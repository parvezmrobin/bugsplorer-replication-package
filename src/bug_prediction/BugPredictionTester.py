import time
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.bug_prediction.BugPredictionArgs import BugPredictionArgs, model_class_of
from src.bug_prediction.DatasetManager import DatasetManager
from src.bug_prediction.ModelLoader import ModelLoader
from src.bug_prediction.Scorer import Score, Scorer

K = 1500


class IndicesValues(NamedTuple):
    indices: torch.Tensor
    values: torch.Tensor


class BugPredictionTester(ModelLoader):
    __slots__ = ()

    def __init__(self):
        args = BugPredictionArgs().parse_args()
        assert args.output_path is not None
        super().__init__(self.get_log_file_name(), args)

        dataset_manager = DatasetManager(
            model_class_of[self.args.model_type].tokenizer,
            self.args.tokenizer_name,
            self.args.dataset_path,
            "line",  # for testing, always use line-level data
            ("test",),
            self.args.cache_dir,
            info=self.info,  # info will still be bound to self
            max_line_len=self.args.max_line_length,
            max_file_len=self.args.max_file_length,
        )

        (test_dataset,) = dataset_manager.load_dataset()
        self.test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.args.batch_size,
        )

        self._load_models(dataset_manager.tokenizer.pad_token_id, is_checkpoint=True)

    def test(self):
        start_time = time.perf_counter()
        self.info(f"Metrics: {Score.get_scores_names()}")
        if self.args.encoder_type == "file":
            true_prob, labels, loss = self._predict_from_file_classifier(
                self.test_dataloader,
            )
            np.savez(
                self.args.output_path,
                true_prob=true_prob,
                labels=labels,
            )
            test_score = Scorer.compute_score(
                Score.get_scores_names(),
                labels,
                true_prob,
                loss=loss,
            )
            self.info(f"Test: {test_score}")
        elif self.args.encoder_type == "line":
            self.info("Regular Evaluation:")
            test_score = self.evaluate(
                self.test_dataloader,
                Score.get_scores_names(),
                output_file=f"{self.args.output_path}-regular",
            )
            self.info(f"Test: {test_score}")
        else:
            raise ValueError(f"Unknown encoder type: {self.args.encoder_type}")

        Path(self.args.output_path).parent.mkdir(parents=True, exist_ok=True)

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        self.info(f"Testing completed in {timedelta(seconds=elapsed_seconds)}")

    def _predict_from_line_classifier(self, data_loader: DataLoader):
        """
        Evaluate the model on the given data loader.
        It will return a mapping from metric name to its score.
        If the number of lines in a file is greater than the maximum number of lines,
        we will pad the logits with zeros (i.e., consider them as not buggy).
        However, we will pad the labels with 1 (i.e., consider them as buggy).
        It is a ***conservative approach***, where we consider the model is failing
        if the file is larger than the maximum number of lines.

        If write_predictions_to is not None, it will write the predictions to the
        filepath specified by write_predictions_to.
        """
        # noinspection PyTypeChecker
        self.info(f"Num examples: {len(data_loader.dataset)}")
        self.info(f"Num batches: {len(data_loader)}")

        total_loss = 0.0
        logit_list = []
        label_list = []
        self.model.eval()
        test_progress = tqdm(data_loader, desc="Testing", position=1, smoothing=0.001)
        for batch_i, batch in enumerate(test_progress):
            source_batch_tensor, labels_batch_tensor, num_lines = batch

            with torch.no_grad():
                loss, logit_batch_tensor = self.model(
                    source_batch_tensor.to(self.device),
                    labels_batch_tensor.type(torch.long).to(self.device),
                )
                total_loss += loss.mean().item()

                # logit_batch_tensor: (batch_size, max_num_lines, 2)
                # logit_batch.shape == labels_batch.shape == (batch_size, max_num_lines)
                logit_batch = logit_batch_tensor[:, :, 1].cpu().numpy()
                label_batch = labels_batch_tensor.cpu().numpy()
                # len(num_lines) == batch_size. Thus, enumerate(num_lines) will
                # return a tuple (i, num_line) where i is the index of the batch.
                for i, num_line in enumerate(num_lines):
                    # logit_batch.shape[1] is the maximum number of lines.
                    # num_line is the actual number of lines in the file.
                    if num_line <= logit_batch.shape[1]:
                        # This means the file was padded or of original length.
                        # We will remove the padded logits and labels, if any.
                        logit_list.append(logit_batch[i, :num_line])
                        label_list.append(labels_batch_tensor[i, :num_line])
                    else:
                        # This means the file was truncated. We will evaluate in a conservative
                        # approach where we pad the logits with 0.49 and the labels with ones.
                        # This is because we consider the model is failing if the file is larger
                        # than the maximum number of lines.
                        remaining_logits = np.zeros(num_line - logit_batch.shape[1])
                        remaining_logits.fill(0.49)
                        extended_logits = np.concatenate(
                            [logit_batch[i], remaining_logits]
                        )
                        logit_list.append(extended_logits)
                        extended_labels = np.concatenate(
                            [
                                label_batch[i],
                                np.ones(num_line - logit_batch.shape[1]),
                            ]
                        )
                        label_list.append(extended_labels)

        logits = np.concatenate(logit_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        loss = total_loss / len(data_loader)
        true_prob = logits

        assert len(true_prob.shape) == len(labels.shape) == 1
        assert true_prob.shape == labels.shape

        return true_prob, labels, loss

    def _predict_from_file_classifier(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        logit_list = []
        label_list = []

        # noinspection PyTypeChecker
        self.info(f"Num examples: {len(data_loader.dataset)}")
        self.info(f"Num batches: {len(data_loader)}")

        test_progress = tqdm(data_loader, desc="Testing", position=1, smoothing=0.001)
        with torch.no_grad():
            for batch_i, batch in enumerate(test_progress):
                assert len(batch) == 3
                source_batch_tensor, labels_batch_tensor, num_lines = batch
                # file_labels_batch_tensor contains the labels for each file.
                # It is a tensor of shape (batch_size,)
                # If any line in the file is buggy, the file is buggy.
                file_labels_batch_tensor = labels_batch_tensor.max(dim=1).values

                # logit_batch_tensor is a tensor of shape (batch_size, 2)
                # attention is a list[tuple[Tensor]]
                #    where each tuple contains 6 tensors of shape
                #    (file_length=512, num_heads, seq_len, seq_len)
                loss, logit_batch_tensor, attentions = self.model(
                    source_batch_tensor.to(self.device),
                    file_labels_batch_tensor.type(torch.long).to(self.device),
                    output_attentions=True,
                )

                # attention_by_file is a list[Tensor]
                #    where each tensor is of shape
                #    (num_layers=6, file_length=512, num_heads, seq_len, seq_len)
                #    and the list is of length batch_size
                attention_by_file = [torch.stack(attention) for attention in attentions]

                # However, if the model is running on N GPUs, len(attention_by_file) will be
                # batch_size / N and the second dimension in each attention will be
                # (file_length * N) instead of file_length. Thus, we will split that dimension
                # into N parts and append them to the first dimension.
                if len(attention_by_file) < len(source_batch_tensor):
                    attention_by_file = [
                        split
                        for attention in attention_by_file
                        for split in torch.split(attention, 512, dim=1)
                    ]

                # reduce attention dimensions
                attention_by_file = [
                    attention.mean(dim=(0, 2, 3)) for attention in attention_by_file
                ]

                assert all(
                    attention.shape == (512, 16) for attention in attention_by_file
                ), [attention.shape for attention in attention_by_file]

                # Now we will take top K = 1500 attention values from each file in
                # attention_by_file. Then we will sum the attention values by line and rank the
                # lines by their summed attention values.

                highest_attentions_by_file = [
                    # torch.sort returns a namedtuple of (values, indices).
                    # keep_k keeps the top k values from each tensor.
                    keep_k(torch.sort(attention.flatten(), descending=True), K)
                    for attention in attention_by_file
                ]

                # assert all indices are less than the maximum number of tokens
                # in a file = 512 * 16
                highest_attention_indices = [
                    highest_attentions.indices.max()
                    for highest_attentions in highest_attentions_by_file
                ]
                assert all(
                    0 <= index < 512 * 16 for index in highest_attention_indices
                ), highest_attention_indices

                # Now we will sum the attention values by line and rank the lines by their summed
                # attention values.
                # attention_by_line_by_file is a Tensor of shape (batch_size, file_length=512)
                attention_by_line_by_file = torch.zeros(
                    (len(attention_by_file), 512), device=self.device
                )
                for file_i, highest_attentions_of_file in enumerate(
                    highest_attentions_by_file
                ):
                    for token_i, attn_value in zip(
                        highest_attentions_of_file.indices,
                        highest_attentions_of_file.values,
                    ):
                        # token_i is the index of the token in the flattened file. Since each line
                        # has 16 tokens, we will divide the token index by 16 to get the line index.
                        line_i = token_i // self.args.max_line_length
                        attention_by_line_by_file[file_i][line_i] += attn_value

                # Now we will normalize the attention values for the lines in each file.
                attention_by_line_by_file = torch.softmax(
                    attention_by_line_by_file, dim=1
                )
                total_loss += loss.mean().item()
                logit_batch = attention_by_line_by_file.cpu().numpy()
                label_batch = labels_batch_tensor.cpu().numpy()
                logit_list.append(logit_batch)
                label_list.append(label_batch)
        logits = np.concatenate(logit_list, axis=0).reshape(-1)
        labels = np.concatenate(label_list, axis=0).reshape(-1)

        # If the number of batches is not divisible by the numer of GPUs,
        # the last batch will be ignored. Thus, we will discard that batch
        # from the labels.
        if len(logits) < len(labels):
            labels = labels[: len(logits)]
        assert len(logits) == len(labels), (len(logits), len(labels))

        return logits, labels, total_loss / len(data_loader)


def keep_k(s: IndicesValues, k: int):
    return IndicesValues(
        indices=s.indices[:k],
        values=s.values[:k],
    )


if __name__ == "__main__":
    BugPredictionTester().test()
