import random
from typing import Iterable

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.AbstractBaseLogger import AbstractBaseLogger
from src.bug_prediction.BugPredictionArgs import model_class_of, BugPredictionArgs
from src.bug_prediction.BugPredictionModel import BugPredictionModel
from src.bug_prediction.Scorer import Scorer, Score


class ModelLoader(AbstractBaseLogger):
    def __init__(self, name: str, args: BugPredictionArgs):
        super().__init__(name)
        self.args = args
        if self.args.no_gpu:
            self.device = torch.device("cpu")
        else:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda")

        self.info(f"Args: {self.args}")
        self.info(f"{self.device=}")

        self._seed()

    def evaluate(
        self,
        data_loader: DataLoader,
        metric_names: Iterable[str],
        output_file: str = None,
    ) -> Score:
        """
        Evaluate the model on the given data loader.
        It will return a mapping from metric name to its score.
        """
        # noinspection PyTypeChecker
        self.info(f"Num examples: {len(data_loader.dataset)}")
        self.info(f"Num batches: {len(data_loader)}")
        self.info(f"Metrics: {metric_names}")

        total_loss = 0.0
        logit_batches, label_batches = [], []
        self.model.eval()
        eval_progress = tqdm(
            data_loader, desc="Evaluating", position=1, smoothing=0.001
        )
        for i, batch in enumerate(eval_progress):
            # the third element is the actual number of lines in the file
            inputs, labels, *_ = batch

            with torch.no_grad():
                loss, logit = self.model(
                    inputs.to(self.device),
                    labels.type(torch.long).to(self.device),
                )
                total_loss += loss.mean().item()
                logit_batches.append(logit.cpu().numpy())
                label_batches.append(labels.cpu().numpy())

        logits = np.concatenate(logit_batches, axis=0)
        labels = np.concatenate(label_batches, axis=0).reshape(-1)
        loss = total_loss / len(data_loader)
        if len(logits.shape) == 3:
            # for line-level defect prediction
            true_prob = logits[:, :, 1].reshape(-1)
        elif len(logits.shape) == 2:
            # for file-level defect prediction
            true_prob = logits[:, 1].reshape(-1)
        else:
            raise ValueError(f"Invalid {logits.shape=}")

        if output_file is not None:
            # write the prediction results to a file in npz format
            np.savez(output_file, true_prob=true_prob, labels=labels)

        return Scorer.compute_score(metric_names, labels, true_prob, loss)

    def _load_models(self, pad_token_id: int, is_checkpoint=False):
        model_classes = model_class_of[self.args.model_type]
        self.config = model_classes.config.from_pretrained(
            self.args.config_name or self.args.model_name
        )
        self.info(
            f"Loading {self.args.model_type} model from {self.args.model_name} "
            f"{is_checkpoint=}",
        )
        self.model = BugPredictionModel(
            self.args.model_name,
            self.config,
            self.args.encoder_type,
            is_checkpoint,
            pad_token_id,
            self.args.model_type,
            self.args.max_line_length,
            self.args.max_file_length,
            class_weight=torch.tensor(
                [1, 1 if self.args.class_weight is None else self.args.class_weight],
                device=self.device,
                dtype=torch.float32,
            ),
        )

        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
        self.model = self.model.to(self.device)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        self.info(
            f"Finished loading model {self.args.model_type} of size {int(model_size // 1e6)}M"
        )

    def _seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
