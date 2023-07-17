# coding=utf-8

"""
Fine-tuning the library models for language modeling on a text file
(BERT, RoBERTa). BERT and RoBERTa are fine-tuned using a masked language
modeling (MLM) loss.
"""
import logging
import os
import time
from datetime import timedelta
from itertools import cycle, count
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
)

from src.bug_prediction.BugPredictionArgs import BugPredictionArgs, model_class_of
from src.bug_prediction.DatasetManager import DatasetManager
from src.bug_prediction.ModelLoader import ModelLoader
from src.bug_prediction.Scorer import Score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


class BugPredictorTrainer(ModelLoader):
    __slots__ = (
        "args",
        "device",
        "config",
        "model",
        "num_data_loader_worker",
    )
    MAX_GRADIENT_NORM = 1.0

    def __init__(self):
        args = BugPredictionArgs().parse_args()
        assert args.checkpoint_dir is not None
        assert args.model_type in model_class_of
        super().__init__(self.get_log_file_name(), args)
        self.num_data_loader_worker = min(
            4,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count(),
        )
        self.info(f"{self.num_data_loader_worker=}")

        dataset_manager = DatasetManager(
            model_class_of[self.args.model_type].tokenizer,
            self.args.tokenizer_name,
            self.args.dataset_path,
            self.args.encoder_type,
            ("train", "val"),
            self.args.cache_dir,
            info=self.info,  # info will still be bound to self
            max_line_len=self.args.max_line_length,
            max_file_len=self.args.max_file_length,
        )
        train_dataset, val_dataset = dataset_manager.load_dataset()
        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.num_data_loader_worker,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
        )
        self._load_models(dataset_manager.tokenizer.pad_token_id)

    def start(self):
        start_time = time.perf_counter()
        self.train()
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        self.info(f"Training completed in {timedelta(seconds=elapsed_seconds)}")

    def train(self):
        Path(self.args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        num_epoch_steps = len(self.train_dataloader)
        num_training_steps = num_epoch_steps * self.args.num_epoch

        # prepare optimizer and scheduler
        frozen_layer_names = {"bias", "LayerNorm.weight"}
        optimizer_params = [
            {
                "params": [
                    params
                    for names, params in self.model.named_parameters()
                    if not frozen_layer_names.intersection(names)
                ],
                "weight_decay": 0.0,  # if we want to integrate weight decay, it will go here
            },
            {
                "params": [
                    params
                    for names, params in self.model.named_parameters()
                    if frozen_layer_names.intersection(names)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_params, lr=self.args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.info("********** Running Training **********")
        # noinspection PyTypeChecker
        self.info(f"Num examples: {len(self.train_dataloader.dataset)}")
        self.info(f"Batch size: {self.args.batch_size}")
        self.info(f"Num batches/epoch steps: {num_epoch_steps}")
        self.info(f"Total training steps: {num_training_steps}")

        best_score = 0.0
        no_improv_epoch_count = 0

        overall_progress_bar = tqdm(
            total=num_training_steps,
            initial=self.args.start_epoch * num_epoch_steps,
            desc="Training Progress",
            position=0,
            smoothing=0.01,
        )

        cyclic_data_loader = cycle(self.train_dataloader)
        global_step = 0

        for curr_epoch in count(1):
            training_loss = 0.0
            self.model.train()
            epoch_progress_bar = tqdm(
                range(1, num_epoch_steps + 1),
                desc="Epoch",
                position=1,
                smoothing=0.001,
            )
            for step_i, step_of_epoch in enumerate(epoch_progress_bar):
                global_step += 1
                batch = next(cyclic_data_loader)
                if len(batch) == 3:
                    source_tensor, label_tensor, _ = batch
                else:
                    source_tensor, label_tensor = batch

                # forward pass
                loss: torch.Tensor
                logits: torch.Tensor
                loss, logits = self.model(
                    source_tensor.to(self.device),
                    label_tensor.type(torch.long).to(self.device),
                )
                training_loss += loss.mean().item()

                # compute gradient
                loss.backward(torch.ones_like(loss))
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.MAX_GRADIENT_NORM
                )

                # run optimizer
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # update process description
                overall_progress_bar.update()
                avg_training_loss = training_loss / step_of_epoch
                epoch_progress_bar.set_description(
                    f"{curr_epoch:>3}. Training loss: {avg_training_loss:.4f}"
                )

            # Evaluate
            self.debug("********** CUDA.empty_cache() **********")
            torch.cuda.empty_cache()
            self.debug(f"******** Running Evaluation For Epoch {curr_epoch} ********")
            validation_score = self.evaluate(
                self.val_dataloader,
                metric_names=Score.get_scores_names(),
            )
            self.info(
                f"{curr_epoch:03} Validation: "
                f"Loss={validation_score.loss:2.3f}, "
                f"AucRoc={validation_score.roc_auc_score:2.3f}, "
                f"BAcc={validation_score.balanced_accuracy_score:2.3f}, "
                f"MCC={validation_score.matthews_corrcoef:2.3f}, "
                f"Recall={validation_score.recall_score:2.3f}, "
                f"R@20%={validation_score.recall_at_20_percent_loc:2.3f}, "
                f"E@20%={validation_score.effort_at_20_percent_recall:2.3f} "
                f"IFA={validation_score.initial_false_alarm:2.3f} "
            )

            score = (
                validation_score.roc_auc_score / 0.8
                + validation_score.balanced_accuracy_score / 0.6
                + validation_score.recall_at_20_percent_loc / 0.8
                + (1 - validation_score.effort_at_20_percent_recall) / 0.9
            )

            # Save best model
            if score > best_score:
                no_improv_epoch_count = 0
                self.info(f"Best score: {score:.3f}")
                best_score = score

                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_output_filepath = os.path.join(
                    self.args.checkpoint_dir, "pytorch_model.bin"
                )
                torch.save(model_to_save.state_dict(), model_output_filepath)
                self.info(f'Saved best model into "{model_output_filepath}"')

            else:
                no_improv_epoch_count += 1
                self.info(f"No improvement in {no_improv_epoch_count} epochs")

                if no_improv_epoch_count > self.args.patient:
                    self.warning(
                        f"Early stop as score does not increase in {no_improv_epoch_count} epochs"
                    )
                    self.info(f"Best score: {best_score}")
                    break

            steps_completed = global_step >= (
                num_training_steps - (self.args.start_epoch * num_epoch_steps)
            )
            if steps_completed:
                self.info(
                    f"Training completed after {curr_epoch} epoch and {global_step} steps"
                )
                break

            overall_progress_bar.update()
            self.debug("********** CUDA.empty_cache() **********")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    BugPredictorTrainer().start()
