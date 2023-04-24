from typing import NamedTuple, Type, Optional

# noinspection PyPackageRequirements
from tap import Tap
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaTokenizerFast,
    T5Config,
    RobertaModel,
    T5EncoderModel,
)

from src.bug_prediction.FileEncoders import (
    RobertaForLineClassification,
    BertForLineClassification,
    T5ForLineClassification,
)


class BugPredictionArgs(Tap):
    seed: int = 4
    config_name: str = None
    model_type: str
    model_name: str = None
    encoder_type: str
    tokenizer_name: str
    # covers 90% of the lines in defectors, 99% in linedp
    max_line_length: int = 16
    # covers 82% in linedp
    max_file_length: int = 512
    class_weight: int = None
    dataset_path: str = "dataset/python/data/bug_prediction_splits"
    cache_dir: str
    checkpoint_dir: str
    output_path: Optional[str] = None
    batch_size: int = 16
    start_epoch: int = 0
    num_epoch: int = 20  # number of epoch
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    patient: int = 5
    no_gpu: bool = False

    def configure(self) -> None:
        self.add_argument("--model_type", choices=model_class_of.keys())
        self.add_argument("--encoder_type", choices=["line", "file"])


class ModelClasses(NamedTuple):
    config: Type[PretrainedConfig]
    line_encoder: Type[PreTrainedModel]
    file_encoder: Type[PreTrainedModel]
    tokenizer: Type[PreTrainedTokenizerFast]


model_class_of: dict[str, ModelClasses] = {
    "roberta": ModelClasses(
        RobertaConfig, RobertaModel, RobertaForLineClassification, RobertaTokenizerFast
    ),
    "bert": ModelClasses(
        RobertaConfig, RobertaModel, BertForLineClassification, RobertaTokenizerFast
    ),
    "t5": ModelClasses(
        T5Config,
        T5EncoderModel,
        T5ForLineClassification,
        RobertaTokenizerFast,
    ),
}
