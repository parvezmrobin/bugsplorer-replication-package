import os.path

import torch
from torch import nn, Tensor
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
    SequenceClassifierOutput,
)

from src.bug_prediction.BugPredictionArgs import model_class_of
from src.bug_prediction.FileEncoders import (
    RobertaForFileClassification,
    T5Pooler,
)


class BugPredictionModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        config: PretrainedConfig,
        encoder_type: str,  # line or file
        is_checkpoint: bool,
        pad_token_id: int,
        model_type: str,
        max_line_length: int,
        max_file_length: int,
        class_weight: torch.Tensor,
    ):
        super(BugPredictionModel, self).__init__()

        # if the model is a checkpoint, we load the whole BugPredictionModel
        # from the checkpoint. Otherwise, we load the token encoder from the
        # pretrained model and initialize the line/file encoder from scratch.

        model_class = model_class_of[model_type]
        print(f"{model_class=}")

        if is_checkpoint or not pretrained_model_name:
            self.token_encoder = model_class.line_encoder(config=config)
        else:
            self.token_encoder = model_class.line_encoder.from_pretrained(
                pretrained_model_name, config=config
            )

        if model_type == "t5":
            self.t5_pooler = T5Pooler(config)

        if encoder_type == "line":
            self.line_encoder = model_class.file_encoder(
                config, class_weight=class_weight
            )
        elif encoder_type == "file":
            assert model_type == "roberta"
            self.file_encoder = RobertaForFileClassification(
                config, class_weight=class_weight
            )
        else:
            raise ValueError(f"Invalid {encoder_type=}")

        if is_checkpoint:
            self.load_state_dict(
                torch.load(os.path.join(pretrained_model_name, "pytorch_model.bin"))
            )

        self.config = config
        self.pad_token_id = pad_token_id
        self.model_type = model_type
        self.max_line_length = max_line_length
        self.max_file_length = max_file_length

    def get_roberta_vec(
        self, source_tensor: Tensor, label_tensor: Tensor, output_attentions=False
    ):
        token_encoder_outputs = []
        token_attentions = []
        for file_tensor in source_tensor:
            token_encoder_output: BaseModelOutputWithPoolingAndCrossAttentions = (
                self.token_encoder(
                    input_ids=file_tensor,
                    attention_mask=file_tensor.ne(self.pad_token_id),
                    output_attentions=output_attentions,
                )
            )

            if self.model_type == "t5":
                token_encoder_output.pooler_output = self.t5_pooler(
                    token_encoder_output.last_hidden_state
                )

            token_encoder_outputs.append(token_encoder_output.pooler_output)
            if output_attentions:
                token_attentions.append(token_encoder_output.attentions)

        line_tensor = torch.stack(token_encoder_outputs)
        assert line_tensor.shape[1:] == (self.max_file_length, self.config.hidden_size)

        # for line attention, we do not attend to empty lines
        # emtpy lines do not contain any token and thus starts with a padding token
        # this attention will be repeated for the whole 2nd dimension,
        # i.e., the hidden state dimension
        attention_mask = source_tensor[:, :, 0].ne(self.pad_token_id)
        if self.model_type == "t5":
            attention_mask = attention_mask[:, None, :]
        else:
            attention_mask = attention_mask[:, None, None, :]
        # assert (
        #     attention_mask.shape == line_tensor.shape
        # ), f"{attention_mask.shape} != {line_tensor.shape}"

        encoder_output: TokenClassifierOutput | SequenceClassifierOutput
        if hasattr(self, "line_encoder"):
            encoder_output = self.line_encoder(
                hidden_states=line_tensor,
                attention_mask=attention_mask,
                labels=label_tensor,
            )
        elif hasattr(self, "file_encoder"):
            encoder_output = self.file_encoder(
                hidden_states=line_tensor,
                attention_mask=attention_mask,
                labels=label_tensor,
            )
        else:
            raise ValueError("line_encoder or file_encoder should be defined")

        # line encoder_output.logits.shape == (BATCH_SIZE, self.max_file_length, 2)
        # file encoder_output.logits.shape == (BATCH_SIZE, 2)

        if output_attentions:
            return (
                encoder_output.loss,
                encoder_output.logits,
                token_attentions,
            )
        return encoder_output.loss, encoder_output.logits

    def forward(
        self,
        source_tensor: Tensor,
        label_tensor: Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.float32, torch.Tensor] | tuple[torch.float32, torch.Tensor, list]:
        assert (
            (source_tensor.dim() == 3)
            and (source_tensor.size(1) == self.max_file_length)
            and (source_tensor.size(2) == self.max_line_length)
        ), source_tensor.size()

        # if output_attentions is False, token_attentions will be an empty list
        loss, logits, *token_attentions = self.get_roberta_vec(
            source_tensor, label_tensor, output_attentions=output_attentions
        )

        softmax_logits = nn.functional.softmax(logits, dim=-1)

        if softmax_logits.dim() == 3:
            assert softmax_logits.shape[1:] == (self.max_file_length, 2)
        else:
            assert softmax_logits.dim() == 2 and softmax_logits.size(1) == 2

        if output_attentions:
            return loss, softmax_logits, token_attentions[0]

        return loss, softmax_logits
