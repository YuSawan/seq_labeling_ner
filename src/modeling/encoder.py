import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.modernbert import ModernBertConfig

from ..config import BertNERConfig
from .pooler import POOLERS

TOKEN = os.environ.get('TOKEN', True)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(hidden_size, hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class Bert(nn.Module):
    def __init__(
            self,
            model_name: str,
            config: dict | PretrainedConfig | None,
            from_pretrained: bool = False
        ) -> None:
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(model_name, token=TOKEN)

        if config.__class__.__name__ in ['DebertaV2Config', 'ModernBertConfig']:
            custom = True
        else:
            custom = False

        if not isinstance(config, PretrainedConfig):
            raise ValueError(f"Unspecified types: {type(config)}. Expected: PretrainedConfig")

        if from_pretrained:
            if custom:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=TOKEN)
            else:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, add_pooling_layer=False, token=TOKEN)
        else:
            if custom:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=TOKEN)
            else:
                self.model = AutoModel.from_config(config, trust_remote_code=True, add_pooling_layer=False)

        self.config = config

    def forward(self, *args: int , **kwargs: str) -> BaseModelOutput | BaseModelOutputWithPoolingAndCrossAttentions:
        output = self.model(*args, return_dict = True, output_hidden_states = True, **kwargs)
        return output


class Encoder(nn.Module):
    def __init__(self, config: BertNERConfig, from_pretrained: bool = False) -> None:
        super().__init__()

        self.bert = Bert(config.model_name, config.encoder_config, from_pretrained)
        self.config = self.bert.config
        self.pooler = POOLERS[f'{config.pooler}']

        if isinstance(self.config, ModernBertConfig):
            bert_hidden_size = self.config.hidden_size
            self.head = ModernBertPredictionHead(
                hidden_size = bert_hidden_size if config.pooler in ('last', 'sum') else 4 * bert_hidden_size,
                config=self.config
            )

        if config.lstm_layers and config.lstm_hidden_size:
            bert_hidden_size = self.config.hidden_size
            self.lstm = nn.LSTM(
                input_size = bert_hidden_size if config.pooler in ('last', 'sum') else 4 * bert_hidden_size,
                hidden_size= config.lstm_hidden_size,
                num_layers = config.lstm_layers,
                batch_first = True,
                bidirectional = True,
            )

    @property
    def device(self) -> torch.device:
        return self.bert.model.device

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> nn.Embedding:
        return self.bert.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_input_embeddings(self) -> nn.Module:
        return self.bert.model.get_input_embeddings()

    def freeze_bert(self) -> None:
        """Freeze all BERT parameters. Only the classifier weights will be updated."""
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if token_type_ids is not None:
            logits = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).hidden_states
        else:
            logits = self.bert(input_ids, attention_mask=attention_mask).hidden_states

        sequence_outputs = self.pooler(logits)

        if hasattr(self, 'head'):
            sequence_outputs = self.head(sequence_outputs)

        if hasattr(self, 'lstm'):
            sequence_outputs, (_, _) = self.lstm(sequence_outputs)

        return sequence_outputs
