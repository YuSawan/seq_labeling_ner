from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    ModernBertConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
    logging,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.modernbert.modeling_modernbert import (
    MODERNBERT_INPUTS_DOCSTRING,
    MODERNBERT_START_DOCSTRING,
    ModernBertPredictionHead,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .crf import CRF

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "answerdotai/ModernBERT-base"
_CONFIG_FOR_DOC = "ModernBertConfig"


@add_start_docstrings(
    "The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(
            self,
            config: ModernBertConfig,
            weight_O: float = 0.01,
            bias_O: Optional[float] = None,
        ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights_list = [1.] * self.num_labels
        weights_list[0] = weight_O
        weights = torch.tensor(weights_list)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

        self.frozen_bert = False

        # Initialize weights and apply final processing
        self.post_init()

    def set_bias_tag_O(self, bias_O: Optional[float] = None) -> None:
        """Increase tag "O" bias to produce high probabilities early on and
        reduce instability in early training."""
        if bias_O is not None:
            logger.info('Setting bias of OUT token to %s.', bias_O)
            self.classifier.bias.data[0] = bias_O

    def freeze_bert(self) -> None:
        """Freeze all BERT parameters. Only the classifier weights will be
        updated."""
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        self.frozen_bert = True


    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.head(sequence_output)
        sequence_output = self.drop(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertLSTM(ModernBertForTokenClassification):
    """BERT-LSTM model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """
    def __init__(self, config: BertConfig, weight_O: float = 0.01, bias_O: Optional[float] = None, lstm_hidden_size: int = 200, lstm_layers: int = 1) -> None:
        super().__init__(config, weight_O, bias_O)
        del self.classifier
        self.freeze_bert()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size= lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = torch.nn.Linear(self.lstm.hidden_size * 2, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.head(sequence_output)
        sequence_output, (_, _) = self.lstm(sequence_output)
        sequence_output = self.drop(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ModernBertCRF(ModernBertForTokenClassification):
    """BERT-CRF model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        del self.loss_fct  # Delete unused CrossEntropyLoss
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        if labels is not None:
            assert input_ids
            loss = self.crf(logits, labels, reduction='token_mean')
            loss = loss / float(logits.size(0))
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertLSTMCRF(ModernBertForTokenClassification):
    """BERT-LSTM-CRF model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, **kwargs: Any):
        super().__init__(config, **kwargs)
        del self.loss_fct  # Delete unused CrossEntropyLoss
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size= self.bert.config.hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
