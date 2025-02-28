from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    logging,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .crf import CRF
from .pooler import POOLERS

logger = logging.get_logger(__name__)


# TokenClassification docstring
_CONFIG_FOR_DOC = "BertConfig"
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01



@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(
            self,
            config: BertConfig,
            weight_O: float = 0.01,
            bias_O: Optional[float] = None,
            pooler: str = 'last',
        ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        if pooler not in POOLERS:
            message = (
                "Invalid pooler: %s. Pooler must be one of %s."
                % (pooler, list(POOLERS.keys()))
            )
            raise ValueError(message)

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = self._build_classifier(config, pooler)

        if bias_O is not None:
            self.set_bias_tag_O(bias_O)

        assert isinstance(weight_O, float) and 0 < weight_O < 1
        weights_list = [1.] * self.num_labels
        weights_list[0] = weight_O
        weights = torch.tensor(weights_list)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

        self.frozen_bert = False
        self.pooler = POOLERS.get(pooler)

        # Initialize weights and apply final processing
        self.post_init()

    def _build_classifier(self, config: BertConfig, pooler: str) -> nn.Linear:
        if pooler in ('last', 'sum'):
            return torch.nn.Linear(config.hidden_size, config.num_labels)
        else:
            assert pooler == 'concat'
            return torch.nn.Linear(4 * config.hidden_size, config.num_labels)

    def set_bias_tag_O(self, bias_O: Optional[float] = None) -> None:
        """Increase tag "O" bias to produce high probabilities early on and
        reduce instability in early training."""
        if bias_O is not None:
            logger.info('Setting bias of OUT token to %s.', bias_O)
            self.classifier.bias.data[0] = bias_O

    def freeze_bert(self) -> None:
        """Freeze all BERT parameters. Only the classifier weights will be
        updated."""
        for p in self.bert.parameters():
            p.requires_grad = False
        self.frozen_bert = True

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        all_layers_sequence_outputs = outputs[1]
        # Use the defined pooler to pool the hidden representation layers
        assert self.pooler
        sequence_output = self.pooler(all_layers_sequence_outputs)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq, tags)

        loss = None
        if labels is not None:
            # ToDo: Add Prediction mask
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLSTM(BertForTokenClassification):
    """BERT-LSTM model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """
    def __init__(self, config: BertConfig, weight_O: float = 0.01, bias_O: Optional[float] = None, pooler: str = 'last', lstm_hidden_size: int = 200, lstm_layers: int = 1) -> None:
        super().__init__(config, weight_O, bias_O, pooler)
        del self.classifier
        self.freeze_bert()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size if pooler in ('last', 'sum') else 4 * config.hidden_size,
            hidden_size= lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = torch.nn.Linear(self.lstm.hidden_size * 2, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        all_layers_sequence_outputs = outputs[1]
        # Use the defined pooler to pool the hidden representation layers
        assert self.pooler
        sequence_output = self.pooler(all_layers_sequence_outputs)
        sequence_output, (_, _) = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq, tags)

        loss = None
        if labels is not None:
            # ToDo: Add Prediction mask
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertCRF(BertForTokenClassification):
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=return_dict,
            **kwargs
        )

        all_layers_sequence_outputs = outputs[1]
        # Use the defined pooler to pool the hidden representation layers
        assert self.pooler
        sequence_output = self.pooler(all_layers_sequence_outputs)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq, tags)

        if labels is not None:
            assert input_ids
            loss = self.crf(logits, labels, reduction='token_mean')
            loss = loss / float(input_ids.size(0))
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


class BertLSTMCRF(BertForTokenClassification):
    """BERT-LSTM-CRF model.
    Args:
        config: BertConfig instance to build BERT model.
        kwargs: arguments to be passed to superclass.
    """

    def __init__(self, config: BertConfig, weight_O: float = 0.01, bias_O: Optional[float] = None, pooler: str = 'last', lstm_hidden_size: int = 200, lstm_num_layers: int = 2) -> None:
        super().__init__(config, weight_O, bias_O, pooler)
        self.freeze_bert()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size if pooler in ('last', 'sum') else 4 * config.hidden_size,
            hidden_size= lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
