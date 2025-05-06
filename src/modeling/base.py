from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import logging

from ..config import BertNERConfig
from .crf import CRF
from .encoder import Encoder
from .model_output import TokenClassifierOutput

logger = logging.get_logger(__name__)


def viterbi(scores_bert: np.ndarray, num_labels: int, penalty: int = 10000) -> list[int]:
    num_entity_type = num_labels // 2
    m = 2 * num_entity_type + 1
    penalty_matrix = np.zeros([m, m])
    for i in range(m):
        for j in range(1+num_entity_type, m):
            if not ( (i == j) or (i + num_entity_type == j) ):
                penalty_matrix[i, j] = penalty

    path = [ [i] for i in range(m) ]
    scores_path = scores_bert[0] - penalty_matrix[0,:]
    scores_bert = scores_bert[1:]

    for scores in scores_bert:
        assert len(scores) == 2 * num_entity_type + 1
        score_matrix = np.array(scores_path).reshape(-1,1) \
            + np.array(scores).reshape(1,-1) \
            - penalty_matrix
        scores_path = score_matrix.max(axis=0)
        argmax = score_matrix.argmax(axis=0)
        path_new = []
        for i, idx in enumerate(argmax):
            path_new.append( path[idx] + [i] )
        path = path_new

    labels_optimal = path[np.argmax(scores_path)]
    return labels_optimal


class TokenModel(nn.Module):
    def __init__(self, config: BertNERConfig, encoder_from_pretrained: bool = False):
        super().__init__()
        self.num_labels = config.num_labels
        self.encoder = Encoder(config, encoder_from_pretrained)
        if not config.encoder_config:
            config.encoder_config = self.encoder.config
        if config.freeze_bert:
            self.encoder.freeze_bert()
        self.config = config

        if config.classifier_dropout is not None:
            classifier_dropout = ( config.classifier_dropout )
        else:
            classifier_dropout = (
                config.encoder_config.mlp_dropout if config.encoder_config.model_type == 'modernbert' else config.encoder_config.hidden_dropout_prob
            )

        self.dropout = nn.Dropout(classifier_dropout)
        assert 0 < config.weight_O < 1
        self.classifier = self._build_classifier()

        if config.no_crf:
            pass
        else:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def _build_classifier(self) -> nn.Linear:
        if self.config.lstm_hidden_size:
            classifier = nn.Linear(2 * self.config.lstm_hidden_size, self.config.num_labels)
        else:
            if self.config.pooler in ('last', 'sum'):
                classifier = nn.Linear(self.config.encoder_config.hidden_size, self.config.num_labels)
            else:
                assert self.config.pooler == 'concat'
                classifier = nn.Linear(4 * self.config.encoder_config.hidden_size, self.config.num_labels)

        bias_O = self.config.bias_O
        """Increase tag "O" bias to produce high probabilities early on and
        reduce instability in early training."""
        if bias_O is not None:
            logger.info('Setting bias of OUT token to %s.', bias_O)
            classifier.bias.data[0] = bias_O

        return classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        """Performs the forward pass of the network.
        If `labels` is not `None`, it will calculate and return the the loss,
        that is the negative log-likelihood of the batch.
        Args:
            input_ids: tensor of input token ids.
            prediction_mask: mask tensor should have value 0 for tokens that do
                not have an associated prediction, such as [CLS] and WordPìece
                subtoken continuations (that start with ##).
            attention_mask: mask tensor that should have value 0 for [PAD]
                tokens and 1 for other tokens.
            token_type_ids: tensor of input sentence type id (0 or 1). Should be
                all zeros for NER. Can be safely set to `None`.
            labels: tensor of gold NER tag label ids. Values should be ints in
                the range [0, config.num_labels - 1].
        Returns a dict with calculated tensors:
            - "loss" (if `labels` is not `None`)
            - "logits"
        """

        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.no_crf:
                weights = torch.ones(self.num_labels)
                weights[0] = self.config.weight_O
                weights.to(self.encoder.device)
                loss_fct = CrossEntropyLoss(weight=weights, reduction='mean')
                masked_labels = labels.masked_fill(~prediction_mask.bool(), -100)
                loss = loss_fct(logits.view(-1, self.num_labels), masked_labels.view(-1))
            else:
                # Negative of the log likelihood.
                # Loop through the batch here because of 2 reasons:
                # 1- the CRF package assumes the mask tensor cannot have interleaved
                # zeros and ones. In other words, the mask should start with True
                # values, transition to False at some moment and never transition
                # back to True. That can only happen for simple padded sequences.
                # 2- The first column of mask tensor should be all True, and we
                # cannot guarantee that because we have to mask all non-first
                # subtokens of the WordPiece tokenization.
                loss = 0.
                for seq_logits, seq_labels, seq_mask in zip(logits, labels, prediction_mask):
                    # Index logits and labels using prediction mask to pass only the
                    # first subtoken of each word to CRF.
                    seq_logits = seq_logits[seq_mask.bool()].unsqueeze(0)
                    seq_labels = seq_labels[seq_mask.bool()].unsqueeze(0)
                    loss -= self.crf(seq_logits, seq_labels, reduction='token_mean')
                loss /= sequence_output.size(0)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            prediction_mask=prediction_mask
        )

    @torch.no_grad()
    def decode(
        self,
        logits: torch.Tensor,
        prediction_mask: torch.Tensor
    ) -> torch.Tensor:

        if self.config.no_crf:
            # predictions = torch.argmax(logits, dim=2)
            # y_preds = []
            # for pred, seq_mask in zip(predictions, prediction_mask):
            #     tags = pred[seq_mask.bool()].tolist()
            #     y_preds.append(tags)

            # If viterebi decoding
            y_preds = []
            for seq_logits, seq_mask in zip(logits, prediction_mask):
                seq_logits = seq_logits[seq_mask.bool()]
                y_pred = viterbi(seq_logits.cpu().detach().numpy(), num_labels=self.num_labels)
                y_preds.append(torch.tensor(y_pred))
        else:
            y_preds = []
            for seq_logits, seq_mask in zip(logits, prediction_mask):
                seq_logits = seq_logits[seq_mask.bool()].unsqueeze(0)
                tags = self.crf.decode(seq_logits)
                # Unpack "batch" results
                y_preds.append(torch.tensor(tags[0]))

        return pad_sequence(y_preds, batch_first=True, padding_value = -100)
