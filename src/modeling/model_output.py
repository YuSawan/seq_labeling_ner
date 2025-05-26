from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before classifier).
        predictions (`torch.Tensor`, of shape `(batch_size, sequence_length)`):
            Classification predictions.
        prediction_mask (`torch.Tensor`, of shape `(batch_size, sequence_length)`):
            Classification loss.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.Tensor] = None
    prediction_mask: Optional[torch.Tensor] = None
