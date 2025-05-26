import torch


def sum_last_4_layers(sequence_outputs: tuple[torch.Tensor]) -> torch.Tensor:
    """Sums the last 4 hidden representations of a sequence output of BERT.
    Args:
    -----
    sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
        For BERT base, the Tuple has length 13.
    Returns:
    --------
    summed_layers: Tensor of shape (batch, seq_length, hidden_size)
    """
    last_layers = sequence_outputs[-4:]
    return torch.stack(last_layers, dim=0).sum(dim=0)


def get_last_layer(sequence_outputs: tuple[torch.Tensor]) -> torch.Tensor:
    """Returns the last tensor of a list of tensors."""
    return sequence_outputs[-1]


def concat_last_4_layers(sequence_outputs: tuple[torch.Tensor]) -> torch.Tensor:
    """Concatenate the last 4 tensors of a tuple of tensors."""
    last_layers = sequence_outputs[-4:]
    return torch.cat(last_layers, dim=-1)


POOLERS = {
    'sum': sum_last_4_layers,
    'last': get_last_layer,
    'concat': concat_last_4_layers,
}
