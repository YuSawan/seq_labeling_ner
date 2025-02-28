import warnings
from typing import Any

import torch
from transformers import DataCollatorWithPadding


class Collator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [f.copy() for f in features]
        labels = [f.pop("labels") for f in features]
        _ = [f.pop("token_type_ids", None) for f in features]
        extra_fields = {}
        extra_field_names = {"id"}
        for k in list(features[0].keys()):
            if k in extra_field_names:
                extra_fields[k] = [f.pop(k) for f in features]

        batch = super().__call__(features)
        batch.update(extra_fields)

        entity_length = batch["input_ids"].shape[1]
        if self.tokenizer.padding_side == "right":
            batch["labels"] = [label + [-100] * (entity_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[-100] * (entity_length - len(label)) + label for label in labels]

        if self.return_tensors == "pt":
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        else:
            warnings.warn(f"return_tensors='{self.return_tensors}' is not supported.")

        return batch
