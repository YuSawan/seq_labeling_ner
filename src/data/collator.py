import warnings
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    TrainingArguments,
)


def get_dataloader(dataset: Dataset, tokenizer: PreTrainedTokenizerFast, batch_size: int, training_args: TrainingArguments) -> DataLoader:
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": Collator(tokenizer),
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
    }
    dataloader_params["sampler"] = SequentialSampler(dataset)
    dataloader_params["drop_last"] = training_args.dataloader_drop_last
    dataloader_params["prefetch_factor"] = training_args.dataloader_prefetch_factor
    return DataLoader(dataset, **dataloader_params)


class Collator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [f.copy() for f in features]

        labels = [f.pop("labels") for f in features] if "labels" in list(features[0].keys()) else None
        prediction_mask = [f.pop("prediction_mask") for f in features]
        _ = [f.pop("token_type_ids", None) for f in features]
        extra_fields = {}
        extra_field_names = {"id", "offsets"}
        for k in list(features[0].keys()):
            if k in extra_field_names:
                extra_fields[k] = [f.pop(k) for f in features]

        batch = super().__call__(features)
        entity_length = batch["input_ids"].shape[1]

        if self.tokenizer.padding_side == "right":
            batch["labels"] = [label + [-100] * (entity_length - len(label)) for label in labels] if labels else None
            batch["prediction_mask"] = [mask + [0] * (entity_length - len(mask)) for mask in prediction_mask]
        else:
            batch["labels"] = [[-100] * (entity_length - len(label)) + label for label in labels] if labels else None
            batch["prediction_mask"] = [[0] * (entity_length - len(mask)) + mask for mask in prediction_mask]

        if self.return_tensors == "pt":
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64) if batch["labels"] else None
            batch["prediction_mask"] = torch.tensor(batch["prediction_mask"], dtype=torch.int64)
        else:
            warnings.warn(f"return_tensors='{self.return_tensors}' is not supported.")

        return batch
