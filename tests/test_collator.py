from typing import Any

import pytest
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, BatchEncoding, TrainingArguments

from src.data import Collator, Preprocessor, get_sequence_labels

dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = load_dataset("json", data_files={"train": dataset_path}, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format = "iob2")
training_args = TrainingArguments(output_dir=".tmp/")


@pytest.mark.parametrize("model", ['google-bert/bert-base-uncased', 'tohoku-nlp/bert-base-japanese', 'answerdotai/ModernBERT-base', 'sbintuitions/modernbert-ja-130m'])
def test_collator(model: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model)
    preprocessor = Preprocessor(tokenizer, labels, format="iob2", extend_context=False)

    def preprocess(documents: dict[str, Any]) -> dict[str, Any]:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    column_names = next(iter(raw_datasets.values())).column_names
    splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)
    collator = Collator(tokenizer)
    dataloader_params = {
        "batch_size": 2,
        "collate_fn": collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
    }
    dataloader_params["sampler"] = SequentialSampler(splits['train'])
    dataloader_params["drop_last"] = training_args.dataloader_drop_last
    dataloader_params["prefetch_factor"] = training_args.dataloader_prefetch_factor
    dataloader = DataLoader(splits['train'], **dataloader_params)

    for batch in dataloader:
        assert isinstance(batch, BatchEncoding)
        keys = list(batch.keys())
        assert keys == ['input_ids', 'attention_mask', 'id', 'labels']
        assert isinstance(batch['input_ids'], torch.Tensor)
        assert isinstance(batch['attention_mask'], torch.Tensor)
        assert isinstance(batch['labels'], torch.Tensor)
        assert batch['input_ids'].size(0) == batch['attention_mask'].size(0) == batch['labels'].size(0) == 2
        assert batch['input_ids'].size(1) == batch['attention_mask'].size(1)
        assert batch['input_ids'].size(1) == batch['labels'].size(1)
