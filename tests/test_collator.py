
import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BatchEncoding, TrainingArguments

from src.data import Preprocessor, get_dataloader, get_sequence_labels, get_splits

TEST_MODEL = [
    "google-bert/bert-base-uncased",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    'llm-jp/llm-jp-modernbert-base',
    'tohoku-nlp/bert-base-japanese-v3',
    'sbintuitions/modernbert-ja-130m'
    ## TODO: xlm-roberta with pretokenize is not working due to tokenizing error.
    # "FacebookAI/xlm-roberta-base",
]
dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = load_dataset("json", data_files={"train": dataset_path}, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format = "iob2")
training_args = TrainingArguments(output_dir=".tmp/")


@pytest.mark.parametrize("model", TEST_MODEL)
@pytest.mark.parametrize("pretokenize", [True, False])
def test_collator(model: str, pretokenize: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model)
    preprocessor = Preprocessor(tokenizer, labels, format="iob2", pretokenize=pretokenize)
    splits = get_splits(raw_datasets, preprocessor, training_args)
    dataloader = get_dataloader(splits['train'], tokenizer, 2, training_args)

    for batch in dataloader:
        assert isinstance(batch, BatchEncoding)
        keys = list(batch.keys())
        assert keys == ['input_ids', 'attention_mask', 'labels', 'prediction_mask']
        assert isinstance(batch['input_ids'], torch.Tensor)
        assert isinstance(batch['attention_mask'], torch.Tensor)
        assert isinstance(batch['labels'], torch.Tensor)
        assert batch['input_ids'].size(0) == batch['attention_mask'].size(0) == batch['labels'].size(0) == 2
        assert batch['input_ids'].size(1) == batch['attention_mask'].size(1)
        assert batch['input_ids'].size(1) == batch['labels'].size(1)
        assert batch['labels'].size(0) == batch['prediction_mask'].size(0)
        assert batch['labels'].size(1) == batch['prediction_mask'].size(1)
