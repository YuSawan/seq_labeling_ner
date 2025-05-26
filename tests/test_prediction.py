from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from src import BertNER, BertNERConfig, get_splits, read_dataset
from src.data import Collator, Preprocessor, get_sequence_labels
from src.prediction import predict
from src.prediction.pred import _word_offsets, label_to_charspan
from src.training import TokenClassificationTrainer

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = read_dataset(train_file=dataset_path, validation_file=dataset_path)
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format='iob2')
training_args = TrainingArguments(output_dir="test_model/")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertNERConfig(model_name, freeze_bert=True)
config.num_labels = len(labels)
config.label2id = {label: i for i, label in enumerate(labels)}
config.id2label = {i: label for i, label in enumerate(labels)}
model = BertNER(config, encoder_from_pretrained=False)

preprocessor = Preprocessor(tokenizer, labels, format='iob2', pretokenize=True)
training_args = TrainingArguments(output_dir="test_model/", remove_unused_columns=False, report_to='none')
splits = get_splits(raw_datasets, preprocessor, training_args)

trainer = TokenClassificationTrainer(
    model = model,
    args=training_args,
    train_dataset = splits['train'],
    eval_dataset = splits['validation'],
    data_collator = Collator(tokenizer),
    seq_scheme='iob2',
    classifier_lr=1e-3
)


def test_word_offsets() -> None:
    for example in splits['validation']:
        prediction_mask = example['prediction_mask'][1:-1]
        offsets = example['offsets']
        assert len(offsets) == len(prediction_mask)
        assert prediction_mask[0] == 1
        word_offsets = _word_offsets(offsets, prediction_mask)
        assert len(word_offsets) == len([p for p in prediction_mask if p == 1])


def test_labels_to_charspan() -> None:
    golds = {}
    for document in raw_datasets['validation']:
        for example in document['examples']:
            pid = example['id']
            entities = []
            for ent in example['entities']:
                entities.append((ent['start'], ent['end'], ent['label']))
            golds[pid] = entities

    for example in splits['validation']:
        pid = example['id']
        labels = [config.id2label[label] for mask, label in zip(example['prediction_mask'][1:-1], example['labels'][1:-1]) if mask == 1]
        word_offsets = _word_offsets(example['offsets'], example['prediction_mask'][1:-1])
        assert len(word_offsets) == len(labels)
        char_spans = label_to_charspan(labels, word_offsets, 'iob2')
        if char_spans != golds[pid]:
            assert char_spans + [(11, 16, 'ORG')] == golds[pid]


def test_predict() -> None:
    _ = trainer.evaluate()
    logits = trainer.last_prediction.predictions
    predictions = predict(logits, splits["validation"], config.id2label, 'iob2')
    assert isinstance(predictions, dict)
    for pid, pred in predictions.items():
        assert isinstance(pid, str)
        assert isinstance(pred, set)
