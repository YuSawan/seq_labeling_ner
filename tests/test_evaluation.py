import pytest
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from src import BertNER, BertNERConfig, get_splits, read_dataset
from src.data import Collator, Preprocessor, get_sequence_labels
from src.evaluation import compute_score, evaluate
from src.prediction import predict
from src.training import TokenClassificationTrainer

schemes = ['iob1', 'iob1', 'ioe1', 'ioe2', 'iobes', 'bilou']

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = read_dataset(train_file=dataset_path, validation_file=dataset_path)
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])


@pytest.mark.parametrize('scheme', schemes)
def test_compute_score(scheme: str) -> None:
    labels = get_sequence_labels(sorted(label_set), format=scheme)
    id2label = {i: label for i, label in enumerate(labels)}
    preprocessor = Preprocessor(tokenizer, labels, format=scheme, pretokenize=True)

    true_labels: list[list[str]] = []
    for document in raw_datasets['train']['examples']:
        for encodings in preprocessor(document):
            labels = encodings['labels']
            masks = encodings['prediction_mask']
            assert isinstance(labels[0], int)
            true_label = [ id2label[label] for label, m in zip(labels, masks) if m != 0 ]
            true_labels.append(true_label)

    results = compute_score(
        predictions=true_labels,
        references=true_labels,
        mode = 'strict',
        scheme=scheme
    )
    assert results['overall_precision'] == 1.0
    assert results['overall_recall'] == 1.0
    assert results['overall_f1'] == 1.0
    assert results['overall_accuracy'] == 1.0
    for label_type in ['PER', 'LOC', 'ORG', 'MISC']:
        assert results[label_type]['precision'] == 1.0
        assert results[label_type]['recall'] == 1.0
        assert results[label_type]['f1'] == 1.0
        if label_type == 'PER':
            assert results[label_type]['number'] == 1
        if label_type == 'LOC':
            assert results[label_type]['number'] == 2
        if label_type == 'ORG':
            assert results[label_type]['number'] == 5
        if label_type == 'MISC':
            assert results[label_type]['number'] == 1


def test_evaluate() -> None:
    labels = get_sequence_labels(sorted(label_set), format='iob2')

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
    _ = trainer.evaluate()
    logits = trainer.last_prediction.predictions
    predictions = predict(logits, splits["validation"], config.id2label, scheme='iob2')
    new_metrics = evaluate(predictions, raw_datasets["validation"])
    assert isinstance(new_metrics, dict)
    assert "overall_precision" in new_metrics
    assert "overall_recall" in new_metrics
    assert "overall_f1" in new_metrics

    for label in sorted(label_set):
        assert f"{label}_precision" in new_metrics
        assert f"{label}_recall" in new_metrics
        assert f"{label}_f1" in new_metrics
