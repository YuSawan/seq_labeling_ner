import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from transformers.trainer_utils import PredictionOutput

from src import BertNER, BertNERConfig, get_splits, read_dataset
from src.data import Collator, Preprocessor, get_dataloader, get_sequence_labels
from src.training import TokenClassificationTrainer

model_name = "google-bert/bert-base-uncased"
dataset_path = "tests/test_data/dataset_toy.jsonl"
format = 'iob2'
raw_datasets = read_dataset(train_file=dataset_path, validation_file=dataset_path)
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format=format)
training_args = TrainingArguments(output_dir="test_model/")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertNERConfig(model_name, freeze_bert=True)
config.num_labels = len(labels)
config.label2id = {label: i for i, label in enumerate(labels)}
config.id2label = {i: label for i, label in enumerate(labels)}
model = BertNER(config, encoder_from_pretrained=False)

preprocessor = Preprocessor(tokenizer, labels, format=format, pretokenize=True)
training_args = TrainingArguments(output_dir="test_model/", remove_unused_columns=False, num_train_epochs=1)
splits = get_splits(raw_datasets, preprocessor, training_args)
dataloader = get_dataloader(splits['train'], tokenizer, 2, training_args)


class TestTokenClassificationTrainer:
    def test__init__(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        assert trainer.seq_scheme == format

    def test_get_no_classifier_parameter_name(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        parameters = trainer.get_no_classifier_parameter_name(trainer.model)
        for p in parameters:
            assert 'crf' not in p
            assert 'classifier' not in p

    def test_create_optimizer(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        optimizer = trainer.create_optimizer()
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.param_groups[0]['lr'] == training_args.learning_rate
        assert optimizer.param_groups[0]['weight_decay'] == training_args.weight_decay
        assert optimizer.param_groups[1]['lr'] == 1e-3
        assert optimizer.param_groups[1]['weight_decay'] == training_args.weight_decay
        assert optimizer.param_groups[2]['lr'] == training_args.learning_rate
        assert optimizer.param_groups[2]['weight_decay'] == 0.0

    def test_prediction_step(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        eval_dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)
        for inputs in eval_dataloader:
            loss, logits, labels = trainer.prediction_step(trainer.model, inputs, prediction_loss_only=False)
            assert loss is None
            assert isinstance(logits, tuple)
            assert isinstance(labels, torch.Tensor)
            loss, logits, predictions, prediction_mask = logits
            assert isinstance(loss, torch.Tensor)
            assert isinstance(logits, torch.Tensor)
            assert isinstance(predictions, torch.Tensor)
            assert isinstance(prediction_mask, torch.Tensor)

    def test_evaluate(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        metrics = trainer.evaluate()
        assert isinstance(metrics, dict)
        assert 'eval_loss' in metrics.keys()
        assert 'eval_precision' in metrics.keys()
        assert 'eval_recall' in metrics.keys()
        assert 'eval_f1' in metrics.keys()
        assert 'eval_accuracy' in metrics.keys()

        logits = trainer.last_prediction.predictions
        assert isinstance(logits, tuple)
        loss, logits, predictions, prediction_mask = logits
        assert isinstance(loss, np.ndarray)
        assert isinstance(logits, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(prediction_mask, np.ndarray)

    def test_predict(self) -> None:
        trainer = TokenClassificationTrainer(
            model = model,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = Collator(tokenizer),
            seq_scheme=format,
            classifier_lr=1e-3
        )
        results = trainer.predict(splits['validation'])
        assert isinstance(results, PredictionOutput)
        metrics = results.metrics
        assert isinstance(metrics, dict)
        assert 'test_loss' in metrics.keys()
        assert 'test_precision' in metrics.keys()
        assert 'test_recall' in metrics.keys()
        assert 'test_f1' in metrics.keys()
        assert 'test_accuracy' in metrics.keys()

        logits = trainer.last_prediction.predictions
        assert isinstance(logits, tuple)
        loss, logits, predictions, prediction_mask = logits
        assert isinstance(loss, np.ndarray)
        assert isinstance(logits, np.ndarray)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(prediction_mask, np.ndarray)
