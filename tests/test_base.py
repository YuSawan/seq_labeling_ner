import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments

from src import BertNERConfig, get_splits, read_dataset
from src.data import Preprocessor, get_dataloader, get_sequence_labels
from src.modeling import TokenModel
from src.modeling.crf import CRF
from src.modeling.encoder import Encoder
from src.modeling.model_output import TokenClassifierOutput
from src.modeling.pooler import get_last_layer

TEST_MODEL = ['google-bert/bert-base-uncased', 'answerdotai/ModernBERT-base']
dataset_path = 'tests/test_data/dataset_toy.jsonl'
raw_datasets = read_dataset(train_file=dataset_path, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format='iob2')
training_arguments = TrainingArguments(output_dir='tmp/')
format='iob2'


class TestTokenModel:
    @pytest.mark.parametrize('model_name', TEST_MODEL)
    @pytest.mark.parametrize('classifier_dropout', [0.5, None])
    @pytest.mark.parametrize('bias_O', [6., None])
    @pytest.mark.parametrize('no_crf', [True, False])
    def test__init__(self, model_name: str, classifier_dropout: float | None, bias_O: float | None, no_crf: bool) -> None:
        config = BertNERConfig(
            model_name,
            pooler = 'last',
            freeze_bert = True,
            classifier_dropout=classifier_dropout,
            no_crf = no_crf,
            bias_O = bias_O
        )
        config.num_labels = len(labels)
        model = TokenModel(config)

        assert isinstance(model, TokenModel)
        assert isinstance(model.encoder, Encoder)
        assert isinstance(model.config, BertNERConfig)
        assert model.config.encoder_config
        assert model.encoder.pooler is get_last_layer
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.classifier, nn.Linear)
        if bias_O is not None:
            assert model.classifier.bias.data[0] == bias_O
        else:
            assert model.classifier.bias.data[0] != bias_O

        for param in model.encoder.parameters():
            assert not param.requires_grad
        for param in model.classifier.parameters():
            assert param.requires_grad

        if classifier_dropout:
            model.dropout.p == classifier_dropout
        else:
            if model_name == 'answerdotai/ModernBERT-base':
                model.dropout.p == model.config.encoder_config.mlp_dropout
            else:
                model.dropout.p == model.config.encoder_config.hidden_dropout_prob

        if no_crf:
            assert not hasattr(model, 'crf')
        else:
            assert hasattr(model, 'crf')
            assert isinstance(model.crf, CRF)


    @pytest.mark.parametrize('model_name', TEST_MODEL)
    @pytest.mark.parametrize('no_crf', [True, False])
    @pytest.mark.parametrize('pretokenize', [True, False])
    def test_forward(self, model_name: str, no_crf: bool, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = BertNERConfig(model_name, freeze_bert=True, no_crf = no_crf)
        config.num_labels = len(labels)
        model = TokenModel(config)
        preprocessor = Preprocessor(tokenizer, labels, format=format, pretokenize=pretokenize)
        splits = get_splits(raw_datasets, preprocessor, training_arguments)
        dataloader = get_dataloader(splits['train'], tokenizer, 2, training_arguments)

        for batch in dataloader:
            outputs = model(**batch)
            assert isinstance(outputs, TokenClassifierOutput)
            assert isinstance(outputs.loss, torch.Tensor)
            assert outputs.logits is not None and outputs.prediction_mask is not None
            assert outputs.logits.size(0) == 2 and outputs.logits.size(-1) == config.num_labels
            assert outputs.predictions is None

        for batch in dataloader:
            _ = batch.pop('labels')
            outputs = model(**batch)
            assert isinstance(outputs, TokenClassifierOutput)
            assert outputs.loss is None and outputs.logits is not None and outputs.prediction_mask is not None
            assert outputs.logits.size(0) == 2 and outputs.logits.size(-1) == config.num_labels
            assert outputs.predictions is None


    @pytest.mark.parametrize('model_name', TEST_MODEL)
    @pytest.mark.parametrize('no_crf', [True, False])
    @pytest.mark.parametrize('pretokenize', [True, False])
    def test_decode(self, model_name: str, no_crf: bool, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = BertNERConfig(model_name, freeze_bert=True, no_crf = no_crf)
        config.num_labels = len(labels)
        model = TokenModel(config)
        preprocessor = Preprocessor(tokenizer, labels, format=format, pretokenize=pretokenize)
        splits = get_splits(raw_datasets, preprocessor, training_arguments)
        dataloader = get_dataloader(splits['train'], tokenizer, 2, training_arguments)

        for batch in dataloader:
            outputs = model(**batch)
            predictions = model.decode(outputs.logits, outputs.prediction_mask)
            assert isinstance(predictions, torch.Tensor)
            assert predictions.size(0) == 2
            for pred, seq_mask in zip(predictions, outputs.prediction_mask):
                assert pred[pred != -100].size(0) == seq_mask.count_nonzero().item()
