import pytest
import torch
from transformers import AutoTokenizer, TrainingArguments

from src import BertNER, BertNERConfig, get_splits, read_dataset
from src.data import Preprocessor, get_dataloader, get_sequence_labels
from src.modeling import TokenModel
from src.modeling.model_output import TokenClassifierOutput

TEST_MODEL = [
    'google-bert/bert-base-uncased',
    'microsoft/deberta-v3-base',
    'FacebookAI/roberta-base',
    'answerdotai/ModernBERT-base',
    'llm-jp/llm-jp-modernbert-base',
    'tohoku-nlp/bert-base-japanese-v3',
    'sbintuitions/modernbert-ja-130m'
    ## TODO: xlm-roberta with pretokenize is not working due to tokenizing error.
    # "FacebookAI/xlm-roberta-base",
]
dataset_path = "tests/test_data/dataset_toy.jsonl"
raw_datasets = read_dataset(train_file=dataset_path, cache_dir='tmp/')
label_set = set()
for document in raw_datasets["train"]:
    for example in document["examples"]:
        for entity in example["entities"]:
            label_set.add(entity["label"])
labels = get_sequence_labels(sorted(label_set), format='iob2')
training_arguments = TrainingArguments(output_dir='tmp/')
format='iob2'


class TestBertNER:
    @pytest.mark.parametrize('model_name', TEST_MODEL)
    @pytest.mark.parametrize('no_crf', [True, False])
    def test__init__(self, model_name: str, no_crf: bool) -> None:
        config = BertNERConfig(model_name, freeze_bert=True, no_crf = no_crf)
        config.num_labels = len(labels)
        model = BertNER(config, encoder_from_pretrained=False)

        assert isinstance(model, BertNER)
        assert isinstance(model.model, TokenModel)
        assert model.config.encoder_config

    @pytest.mark.parametrize('model_name', TEST_MODEL)
    def test_device(self, model_name: str) -> None:
        config = BertNERConfig(model_name, freeze_bert=True)
        config.num_labels = len(labels)
        model = BertNER(config)
        assert model.device == torch.device('cpu')

    @pytest.mark.parametrize('model_name', TEST_MODEL)
    @pytest.mark.parametrize('no_crf', [True, False])
    @pytest.mark.parametrize('pretokenize', [True, False])
    def test_forward(self, model_name: str, no_crf: bool, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = BertNERConfig(model_name, freeze_bert=True, no_crf = no_crf)
        config.num_labels = len(labels)
        model = BertNER(config)
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
        model = BertNER(config)
        preprocessor = Preprocessor(tokenizer, labels, format=format, pretokenize=pretokenize)
        splits = get_splits(raw_datasets, preprocessor, training_arguments)
        dataloader = get_dataloader(splits['train'], tokenizer, 2, training_arguments)

        for batch in dataloader:
            _ = batch.pop('labels')
            outputs = model.decode(**batch)
            assert isinstance(outputs, TokenClassifierOutput)
            assert isinstance(outputs.predictions, torch.Tensor)
            assert outputs.predictions.size(0) == 2
            for pred, seq_mask in zip(outputs.predictions, batch['prediction_mask']):
                assert pred[pred != -100].size(0) == seq_mask.count_nonzero().item()

        for batch in dataloader:
            outputs = model.decode(**batch)
            assert isinstance(outputs, TokenClassifierOutput)
            assert isinstance(outputs.predictions, torch.Tensor)
            assert outputs.predictions.size(0) == 2
            for pred, seq_mask in zip(outputs.predictions, batch['prediction_mask']):
                assert pred[pred != -100].size(0) == seq_mask.count_nonzero().item()


    @pytest.mark.parametrize('model_name', TEST_MODEL)
    def test_resize_token_embedding(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = BertNERConfig(model_name)
        config.num_labels = len(labels)
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.id2label = {i: label for i, label in enumerate(labels)}
        model = BertNER(config)

        tokenizer.add_tokens(['TEST'])
        parameters = model.resize_token_embeddings(len(tokenizer))
        assert parameters.num_embeddings == len(tokenizer)
        assert model.config.encoder_config.vocab_size == len(tokenizer)


    @pytest.mark.parametrize('model_name', TEST_MODEL)
    def test_save_and_load(self, model_name: str) -> None:
        config = BertNERConfig(model_name)
        config.num_labels = len(labels)
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.id2label = {i: label for i, label in enumerate(labels)}
        model = BertNER(config)

        model.save_pretrained("test_model")
        del model, config

        model = BertNER.from_pretrained("test_model")
        config = model.config
        assert isinstance(config, BertNERConfig)
        assert isinstance(model, BertNER)
        assert isinstance(model.model, TokenModel)
        assert config.num_labels == len(labels)
        assert config.label2id == {label: i for i, label in enumerate(labels)}
        assert config.id2label == {i: label for i, label in enumerate(labels)}
