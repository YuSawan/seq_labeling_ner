import copy

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer, BatchEncoding

from src.data import Preprocessor, get_sequence_labels
from src.data.dataset import (
    _offset_to_seqlabels,
    _remove_nested_mentions,
)
from src.data.tokenizer import BertJapaneseTokenizerFast

TEST_MODEL = [
    "google-bert/bert-base-uncased",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    'llm-jp/llm-jp-modernbert-base',
    'tohoku-nlp/bert-base-japanese-v3',
    'sbintuitions/modernbert-ja-130m',
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


class TestPreprocessor:
    @pytest.mark.parametrize("model", TEST_MODEL)
    @pytest.mark.parametrize("format", ["iob1", "iob2", "ioe1", "ioe2", "iobes", "bilou"])
    def test___init__(self, model: str, format: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        labels = get_sequence_labels(sorted(label_set), format)

        preprocessor = Preprocessor(tokenizer, labels, format=format)
        if model == 'tohoku-nlp/bert-base-japanese-v3':
            assert isinstance(preprocessor._fast_tokenizer, BertJapaneseTokenizerFast)
        else:
            preprocessor._fast_tokenizer.is_fast
        assert sorted(preprocessor.types) == sorted(['ORG', 'PER', 'MISC', 'LOC'])
        assert len(preprocessor.labels) == 9 if format not in ["iobes", "bilou"] else 17
        assert len(preprocessor.label2id.keys()) == 9 if format not in ["iobes", "bilou"] else 17
        if model in ['google-bert/bert-base-uncased', 'tohoku-nlp/bert-base-japanese-v3']:
            assert preprocessor.max_sequence_length == 512
            assert preprocessor.max_num_tokens == 510
        if model == 'answerdotai/ModernBERT-base':
            assert preprocessor.max_sequence_length == 8192
            assert preprocessor.max_num_tokens == 8190
        if model == 'sbintuitions/modernbert-ja-130m':
            assert preprocessor.max_sequence_length == 1000000000000000019884624838656
            assert preprocessor.max_num_tokens == 1000000000000000019884624838654

    @pytest.mark.parametrize("model", TEST_MODEL)
    @pytest.mark.parametrize("pretokenize", [True, False])
    def test_tokenize(self, model: str, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        labels = get_sequence_labels(sorted(label_set), format="iob2")
        preprocessor = Preprocessor(tokenizer, labels, format="iob2", pretokenize=pretokenize)
        for document in raw_datasets['train']['examples']:
            segments = [example['word_positions'] for example in document] if pretokenize else None
            for i, tokenization in enumerate(preprocessor.tokenize([e["text"] for e in document], segments)):
                assert isinstance(tokenization, dict)
                assert "token_ids" in tokenization
                assert "context_boundary" in tokenization
                assert "offsets" in tokenization
                assert tokenization["context_boundary"][0] == 0
                assert "prediction_mask" in tokenization
                assert len(tokenization["prediction_mask"]) == len(tokenization['token_ids'])
                if pretokenize:
                    tokenization["prediction_mask"] != [1] * len(tokenization['token_ids'])
                else:
                    tokenization["prediction_mask"] == [1] * len(tokenization['token_ids'])


    @pytest.mark.parametrize("model", TEST_MODEL)
    @pytest.mark.parametrize("pretokenize", [True, False])
    def test___batch_spans(self, model: str, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        labels = get_sequence_labels(sorted(label_set), format="iob2")
        preprocessor = Preprocessor(tokenizer, labels, format="iob2", pretokenize=pretokenize)
        for document in raw_datasets['train']['examples']:
            segments = [example['word_positions'] for example in document] if pretokenize else None
            for example, tokenization in zip(document, preprocessor.tokenize([e["text"] for e in document], segments)):
                entity_map = {(ent["start"], ent["end"]): ent["label"] for ent in example["entities"]}
                for token_spans, char_spans in preprocessor._batch_spans(example, tokenization):
                    entities = []
                    assert isinstance(token_spans, list)
                    assert isinstance(char_spans, list)
                    assert len(token_spans) == len(char_spans)
                    for (t_s, t_e), (c_s, c_e) in zip(token_spans, char_spans):
                        assert type(t_s) is int and type(t_e) is int
                        assert type(c_s) is int and type(c_e) is int
                        entity_type = entity_map.pop((c_s, c_e), None)
                        if entity_type:
                            entities.append((t_s, t_e, entity_type))
                    entities_copy = copy.copy(entities)
                    entities_copy.sort(key=lambda x: x[0])
                    assert entities == entities_copy

    @pytest.mark.parametrize("model", TEST_MODEL)
    @pytest.mark.parametrize("pretokenize", [True, False])
    def test___call__(self, model: str, pretokenize: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        labels = get_sequence_labels(sorted(label_set), format="iob2")
        preprocessor = Preprocessor(tokenizer, labels, format="iob2", pretokenize=pretokenize)
        cnt = 0
        for document in raw_datasets['train']['examples']:
            for encodings in preprocessor(document):
                assert isinstance(encodings, BatchEncoding)
                assert len(encodings['input_ids']) == len(encodings["labels"])
                assert len(encodings["labels"]) == len(encodings['attention_mask'])
                assert hasattr(encodings, "prediction_mask")
                assert len(encodings["prediction_mask"]) == len(encodings['attention_mask'])
                assert hasattr(encodings, "offsets")
                assert len(encodings["offsets"]) == len(encodings['attention_mask']) - 2
                if not pretokenize:
                    assert encodings["prediction_mask"] == [0] + [1] * (len(encodings['input_ids'])-2) + [False]
                cnt += 1
        assert cnt == 8


@pytest.mark.parametrize(
        'entities',
        [[(0, 3, 'ORG'), (0, 2, 'LOC'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')], [(0, 3, 'ORG'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')]],
)
def test__remove_nested_mentions(entities: list[tuple[int, int, str]]) -> None:
    surface_entities, nested_entities = _remove_nested_mentions(entities)
    if entities == [(0, 3, 'ORG'), (0, 2, 'LOC'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')]:
        assert surface_entities == [(0, 3, 'ORG'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')]
        assert nested_entities == [(0, 2, 'LOC')]
    else:
        assert surface_entities == [(0, 3, 'ORG'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')]
        assert nested_entities == []

@pytest.mark.parametrize("format", ["iob1", "iob2", "ioe1", "ioe2", "iobes", "bilou"])
def test___offset_to_seqlabel(format: str) -> None:
    entities = [(0, 3, 'ORG'), (4, 5, 'LOC'), (5, 6, 'LOC'), (6, 7, 'LOC')]
    labels = _offset_to_seqlabels(entities, format=format, token_len=8)
    if format == 'iob1':
        assert labels == ['I-ORG', 'I-ORG', 'I-ORG', 'O', 'I-LOC', 'B-LOC', 'B-LOC', 'O']
    if format == 'iob2':
        assert labels == ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'B-LOC', 'B-LOC', 'O']
    if format == 'ioe1':
        assert labels == ['I-ORG', 'I-ORG', 'I-ORG', 'O', 'E-LOC', 'E-LOC', 'I-LOC', 'O']
    if format == 'ioe2':
        assert labels == ['I-ORG', 'I-ORG', 'E-ORG', 'O', 'E-LOC', 'E-LOC', 'E-LOC', 'O']
    if format == 'iobes':
        assert labels == ['B-ORG', 'I-ORG', 'E-ORG', 'O', 'S-LOC', 'S-LOC', 'S-LOC', 'O']
    if format == 'bilou':
        assert labels == ['B-ORG', 'I-ORG', 'L-ORG', 'O', 'U-LOC', 'U-LOC', 'U-LOC', 'O']
