import pytest
import torch
from transformers import AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from src import BertNERConfig
from src.modeling.encoder import Bert, Encoder
from src.modeling.pooler import POOLERS

TEST_MODEL = [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    'llm-jp/llm-jp-modernbert-base',
    'tohoku-nlp/bert-base-japanese-v3',
    'sbintuitions/modernbert-ja-130m'
]


@pytest.mark.parametrize("model_name", TEST_MODEL)
@pytest.mark.parametrize("from_pretrained", [True, False])
def test_Bert(model_name: str, from_pretrained: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BertNERConfig(model_name)
    bert = Bert(config.model_name, config.encoder_config, from_pretrained)
    assert bert.model.name_or_path == model_name
    assert isinstance(bert, Bert)
    assert hasattr(bert, "config") and isinstance(bert.config, PretrainedConfig)

    encodings = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = bert(**encodings)
    assert isinstance(output, BaseModelOutputWithPoolingAndCrossAttentions) or isinstance(output, BaseModelOutput)
    assert hasattr(output, "hidden_states")

@pytest.mark.parametrize("model_name", TEST_MODEL)
@pytest.mark.parametrize('pooler', ['sum', 'last', 'concat'])
def test_Encoder(model_name: str, pooler: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BertNERConfig(model_name)
    config.pooler = pooler
    pooler_func = POOLERS[pooler]

    encoder = Encoder(config)
    assert isinstance(encoder, Encoder)
    assert hasattr(encoder, "config") and isinstance(encoder.config, PretrainedConfig)
    assert hasattr(encoder, "pooler") and encoder.pooler is pooler_func
    assert isinstance(encoder.bert, Bert)
    assert encoder.bert.model.name_or_path == model_name

    encodings = tokenizer("Hello, my dog is cute", return_tensors="pt")

    output = encoder(**encodings)
    assert isinstance(output, torch.Tensor)

    if pooler in ['last', 'sum']:
        bert_hidden_size = encoder.config.hidden_size
    if pooler == 'concat':
        bert_hidden_size = 4 * encoder.config.hidden_size
    assert output.size() == (1, encodings.input_ids.size(1), bert_hidden_size)
