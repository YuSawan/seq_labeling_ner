import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from typing import Optional

import yaml
from transformers import HfArgumentParser, TrainingArguments


def load_config_as_namespace(config_file: str | os.PathLike) -> Namespace:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


@dataclass
class DatasetArguments:
    """Dataset arguments."""
    train_file : str
    validation_file : str
    test_file: str
    format: str
    pretokenize: bool


@dataclass
class ModelArguments:
    """Model arguments."""
    model_name: str
    model_max_length: int
    pooler: str
    freeze_bert: bool
    classifier_lr: float
    classifier_dropout: float
    no_crf: bool
    lstm_layers: Optional[int] = None
    lstm_hidden_size: Optional[int] = None
    weight_O: Optional[float] = 0.01
    bias_O: Optional[float] = None
    cache_dir: Optional[str] = None
    prev_path: Optional[str] = None


def parse_args() -> tuple[DatasetArguments, ModelArguments, TrainingArguments]:
    parser = ArgumentParser()
    hfparser = HfArgumentParser(TrainingArguments)

    parser.add_argument(
        "--config_file", metavar="FILE", required=True
    )
    parser.add_argument(
        '--format', type=str, default=None,
    )
    parser.add_argument(
        '--prev_path', metavar="DIR", default=None
    )

    args, extras = parser.parse_known_args()
    config = vars(load_config_as_namespace(args.config_file))
    training_args = hfparser.parse_args_into_dataclasses(extras)[0]

    data_config = config.pop("dataset")
    model_config = config.pop("model")

    arguments = DatasetArguments(**data_config)
    model_args = ModelArguments(**model_config)
    training_args = replace(training_args, **config)

    arguments.format = args.format if args.format else arguments.format
    model_args.prev_path = args.prev_path if args.prev_path else model_args.prev_path

    return arguments, model_args, training_args
