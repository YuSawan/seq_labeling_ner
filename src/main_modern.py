from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    logging,
    set_seed,
)

from data import Collator, Preprocessor, get_sequence_labels
from models import (
    ModernBertCRF,
    ModernBertForTokenClassification,
    ModernBertLSTM,
    ModernBertLSTMCRF,
)
from training import LoggerCallback, TokenClassificationTrainer, setup_logger

logger = logging.get_logger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    format: str = "iob2"
    model: str = "answerdotai/ModernBERT-base"
    cache_dir: Optional[str] = 'tmp/'
    max_context_length: int = 512
    freeze_bert: bool = False
    lstm_layers: int = 1
    lstm_hidden_size: int = 100
    no_crf: bool = False
    pretokenize: bool = False
    extend_context: bool = False


def get_model_and_kwargs_for_args(
        args: Arguments,
        training: bool = True,
) -> tuple[Union[type[ModernBertLSTM], type[ModernBertLSTMCRF], type[ModernBertForTokenClassification], type[ModernBertCRF]], dict[str, Any]]:
    """Given the parsed arguments, returns the correct model class and model
    args.
    Args:
        args: a Namespace object (from parsed argv command).
        training: if True, sets a high initialization value for classifier bias
            parameter after model initialization.
    """
    bias_O = 6. if training else None
    model_args = {
        'bias_O': bias_O,
    }

    if args.freeze_bert:
        # Possible models: BERT-LSTM or BERT-LSTM-CRF
        model_args['lstm_layers'] = args.lstm_layers
        model_args['lstm_hidden_size'] = args.lstm_hidden_size
        if args.no_crf:
            return ModernBertLSTM, model_args
        else:
            return ModernBertLSTMCRF, model_args
    else:
        # Possible models: BertForNERClassification or BertCRF
        if args.no_crf:
            return ModernBertForTokenClassification, model_args
        else:
            return ModernBertCRF, model_args


def main(args: Arguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")
    set_seed(training_args.seed)

    data_files = {k: getattr(args, f"{k}_file") for k in ["train", "validation", "test"]}
    data_files = {k: v for k, v in data_files.items() if v is not None}
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    config = AutoConfig.from_pretrained(args.model)
    if config.label2id == PretrainedConfig().label2id:
        if "train" not in raw_datasets:
            raise RuntimeError("Cannot retrieve labels from dataset")
        label_set = set()
        for document in raw_datasets["train"]:
            for example in document["examples"]:
                for entity in example["entities"]:
                    label_set.add(entity["label"])
        labels = get_sequence_labels(sorted(label_set), args.format)
        config.num_labels = len(labels)
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.id2label = {i: label for i, label in enumerate(labels)}
        config.format = args.format
        logger.info(f"labels: {labels}")
        logger.info(f"format: {args.format}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=args.max_context_length)
    preprocessor = Preprocessor(
        tokenizer,
        labels=[v for _, v in sorted(config.id2label.items())],
        format=config.format,
        extend_context=args.extend_context,
        pretokenize=args.pretokenize,
    )

    def preprocess(documents: dict[str, Any]) -> dict[str, Any]:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    model_class, model_kwargs = get_model_and_kwargs_for_args(args, training=True if training_args.do_train else False)
    model = model_class.from_pretrained(args.model, config=config, **model_kwargs)

    trainer = TokenClassificationTrainer(
        model = model,
        args=training_args,
        train_dataset = splits['train'],
        eval_dataset = splits['validation'],
        data_collator = Collator(tokenizer)
    )
    trainer.add_callback(LoggerCallback(logger))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        pass
        # results = evaluate(model=model, dataset=splits['test'])

    if training_args.do_predict:
        pass
        # predicts = predict(
        #     model=model,
        #     dataset=splits['validation'].remove_columns('negatives') if args.negative != 'inbatch' else splits['validation'],
        #     retriever=retriever,
        #     reset_index=False if training_args.do_eval else True
        # )

if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    if args.validation_file is None:
        training_args.evaluation_strategy = "no"
    main(args, training_args)
