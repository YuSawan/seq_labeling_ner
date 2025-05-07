import os

from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    TrainingArguments,
    logging,
    set_seed,
)

from src import BertNER, BertNERConfig, get_splits, parse_args, read_dataset
from src.argparser import DatasetArguments, ModelArguments
from src.data import Collator, Preprocessor, get_sequence_labels
from src.evaluation import evaluate, submit_wandb_evaluate
from src.prediction import predict, submit_wandb_predict
from src.training import LoggerCallback, TokenClassificationTrainer, setup_logger

logger = logging.get_logger(__name__)
TOKEN = os.environ.get('TOKEN', True)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {data_args}")
    logger.info(f"args: {model_args}")
    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)
    if not model_args.prev_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, model_max_length=model_args.model_max_length, token=TOKEN)
        config = BertNERConfig(
            model_args.model_name,
            pooler=model_args.pooler,
            freeze_bert=model_args.freeze_bert,
            lstm_layers=model_args.lstm_layers,
            lstm_hidden_size=model_args.lstm_hidden_size,
            classifier_dropout=model_args.classifier_dropout,
            no_crf=model_args.no_crf,
            weight_O=model_args.weight_O,
            bias_O=model_args.bias_O
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.prev_path)
        model = BertNER.from_pretrained(model_args.prev_path)
        config = model.config

    raw_datasets = read_dataset(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        model_args.cache_dir,
    )
    if config.label2id == PretrainedConfig().label2id:
        if "train" not in raw_datasets:
            raise RuntimeError("Cannot retrieve labels from dataset")
        label_set = set()
        for document in raw_datasets["train"]:
            for example in document["examples"]:
                for entity in example["entities"]:
                    label_set.add(entity["label"])
        labels = get_sequence_labels(["O"] + sorted(label_set), data_args.format)
        config.num_labels = len(labels)
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.id2label = {i: label for i, label in enumerate(labels)}
        logger.info(f"labels: {labels}")
        logger.info(f"format: {data_args.format}")
        model = BertNER(config)

    preprocessor = Preprocessor(
        tokenizer,
        labels=[v for _, v in sorted(config.id2label.items())],
        format=data_args.format,
        pretokenize=data_args.pretokenize,
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)

    trainer = TokenClassificationTrainer(
        model = model,
        args=training_args,
        train_dataset = splits['train'],
        eval_dataset = splits['validation'],
        data_collator = Collator(tokenizer),
        seq_scheme=data_args.format,
        classifier_lr=model_args.classifier_lr,
    )
    trainer.add_callback(LoggerCallback(logger))

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model(training_args.output_dir)
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate(splits['test'])
        logits = trainer.last_prediction.predictions
        predictions = predict(logits, splits["test"], config.id2label, data_args.format)
        new_metrics = evaluate(predictions, raw_datasets["test"])
        metrics.update({f"eval_exact_{k}": v for k, v in new_metrics.items()})

        logger.info(f"eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        submit_wandb_evaluate(metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        result = trainer.predict(splits["validation"])
        logits = trainer.last_prediction.predictions
        predictions = predict(logits, splits["validation"], config.id2label, data_args.format)
        new_metrics = evaluate(predictions, raw_datasets["validation"])
        result.metrics.update({f"test_exact_{k}": v for k, v in new_metrics.items()})

        logger.info(f"test metrics: {result.metrics}")
        trainer.log_metrics("predict", result.metrics)
        submit_wandb_predict(predictions, raw_datasets['validation'])
        if training_args.save_strategy != "no":
            trainer.save_metrics("predict", result.metrics)


def cli_main() -> None:
    data_args, model_args, training_args = parse_args()
    if data_args.validation_file is None:
        training_args.eval_strategy = "no"
    main(data_args, model_args, training_args)



if __name__ == "__main__":
    cli_main()
