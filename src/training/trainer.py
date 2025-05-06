
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers import EvalPrediction, Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names, nested_detach

from ..evaluation import compute_score
from ..model import BertNER


class TokenClassificationTrainer(Trainer):
    def __init__(self, *args: Any, seq_scheme: str, classifier_lr: float, **kwargs: Any) -> None:
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)
        self.seq_scheme = seq_scheme
        self.classifier_lr = classifier_lr

    def get_no_classifier_parameter_name(self, model: nn.Module) -> list[str]:
        no_classifier_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["crf", "classifier"])
        return no_classifier_parameters

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            no_classifier_parameters = self.get_no_classifier_parameter_name(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in no_classifier_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in no_classifier_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.classifier_lr
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError
                # import bitsandbytes

                # manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                # skipped = 0
                # for module in opt_model.modules():
                #     if isinstance(module, nn.Embedding):
                #         skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                #         logger.info(f"skipped {module}: {skipped / 2**20}M params")
                #         manager.register_module_override(module, "weight", {"optim_bits": 32})
                #         logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                # logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def _compute_metrics(self, p: EvalPrediction) -> dict[str, float]:
        self.last_prediction = p
        return _compute_metrics(p, self.model.config.id2label, self.seq_scheme)

    def prediction_step(
        self,
        model: BertNER,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    outputs = model.decode(**inputs)
                    loss = outputs.loss
                # loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model.decode(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        detached_logits = nested_detach(logits)
        if len(detached_logits) == 1:
            detached_logits = detached_logits[0]
        labels = inputs.get('labels')

        return (loss, detached_logits, labels)


def _compute_metrics(p: EvalPrediction, id2label: dict[int, str], scheme: str) -> dict[str, float]:
    # NOTE: This is not an accurate calculation of recall because some gold entities may be discarded during preprocessing.
    loss, _, preds, prediction_mask = p.predictions

    true_predictions = [
        [id2label[p] for p in pred if p != -100]
        for pred in preds
    ]

    labels = []
    for label, p_mask in zip(p.label_ids, prediction_mask):
        mask = p_mask != 0
        label = label[mask]
        labels.append(label)
    true_labels = [
        [id2label[lb] for lb in label]
        for label in labels
    ]

    results = compute_score(predictions=true_predictions, references=true_labels, mode='strict', scheme=scheme)

    return {
        "loss": loss,
        "precision": results["overall_precision"],
        "recall": results["overall_precision"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
