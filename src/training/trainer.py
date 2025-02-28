
from typing import Any

from transformers import EvalPrediction, Trainer


class TokenClassificationTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)

    def _compute_metrics(self, p: EvalPrediction) -> dict[str, float]:
        self.last_prediction = p
        return _compute_metrics(p)


def _compute_metrics(p: EvalPrediction) -> dict[str, float]:
    # NOTE: This is not an accurate calculation of recall because some gold entities may be discarded during preprocessing.
    preds = p.predictions.argmax(axis=-1).ravel()
    labels = p.label_ids.ravel()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    pred_entity_mask = preds != 0
    gold_entity_mask = labels != 0
    num_corrects = (preds[gold_entity_mask] == labels[gold_entity_mask]).sum().item()
    num_preds = pred_entity_mask.sum().item()
    num_golds = gold_entity_mask.sum().item()
    precision = num_corrects / num_preds if num_preds > 0 else float("nan")
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")

    return {"precision": precision, "recall": recall, "f1": f1}
