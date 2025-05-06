import importlib
from typing import Any, Optional, Union

from seqeval.metrics import accuracy_score, classification_report


def compute_score(
        predictions: list[list[str]],
        references: list[list[str]],
        suffix: bool = False,
        scheme: Optional[str] = None,
        mode: Optional[str] = None,
        sample_weight: Optional[list[int]] = None,
        zero_division: Union[str, int] = "warn",
    ) -> dict[str, Any]:
    if scheme is not None:
        try:
            scheme_module = importlib.import_module("seqeval.scheme")
            scheme = getattr(scheme_module, scheme.upper())
        except AttributeError:
            raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")

    report = classification_report(
        y_true=references,
        y_pred=predictions,
        suffix=suffix,
        output_dict=True,
        scheme=scheme,
        mode=mode,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "number": score["support"],
        }
        for type_name, score in report.items()
    }
    scores["overall_precision"] = overall_score["precision"]
    scores["overall_recall"] = overall_score["recall"]
    scores["overall_f1"] = overall_score["f1-score"]
    scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)

    return scores
