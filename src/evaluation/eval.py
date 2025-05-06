from collections import OrderedDict

from datasets import Dataset


def calculate_metrics(num_corrects: int, num_preds: int, num_golds: int) -> tuple[float, float, float]:
    precision = num_corrects / num_preds if num_preds > 0 else float("nan")
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return precision, recall, f1


def evaluate(predictions: dict[str, set[tuple[int, int, str]]], dataset: Dataset) -> dict[str, float]:
    pred_entities = predictions
    true_entities = OrderedDict()

    labels = set()
    for document in dataset:
        for example in document["examples"]:
            entities = set()
            for ent in example["entities"]:
                entities.add((ent["start"], ent["end"], ent["label"]))
                labels.add(ent["label"])
            true_entities[example["id"]] = entities

    assert len(pred_entities) == len(true_entities)
    result = {}
    all_num_corrects, all_num_preds, all_num_golds = 0, 0, 0
    for label in labels:
        num_corrects, num_preds, num_golds = 0, 0, 0
        for y, t in zip(pred_entities.values(), true_entities.values()):
            num_corrects += len([(s, e, lb) for s, e, lb in list(y & t) if lb == label])
            num_preds += len([(s, e, lb) for s, e, lb in list(y) if lb == label])
            num_golds += len([(s, e, lb) for s, e, lb in list(t) if lb == label])
        precision, recall, f1 = calculate_metrics(num_corrects, num_preds, num_golds)
        result.update({f"{label}_precision": precision, f"{label}_recall": recall, f"{label}_f1": f1})
        all_num_corrects += num_corrects
        all_num_golds += num_golds
        all_num_preds += num_preds

    precision, recall, f1 = calculate_metrics(all_num_corrects, all_num_preds, all_num_golds)
    result.update({"overall_precision": precision, "overall_recall": recall, "overall_f1": f1})

    print(result)
    return result
