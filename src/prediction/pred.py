
from collections import OrderedDict
from typing import Any

from datasets import Dataset
from seqeval.scheme import BILOU, IOB1, IOB2, IOBES, IOE1, IOE2, Entities

import wandb

SCHEME = {'iob1': IOB1, 'iob2': IOB2, 'ioe1': IOE1, 'ioe2': IOE2, 'iobes': IOBES, 'bilou': BILOU}


def _word_offsets(char_offsets: list[tuple[int, int]], prediction_mask: list[int]) -> list[tuple[int, int]]:
    word_offset = None
    word_offsets: list[tuple[int, int]] = []
    char_start = 0
    for (s, e), mask in zip(char_offsets, prediction_mask):
        if mask == 1:
            if word_offset:
                word_offsets.append(word_offset)
            char_start = s
        word_offset = (char_start, e)
    if word_offset:
        word_offsets.append(word_offset)
    return word_offsets


def label_to_charspan(labels: list[str], word_offsets: list[tuple[int, int]], scheme: str) -> list[tuple[int, int, str]]:
    entities = Entities([labels], scheme=SCHEME[scheme]).entities[0]
    char_spans = []
    for ent in entities:
        offsets = word_offsets[ent.start: ent.end]
        s, _ = offsets[0]
        _, e = offsets[-1]
        char_spans.append((s, e, ent.tag))
    return char_spans


def predict(logits: tuple[Any, ...], dataset: Dataset, id2label: dict[int, str], scheme: str) -> dict[str, set[tuple[int, int, str]]]:
    _, _, predictions, _ = logits
    true_predictions = [
        [id2label[p] for p in pred if p != -100]
        for pred in predictions
    ]
    assert len(true_predictions) == len(dataset)

    results = {}
    for example, prediction in zip(dataset, true_predictions):
        pid = example['id']
        prediction_mask = example['prediction_mask'][1:-1]
        offsets = example['offsets']
        assert len(offsets) == len(prediction_mask)
        assert prediction_mask[0] == 1
        word_offsets = _word_offsets(offsets, prediction_mask)
        char_spans = label_to_charspan(prediction, word_offsets, scheme=scheme)
        results[pid] = set(char_spans)

    return results


def submit_wandb_predict(predictions: dict[str, set[tuple[int, int, str]]], dataset: Dataset) -> None:
    columns = ["pid", "text", "gold", "predictions"]
    result_table = wandb.Table(columns=columns)

    pred_entities = predictions
    true_entities = OrderedDict()

    for document in dataset:
        for example in document["examples"]:
            entities = set((ent["start"], ent["end"], ent["label"]) for ent in example["entities"])
            true_entities[example["paragraph_id"]] = {"text": example["text"], "entities": entities}

    assert len(pred_entities) == len(true_entities)
    for (pid, y), (tid, t) in zip(pred_entities.items(), true_entities.items()):
        assert pid == tid
        text = t['text']
        y_span = [f"{text[s:e]}({lb})" for s, e, lb in y]
        t_span = [f"{text[s:e]}({lb})" for s, e, lb in t['entities']]
        result_table.add_data(pid, text, ', '.join(t_span), ', '.join(y_span))
    wandb.log({"predictions": result_table})
