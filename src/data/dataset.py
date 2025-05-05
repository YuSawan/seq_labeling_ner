import itertools
from typing import Any, Iterable, Optional, TypedDict

from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    BatchEncoding,
    BertJapaneseTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TrainingArguments,
    logging,
)

from .tokenizer import BertJapaneseTokenizerFast

logger = logging.get_logger(__name__)


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Example(TypedDict):
    id: str
    text: str
    entities: list[Entity]
    word_positions: Optional[list[tuple[int, int]]]


def read_dataset(
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        cache_dir: Optional[str] = None
        ) -> DatasetDict:
    """
    DatasetReader is for processing
    Input:
        train_file: dataset path for training
        validation_file: dataset path for validation
        test_file: dataset path for test
    Output: DatasetDict
    """
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
    cache_dir = cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    return raw_datasets


def get_sequence_labels(label_set: list[str], format: str) -> list[str]:
    labels = ['O']
    for label in sorted(label_set):
        if format in ["iob1", "iob2"]:
            labels.extend([f"B-{label}", f"I-{label}"])
        elif format in ["ioe1", "ioe2"]:
            labels.extend([f"I-{label}", f"E-{label}"])
        elif format == 'iobes':
            labels.extend([f"B-{label}", f"I-{label}", f"E-{label}", f"S-{label}"])
        else:
            assert format == 'bilou'
            labels.extend([f"B-{label}", f"I-{label}", f"L-{label}", f"U-{label}"])
    return labels


def _offset_to_seqlabels(entities: list[tuple[int, int, str]], format: str, token_len: int) -> list[str]:
    labels: list[str] = []
    cur = 0
    if not entities:
        return ['O'] * token_len

    for start, end, label in entities:
        if end - start == 1:
            if format in ['iobes', 'bilou']:
                segment = ['S-'+label] if format == 'iobes' else ['U-'+label]
            elif format == 'iob2':
                segment = ['B-'+label]
            elif format == 'ioe2':
                segment = ['E-'+label]
            else:
                assert format.endswith('1')
                segment = ['I-'+label]
        else:
            segment = ['I-'+label] * (end - start)
            if format in ['iob2', 'iobes', 'bilou']:
                segment[0] = 'B-'+label
            if format in ['ioe2', 'iobes']:
                segment[-1] = 'E-'+label
            if format == 'bilou':
                segment[-1] = 'L-'+label

        if cur == start:
            if format in ['iob2', 'ioe2', 'iobes', 'bilou']:
                labels.extend(segment)
            else: # iob1, ioe2
                if labels:
                    prev_label = labels[-1]
                    if format == 'ioe1':
                        if prev_label == segment[0]:
                            labels[-1] = 'E-'+labels[-1].split('-')[1]
                    else: # iob1
                        if prev_label.split('-')[1] == segment[0].split('-')[1]:
                            segment[0] = 'B-'+segment[0].split('-')[1]
                labels.extend(segment)
        else:
            labels.extend(['O'] * (start - cur))
            labels.extend(segment)

        cur = end

    if cur != token_len:
        assert cur < token_len
        labels.extend(['O'] * (token_len - cur))

    return labels


def _remove_nested_mentions(entities: list[tuple[int, int, str]]) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    nested_mentions, surface_mentions = [], []
    for (start, end, label) in entities:
        flag = True
        for (s_start, s_end, _) in entities:
            if start == s_start and end == s_end:
                continue
            if start >= s_start and end <= s_end:
                flag = False
                break
        if flag:
            surface_mentions.append((start, end, label))
        else:
            nested_mentions.append((start, end, label))

    return surface_mentions, nested_mentions


class Preprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        labels: list[str],
        format: str = "iob2",
        max_sequence_length: Optional[int] = None,
        pretokenize: bool = False
    ):
        if not isinstance(tokenizer, (PreTrainedTokenizerFast, BertJapaneseTokenizer)):
            raise RuntimeError(
                "Only ``PreTrainedTokenizerFast`, `BertTokenizerFast` and `BertJapaneseTokenizer`, `LlamaTokenizerFast` are currently supported,"
                f" but got `{type(tokenizer).__name__}`."
            )
        if format not in [ "iob1", "iob2", "ioe1", "ioe2", "iobes", "bilou"]:
            raise ValueError(f"Invalid format: {format}")
        if tokenizer.is_fast:
            self._fast_tokenizer = tokenizer
        else:
            if isinstance(tokenizer, BertJapaneseTokenizer):
                self._fast_tokenizer = BertJapaneseTokenizerFast.from_pretrained(tokenizer.name_or_path, model_max_length=tokenizer.model_max_length)
            else:
                raise RuntimeError(
                    "Only `PreTrainedTokenizerFast` is supported," f" but got `{type(tokenizer).__name__}`(PreTrainedTokenizerBase)."
                )
        type_set = set()
        for label in labels:
            if label != 'O':
                type_set.add(label.split('-')[1])
        self.types = sorted(type_set)
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.format = format

        if max_sequence_length is None:
            max_sequence_length = tokenizer.model_max_length
        num_specials = len(tokenizer.build_inputs_with_special_tokens([]))
        if type(self._fast_tokenizer) is PreTrainedTokenizerFast:
            num_specials = 2 # [cls, sep] tokens
        assert num_specials == 2 # [bos, eos] tokens, or [cls, sep] tokens
        self.max_sequence_length = max_sequence_length
        self.max_num_tokens = max_sequence_length - num_specials
        self.pretokenize = pretokenize

    def __call__(self, document: list[Example]) -> Iterable[BatchEncoding]:
        segments = None
        if self.pretokenize:
            segments = []
            for example in document:
                if example["word_positions"] is None:
                    raise ValueError("`word_positions` is required for pretokenization.")
                segments.append(example['word_positions'])

        for example, tokenization in zip(document, self.tokenize([e["text"] for e in document], segments)):
            example_id = example["id"]
            yield self._get_encoding(tokenization['token_ids'], self._get_entities(example, tokenization), tokenization['prediction_mask'], example_id)


    def _get_encoding(self, token_ids: list[int], entities: list[tuple[int, int, str]], prediction_mask: list[bool], example_id: str) -> BatchEncoding:
        encoding = self._fast_tokenizer.prepare_for_model(
            token_ids,
            add_special_tokens=True,
        )
        # PreTrainedTokenizerFastだけprepare_for_modelのadd_special_tokensに非対応
        if type(self._fast_tokenizer) is PreTrainedTokenizerFast:
            encoding['input_ids'] = [self._fast_tokenizer.cls_token_id] + encoding['input_ids'] + [self._fast_tokenizer.sep_token_id]
            encoding['attention_mask'] = [1] + encoding['attention_mask'] + [1]

        encoding["id"] = example_id
        labels = [self.label2id[label] for label in _offset_to_seqlabels(entities, self.format, len(token_ids))]
        labels = [-100] + labels + [-100]
        encoding["labels"] = labels
        encoding["prediction_mask"] = [False] + prediction_mask + [False]

        return encoding


    def _get_entities(self, example: Example, tokenization: dict[str, Any]) -> list[tuple[int, int, str]]:
        entity_map = {(ent["start"], ent["end"]): ent["label"] for ent in example["entities"]}
        entities = []
        for token_spans, char_spans in self._batch_spans(example, tokenization):
            for (char_start, char_end), (token_start, token_end) in zip(char_spans, token_spans):
                entity_type = entity_map.pop((char_start, char_end), None)
                if entity_type:
                    entities.append((token_start, token_end, entity_type))
        entities, nested_entities = _remove_nested_mentions(entities)
        if nested_entities:
            offset_map = {(t_s, t_e): (c_s, c_e) for (t_s, t_e), (c_s, c_e) in zip(token_spans, char_spans)}
            assert offset_map
            for start, end, label in nested_entities:
                nested_char_start, nested_char_end = offset_map.pop((start, end))
                entity_map.update({(nested_char_start, nested_char_end): label})
        if entity_map:
            logger.warning(f"Some entities are discarded during preprocessing: {entity_map}")
        return entities


    def _batch_spans(self, example: Example, tokenization: dict[str, Any]) -> Iterable[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
        def _enumerate_spans(left: int, right: int, max_length: Optional[int] = None) -> Iterable[tuple[int, int]]:
            if max_length is None:
                max_length = right - left
            for i in range(left, right):
                for j in range(i + 1, right + 1):
                    if j - i > max_length:
                        continue
                    yield i, j

        token_spans = []
        char_spans = []

        text = example["text"]
        offsets = tokenization["offsets"]
        boundaries = set(itertools.chain.from_iterable(example["word_positions"] or []))
        seq_start, seq_end = tokenization["context_boundary"]
        for start, end in _enumerate_spans(seq_start, seq_end):
            char_start, char_end = offsets[start - seq_start][0], offsets[end - seq_start - 1][1]
            if text[char_start] == " ":
                char_start += 1
            if char_start == offsets[start - seq_start][1]:
                continue
            assert char_start < char_end
            if boundaries and not (char_start in boundaries and char_end in boundaries):
                continue
            token_spans.append((start, end))
            char_spans.append((char_start, char_end))
        assert len(char_spans) == len(set(char_spans))
        yield token_spans, char_spans

    def tokenize(
        self, document: list[str], segments: Optional[list[list[tuple[int, int]]]] = None
    ) -> Iterable[dict[str, Any]]:
        if segments is not None:
            assert len(document) == len(segments)
            batch_text = [
                text[start:end]
                for text, positions in zip(document, segments)
                for start, end in positions
            ]
        else:
            batch_text = document

        encoding = self._fast_tokenizer(
            batch_text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )

        _post_process_encoding(encoding, batch_text, self._fast_tokenizer)
        encoding = _merge_encoding(encoding, segments, self._fast_tokenizer)

        offset = 0
        for i, tokens in enumerate(encoding["input_ids"]):
            n = len(tokens)
            if n > self.max_num_tokens:
                logger.info(f"truncate sequence: {encoding['input_ids'][i]}")
                tokens = tokens[: self.max_num_tokens]
                boundary = (offset, offset + self.max_num_tokens)
            else:
                boundary = (offset, offset+n)

            yield {
                "token_ids": tokens,
                "context_boundary": boundary,
                "offsets": encoding["offset_mapping"][i],
                "prediction_mask": encoding["prediction_mask"][i]
            }


def _post_process_encoding(encoding: BatchEncoding, batch_text: list[str], tokenizer: PreTrainedTokenizerBase) -> None:
    prefix_token_id = tokenizer.convert_tokens_to_ids("▁")
    if prefix_token_id == tokenizer.unk_token_id:
        return
    if not getattr(tokenizer, "add_prefix_space", True):
        return

    for i, text in enumerate(batch_text):
        if len(encoding["input_ids"][i]) == 0:
            # some tokenizers return empty ids for illegal characters.
            continue
        if encoding["input_ids"][i][0] != prefix_token_id or text[0].isspace():
            continue
        # set to (0, 0) because there is no corresponding character.
        encoding["offset_mapping"][i][0] = (0, 0)


def _merge_encoding(
        encoding: BatchEncoding,
        segments: list[list[tuple[int, int]]] | None,
        tokenizer: PreTrainedTokenizerBase,
    ) -> BatchEncoding:
    if not segments:
        encoding['prediction_mask'] = [[True] * len(input_ids) for input_ids in encoding['input_ids']]
        return encoding

    new_encoding: dict[str, Any] = {"input_ids": [], "offset_mapping": [], "prediction_mask": []}

    add_prefix_space = getattr(tokenizer, "add_prefix_space", True)
    index = 0
    for positions in segments:
        merged_input_ids = []
        merged_offsets: list[tuple[int, int]] = []
        merged_prediction_mask = []

        batch_input_ids = encoding["input_ids"][index : index + len(positions)]
        batch_offsets = encoding["offset_mapping"][index : index + len(positions)]

        for i, (input_ids, offsets, (char_start, _)) in enumerate(
            zip(batch_input_ids, batch_offsets, positions)
        ):
            if not input_ids:  # skip illegal input_ids
                continue
            if add_prefix_space and i > 0 and offsets[0] == (0, 0):
                input_ids = input_ids[1:]
                offsets = offsets[1:]
            merged_input_ids.extend(input_ids)
            merged_offsets.extend((char_start + ofs[0], char_start + ofs[1]) for ofs in offsets)
            merged_prediction_mask.extend([True] + [False] * (len(input_ids)-1))

        assert all(
            merged_offsets[i][0] >= merged_offsets[i - 1][1] for i in range(1, len(merged_offsets))
        )

        new_encoding["input_ids"].append(merged_input_ids)
        new_encoding["offset_mapping"].append(merged_offsets)
        new_encoding["prediction_mask"].append(merged_prediction_mask)
        index += len(positions)
    assert index == len(encoding["input_ids"])

    return new_encoding


def get_splits(
        raw_datasets: DatasetDict,
        preprocessor: Preprocessor,
        training_arguments: Optional[TrainingArguments]=None
        ) -> dict[str, Dataset]:
    def preprocess(documents: Dataset) -> Any:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = next(iter(raw_datasets.values())).column_names
            splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)
    else:
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    return splits
