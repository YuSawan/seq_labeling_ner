# seq_labeling_ner
Sequential labeling based named entity recognition toolkit

## Usage

### Instllation
```
git clone git@github.com:YuSawan/seq_labeling_ner.git
cd seq_labeling_ner
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Dataset preparation
#### Dataset
```
{
  "id": "doc-001",
  "examples": [
    {
      "id": "doc-001-P1",
      "text": "She graduated from NAIST.",
      "entities": [
        {
          "start": 19,
          "end": 24,
          "label": "ORG"
        }
      ],
      # Optional
      "word_positions": [
        [0, 3], [4, 13], [14, 18], [19, 24], [24, 25]
      ]
    }
  ]
}
```


### Finetuning
BERT enclosed in parentheses () has its parameters frozen.
#### BERT Token Classification
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config.yaml
    --output_dir ./output/
```

#### BERT-CRF
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config_bert-crf.yaml
    --output_dir ./output/
```

#### (BERT)-CRF
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config_crf.yaml
    --output_dir ./output/
```

#### (BERT)-BiLSTM
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config_bert-lstm.yaml
    --output_dir ./output/
```

#### (BERT)-BiLSTM-CRF
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config_bert-lstm-crf.yaml
    --output_dir ./output/
```
