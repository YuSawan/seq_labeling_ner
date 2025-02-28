# seq_labeling_ner
Sequential labeling based named entity recognition toolkit


## LSTM series (ToDo)
### BERT-BiLSTM

### BERT-BiLSTM-CRF


## BERT series
### BERT Token Classification
```
torch run --nproc_per_node 1 main_bert.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/eng.train.jsonl \
    --validation_file data/eng.testa.jsonl \
    --test_file data/eng.testb.jsonl \
    --model "google-bert/bert-base-uncased" \
    --output_dir ./output/ \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --max_entity_length 64 \
    --max_mention_length 16 \
    --save_strategy epoch \
```

### BERT-CRF
```
```
