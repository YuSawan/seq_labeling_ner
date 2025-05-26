#!/bin/bash
#SBATCH --job-name=seq_labeling_ner
#SBATCH -p gpu_long
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a6000:1

config_file=configs/config.yaml
output_dir=save_models/bert-base-uncased/

mkdir -p $output_dir

for seed in 0 ; do
uv run python src/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file ${config_file} \
    --output_dir ${output_dir} \
    --seed $seed
done
