#!/bin/bash

DATASET="NeedleInHaystack"

MODEL_NAME="Mistral-7B-Instruct-v0.2"
MODEL=TODO
MAX_LEN=31500

echo "Testing $MODEL $DATASET $MAX_LEN"

methods=('lava')
budget=1024
step=400

python run_needle_in_haystack.py \
--model_path $MODEL \
--s_len 1000 \
--e_len $MAX_LEN \
--mode lava \
--compress_args_path c"$budget"_w32_k7_maxpool.json \
--floor_alpha 0.2 \
--gqa_support \
--model_version $MODEL_NAME \
--model_provider Mistral \
--save_dir ./needle_results/lava_"$budget"_"$MODEL_NAME" \
--step $step

