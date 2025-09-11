#!/bin/bash

MODEL_NAME="Mistral-7B-128K"
MODEL=TODO

device="a800"
budget=1024
python eval_mistral_lava.py \
    -m $MODEL \
    --compress_args_path c10000_w64_k7_maxpool.json \
    --mode lava \
    --out_name "$device"_"$budget"_10000_w64_k7_"$MODEL_NAME" \
    --gqa_support \
    --start_idx 0 \
    --stop_idx 50 \
    --task longbook_sum_eng




