#!/bin/bash

MODEL_NAME="Mistral-7B-Instruct-v0.2"
MODEL=TODO
MAX_LEN=31500

echo "Testing $MODEL $DATASET $MAX_LEN"

device="a800"
budget=1024
python pred.py -m $MODEL \
    --max_length $MAX_LEN \
    --mode lava \
    --compress_args_path c"$budget"_w32_k7_maxpool.json \
    --out_name "$device"_origin_snapkv_"$budget"_"$MODEL_NAME" \
    --gqa_support 

python3 eval.py