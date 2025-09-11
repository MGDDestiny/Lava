#!/bin/bash

MODEL_NAME="Mistral-7B-Instruct-v0.2"
MODEL=TODO
MAX_LEN=31500

echo "Testing $MODEL $DATASET $MAX_LEN"

device="a800"

mode='lava'
budget=1024
python run_ruler.py -m $MODEL \
    --max_length $MAX_LEN \
    --mode $mode \
    --compress_args_path c"$budget"_w32_k7_maxpool.json \
    --floor_alpha 0.2 \
    --out_name "$device"_"$mode"_"$budget"_"$MODEL_NAME" \
    --gqa_support 

python3 eval_ruler.py --results_dir TODO
