import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_fixed, replace_mistral_adaptive, replace_qwen2_adaptive, replace_qwen2_fixed
from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_lava, replace_qwen2_lava


context_length_list = [4096, 8192, 16384]

datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def main(args):
    
    print("Loading data...")
    
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []
    
    input_max_len = 0
    model_path = args.model_path.lower()
    

    model_max_len = args.max_length
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:

            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: 
                input_max_len = length

            prompt = example["input"] #TODO tokenizer.apply_chat_template ?

            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for example in test_data:
        prompt_list.append(example["prompt"])
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example["index"])

    print("Finish loading model and tokenizer")
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}", str(args.context_length), args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}", str(args.context_length), args.dataset, f"{args.out_name}.json"), "w")
    
    for i in tqdm(range(0, len(prompt_list), args.eval_batch_size)):
        
        batch_prompts = prompt_list[i:i+args.eval_batch_size]
        batch_inputs = input_list[i:i+args.eval_batch_size]
        batch_answers = outputs_list[i:i+args.eval_batch_size]
        batch_lengths = length_list[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask


        context_length = batch_input_ids.shape[-1]
        output = model.generate(
            **tokenized_prompts,
            max_new_tokens=output_max_len,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[tokenizer.eos_token_id]
        )

        batch_outputs = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        batch_generations = batch_outputs

        torch.cuda.empty_cache()
        
        for j in range(args.eval_batch_size):
            
            example = {}
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["answers"] = batch_answers[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]

            fout.write(json.dumps(example) + "\n")
    
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map=args.device_map,
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("-m", '--model_path', type=str, required=True)
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")

    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--skip",type=int, default=0, help="skip layer number")
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--mode', type=str, choices=['ada', 'my', 'fix', 'lava', 'base'], help="Ada mode, fix mode , lava mode or normal")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    parser.add_argument('--device_map',type=str,default="auto",help="device map")

    args = parser.parse_args()
    
    set_seed(args.seed)

    # NOTE: Compress config
    compress_args = {}
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress_args['floor_alpha'] = args.floor_alpha
        compress_args['gqa_support'] = args.gqa_support
        compress_args['normalize'] = args.normalize
        compress_args['pyram_mode']= args.pyram
        compress_args['skip'] = args.skip
        compress_args['pyram_beta'] = args.pyram_beta
        compress = True
        # if args.adaptive:
        if args.mode == "ada":
            print("Ada mode")
            replace_mistral_adaptive()
            replace_qwen2_adaptive()
        elif args.mode == "fix":
            print("Fix mode")
            replace_mistral_fixed()
            replace_qwen2_fixed()
        elif args.mode == "lava":
            print("Lava mode")
            replace_mistral_lava()
            replace_qwen2_lava()
        else:
            print("Base mode")
    else:
        print("Base mode")

    def config_compress(model, window_size=32, base_capacity=512, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, pyram_beta = 20, normalize=True, skip=0, gqa_support=False):
        model.model.config.window_size = window_size
        model.model.config.base_capacity = base_capacity
        model.model.config.kernel_size = kernel_size

        model.model.config.normalize = normalize
        model.model.config.pooling = pooling
        model.model.config.floor_alpha = floor_alpha

        model.model.config.pyram_mode = pyram_mode
        model.model.config.pyram_beta = pyram_beta
        model.model.config.skip = skip
        model.model.config.gqa_support = gqa_support
        return model
    

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    

    if args.compress_args_path:
        model = config_compress(model, **compress_args)
    if args.mode == "ada":
        if args.pyram:
            args.out_name = f"{args.out_name}_pyram_beta_{args.pyram_beta}"
    elif args.mode == "fix":
        if args.pyram:
            args.out_name = f"{args.out_name}_pyram_beta_{args.pyram_beta}"

    save_dir = args.save_dir
    max_capacity_prompts = args.max_capacity_prompts
    
    for context_length in context_length_list:
        for idx, dataset in enumerate(datasets):

            print(f"Working on context length {context_length}, max_capacity_prompts: {args.max_capacity_prompts}, dataset: {dataset} - {idx}/{len(datasets)}")
            args.context_length = context_length
            args.dataset = dataset
            args.data_file = f"../data/RULER/{context_length}/{args.dataset}.jsonl"

            main(args)
