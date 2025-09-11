import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
import os
import argparse
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_fixed, replace_mistral_adaptive, replace_qwen2_adaptive, replace_qwen2_fixed
from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_lava, replace_qwen2_lava

MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1000

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=None)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-m", '--model_name_or_path', type=str, required=True)
    parser.add_argument("--skip",type=int, default=0, help="skip layer number")
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', 'base', 'lava'], help="Ada mode, fix mode, lava mode or normal")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    return parser.parse_args(args)

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = 128 * 1024,
    chunk_size: int = 2500,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)
        attention_mask: Tensor = inputs.attention_mask  # (b, n)
        position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, value=1)
        seq_len = input_ids.shape[-1]
        print("seq_len:", seq_len)
        kv_cache: Any = None
        # Split into chunks for pre-filling
        chunk_idxs = []
        n = seq_len - 1
        while n > 0:
            chunk_idxs.append(n)
            n -= chunk_size
        chunk_idxs.append(0)
        chunk_idxs = chunk_idxs[::-1]
        chunk_lo = chunk_idxs[:-1]
        chunk_hi = chunk_idxs[1:]
        print(f"Number of chunks: {len(chunk_lo)}, generating...")
        start_time = time.time()
        for chunk_i, (chunk_lo, chunk_hi) in enumerate(
            zip(chunk_lo, chunk_hi)
        ):
            if verbose:
                print(
                    f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                    round(time.time() - start_time),
                )
            chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
            if kv_cache is not None:
                mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
            else:
                mask_start_idx = chunk_lo
            chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
            chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
            outputs: BaseModelOutputWithPast = model.model.forward(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                position_ids=chunk_position_ids,
                past_key_values=kv_cache,
                return_dict=True,
                use_cache=True,
            )
            kv_cache = outputs.past_key_values
            # Discard KV states on the left beyond the window
            new_cache = ()
            n_layers = len(kv_cache)
            for layer_i in range(n_layers):
                keys = kv_cache[layer_i][0][:, :, -sliding_window:]
                values = kv_cache[layer_i][1][:, :, -sliding_window:]
                new_cache += ((keys, values),)
            kv_cache = new_cache
        kv_cache_len = kv_cache[0][0].shape[2]
        outputs = model.generate(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask[:, -kv_cache_len - 1 :],
            max_new_tokens=max_tokens,
            past_key_values=kv_cache,
            eos_token_id=tok.pad_token_id,
            use_cache=True,
        )
        responses = [
            tok.decode(t[1:], skip_special_tokens=True) for t in outputs
        ]
    return responses


def get_pred(
    model,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tokenizer, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    # output = chunk_generate(
    #     model,
    #     tok,
    #     [input_text],
    #     max_tokens=max_tokens,
    #     chunk_size=128,
    #     verbose=verbose,
    # )[0]
    # print("Chunked generation:", output)
    tokenized_input = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to('cuda')
    context_length = tokenized_input.input_ids.shape[-1]
    output = model.generate(
        **tokenized_input,
        max_new_tokens=max_tokens,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        min_length=context_length+1,
        eos_token_id=[tokenizer.eos_token_id]
    )[0]
    output = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    print(output)
    return output



def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map='auto',
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer# type: ignore


if __name__ == "__main__":

    args = parse_args()
    model_name_or_path = args.model_name_or_path
    model_name = args.model_name_or_path.split("/")[-1]

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
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

    # NOTE: load model after replace
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    if args.compress_args_path:
        model = config_compress(model, **compress_args)
    if args.mode == "ada":
        if args.pyram:
            args.out_name = f"{args.out_name}_pyram_beta_{args.pyram_beta}"
    elif args.mode == "fix":
        if args.pyram:
            args.out_name = f"{args.out_name}_pyram_beta_{args.pyram_beta}"

    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}_{args.out_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}_{args.out_name}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tokenizer, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
