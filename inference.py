import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=400)
    parser.add_argument('--normalize',action='store_true')
    return parser.parse_args(args)

args = parse_args()

from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_fixed, replace_mistral_adaptive, replace_qwen2_adaptive, replace_qwen2_fixed
from adaptive_snapkv.monkeypatch.monkeypatch import replace_mistral_lava, replace_qwen2_lava

replace_qwen2_lava()
replace_mistral_lava()

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(path,
                                            #  torch_dtype=torch.bfloat16,
                                             torch_dtype=torch.bfloat16,
                                             # TODO: hard code
                                             device_map="auto",
                                             attn_implementation="flash_attention_2",
                                             trust_remote_code=True,
                                             )
    model = model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

# config hyperparameters
compress_args = {}
compress_args['normalize'] = args.normalize
def config_compress(model, window_size=32, base_capacity=128, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, pyram_beta = 20, normalize=False, skip=0, gqa_support=False):
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

model = config_compress(model, **compress_args)
print(model.model.config.normalize)
prompt = "To further reduce the latency overhead in sparse attention, including fetching the selected value cache from the CPU and reconstructing the corresponding key cache, an accurate KV selection method is needed to minimize the sparse KV cache budget while maintaining the accuracy.Observation. We found most post-RoPE key cache exhibits spatial locality, with high cosine similarity to adjacent tokens, except for a few outliers. To quantify this, we conducted inference experiments on 128K contexts. We divided the post-RoPE keys into chunks of eight tokens and visualized the minimum cosine similarity between the chunk's mean and its key cache, as shown in Figure 3b. The results indicate that, apart from a few outliers, there is generally high cosine similarity, suggesting the mean values can serve as landmarks to approximate attention well within normal chunks.Analysis. This finding suggests that for the majority of chunks, we can maintain the mean value as compressed landmarks to select minimal important KV pairs (1.56%) accurately during decoding. Outlier chunks, which may contain dense or critical information and are difficult to approximate, are retained to ensure accuracy. Given their relatively small number (0.2–0.3%), storing them on the GPU is feasible without affecting memory capacity. Furthermore, as shown in Figure 3c, considering the temporal locality of the KV cache—meaning that the KV pairs selected by the queries of two adjacent decoding steps have a high repetition rate, a cache policy (Zhang et al., 2024a) can be leveraged to further reduce the latency overhead by 60% during decoding with optimized kernels."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
print(input_ids)
output = model.generate(input_ids, max_new_tokens=100, temperature=0.0, do_sample=False)
print(tokenizer.decode(output[0]))