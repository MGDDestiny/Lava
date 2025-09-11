# LAVa

LAVa is a kv cache compression method aiming to imporve kv cache eviction performance based on theoretical analysis. It supports dynamic head budget allocation and dynamic layer budget allocation. 


## Usage

### Requirements

```
transformers==4.41.1
flash-attn==2.4.0

datasets
tiktoken
jieba
rouge_score
```

### Installation

```
https://github.com/MGDDestiny/Lava/
cd Lava
make i
```

### Quick Start

```python
python inference.py -m /path/of/mistral_or_qwen/model
```


## Evaluations

### LongBench 

```bash
cd ./experiments/LongBench
bash eval_longbench_lava.sh
```

### Needle_In_A_Haystack

```bash
cd ./experiments/needle_in_haystack
bash eval_needle_lava.sh
```

### Ruler

```bash
cd ./experiments/ruler
bash eval_ruler_lava.sh
```

### Ruler

```bash
cd ./experiments/InfiniteBench
bash src/eval_infinite_lava.sh
```

## Citation

If you found our work valuable, please cite:

```
@inproceedings{
shen2025lava,
title={{LAV}a: Layer-wise {KV} Cache Eviction with Dynamic Budget Allocation},
author={Yiqun Shen and Song Yuan and Zhengze Zhang and Xiaoliang Wang and Daxin Jiang and Nguyen Cam-Tu},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=2wLIS98ZDp}
}
```

## Acknowledgement

We extend our gratitude to [Adakv](https://github.com/FFY0/AdaKV),  [SnapKV](https://github.com/FasterDecoding/SnapKV)  and [PyramidKV](https://github.com/Zefan-Cai/PyramidKV) for their contributions of open-source code, which have significantly facilitated the advancement of this project.

