# Qualification report

2026-11-18, `tools/qualify.sh` at `112a245-dirty` (SGLang `8c7ec052`, Yuzuri `303e487`).

## Machine

- GPU: NVIDIA GeForce RTX 5070 Ti, 12.0, 16303 MiB, 595.91.07, PCIe gen 4 x16 (max gen 4 x16)
- CPU: AMD EPYC 7452 32-Core Processor, 2 socket(s) x 32 cores, 2 NUMA node(s); avx2
- RAM: 499 GiB
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic
- Python 3.12.14, torch 2.14.1+cu130 (CUDA 13.0), FlashInfer 0.7.4; CUDA toolkit 13.1
- checkpoint: model.safetensors.index.json sha256 `74b0686a3d28`

## Checks

| check | result | detail |
|---|---|---|
| answer | pass | 226 characters |
| tool calls | pass | 4 of 4 passed |
| NLL | pass | served 0.7167, minus reference -0.0018, SE 0.0048 over 24 pieces (z -0.38); mean \|diff\| a token 0.1177; 645 of 11496 tokens identical |
| prefix cache | pass | first 40.1 s, again 5.4 s with 34816 of 35000 tokens from the cache |

## Measured

- server up after 217 s; hot slots at the first lend: 215
```
decode median: 24.1 tok/s
prefill, first 35000-token prompt: 42.19 s = 830 tok/s
prefill, second 35000-token prompt: 41.73 s = 839 tok/s
decode, Chinese prose: 1024 tokens in 48.8 s = 21.0 tok/s
decode, English prose: 1024 tokens in 54.0 s = 18.9 tok/s
decode, English technical prose: 1024 tokens in 62.5 s = 16.4 tok/s
decode, math: 1024 tokens in 54.0 s = 19.0 tok/s
decode, JSON: 1024 tokens in 57.1 s = 17.9 tok/s
decode, Python code: 1024 tokens in 63.7 s = 16.1 tok/s
```

## Launch

```
profile auto: DeepseekV41ForCausalLM, experts MXFP4 (packed weights I8, block scales F8_E8M0), stream group 4
deepseek-v41-flash-1017 tp 1: hot 1 per layer (plan 1), chunk 31744, threads 30 on nodes [0], context 1048576 (pool = context), KV fp8, static 1.0 of the card
[yuzuri-features] 32 hooks on, 0 the tree's own, 0 off, 0 gone, 7 idle 
```
