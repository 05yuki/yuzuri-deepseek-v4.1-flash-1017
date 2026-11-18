# GGUF packs

`serve.sh` also serves a GGUF pack of the model.
The pack measured here is [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), the IQ2_XXS build with its Engram file.
The pack is served as it is: nothing is converted or re-quantized.

```
GGUF=<dir with the GGUF shards and the Engram file> ./setup.sh
GGUF=<the same dir> ./serve.sh
```

- `setup.sh` writes `state/gguf-<dir name>/config/`: the base model's `config.json`, tokenizer and chat encoding, with no weights.
  `BASE=<dir>` takes them from a local copy of deepseek-ai/DeepSeek-V4.1-Flash-1017 instead of downloading them.
- `setup.sh` also writes `state/gguf-<dir name>/engram/engram-manifest.json`: where each engram table lies in the Engram GGUF.
- `serve.sh` passes the pack to Yuzuri with `--gguf`; the plan is made from the GGUF headers and the machine.

## What the pack holds

| part | format | size |
|---|---|---|
| routed experts | IQ1_M, IQ2_XXS and IQ2_XS, by layer | 123 GiB |
| shared experts | Q5_K | on the GPUs |
| engram tables | IQ4_NL, one file | 103 GiB, read from the drive |

## How it is run

- The routed experts run on the CPU in their GGUF blocks, through kt-kernel's LLAMAFILE backend.
- Hot experts and the streamed prefill run on the GPUs in the same GGUF blocks, through llama.cpp's int8 kernels (MMQ).
- The context is 1,048,576 tokens on one GPU and on two.
- On two GPUs each shared expert is whole on both ranks.
  Its down projection has 2,304 inputs, and half of that is not a whole number of 256-value GGUF blocks.
- `06-gguf-keeps-quant-config` is the one SGLang patch the pack needs ([patches.md](patches.md)).

### RAM

Yuzuri chooses between two ways of holding the experts, from the RAM the server may use.

| | when | experts in RAM | the prefill reads them from |
|---|---|---|---|
| banks | twice the experts fit | twice: kt-kernel's copy and each rank's pinned bank | the banks |
| expert cache | otherwise | once: kt-kernel's cache | the cache, and the drive for what it does not hold |

- With banks, the server here holds 131 GiB of anonymous memory and 100 GiB of pinned shared memory for 123 GiB of experts.
- The banks and kt-kernel's copy are placed by first touch, so Yuzuri launches a GGUF pack with `SGLANG_NUMA_BIND_V2=0`.
  Under SGLang's hard bind they all land on the first rank's node, and the kernel kills that rank on a 256 GB node.
- The expert cache is the RAM left after the server's own share, and never more than the experts, their decode copies and 4 GiB.
- When the cache holds every expert and a decode copy of each IQ1_M tensor, the copy is kept (`KT_GGUF_CACHE_IQ1M_T=1`).
  The copy takes 76 bytes for a block of 256 values where IQ1_M takes 56, and its GEMM gives the same result bit for bit.
- A cgroup memory limit counts as the machine's RAM.

### The first launch

The first launch with a GGUF pack compiles CUDA kernels during its warmup.

- It took 857 s to come up here; later launches take 166 to 265 s.
- The compiler writes its temporary files to `/tmp`.
  When `/tmp` is a tmpfs, they are in RAM: about 6 GiB while it runs.
- A launch that is stopped during the build leaves those files (`/tmp/tmpxft_*`) behind.

## Results

The machine and software are those of the [README](../README.md#results).
The pack is on the same NVMe drive.
One request at a time, `tools/bench.py`.

| | two GPUs, two sockets | one GPU, one socket (251 GiB) |
|---|---|---|
| experts held as | banks | expert cache, 195 GiB, with the IQ1_M decode copy |
| decode, Japanese prose, median of four prompts | 58.7 tok/s | 34.9 tok/s |
| prefill, 35,000 tokens | 19.9 s (1,760 tok/s) | 28.5 s (1,227 tok/s) |
| prefill, a second 35,000-token prompt | 17.2 s (2,035 tok/s) | 29.0 s (1,206 tok/s) |
| server up after | 265 s | 196 s |
| hot expert slots at the first lend | 886 | 860 |
| NLL, served | 1.0465 | 1.0484 |

- The NLL is over the same 24 public-domain texts as the MXFP4 checkpoint's, whose reference is 0.7185.
  The difference is the pack's quantization.
- The one-socket column is the server under `numactl -N 0 -m 0` with one GPU visible.
- The one-socket decode is the second pass over the four prompts.
  The cache fills from the drive as experts are routed: the first pass after the launch measured 8.3, 18.5, 24.8 and 35.0 tok/s.
- Two of the one-socket 35,000-token prefills took 37.4 s and 45.7 s in the same launch.
- On two GPUs the first of the four decode prompts measured 39.1 tok/s and the other three 58.3 to 62.4.
- A node of 256 GB on this two-socket machine is not a single-socket machine with more RAM: compare only the GPU side.

## Not covered

- A memory limit below the socket (64 GB, 128 GB) has not been run with this repository.
- The IQ1_M and IQ2_XS builds of the pack have not been run with this repository.
- An 8 GB or SM75 GPU has not been run with this repository.
