# DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers + Yuzuri

Run [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on one machine with 512 GB of RAM and one or two 16 GB GPUs.

- The 384 routed experts a layer run on the CPU in the checkpoint's MXFP4, through [KTransformers](https://github.com/kvcache-ai/ktransformers)' kt-kernel.
- Attention, the hottest experts and the prefill run on the GPUs, through [SGLang](https://github.com/sgl-project/sglang).
- [Yuzuri](https://github.com/05yuki/yuzuri) plans the launch from the checkpoint and the machine, and lends the free pages of the KV cache to hot experts.
- The engram tables (about 200 GB) are read from the checkpoint on disk, with no extracted copy.
- The context is 1,048,576 tokens.

## What is in here

SGLang is the public commit `8c7ec052` plus five patches.
The five patches change the model's own files: `deepseek_v4.py`, its attention kernels and its KV pool.
Nothing else in SGLang is patched.
Everything else that the launch needs comes from Yuzuri's hooks at startup.

| piece | where it comes from |
|---|---|
| SGLang | main at `8c7ec052`, with `patches/sglang-v41/` (five patches) |
| kt-kernel | KTransformers `95009ea`, with `patches/kt-*.patch` |
| Yuzuri | a pinned commit of [05yuki/yuzuri](https://github.com/05yuki/yuzuri), fetched by `setup.sh` |
| engram row store | `engram-adapter/`, built by `setup.sh` |

[docs/patches.md](docs/patches.md) lists every patch and its switch.

The same model also runs from
[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers),
which carries its whole stack as patches against SGLang.
This repository is the other arrangement: the model's patches here, the rest in Yuzuri.

## Quick start

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017
MODEL=/models/dsv41-1017 ./setup.sh
MODEL=/models/dsv41-1017 ./serve.sh
```

- `setup.sh` builds everything inside this directory.
  Re-running it skips the finished steps.
- `PREBUILT=<archive> ./setup.sh` uses the release archive of deepseek-v4.1-flash-1017-ktransformers and skips the kt-kernel build.
  Without it, the build needs a system CUDA 13 toolkit.
- `serve.sh` starts one tensor-parallel rank per visible GPU.
  `CUDA_VISIBLE_DEVICES` picks the GPUs.
- `MODEL=<dir> tools/qualify.sh` starts the server, checks it end to end, measures the numbers below, stops it and writes one Markdown report.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`.
`SGLANG_HOST=0.0.0.0` serves other machines; set `SGLANG_API_KEY` with it.

## Requirements

- An NVIDIA GPU of 16 GB, SM120 (RTX 50).
  SM89 (RTX 40) is selected by `03-sm89-attention` but has not been run with this repository.
- 512 GB of RAM.
- An x86-64 CPU with AVX2.
- The checkpoint (476 GiB) on an NVMe drive.
- Python 3.12, and CUDA 13 wheels from `requirements.txt`.

## Results

Measured by `tools/qualify.sh` on 2 x EPYC 7452 (2 x 32 cores), 499 GiB of RAM and RTX 5070 Ti 16 GB cards on PCIe 4.0 x16.
The checkpoint is on a WD Blue SN5100 2TB.
One request at a time.

| | two GPUs | one GPU, both sockets |
|---|---|---|
| decode, median | 55.9 tok/s | 39.3 tok/s |
| prefill, 35,000 tokens | 22.1 s (1,587 tok/s) | 28.9 s (1,212 tok/s) |
| the same prompt again, from the prefix cache | 1.9 s | 2.0 s |
| server up after | 213 s | 233 s |
| hot expert slots at the first lend | 854 | 202 |
| NLL, served minus reference | -0.0045 (SE 0.0042) | -0.0029 (SE 0.0037) |

- The prefill prompt is 35,000 token ids drawn at random.
  Every engram row it needs is a miss, so the time depends on the drive's random read rate.
  On this drive, 15 of the 22 seconds are spent reading engram rows.
- The NLL is over 24 public-domain texts, against the scores of the checkpoint's own `inference/model.py` (`tools/ref-nll.py`).
- The one-GPU column uses the CPUs of both sockets.

Decode by kind of text, two GPUs:

| text | tok/s |
|---|---|
| Chinese prose | 47.1 |
| English prose | 43.1 |
| English technical prose | 38.9 |
| math | 43.8 |
| JSON | 42.2 |
| Python code | 39.1 |

## Not covered

- More than one request at a time has not been measured.
- GGUF packs and the one-socket, 256 GB launch are in deepseek-v4.1-flash-1017-ktransformers, not here.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for the parts under MIT.
