# DeepSeek-V4.1-Flash-1017 on SGLang + KTransformers + Yuzuri

Run [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on one machine with 512 GB of RAM and one or two 16 GB GPUs.

- The 384 routed experts a layer run on the CPU in the checkpoint's MXFP4, through [KTransformers](https://github.com/kvcache-ai/ktransformers)' kt-kernel.
- Attention, the hottest experts and the prefill run on the GPUs, through [SGLang](https://github.com/sgl-project/sglang).
- [Yuzuri](https://github.com/05yuki/yuzuri) plans the launch from the checkpoint and the machine, and lends the free pages of the KV cache to hot experts.
- The engram tables (about 200 GB) are read from the checkpoint on disk, with no extracted copy.
- The context is 1,048,576 tokens.

## What is in here

SGLang is the public commit `a9a27e92` plus six patches.
The six patches change the model's own files: `deepseek_v4.py`, its attention kernels and its KV pool.
Nothing else in SGLang is patched.
Everything else that the launch needs comes from Yuzuri's hooks at startup.

| piece | where it comes from |
|---|---|
| SGLang | main at `a9a27e92`, with `patches/sglang-v41/` (six patches) |
| kt-kernel | KTransformers `ca198c5`, with `patches/kt-*.patch` |
| Yuzuri | a pinned commit of [05yuki/yuzuri](https://github.com/05yuki/yuzuri), fetched by `setup.sh` |
| engram row store | `engram-adapter/`, built by `setup.sh` |

[docs/patches.md](docs/patches.md) lists every patch and its switch.

## Quick start

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /models/dsv41-1017
MODEL=/models/dsv41-1017 ./setup.sh
MODEL=/models/dsv41-1017 ./serve.sh
```

- `setup.sh` builds everything inside this directory.
  Re-running it skips the finished steps.
- `PREBUILT=<archive> ./setup.sh` installs kt-kernel from the archive on this repository's [release page](https://github.com/05yuki/yuzuri-deepseek-v4.1-flash-1017/releases) and skips its build.
  Without it, the build needs a system CUDA 13 toolkit.
- `serve.sh` starts one tensor-parallel rank per visible GPU.
  `CUDA_VISIBLE_DEVICES` picks the GPUs.
- `MODEL=<dir> tools/qualify.sh` starts the server, checks it end to end, measures the numbers below, stops it and writes one Markdown report.

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`.
`SGLANG_HOST=0.0.0.0` serves other machines; set `SGLANG_API_KEY` with it.
Tool calls arrive in `message.tool_calls` and thinking in `reasoning_content`.
Thinking is off unless a request sets `"chat_template_kwargs": {"thinking": true}`.

## Documents

| document | what it covers |
|---|---|
| [docs/requirements.md](docs/requirements.md) | RAM, GPU, disk, software, host settings |
| [docs/setup.md](docs/setup.md) | what `setup.sh` and `serve.sh` do, their variables, the prebuilt archive |
| [docs/serving.md](docs/serving.md) | the API, the prefix cache, tool calls and thinking |
| [docs/patches.md](docs/patches.md) | every patch and its switch |
| [docs/features.md](docs/features.md) | each Yuzuri feature, and the launch without it |
| [docs/gguf.md](docs/gguf.md) | serving a GGUF pack, with its numbers |
| [docs/model.md](docs/model.md) | what 1017 changes against the earlier release, and how it is run here |
| [docs/kv-cache.md](docs/kv-cache.md) | V4.1's KV formats, and the KV pool past the GPU |
| [docs/sm89.md](docs/sm89.md) | the attention fallback for RTX 40 |
| [docs/qualification.md](docs/qualification.md) | `tools/qualify.sh`, the NLL check, the tests |
| [docs/glossary.md](docs/glossary.md) | the words used here |

## Requirements

- An NVIDIA GPU of 16 GB, SM120 (RTX 50).
  SM89 (RTX 40) is selected by `03-sm89-attention` but has not been run with this repository.
- 512 GB of RAM.
  The server uses about 420 GiB.
  With 256 GB the experts that do not fit are read from the drive ([docs/requirements.md](docs/requirements.md#with-256-gb)).
- An x86-64 CPU with AVX2.
- The checkpoint (476 GiB) on an NVMe drive.
  Its random read rate limits the prefill; see [Prefill and the drive](#prefill-and-the-drive).
- Python 3.12, and CUDA 13 wheels from `requirements.txt` (torch 2.14.1, FlashInfer 0.7.4).

Three host settings are recommended, set as root.

- `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`
- `vm.min_free_kbytes = 8388608`
- Swap off.

[docs/requirements.md](docs/requirements.md) says what each one is for.

## Results

Measured by `tools/qualify.sh` on 2 x EPYC 7452 (2 x 32 cores), 499 GiB of RAM and RTX 5070 Ti 16 GB cards on PCIe 4.0 x16.
The checkpoint is on a WD Blue SN5100 2TB.
The software is Ubuntu 26.04, driver 615.71.09, torch 2.14.1 + cu130, FlashInfer 0.7.4 and CUDA toolkit 13.1.
One request at a time.

| | two GPUs | one GPU, both sockets | one GPU, one socket |
|---|---|---|---|
| decode, Japanese prose, median of four prompts | 58.2 tok/s | 40.6 tok/s | 22.0 tok/s |
| prefill, 35,000 tokens | 19.7 s (1,780 tok/s) | 27.2 s (1,289 tok/s) | 42.3 s (827 tok/s) |
| the same prompt again, from the prefix cache | 2.0 s | 2.1 s | 5.9 s |
| server up after | 198 s | 208 s | 193 s |
| hot expert slots at the first lend | 868 | 206 | 221 |
| NLL, served | 0.7191 | 0.7187 | 0.7181 |
| NLL, served minus reference | +0.0006 (SE 0.0040) | +0.0001 (SE 0.0032) | -0.0004 (SE 0.0032) |

- The three reports are in [reports/](reports/).
- The server-up times are of launches after the first.
- The first launch of a setup also compiles kernels and measures the prefill plan.
- The prefill prompt is 35,000 token ids drawn at random.
- Each decode reply is 1,024 tokens.
- The NLL is over 24 public-domain texts, against the scores of the checkpoint's own `inference/model.py` (`tools/ref-nll.py`).
- The reference's mean NLL is 0.7185.
- The second column uses the CPUs of both sockets.
- The third column is the server under `numactl -N 0 -m 0` with one GPU visible: 32 cores and 251 GiB of RAM.
  The experts that do not fit in that RAM are read from the drive.
- A node of this two-socket machine is not a single-socket machine with more RAM.

Decode by kind of text:

| text | two GPUs | one GPU, both sockets | one GPU, one socket |
|---|---|---|---|
| Chinese prose | 51.3 tok/s | 35.9 tok/s | 19.5 tok/s |
| English prose | 47.5 tok/s | 34.4 tok/s | 18.2 tok/s |
| English technical prose | 41.7 tok/s | 33.3 tok/s | 16.2 tok/s |
| math | 45.8 tok/s | 34.4 tok/s | 18.8 tok/s |
| JSON | 45.2 tok/s | 33.4 tok/s | 17.3 tok/s |
| Python code | 41.2 tok/s | 32.5 tok/s | 15.5 tok/s |

- The hot experts on the GPU are swapped for the ones the text being written routes to most.
- Code, JSON, math and English spread their routes over more experts than Japanese prose, so the CPU computes more of them.

### Long prompts

Prompts of random token ids, sent one after another to one server, shortest first.
No prompt shares a prefix with the one before it.

| prompt | two GPUs | one GPU, both sockets | one GPU, one socket |
|---|---|---|---|
| 65,536 tokens | 31.2 s (2,098 tok/s) | 44.0 s (1,489 tok/s) | 74.1 s (885 tok/s) |
| 131,072 tokens | 62.1 s (2,112 tok/s) | 91.9 s (1,427 tok/s) | 144.6 s (907 tok/s) |
| 524,288 tokens | 219.5 s (2,389 tok/s) | 351.8 s (1,490 tok/s) | 568.8 s (922 tok/s) |
| 1,000,000 tokens | 450.9 s (2,218 tok/s) | 713.7 s (1,401 tok/s) | 1,196.5 s (836 tok/s) |

### Prefill and the drive

The NVMe drive's random read rate limits the prefill.

- The engram rows are read from the checkpoint as 4 KiB random reads with `O_DIRECT`.
- Rows already read stay in an 8 GiB cache in RAM.
- A prompt of random token ids misses that cache on every row.
- One such prompt of 35,000 tokens takes about 1.86 million reads.
- The drive here has no DRAM, and its random read rate falls as the span of the file being read grows: 249,000 reads a second over a 7 GB file, 143,000 over one 101 GB engram shard.
- A 35,000-token prompt that needs no reads from the drive (one token repeated) prefills in 17.5 s on two GPUs and 25.7 s on one.
- The drive therefore adds about 2 s to the two-GPU prefill above, and 1 to 2 s on one GPU.
- 17.5 s and 25.7 s are what a faster drive can approach; no other drive has been measured.

The state of the drive matters as well.

- With this drive 99% full and never trimmed, the rate over the engram shard was 100,000 to 110,000 reads a second.
- The two-GPU prefill of 35,000 tokens was then 22 s.
- Freeing space and running `fstrim` on the mount gave the numbers above.

## Not covered

- More than one request at a time has not been measured.
- An SM89 GPU has not been run with this repository.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for the parts under MIT.
