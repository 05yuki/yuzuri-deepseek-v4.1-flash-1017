# setup.sh and serve.sh

## What setup.sh does

```
[MODEL=<checkpoint dir>] [PREBUILT=<archive or directory>] ./setup.sh
```

1. Clones sgl-project/sglang at `8c7ec052` and applies the six patches of `patches/sglang-v41/`.
2. Clones kvcache-ai/ktransformers at `95009ea` with its llama.cpp and pybind11 submodules, and applies `patches/kt-*.patch`.
3. Creates `.venv` (Python 3.12) from `requirements.txt` and installs the patched SGLang in place.
4. Chooses the CUDA toolkit for the kernels compiled at run time: the system's when it is usable, otherwise NVIDIA's wheels under `build/`.
5. Builds kt-kernel for this CPU and the visible GPUs' compute capability, or installs the wheel of a prebuilt archive.
6. Builds the engram row store, `engram-adapter/librow_store.so`.
7. Clones Yuzuri at the commit pinned in `YUZURI_COMMIT` into `third_party/yuzuri` and attaches it to the venv (`setup.sh --package`).
8. With `MODEL` set, writes the engram manifest from the checkpoint's headers (`state/engram-<model>/engram-manifest.json`).
9. Runs the test of the attention-output FP8 kernel on this GPU.

- Each patch is checked against its pinned tree before it is applied.
- A patch that does not fit stops the setup with nothing applied.
- Re-running skips the finished steps.
- A changed patch or a moved pin is checked out, applied and built again.
- Everything it makes stays inside this directory: `third_party/`, `.venv`, `build/`, `state/`, `engram-adapter/librow_store.so`.

### Variables

| variable | meaning |
|---|---|
| `MODEL` | The checkpoint directory. Without it the engram manifest is written by `serve.sh` at the first launch. |
| `PREBUILT` | A prebuilt archive or its unpacked directory (below). |
| `CUDA_HOME` | The system toolkit, default `/usr/local/cuda`. |
| `CUDA_WHEELS=0` | Forbids the download of NVIDIA's toolkit wheels. |
| `KT_CUDA_ARCHS` | kt-kernel's GPU archs, default the visible GPUs' compute capability. |
| `CPUINFER_CPU_INSTRUCT` | kt-kernel's CPU target, default `NATIVE` (this machine's ISA). |
| `YUZURI_REPO` | Where Yuzuri is cloned from. |

Headers and libraries outside the system paths go in through `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`.

### Prebuilt pieces

`PREBUILT=<archive> ./setup.sh` installs the pieces `setup.sh` otherwise compiles.

- The archive is on this repository's [release page](https://github.com/05yuki/yuzuri-deepseek-v4.1-flash-1017/releases), with its SHA-256 beside it (36 MB).
- It holds kt-kernel with six CPU variants (AVX2, AVX-512 base / VNNI / VBMI / BF16, AMX); one is picked at import.
- It holds the streamed prefill's MMQ extension for GPU archs sm89, sm90, sm100 and sm120.
- It holds the engram row store and hwloc.
- It is built from KTransformers `95009ea` with the patches of this repository, on Ubuntu 24.04, so it runs on glibc 2.39 and later.
- It is for the pinned torch (`requirements.txt`) and Python 3.12.
- Its `manifest.json` names the pinned trees, the torch, CUDA and Python versions and every file's SHA-256; `setup.sh` checks them all.
- With it, nothing of kt-kernel is built at setup, and no system CUDA toolkit is needed.

Without an archive, kt-kernel's CMake build needs a system CUDA 13 toolkit.
The MMQ extension is then built by Yuzuri at the first launch, from 33 files of llama.cpp at `23b0202` that `lib/llama-mmq.sh` fetches, patches and checks byte for byte.

## What serve.sh does

```
MODEL=<checkpoint dir> ./serve.sh [--foreground|--background]
```

Yuzuri reads the checkpoint and the machine and plans the launch (`third_party/yuzuri/tools/serve.py --profile auto`).

- One tensor-parallel rank per visible GPU.
- One CPU expert pool per NUMA node, on every physical core.
- The hot experts, the KV pool and the prefill chunk.
- The prefix cache with its copy in host memory.
- One request at a time.
- FlashInfer autotune off, so outputs reproduce across launches.

`serve.sh` adds what belongs to this model.

- The switches of the SGLang patches ([patches.md](patches.md)).
- The engram tables, read from the checkpoint shards on disk (`engram-adapter/`).
- The attention fallback's settings on SM89 ([sm89.md](sm89.md)).

Logs go to `logs/<model>/<timestamp>/`, and `logs/<model>/current` links to the newest launch.
`serve.sh --background` returns once the server process is started; `server.pid` in the log directory holds its pid.

The first launch of a setup compiles SGLang's and FlashInfer's kernels and measures the prefill plan.
Later launches reuse the cache (`.cache/` in the checkout).
On the test machine the first launch was up after 519 s on two GPUs, and later launches after 213 s.

### Variables

| variable | default | meaning |
|---|---|---|
| `MODEL` | | The checkpoint directory. |
| `CUDA_VISIBLE_DEVICES` | all GPUs | The GPUs to use. |
| `SGLANG_TP` | the visible GPUs | Fewer tensor-parallel ranks than visible GPUs. |
| `SGLANG_PORT` | 8080 | The port. |
| `SGLANG_HOST` | 127.0.0.1 | The address to listen on. |
| `SGLANG_API_KEY` | | A key every request must carry. |
| `SGLANG_CONTEXT_LENGTH` | 1,048,576 | A shorter context, and with it a smaller KV pool. |
| `SGLANG_SERVED_NAME` | | The model name the API reports. |
| `SGLANG_EXTRA_ARGS` | | More arguments for the SGLang server. |
| `DSV41_HICACHE` | 1 | 0 keeps the prefix cache on the GPU only. |
| `DSV41_HICACHE_RATIO` | 2 | The host copy's size, in GPU pools. |
| `DSV41_ENGRAM_DIR` | `state/engram-<model>` | Where the engram manifest is. |
| `DSV41_CACHE_GIB` | 8 | The RAM cache of engram rows. |

The switches of the model patches are in [patches.md](patches.md).
