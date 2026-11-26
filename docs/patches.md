# Patches

`setup.sh` applies every patch to a pinned public commit and stops when one does not fit.

## SGLang: the six model patches (`patches/sglang-v41/`)

They are applied to SGLang main at `a9a27e92`, in this order.
They change `python/sglang/srt/models/deepseek_v4.py`, its attention kernels and its KV pool: the model's own files.
Nothing else in SGLang is patched; the rest comes from Yuzuri's hooks at startup.

| patch | what it does | switch (`serve.sh` sets it) |
|---|---|---|
| `01-o-fp8-qdq` | Rounds the attention output to FP8 and back before a bf16 `wo_a`, as the checkpoint's `inference/model.py` does. | `SGLANG_DSV4_O_FP8_QDQ_BLOCK`, read from the checkpoint |
| `02-v41-kv-layout` | Stores the KV in V4.1's own layouts on SM120 / SM121: the window KV in FP8 blocks of 32, the compressed KV in FP4 blocks of 16. | `SGLANG_DSV4_KV_LAYOUT=v41` |
| `03-sm89-attention` | Selects portable Triton attention kernels on compute capability 8.9 (RTX 40). | none; chosen by the GPU |
| `04-engram-row-prefetch` | Starts each engram layer's row lookup at the beginning of the forward, on a side stream. | `SGLANG_DSV41_ENGRAM_PREFETCH_ROWS=1` |
| `05-wo-a-fp8` | Keeps `wo_a` in the checkpoint's FP8 (32 x 32 blocks); decode reads it with an FP8 GEMV. | `SGLANG_DSV4_WO_A_FP8_NATIVE=1` |
| `06-gguf-keeps-quant-config` | Lets `wo_a` and the token embedding keep a GGUF pack's quantization config, so their GGUF tensors load. | none; a GGUF pack |

With `05-wo-a-fp8` on, the FP8 path rounds the attention output itself and `01-o-fp8-qdq` covers only the bf16 path (`SGLANG_DSV4_WO_A_FP8_NATIVE=0`).

What each patch answers to in the model is in [model.md](model.md) and [kv-cache.md](kv-cache.md).
`03-sm89-attention` is described in [sm89.md](sm89.md).
This repository has not been run on an SM89 GPU.

## KTransformers: kt-kernel (`patches/kt-*.patch`)

They are applied to KTransformers `ca198c5` and its llama.cpp submodule.
A prebuilt archive holds the same build ([setup.md](setup.md#prebuilt-pieces)).

| patch | what it does |
|---|---|
| `kt-kernel-tree` | kt-kernel's changes for the CPU experts and the streamed prefill. |
| `kt-kernel-llama-cpp` | The same for kt-kernel's llama.cpp submodule. |
| `kt-kernel-cpu-handoff` | The CPU experts run while the GPU works on the same layer. |
| `kt-kernel-expert-cache` | A cache of the experts in RAM for a NUMA node that cannot hold them all. |
| `kt-kernel-gguf`, `kt-llamafile-iq2xs`, `kt-llamafile-iquants`, `kt-llamafile-iq4lut`, `kt-kernel-gguf-kernel-guard` | kt-kernel's backend for GGUF experts, its x86 kernels and the check at load. `serve.sh` here does not serve a GGUF pack; the patches keep the build equal to the archive's. |
| `kt-kernel-decode-tasks` | The CPU experts' decode: the end of a gate+up job is handed out in smaller tasks so the threads finish together, a thread keeps its copy of the input row for the call, and the decode loop reads two-bit scale codes where a matrix holds at most four scales. Bit-identical output. |
| `kt-kernel-load-prefetch.py` | Changes the installed kt-kernel: each expert layer is read into the page cache in parallel before it is copied. |

`kt-kernel-tree` in parts:

- **MXFP4 experts on AVX2** (`operators/avx2/mxfp4-moe.hpp`, `moe_base.hpp`): the CPU expert kernels for this model.
  Activations are reordered once per expert to match the weight decode, and the SwiGLU clamp is applied.
- **Scale codes** (`KT_MXFP4_SCALE_CODES`, on): one byte per 32-value scale group instead of an fp32 scale.
  The results are bit-identical.
- **Decode reads** (`KT_MXFP4_DEC_VCONV`, `KT_MXFP4_DEC_PFS=1024`, on): each task converts the activation row to the kernel's order with PSHUFB, and the weights are prefetched 1 KB ahead.
  The results are bit-identical.
- **A few tokens on one expert** (`KT_MXFP4_SMALL_M`, on): with 2 to 8 rows on an expert, each weight group is decoded once for up to four rows instead of once a row.
  Every row's output is byte-identical to computing that row alone.
- **Shared expert memory** (`KT_EXPERT_SHM`, on): the experts live in shared memory on 2 MB pages, which the GPU ranks map to stream from.
- **Placement on the CPU side**: each NUMA part loads the experts the placement assigns to it.
- **Loader** (`python/utils/loader.py`, `kt-kernel-load-prefetch.py`): safetensors experts are read in parallel, a layer ahead, through the page cache.
- **Task queue and worker pool**: the GPU stream hands work to the CPU through stream memory operations (`KT_STREAM_MEMOPS`, on), and waiting threads spin briefly before sleeping.
- **Diagnostics**: `KT_DECODE_PHASES=<calls>` and `KT_TASKQUEUE_TIMING=<syncs>` print where the CPU's time goes.
- Code for other expert formats also comes with the tree (GGUF, EXL3, BF16).
  `serve.sh` here does not use it, but the build needs it.

`kt-kernel-expert-cache` adds `operators/avx2/mxfp4-expert-cache.hpp`: a RAM cache of the routed experts in fixed slots of one memfd, filled from the checkpoint's safetensors files with `O_DIRECT` reads.
`serve.sh` here does not turn it on.

## llama.cpp MMQ sources (`patches/llama-mmq.patch`)

The streamed prefill multiplies the MXFP4 experts with llama.cpp's MMQ kernels.
With a prebuilt archive the built extension comes from the archive.
Without one, `lib/llama-mmq.sh` fetches 33 files of llama.cpp at `23b0202`, applies `patches/llama-mmq.patch` and checks every file against `patches/llama-mmq.sha256`.

## Yuzuri

`setup.sh` clones [Yuzuri](https://github.com/05yuki/yuzuri) at the commit pinned in `YUZURI_COMMIT` into `third_party/yuzuri` and runs its `setup.sh --package .venv`.
Yuzuri attaches to SGLang at startup through hooks and changes no file of the SGLang tree.
The server log lists each hook as on or off.
