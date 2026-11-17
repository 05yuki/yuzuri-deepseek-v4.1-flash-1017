# Patches

`setup.sh` applies every patch to a pinned public commit and stops when one does not fit.

## SGLang: the five model patches (`patches/sglang-v41/`)

They are applied to SGLang main at `8c7ec052`, in this order.
They change `python/sglang/srt/models/deepseek_v4.py`, its attention kernels and its KV pool: the model's own files.
Nothing else in SGLang is patched; the rest comes from Yuzuri's hooks at startup.

| patch | what it does | switch (`serve.sh` sets it) |
|---|---|---|
| `01-o-fp8-qdq` | Rounds the attention output to FP8 and back before a bf16 `wo_a`, as the checkpoint's `inference/model.py` does. | `SGLANG_DSV4_O_FP8_QDQ_BLOCK`, read from the checkpoint |
| `02-v41-kv-layout` | Stores the KV in V4.1's own layouts on SM120 / SM121: the window KV in FP8 blocks of 32, the compressed KV in FP4 blocks of 16. | `SGLANG_DSV4_KV_LAYOUT=v41` |
| `03-sm89-attention` | Selects portable Triton attention kernels on compute capability 8.9 (RTX 40). | none; chosen by the GPU |
| `04-engram-row-prefetch` | Starts each engram layer's row lookup at the beginning of the forward, on a side stream. | `SGLANG_DSV41_ENGRAM_PREFETCH_ROWS=1` |
| `05-wo-a-fp8` | Keeps `wo_a` in the checkpoint's FP8 (32 x 32 blocks); decode reads it with an FP8 GEMV. | `SGLANG_DSV4_WO_A_FP8_NATIVE=1` |

With `05-wo-a-fp8` on, the FP8 path rounds the attention output itself and `01-o-fp8-qdq` covers only the bf16 path (`SGLANG_DSV4_WO_A_FP8_NATIVE=0`).

The same five patches are part of
[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)
(`sglang-tier0/01`, `11`, `13` and `sglang-tier1/04`, `05` there).
`03-sm89-attention` is described in that repository's
[docs/sm89.md](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers/blob/main/docs/sm89.md).
This repository has not been run on an SM89 GPU.

## KTransformers: kt-kernel (`patches/kt-*.patch`)

They are applied to KTransformers `95009ea` and its llama.cpp submodule.
The build is the same as that repository's, so its prebuilt archive fits (`PREBUILT=<archive> ./setup.sh`).

| patch | what it does |
|---|---|
| `kt-kernel-tree` | kt-kernel's changes for the CPU experts and the streamed prefill. |
| `kt-kernel-llama-cpp` | The same for kt-kernel's llama.cpp submodule. |
| `kt-kernel-cpu-handoff` | The CPU experts run while the GPU works on the same layer. |
| `kt-kernel-expert-cache` | A cache of the experts in RAM for a NUMA node that cannot hold them all. |
| `kt-kernel-gguf`, `kt-llamafile-iq2xs`, `kt-llamafile-iquants`, `kt-llamafile-iq4lut`, `kt-kernel-gguf-kernel-guard` | kt-kernel's backend for GGUF experts, its x86 kernels and the check at load. `serve.sh` here does not serve a GGUF pack; the patches keep the build equal to the archive's. |
| `kt-kernel-load-prefetch.py` | Changes the installed kt-kernel: each expert layer is read into the page cache in parallel before it is copied. |

Each is described in that repository's [docs/patches.md](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers/blob/main/docs/patches.md).

## llama.cpp MMQ sources (`patches/llama-mmq.patch`)

The streamed prefill multiplies the MXFP4 experts with llama.cpp's MMQ kernels.
With a prebuilt archive the built extension comes from the archive.
Without one, `lib/llama-mmq.sh` fetches 33 files of llama.cpp at `23b0202`, applies `patches/llama-mmq.patch` and checks every file against `patches/llama-mmq.sha256`.

## Yuzuri

`setup.sh` clones [Yuzuri](https://github.com/05yuki/yuzuri) at the commit pinned in `YUZURI_COMMIT` into `third_party/yuzuri` and runs its `setup.sh --package .venv`.
Yuzuri attaches to SGLang at startup through hooks and changes no file of the SGLang tree.
The server log lists each hook as on or off.
