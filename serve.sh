#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 with SGLang + KTransformers + Yuzuri from what setup.sh built.
#
#   MODEL=<checkpoint dir> ./serve.sh [--foreground|--background]
#   GGUF=<GGUF pack dir> ./serve.sh [--foreground|--background]
#
# Yuzuri reads the checkpoint and the machine and plans the launch (third_party/yuzuri/tools/serve.py
# --profile auto): one tensor-parallel rank per visible GPU (CUDA_VISIBLE_DEVICES picks them), the CPU expert
# pools, the hot experts, the KV pool and the prefill chunk. This script adds what belongs to this model:
# the switches of the SGLang patches (patches/sglang-v41/) and the engram tables read from the
# checkpoint shards on disk (engram-adapter/). Logs go to logs/<model>/<timestamp>/.
set -Eeuo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---foreground}"
if [[ "$mode" != "--foreground" && "$mode" != "--background" ]]; then
  echo "Usage: MODEL=<checkpoint dir> $0 [--foreground|--background]" >&2
  exit 2
fi
gguf="${GGUF:-}"
if [[ -n "$gguf" ]]; then
  # a GGUF pack: the config directory setup.sh wrote for it (no weights), the pack's own Engram rows
  [[ -d "$gguf" ]] || { echo "GGUF is not a directory: $gguf" >&2; exit 4; }
  gguf="$(cd "$gguf" && pwd)"
  state="$here/state/gguf-$(basename "$gguf")"
  model="$state/config"
  [[ -f "$model/config.json" ]] || { echo "run GGUF=$gguf ./setup.sh first ($model missing)" >&2; exit 3; }
  name="$(basename "$gguf")"
  engram_default="$state/engram"
  engram_base="$gguf"
else
  model="${MODEL:-}"
  [[ -n "$model" && -f "$model/config.json" ]] || { echo "set MODEL to the checkpoint directory (or GGUF to a GGUF pack)" >&2; exit 4; }
  model="$(cd "$model" && pwd)"
  name="$(basename "$model")"
  engram_default="$here/state/engram-$name"
  engram_base="$model"
fi
venv="$here/.venv"
python="$venv/bin/python"
source_dir="$here/third_party/sglang/python"
adapter_dir="$here/engram-adapter"
yuzuri="$here/third_party/yuzuri"
engram_dir="${DSV41_ENGRAM_DIR:-$engram_default}"
port="${SGLANG_PORT:-8080}"
log_dir="$here/logs/$name"
[[ -x "$python" ]] || { echo "run ./setup.sh first (.venv missing)" >&2; exit 3; }
[[ -f "$here/third_party/sglang/.yuzuri-1017-patched" ]] || { echo "run ./setup.sh first (patched SGLang missing)" >&2; exit 3; }
[[ -f "$adapter_dir/librow_store.so" ]] || { echo "run ./setup.sh first (engram-adapter/librow_store.so missing)" >&2; exit 3; }
[[ -f "$yuzuri/tools/serve.py" ]] || { echo "run ./setup.sh first (third_party/yuzuri missing)" >&2; exit 3; }
export PATH="$venv/bin:$PATH"
if [[ ! -f "$engram_dir/engram-manifest.json" ]]; then
  if [[ -n "$gguf" ]]; then
    "$python" "$here/tools/make-engram-manifest-gguf.py" "$gguf" "$engram_dir" >&2
  else
    "$python" "$here/tools/make-engram-manifest.py" "$model" "$engram_dir" >&2
  fi
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 3; }
# one tensor-parallel rank per visible GPU (SGLANG_TP=<n> for fewer)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra gpus <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
n_gpus="${#gpus[@]}"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${gpus[0]}" | head -1)"
# 03-sm89-attention supplies a portable Triton attention for Ada (SM89).
# Other pre-Hopper architectures remain unsupported; do not load the model first.
if [[ "$cap" != "8.9" && "${cap%%.*}" =~ ^[0-9]+$ ]] && (( ${cap%%.*} < 9 )); then
  echo "GPU ${gpus[0]} is compute capability $cap: attention needs SM89 (Ada) or SM90+" >&2
  exit 3
fi
if [[ "$cap" == "8.9" ]]; then
  # Ada uses the portable indexer and mHC paths as well as sparse attention.
  # The DeepGEMM and TileLang alternatives require newer GPU architectures.
  export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
  export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-0}"
  export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}"
fi

# --- the model patches' switches ----------------------------------------------
# DeepSeek-V4.1-Flash-1017 rounds the attention output to FP8 before wo_a; the block comes from the
# checkpoint's own inference code (01-o-fp8-qdq; the native FP8 wo_a below does the rounding itself)
source "$here/tools/o-fp8-qdq-block.sh"
# V4.1 keeps its KV in its own formats (02-v41-kv-layout): the window KV as all 512 values in FP8 blocks of
# 32, the RoPE part too, and the compressed KV in FP4 blocks of 16. SGLang reads them on SM100 and, with
# 03-sm89-attention, on SM120 / SM121; other GPUs keep the V4 layout
if [[ -z "${SGLANG_DSV4_KV_LAYOUT:-}" ]] && [[ "${cap%%.*}" == 10 || "${cap%%.*}" == 12 ]]; then
  export SGLANG_DSV4_KV_LAYOUT=v41
fi
# wo_a stays in the checkpoint's FP8 (32x32 E8M0 blocks) instead of a bf16 copy (05-wo-a-fp8; 0: bf16)
if [[ -n "$gguf" ]]; then
  # a GGUF pack: wo_a is a GGUF tensor, and wq_a and wkv load as two layers (GGUF bytes do not go through the
  # fused load)
  export SGLANG_DSV4_WO_A_FP8_NATIVE="${SGLANG_DSV4_WO_A_FP8_NATIVE:-0}"
  export SGLANG_OPT_FUSE_WQA_WKV="${SGLANG_OPT_FUSE_WQA_WKV:-0}"
else
  export SGLANG_DSV4_WO_A_FP8_NATIVE="${SGLANG_DSV4_WO_A_FP8_NATIVE:-1}"
fi
# decode's engram row lookups start with the token on a side stream, not at their layers
# (04-engram-row-prefetch; 0: at the layers)
export SGLANG_DSV41_ENGRAM_PREFETCH_ROWS="${SGLANG_DSV41_ENGRAM_PREFETCH_ROWS:-1}"
# the GPU experts in the checkpoint's MXFP4 layout through kernels that round once, at the weighted
# top-k sum (0: Marlin)
export SGLANG_MXFP4_MOE_FP32="${SGLANG_MXFP4_MOE_FP32:-1}"

# --- engram from disk -----------------------------------------------------------
export DSV41_ENGRAM_DIR="$engram_dir"
export DSV41_ENGRAM_BASE="${DSV41_ENGRAM_BASE:-$engram_base}"
export DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-8}"
export DSV41_ENGRAM_MODE="${DSV41_ENGRAM_MODE:-nvme}"
# the patched tree shadows any installed sglang; the adapter's sitecustomize.py installs the engram hook
# in every worker
export PYTHONPATH="$source_dir:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"
# the archive's MMQ kernels for the streamed prefill (setup.sh with PREBUILT=)
if [[ -z "${KT_GGUF_MOE_MMQ_PREBUILT:-}" && -d "$here/prebuilt/mmq" ]]; then
  export KT_GGUF_MOE_MMQ_PREBUILT="$here/prebuilt/mmq"
fi

# --- the server's arguments -------------------------------------------------------
args=(--profile auto --config-dir "$model" --package --venv "$venv"
      --log-dir "$log_dir" --cache-dir "$here/.cache" --port "$port" --host "${SGLANG_HOST:-127.0.0.1}")
[[ -n "$gguf" ]] && args+=(--gguf "$gguf" --served-model-name "${SGLANG_SERVED_NAME:-$name}")
args+=(--tp "${SGLANG_TP:-$n_gpus}")
[[ -n "${SGLANG_CONTEXT_LENGTH:-}" ]] && args+=(--context "$SGLANG_CONTEXT_LENGTH")
[[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]] && args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
[[ -z "$gguf" && -n "${SGLANG_SERVED_NAME:-}" ]] && args+=(--served-model-name "$SGLANG_SERVED_NAME")
# The prefix cache: a prompt that starts like an earlier one reads that part from the cache. Its pages are
# also kept in host memory (HiCache, write-back, twice the GPU pool), so a prefix the GPU has given up is
# loaded back instead of computed again. DSV41_HICACHE=0 keeps the cache on the GPU only.
if [[ "${DSV41_HICACHE:-1}" != "0" ]]; then
  args+=(--server-arg="--enable-hierarchical-cache --hicache-write-policy write_back --hicache-ratio ${DSV41_HICACHE_RATIO:-2}")
fi
# RAM this launch takes beside the experts and Yuzuri's own count: the engram row cache, and HiCache's pages
# (6 GiB a rank at ratio 2). Yuzuri reads it only when the experts do not fit in RAM (one socket of 256 GB):
# what is left is then kt-kernel's cache of the experts, the rest read from the checkpoint as they are routed
extra_gib="$DSV41_CACHE_GIB"
[[ "${DSV41_HICACHE:-1}" != "0" ]] && extra_gib=$(( extra_gib + 3 * ${DSV41_HICACHE_RATIO:-2} * ${SGLANG_TP:-$n_gpus} ))
args+=(--host-extra-gb "$extra_gib")
# SGLANG_HOST=0.0.0.0 serves other machines too; then set SGLANG_API_KEY, which every request must carry
# YUZURI_WITHOUT=<features>: a launch without some of Yuzuri's features (docs/features.md measures each this way)
[[ -n "${YUZURI_WITHOUT:-}" ]] && args+=(--without "$YUZURI_WITHOUT")
[[ -n "${SGLANG_API_KEY:-}" ]] && args+=(--server-arg="--api-key $SGLANG_API_KEY")
[[ -n "${SGLANG_EXTRA_ARGS:-}" ]] && args+=(--server-arg="$SGLANG_EXTRA_ARGS")
# Yuzuri's front.py passes its serve.py arguments (port, host, log-dir, YUZURI_PARK=1, --beside, ...) here
if [[ -n "${YUZURI_SERVE_ARGS:-}" ]]; then
  mapfile -d '' -t yuzuri_serve_args < <(
    "$python" -c 'import shlex, sys
for a in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(a.encode() + b"\0")' "$YUZURI_SERVE_ARGS"
  )
  (( ${#yuzuri_serve_args[@]} )) && args+=("${yuzuri_serve_args[@]}")
fi

mkdir -p "$log_dir"
if [[ "$mode" == "--foreground" ]]; then
  exec "$python" "$yuzuri/tools/serve.py" "${args[@]}"
fi
out="$("$python" "$yuzuri/tools/serve.py" "${args[@]}" --background)" || { printf '%s\n' "$out"; exit 1; }
printf '%s\n' "$out"
run="$(printf '%s\n' "$out" | sed -n 's/^log directory: //p' | tail -1)"
[[ -n "$run" && -f "$run/server.pid" ]] || { echo "no server was started" >&2; exit 1; }
# logs/<model>/current is the newest launch (lib/server-up.sh waits on it)
ln -sfn "$run" "$log_dir/current"
