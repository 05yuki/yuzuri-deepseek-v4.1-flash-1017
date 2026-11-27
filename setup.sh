#!/usr/bin/env bash
# Build everything serve.sh needs, inside this directory.
#
#   [MODEL=<checkpoint dir>] [GGUF=<GGUF pack dir> [BASE=<dir with the model's config and tokenizer>]]
#   [PREBUILT=<archive or directory>] ./setup.sh
#
# 1. SGLang at SGLANG_COMMIT with the six model patches of patches/sglang-v41/ (nothing else is changed in it);
# 2. kt-kernel at KT_COMMIT with patches/kt-*.patch, built here, or the wheel of a PREBUILT archive;
# 3. a Python 3.12 venv (.venv) from requirements.txt;
# 4. the engram row store (engram-adapter/librow_store.so);
# 5. Yuzuri at YUZURI_COMMIT (third_party/yuzuri), attached to the venv by its own setup.sh --package.
# Re-running skips the finished steps. PREBUILT is the archive on this repository's release page
# (kt-kernel and the MMQ kernels, built); without it kt-kernel's CMake
# build needs a system CUDA 13 toolkit.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
SGLANG_COMMIT=a9a27e9255863d44e15a2965b9cf550ffcd19e9b
KT_COMMIT=ca198c50de70abb6827ba6f573bb6671e54436e2
YUZURI_REPO="${YUZURI_REPO:-https://github.com/05yuki/yuzuri.git}"
YUZURI_COMMIT=0da581fd4e0fd5c5079815dda61df4005627be1b
sglang_patches=()
for f in "$here"/patches/sglang-v41/[0-9][0-9]-*.patch; do sglang_patches+=("sglang-v41/$(basename "$f")"); done
kt_patches=(kt-kernel-tree.patch kt-kernel-cpu-handoff.patch kt-kernel-expert-cache.patch kt-kernel-gguf.patch
            kt-llamafile-iq2xs.patch kt-llamafile-iquants.patch kt-llamafile-iq4lut.patch
            kt-kernel-gguf-kernel-guard.patch kt-kernel-decode-tasks.patch)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p third_party build state
step() { printf '\n== %s\n' "$*"; }
# a step's marker file holds what it built (a hash of the patches and the pinned commit), so a changed
# patch or a moved pin is checked out, applied and built again
built_as() { [[ -f "$1" && "$(cat "$1")" == "$2" ]]; }
patch_hash() { (cd "$here/patches" && cat "$@") | sha256sum | cut -c1-12; }
# a patch is checked against the pinned tree first; nothing is applied when it does not fit
apply_checked() {
  local tree="$1" patch="$2"
  if ! git -C "$tree" apply --check --whitespace=nowarn "$patch"; then
    echo "$patch does not apply to $tree; the tree was left as checked out" >&2
    exit 4
  fi
  git -C "$tree" apply --whitespace=nowarn "$patch"
}

sglang_state="$(patch_hash "${sglang_patches[@]}") ${SGLANG_COMMIT:0:12}"
step "SGLang @ ${SGLANG_COMMIT:0:8} with ${#sglang_patches[@]} model patches"
if [[ ! -d third_party/sglang/.git ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git third_party/sglang
fi
if ! built_as third_party/sglang/.yuzuri-1017-patched "$sglang_state"; then
  # a commit on sgl-project/sglang's main branch
  git -C third_party/sglang fetch --quiet origin "$SGLANG_COMMIT"
  git -C third_party/sglang checkout --quiet --force "$SGLANG_COMMIT"
  git -C third_party/sglang clean -fdq
  for p in "${sglang_patches[@]}"; do apply_checked third_party/sglang "$here/patches/$p"; done
  echo "$sglang_state" > third_party/sglang/.yuzuri-1017-patched
fi

kt_variant="patched $(patch_hash "${kt_patches[@]}" kt-kernel-llama-cpp.patch kt-kernel-load-prefetch.py) ${KT_COMMIT:0:12}"
step "KTransformers @ ${KT_COMMIT:0:8}"
if [[ ! -d third_party/ktransformers/.git ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git third_party/ktransformers
fi
if ! built_as third_party/ktransformers/.yuzuri-1017-patched "$kt_variant"; then
  git -C third_party/ktransformers checkout --quiet --force "$KT_COMMIT"
  git -C third_party/ktransformers clean -fdq
  git -C third_party/ktransformers submodule update --init --force --quiet third_party/llama.cpp third_party/pybind11
  git -C third_party/ktransformers/third_party/llama.cpp clean -fdq
  for p in "${kt_patches[@]}"; do apply_checked third_party/ktransformers "$here/patches/$p"; done
  apply_checked third_party/ktransformers/third_party/llama.cpp "$here/patches/kt-kernel-llama-cpp.patch"
  # kt-kernel's CMake applies its own two llama.cpp patches and stops on a tree that has more than them;
  # kt-kernel-llama-cpp.patch holds both and the GGUF types of this recipe, so the two files go
  rm third_party/ktransformers/kt-kernel/third_party_patches/llama.cpp/0001-ggml-mxfp4-type.patch \
     third_party/ktransformers/kt-kernel/third_party_patches/llama.cpp/0002-gguf-numpy2-byteorder.patch
  echo "$kt_variant" > third_party/ktransformers/.yuzuri-1017-patched
fi

step "Python environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 --seed .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    echo "neither uv nor python3.12 is on PATH: install uv (https://docs.astral.sh/uv/, it fetches Python 3.12) or Python 3.12" >&2
    exit 2
  fi
fi
py="$here/.venv/bin/python"
pip_install() {
  if command -v uv >/dev/null; then
    uv pip install --python "$py" "$@"
  else
    "$py" -m pip install "$@"
  fi
}
if [[ ! -f .venv/.requirements-done ]] || [[ requirements.txt -nt .venv/.requirements-done ]]; then
  # torch and its CUDA 13 wheels come from PyTorch's index, the rest from PyPI
  if command -v uv >/dev/null; then
    pip_install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  else
    pip_install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
  fi
  touch .venv/.requirements-done
fi
# The toolkit for the run-time JITs and the small builds. A system toolkit is used when it is CUDA 13
# (torch's cu130 build refuses another major) and its headers match this glibc (lib/cuda-home.sh).
# Otherwise NVIDIA's toolkit wheels (lib/cuda-wheels.txt, 1.3 GB) go under build/ and serve as CUDA_HOME:
# nothing outside this directory changes. CUDA_WHEELS=0 forbids that download.
source "$here/lib/cuda-home.sh"
why="$(cuda_home_unusable "$CUDA_HOME")"
if [[ -n "$why" ]]; then
  [[ -x "$CUDA_HOME/bin/nvcc" ]] && echo "the system toolkit is not used: $why"
  if [[ "${CUDA_WHEELS:-1}" == 0 ]]; then
    echo "no usable CUDA 13 toolkit under CUDA_HOME ($CUDA_HOME: $why) and CUDA_WHEELS=0: install one (13.2 or later) or allow the wheels" >&2
    exit 2
  fi
  pip_cuda="$(cuda_home_from_wheels "$py" "$here/build")"
  if [[ -n "$pip_cuda" ]]; then
    export CUDA_HOME="$pip_cuda"
    echo "CUDA toolkit: NVIDIA's wheels, $("$pip_cuda/bin/nvcc" --version | tail -1 | sed 's/.*cuda_//; s/,.*//'), under build/cuda-home (lib/cuda-wheels.txt)"
  else
    echo "no usable CUDA 13 toolkit under CUDA_HOME and the toolkit wheels could not be installed (lib/cuda-wheels.txt)" >&2
    exit 2
  fi
fi
# the SGLang tree, installed in place (serve.sh also puts it first on PYTHONPATH)
if ! "$py" -c "import sglang, sys; sys.exit(0 if sglang.__file__.startswith('$here/third_party/sglang') else 1)" 2>/dev/null; then
  # the Rust extensions (router, gRPC) are not used here and would need cargo
  SGLANG_BUILD_RUST_EXTS=none pip_install --no-deps -e third_party/sglang/python
fi

prebuilt=""
if [[ -n "${PREBUILT:-}" ]]; then
  step "prebuilt pieces from $PREBUILT"
  if [[ -d "$PREBUILT" ]]; then
    [[ "$(cd "$PREBUILT" && pwd)" == "$here/prebuilt" ]] || { rm -rf prebuilt; cp -a "$PREBUILT" prebuilt; }
  else
    rm -rf prebuilt build/prebuilt-unpack; mkdir -p build/prebuilt-unpack
    tar -xzf "$PREBUILT" -C build/prebuilt-unpack
    mv "$(ls -d build/prebuilt-unpack/*/ | head -1)" prebuilt; rmdir build/prebuilt-unpack
  fi
  "$py" - "$here/prebuilt" <<'PY'
import hashlib, json, os, platform, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "manifest.json")))
bad = [f for f, h in m["files"].items() if hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() != h]
if bad:
    sys.exit("prebuilt files do not match manifest.json: " + ", ".join(bad))
want = [l.split("==")[1].strip() for l in open("requirements.txt") if l.startswith("torch==")][0]
if m["torch"] != want:
    sys.exit(f"prebuilt is for torch {m['torch']}, requirements.txt pins {want}")
if platform.python_version().rsplit(".", 1)[0] != m["python"].rsplit(".", 1)[0]:
    sys.exit(f"prebuilt is for Python {m['python']}, this is {platform.python_version()}")
kt = [l.split("=")[1].strip() for l in open("setup.sh") if l.startswith("KT_COMMIT=")][0]
if m["pins"]["ktransformers"] != kt:
    sys.exit(f"prebuilt holds kt-kernel of KTransformers {m['pins']['ktransformers'][:8]}, setup.sh pins {kt[:8]}")
print(f"recipe {m['recipe']}, torch {m['torch']}, CUDA {m['cuda']}, GPU archs {' '.join(m['gpu_archs'])}, "
      f"{len(m['files'])} files verified")
PY
  prebuilt="$here/prebuilt"
fi

step "kt-kernel (CPU experts)"
site="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -n "$prebuilt" ]]; then
  kt_built="prebuilt $(sha256sum "$prebuilt"/kt_kernel-*.whl | cut -c1-12)"
  if ! built_as build/.kt-kernel-done "$kt_built"; then
    echo "the archive's wheel: $(basename "$prebuilt"/kt_kernel-*.whl) (every CPU variant, picked at import)"
    pip_install --no-deps "$prebuilt"/kt_kernel-*.whl
    "$py" patches/kt-kernel-load-prefetch.py "$site"
    echo "$kt_built" > build/.kt-kernel-done
  fi
elif ! built_as build/.kt-kernel-done "$kt_variant"; then
  if [[ -z "${KT_CUDA_ARCHS:-}" ]] && command -v nvidia-smi >/dev/null; then
    KT_CUDA_ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
  fi
  export CPUINFER_CUDA_ARCHS="${KT_CUDA_ARCHS:-80;86;89;90;120}"
  export CPUINFER_CPU_INSTRUCT="${CPUINFER_CPU_INSTRUCT:-NATIVE}"
  # the CUDA side (stream memops, the streamed prefill's helpers)
  export CPUINFER_USE_CUDA="${CPUINFER_USE_CUDA:-1}"
  echo "CUDA archs $CPUINFER_CUDA_ARCHS, CPU target $CPUINFER_CPU_INSTRUCT"
  PATH="$CUDA_HOME/bin:$here/.venv/bin:$PATH" pip_install --no-deps --no-build-isolation third_party/ktransformers/kt-kernel
  # each expert layer is read into the page cache in parallel before kt-kernel copies it
  "$py" patches/kt-kernel-load-prefetch.py "$site"
  echo "$kt_variant" > build/.kt-kernel-done
fi

step "engram row store (engram-adapter/librow_store.so)"
if [[ -n "$prebuilt" ]]; then
  cp "$prebuilt/lib/librow_store.so" engram-adapter/librow_store.so
elif [[ ! -f engram-adapter/librow_store.so || engram-adapter/row_store.cpp -nt engram-adapter/librow_store.so ]]; then
  g++ -O2 -std=c++17 -shared -fPIC -pthread engram-adapter/row_store.cpp -o engram-adapter/librow_store.so
fi

step "Yuzuri @ ${YUZURI_COMMIT:0:7} (third_party/yuzuri)"
if [[ ! -d third_party/yuzuri/.git ]]; then
  git clone --quiet "$YUZURI_REPO" third_party/yuzuri
fi
if [[ "$(git -C third_party/yuzuri rev-parse HEAD)" != "$YUZURI_COMMIT" ]]; then
  git -C third_party/yuzuri cat-file -e "$YUZURI_COMMIT^{commit}" 2>/dev/null || git -C third_party/yuzuri fetch --quiet origin
  git -C third_party/yuzuri checkout --quiet --detach "$YUZURI_COMMIT"
fi
# The streamed prefill multiplies the MXFP4 experts with llama.cpp's MMQ, a torch extension. The archive has
# it built for the common GPU archs (serve.sh hands it over); otherwise its llama.cpp sources are fetched,
# patched and checked here (lib/llama-mmq.sh) and given to Yuzuri, which builds the extension at the first launch.
mmq_arch=""
if command -v nvidia-smi >/dev/null; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
  case "$cap" in 9.0|10.0|10.3|12.0|12.1) mmq_arch="${cap/./}a" ;; *) mmq_arch="${cap/./}" ;; esac
fi
fetched=()
if [[ -n "$prebuilt" && -n "$mmq_arch" && -f "$prebuilt/mmq/kt_gguf_moe_mmq_$mmq_arch.so" ]]; then
  echo "MMQ kernels: prebuilt/mmq/kt_gguf_moe_mmq_$mmq_arch.so"
else
  mmq_tree="$here/build/mmq-src"
  if [[ ! -f "$mmq_tree/.done" ]]; then
    rm -rf "$mmq_tree"
    mkdir -p "$mmq_tree/python/sglang/srt/layers/moe/kt_gguf_moe_mmq/llama_mmq"
    git -C "$mmq_tree" init -q
    bash "$here/lib/llama-mmq.sh" "$mmq_tree"
    touch "$mmq_tree/.done"
  fi
  fetched=(--fetched-from "$mmq_tree")
fi
[[ -n "$prebuilt" ]] && export LD_LIBRARY_PATH="$prebuilt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHONPATH="$here/third_party/sglang/python" bash third_party/yuzuri/setup.sh --package "$here/.venv" "${fetched[@]}"

if [[ -n "${MODEL:-}" ]]; then
  step "engram manifest for $MODEL"
  "$py" tools/make-engram-manifest.py "$MODEL" "state/engram-$(basename "$MODEL")"
fi

if [[ -n "${GGUF:-}" ]]; then
  # a GGUF pack of the model (llama.cpp's "deepseek41" layout): a config directory with no weights, and the engram
  # manifest over the pack's own Engram rows
  gguf_dir="$(cd "$GGUF" && pwd)"
  out="state/gguf-$(basename "$gguf_dir")"
  step "SGLang config for $gguf_dir"
  if [[ ! -f "$out/config/config.json" ]]; then
    base="${BASE:-${MODEL:-}}"
    if [[ -z "$base" ]]; then
      # the base model's config, tokenizer and chat encoding only
      base="state/base-DeepSeek-V4.1-Flash-1017"
      "$here/.venv/bin/hf" download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$base" \
        --include config.json --include tokenizer.json --include tokenizer_config.json \
        --include "encoding/*" --include "inference/*"
    fi
    "$py" tools/make-gguf-config.py "$base" "$out/config"
  fi
  step "engram manifest for $gguf_dir"
  [[ -f "$out/engram/engram-manifest.json" ]] || "$py" tools/make-engram-manifest-gguf.py "$gguf_dir" "$out/engram"
fi

step "check: the attention-output FP8 kernel on this GPU"
"$py" tools/dsv4-o-fp8-qdq-test.py ${MODEL:+"$MODEL"} 2>&1 | grep -v -i warn | tail -3

printf '\nDone. Start the server with: MODEL=<checkpoint dir> ./serve.sh   (a GGUF pack: GGUF=<dir> ./serve.sh)\n'
