# What it needs

RAM and VRAM play different parts.
RAM holds every routed expert, so it has to be large enough.
VRAM has to hold the dense layers and the part of the KV pool the attention scans in full.
The rest of the GPU memory is working room, and each use of it shrinks when there is less.

## RAM

- A 512 GB class machine.
- The server uses about 420 GiB.
- Most of it is the MXFP4 CPU experts on 2 MB pages.
- The prefix cache's copy in host memory adds about 5 GiB a GPU.
- Yuzuri makes one expert pool per NUMA node and uses every physical core.
- kt-kernel's MXFP4 kernels need AVX2.

### With 256 GB

- One socket of the test machine (32 cores, 251 GiB) with one GPU runs the model.
- The launch is `numactl -N 0 -m 0 env CUDA_VISIBLE_DEVICES=0 MODEL=<checkpoint dir> ./serve.sh`.
- Yuzuri then keeps as many experts in an expert cache as the node's free memory holds.
- The other experts are read from the checkpoint on the drive when the routing picks them.
- Decode is 24.1 tok/s and a 35,000-token prefill 42.2 s there, against 39.2 tok/s and 27.4 s with both sockets ([README](../README.md#results)).
- A machine with one socket and 256 GB has not been run with this repository.

## GPU

- One or two cards of 16 GB, SM120 (RTX 50).
- `CUDA_VISIBLE_DEVICES` picks the GPUs, and each one is a tensor-parallel rank.
- The KV pool always covers the whole 1,048,576-token context.
- On two cards the whole pool is on the GPUs.
- On one card the compressed KV that does not fit (0.70 GiB at 1M tokens) lives in host memory ([kv-cache.md](kv-cache.md#the-kv-pool-past-the-gpu)).
- The prefill chunk is sized to the memory that is left, so a card with less room runs smaller chunks and a slower prefill.
- In decode, extra hot experts go into the KV pool's free pages and the driver's idle memory, as many as fit.
- The cards therefore show nearly full whatever room is left.
- The KV pool is never made smaller to gain speed: the lends take only its free pages.
- If the server runs out of memory at launch, a shorter context (`SGLANG_CONTEXT_LENGTH`) makes the pool smaller.
- SM89 (Ada, RTX 40) is selected by `03-sm89-attention` but has not been run with this repository ([sm89.md](sm89.md)).
- GPUs older than SM89 are refused by `serve.sh` before the model loads.

## Disk

- About 500 GB on NVMe.
- The checkpoint is 476 GiB (510 GB).
- `.venv` is 8.6 GB and the upstream clones about 1 GB.
- The load runs at the drive's speed.
- The engram rows are read from the drive during inference, so its random read rate limits the prefill ([README](../README.md#prefill-and-the-drive)).
- Keep free space on the drive and run `fstrim` on its mount.

## Software

- Linux, bash 4.4 or newer, git, gcc/g++.
- `uv`, or Python 3.12.
- The libnuma and hwloc headers (Debian/Ubuntu: `libnuma-dev libhwloc-dev`) for the kt-kernel build.
- A CUDA 13 toolkit for the kt-kernel build.
- With a prebuilt archive (`PREBUILT=`, [setup.md](setup.md#prebuilt-pieces)) the kt-kernel build is skipped, and gcc/g++, `libnuma1` and `libgomp1` are enough.
- The kernels compiled at run time use the system toolkit when it is CUDA 13.x.
- With glibc 2.42 or later that toolkit must be 13.2 or newer: older headers declare `rsqrt` differently from glibc.
- Without a usable toolkit, `setup.sh` downloads NVIDIA's toolkit wheels (CUDA 13.2, pinned in `lib/cuda-wheels.txt`, 1.3 GB) into `build/`.
- Nothing outside the checkout changes, and `CUDA_WHEELS=0` forbids the download.
- The Python packages are pinned in `requirements.txt`.

The versions the results were measured with:

| piece | version |
|---|---|
| OS | Ubuntu 26.04.1, kernel 7.0.0 |
| NVIDIA driver | 595.91.07 |
| Python | 3.12.14 |
| torch | 2.14.1 + cu130 |
| FlashInfer | 0.7.4 |
| CUDA toolkit | 13.1 |

## Host settings

None of them is needed to start.
They are recommended, set as root and persistently.

- `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled`.
  The expert arenas want 2 MB pages.
  On 4 KB pages, pinning them can take most of an hour at launch.
- `vm.min_free_kbytes = 8388608`.
  Without it, the CUDA JIT can hit a node-local OOM while the experts fill the nodes.
- Swap off.
  With swap on, pressure on one NUMA node can push the server's memory to swap while the other node still has room.
