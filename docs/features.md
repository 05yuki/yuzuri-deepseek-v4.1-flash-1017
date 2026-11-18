# Features

What each of Yuzuri's features does for this model, and what the launch loses without it.

`serve.sh` turns every feature on.
`YUZURI_WITHOUT=<names> ./serve.sh` starts the server without the named features, comma separated.
A feature that needs another one goes off with it.

## How the numbers were taken

- Two GPUs, both sockets, the MXFP4 checkpoint, one request at a time.
- One launch for each row, with that row's features off and everything else on.
- Decode is the median of four Japanese prose replies of 1,024 tokens.
- Prefill is a prompt of 35,000 random token ids, sent twice with different ids.
- The first row is the same launch with nothing off.
- Decode moved between 52.1 and 55.1 tok/s across launches that differ in nothing that touches decode.
  A decode difference inside that band is not a result.

## Measured

| off | decode | 35,000-token prefill, first / second | what else changed |
|---|---|---|---|
| nothing | 52.6 tok/s | 19.4 s / 19.0 s | 854 hot slots, pool 1,048,576, NLL 0.7167 |
| `hot-lend` | 27.0 tok/s | 19.2 s / 19.1 s | no lent hot slots |
| `fp8-decode` | 21.1 tok/s | 19.6 s / 19.2 s | NLL 0.7167 |
| `host-allreduce` | 50.1 tok/s | 24.9 s / 24.3 s | |
| `kv-lend-plan` | 28.2 tok/s | 43.6 s / 41.6 s | pool 974,848 |
| `kv-lend-plan`, `prefill-lend` | 28.3 tok/s | 140.5 s / 41.2 s | pool 974,848 |
| `stream-prefill` | 30.0 tok/s | 4,096 tokens: 198.0 s / 20.4 s | pool 974,848 |
| `host-kv` | 54.2 tok/s | 19.8 s / 19.5 s | pool 974,848 |
| `rope-tables` | 53.9 tok/s | 19.7 s / 19.1 s | 792 hot slots |
| `embed-on-host` | 53.5 tok/s | 19.7 s / 19.2 s | 777 hot slots |
| `gpu-moe-fp32` | 52.5 tok/s | 20.1 s / 19.7 s | NLL 0.7188 |
| `stream-mmq` | 54.4 tok/s | 22.9 s / 19.2 s | |
| `prefill-warmup` | 54.5 tok/s | 22.5 s / 18.6 s | server up 15 s sooner |
| `loading` | 55.1 tok/s | 20.2 s / 19.2 s | server up after 453 s, against 207 s |
| `cpu-gpu-overlap` | 52.1 tok/s | 19.6 s / 19.0 s | |
| `live-topk` | 54.9 tok/s | 19.7 s / 19.1 s | |

- `kv-lend-plan` off takes `swa-lend`, `hot-lend` and `host-kv` off with it.
- `stream-prefill` off takes `stream-mmq`, `expert-cache`, `prefill-lend` and the `kv-lend-plan` group off with it.
  The prefill then runs on the CPU, so that row's prompt is 4,096 tokens.

## What each feature is for

**`hot-lend`** — the free pages of the KV pool hold more experts on the GPUs, chosen by the routing.
Without it decode halves.

**`fp8-decode`** — decode-sized block-FP8 GEMMs on their own kernels.
Without it decode falls to 21.1 tok/s, with the same NLL.

**`stream-prefill`** — a prompt's experts are streamed to the GPUs from kt-kernel's memory.
Without it the prefill runs on the CPU at about 200 tok/s.

**`prefill-lend`** — the VRAM of the GPU-resident experts is lent to a streamed prefill.
Without it the first long prompt of a launch took 140.5 s.

**`kv-lend-plan`** — the KV pool's free pages are lent to the prefill, and each request's chunk is planned.
Without it the 35,000-token prefill takes twice as long.

**`host-allreduce`** — all-reduce through host memory, for two cards without peer access.
Without it the prefill is 28% slower.

**`host-kv`** — the pool is the model's full context, with the rows past the cards in host memory.
Without it the pool is 974,848 tokens.

**`rope-tables`** — the RoPE tables are mapped as the context grows.
Without it 62 hot slots are lost.

**`embed-on-host`** — the token embedding is read from pinned host memory.
Without it 77 hot slots are lost.

**`gpu-moe-fp32`** — the GPU experts keep fp32 between their GEMMs.
Without it the NLL rises from 0.7167 to 0.7188.

**`stream-mmq`** — streamed groups go through llama.cpp's int8 kernels.
Without it the first long prefill of a launch took 22.9 s.

**`prefill-warmup`** — two prefills run before the server reports ready.
Without it the first long prompt pays for them: 22.5 s.

**`loading`** — the loader skips what the CPU experts hold and reads ahead.
Without it the server is up after 453 s.

**`cpu-gpu-overlap`** — the hand-off to the CPU experts is packed by one kernel, and the shared expert runs after the CPU request is sent.
No difference was measured here.

**`live-topk`** — the sparse-attention indexer selects its top-k while the scores are made.
No difference was measured at these short contexts.

## Not measured here

- `kv-vmm` and `swa-lend` were not turned off alone.
- `expert-cache` matters on one socket; see the one-socket column in the [README](../README.md).
- `mm-on-demand`, `hicache` and `swa-checkpoint-tail` were not turned off.
- `cutile-nvfp4` does not apply to this checkpoint.
