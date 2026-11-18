# The KV cache

## The KV cache in V4.1's formats

V4.1 defines its own KV cache formats.
The checkpoint's inference code and DeepSeek's DeepInfer engine (`V41_FORMAT`) agree on them.

- The window KV keeps all 512 values in FP8, RoPE part included, in blocks of 32 with power-of-two scales.
- The compressed KV of the ratio-1 and ratio-2 layers is FP4, in blocks of 16 with FP8 scales.

SGLang has both formats but reads them only on SM100.
Every other GPU falls back to V4's layout, which keeps the RoPE part in bf16 and quantizes the rest in blocks of 64.
`02-v41-kv-layout` adds SM120 and SM121.
The attention reads the formats through FlashInfer's SM120 sparse MLA on its full-BF16 route.
That route computes as the reference does: bf16 queries and probabilities, exact dequantization on chip, and fp32 softmax.
`serve.sh` selects the formats on SM100 and SM12x (`SGLANG_DSV4_KV_LAYOUT=v41`; `v4` gives the old layout).

- **Writers:** SGLang's writers for both formats match the reference's quantizers bit for bit.
- **Accuracy:** layer 0's attention output moves from 3.8% off the reference to 1.0%.
- **Memory:** a compressed-KV token takes 288 bytes instead of 584.
  The memory this frees holds more hot experts.
- **Decode attention:** it reads the cache in place, without V4's page split, and runs in 10-11 µs a call against 16-18 µs.

## The KV pool past the GPU

When the GPU cannot hold the KV pool for the whole context, the pool is made at the context length anyway.
The compressed KV it cannot hold lives in host memory.
This is Yuzuri's host KV.

- The window cache stays on the GPU.
- The indexer keys stay on the GPU: decode scores them in full at every step.
- The compressed KV is 720 bytes a token; at 1M tokens, 0.70 GiB of RAM.
- Each pool layer is one range on remappable memory.
- Its first pages are on the GPU, as many as the budget holds, and the rest is host memory of the GPU's NUMA node.
- Requests fill the GPU's pages first.
- When the whole context fits, as on two 16 GB cards, nothing changes.
- Each decode step reads only the 512 entries the indexer picks for each of 8 indexer layers.
- They are gathered from host memory into a small GPU buffer.
- The output is bit-identical to the same pool on the GPU.
