# What DeepSeek-V4.1-Flash-1017 changes

Compared against the earlier DeepSeek-V4.1-Flash release (Hub revision `314113f8`; 1017 is `1bae0e10`).

## Weights

- All 48 safetensors shards have new contents.
- The shapes and dtypes are the same.
- The safetensors headers of the two engram shards (47 and 48) are the same.
- `config.json` and `tokenizer.json` are byte-identical.

## Reference inference code (`inference/model.py`)

**The attention output is quantized to FP8 and back before `wo_a`.**
Both `Attention.forward` and `DSparkAttention.forward` gained

```python
act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)   # block 32, "ue8m0", in place
```

The line sits between the inverse RoPE and the `wo_a` einsum.

- Per 32 values, amax is floored at 1e-4.
- The scale is `2^ceil(log2(amax / 448))`.
- `x / scale` is clamped to ±448, rounded to e4m3, and multiplied back.

The new weights were made with this step.
A server that feeds `wo_a` an unrounded bf16 `o` runs 1017 off its training numerics.

**The indexer's `shared_attn.index_k` is published by every key-owning layer on every call.**
Before, a reading layer could see the previous owner's cache.
SGLang's port maps each layer to its source layer statically, so it never had this bug.

`kernel.py`, `convert.py`, `generate.py` and `engram.py` are unchanged.

## How this repository runs the FP8 step

- `01-o-fp8-qdq` adds an in-place Triton kernel for the step, bit-exact against the checkpoint's own `act_quant` (`tools/dsv4-o-fp8-qdq-test.py`; `setup.sh` runs it on the GPU it builds on).
- `tools/o-fp8-qdq-block.sh`, sourced by `serve.sh`, reads the block size from the checkpoint's own `inference/model.py`.
- The step therefore turns on for 1017 (block 32) and stays off for the earlier release.
- With `05-wo-a-fp8` (the default), `wo_a` stays in the checkpoint's FP8 and the FP8 path rounds the attention output itself.
- The 32-value blocks lie inside a head (512 values), so splitting heads across GPUs never cuts a block.

## Encoding (`encoding/`)

- Without a tools declaration, reasoning is now stripped only before the latest user request.
- A tool can carry a `namespace`; it is encoded as `namespace::name`.
- Neither change touches a plain chat without tools.

SGLang renders the prompt through DeepSeek's `deepseek-recipe` package.
It differs from the checkpoint's `encoding/encoding.py` in three places.

- Tool namespaces are ignored, and same-named tools in different namespaces are refused.
- The current turn's reasoning before tool results is dropped.
- An assistant message's `action` does not render as `<｜action｜>`.

## Sampling

The 1017 model card recommends `temperature` 1.0 and `top_p` 1.0.
These are SGLang's defaults for a request that sets neither, since the checkpoint has no `generation_config.json`.
