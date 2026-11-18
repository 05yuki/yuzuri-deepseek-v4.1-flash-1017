# Glossary

The words the README and the docs use.

**CPU experts.** The routed experts that kt-kernel runs on the CPU in MXFP4.
They live in RAM on 2 MB pages, one pool per NUMA node.

**Resident expert.** The one expert a layer whose weights always stay on the GPU.

**Hot experts.** More experts a layer held on the GPU during decode, next to the resident one.
The GPU computes them, so the CPU skips them.

**Hot expert slot.** The room for one hot expert on one GPU.
The slots follow the text being written: the experts routed to most are copied in from RAM.
No routing has to be recorded beforehand.

**Hot lend.** Yuzuri holds hot experts in GPU memory nothing else is using: the KV pool's free pages and the driver's idle memory.
It gives that memory back when the KV pool or a prefill needs it.

**Streamed prefill.** A long prompt's experts run on the GPU.
The CPU experts are copied to it a group at a time while the previous group computes.

**MMQ.** llama.cpp's integer matrix-multiply kernels for quantized weights.
The streamed prefill runs the MXFP4 experts through them.

**Prefill lend.** During a streamed prefill, the GPU-resident weights and the KV pool's free pages give their memory to the prefill.
They come back before decode.

**Chunk.** The part of a long prompt prefilled at once.
Its size is measured on the first launch of a setup, from the GPU memory that is left.

**KV pool.** The memory that holds the KV cache, sized for the whole context.
The lends take only its free pages, so it never shrinks to gain speed.

**Host KV.** The compressed KV that does not fit the GPU, kept in host memory ([kv-cache.md](kv-cache.md#the-kv-pool-past-the-gpu)).

**Engram.** V4.1's n-gram memory tables, about 200 GB.
They are read straight from the checkpoint on disk, a few rows a token.

**Engram row cache.** 8 GiB of RAM that keeps the engram rows already read.

**Prefix cache.** SGLang's radix cache: a prompt that starts like an earlier one reads that part from the cache.

**HiCache.** SGLang's copy of the prefix cache in host memory, so an evicted prefix can come back without a prefill.

**Prebuilt archive.** The compiled pieces of setup in one file ([setup.md](setup.md#prebuilt-pieces)).

**Qualification report.** The output of `tools/qualify.sh`: checks and measurements of one machine ([qualification.md](qualification.md)).

**Reference NLL.** The scores of DeepSeek's own `inference/model.py` on 24 public-domain pieces.
The served model is compared against them.

**TP.** Tensor parallelism: the dense layers split over the GPUs, one rank a GPU.
