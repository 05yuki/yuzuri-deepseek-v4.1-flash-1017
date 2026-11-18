# Qualification

`MODEL=<dir> tools/qualify.sh [report.md]` writes a report for the machine it runs on, as one Markdown file.

- It starts `./serve.sh`, runs the checks and the measurements, and stops the server.
- It takes about 15 minutes after the server is up.
- The report names the hardware and software.
- It holds no host name, user, path, address or serial number.
- Exit status 0 means every check passed.

## Checks

| check | passes when |
|---|---|
| answer | A chat request comes back with text. |
| tool calls | `tools/tool-call-check.py` passes. |
| NLL | The served model's mean NLL is within 0.01 nats a token of the reference's (`QUALIFY_NLL_TOL`). |
| prefix cache | The same 35,000-token prompt, sent twice, is read from the cache the second time. |

## Measured

- Decode on Japanese prose: four prompts, 1,024 tokens each, the median.
- Decode on six other kinds of text (`tools/bench.py --other-text`).
- The prefill of a 35,000-token prompt of random token ids, twice with different ids.
- The time until the server is up, and the hot expert slots at the first lend.

`tools/bench.py` measures the same against a running server.
`tools/bench.py --prefill-only --lengths 131072,524288,1000000` measures longer prompts; it takes about an hour.

## The NLL

The served model is scored against the checkpoint's own reference implementation.

- The reference is DeepSeek's `inference/model.py`, run one layer at a time on one 16 GB GPU (`tools/ref-nll.py`).
- Both score the same 24 pieces of 480 tokens, teacher-forced.
- The pieces come from two public-domain texts in `tools/nll-text/`, one Japanese and one English.
- `tools/nll-ref-1017.json` holds the reference's scores on an RTX 5070 Ti; its mean is 0.7185.
- `QUALIFY_REF=<file>` compares against a reference run on this machine instead.
- Launches of one build differ by about 0.002.
- The texts are well known, so the model predicts them closely and the NLL is low.
- The NLL compares the two implementations on the same tokens; it does not measure the model.

To repeat it by hand:

```
python tools/ref-nll.py <checkpoint dir> ref.json      # about 4 minutes on one 16 GB GPU, no server
MODEL=<checkpoint dir> ./serve.sh --background
python tools/nll.py score <checkpoint dir> served.json
python tools/nll.py compare ref.json served.json
```

## Reports

The reports of the test machine are in [reports/](../reports/).
Reports from other GPUs, CPUs and drives are welcome as an issue or a pull request.

## Tests

The CI job "patches apply" runs on each push and once a week, without a GPU.

- Every patch applies to its pinned commit.
- The pinned Yuzuri commit exists.
- The llama.cpp MMQ sources are fetched, patched and checked byte for byte.
- `tools/test_sm89_dispatch.py` checks the SM89 dispatch on the CPU.
- Every script parses.

| test | needs | where it runs |
|---|---|---|
| `tools/test_sm89_dispatch.py` | the patched SGLang tree | CI |
| `tools/dsv4-o-fp8-qdq-test.py` | a GPU | `setup.sh`, on the machine it builds |
| `tools/e4m3-emulation-test.py` | torch | by hand |
| `tools/sm89-attention-test.py` | an SM89 GPU | by hand |
| `tools/dsv41-encoding-check.py` | the checkpoint's `encoding/` and tokenizer | by hand |
