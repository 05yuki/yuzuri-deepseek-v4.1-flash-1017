# Serving

The server speaks SGLang's OpenAI-compatible API on `127.0.0.1:8080`.
It runs one request at a time.
More than one request at a time has not been measured with this repository.

To serve other machines, set `SGLANG_HOST=0.0.0.0` and `SGLANG_API_KEY=<key>`.
Clients then send `Authorization: Bearer <key>`.

## The prefix cache

The prefix cache is on, with a copy in host memory (SGLang's HiCache).

- A prompt that starts like an earlier one reads that part from the cache.
- The same 35,000-token prompt again takes 2 s instead of 20 s.
- A prefix the GPU has given up is loaded back from host memory instead of computed again.
- The host copy is twice the GPU pool (`DSV41_HICACHE_RATIO`).
- `DSV41_HICACHE=0` keeps the cache on the GPU only.

## Sampling

The model card recommends `temperature` 1.0 and `top_p` 1.0.
A request that sets neither gets these.

## Tool calls and thinking

The server starts with V4.1's tool-call parser and reasoning parser.
The OpenAI API returns a tool call in `message.tool_calls` and the model's thinking in `reasoning_content`, streamed or not.

- Thinking is off unless a request turns it on: `"chat_template_kwargs": {"thinking": true}`.
- The prompt is rendered by SGLang's `deepseek-recipe` package.
- It differs from the checkpoint's `encoding/encoding.py` in three places, tool namespaces among them ([model.md](model.md#encoding-encoding)).
- `tools/tool-call-check.py [URL]` checks a running server: a call, the same call streamed, the answer after the tool's result, and a call with thinking on.
- All four pass; a namespaced tool is reported as the known gap above.
