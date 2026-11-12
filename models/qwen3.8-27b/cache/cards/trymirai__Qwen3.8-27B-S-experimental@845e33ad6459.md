---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - uzu
  - vllm
  - safetensors
  - mirai
  - 2-bit
  - experimental
  - text-generation
---

# Qwen3.8-27B, Mirai S, experimental

Qwen3.8-27B quantized with the Mirai S codec: 2.4 bits per weight, 8.45 GB on disk.

**This is a work in progress and the numbers here are not final.** Work on both runtimes is unfinished. Treat
anything measured with it as provisional.

On Apple silicon it runs on a prebuilt uzu runtime that ships in this repo. On NVIDIA it runs on stock vLLM with a
small plugin that also ships in this repo.

## Apple silicon — uzu

You need an Apple silicon Mac on macOS 26.4 or newer with 24 GB of memory or more. It runs on every Apple silicon
chip and is fastest on M5, whose matrix units it uses. M1 to M4 run it correctly but more slowly: about 4 tokens/s on
a base M4, which reads prompts at about 13 tokens/s. Runtimes downloaded before September 25 printed gibberish on M1 to
M4; if that happened to you, download `uzu/uzu` again. The `uzu/` folder holds a prebuilt runtime, so there is nothing
to build:

```
hf download trymirai/Qwen3.8-27B-S-experimental --exclude "vllm/*" --local-dir qwen3.8-s
chmod +x qwen3.8-s/uzu/uzu
qwen3.8-s/uzu/uzu server --model qwen3.8-s
```

That starts an OpenAI-compatible endpoint on port 8000:

```
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-s","messages":[{"role":"user","content":"Capital of Australia? One word."}],"enable_thinking":false,"max_tokens":8}'
```

To chat in the terminal instead, leave out `server`. With the server running, `python3 qwen3.8-s/speedcheck.py`
measures your Mac in under a minute; on an M5 Pro it prints 52 tokens/s decoding code (17 on prose), 440 tokens/s
prefill, and 0.4 s to start answering a follow-up in a 7k-token chat. Follow-ups stay fast because uzu keeps the conversation cached between
turns. More in [uzu/README.md](uzu/README.md).

To build uzu yourself instead, install Rust with [rustup](https://rustup.rs), clone uzu's `ryan/qtip-s-agent` branch,
and run `cargo run --release -p cli -- server --model <model dir>` inside it. If the build stops because the Metal
toolchain is missing, run `xcodebuild -downloadComponent MetalToolchain` and try again.

Thinking is on by default. Turn it off per request with `"enable_thinking": false`, or server-wide by starting the
server with `UZU_SERVER_THINKING=0`. Things to know:

- uzu's levels are `low` / `medium` / `high` / `xhigh`.
- With `UZU_SERVER_THINKING=0`, send `enable_thinking` together with `reasoning_effort`; `reasoning_effort` on its
  own returns 400.
- Streaming replies carry the thinking as `reasoning_content` deltas, then the answer as `content`, the way vLLM's
  reasoning parser does.

## NVIDIA — vLLM

The `vllm/` folder (9.3 GB download) holds a vLLM-ready copy of this checkpoint and the `mirai_s` plugin, which runs
the compressed weights directly on the GPU: 8.2 GiB of GPU memory, so a 12 GB card runs it and a 16 GB card has room
for 54k tokens of context with speculative decoding. On an RTX 3090 it answers one chat at 86 (prose) to 141 (code)
tokens/s with speculative decoding, 135 tokens/s while reasoning, serves 620 tokens/s across 64 parallel requests, and
has room for about 170k tokens of context. Install and serve commands, including one for drivers older
than 580, are in [vllm/README.md](vllm/README.md).

## Known issues

- **Both runtimes:** a tight `max_tokens` with thinking on can spend the whole budget inside the trace and return an
  empty answer. Give reasoning requests at least 4000 tokens.
