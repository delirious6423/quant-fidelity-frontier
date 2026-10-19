---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- gbx-lm
- gbx
---

# Qwen3.8-27B, 4-bit, with its draft head

Qwen3.8-27B quantized to four bits for MLX, carrying the model's own multi-token
prediction head in `mtp/`. Speculative decoding therefore works from this
repository alone -- nothing else to fetch, no environment variable pointing
somewhere else.

**Generation runs 1.8-2.8x faster with the head on, and says the same thing.**
Every token it proposes is checked by the model itself, so the reply is the
model's own either way; the head only saves passes over the weights.

## Speed

Measured on a 48 GB MacBook Pro (M4 Pro), greedy decoding, 96 tokens, with
`gbx_lm`. Decode is timed from the first token, so prefill is not in it.

| context | decode, head off | decode, head on | draft acceptance |
|---|---|---|---|
| 1,024 | 14.7 tok/s | **33.8 tok/s** | 0.95 |
| 4,096 | 14.3 | **27.3** | 0.78 |
| 16,384 | 13.5 | **24.8** | 0.78 |

On a Mac Studio:

Measured 2026-10-13 with `gbx_lm` v0.8.0 on a 512 GB Mac Studio (M3 Ultra), weights
resident, greedy, thinking off, 256 tokens a reply, decode timed from the first token.
Short prompts: three each of code, prose and chat, three runs each, the median per kind.
Long contexts: a slice of source code and a question about it, the median of three runs.

| prompt | head off | head on | speedup |
|---|---|---|---|
| short, code | 37.9 tok/s | **106.1** | 2.80x |
| short, prose | 37.9 | **67.7** | 1.79x |
| short, chat | 38.0 | **86.1** | 2.27x |
| 3,600-token context | 37.0 | **83.9** | 2.27x |
| 15,000-token context | 34.4 | **68.6** | 1.99x |

## Requirements

**Mac**

| | |
| --- | --- |
| macOS | **15.0 or later** |
| chip | **Apple Silicon** (arm64). There is no Intel build. |
| Python | none -- the binary carries what it needs |

**Linux** (gbx-lm v0.8.1 or later)

| | |
| --- | --- |
| machine | **NVIDIA DGX Spark** (GB10). Other Linux machines are not supported yet. |
| system | arm64, glibc 2.39 or later (DGX OS / Ubuntu 24.04) |
| driver | an NVIDIA driver with CUDA 13 support. No CUDA toolkit is needed: the binary carries the CUDA libraries it uses. |
| Python | none |

On a DGX Spark (128 GB) the model fits in memory.

These weights are held resident, not paged: on a 512 GB Mac Studio the model
settles at about **16 GB**, and a machine needs room for that much plus the
conversation's cache.

## Install

```bash
# upgrading? clear the previous version's unpack directory first
rm -rf ~/.libra/cache/onefile/gbx_lm

# on a DGX Spark, use gbx_lm-linux-arm64.tar.gz in place of gbx_lm-darwin-arm64.tar.gz
curl -fL -o gbx_lm-darwin-arm64.tar.gz 'https://github.com/GreenBitAI/gbx-lm/releases/latest/download/gbx_lm-darwin-arm64.tar.gz' \
  && tar -xzf gbx_lm-darwin-arm64.tar.gz gbx_lm \
  && mkdir -p "$HOME/.local/bin" \
  && mv gbx_lm "$HOME/.local/bin/gbx_lm" \
  && chmod +x "$HOME/.local/bin/gbx_lm"

gbx_lm -h
```

The Mac build is signed with a Developer ID and notarised, so macOS runs it
without the usual detour for a downloaded binary.

**`command not found`** -- `$HOME/.local/bin` is not on your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc     # zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile && source ~/.bash_profile   # bash
```

**`Killed: 9`** (Mac) -- a previous version's files are still in the unpack directory,
and macOS refuses to mix two builds. Run the `rm -rf` line above, then try again.

## Run

```bash
gbx_lm --model GreenBitAI/Qwen3.8-27B-4bit

# the draft head is off unless asked for, and found in `mtp/` without a path
GBX_QWEN35_MTP=on gbx_lm --model GreenBitAI/Qwen3.8-27B-4bit
```

That serves an OpenAI-compatible API on port 11688, which is its default. The
weights download on first use into `~/.libra/cache/models`; set `HF_HOME` to put
them elsewhere, and `HF_TOKEN` if you meet the Hub's rate limits for anonymous
downloads.

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/Qwen3.8-27B-4bit","messages":[{"role":"user","content":"Hello"}]}'
```

### Images

The model reads images as well as text. Send them as OpenAI `image_url` parts --
an http(s) URL or a base64 `data:` URL:

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/Qwen3.8-27B-4bit","messages":[{"role":"user","content":[
        {"type":"text","text":"What is in this picture?"},
        {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}]}]}'
```

From the command line, `--images` takes one or more files:
`gbx_lm generate --model GreenBitAI/Qwen3.8-27B-4bit --images photo.jpg --prompt "What is in this picture?" --max-tokens 2048`.

### From the command line

From `gbx_lm` v0.7.1, the binary also runs the model without a server -- one
prompt, or an interactive chat:

```bash
GBX_QWEN35_MTP=on gbx_lm generate --model GreenBitAI/Qwen3.8-27B-4bit --prompt "Hello" --max-tokens 2048
GBX_QWEN35_MTP=on gbx_lm chat --model GreenBitAI/Qwen3.8-27B-4bit --max-tokens 2048
```

`gbx_lm generate -h` and `gbx_lm chat -h` list their options. Keep
`--max-tokens` generous: the default is 100 for `generate` and 256 for `chat`,
and the model's reasoning before it answers counts against it. Leave out
`GBX_QWEN35_MTP=on` to run without the draft head.

## Coding agents

The server speaks three wire protocols on the same port, so the tools that
expect a hosted API can be pointed at this one:

| path | for |
| --- | --- |
| `/v1/chat/completions` | anything written against the OpenAI API |
| `/v1/responses` | **Codex** |
| `/v1/messages` | **Claude Code** |

**Codex** -- a provider in `~/.codex/config.toml`:

```toml
[model_providers.gbx]
name = "gbx-lm"
base_url = "http://127.0.0.1:11688/v1"
wire_api = "responses"
```

and a profile in `~/.codex/gbx.config.toml`:

```toml
model_provider = "gbx"
model = "GreenBitAI/Qwen3.8-27B-4bit"
model_context_window = 262144
```

**Claude Code** -- `~/.claude/gbx.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11688",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "GreenBitAI/Qwen3.8-27B-4bit",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GreenBitAI/Qwen3.8-27B-4bit"
  }
}
```

Both clients ask for a small model for their own background work, so every name
in the settings has to be one this server is serving.

## What is in here

`mtp/mtp.safetensors` is built from the draft head
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ships under `mtp.`,
quantized to match these weights. Apache 2.0, as the original is.
