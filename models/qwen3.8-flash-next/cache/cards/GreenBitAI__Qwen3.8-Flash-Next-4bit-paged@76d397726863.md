---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gbx-lm
  - mlx-vlm
  - omlx
  - qwen
  - qwen3.8
  - mixture-of-experts
  - vision-language
  - quantized
  - apple-silicon
  - 4-bit
---

# Qwen3.8-Flash-Next-4bit-paged

Expert-paged build of `Vontra/Qwen3.8-Flash-Next-MLX-4bit`. The weights that are read a fraction at a time
live in their own containers, so a machine loads what it needs rather than all
of it.

| file | size | holds |
| --- | --- | --- |
| `model.safetensors` | 3.80 GiB | resident weights |
| `experts.bin` | 70.31 GiB | routed experts |
| `ple-q4.rows` | 29.80 GiB | n-gram table |
| `mtp/` | 1.52 GiB | draft head, off by default |

Total 105.46 GiB. Of that, 103.94 GiB is the source build, whose bytes moved into

These containers are not a format `mlx-lm` reads. The model runs on
**`gbx_lm`**, a single signed binary for Apple Silicon; there is nothing to
`pip install`.
containers rather than being copied, and 1.52 GiB is the draft head, which no
published build of this model carries.

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

On a DGX Spark (128 GB) the model fits in memory and nothing is paged. Prompt processing there,
gbx-lm v0.8.2 against v0.8.3 (the median of three runs, same machine and prompts) -- long
prompts no longer slow down:

| | v0.8.2 | v0.8.3 |
|---|---|---|
| 4k-token prompt | 1028 tok/s | **1274** tok/s |
| 16k-token prompt | 911 | **1243** |
| 54k-token prompt | 480 | **1065** |
| 54k-token conversation, first answer / next 1.3k-token message | 117 s / 4.9 s | **54 s / 2.2 s** |

Decode on short prompts is 24 tok/s, and **41** with the draft head (averaged over
code, prose and chat).

Memory is not a fixed figure for a paged build, and that is the point of one: it
fills what fits and streams the rest from disk. On a 512 GB Mac Studio with room
to spare this model settles at about **76 GB** resident. A smaller machine holds
less and reads more from disk -- slower, but it runs.

How much slower depends on how far the machine is from holding the experts, and
on how fast its disk is. Each token routes to a few experts; the ones already in
memory cost nothing to reach, and the ones that are not have to be read before
that token can finish. A machine holding most of them waits rarely, one holding
few waits often. We have not measured this across machine sizes and will not
guess a figure: what we can say is that the model answers either way, and that
the wait is the SSD's, not the model's.

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
gbx_lm --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged
```

That serves an OpenAI-compatible API on port 11688, which is its default. The
weights download on first use into `~/.libra/cache/models`; set `HF_HOME` to put
them elsewhere, and `HF_TOKEN` if you meet the Hub's rate limits for anonymous
downloads.

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/Qwen3.8-Flash-Next-4bit-paged","messages":[{"role":"user","content":"Hello"}]}'
```

Where the weights fit they are filled from `experts.bin` and the model runs the
stock path at stock speed; where they do not, they stream from disk. Reading
the machine decides that, not a flag.

To override that: `GBX_PAGING=off` holds the experts resident, `GBX_PLE=off` holds the n-gram table resident.

Checked at build time, while the source checkpoint was still there to compare against:

- **PASS** bit-identical logits — exact on 5 prompt(s) to 160 tokens; 1 longer differ by at most 6.8, same greedy token throughout
- **PASS** layer-wise vs resident — 48 layers x 2 draws exact, 88.89 GiB peak for this gate

Quantization, tokenizer, chat template and licence are unchanged from `Vontra/Qwen3.8-Flash-Next-MLX-4bit`.

### Images

The model reads images as well as text. Send them as OpenAI `image_url` parts --
an http(s) URL or a base64 `data:` URL:

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/Qwen3.8-Flash-Next-4bit-paged","messages":[{"role":"user","content":[
        {"type":"text","text":"What is in this picture?"},
        {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}]}]}'
```

From the command line, `--images` takes one or more files:
`gbx_lm generate --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged --images photo.jpg --prompt "What is in this picture?" --max-tokens 2048`.

### From the command line

From `gbx_lm` v0.7.1, the binary also runs the model without a server -- one
prompt, or an interactive chat:

```bash
GBX_QWEN4_MTP=on gbx_lm generate --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged --prompt "Hello" --max-tokens 2048
GBX_QWEN4_MTP=on gbx_lm chat --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged --max-tokens 2048
```

`gbx_lm generate -h` and `gbx_lm chat -h` list their options. Keep
`--max-tokens` generous: the default is 100 for `generate` and 256 for `chat`,
and the model's reasoning before it answers counts against it. Leave out
`GBX_QWEN4_MTP=on` to run without the draft head.

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
model = "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged"
model_context_window = 262144
```

**Claude Code** -- `~/.claude/gbx.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11688",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged"
  }
}
```

Both clients ask for a small model for their own background work, so every name
in the settings has to be one this server is serving.

## The draft head

The `mtp/` folder carries the model's own multi-token prediction head, so
speculative decoding works from this repository alone. It is off unless asked
for:

```bash
GBX_QWEN4_MTP=on gbx_lm --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged
```

**Up to 2.45x.**
Measured 2026-10-13 with `gbx_lm` v0.8.0 on a 512 GB Mac Studio (M3 Ultra), weights
resident, greedy, thinking off, 256 tokens a reply, decode timed from the first token.
Short prompts: three each of code, prose and chat, three runs each, the median per kind.
Long contexts: a slice of source code and a question about it, the median of three runs.

| prompt | head off | head on | speedup |
|---|---|---|---|
| short, code | 51.8 tok/s | **126.8** | 2.45x |
| short, prose | 51.6 | **83.1** | 1.61x |
| short, chat | 52.0 | **95.9** | 1.84x |
| 3,600-token context | 43.5 | **84.6** | 1.94x |
| 15,000-token context | 42.0 | **70.9** | 1.69x |
| 31,000-token context | 40.3 | **63.2** | 1.57x |

Code gains most and prose least. The head accepts about 60% of what it proposes,
short prompt or long.

Every token the head proposes is checked by the model itself, so the reply is the
model's own either way; the head only saves passes over the weights.
