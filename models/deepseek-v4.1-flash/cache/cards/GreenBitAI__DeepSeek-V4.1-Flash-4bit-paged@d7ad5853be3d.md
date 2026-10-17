---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
- mlx
- gbx-lm
- apple-silicon
- deepseek_v41
- mixture-of-experts
- 4-bit
- expert-paging
library_name: mlx
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash-4bit-paged

Expert-paged build of `pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit`. The weights that are read a fraction at a time
live in their own containers, so a machine loads what it needs rather than all
of it.

| file | size | holds |
| --- | --- | --- |
| `model.safetensors` | 10.48 GiB | resident weights |
| `experts.bin` | 284.77 GiB | routed experts |
| `engram-layer-1/` | 51.50 GiB | engram table, layer 1 |
| `engram-layer-14/` | 51.50 GiB | engram table, layer 14 |
| `mtp/` | 7.85 GiB | draft head, off by default |

Total 406.10 GiB. The bytes moved into containers of their own; they were not copied.

These containers are not a format `mlx-lm` reads. The model runs on
**`gbx_lm`**, a single signed binary for Apple Silicon; there is nothing to
`pip install`.

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

On a DGX Spark (128 GB) the model does not fit, so its experts are paged from the SSD.

Memory is not a fixed figure for a paged build, and that is the point of one: it
fills what fits and streams the rest from disk. On a 512 GB Mac Studio with room
to spare this model settles at about **306 GB** resident. A smaller machine holds
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
gbx_lm --model GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged
```

That serves an OpenAI-compatible API on port 11688, which is its default. The
weights download on first use into `~/.libra/cache/models`; set `HF_HOME` to put
them elsewhere, and `HF_TOKEN` if you meet the Hub's rate limits for anonymous
downloads.

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged","messages":[{"role":"user","content":"Hello"}]}'
```

Where the weights fit they are filled from `experts.bin` and the model runs the
stock path at stock speed; where they do not, they stream from disk. Reading
the machine decides that, not a flag.

To override that: `GBX_PAGING=off` holds the experts resident, `GBX_ENGRAM=off` holds the engram tables resident.

The directory also carries DeepSeek's own DSpark draft head, off unless
asked for: `GBX_DEEPSEEK_MTP=on` decodes speculatively against it. It is
converted from `deepseek-ai/DeepSeek-V4.1-Flash` and keeps that licence; no
published MLX build carries it, which is why it is here.

Checked at build time, while the source checkpoint was still there to compare against:

- **PASS** bit-identical logits — exact on 5 prompt(s) to 160 tokens; 1 longer differ by at most 15, same greedy token throughout
- **PASS** layer-wise vs resident — 40 layers x 2 draws exact, 377.86 GiB peak for this gate

Quantization, tokenizer, chat template and licence are unchanged from `pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit`.

### From the command line

From `gbx_lm` v0.7.1, the binary also runs the model without a server -- one
prompt, or an interactive chat:

```bash
GBX_DEEPSEEK_MTP=on gbx_lm generate --model GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged --prompt "Hello" --max-tokens 2048
GBX_DEEPSEEK_MTP=on gbx_lm chat --model GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged --max-tokens 2048
```

`gbx_lm generate -h` and `gbx_lm chat -h` list their options. Keep
`--max-tokens` generous: the default is 100 for `generate` and 256 for `chat`,
and the model's reasoning before it answers counts against it. Leave out
`GBX_DEEPSEEK_MTP=on` to run without the draft head.

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
model = "GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged"
model_context_window = 1048576
```

**Claude Code** -- `~/.claude/gbx.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11688",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged"
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
GBX_DEEPSEEK_MTP=on gbx_lm --model GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged
```

**Up to 2.3x, and it holds at a long context.** Measured 2026-09-18 on a 512 GB
Mac Studio (M3 Ultra) over HTTP, in an agent loop carrying tool definitions and a
session cache, decode timed from the first token; the median request of each
session:

| context | temperature | head off | head on | speedup | accepted |
|---|---|---|---|---|---|
| 38-41k | 0 | 20.3 tok/s | **46.6** | **2.29x** | 74% |
| 38-47k | 0.7 | 20.3 | **42.1** | 2.08x | 79% |
| 21-23k | 0 | 20.5 | **43.8** | 2.14x | 76% |
| 21-22k | 0.7 | 20.4 | **37.7** | 1.85x | 71% |

Outside an agent loop -- short prompts, and a plain long context without tools --
the head accepts less (about 52% on short prompts, 64-65% with a long context) and
the gain is smaller.
Measured 2026-10-13 with `gbx_lm` v0.8.0 on a 512 GB Mac Studio (M3 Ultra), weights
resident, greedy, thinking off, 256 tokens a reply, decode timed from the first token.
Short prompts: three each of code, prose and chat, three runs each, the median per kind.
Long contexts: a slice of source code and a question about it, the median of three runs.

| prompt | head off | head on | speedup |
|---|---|---|---|
| short, code | 22.5 tok/s | **41.4** | 1.84x |
| short, prose | 22.6 | **30.6** | 1.35x |
| short, chat | 22.6 | **33.2** | 1.47x |
| 3,700-token context | 22.5 | **42.7** | 1.90x |
| 15,000-token context | 22.9 | **41.5** | 1.81x |
| 31,000-token context | 22.6 | **42.6** | 1.88x |

It does not fade as the history grows: `index_topk` bounds attention during
decode, so the backbone's cost per token is nearly flat and the head's saving is
not swamped by a growing denominator. It wins most where the next token is
guessable -- source code over prose. Every token it proposes is verified by the
model itself, so the reply is the model's own either way; the head only saves
passes over the weights.
