---
license: other
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: gguf
quantized_by: PollardWeights
pipeline_tag: text-generation
language:
- en
tags:
- pollard-weights
- pollard
- gguf
- llama.cpp
- ik_llama.cpp
- trellis
- imatrix
- mlx
- apple-silicon
- gptq
- vllm
- compressed-tensors
- lfm2_moe
- quantized
- mixed-precision
- measured-allocation
- conversational
---

# LFM2.5-24B-A2B -- Pollard

> ### Pollard shrank this model: **47.94 GB (f16) -> 5.82 GB** -- **88% smaller, 8.2x down**.
> The smallest rung here; larger, higher-fidelity rungs are listed below.
>
> | format | this model's size |
> |---|---:|
> | f16 | 47.94 GB |
> | Q8_0 | ~25.41 GB |
> | Q6_K | 19.68 GB |
> | Q4_K_M | ~13.90 GB |
> | **PollardMix (this repo's IQ1_KT)** | **5.82 GB** |

Pollard builds of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) made with [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- a ladder of **measured-allocation** quants (bits placed by per-layer sensitivity, not a uniform crush).

**Standard GGUF -- runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio, except where noted.** `IQ1_KT` needs [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): its allocation puts ik_llama-only atoms on the tensors it protects. The rest run anywhere.

## Model details

| | |
|---|---|
| Parameter count | ~24.0B |
| Architecture | `lfm2_moe` |
| Input support | text |
| imatrix | **yes** -- see [calibration](#imatrix-calibration) |
| Perplexity measured | **yes** -- table below |

## Which file should I choose?

Every rung is the **same weights**, sized to a different RAM budget by the measured allocation. Pick the largest one that fits your machine with room for context:

- **~21.7 GB RAM / VRAM** -> **`Q6_K`** (19.68 GB).
- **~16.9 GB RAM / VRAM** -> **`IQ4_XS`** (14.85 GB).
- **~13.5 GB RAM / VRAM** -> **`IQ3_S`** (11.53 GB).
- **~10.9 GB RAM / VRAM** -> **`IQ2_XXS`** (8.89 GB).
- **~7.8 GB RAM / VRAM** -> **`IQ1_KT`** (5.82 GB). *(ik_llama.cpp)*

## Available files

| file | PPL | size | Mean KLD | Top-1 agree | runs in | Tool-ready | verified | notes |
|---|---:|---:|---:|---:|---|---|---|---|
| `LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf` | 3.312 | 5.82 GB | 0.3807 | 79.58% | ik_llama | no (78% of f16; CI 67-88) | yes | smallest |
| `LFM2.5-24B-A2B-Pollard-IQ2_XXS.gguf` | 3.7618 | 8.89 GB | 0.5103 | 76.58% | any llama.cpp | yes (92% of f16; CI 83-98) | yes |  |
| `LFM2.5-24B-A2B-Pollard-IQ3_S.gguf` | 2.8826 | 11.53 GB | 0.2111 | 85.20% | any llama.cpp | yes (97% of f16; CI 92-100) | yes |  |
| `LFM2.5-24B-A2B-Pollard-IQ4_XS.gguf` | 2.6572 | 14.85 GB | 0.1119 | 89.37% | any llama.cpp | yes (100% of f16; CI 100-100) | yes |  |
| `LFM2.5-24B-A2B-Pollard-Q6_K.gguf` | 2.5736 | 19.68 GB | 0.0217 | 95.85% | any llama.cpp | yes (100% of f16; CI 100-100) | yes | recommended default |

_Top-1 agree = share of tokens where the rung's most-likely token is the same one the f16 would have picked (llama-perplexity's `Same top p` for GGUF, the same logit comparison in pollard-verify for the other lanes). Higher is closer to the original model._

_Verified = the rung passed `pollard-verify --gate`: its KLD / top-1 against the source and every other threshold in the gate file (retention, tool calls, runtime, coherence) were measured and met. A rung that failed is not published._

_Tool-ready = the rung still calls tools ([pollard-toolcall](https://github.com/WestWaters/pollard-weights)): served with `llama-server --jinja`, asked 60 tool probes (single call, choose among eight tools, no call needed, nested arguments) through OpenAI-style `tools`, and scored on the calls that server parses -- shown as success relative to the reference (the f16 for GGUF, the bf16 original for the other lanes) with a 95% paired-bootstrap CI, or as a bare success rate when no reference run was recorded. "cross-backend" = the reference ran on a different server than the rung; "not measured" = this lane's server could not run where the rung was built._

## Retention vs f16

What each rung keeps of the f16 on real work -- quantized score / f16 score on the same tasks, the same settings, the same harness (100% = no loss). Measured against our own f16, so the rungs compare directly; it is not comparable to a retention figure published on a different suite.

| file | Tool calls |
|---|---:|
| `LFM2.5-24B-A2B-Pollard-Q6_K.gguf` | 100.0% |
| `LFM2.5-24B-A2B-Pollard-IQ4_XS.gguf` | 100.0% |
| `LFM2.5-24B-A2B-Pollard-IQ3_S.gguf` | 96.7% |
| `LFM2.5-24B-A2B-Pollard-IQ2_XXS.gguf` | 91.7% |
| `LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf` | 78.3% |

_Tool calls = the rung's tool-call success as a share of the f16's ([pollard-toolcall](https://github.com/WestWaters/pollard-weights)); "valid" when no f16 run was recorded. Decisions = share of typed decisions identical to the f16's._

## Every format in this repo

One repo holds every Pollard build of this model: the GGUF files above sit at the root, each other format in its own folder: `MLX/`, `GPTQ/`

## MLX (Apple Silicon) -- `MLX/`

Each rung is a complete checkpoint in its own folder under `MLX/`. Download only the one you want:

| folder | size | avg bits | PPL | Mean KLD | Top-1 agree | Tool-ready | verified |
|---|---:|---:|---:|---:|---:|---|---|
| `MLX/6bit/` | 21.73 GB | 6.7 | 3.6891 | 0.0429 | 94.86% | not measured (mlx_lm.server with the whole rung needs ~26.0 GB of RAM and 5.0 GB is usable here (7.0 GB free, 2 GB floor)) | yes |
| `MLX/4bit/` | 17.98 GB | 5.4 | 3.6652 | 0.1065 | 91.57% | not measured (mlx_lm.server with the whole rung needs ~21.7 GB of RAM and 6.5 GB is usable here (8.5 GB free, 2 GB floor)) | yes |
| `MLX/3bit/` | 13.87 GB | 4.05 | 3.6628 | 0.2104 | 86.96% | not measured (mlx_lm.server with the whole rung needs ~16.9 GB of RAM and 6.2 GB is usable here (8.2 GB free, 2 GB floor)) | yes |

```bash
hf download PollardWeights/LFM2.5-24B-A2B-Pollard --include "MLX/3bit/*" --local-dir ./LFM2.5-24B-A2B-Pollard
mlx_lm.server --model ./LFM2.5-24B-A2B-Pollard/MLX/3bit
```

## GPTQ (vLLM / SGLang) -- `GPTQ/`

Each rung is a complete checkpoint in its own folder under `GPTQ/`. Download only the one you want:

| folder | size | avg bits | PPL | Mean KLD | Top-1 agree | Tool-ready | verified |
|---|---:|---:|---:|---:|---:|---|---|
| `GPTQ/int8/` | 24.87 GB | 8.0 | 3.6871 | 0.0289 | 96.50% | not measured (47.9GB model needs ~56.1GB of RAM, 18.7GB free) | yes |
| `GPTQ/int4/` | 17.31 GB | 5.47 | 3.6627 | 0.1367 | 89.43% | not measured (47.9GB model needs ~56.1GB of RAM, 16.7GB free) | yes |

```bash
hf download PollardWeights/LFM2.5-24B-A2B-Pollard --include "GPTQ/int4/*" --local-dir ./LFM2.5-24B-A2B-Pollard
vllm serve ./LFM2.5-24B-A2B-Pollard/GPTQ/int4 --enable-auto-tool-choice --tool-call-parser lfm2
```

## Download a specific file

```bash
pip install -U "huggingface_hub[cli]"
hf download PollardWeights/LFM2.5-24B-A2B-Pollard \
  --include "LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf" --local-dir ./
```

## How to run

`IQ1_KT` is built on ik_llama-only atoms, so it runs with **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**:

```bash
llama-cli    -m LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf -ngl 99 -p "Explain why the sky is blue."
llama-server -m LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf -ngl 99
```

For stock llama.cpp, Ollama or LM Studio, use `Q6_K` instead:

```bash
llama-server -hf PollardWeights/LFM2.5-24B-A2B-Pollard:Q6_K
llama-cli    -m LFM2.5-24B-A2B-Pollard-Q6_K.gguf -ngl 99 -p "Explain why the sky is blue."
```

## Tool calling

The chat template writes tool calls in the **`lfm2`** syntax (detected by rendering it with [pollard-modelkind](https://github.com/WestWaters/pollard-weights)):

```
<|tool_call_start|>[get_weather(city='Paris', days=7)]<|tool_call_end|>
```

Serve with:

- **llama.cpp** (GGUF): `llama-server -m LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf --jinja` -- the model's own template renders the tools and llama.cpp parses the calls (specialised: LFM2.5); pass `tools` in `/v1/chat/completions`.
- **vLLM**: `vllm serve <the original weights> --enable-auto-tool-choice --tool-call-parser lfm2` (parser name not yet confirmed for this model)

## Use it in your harness

Any agent harness that speaks the OpenAI, Anthropic or Responses API can drive this model through llama-server. [Pollard Weights](https://github.com/WestWaters/pollard-weights)' `pollard serve` picks the runtime this rung needs, sizes the context to the harness and prints its exact config (`--write` merges it in, after backing the original up):

```bash
pollard serve PollardWeights/LFM2.5-24B-A2B-Pollard:IQ1_KT --harness claude-code      # or omp, codex, opencode, aider, ...
```

| harness | calls | min. context | setup |
|---|---|---:|---|
| Oh My Pi (`omp`) | `/v1/chat/completions` | 32k | `~/.omp/agent/models.yml`: provider `api: openai-completions`, `baseUrl: http://127.0.0.1:8080/v1`, `auth: none` |
| Claude Code | `/v1/messages` | 64k | `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` `ANTHROPIC_AUTH_TOKEN=pollard` `ANTHROPIC_MODEL=LFM2.5-24B-A2B-Pollard-IQ1_KT` |
| Codex CLI | `/v1/responses` | 32k | `~/.codex/config.toml`: `[model_providers.pollard]` `base_url = "http://127.0.0.1:8080/v1"`, `wire_api = "responses"` |
| OpenCode | `/v1/chat/completions` | 32k | `opencode.json`: provider `npm: @ai-sdk/openai-compatible`, `options.baseURL: http://127.0.0.1:8080/v1`, `limit.context` |
| Cline / Roo Code | `/v1/chat/completions` | 32k | API Provider "OpenAI Compatible", Base URL `http://127.0.0.1:8080/v1` |
| Aider | `/v1/chat/completions` | 16k | `OPENAI_API_BASE=http://127.0.0.1:8080/v1 aider --model openai/LFM2.5-24B-A2B-Pollard-IQ1_KT` |
| Continue | `/v1/chat/completions` | 32k | `config.yaml`: `provider: llama.cpp`, `apiBase: http://127.0.0.1:8080` |
| Goose | `/v1/chat/completions` | 32k | `GOOSE_PROVIDER=openai` `OPENAI_HOST=http://127.0.0.1:8080` |
| OpenHands | `/v1/chat/completions` | 32k | model `openai/LFM2.5-24B-A2B-Pollard-IQ1_KT`, base URL `http://host.docker.internal:8080/v1` |
| Open WebUI | `/v1/chat/completions` | 8k | Admin Settings -> Connections -> OpenAI API: `http://127.0.0.1:8080/v1` |
| Ollama | own runtime | 8k | `ollama run hf.co/PollardWeights/LFM2.5-24B-A2B-Pollard:Q6_K`, then `/set parameter num_ctx 8192` or more |
| LM Studio / Jan | own runtime | 8k | load the GGUF, set the context length to 8192 or more |

Without Pollard, start the llama-server from ik_llama.cpp yourself (this rung needs it) with a context the harness fits in and the model's own template:

```bash
llama-server -m LFM2.5-24B-A2B-Pollard-IQ1_KT.gguf -c 65536 --jinja --alias LFM2.5-24B-A2B-Pollard-IQ1_KT -np 1 --port 8080
```

_Too small a context fails silently: the harness's system prompt and tool schemas are truncated and the model looks like it ignores its tools. Claude Code needs 64k._

## imatrix (calibration)

The importance matrix (`LFM2.5-24B-A2B.imatrix`, included) was computed on the pollard-calib 'auto' profile (sources, revisions and licences in the shipped .sources.json).

The exact corpus is included as `LFM2.5-24B-A2B-calib.txt`, so the allocation can be reproduced rather than taken on trust.

## ARM / AVX

llama.cpp repacks weights into an interleaved layout at load time for faster inference on ARM and AVX machines -- no special file needed, online repacking covers these quants. The old `Q4_0_4_4/4_8/8_8` variants are not required.

## Errata

- `IQ1_KT` carries ik_llama-only atoms and needs ik_llama.cpp to run; stock llama.cpp rejects any ggml type above 42 outright. Checked with `pollard-ggufcheck`, from the files' tensor types rather than their names.
- Measured allocation places bits by per-layer sensitivity under a size budget.
- Single machine; replication invited.

## Credits & license

- Base model: [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) (LiquidAI)
- Quantization tooling: [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org)
- Method + tooling: [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- *measure first, no claim before a number.*
- License: `other`, inherited from the base model.

*Built with [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- frontier models, small hardware, no compromise.*
