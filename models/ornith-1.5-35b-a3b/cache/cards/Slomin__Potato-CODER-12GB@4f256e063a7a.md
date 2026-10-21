---
license: mit
base_model:
  - peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
language:
  - en
  - zh
tags:
  - qwen
  - qwen35moe
  - moe
  - 35b-a3b
  - potato-coder
  - potato
  - coding
  - agentic-coding
  - long-context
  - 131k
  - iq3_xxs
  - gguf
  - llama.cpp
  - 12gb
---

# Potato-CODER-12GB (v1.0-rc1)

A 35B-A3B coding **agent** that runs on **one 12 GB NVIDIA card** (RTX 4070 / 3060 class) at a **131,072-token context**, with
vision and tool calling on. The weights are
[peculiar-ragdoll's Tiel-Coder-35B-A3B UD-IQ3_XXS](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF), unmodified;
what is ours is the **deployment recipe** — the context, offload, cache, sampling and thinking-budget settings that make a model
this size work on a card this size, and the fit testing behind them. Nothing was trained or requantized here.

| | |
|---|---|
| architecture | Ornith-1.5-35B-A3B, a mixture-of-experts model (`qwen35moe` in GGUF), ~35B total / ~3B active, 26 of the expert layers served from system RAM |
| quantization | UD-IQ3_XXS (unsloth-dynamic recipe, peculiar-ragdoll's own imatrix), ~3.0 bits/weight, 12.3 GiB on disk |
| context | **131,072** tokens, KV cache q8_0 / q8_0, one slot |
| VRAM | peak **8,949 MiB** of an 11,836 MiB envelope (12 GB card less a 452 MiB desktop/driver reserve) — 2,886 MiB spare, with the Q8_0 vision projector resident |
| system RAM | 16 GB recommended; our runs capped the server at 10 GiB and it stayed inside that |
| speed (RTX 3090, one slot, in the 12 GB envelope) | decode **53 tok/s**, prefill **744 tok/s** at 60k, load 18 s, image turn 4.9 s, tool call 2.7 s |
| vision | Q8_0 projector resident on the GPU; screenshots and photographs verified at full context depth |
| tools | tool calling verified in this configuration |

This is the stronger of the two on repository work. If you want speed instead, see
**[Potato-CODER-12GB-Fast](https://huggingface.co/Slomin/Potato-CODER-12GB-Fast)**: about 2.4× faster on the same work, at the
same 131,072 context, with the whole model resident on the GPU and nothing in system RAM — and the weaker of the two on
multi-step repository work.

## Files

| file | what it is |
|---|---|
| `Potato-CODER-12GB-UD-IQ3_XXS.gguf` | the model — peculiar-ragdoll's `Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf` (sha256 `3317d98a…`) with the `general.*` metadata rewritten for this release; **every tensor verified byte-identical** to the source |
| `mmproj-Potato-CODER-12GB-Q8_0.gguf` | vision projector, Q8_0 — peculiar-ragdoll's `mmproj-Q8_0.gguf` (sha256 `18bd953a…`) from their Cyber-Tiel MTP repository, same treatment; it is the vision tower published with this model family |
| `serve-potato.bat` | Windows start script |
| `LICENSE`, `NOTICE`, `SHA256SUMS`, `recipe.txt` | licence, attribution, checksums, the exact serving recipe |

The chat template (peculiar-ragdoll's "Sharp" template) is **inside** the GGUF; `--jinja` is what activates it.

## Running it (llama.cpp)

This model needs no patched runtime — current upstream `llama.cpp` serves it. We publish binaries at
[github.com/potato-os/llama.cpp/releases](https://github.com/potato-os/llama.cpp/releases) built for a wider set of GPU
architectures (SASS for sm_75/80/86/89/120 plus generic PTX, so newer cards still run), with the Windows script below tested on
every build.

### Linux

```bash
./llama-server -m Potato-CODER-12GB-UD-IQ3_XXS.gguf \
  -c 131072 -ngl 99 --n-cpu-moe 26 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 --no-context-shift \
  --mmproj mmproj-Potato-CODER-12GB-Q8_0.gguf \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning-budget 16384 \
  --reasoning-budget-message $'\n\nThinking budget reached. Stop deliberating now: state the approach in one or two sentences and write the complete final code.\n' \
  --chat-template-kwargs '{"preserve_thinking": false, "reasoning_effort": "medium"}'
```

The server listens on `http://127.0.0.1:8080` with an OpenAI-compatible API; point your agent (OpenCode, Cline, mini-SWE-agent…)
at `http://127.0.0.1:8080/v1`.

### Windows

1. Download this repository into one folder.
2. From the [releases page](https://github.com/potato-os/llama.cpp/releases) take `llama-potato-bin-win-cuda-12.4-x64.zip`
   (RTX 20xx–40xx) or `...-13.4-...` (RTX 50xx) **and** the matching `cudart-` zip; unzip both into that same folder.
3. Double-click `serve-potato.bat`. The model is ready when the window shows `listening on http://127.0.0.1:8080`.

Two Windows specifics:
- **Turn off the driver's RAM spill-over**: NVIDIA Control Panel → Manage 3D settings → Program Settings → add
  `llama-server.exe` → *CUDA - Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**. Otherwise a full card quietly spills into
  system RAM and generation collapses to a few tokens per second instead of reporting an error.
- The desktop takes part of the card. If the server fails to allocate, lower `-c` to `114688` first, then raise `--n-cpu-moe`.

## The knobs that matter

- **`--n-cpu-moe 26`** is the setting, not a fallback. 131k context fits with 22 offloaded layers too — and at 22 the model
  *fails our retrieval-at-depth check*, while at 26 it passes. Every failure we saw on the context ladder tracked the offload
  setting rather than the context length, and none of them was an out-of-memory error. If you shorten the context, keep the
  offload.
- **KV cache q8_0 / q8_0.** Not lower.
- **Thinking budget 16,384 tokens + the wrap-up message** (both in the commands above). The budget alone truncates; the message
  is what turns a capped-out chain into a finished answer. `reasoning_effort` is accepted by this model's template.
- **`--presence-penalty 0`.** A repetition penalty is a reasonable chat setting and a bad agent setting: an agent legitimately
  repeats file paths and re-issues similar commands, and penalising that costs solved tasks.
- One slot. Every number here is single-slot; more slots divide the context.

## How it was put together

19–20 September 2026, on RTX 3090s held down to a 12 GB envelope. Six candidate models were served in that envelope and compared
on fit, context depth, speed and agentic coding; this one and its fast sibling were the two kept. The configuration above is the
highest context/offload rung that passed every fit gate: single-slot property checks, retrieval at half and full depth, a
near-limit turn, three repeated image rounds (screenshots and photographs, including after a deep context fill), a VRAM-growth
check and a projector-placement check.

Our comparison numbers are internal and not published here.

## Limits

- The 2,886 MiB of spare VRAM is measured on a headless RTX 3090 held to a 12 GB envelope; a real 12 GB card with a desktop on it
  behaves slightly differently, and other drivers differ by a few hundred MiB.
- Everything was measured on Linux. The Windows script is tested for start-up in CI on every build; Windows speed is unmeasured.
- Nothing here measures multi-user throughput or quantizations other than the one shipped.

## Licence and attribution

The model weights are MIT, as declared by their upstream repositories (neither ships a licence file; `LICENSE` here is the MIT
text with the upstream copyright holders named — see `NOTICE`).

- **ornith-ai (Deep Reinforce)** — Ornith-1.5-35B-A3B, the model (MIT). Its card describes it as extending Ornith-1.0, which
  was built on Qwen3.5 and Gemma4 with continued pretraining, mid-training and post-training; the MIT grant over the resulting
  weights is ornith-ai's.
- **peculiar-ragdoll** — the UD-IQ3_XXS quantization, its imatrix, the Q8_0 vision projector and the Sharp chat template embedded
  in the GGUF (MIT).
- **Slomin** — the 12 GB deployment recipe and the fit testing behind it. No weights were modified.
