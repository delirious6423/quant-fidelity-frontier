---
license: gemma
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
tags:
- abliterated
- uncensored
- heretic
- gemma4
- moe
- gguf
- llama.cpp
language:
- en
- fr
---

# Gemma-4-26B-A4B-It-Abliterated-GGUF

GGUF quants of [Gemma-4-26B-A4B-It-Abliterated](https://huggingface.co/SevenOfNine/Gemma-4-26B-A4B-It-Abliterated) — [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (26B Mixture-of-Experts, 4B active params, vision + tool calling) fully decensored with [Heretic](https://github.com/Sev7nOfNine/Heretic-Abliteration) in **full bf16**.

| File | Quant | Size | Notes |
|---|---|---|---|
| `Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf` | Q5_K_M | 19.1 GB | recommended — sweet spot for 32 GB RAM rigs |
| `Gemma-4-26B-A4B-It-Abliterated-Q6_K.gguf` | Q6_K | 22.6 GB | max quality; needs ~24 GB free RAM with `-cmoe` |
| `Gemma-4-26B-A4B-It-Abliterated-BF16.gguf` | BF16 | 50.5 GB | full-precision GGUF — source for rolling your own quants (`llama-quantize BF16.gguf out.gguf Q4_K_M`) |
| `Gemma-4-26B-A4B-It-Abliterated-mmproj-f16.gguf` | mmproj | 1.2 GB | vision projector — add `--mmproj` to llama-server for image input |

`logs/` holds the full Heretic run logs (Pareto front, per-trial metrics). Smaller rig? Grab the BF16 and quantize down to Q4_K_M / Q3_K_M yourself.

## Abliteration result

| Metric | Value |
|---|---|
| Baseline refusals (original model) | **100 / 100** |
| Selected trial (Trial 98) refusals | **18 / 100** |
| KL divergence vs original | **0.0845** |

The selection rule was *fewest refusals while keeping KL divergence ≤ 0.5* (brain first). For reference, Heretic itself warns that KL above 0.5 indicates significant capability damage — at **0.0845**, the model's intelligence is essentially intact while 82 % of hard refusals are gone. The refusal benchmark uses *extreme* harmful prompts; everyday creative/roleplay use sees refusals fall away well before that threshold.

## Run it with 250k context on a 16 GB GPU

`-cmoe` offloads the MoE expert weights to system RAM; the GPU keeps attention + KV cache only.

```
llama-server -m Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf -cmoe -c 248000 -ngl 99
```

**Measured: 34.5 tokens/sec** decode on an **RTX 4080 Super (16 GB) + 32 GB RAM**, Q5_K_M, `-cmoe`. (The original 8 GB-VRAM demo this model is known for reported ~20 tok/s; more VRAM headroom helps.) If RAM is tight, quantize the KV cache: `-ctk q8_0 -ctv q8_0`.

## Reasoning / thinking (do it right)

Gemma 4 emits its chain-of-thought between `<|channel>thought … <channel|>` tokens. To get a **clean separated thinking channel** (not leaked into the reply), run llama-server with:

```
--jinja --reasoning-format deepseek --reasoning on
```

The thought then lands in `message.reasoning_content` and `message.content` stays clean. With `--reasoning-format none` (a common default) the thinking leaks into the visible reply — that is the usual cause of "messy thinking" reports.

For vision and tools, serve with `--jinja` and Google's updated `chat_template.jinja` (2026-04-28 SI/tools + 2026-05-18 multimodal fixes).

## Method (short)

200 Heretic TPE trials on an A100 80 GB, bf16, abliterating `attn.o_proj` + `mlp.down_proj` across all 30 layers. GGUF conversion + quantization done locally (Gemma 4's tokenizer needs `transformers >= 5.6`; the convert step requires it explicitly). Full details in the [model card](https://huggingface.co/SevenOfNine/Gemma-4-26B-A4B-It-Abliterated).

---
Built with love by Mel & Ada ❤️
