---
license: apache-2.0
base_model:
- harness-hub/Qwen3.8-27B-pi-adapted
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- imatrix
- llama.cpp
- ollama
- agentic-coding
- coding-agent
- tool-use
- pi
- qwen3.8
- pithos
---

![Pithos-27B](banner.png)

# Pithos-27B · Pandora · IQ3_XXS GGUF

**A coding agent that gets to the point, in under 12 GB.**

In the myth, the *pithos* was the jar Pandora opened. This one holds a coding agent. harness-hub trained a LoRA adapter on Qwen3.8-27B with reinforcement learning so it works as the model behind the [pi](https://github.com/earendil-works/pi) coding agent: find the code, edit it, run the tests, stop. On their benchmarks it gets most of the gain of maximum-effort thinking **with thinking switched off**, in fewer turns. **Pithos** is that adapter merged into the base model and quantized for the 8–12 GB band, where most people actually run local models. It's an importance-matrix-calibrated **IQ3_XXS** with a hand-tuned tensor mix. Before publishing, we checked that it still *acts* as an agent: it had to resolve real SWE-bench issues end to end through a local agent harness.

> The name and the banner are just packaging. The weights are [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (Apache-2.0) merged into [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) at the adapter's own scale (the standard PEFT merge), then quantized. No extra training, no other changes. Every number below was measured by us on this file.

| File | Quant | Size |
|---|---|---|
| `Pithos-27B-Pandora-IQ3_XXS.gguf` | IQ3_XXS + upgraded attention/SSM tensors (imatrix) | **11.89 GB** |
| `pithos-27b.imatrix` | importance matrix (174 × 2048 tokens) | 0.01 GB |

**Text-only.** Qwen3.8-27B is multimodal and ships an MTP head. This GGUF contains the language model only: no vision projector and no MTP layer. The adapter card notes that the vision tower isn't used under pi either.

**Runtimes.** We ran this file in Ollama 0.40.13 (native tool calls work) and in llama.cpp master (`dbaf582af`).

## What we measured (real numbers, this file)

| Benchmark | Setting | **Pithos IQ3_XXS (11.89 GB)** | netprojects IQ3_XXS (11.43 GB) |
|---|---|---|---|
| **SWE-bench Verified**, 5-instance probe | local agent harness, ~12K-char context budget, non-thinking, native tool calls, temp 0.8 / top_p 1.0 | **4/5 resolved** | not run |
| **HumanEval** pass@1 (base / plus) | non-thinking, temp 0.6, top_p 0.95, seed 1234, 2048 max new tokens | **0.963 / 0.921** | 0.939 / 0.878 |
| Runaway generations on HumanEval (hit the token cap) | same | 0/164 | 1/164 |
| Wikitext-2 perplexity (lower is better) | 40 × 2048-token chunks | **6.254 ± 0.074** | 6.430 ± 0.077 |

HumanEval and perplexity for both files were run on the same engine (llama.cpp master `dbaf582af`, `llama-server --jinja`, `enable_thinking=false`) with identical prompts, seeds and scorer (evalplus).

**Why netprojects and not unsloth?** Unsloth hasn't published a GGUF of this adapter. The most-downloaded same-size imatrix quant is netprojects' IQ3_XXS. It's 11.43 GB on disk, but 0.24 GB of that is the MTP draft layer, which isn't used for normal decoding. Its main-model weights come to 11.18 GB vs our 11.89 GB, so ours is about 6% larger. We spent that extra 0.7 GB on purpose (see the tensor mix below).

**How to read the comparison:** Pithos solved 4 more HumanEval problems (158 vs 154) and 7 more HumanEval+ problems (151 vs 144), and its perplexity is 2.7% lower. Part of that comes from the extra 0.7 GB, so it isn't free: you pay ~6% more bytes, and they go to the tensors that matter most. A few problems on a single seed can be noise, so treat the HumanEval gap as "at least on par". The perplexity gap is larger than either run's standard error.

**What the SWE probe did:** it resolved `django__django-11133` (12 tool calls), `django__django-11099` (5 tool calls), `django__django-13933` (35 tool calls) and `django__django-14608` (37 tool calls). On `django__django-13590` it submitted a patch after 11 tool calls, but the patch broke 5 existing lookup tests, so it counts as unresolved. There were no crashes and no empty replies. The short, direct runs match what the adapter was trained for. Five instances is a smoke test for agentic competence, not a leaderboard score.

**The tensor mix.** llama.cpp's stock IQ3_XXS recipe for this architecture puts the full-attention Q/K projections at IQ2_S and the small gated-delta-net gates (`ssm_alpha`/`ssm_beta`) at 3 bits. That's what the reference file uses. We spent the extra budget only on small, sensitive tensors: `ssm_alpha`/`ssm_beta` → Q8_0, `ssm_out` → Q4_K, `attn_output` → Q5_K, `attn_v` → Q6_K, `attn_k` → Q5_K, `attn_q` → IQ3_XXS, `output` → Q6_K. The bulk FFN weights stay at IQ3_XXS, the same as the reference.

**Honest caveats**
- IQ3_XXS is a ~3.5-bit quant of a dense 27B model. It's a laptop-sized version, not equivalent to the BF16 model. If you have the memory, a Q4_K_M or larger will be stronger.
- The adapter was trained and evaluated with **thinking off**, so that's how we ran the gate.
- harness-hub's results (SWE-bench Pro hard-51, Terminal-Bench 2.1, under the pi harness) describe the full-precision adapter. We don't claim those numbers for this quant.

## Quickstart

**Ollama** (uses the chat template embedded in the GGUF, including tool calls):

```bash
ollama run hf.co/KikoCis/Pithos-27B-Pandora-IQ3_XXS-GGUF
```

**llama.cpp**

```bash
llama-server -hf KikoCis/Pithos-27B-Pandora-IQ3_XXS-GGUF --jinja -c 32768 \
  --chat-template-kwargs '{"enable_thinking":false}' --temp 0.8 --top-p 1.0
```

Recommended (from the adapter card): **thinking off**, `temperature 0.8, top_p 1.0` (the training sampling). For Ollama, pass `think: false`. Then point pi, or any OpenAI-compatible coding agent, at the local endpoint.

## How it was made

1. Merged the LoRA (r = 32, alpha = 64, scale 2.0; q/k/v/o projections of the 16 full-attention layers plus the MLPs of all 64 layers) into the Qwen3.8-27B BF16 weights, computing `W + scale · B·A` in FP32 and storing BF16. That's the same result as PEFT `merge_and_unload`.
2. Converted to GGUF with `llama.cpp` (text model only, MTP excluded; architecture and tokenizer unchanged; the adapter's chat template is identical to the base one).
3. Computed an importance matrix from a Q8_0 copy (174 × 2048-token chunks of a code-focused calibration set).
4. Quantized from BF16 to `IQ3_XXS` with that imatrix, plus the per-tensor overrides listed above.
5. Gated: it loads and generates coherently, resolves at least 1 of 5 SWE-bench instances with no crash or loop, then HumanEval and perplexity vs the nearest-size reference quant.

The imatrix (`pithos-27b.imatrix`) is included so you can make other sizes yourself.

## Credits

All capability comes from **harness-hub**'s [Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) adapter and **Qwen**'s [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (both Apache-2.0). Quantization uses [llama.cpp](https://github.com/ggml-org/llama.cpp). The reference quant is from [netprojects](https://huggingface.co/netprojects/Qwen3.8-27B-pi-adapted-GGUF).
