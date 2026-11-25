---
license: apache-2.0
base_model:
- harness-hub/Qwen3.8-27B-opencode-adapted
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
- opencode
- qwen3.8
- codyssey
---

![Codyssey-27B](banner.png)

# Codyssey-27B · Ithaca · IQ3_XXS GGUF

**A coding agent that finds its way home to a passing test suite, in under 12 GB.**

Odysseus spent years at sea before he reached Ithaca. A coding agent's trip is shorter: read the repo, find the bug, edit, run the tests, stop when they pass. harness-hub trained a LoRA adapter on Qwen3.8-27B with reinforcement learning so it works as the model behind the [OpenCode](https://opencode.ai) coding agent. On their benchmarks, with **thinking switched off**, it matches the base model thinking at maximum effort while using a fraction of the turns and tokens. **Codyssey** is that adapter merged into the base model and quantized for the 8–12 GB band, where most people actually run local models. It's an importance-matrix-calibrated **IQ3_XXS** with a hand-tuned tensor mix. Before publishing, we checked that it still *acts* as an agent: it had to resolve real SWE-bench issues end to end through a local agent harness.

> The name and the banner are just packaging. The weights are [harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted) (Apache-2.0) merged into [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) at the adapter's own scale (the standard PEFT merge), then quantized. No extra training, no other changes. Every number below was measured by us on this file.

| File | Quant | Size |
|---|---|---|
| `Codyssey-27B-Ithaca-IQ3_XXS.gguf` | IQ3_XXS + upgraded attention/SSM tensors (imatrix) | **11.89 GB** |
| `codyssey-27b.imatrix` | importance matrix (174 × 2048 tokens) | 0.01 GB |

**Text-only.** Qwen3.8-27B is multimodal and ships an MTP head. This GGUF contains the language model only: no vision projector and no MTP layer. The adapter card notes that the vision tower isn't served under OpenCode either.

**Runtimes.** We ran this file in Ollama 0.40.13 (native tool calls work) and in llama.cpp master (`dbaf582af`).

## What we measured (real numbers, this file)

| Benchmark | Setting | **Codyssey IQ3_XXS (11.89 GB)** | netprojects IQ3_XXS (11.43 GB) |
|---|---|---|---|
| **SWE-bench Verified**, 5-instance probe | local agent harness, ~12K-char context budget, non-thinking, native tool calls, temp 0.8 / top_p 1.0 | **3/5 resolved** | not run |
| **HumanEval** pass@1 (base / plus) | non-thinking, temp 0.6, top_p 0.95, seed 1234, 2048 max new tokens | **0.982 / 0.921** | 0.945 / 0.896 |
| Runaway generations on HumanEval (hit the token cap) | same | 0/164 | 1/164 |
| Wikitext-2 perplexity (lower is better) | 40 × 2048-token chunks | **6.183 ± 0.073** | 6.433 ± 0.078 |

HumanEval and perplexity for both files were run on the same engine (llama.cpp master `dbaf582af`, `llama-server --jinja`, `enable_thinking=false`) with identical prompts, seeds and scorer (evalplus).

**Why netprojects and not unsloth?** Unsloth hasn't published a GGUF of this adapter. The natural comparison is netprojects' IQ3_XXS: same quant type, imatrix-calibrated, and the most-downloaded full-model GGUF of this adapter. It's 11.43 GB on disk, but 0.24 GB of that is the MTP draft layer, which isn't used for normal decoding. Its main-model weights come to 11.18 GB vs our 11.89 GB, so ours is about 6% larger. We spent that extra 0.7 GB on purpose (see the tensor mix below).

**How to read the comparison:** Codyssey solved 6 more HumanEval problems (161 vs 155) and 4 more HumanEval+ problems (151 vs 147), and its perplexity is 3.9% lower. Part of that comes from the extra 0.7 GB, so it isn't free: you pay ~6% more bytes, and they go to the tensors that matter most. A handful of problems on a single seed can be noise, so treat the HumanEval gap as "at least on par". The perplexity gap is larger than either run's standard error.

**What the SWE probe did:** it resolved `django__django-11133` (13 tool calls), `django__django-11099` (18 tool calls) and `django__django-13933` (40 tool calls). On `django__django-13590` it submitted a patch after 11 tool calls, but the patch broke 5 existing lookup tests, so it counts as unresolved. On `django__django-14608` it reached the harness's 80-message cap with an incomplete fix (4 formset tests still failing). There were no crashes and no empty replies. Five instances is a smoke test for agentic competence, not a leaderboard score.

**The tensor mix.** llama.cpp's stock IQ3_XXS recipe for this architecture puts the full-attention Q/K projections at IQ2_S and the small gated-delta-net gates (`ssm_alpha`/`ssm_beta`) at 3 bits. That's what the reference file uses. We spent the extra budget only on small, sensitive tensors: `ssm_alpha`/`ssm_beta` → Q8_0, `ssm_out` → Q4_K, `attn_output` → Q5_K, `attn_v` → Q6_K, `attn_k` → Q5_K, `attn_q` → IQ3_XXS, `output` → Q6_K. The bulk FFN weights stay at IQ3_XXS, the same as the reference.

**Honest caveats**
- IQ3_XXS is a ~3.5-bit quant of a dense 27B model. It's a laptop-sized version, not equivalent to the BF16 model. If you have the memory, a Q4_K_M or larger will be stronger.
- The adapter was trained and evaluated with **thinking off**, so that's how we ran the gate.
- harness-hub's results (SWE-bench Pro hard-51, Terminal-Bench 2.1, under the OpenCode harness) describe the full-precision adapter. We don't claim those numbers for this quant.

## Quickstart

**Ollama** (uses the chat template embedded in the GGUF, including tool calls):

```bash
ollama run hf.co/KikoCis/Codyssey-27B-Ithaca-IQ3_XXS-GGUF
```

**llama.cpp**

```bash
llama-server -hf KikoCis/Codyssey-27B-Ithaca-IQ3_XXS-GGUF --jinja -c 32768 \
  --chat-template-kwargs '{"enable_thinking":false}' --temp 0.8 --top-p 1.0
```

Recommended (from the adapter card): **thinking off**, `temperature 0.8, top_p 1.0` (the training sampling). For Ollama, pass `think: false`. Then point OpenCode, or any OpenAI-compatible coding agent, at the local endpoint.

## How it was made

1. Merged the LoRA (r = 32, alpha = 64, scale 2.0; q/k/v/o projections of the 16 full-attention layers plus the MLPs of all 64 layers) into the Qwen3.8-27B BF16 weights, computing `W + scale · B·A` in FP32 and storing BF16. That's the same result as PEFT `merge_and_unload`.
2. Converted to GGUF with `llama.cpp` (text model only, MTP excluded; architecture and tokenizer unchanged; the adapter's chat template is identical to the base one).
3. Computed an importance matrix from a Q8_0 copy (174 × 2048-token chunks of a code-focused calibration set).
4. Quantized from BF16 to `IQ3_XXS` with that imatrix, plus the per-tensor overrides listed above.
5. Gated: it loads and generates coherently, resolves at least 1 of 5 SWE-bench instances with no crash or loop, then HumanEval and perplexity vs the nearest-size reference quant.

The imatrix (`codyssey-27b.imatrix`) is included so you can make other sizes yourself.

## Credits

All capability comes from **harness-hub**'s [Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted) adapter and **Qwen**'s [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (both Apache-2.0). Quantization uses [llama.cpp](https://github.com/ggml-org/llama.cpp). The reference quant is from [netprojects](https://huggingface.co/netprojects/Qwen3.8-27B-opencode-adapted-GGUF).
