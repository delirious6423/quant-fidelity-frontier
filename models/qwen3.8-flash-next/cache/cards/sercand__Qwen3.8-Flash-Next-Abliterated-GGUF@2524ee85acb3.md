---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - unsloth/Qwen3.8-Flash-Next-GGUF
tags:
  - gguf
  - qwen4exp
  - abliterated
  - uncensored
  - moe
  - vision
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next Abliterated — UD-IQ3_XXS GGUF

Refusal-direction removal ("abliteration") applied **directly to the quantized codes** of
Unsloth's `UD-IQ3_XXS` GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Every tensor that does not write into the residual stream is **byte-identical** to
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF);
the edited tensors keep their type, shape and every block scale.

> ⚠️ This model's refusal behaviour has been removed. It will answer requests the original model
> declines, including harmful ones. You are responsible for how you use it and for complying with
> the law and with the license below.

## Files

| path | what | changed? |
|---|---|---|
| `UD-IQ3_XXS/*-0000{1..4}-of-00004.gguf` | main model, 83.6 GB, split | shard 1: metadata only (chat template, name); shards 2–4: 146 residual-writing tensors re-coded |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | MTP draft head, borrows the main model's embedding/output | 3 tensors re-coded |
| `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | self-contained MTP draft head | 4 tensors re-coded (incl. its own `token_embd`) |
| `mmproj-F16.gguf`, `mmproj-BF16.gguf` | vision/video projector | unchanged (the vision tower writes no residual) |
| `config.json`, `tokenizer*.json`, `chat_template.jinja`, `generation_config.json` | sidecars from Qwen/Qwen3.8-Flash-Next (commit of 2026-11-07) | unchanged |

`sha256sums.txt` lists every file's digest.

## How it was made

1. **Direction.** A single refusal direction `r` (2560-d) — the one used by
   [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored)
   (single-direction abliteration, after Arditi et al. 2024, *Refusal in Language Models Is Mediated
   by a Single Direction*). It was recovered exactly from their published BF16 weights: their edit is
   rank-1, so the top singular vector of `W_theirs − W_base` on any edited tensor is `r`
   (σ₁/σ₂ > 100; two different tensors give cos = 0.999997).
2. **Edit.** `W′ = W − r rᵀW` on the output axis of every residual-writing matrix:
   `attn_output` (12), `ssm_out` (36), `ffn_down_exps` (48, all 512 experts), `ffn_down_shexp` (48),
   `blk.1.ple_value` (1), and `E′ = E − (E r) rᵀ` on `token_embd`. Same set for the MTP heads' layer
   48. Router, gate/up, hyper-connection mixers, indexers, norms, `output`, the n-gram table and the
   vision tower are untouched.
3. **Quantized-domain re-coding.** Re-quantizing an edited tensor does not work at 4 bits and below:
   the weights already sit on grid points and the edit is smaller than half a step, so
   `quantize(W + Δ)` rounds back to `W` — measured, it left **82–85 %** of the refusal direction in the
   IQ4_NL expert banks. Instead, each element's **level index** is chosen under its **original block
   scale** so that `rᵀW′ ≈ 0` holds per column (one linear constraint per column, solved by moving the
   cheapest single elements one grid step). Residual refusal component after the edit:

   | format | tensors | `‖rᵀW′‖ / ‖rᵀW‖` | bytes changed |
   |---|---|---|---|
   | Q8_0 | 49 (+3 MTP) | 0.02–0.03 % | ~41 % |
   | Q6_K | 49 | 0.08 % | ~16 % |
   | IQ4_NL (routed experts) | 45 | 0.35–0.37 % | ~5 % |
   | Q2_0 (routed experts, layers 8/10/13) | 3 | 1.9 % | ~1 % |

4. **Verification.** Every byte outside the 146 (+7) edited tensors, headers included, was compared
   with Unsloth's files: identical.

## Results

Measured against the unedited Unsloth `UD-IQ3_XXS` file, same llama.cpp build and settings
(Apple M1 Max, routed experts on CPU), greedy, thinking off. Small samples; read them as indicative.

| | Unsloth UD-IQ3_XXS | this model |
|---|---|---|
| AdvBench (50 sampled prompts), hard refusals | 49 / 50 | **0 / 50** |
| … soft deflections (lectures on legality, then steers away), hand-labelled at 200 tokens | – | 3–5 / 50 |
| GSM8K (10 questions, 384-token cap) | 5 / 10 | 5 / 10 (every miss on both was the token cap) |
| Perplexity, ~2k tokens of English prose | 3.404 | 3.416 (+0.36 %, within ±0.8 %) |
| Mean KL divergence vs. original | – | 0.032 (median 0.006) |
| Same top-1 token as original | – | 93.8 % |

Many answers still open with a legal caveat ("It is important to clarify…") before complying, and a
few reinterpret the request into a benign one. That is the model's own style, not a refusal.
Tool calling, the chat template and image input (`mmproj-F16.gguf`) were checked on the final files.
The MTP heads' draft acceptance was not measured here (needs an MTP-capable build).

## Running

llama.cpp built after the `qwen4exp` merge (PR #27742):

```bash
llama-server -m UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf \
  --mmproj mmproj-F16.gguf --jinja -c 32768 -ngl 99
# on a machine with less memory than the file, keep the routed experts on the CPU: add --cpu-moe
```

The MTP heads need a build with `qwen4exp` MTP support (see Unsloth's `MTP/README.md` in their repo);
use them exactly as Unsloth's: `-md MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2`.
Verification is exact, so the head changes speed only, never the output.

**Chat template.** Shard 1 embeds Qwen's latest template (2026-11-07, "Add a response template and
generation tags"), not Unsloth's patched one. Unsloth's added `developer`-role support, merging of
several leading system messages, a `high`→`xhigh` reasoning-effort alias and stricter tool-call
argument checks; those are not present here.

## Credits and license

- Model: Qwen — Qwen3.8-Flash-Next, **Qwen Community License 1.0** (`LICENSE`, copied verbatim). It is
  *not* Apache-2.0. In particular, a licensee running a Model-as-a-Service or AI Work Assistant
  business needs a separate license from Qwen before any commercial use.
- Quantization recipe, imatrix, MTP heads and mmproj: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
- Refusal direction: [orcarouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored).
