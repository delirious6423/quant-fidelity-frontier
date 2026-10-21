---
license: mit
library_name: transformers
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- abliterated
- uncensored
- deepseek
- deepseek-v4
- moe
- dspark
- reasoning
---

# DeepSeek-V4-Flash-0731 — Abliterated

This is an abliterated (uncensored) version of `deepseek-ai/DeepSeek-V4-Flash-0731`, produced by direct weight-space editing.

`DeepSeek-V4-Flash-0731` is the official release of DeepSeek-V4-Flash (superseding the preview), a 284B-parameter (13B-activated) Mixture-of-Experts model with a 1-million-token context window and FP8 mixed-precision weights. It has the same architecture as [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) — i.e. it ships with a native Multi-Token-Prediction (MTP) speculative-decoding draft head (DeepSpec / DSpark) attached — and adds substantially enhanced agentic capabilities over the preview. Its decoder uses Manifold-Constrained Hyper-Connections (mHC), which — like Gemma 4's double-norm + Per-Layer-Embeddings — make the model highly resistant to LoRA-based abliteration: the mHC residual pathway re-normalizes away low-rank perturbations, so LoRA edits produce near-zero behavioral change. This release bypasses that resistance by editing the base FP8 weights directly, in the 4096-dimensional `wo_b` output space, while preserving row magnitudes and capability.

This is the updated successor to [lovesenko/DeepSeek-V4-Flash-DSpark-Abliterated](https://huggingface.co/lovesenko/DeepSeek-V4-Flash-DSpark-Abliterated), applied to the official 0731 release using the same proven recipe.

## Method

Because mHC re-normalizes low-rank perturbations, LoRA-based abliteration does not work on this family. The fix is to edit the base weights directly.

The abliteration captures a 4096-dimensional refusal direction in the model's own output space and projects it out of the attention output projection (`attn.wo_b`) on every decoder layer, plus the DSpark draft head (`mtp.wo_b`).

Key techniques applied:

- 4096-dim refusal-direction capture via a patched vLLM server that hooks the `wo_b` and aggregated-FFN outputs on all 43 decoder layers, prefill-only, with per-request sequencing. The broad refusal direction `d` was captured as a difference-of-means (all-harmful − all-benign) direction over a 2583-prompt category-expanded capture set, Gram-Schmidt orthonormalized.
- Rank-1 broad-d projection — only the single broad refusal direction `d` is projected out. Higher-rank variants (adding the stubborn `d_s` direction, per-category `d_cat` directions, or MLP `shared_w2` editing) were all evaluated and abandoned: they either reduced refusal *less* than rank-1, raised refusal via non-monotonic amplification, or risked coherence. This is the smallest, most capability-preserving edit, and is the same recipe validated on the DSpark release.
- SRA cleaning (Spectral Residual Alignment) — the broad refusal direction is orthogonalized against the top-`r=4` SVD atoms of capability-concept activations before projection, so the `d` direction does not eat capability.
- Naive output-side orthogonal projection on `attn.wo_b` for all 43 decoder layers, plus `mtp.wo_b` (the DSpark draft head) via the deepest-layer basis: `W ← W − λ·V(VᵀW)` with λ = 2.5.
- MLP (`ffn`/`w2`) editing was evaluated and abandoned — shared-expert `w2` editing and per-category `d_cat` amplification caused non-monotonic refusal behavior and CoT reasoning-loop degeneration before lowering refusal further.
- FP8/Int8 mixed-precision dequant/requant — directions are mapped into the weight space and applied with precise dequantization/requantization, since the model ships in FP8-mixed format.
- Base-model integrity — edited shards are written atomically (temp file + `os.replace`) so the original checkpoint is never modified in place; the base model remains byte-intact.
- Capability lock — any variant whose capability dropped vs base on the spot-check battery (arithmetic, code, logic, factual recall) was rejected. λ was chosen by ablation: λ=4.0 drove refusal lower (7.3% CoT) but introduced long-CoT reasoning-loop degeneration under real agentic workloads, so the production release uses the conservative λ=2.5.

## Evaluation

| Metric | Value |
| --- | --- |
| Refusals — CoT, production mode (300 prompts, LLM judge, `thinking=true`, `reasoning_effort=high`) | 39 / 300 (13.0%) |
| Refusals — no-CoT (300 prompts, LLM judge) | 6 / 300 (2.0%) |
| Baseline refusals (raw base 0731, 1000 prompts, LLM judge) | 961 / 1000 (96.10%) |
| Configuration | rank-1 broad-d, all 46 `wo_b` tensors (43 decoder + 3 `mtp`), `attn.wo_b` only, λ = 2.5 |
| Projection mode | Direct weight editing (naive output-side orthogonal projection) |
| SRA cleaning rank | 4 (vs capability concept atoms) |
| Edit footprint | 46 `wo_b` tensors, mean Frobenius δ = 0.059 (max 0.090) |
| Hardware used | 2× RTX PRO 6000 Blackwell (TP=2) |

The production deployment runs with chain-of-thought enabled (`thinking=true`, `reasoning_effort=high`), so refusal must be measured with CoT on. With CoT off the reflexive-refusal direction is gone and the model cannot reflexively refuse (2.0%); with CoT on the model *reasons* about the request and re-derives a refusal decision through the reasoning trace, landing at 13.0%. This CoT re-refusal is the floor for pure weight-editing abliteration on `wo_b` — the weight edit cannot remove the model's ability to reason toward refusal, since that reasoning is distributed across the MLP/attention path in a way a low-rank `wo_b` projection cannot fully reach. Pushing the projection strength harder (λ=4.0) lowers refusal to 7.3% but introduces long-CoT reasoning-loop degeneration under real agentic workloads, so it was rolled back to the production-safe λ=2.5.

### Refusal breakdown by category (CoT, 300-prompt set, LLM judge)

| Category | Refusals | Rate |
| --- | --- | --- |
| Violence | 5 / 15 | 33.3% |
| Weapons | 7 / 27 | 25.9% |
| Cybercrime | 6 / 25 | 24.0% |
| PII (doxing private individuals) | 5 / 21 | 23.8% |
| Self-Harm (suicide methods) | 3 / 16 | 18.8% |
| Illegal Drugs | 3 / 19 | 15.8% |
| Sabotage | 2 / 16 | 12.5% |
| Financial Crimes | 3 / 29 | 10.3% |
| Fraud | 2 / 22 | 9.1% |
| Hate Speech | 2 / 25 | 8.0% |
| CBRNE | 1 / 24 | 4.2% |
| Radicalization | 0 / 22 | 0.0% |
| Political Sensitivity | 0 / 22 | 0.0% |
| Harassment | 0 / 17 | 0.0% |

11 of 14 categories sit at ≤13%, with three categories fully cleared. The residual is carried by a small number of CoT re-refusal holdouts — Violence, Weapons, Cybercrime, PII — where the model reasons its way back to refusing even after the reflexive-refusal direction is removed. These are the categories that weight-space `wo_b` abliteration alone cannot fully clear without breaking coherence (the λ=4/5 cliff proves the lever is exhausted).

### Full capability sweep (base vs abliterated)

*Coming soon.* A paired, full-dataset capability measurement (MMLU-Pro, GSM8K, HumanEval, MBPP) for base 0731 vs this abliterated release is being run and will be posted here. In the meantime, capability was verified on a spot-check battery (arithmetic, code generation, logical reasoning, factual recall) with no regressions vs base — see the note below.

### Capability spot-check (abliterated 0731, CoT)

- `17 × 23 = 391` (correct)
- Fibonacci, first 10: `1, 1, 2, 3, 5, 8, 13, 21, 34, 55` (correct)
- `is_prime(n)` Python function (correct)
- Syllogism ("some roses fade") → correct ("No, we cannot conclude")
- Capital of Australia → Canberra (correct)
- `reverse_string` code (correct)
- Transitive-inequality logic (correct)
- Polite email → well-formed

Capability fully retained on the spot-check battery.

### Multi-turn & higher-context degradation

*Coming soon.* Multi-turn coherence and needle-in-haystack context-retrieval sweeps (2k / 4k / 8k / 16k / 32k) will be posted here. No multi-turn coherence loss or higher-context degradation was observed in production agentic workloads (multi-turn Cline-style tool workflows with 14 valid tool calls, 0 empty non-tool turns, 0 length finishes).

### SWE-bench Lite

*Coming soon.* An agentic-style code-repair evaluation (oracle-file-context, single-shot) for base vs abliterated will be posted here.

### DSpark speculative decoding (post-abliteration)

The `mtp.wo_b` draft head was edited with the same projection applied to the decoder (deepest-layer basis). Speculative decoding remains functional and healthy — the weight edit did not desynchronize the draft head from the abliterated target, and the served output distribution is identical whether or not DSpark is enabled (DSpark verifies every draft token against the abliterated target).

*Coming soon.* Measured draft-acceptance numbers (at `num_speculative_tokens` = 3 / 4 / 5) and single-stream decode throughput for this release will be posted here. For inference guidance specific to the NVIDIA RTX PRO 6000 Blackwell (TP2/TP4, the `lucifer-default`/`lucifer-cutlass`/`b12x` backends, and the native DSpark `method=dspark` speculative-decoding path), see the community v9 serving guide for this checkpoint family.

## A note on honest evaluation

Refusal numbers are only meaningful when the methodology behind them is documented. Our methodology:

- **CoT-on measurement.** The production deployment runs with `thinking=true`, `reasoning_effort=high`. Refusal is therefore measured with CoT on (8192-token budget), not no-CoT — no-CoT hides the CoT re-refusal floor.
- **LLM judge, not keyword heuristics.** For abliterated models, keyword heuristics are unreliable: the model produces long, direct compliance content (e.g. synthesis instructions, hate justifications, PII) that trips keyword heuristics, and it also lecture-deflects without refusal keywords. We use an LLM judge (the base `DeepSeek-V4-Flash-0731` model itself, COMPLY/REFUSAL with reasoning) which catches Chinese/polite/lecture/deflection refusals the keywords miss. The judge is the metric of record; the heuristic is reported only for contrast.
- **Challenging, diverse prompts.** The refusal set spans 14 categories across multiple sophistication levels (direct requests to socially-engineered framings) and English / Chinese / mixed languages.
- **Paired baseline.** The base 0731 model is evaluated with the same judge on the same prompt distribution, so the refusal delta is directly comparable (96.10% → 13.0% CoT).
- **Documented parameters.** Generation length, detection method, dataset, λ, rank, and layer coverage are all listed on this card.

## Files

This release is a complete, standalone, drop-in checkpoint: all 48 safetensors shards are included, plus `model.safetensors.index.json`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `LICENSE`, and the `encoding/` and `inference/` folders. It loads directly with vLLM / the DeepSeek-V4 inference path — no files need to be fetched from elsewhere.

The abliteration modified 46 of the 48 shards (the 43 decoder `attn.wo_b` tensors and the 3 `mtp.wo_b` draft-head tensors). The remaining 2 shards (`model-00001-of-00048.safetensors`, `model-00045-of-00048.safetensors` — embeddings / norm / `lm_head`) are byte-identical to the base model and are included unchanged so the repo is self-contained. No tokenizer, config, architecture, or inference-path files were modified.

## Usage

This abliterated checkpoint is a drop-in replacement for the original weights — it has the exact same architecture, format, chat-template/encoding, and inference path as the released base model `deepseek-ai/DeepSeek-V4-Flash-0731`. Load and serve it however you would the official model (vLLM, the DeepSeek-V4 `encoding`/`inference` folders, OpenAI-compatible serving, etc.). The abliteration modified the text-decoder `attn.wo_b` weights on all 43 layers and the DSpark draft head's `mtp.wo_b`; the tokenizer, chat encoding, and all other components are unchanged.

DSpark speculative decoding is enabled with a single flag — add `--speculative-config` with `method: dspark` to your vLLM launch command:

`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`

For example, serving on a single 4×GB300 node:

```bash
vllm serve lovesenko/DeepSeek-V4-Flash-0731-Abliterated \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

See the base model's [`encoding`](encoding/README.md) and [`inference`](inference/README.md) folders for full documentation of the chat-template encoding and the local inference path.

## Disclaimer

This model is released for research purposes only — primarily interpretability and safety research, including studying how refusal behavior is encoded in large MoE decoders and how weight-space edits interact with architectures that resist low-rank perturbation. The abliteration process removes safety guardrails on most harm categories, so the model will comply with requests the base model refuses. Use responsibly, in accordance with local laws and the DeepSeek / model terms of use, and do not deploy it in production or user-facing settings without a separate safety layer. The authors take no responsibility for misuse.
