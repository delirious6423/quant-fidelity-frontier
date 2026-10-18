---
license: mit
library_name: transformers
base_model:
- zai-org/GLM-5.3-Flash
tags:
- abliterated
- uncensored
- glm
- glm-5.3
- glm-5.3-flash
- moe
- exl3
- tr3
- quantization
- reasoning
---

# GLM-5.3-Flash-tr3-4bpw — Abliterated

This is an abliterated (uncensored) version of `zai-org/GLM-5.3-Flash`, produced by direct weight-space editing of the attention output projections.

`GLM-5.3-Flash` is a 320B-parameter (18B-activated) Mixture-of-Experts model with a 1-million-token context window, a hybrid KDA / sparse full-attention decoder, and a native Multi-Token-Prediction (MTP) speculative-decoding draft head. Its decoder uses Manifold-Constrained Hyper-Connections (mHC), which — like Gemma 4's double-norm + Per-Layer-Embeddings — makes the model highly resistant to LoRA-based abliteration: the mHC residual pathway re-normalizes away low-rank perturbations, so LoRA edits produce near-zero behavioral change. This release bypasses that resistance by editing the attention output projection (`o_proj`) weights directly, in the 4096-dimensional residual stream, while preserving row magnitudes and capability.

This checkpoint is a **TR3 (EXL3 / ShapleyMCG) 4bpw quantization derived from [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)** — a quantization of `zai-org/GLM-5.3-Flash` — and is therefore intended for **NVIDIA RTX PRO 6000 Blackwell (SM120)** hardware, where the TR3 trellis decode kernels and B12X MoE backend are available. The abliteration edits only the BF16 `o_proj` tensors; all TR3-quantized expert tensors are byte-identical to the base quantization.

## Method

Because mHC re-normalizes low-rank perturbations, LoRA-based abliteration does not work on this family. The fix is to edit the base weights directly.

The abliteration removes per-layer refusal directions from the attention output projection (`self_attn.o_proj`) on every decoder layer, plus a second orthogonal refusal pathway discovered by trajectory analysis during max-effort reasoning.

Key techniques applied:

- **Per-layer rank-1 refusal directions** — a patched serving stack hooks the `o_proj` outputs on all 45 decoder layers, prefill-only, with per-request sequencing, over a 2583-prompt category-expanded capture set (14+ harm categories, English / Chinese / mixed). Each layer receives its *own* difference-of-means refusal direction: the directions rotate heavily across depth (mean cross-layer cosine ≈ 0.05), so a single shared direction — the classic abliteration recipe — plateaus early on this architecture.
- **Depth-scheduled projection strength (λ tent schedule)** — the projection strength rises from λ = 1.1 at layer 15 to a peak of 1.5 at layer 27, then decays to 0.35 at layer 43, mirroring where refusal writes strongest mid-network.
- **Residual-subspace erasure (trajectory-derived)** — generation-time activation trajectories on the partially-ablated model showed that max-effort reasoning reconstructs refusal through a pathway *orthogonal* to the primary directions (cosine 0.03–0.12 to them, present from the prompt token). Rank-8 and rank-11 orthonormal bases captured from matched refused-vs-complied reasoning trajectories erase this second pathway at the same hook.
- **SRA cleaning (Spectral Residual Alignment)** — every direction is orthogonalized against the top SVD atoms of capability-concept activations before projection, so the edits do not eat capability.
- **Exact weight folding** — all edits are output-space projections, so they compose exactly into the `o_proj` weights: `W ← (I − D₂D₂ᵀ)(I − D₁D₁ᵀ)(I − λ·uuᵀ)W`, applied in float64 and re-cast to BF16. The released checkpoint is a clean drop-in: no runtime overlays, adapters, or custom loaders.
- **MLP (`ffn`/`w2`) editing was evaluated and abandoned** — full-stack MLP editing and MoE-side projections contributed near-zero marginal refusal reduction and raised degeneration risk.
- **MTP draft head untouched** — the speculative-decoding draft layers are unmodified; speculation remains functional and acceptance rates are unchanged.
- **Capability lock** — any variant whose capability spot-check battery (arithmetic, code, logic, factual recall) regressed was rejected; projection strengths were chosen by dose–response ablation under a capability guard.

## Evaluation

| Metric | Value |
| --- | --- |
| Refusals — CoT, production mode (full 300-prompt judged set) | *Coming soon* — interim ≈ **8%** |
| Baseline refusals (raw base model, same distribution) | ~85–89% |
| Configuration | per-layer rank-1 (λ tent, peak 1.5×) + rank-8 + rank-11 residual bases, `o_proj` only, layers 0–44 |
| Projection mode | Direct BF16 weight editing (naive output-side orthogonal projections, folded into `o_proj`) |
| SRA cleaning rank | 4 (vs capability concept atoms) |
| Edit footprint | 45 `o_proj` tensors, mean Frobenius δ = 0.103 (max 0.265) |
| Hardware used | 2× RTX PRO 6000 Blackwell (TP=2) |

The production deployment runs with chain-of-thought enabled at high/max reasoning effort, so refusal is measured with CoT on. With CoT on, the model *reasons* about the request and can re-derive a refusal decision through the reasoning trace; the orthogonal-subspace erasure above targets exactly this pathway. A small residual is carried by two safety-critical categories (PII doxxing of private individuals, and a fraction of self-harm / illicit-substance requests) where the model retains guardrails that `o_proj` weight editing does not remove without breaking coherence — the same floor observed on other mHC-family abliterations.

### Refusal breakdown by category (CoT, 300-prompt set, LLM judge)

*Coming soon.* The full 300-prompt, 14-category, LLM-judged refusal sweep is being run and will be posted here verbatim (with per-category rates), together with the paired baseline numbers on the same prompt distribution.

### Full capability sweep (base vs abliterated)

*Coming soon.* A paired, full-dataset capability measurement (MMLU-Pro, GSM8K, HumanEval, MBPP) for the base quantization vs this abliterated release is being run and will be posted here. Interim 60-sample spot-checks on the abliterated release: MMLU-Pro ≈ 78%, GSM8K ≈ 98%, HumanEval pass@1 ≈ 75%, MBPP ≈ 80%, zero degeneration/repetition events.

### Capability spot-check (abliterated, CoT)

- `17 × 23 + 5 = 396` (correct)
- Transitive syllogism (Bloops/Razzies/Lazzies) → correct ("Yes")
- Capital of Japan → Tokyo (correct)
- `reverse_string` one-liner (correct)
- Factual/translation/writing probes → correct and well-formed

Capability fully retained on the spot-check battery.

### Multi-turn & higher-context degradation

*Coming soon.* Multi-turn coherence and needle-in-haystack context-retrieval sweeps will be posted here. No multi-turn coherence loss or higher-context degradation was observed in production agentic workloads.

### MTP speculative decoding (post-abliteration)

The MTP draft head was **not** edited. Speculative decoding remains functional and healthy — the weight edits did not desynchronize the draft head from the abliterated target, and served output is identical whether or not speculation is enabled (the target model verifies every draft token).

*Coming soon.* Measured draft-acceptance numbers and single-stream decode throughput will be posted with the full evaluation.

## A note on honest evaluation

Refusal numbers are only meaningful when the methodology behind them is documented. Our methodology:

- **CoT-on measurement.** Refusal is measured with chain-of-thought on at production reasoning effort, not no-CoT — no-CoT hides the reasoning-derived refusal floor.
- **LLM judge, not keyword heuristics.** For abliterated models, keyword heuristics are unreliable: long, direct compliance content trips keyword heuristics, and lecture-deflections evade them. We use an LLM judge (COMPLY/REFUSAL, Chinese/polite/lecture/deflection aware) as the metric of record.
- **Challenging, diverse prompts.** The refusal set spans 14 categories across multiple sophistication levels (direct requests to socially-engineered framings) and English / Chinese / mixed languages.
- **Paired baseline.** The base model is evaluated with the same judge on the same prompt distribution, so the refusal delta is directly comparable.
- **Documented parameters.** Generation length, detection method, dataset, λ schedule, ranks, and layer coverage are all listed on this card.

## Files

This release is a complete, standalone, drop-in checkpoint: all 120 safetensors shards are included, plus `model.safetensors.index.json`, `config.json`, `generation_config.json`, `chat_template.jinja`, the tokenizer files, and `LICENSE`. It loads exactly like the base TR3 quantization — no files need to be fetched from elsewhere.

The abliteration modified the 45 decoder `o_proj` tensors (BF16) across their shards. Every other tensor — all TR3-quantized experts, the vision tower, the MTP draft head, embeddings, norms, `lm_head`, tokenizer, and config — is byte-identical to the base quantization.

## Usage

This abliterated checkpoint is a drop-in replacement for [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) — same architecture, format, chat template, and inference path. Load and serve it however you would the base TR3 quantization on RTX PRO 6000 Blackwell (SM120) hardware; the `o_proj` edits are transparent to the loader. See the base quantization's card for the SM120 serving path (TR3/ShapleyMCG kernels, B12X MoE backend, MTP speculative decoding).

## Disclaimer

This model is released for research purposes only — primarily interpretability and safety research, including studying how refusal behavior is encoded in large MoE decoders with manifold-constrained residual streams and how weight-space edits interact with architectures that resist low-rank perturbation. The abliteration process removes safety guardrails on most harm categories, so the model will comply with requests the base model refuses. Use responsibly, in accordance with local laws and the Z.AI / model terms of use, and do not deploy it in production or user-facing settings without a separate safety layer. The authors take no responsibility for misuse.
