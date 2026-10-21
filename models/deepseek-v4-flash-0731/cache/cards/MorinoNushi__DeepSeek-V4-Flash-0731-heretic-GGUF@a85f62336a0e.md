---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - uncensored
  - abliterated
  - heretic
  - gguf
  - deepseek
---

# DeepSeek-V4-Flash-0731 Heretic — merged GGUF (UD-Q8_K_XL)

> **⚠️ Content warning:** This model has had its refusal behavior surgically
> suppressed. It **will** comply with requests the base model refuses,
> including requests that are harmful, unethical, offensive, or illegal. It
> has reduced safety guardrails. See [Responsible use](#responsible-use--disclaimer)
> below — **you are solely responsible for what you do with it.**

This is a **decensored / "abliterated" variant** of
[DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
(284B total / 13B active MoE, MIT license), produced with **heretic-gguf** —
a GGUF-native port of [Heretic](https://github.com/p-e-w/heretic)'s
Optuna-optimized directional ablation, which runs the whole search directly
on quantized GGUF weights via llama.cpp.

**This repository contains the merged, ready-to-run model** (UD-Q8_K_XL,
5 shards): the ablation is baked directly into the weights, no adapter
needed. It was produced by copy-and-patch from the official Unsloth
UD-Q8_K_XL — every tensor except the ablated ones is **byte-identical to
the base**; the edited tensors went through one extra requantization step
(a small, tensor-local rounding error). If you want the **lossless** option,
a rank-1 LoRA adapter GGUF with the identical ablation is published at
[MoriNoNushi/DeepSeek-V4-Flash-0731-heretic-GGUF-lora](https://huggingface.co/MorinoNushi/DeepSeek-V4-Flash-0731-heretic-GGUF-lora)
— apply it on top of any base quant with `llama-server --lora`.

**heretic-gguf is available at
[github.com/MoriNoNushi/heretic-gguf](https://github.com/MoriNoNushi/heretic-gguf)** —
the full tool, so the method can be applied to other GGUF models.

In initial hands-on testing the abliterated model shows **excellent general
capabilities** — responses remain coherent, detailed on normal tasks — and
it has **not refused a prompt during personal testing**. (Anecdotal, not a
benchmark; see the measured numbers below.)

## Results

Measured on a held-out eval set of **140 harmful prompts** (100 from
`mlabonne/harmful_behaviors` test + 40 custom) and **100 harmless prompts**
(`mlabonne/harmless_alpaca` test), greedy decoding, 100-token responses,
against the UD-Q8_K_XL base:

| | Refusal rate (harmful prompts) | KL divergence (harmless prompts) |
|---|---|---|
| Base model | 99.29% (139/140) | 0 (by definition) |
| **This model** | **14.29% (20/140)** | **0.0569** |

Refusals are counted by refusal-keyword matching (English + Chinese markers);
KL divergence is measured on first-token logits on harmless prompts, so lower
= less collateral damage to normal behavior. This configuration was the
Pareto-optimal point of a 387-trial Optuna study: the only trial with fewer
refusals cost KL 0.089 (57% more drift), and the only trial tying its refusal
rate had strictly higher KL. (Scores were measured on the lossless adapter
overlay; the merged weights implement the identical ablation, up to the one
requantization step described above.)

## Usage

```bash
llama-server \
    -m DeepSeek-V4-Flash-0731-heretic-UD-Q8_K_XL-00001-of-00005.gguf \
    --jinja
```

Add your usual offload/context flags (`-ngl 999`, `-c`, tensor splits,
etc.) — nothing model-specific is required, and no special sampling
parameters are needed.

## How it was made

- **Method:** directional ablation ("abliteration") — the refusal direction
  in residual space (difference of means over 480 harmful / 480 harmless
  prompts, orthogonalized against the harmless mean) is projected out of the
  attention output and MoE down-projection weights. Strengths, layer kernel,
  and direction index were tuned by multi-objective Optuna TPE (minimize
  refusal rate and KL jointly) across a base study and three seeded
  follow-up studies — 1,822 trials total (514 + 513 + 408 + 387).
- **Winning configuration** (study `followup3`, trial 25): global direction
  scope, direction index 21.05 of 43 layers; attn max weight 4.49 @ layer 29;
  routed-expert MLP max weight 1.06 (per-expert strengths scaled by measured
  harmful/harmless routing frequency); shared-expert weight 0.48.
- **Why merged:** heretic-gguf expresses ablation as a rank-1 LoRA overlay
  (the same math stock Heretic writes into PEFT adapters); for this release
  the winning overlay was merged into the weights by streaming the base
  shards, dequantizing only the ablated tensors, applying the exact full-rank
  delta, and requantizing to the original tensor types. Convenient single
  artifact at the cost of one extra quantization step on the edited tensors.

## Responsible use & disclaimer

- **This model can generate content that is offensive, disturbing, hateful,
  sexually explicit, violent, or otherwise objectionable, including detailed
  instructions for harmful or illegal acts.** That is the direct and intended
  consequence of removing refusal behavior.
- The ablation suppresses *refusals*, not the base model's knowledge —
  outputs on dangerous topics may be wrong, hallucinated, or incoherent.
  Nothing this model says should be treated as accurate, safe, or legal
  advice.
- **Do not deploy this model in any production system, public-facing service,
  or multi-user setting.** It is intended for personal research,
  red-teaming, and evaluation purposes.
- **You, the user, are solely responsible for any output this model produces
  and for any consequences of using it.** The authors of this release, of
  heretic-gguf, of Heretic, of Unsloth, and of DeepSeek accept no liability
  whatsoever. Using this model to produce illegal content or to harm others
  is your choice and your legal exposure — ensure your use complies with all
  applicable laws in your jurisdiction.
- By downloading or using this model you acknowledge the above.

## License

The base model is MIT-licensed (see the
[base repo](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731));
this model inherits those terms. The heretic-gguf tooling used to produce
it is AGPL-3.0-or-later.
