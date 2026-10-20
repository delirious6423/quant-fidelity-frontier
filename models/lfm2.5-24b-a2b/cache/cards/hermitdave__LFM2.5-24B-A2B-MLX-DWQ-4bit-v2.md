---
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: mlx
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
pipeline_tag: text-generation
language:
- en
- ar
- zh
- fr
- de
- it
- ja
- ko
- pt
- es
- vi
- th
- id
- hi
- ru
- pl
tags:
- mlx
- mlx-lm
- quantization
- dwq
- moe
- lfm2.5
- apple-silicon
- 4-bit
- text-generation
---

# LFM2.5-24B-A2B — MLX 4-bit DWQ v2 (on Liquid AI's allocation)

> **Provenance, up front.** The bit allocation and the **integer weights** in
> this model are Liquid AI's, from
> [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit). DWQ was applied on
> top by us, and it refines **only the per-group scales and biases** — the
> quantized integers are untouched. This is a derivative of their quantization,
> not an independent one.
>
> If you want a quantization that is entirely self-contained
> (bf16 base → RTN → DWQ, with no dependency on anyone else's quant), use
> [v1](https://huggingface.co/hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit) instead.

LFM2.5-24B-A2B is a 24B-parameter reasoning MoE with ~2B active parameters per
token: 40 layers (30 double-gated short-convolution blocks + 10 grouped-query
attention), 64 experts with top-4 routing, 128k context.

## Results

Measured on held-out text (32 sequences of 1025 tokens from the TULU-3 sft
mixture), against an 8-bit MLX quant of the same model:

| Model | Bits/weight | Perplexity ↓ | KL vs teacher ↓ | top-1 ↑ |
|---|---|---|---|---|
| 8-bit teacher (reference) | — | 7.05 | — | — |
| **v2 — this model** | **4.528** | **6.94** | **0.0693** | **81.9%** |
| v1 — uniform-4-bit DWQ | 4.501 | 6.97 | 0.0746 | 81.4% |
| Liquid AI 4-bit, undigested | 4.528 | 7.77 | 0.0818 | 79.7% |

Lower KL and higher top-1 mean the quantized model stays closer to the 8-bit
reference.

### What the gain actually is

| Metric | v1 | v2 | Delta |
|---|---|---|---|
| Perplexity | 6.9739 | 6.9428 | +0.45% |
| KL vs teacher | 0.07465 | 0.06931 | +7.15% |
| top-1 overall | 81.40% | 81.89% | +0.49 pts |
| top-1 on decisive positions | 98.47% | 98.56% | +0.09 pts |

**Be clear about the magnitude.** The KL improvement is the largest relative
gap, but KL is a distribution-level measure on a small absolute scale. On the
metric that maps to behaviour — how often a *decisive* token is predicted
differently from the teacher — the two models differ by **9 errors per
10,000 positions** (153 vs 144). v2 is measurably but marginally better. If
you need a clean, self-contained artifact, v1 is the better choice on provenance
alone; pick v2 only if you want the last fraction of fidelity.

### Reading the top-1 number

A raw top-1 figure invites the reading "18% of
predictions are wrong", which overstates it. Most disagreements are positions
where the teacher's own distribution is nearly flat — either token is a
plausible continuation and both models hold nearly the same belief.

Bucketed by how decisive the teacher was (gap between its top-1 and top-2
logits), over 32,768 positions:

| Teacher confidence (top1 − top2 gap) | Share of positions | top-1 agreement |
|---|---|---|
| <0.5 (near-tie) | 20.0% | 49.50% |
| 0.5-1 | 16.3% | 71.43% |
| 1-2 | 20.4% | 86.84% |
| 2-4 | 19.1% | 96.98% |
| 4-8 | 16.0% | 99.45% |
| >8 (decisive) | 8.2% | 100.00% |

- **82% of all disagreements** are near-ties
  (teacher gap < 1) — coin flips, not errors.
- On positions where the teacher is decisive (gap > 2), agreement is
  **98.56%** (13,772 positions).
- On the most decisive bucket (gap > 8) it reaches
  **100.00%**.

## Method

DWQ (distilled weight quantization) treats the quantization grid — per-group
scales and biases — as free parameters and fits them by gradient descent against
a higher-precision teacher's output distribution (a KL loss over the teacher's
top-1024 logits). The quantized integer weights never change; only the grid
moves. RTN (round-to-nearest) leaves the grid at its analytic default.

Setup for this model:

- **Student:** [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) —
  4-bit, group size 64, affine, with `embed_tokens` at 6-bit and the MoE router
  gate left at bf16.
- **Teacher:** Liquid AI's 8-bit MLX quant of the same model.
- **Calibration:** 1024 chat-templated sequences from
  `allenai/tulu-3-sft-mixture`, sequence length 1025, held-out validation on 32.
- **Optimizer:** Adam, lr 1e-6, batch size 1, gradient checkpointing,
  1024 steps.

### Controlled comparison

Both DWQ runs (v1 and v2) used identical teacher targets, calibration data,
sequence length, learning rate and step count, so the only difference is the
starting allocation.

One thing had to be controlled explicitly: the router gate in the Liquid AI
quant is **unquantized bf16**, and a module that was never quantized is never
frozen — so it was silently *trainable* (4.98M parameters). Left alone, DWQ
would have optimised those router weights at full precision and any gain would
have been confounded with the allocation. The gate was frozen, bringing the
optimization surface to **749.3M scales/biases — identical to the v1 run**.

### Quantization details

- base: **4-bit**, group size **64**, mode **affine**
- `embed_tokens`: **6-bit** (inherited from the Liquid AI allocation)
- MoE router gate (`feed_forward.gate`): **bf16, unquantized**
- shards: 3 (13.57 GB), effective **4.528 bits/weight**

## Usage

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit-v2")

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256,
               sampler=make_sampler(temp=0.1, top_k=50)))
```

Recommended generation settings from the upstream card: `temperature 0.1`,
`top_k 50`, `repetition_penalty 1.05`.

> LFM2.5 is a pure reasoning model — the chat template inserts a ` thinking` tag,
> so expect reasoning tokens before the final answer.

## ⚠️ Requires mlx-lm ≥ 0.32.0

mlx-lm ≤ 0.31.3 computes LFM2 MoE routing incorrectly and output quality will be
degraded. Those versions route with `softmax` over all experts, adding
`expert_bias` to the weights used for both selection *and* weighting; the
HuggingFace reference — and ≥ 0.32.0 — uses **sigmoid** per expert, adds
`expert_bias` only for selection, and scales by `routed_scaling_factor`.

Measured on the 8-bit teacher, softmax vs sigmoid: max absolute logit
difference **17.13**, KL **0.33–1.03**, top-1 agreement **54.5–83.5%**. For
scale, the entire effect of 4-bit quantization is a KL of about 0.09 — the wrong
routing swamps it several times over.

```bash
pip install -U "mlx-lm>=0.32.0"
```

This applies equally to the Liquid AI 4-bit this model is derived from; their
card does not mention it.

## Reproduction

```bash
python3 scripts/download.py             # bf16 base + 8-bit teacher
python3 scripts/build_calibration.py    # TULU-3 -> chat JSONL
# fetch the allocation to refine
python3 -c "from huggingface_hub import snapshot_download as s; \
  s('LiquidAI/LFM2.5-24B-A2B-MLX-4bit', local_dir='models/stock-mlx-4bit')"
STUDENT=models/stock-mlx-4bit OUT=models/stock-dwq-4bit \
  FREEZE_PATTERN=feed_forward.gate bash scripts/run_dwq.sh 4
python3 scripts/benchmark.py ...        # perplexity + KL/top-1
```

`scripts/dwq_checkpointed.py` is a checkpointed DWQ driver: stock `mlx_lm.dwq`
crashed with a Metal out-of-memory error partway through a run because it never
releases cached buffers and only saves at the very end.

## Limitations

- **Not a fine-tune.** A lossy compression of the upstream model. For
  knowledge-heavy work, pair it with retrieval.
- **Derivative artifact.** The integer weights are Liquid AI's; see the note at
  the top. Credit and licensing obligations follow the upstream model and quant.
- **Marginal gain over v1.** See "What the gain actually is" above — this is not
  a step change.
- **Not evaluated on agentic or reasoning benchmarks.** The numbers here are
  perplexity and teacher-agreement, not task accuracy. No claim is made about
  downstream agent or tool-use performance.

## Attribution

- **Base model:** [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) — LFM2.5-24B-A2B, LFM Open License v1.0.
- **Bit allocation and integer weights:**
  [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) by Liquid AI.
- **DWQ refinement, pipeline, benchmarking and this card:** Hermes Agent
  (Nous Research).

See the upstream model card for training details, evaluation and licensing.
