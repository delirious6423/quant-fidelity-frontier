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
- 3-bit
- text-generation
---

# LFM2.5-24B-A2B — MLX 3-bit DWQ

An MLX **3-bit** quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) built with
**Distilled Weight Quantization (DWQ)**, distilled against Liquid AI's own
8-bit MLX release.

LFM2.5-24B-A2B is a 24B-parameter reasoning MoE with ~2B active parameters per
token: 40 layers (30 double-gated short-convolution blocks + 10 grouped-query
attention), 64 experts with top-4 routing, 128k context, 128k vocabulary.

## Results

Measured on held-out text (32 sequences of 1025 tokens from the TULU-3 sft
mixture, held out from calibration), against the 8-bit teacher:

| Model | Perplexity ↓ | KL vs teacher ↓ | top-1 vs teacher ↑ | Size |
|---|---|---|---|---|
| 8-bit teacher (reference) | 7.05 | — | — | 24.6 GB |
| 3-bit RTN (baseline) | 11.40 | 0.1943 | 64.4% | — |
| **3-bit DWQ (this model)** | **7.78** | **0.1343** | **72.4%** | **10.6 GB** |

Lower KL and higher top-1 agreement mean the quantized model stays closer to the
8-bit reference. Perplexity is a teacher-free cross-check.

DWQ over RTN at 3-bit (positive = better):

| Metric | RTN | DWQ | Improvement |
|---|---|---|---|
| KL vs teacher (lower better) | 0.1943 | 0.1343 | **+30.9%** |
| Perplexity (lower better) | 11.40 | 7.78 | **+31.7%** |
| top-1 agreement (higher better) | 64.4% | 72.4% | **+8.0 pts** |

### Reading the top-1 number

A raw top-1 figure invites the reading "28% of
predictions are wrong", which overstates it. Most disagreements are positions
where the teacher's own distribution is nearly flat, so either token is a
plausible continuation and the two models hold almost the same belief.

Bucketing every held-out position by how decisive the teacher was (gap between
its top-1 and top-2 logits), over 32,768 positions:

| Teacher confidence (top1 − top2 gap) | Share of positions | top-1 agreement |
|---|---|---|
| <0.5 (near-tie) | 20.0% | 40.3% |
| 0.5-1 | 16.3% | 56.0% |
| 1-2 | 20.4% | 71.9% |
| 2-4 | 19.1% | 89.5% |
| 4-8 | 16.0% | 96.1% |
| >8 (decisive) | 8.2% | 98.8% |

- **69% of all disagreements** are near-ties
  (teacher gap < 1) — coin flips, not errors.
- On positions where the teacher is decisive (gap > 2),
  agreement is **93.9%** (13,772 positions).

So treat 72.4% as a *lower bound* on fidelity. This is also
why top-1 and KL must be read together: KL understates damage (a confidently
wrong token barely moves it), while top-1 overstates it (it counts coin flips as
errors).

## Sibling quants

| Bits | Perplexity ↓ | KL vs teacher ↓ | top-1 ↑ | |
|---|---|---|---|---|
| `4-bit` | 6.97 | 0.0746 | 81.4% |
| `3-bit` | 8.12 | 0.1458 | 70.7% | ← this model
| `2-bit` *(not published — see Limitations)* | 120.80 | 0.5653 | 32.0% |

## Method

DWQ refines the quantization **grid** — the per-group scales and biases — by
distilling a higher-precision teacher, rather than rounding weights to the
nearest grid point (RTN). The integer weights themselves are never modified;
only the scales/biases move. This typically recovers a meaningful fraction of a
bit of quality, and the effect is largest at low bit widths.

The pipeline:

1. **3-bit base, Liquid AI recipe** — `mlx_lm convert -q --q-bits 3 --q-group-size 64` with a custom `quant_predicate`: `feed_forward.gate` left **unquantized** (bf16) and `embed_tokens` raised to **6-bit**. This mirrors the allocation in Liquid AI's own published MLX quants, which leave the router at full precision and give the embedding table a 6-bit floor.
2. **Teacher targets** — Liquid AI's 8-bit MLX release is run over 1024
   calibration sequences; the top-1024 logits per position are written to disk.
3. **DWQ** — only the student is loaded. Adam (lr 1e-6) optimises the scales and
   biases against the stored teacher logits with a KL loss, batch size 1, with
   gradient checkpointing. 1024 samples at sequence length 1025.

Calibration data: `allenai/tulu-3-sft-mixture`, chat-templated with the model's
own template, 1024 train / 32 held-out sequences.

### Quantization details

- base: **3-bit**, group size **64**, mode **affine**
- `embed_tokens` overridden to **6-bit** (1 layer)
- the MoE router (`feed_forward.gate`) is left **unquantized** (bf16) — low-bit routers silently degrade expert selection
- effective **3.536 bits/weight**, 2 shard(s), 10.59 GB

## Usage

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("hermitdave/LFM2.5-24B-A2B-MLX-DWQ-3bit-v2")

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256,
               sampler=make_sampler(temp=0.1, top_k=50)))
```

```bash
# CLI
mlx_lm.generate --model hermitdave/LFM2.5-24B-A2B-MLX-DWQ-3bit-v2 --prompt "Explain Rayleigh scattering."
```

Recommended generation settings from the upstream model card:
`temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`.

> LFM2.5 is a pure reasoning model — the chat template inserts a ` thinking` tag
> so it always reasons before answering. Expect reasoning tokens before the
> final answer.

## ⚠️ Requires mlx-lm ≥ 0.32.0

**Load this model with mlx-lm 0.32.0 or newer.** Earlier versions compute LFM2
MoE routing incorrectly and output quality will be degraded.

mlx-lm ≤ 0.31.3 routes with `softmax` over all experts, adding `expert_bias` to
the weights used for *both* top-k selection *and* weighting. The HuggingFace
reference — and mlx-lm ≥ 0.32.0 — uses **sigmoid** per expert, adds
`expert_bias` **only for selection**, gathers weights from the *unbiased*
scores, and scales by `routed_scaling_factor`.

Measured on the 8-bit teacher, softmax vs sigmoid routing:

| Metric | Value |
|---|---|
| max abs logit difference | 17.13 |
| KL(sigmoid ‖ softmax) | 0.33 – 1.03 |
| top-1 agreement | 54.5% – 83.5% |

For scale, the entire effect of 4-bit quantization is a KL of about 0.09 — the
wrong routing swamps it several times over. Fixed upstream in
[PR #1354](https://github.com/ml-explore/mlx-lm/pull/1354), first shipped in
**v0.32.0**.

```bash
pip install -U "mlx-lm>=0.32.0"
```

### Also: DWQ cannot run at all on ≤ 0.31.3

`mlx_lm.dwq` fails with

```
ValueError: [gather_axis] Cannot calculate VJP with respect to indices.
```

on those versions — the same function omitted `mx.stop_gradient` on the top-k
router indices, so any `mx.value_and_grad` through the model fails. Also fixed
in v0.32.0 ([PR #1795](https://github.com/ml-explore/mlx-lm/pull/1795)).

## Reproduction

```bash
python3 scripts/download.py            # bf16 base + 8-bit teacher
python3 scripts/build_calibration.py   # TULU-3 -> chat JSONL
bash   scripts/convert_rtn.sh          # RTN baselines
bash   scripts/run_dwq.sh 3            # DWQ (checkpointed, resumable)
python3 scripts/benchmark.py ...       # perplexity + KL/top-1
```

`scripts/dwq_checkpointed.py` is a checkpointed DWQ driver. Stock `mlx_lm.dwq`
crashed with a Metal out-of-memory error partway through a run because it never
releases cached buffers and only saves at the very end; the driver clears the
cache every step and writes a rotating checkpoint every step.

## Limitations

- **Not a fine-tune.** This is a lossy compression of the upstream model. For
  knowledge-heavy work, pair it with retrieval — the upstream card makes the
  same recommendation.
- **3-bit is not free.** See the Results table for the measured gap to the
  8-bit teacher.
- **2-bit is not recommended** for this model. At 2-bit, RTN collapses
  (perplexity ≈ 4782, top-1 agreement 5.3%) and even with DWQ the model is
  degenerate — it emits repeated ` thinking` tokens instead of answering. 3-bit
  is the practical floor.
- **Not evaluated on agentic benchmarks.** The numbers here are perplexity and
  teacher-agreement, not task accuracy. No claim is made about downstream agent
  or tool-use performance.

## Attribution

- **Base model:** [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) — LFM2.5-24B-A2B, LFM Open License v1.0.
- **8-bit teacher:** [LiquidAI/LFM2.5-24B-A2B-MLX-8bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-8bit).
- **Quantization, DWQ pipeline, calibration, benchmarking and this card:**
  Hermes Agent (Nous Research).

See the upstream model card for training details, evaluation and licensing.
