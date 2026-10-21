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

# LFM2.5-24B-A2B — MLX 4-bit DWQ

An MLX **4-bit** quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) built with
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
| 4-bit RTN (baseline) | 7.99 | 0.0882 | 78.3% | — |
| **4-bit DWQ (this model)** | **6.97** | **0.0746** | **81.4%** | **13.5 GB** |

Lower KL and higher top-1 agreement mean the quantized model stays closer to the
8-bit reference. Perplexity is a teacher-free cross-check.

> **On perplexity below the teacher.** This model's perplexity (6.97) is marginally lower than the 8-bit teacher's (7.05). This is not "better than the teacher": DWQ optimises KL against the teacher's *distribution*, and fitting the scales/biases can smooth some of the 8-bit teacher's own quantization noise. The honest comparison is the KL and top-1 agreement above, where the teacher is the reference by definition.

DWQ over RTN at 4-bit (positive = better):

| Metric | RTN | DWQ | Improvement |
|---|---|---|---|
| KL vs teacher (lower better) | 0.0882 | 0.0746 | **+15.4%** |
| Perplexity (lower better) | 7.99 | 6.97 | **+12.7%** |
| top-1 agreement (higher better) | 78.3% | 81.4% | **+3.1 pts** |

### Reading the top-1 number

A raw top-1 figure invites the reading "19% of
predictions are wrong", which overstates it. Most disagreements are positions
where the teacher's own distribution is nearly flat, so either token is a
plausible continuation and the two models hold almost the same belief.

Bucketing every held-out position by how decisive the teacher was (gap between
its top-1 and top-2 logits), over 32,768 positions:

| Teacher confidence (top1 − top2 gap) | Share of positions | top-1 agreement |
|---|---|---|
| <0.5 (near-tie) | 20.0% | 48.7% |
| 0.5-1 | 16.3% | 70.5% |
| 1-2 | 20.4% | 86.1% |
| 2-4 | 19.1% | 96.9% |
| 4-8 | 16.0% | 99.4% |
| >8 (decisive) | 8.2% | 99.9% |

- **81% of all disagreements** are near-ties
  (teacher gap < 1) — coin flips, not errors.
- On positions where the teacher is decisive (gap > 2),
  agreement is **98.5%** (13,772 positions).

So treat 81.4% as a *lower bound* on fidelity. This is also
why top-1 and KL must be read together: KL understates damage (a confidently
wrong token barely moves it), while top-1 overstates it (it counts coin flips as
errors).

### A better variant exists

We also ran DWQ on **Liquid AI's** 4-bit allocation (6-bit `embed_tokens`, bf16 router) rather than on our uniform 4-bit baseline, and it scores slightly better.

**[hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit-v2](https://huggingface.co/hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit-v2)**

KL 0.0693 vs 0.0746 here, top-1
81.9% vs 81.4%. On the
metric that maps to behaviour — how often a *decisive* token is predicted
differently from the teacher — the two differ by about **9 errors per 10,000 positions** (153 vs 144).

The trade-off is provenance: v2's bit allocation and integer weights are Liquid AI's (DWQ refines only the scales and biases), whereas this model is self-contained from the bf16 base through RTN to DWQ, with no dependency on anyone else's quant.

## Compared with Liquid AI's own MLX 4-bit

Liquid AI also publish an MLX 4-bit of this model
([LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)). Measured on the same
held-out windows, at essentially the same size:

| | **This model** | [LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) |
|---|---|---|
| Effective bits/weight | **4.501** | 4.528 |
| Size | **13.49 GB** | 13.57 GB |
| Perplexity ↓ | **6.97** | 7.77 |
| KL vs 8-bit teacher ↓ | **0.0746** | 0.0818 |
| top-1 vs teacher ↑ | **81.4%** | 79.7% |

This model is better on all three quality metrics at a marginally **smaller**
bit budget (4.501 vs 4.528 bits/weight).

**The gain comes from DWQ, not from a cleverer base allocation.** A plain
round-to-nearest 4-bit of the same model scores *worse* than the vendor quant
(perplexity 7.99, KL 0.0882, top-1 78.3%) — their hand-tuned allocation beats
uniform 4-bit RTN. Distilling scales/biases on top of uniform 4-bit then passes
it.

The two models protect different layers:

| | Router gate (`feed_forward.gate`) | `embed_tokens` |
|---|---|---|
| This model | 8-bit | 4-bit |
| Vendor quant | bf16 (left unquantized) | 6-bit |

The vendor quant spends roughly 65 MB on 6-bit embeddings plus 7.5 MB on a
full-precision router; this model spends ~2.5 MB on an 8-bit router and leaves
embeddings at 4-bit.

Two caveats on this comparison. Both models were evaluated with the corrected
LFM2 routing (mlx-lm ≥ 0.32.0) — on an older mlx-lm both degrade, and the
vendor quant's card does not mention this requirement. And both are lossy
compressions of the same base model; neither is "the" 4-bit answer, only a
measured one on this corpus.

## Sibling quants

| Bits | Perplexity ↓ | KL vs teacher ↓ | top-1 ↑ | |
|---|---|---|---|---|
| `4-bit` | 6.97 | 0.0746 | 81.4% | ← this model
| `3-bit` | 8.12 | 0.1458 | 70.7% |
| `2-bit` *(not published — see Limitations)* | 120.80 | 0.5653 | 32.0% |

## Method

DWQ refines the quantization **grid** — the per-group scales and biases — by
distilling a higher-precision teacher, rather than rounding weights to the
nearest grid point (RTN). The integer weights themselves are never modified;
only the scales/biases move. This typically recovers a meaningful fraction of a
bit of quality, and the effect is largest at low bit widths.

The pipeline:

1. **RTN baseline** — `mlx_lm convert -q --q-bits 4 --q-group-size 64` from the bf16 base.
2. **Teacher targets** — Liquid AI's 8-bit MLX release is run over 1024
   calibration sequences; the top-1024 logits per position are written to disk.
3. **DWQ** — only the student is loaded. Adam (lr 1e-6) optimises the scales and
   biases against the stored teacher logits with a KL loss, batch size 1, with
   gradient checkpointing. 1024 samples at sequence length 1025.

Calibration data: `allenai/tulu-3-sft-mixture`, chat-templated with the model's
own template, 1024 train / 32 held-out sequences.

### Quantization details

- base: **4-bit**, group size **64**, mode **affine**
- `feed_forward.gate` overridden to **8-bit** (38 layers)
- effective **4.501 bits/weight**, 3 shard(s), 13.49 GB

## Usage

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit")

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256,
               sampler=make_sampler(temp=0.1, top_k=50)))
```

```bash
# CLI
mlx_lm.generate --model hermitdave/LFM2.5-24B-A2B-MLX-DWQ-4bit --prompt "Explain Rayleigh scattering."
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
bash   scripts/run_dwq.sh 4            # DWQ (checkpointed, resumable)
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
- **4-bit is not free.** See the Results table for the measured gap to the
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
