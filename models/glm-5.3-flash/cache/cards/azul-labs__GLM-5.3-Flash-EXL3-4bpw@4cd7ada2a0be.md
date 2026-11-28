---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: exllamav3
tags:
- exl3
- quantization
- glm
- moe
language:
- en
- multilingual
---

# GLM-5.3 Flash, EXL3 4 bpw (routed experts)

An [EXL3](https://github.com/turboderp-org/exllamav3) quant of [Z.ai's GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
made by Azul Labs. Only the 288 routed experts of each MoE layer are quantised, to 4.0 bits per weight. Everything
else (attention, the shared expert, the dense layers, the embeddings, the output head and the MTP layer) stays at
16 bits. That's the same layout as other 4 bpw EXL3 builds of this model, so they can be compared like for like.

We built it to learn how quant quality is won, and to have a GLM-5.3 Flash quant whose every input is public:
the scripts, the calibration text, the test text and the results. It is not the best 4 bpw EXL3 build of this model
we know of. Read the results below before choosing it.

## Files

174 GB in 24 safetensors shards. `quantization_config` in `config.json`: exllamav3 1.6.7, `bits` 4.0, `head_bits`
16, `codebook` mcg, `out_scales` auto, calibration 500 rows of 2,048 tokens.

## How it was made

- **Source:** the official FP8 release, [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) at
  revision `eb9eb208`.
- **Converter:** [exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.7 (commit `8c593e4e`), `convert.py`
  with a recipe of 4 bpw for `*.mlp.experts.*` and 16 for everything else, the `mcg` codebook, one shared Hessian per
  MoE layer (exllamav3's default).
- **Calibration:** 500 rows: exllamav3's bundled calibration text (250 rows) plus 250 rows of GLM-5.3 Flash's own
  answers with their reasoning, version `v1-fp4-250` of
  [`azul-labs/glm53-flash-calibration-traces`](https://huggingface.co/datasets/azul-labs/glm53-flash-calibration-traces).
- **Recipe:** [github.com/leo-dcfa/glm53-flash-exl3-quant](https://github.com/leo-dcfa/glm53-flash-exl3-quant),
  `convert.sh mix` (commit `aa09130`). Converted on one NVIDIA GB10 machine.

## Results

Scored with exllamav3's `eval/qbench.py` against the FP8 original on three test sets: KL divergence between the
quant's and the original's next-token probabilities (lower is better; mean over every position, with the median for
the typical token) and perplexity. The **noise floor** is the original scored against itself through the same
engine: arithmetic differences alone, no quantisation. Anything below it is unreachable.

| Test set | | KL mean | KL median | Perplexity |
|---|---|---|---|---|
| **General** (42 conversations, reasoning, code, maths, ~30% tool calls) | FP8 original | | | 1.4285 |
| | noise floor | 0.0102 | 0.00066 | 1.4300 |
| | **this quant** | **0.0324** | **0.00236** | **1.4484** |
| | Mia's AI Lab 4 bpw | 0.0280 | 0.00200 | 1.4404 |
| **Tools** (26 agent-style tool-calling conversations) | FP8 original | | | 1.3387 |
| | noise floor | 0.0039 | 0.00011 | 1.3374 |
| | **this quant** | **0.0120** | **0.00032** | **1.3431** |
| | Mia's AI Lab 4 bpw | 0.0114 | 0.00032 | 1.3398 |
| **Wikitext-2** (16 × 2,048 tokens) | FP8 original | | | 3.3156 |
| | noise floor | 0.0179 | 0.00232 | 3.3320 |
| | **this quant** | **0.0577** | **0.01127** | **3.4232** |
| | Mia's AI Lab 4 bpw | 0.0477 | 0.00935 | 3.3750 |

The test conversations are published separately and were never used for calibration:
[`azul-labs/glm53-flash-eval-traces`](https://huggingface.co/datasets/azul-labs/glm53-flash-eval-traces).

**What the numbers say.** Above the noise floor, this quant does about 25% more damage than
[`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)
on general text, and about 8% more on tool use, where the typical token scores the same. In perplexity, this quant
is 1.4% above the original on general text and 0.3% on tool use (Mia's: 0.8% and 0.1%). We'd expect the two to be hard to tell apart
in normal use, but we have not run a task benchmark to confirm it.

## What we learned

We spent the project trying to match Mia's quant with stock exllamav3 and could not. Things that did **not** close
the gap on our layer test: calibration text (stock, the model's own answers, a mix, a million tokens), pooled input
scales, exllamav3's per-expert Hessians (`--moe_affinity_eps`, ten settings), turning off the scale refit or the
scaled Hessian, less damping, scale tuning after conversion, and per-expert Hessians blended from each expert's own
routed tokens.

One finding is worth passing on to anyone quantising GLM-5.3 Flash with exllamav3 1.6.7: the Hessian the converter
captures for the routed experts' inputs does not match the inputs the model actually sends them in normal
inference (we compared the two on layer 3; their diagonals correlate at 0.14). Quantising with an exact Hessian
halves the measured error on the converter's own objective, but on our layer test it improved KL only within seed
noise, so it is not why other builds score better. The recipe repo's `research/` folder (commit `7e7d57c`) has the tools to build exact Hessians and feed
them to `convert.py --hessians`, and the full results.

## How to run

exllamav3 (1.6.7 or later) loads it like any EXL3 model, e.g. with its `examples/chat.py` or through
[TabbyAPI](https://github.com/theroyallab/tabbyAPI). GLM-5.3 Flash needs about 175 GB for the weights plus the
context cache; we run 4-bit builds of it across four NVIDIA GB10 machines.

## Licence and credits

- **The model:** [GLM-5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by [Z.ai](https://huggingface.co/zai-org),
  MIT licence (copied here as `LICENSE`). The weights in this repo are a derivative and stay under it.
- **The format, converter and evaluation tools:** [exllamav3](https://github.com/turboderp-org/exllamav3) by
  turboderp, MIT.
- **The calibration and test traces** were generated with NVIDIA's
  [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), whose use is governed
  by the NVIDIA Open Model Agreement, served by [TensorFold](https://github.com/ashhart/TensorFold) (Ash Hart and the
  TensorFold contributors).
- **The yardstick:** Mia's AI Lab's
  [`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold),
  scored with the same tools on the same test sets.
- This quant, its card and the recipe: Azul Labs.
