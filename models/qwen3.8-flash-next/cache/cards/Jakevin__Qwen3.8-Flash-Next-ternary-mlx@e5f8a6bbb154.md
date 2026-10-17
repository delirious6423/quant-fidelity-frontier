---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- ternary
- quantized
- mixed-precision
- apple-silicon
---

# Jakevin/Qwen3.8-Flash-Next-ternary-mlx

Mixed-precision MLX checkpoint of Qwen/Qwen3.8-Flash-Next. The start point is the official FP8 weights. Ternary matrices run on [mlx-ternary](https://github.com/Jakevin/mlx-ternary) (Hugging Face: [Jakevin/mlx-ternary](https://huggingface.co/Jakevin/mlx-ternary)).

The checkpoint has 288 units: 48 layers, and in each layer the attention block, the shared expert, and four routed-expert bands. Their storage is

| Format | Units |
| --- | ---: |
| Ternary, about 1.85 bits, one scale per 64 weights | 117 |
| 4-bit affine, group 64 | 92 |
| 3-bit affine, group 64 | 62 |
| 2-bit affine, group 64 | 17 |

Held-out next-token KL against the FP8 reference is 0.167 nats/token. ISTA 2-bit on the same text is 0.217.

On a 216-question multiple-choice set this mix scores 149 and ISTA 2-bit scores 162. That is 13 fewer correct answers (about 6 points), well beyond the roughly half-point noise of one question, so this mix answers fewer questions correctly than ISTA 2-bit. Its advantage is a lower KL divergence, meaning its outputs stay closer to the original model.

Speed depends on the device. Measured tokens per second for the mlx-ternary v3 kernel, with SSD read time split from GPU compute, will be added later.

## Limitations

An all-ternary allocation of the same model (every one of the 288 units ternary) keeps about 57% of the official FP8 score on those 216 questions. This release is the mixed allocation above, not that all-ternary model.

The runtime reads every routed expert from the layer files. It does not ship the allocation, calibration, or KL programs that chose the bit widths. Text generation also needs the official FP8 checkpoint for the tensors that stayed there: norms, hyper-connections, PLE tables, and the routers (the routers are quantized to affine 8-bit at load).

Apple Silicon and MLX only. The qwen4_exp architecture currently comes from oMLX's mlx-vlm build.

## Layout

Each layer directory `L00` … `L47` holds `codes.bin`, `scales.bf16`, `bias.bf16`, and `index.json`. Ternary codes are five base-3 trits per byte. The loader unpacks them into mlx-ternary's four 2-bit codes per byte and calls the v3 kernel. Affine 2/3/4-bit codes stay in the MLX uint32 lane pack. `skeleton.safetensors` holds the affine 8-bit embedding and lm_head.

Weights are under the Qwen Community License 1.0. See `LICENSE` and `NOTICE`.

## Setup

Use the oMLX Python, which provides MLX, mlx-vlm, and the qwen4_exp patch. Point a new virtualenv at that stack, then install mlx-ternary from a checkout of https://github.com/Jakevin/mlx-ternary (the package is also named on https://huggingface.co/Jakevin/mlx-ternary).

```bash
PY=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11
"$PY" -m venv .venv
SITE=.venv/lib/python3.11/site-packages
printf '%s\n' /Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages > "$SITE/omlx-mlx.pth"
printf '%s\n' /Applications/oMLX.app/Contents/Resources > "$SITE/omlx-root.pth"
.venv/bin/pip install --no-deps -e /path/to/mlx-ternary
```

## Generate

`--fp8` is a local copy of [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8). The script wraps the prompt with the Qwen chat template and a generation prompt, then generates. The process keeps RSS plus Metal active memory under 20 GB and caps the MLX cache at 2 GB.

```bash
.venv/bin/python generate.py --fp8 /path/to/Qwen3.8-Flash-Next-FP8 --max-tokens 8
```

Optional check of one ternary expert against the v3 kernel, without loading the whole model:

```bash
.venv/bin/python generate.py --self-check
```
