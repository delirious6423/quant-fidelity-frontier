---
library_name: mlx
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- mlx
- dwq
- qwen3.8
- multilingual
- tool-use
- mtp
- speculative-decoding
---

# Qwen3.8-27B MLX 4-bit DWQ

This is a 4-bit MLX derivative of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), calibrated for
tool calling, agentic software-engineering traces, multilingual chat, and
Python code.

The **language tower** was trained with Distilled Weight Quantization (DWQ)
against the pinned 8-bit MLX teacher. The **vision tower was not DWQ-trained**:
its 333 tensors are copied exactly from the pinned `mlx-community` 4-bit RTN
conversion. For Qwen3.8's MTP path, use the separately loadable official MLX
drafter
[`mlx-community/Qwen3.8-27B-MTP-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit),
pinned here at revision `b643c01b6d3b094e325edb6ebd832e16c486c575`.

## What changed

| Component | Treatment |
| --- | --- |
| Language tower | 4-bit affine, group size 64; DWQ-trained scales and biases |
| Vision tower | Original 4-bit affine RTN tensors, byte-for-byte values unchanged |
| MTP | External official 4-bit MLX drafter, loaded separately by MLX-VLM |

The final language checkpoint contains 1,847 tensors; the complete VLM contains
2,180 tensors (1,847 language + 333 vision).

## Results

The primary metric is KL divergence to sparse top-1,024 logits from the pinned
8-bit teacher on a fixed, disjoint 64-example holdout. Lower is better.

| Checkpoint | Held-out KL |
| --- | ---: |
| Original 4-bit RTN baseline | 0.204094 |
| 200 examples | 0.098783 |
| 400 examples | 0.084881 |
| 600 examples | 0.080500 |
| 800 examples | 0.080442 |
| 1,000 examples | 0.075115 |
| Final, 1,024 examples | **0.074299** |

The final KL is **63.596% lower** than the baseline. The predeclared acceptance
threshold was 0.203890.

A deterministic 11-prompt smoke suite produced the following results with
thinking disabled:

| Model | Tool calling | Multilingual | Code | Total |
| --- | ---: | ---: | ---: | ---: |
| Original 4-bit RTN | 2/2 | 7/7 | 2/2 | 11/11 |
| This DWQ model | 2/2 | 7/7 | 2/2 | 11/11 |

On an Apple M2 Max with 64 GB unified memory, the DWQ model averaged 67.68
prompt tok/s and 23.19 generation tok/s with a 16.05 GB peak in this small
suite. The RTN baseline measured 67.96 and 23.16 tok/s with the same peak.
This is a compatibility smoke test, not a broad capability benchmark.

The pinned official MTP drafter also passed a three-prompt, 256-token greedy
smoke against this DWQ target with 3/3 outputs exactly matching autoregressive
decoding. It accepted 83.8% of drafted tokens and improved mean generation
throughput by 16.1% in that small test; peak MLX memory changed from 18.94 to
19.72 GB. Speedup is prompt-dependent, and one prompt was slightly slower with
MTP.

## Calibration mix

The 1,024 training and 64 validation examples are source-disjoint. Rendered
length is capped at 513 tokens.

| Slice | Train | Valid | Source | License |
| --- | ---: | ---: | --- | --- |
| Function/tool calling | 320 | 20 | `NousResearch/hermes-function-calling-v1` | Apache-2.0 |
| Successful SWE-agent traces | 320 | 20 | `nebius/SWE-agent-trajectories` | CC-BY-4.0 |
| Multilingual chat | 256 | 16 | `CohereLabs/aya_dataset` | Apache-2.0 |
| Python code | 128 | 8 | `openai/openai_humaneval` | MIT |

The multilingual slice covers Russian, Chinese, Ukrainian, Polish, Standard
Arabic, Spanish, and Japanese, with double weight on Russian. Only aggregate
provenance is published; source rows and teacher logits are not redistributed.
See `CALIBRATION.md` and `calibration-manifest.json`.

## Usage

Text-only generation with MLX-LM:

```bash
pip install -U mlx-lm
mlx_lm.generate \
  --model WaveCut/Qwen3.8-27B-MLX-4bit-DWQ \
  --prompt "Как называется столица Польши? Ответь одним словом." \
  --max-tokens 32 \
  --temp 0 \
  --chat-template-config '{"enable_thinking": false}'
```

Image-text generation with MLX-VLM:

```bash
pip install -U mlx-vlm
mlx_vlm.generate \
  --model WaveCut/Qwen3.8-27B-MLX-4bit-DWQ \
  --image /path/to/image.png \
  --prompt "Describe this image." \
  --max-tokens 128 \
  --temperature 0
```

Speculative decoding with the official MTP drafter:

```bash
pip install -U "mlx-vlm>=0.6.8"
mlx_vlm.generate \
  --model WaveCut/Qwen3.8-27B-MLX-4bit-DWQ \
  --draft-model mlx-community/Qwen3.8-27B-MTP-4bit \
  --prompt "Write a robust retrying HTTP client in Python." \
  --max-tokens 256 \
  --temperature 0
```

`--draft-kind mtp` is detected automatically. The drafter is separate because
MLX-VLM binds it to the target model's embeddings and language-model head at
runtime. This DWQ release does not redistribute or claim authorship of the MTP
weights.

The release was built and clean-loaded with `mlx-lm==0.31.3`, `mlx==0.32.0`,
`mlx-vlm==0.6.8`, Python 3.12.9, and `uv==0.12.3`.

## Reproducibility

The exact revisions, hyperparameters, corpus hashes, validation curve, local
MLX-LM compatibility patch, and environment lock are included in `RECIPE.md`,
`calibration-manifest.json`, `evaluation-summary.json`,
`mlx-lm-0.31.3-local-dwq-data.patch`, and `uv.lock`.

## Limitations

- DWQ optimization covered the language tower only; vision quality is that of
  the upstream 4-bit RTN conversion and was only clean-load-smoked here.
- The calibration corpus and behavioral smoke suite are small relative to the
  model's full capability surface. No claim is made for broad benchmark gains.
- Tool-call behavior depends on the supplied tool schema and Qwen chat
  template. Validate formats required by your runtime.
- The external MTP tensors come from the original Qwen3.8 checkpoint and were
  not DWQ-retuned. Greedy compatibility with the pinned official MLX drafter is
  verified, but acceptance and speedup are workload-dependent.
- Quantization can change outputs. Evaluate safety, multilingual quality,
  long-context behavior, and task-specific reliability before deployment.

The original model and this derivative are released under Apache-2.0. See
`LICENSE` and `THIRD_PARTY_NOTICES.md`.
