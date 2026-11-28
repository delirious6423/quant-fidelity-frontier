---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
repo_description: "Compressed GGUF Q4_K_M / Q8_0 / Q2_K + mmproj weights for Qwen/Qwen3.8-27B (24.4B parameters). Fits 24 GB GPU; context length 262144."
tags:
  - gguf
  - dense
  - compressed
  - qwen3_5
  - multimodal
  - llama-cpp
language:
  - en
---

# Qwen3.8-27B-Turbo — compressed

A smaller build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with measured and disclosed quality costs.
Base revision: `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.

## How this build was derived

This build was produced by **structurally pruning feed-forward layers of** the base dense model and then quantising the result.
The Hub's `base_model_relation` field accepts only `adapter`, `finetune`, `merge`, `quantized`; none of them means pruned, so it is set to `quantized` as the closest accepted value and the full derivation is stated here instead.

All numbers were measured on `Qwen3.8-27B-Turbo-Q4_K_M.gguf`, sha256 `519766ba408ef539e003e4db8d90983db13f31b08209115c37c07cb94e66ef63`.
Exact artifact size: 18028549632 bytes.

## What you get

Size on disk: 54.6577 GB → **18.0285 GB** (67.0% smaller).
Memory: ~58 GB → **~20 GB**.
Fits on: 2x48 GB GPU / 80 GB GPU → **24 GB GPU**.
Format: GGUF BF16 → GGUF Q4_K_M.

### Hardware & VRAM Guide
| Quant Level | Est. Disk Size | Target VRAM (Inference) | Recommended GPU Class |
|:---|---:|---:|:---|
| **GGUF Q4_K_M** (this build) | **~18.0285 GB** | **~20 GB** | **24 GB GPU** |
| Base Model (GGUF BF16) | ~54.6577 GB | ~58 GB | 2x48 GB GPU / 80 GB GPU |

## What it costs you

**Lower than base:**

| capability | base | this build | change | 95% CI |
|---|---:|---:|---:|---:|
| none | — | — | — | — |

**Unchanged within measurement error:**

| capability | base | this build | change | 95% CI |
|---|---:|---:|---:|---:|
| ARC-Challenge (MC) | 96.00% | 95.50% | -0.50 pp | [-2.70, 1.70] |
| GSM8K | 86.00% | 83.50% | -2.50 pp | [-7.40, 2.40] |
| MATH-500 | 53.00% | 49.50% | -3.50 pp | [-9.29, 2.29] |

**Higher than base on one run (not replicated):**

| capability | base | this build | change | 95% CI |
|---|---:|---:|---:|---:|
| none | — | — | — | — |

**Not measured / not published:**

| capability | why no number is published |
|---|---|
| none | — |

## Honest notes

- This is compression, not improvement; no capability is added.
- A change inside the confidence interval is unresolved, not proof of no change.

## Licence

Base model licence: apache-2.0.
- Include the base model Apache-2.0 LICENSE text with the published weights.
- Attribute upstream Qwen/Qwen3.8-27B and state that these weights are modified.

## Not included

No claim is made that lost capability can be recovered.


## Included files

| File | Role | Format | Exact size (bytes) | SHA-256 |
|:---|:---|:---|---:|:---|
| `Qwen3.8-27B-Turbo-Q4_K_M.gguf` | primary | GGUF Q4_K_M | 18,028,549,632 | `519766ba408ef539e003e4db8d90983db13f31b08209115c37c07cb94e66ef63` |
| `Qwen3.8-27B-Turbo-Q8_0.gguf` | quant_rung | GGUF Q8_0 | 25,921,889,792 | `ca1b1fb31317f7746ebf312c3158b783145e4a4179ddcd541a7720593d80cc78` |
| `Qwen3.8-27B-Turbo-Q2_K.gguf` | quant_rung | GGUF Q2_K | 14,340,367,872 | `6cdd40ea4dc80a148f95fcfe371d77be0967c53965668194d111873de1e075f1` |
| `Qwen3.8-27B-Turbo-mmproj-BF16.gguf` | mmproj | GGUF BF16 | 931,145,888 | `355817f1e18d1af0c9712f65e475f02c3b0457256f8d9ae300ca25ce6c222ab3` |

### Multimodal usage (`llama.cpp`)

```bash
llama-mtmd-cli -m Qwen3.8-27B-Turbo-Q4_K_M.gguf --mmproj Qwen3.8-27B-Turbo-mmproj-BF16.gguf --image photo.png -p "Describe this image."
```
