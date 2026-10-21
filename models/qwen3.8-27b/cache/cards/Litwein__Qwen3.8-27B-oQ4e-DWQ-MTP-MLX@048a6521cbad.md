---
license: apache-2.0
language:
- en
library_name: mlx
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
datasets:
- SWE-bench/SWE-smith-trajectories
- open-thoughts/OpenThoughts-114k
- open-r1/OpenR1-Math-220k
- bigcode/self-oss-instruct-sc2-exec-filter-50k
- open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled
- nvidia/OpenCodeReasoning
- allenai/tulu-3-sft-mixture
tags:
- mlx
- oq
- imatrix
- dwq
- m4q
- quantization
- 4-bit
- mtp
- speculative-decoding
- qwen3.8
- qwen3_5
---

# Qwen3.8-27B-oQ4e-DWQ-MTP-MLX

> ### Compressed for Apple Silicon
> A benchmark-aligned **oQ4e-XL (imatrix) → M4Q → DWQ** build of Qwen3.8-27B.
> M4Q + DWQ reduced held-out teacher divergence by **20%** (`0.0287 → 0.0229`) over the
> M4Q initializer and by **60%** over the raw oQ4e-XL donor (`0.0579`), with the native
> **MTP** speculative-decoding head preserved.

A mixed-precision **4-bit MLX quantization** of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — the smallest text/code/agent
member of the family, with the donor's mixed-precision MTP head retained for native
speculative decoding in oMLX.

> ⚠️ These are quantized weights. **The model's capability comes from Qwen's base model** —
> please star/cite it first. This repo contributes the oQ/DWQ quantization recipe,
> Hessian-guided M4Q protection and MTP packaging; M4Q and DWQ tune the quantizer's
> scales/biases and do not add new knowledge.

## Model lineage

```
Qwen/Qwen3.8-27B                          (Apache-2.0 · dense 27B · hybrid attention · vision + MTP)
  └─ THIS REPO: oQ4e-XL (imatrix) → M4Q → DWQ   + MTP (text-only)
```

- **Architecture:** dense Qwen3.8-27B (`qwen3_5` MLX architecture), 64 hybrid
  Gated-DeltaNet/full-attention layers.
- **MTP head:** native multi-token prediction head for oMLX's speculative decoding.

## Quantization: `oQ4e-XL` (imatrix) → `M4Q` → single-stage `DWQ`

This is not a plain round-to-nearest 4-bit conversion:

1. **`oQ4e-XL` — importance-aware mixed precision.** oMLX's enhanced quantizer builds an
   importance matrix from **256 × 512-token** calibration samples (`oqe_code_multilingual`,
   adaptive) and allocates additional precision to sensitive tensors. The **XL** recipe then
   surgically promotes the architecture's precision-critical tensors to 8-bit: all
   `linear_attn.*` (Gated-DeltaNet) and `self_attn.*` projections, the embeddings and the
   lm_head — the same layer-swap idea as
   [leonsarmiento/Qwen3.6-35B-A3B-oQ4e-mtp-XL-mlx](https://huggingface.co/leonsarmiento/Qwen3.6-35B-A3B-oQ4e-mtp-XL-mlx).
   The result is an affine **4-bit, group size 64** base with **310 quantized modules
   promoted to 8-bit and 27 modules to 5-bit**.
2. **M4Q — Hessian-guided protection.** The bf16 base is streamed and every module's affine
   quantization (scales/biases, group size 64, 4-bit base) is optimized against
   activation-weighted Hessian objectives from the calibration data, keeping the XL bit map.
3. **DWQ — activation-aligned distillation.** The trainable affine scales/biases of all
   sub-8-bit language modules are optimized toward an **`oQ8e` teacher made from the same
   base model**. The objective is KL divergence over the teacher's top-1024 logits at
   temperature 2.0. Training uses batch 1, 512-token windows, gradient checkpointing, Adam,
   a cosine LR schedule, validation early stop and exports only the best checkpoint.
4. **Component splice.** The donor's mixed-precision MTP head is copied back byte-for-byte
   after DWQ; it is not trained by the text-only DWQ pass.

| | |
|---|---|
| Scheme | affine 4-bit base (g64) · 310×8-bit + 27×5-bit protected modules · one DWQ pass |
| Weight files | **20.7 GB** safetensors (including MTP) |
| Distillation teacher | `oQ8e` enhanced quantization of the same Qwen3.8-27B base |
| DWQ calibration | 4,096 train / 128 valid · document-disjoint splits · ≤512-token windows (ThinkingCap v4 recipe) |
| Held-out DWQ loss | M4Q init 0.0287 → **0.0229** (**−20%**; raw oQ4e-XL donor 0.0579) |
| Tooling | oMLX `oq`/`oQe` + M4Q Hessian init + `mlx_lm.quant.dwq` |

The loss above measures fidelity to this recipe's 8-bit teacher on its held-out calibration
distribution. It is **not task accuracy** and should not be compared with losses from another
dataset, tokenizer, teacher or sequence length. The oQ8e teacher itself sits at KL 0.019 vs
the bf16 base, so the final model (0.023) is within ~20% of the teacher's own noise budget.

### Calibration mix

The single DWQ stage retains general reasoning while emphasizing the model's intended code and
agent workloads. DWQ sees tokenized activation windows, not additional SFT updates.

| Partition | Train / valid | Share | Sources / role |
|---|---:|---:|---|
| Resolved SWE agent trajectories | 1,024 / 32 | 25% | SWE-smith tool calls, patches and terminal output |
| General reasoning | 768 / 24 | 18.75% | OpenThoughts science/math/code traces |
| Verified math | 512 / 16 | 12.5% | OpenR1-Math |
| Concise function code | 512 / 16 | 12.5% | execution-filtered self-OSS-Instruct |
| Verified Python problems | 512 / 16 | 12.5% | decontaminated/tested verifiable coding problems |
| Competitive code reasoning | 512 / 16 | 12.5% | NVIDIA OpenCodeReasoning |
| General instruction/chat | 256 / 8 | 6.25% | Tulu 3 SFT mixture |

Agent + code data is **62.5%** of the mix.

## Evaluation

### Quantization fidelity measured on this build

| Metric | Raw oQ4e-XL | M4Q init | **oQ4e → M4Q → DWQ (this repo)** |
|---|---:|---:|---:|
| Held-out top-K teacher KL | 0.0579 | 0.0287 | **0.0229** |
| Relative reduction vs M4Q init | — | — | **20%** |

No task benchmark score is claimed here yet. The result above demonstrates improved
language-logit fidelity on the held-out calibration split, not guaranteed benchmark
improvement. (The raw-donor number was measured on a 16-batch subset; the M4Q/DWQ numbers on
the full 128-batch split.)

## Repos in this family

| Repo | Vision | Safetensors | Use case |
|---|:--:|---:|---|
| [`Qwen3.8-27B-oQ4e-DWQ-MTP-Vision-MLX`](https://huggingface.co/Litwein/Qwen3.8-27B-oQ4e-DWQ-MTP-Vision-MLX) | ✓ | **21.6 GB** | multimodal |
| **`Qwen3.8-27B-oQ4e-DWQ-MTP-MLX`** (this) | – | **20.7 GB** | smallest · text/code/agents |

The siblings share the **same DWQ language backbone and MTP head** (backbone shards are
hard-linked); the Vision sibling additionally contains the original bf16 vision tower.

## How to run

These are **MLX weights for Apple Silicon**. The tested serving path is
**[oMLX](https://omlx.app)**, which supports Qwen3.8 and native MTP speculative decoding.

```bash
# Download directly into the oMLX model directory
hf download Litwein/Qwen3.8-27B-oQ4e-DWQ-MTP-MLX \
  --local-dir ~/.omlx/models/Qwen3.8-27B-oQ4e-DWQ-MTP-MLX

# Enable native MTP once
curl -X PUT \
  http://127.0.0.1:8003/admin/api/models/Qwen3.8-27B-oQ4e-DWQ-MTP-MLX/settings \
  -H "Authorization: Bearer sk-local" -H "Content-Type: application/json" \
  -d '{"mtp_enabled": true}'

# OpenAI-compatible API
curl -X POST http://127.0.0.1:8003/v1/chat/completions \
  -H "Authorization: Bearer sk-local" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B-oQ4e-DWQ-MTP-MLX",
       "messages":[{"role":"user","content":"Explain what neural network quantization is in one paragraph."}],
       "max_tokens":1024,"temperature":0.7,"top_p":0.8}'
```

The oMLX model id is **case-sensitive** and matches the downloaded folder name. Native MTP
requires a compatible runtime; oMLX is the path validated for this release.

### Recommended sampling

Follow the base model's recommended settings (Qwen3-style defaults: thinking enabled,
**temperature ~0.7, top_p ~0.8**; greedy decoding is useful for reproducible benchmarks).
Give hard reasoning/code tasks a generous output budget (32k or more where practical).

## Intended use & limitations

- **Best suited to:** coding, tool/agent workflows, math, STEM and long-context chat.
- **Quantization is lossy:** for maximum fidelity use the bf16 base or a higher-bit quant.
- **DWQ is calibration, not SFT:** it improves quantized-teacher fidelity on represented
  activations; it does not teach facts or guarantee gains on every benchmark.
- **Calibration coverage:** public resolved SWE trajectories are Python-centric; multilingual
  SWE/terminal/skills behaviour was not directly calibrated.
- **Long context costs memory:** the architectural context maximum is not a promise that every
  Apple Silicon machine can allocate the corresponding KV/cache state.

## Acknowledgements

- **Qwen team** — Qwen3.8-27B (Apache-2.0).
- **Apple MLX** — `mlx`, `mlx-lm`, `mlx-vlm` and `mlx_lm.quant.dwq`.
- **oMLX** — enhanced `oq`/`oQe` imatrix quantization, M4Q support and MTP serving runtime.
- **Calibration-data authors** — SWE-bench/SWE-smith, OpenThoughts, Open-R1, BigCode,
  NVIDIA OpenCodeReasoning and AllenAI Tulu.

## License

**Apache-2.0**, inherited from `Qwen/Qwen3.8-27B`.

## Citation

Please cite the original base model:

```bibtex
@misc{qwen3.8-27b,
  title  = {Qwen3.8-27B},
  author = {Qwen Team, Alibaba Group},
  year   = {2026}
}
```
