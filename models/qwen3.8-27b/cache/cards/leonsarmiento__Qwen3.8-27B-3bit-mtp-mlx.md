---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen3
- mtp
- 3-bit
base_model: Qwen/Qwen3.8-27B
---

# leonsarmiento/Qwen3.8-27B-3bit-mtp-mlx

Qwen3.8-27B VLM with the **MTP drafter baked in** — the speed variant of this quantization
family. Identical 3-bit mixed recipe as [Qwen3.8-27B-3bit-mlx](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx),
plus the source model's Multi-Token Prediction layer bundled inside the checkpoint, so the
engine runs speculative decoding natively — no separate drafter model to download or pair.

**Which one should you get?**

| | [Qwen3.8-27B-3bit-mlx](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx) | **This model (3bit-mtp)** |
|---|---|---|
| Footprint | 12.71 GB | 13.04 GB (+0.33 GB) |
| Peak memory (pp1024/tg128) | 13.16 GB | 15.69 GB with MTP on / 13.16 GB with MTP off |
| Speculative decoding | — | ✅ bundled, toggleable at runtime |
| Best for | The RAM-poor — leanest footprint | Machines with ~3 GB headroom that want extra decode speed |

> **⚠️ Lightning MTP speedup is an oMLX-only feature** (verified on oMLX v0.6.1, build 2323).
> LM Studio does not execute bundled MTP layers — this model loads and runs fine there, but
> the MTP tensors are inert dead weight. **LM Studio users: get the clean
> [Qwen3.8-27B-3bit-mlx](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx) instead.**
> MTP is also not needed for thinking mode — see below.

## Quantization Details

Same recipe as the clean build:
- **Embedding layers**: 4-bit with group_size=64
- **Prediction head (lm_head)**: 4-bit with group_size=64
- **Vision tower**: 8-bit with group_size=64
- **All other layers**: 3-bit with group_size=64

**MTP add-on** (extracted from the source `mtp.*` tensors, scottlowry/oQ checkpoint convention):
- 7 norm tensors: bf16, +1.0 MLX RMSNorm convention
- `fc.weight`: bf16 **unquantized** (bit-identical to source)
- 7 linear layers: 4-bit affine, group_size=64 (per-tensor config overrides — the 3-bit default doesn't apply to them)
- `text_config.mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`

**Stats**: 3.716 bpw base + 0.33 GB MTP · 13.04 GB · 3 shards · mlx-vlm · vision tower preserved · chat template with `reasoning_effort` defaulting to `low`.

## Benchmarks (oMLX v0.6.1, Engine Auto, Context: Code/Python, pp1024/tg128)

| Config | tg TPS (1×) | tg TPS (4×) | Peak Mem |
|---|---:|---:|---:|
| **3bit-mtp, Lightning MTP ON** | 20.0 | **90.9** | 15.69 GB |
| 3bit-mtp, MTP OFF | 15.3 | 67.4 | 13.16 GB |
| oQ4e-mtp reference (4-bit + iMatrix), MTP ON | 22.3 | 87.2 | 18.35 GB |
| oQ4e-mtp reference, MTP OFF | 14.8 | 59.0 | 16.41 GB |

- **+30.7% single-stream decode** with MTP on (15.3 → 20.0 tok/s)
- **Beats the 4-bit oQ4e reference at batch ≥ 2** (90.9 vs 87.2 tok/s @ 4×) while using **2.66 GB less memory**
- Single-stream sits ~10% below a 4-bit verifier: the 3-bit base rejects more speculative drafts (verifier noise) — the accepted trade for the smaller footprint
- **Disable TurboQuant KV cache** — MTP shows no advantage with it enabled

**When to turn MTP off**: thinking/reasoning-heavy tasks. Faster token generation gets spent
on *more* reasoning, not faster answers (measured on Qwen3.6-35B MATHQA: −7pp accuracy with
MTP on, +20% wall time). With MTP off this model behaves exactly like the clean 3-bit build.

## Source

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — dense hybrid-attention VLM,
64 layers (48 linear attention + 16 full attention), 262K context, MTP prediction layer.

## Instruct Benchmarks

| Model | MMLU | MMLU_PRO | HUMANEVAL | MBPP | Avg |
|-------|------|----------|-----------|------|-----|
| Huihui-Gemma-4 26B 3bit-XL | 78.0% | **71.0%** | **98.0%** | 82.0% | **82.3%** |
| Gemma-4 26B 3bit-XL | 80.0% | 65.0% | **98.0%** | **86.0%** | **82.3%** |
| Qwen3.8-27B 3bit-mtp | 80.0% | 61.0% | 94.0% | 70.0% | 76.3% |
| Qwen3.6-35B 3bit-XL | 82.0% | 63.0% | 66.0% | 80.0% | 72.8% |
| Huihui-Qwen3.6-35B 3bit-XL | **83.0%** | 63.0% | 62.0% | 76.0% | 71.0% |

![Family comparison](poor_ram_family_benchmarks.png)

*Instruct mode (thinking off). Sampled: MMLU 100/14042, MMLU_PRO 100/12032, HUMANEVAL 50/164, MBPP 50/500. Wall times not directly comparable across architectures (dense vs MoE active params).*

**Highlights**: HUMANEVAL **94%** at dense 3-bit — strong code for the size class. Benchmarks ran in instruct mode; engage Lightning MTP (oMLX only) for decode speed, not for these quality numbers.
