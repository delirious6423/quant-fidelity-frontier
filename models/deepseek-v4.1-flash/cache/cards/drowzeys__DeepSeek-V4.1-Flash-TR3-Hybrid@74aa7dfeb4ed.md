---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - moe
  - exl3
  - mxfp4
  - hybrid-quant
  - dgx-spark
base_model: deepseek-ai/DeepSeek-V4.1-Flash
---

# DeepSeek-V4.1-Flash TR3 Hybrid

Custom **hybrid quant** of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) for 4× NVIDIA DGX Spark (GB10).

This is **not** a uniform 3.5 bpw Pollard EXL3 pack. It is a **TR3 hybrid**: most routed experts are EXL3 **3.0 bpw**, the 64 hardest experts per layer stay **native MXFP4**.

## Bits per weight

| Piece | Format | BPW |
|---|---|---|
| Routed **tail** | EXL3-TR3 **K=3**, codebook **mcg** | **3.0** |
| Routed **keep-64** / layer | native MXFP4 (E2M1 + UE8M0/32) | **4.25** |
| **Expert-weighted mix** | 64 keep + rest tail | **≈3.22** |
| Attention, shared expert, norms, embeddings | official FP8 / native | unchanged |
| Engram (shards 47–48) | official, hardlinked | unchanged (~203 GB) |

Do not label this repo “3 bpw only.” The honest headline is **3.0 bpw EXL3-TR3 tail + 64 MXFP4 keeps/layer (~3.22 bpw MoE mix)**.

Disk: official **476 GB** → this pack **410 GB** (48 shards). Non-Engram 273 → 207 GB (~0.76×), matching 320/384 experts at 3/4.25 and 64/384 kept MXFP4.

## Validation - KLD & tier-1 fidelity

Measured on 4x DGX Spark (vLLM, TP4). **Reference = native**, i.e. the shipped
DeepSeek-V4.1-Flash MXFP4-experts/FP8 checkpoint this pack was quantized from
(there is no public BF16 original), so KL(native || TR3) is the loss this hybrid
adds on top of the release.

KLD is measured to [brandonmusic](https://huggingface.co/brandonmusic)'s spec:
the teacher's (native's) logits over a representative **32-window x 2047-token
corpus = 66,599 teacher-forced positions (>50K)**, top-20 truncation, KL averaged
per position (no alignment artifact).

| metric | native (ref) | **TR3-Hybrid** | EXL3 3.5bpw Pollard |
|---|---|---|---|
| KL vs native (nats), lower=better | 0 | **0.032** | 0.057 |
| top-1 agreement (teacher-forced) | 1.000 | **0.984** | 0.976 |
| objective battery (30 items) | 24/27 | 25/27 | 25/27 |

**Tier-1 (the K3 trellis tail) is high fidelity.** TR3's KL is **~44% below** the
uniform EXL3 3.5 bpw pack, and both track native tightly (top-1 > 97.5%); for
scale a third-party GLM-5.3 EXL3 KLD was ~0.102 nats. An early 274-position
smoke test read ~0.26 nats and swung with alignment - that sample was far too
small (per brandonmusic, use >=50K positions), not a real tier-1 problem.

Full tables, methodology, reproducible scripts and raw captures:
https://github.com/drowzeys/keys-DeepSeekV4.1-Flash-TR3-Hybrid-1M-Context-4M-KV-Four-DGX-Sparks

## Recipe

- **Source:** `deepseek-ai/DeepSeek-V4.1-Flash` (552B backbone, 384 routed + 1 shared, top-6, `moe_intermediate_size=2304`, 1M context).
- **Keep:** 64 experts/layer with the **highest trellis round-trip error**, left as MXFP4.
- **Tail:** EXL3 **K3 mcg**, full expert matrices (2304 and 5120 are both `% 128 == 0`).
- **DSpark layers 37–39:** 128 routed experts; still keep-64, so 64 tail on those layers.
- **Hessian:** identity / `q_fallback` (Pass A does not fit 4×128 GB unified memory).
- **Smoke:** identity-H mcg 5120×2304, nmse **1.74e-2**.
- Encoder: `encode_dsv41_tr3.py` wrapping the TR3 numeric core.

`config.json` field `hybrid_tr3_tail`: `bits: 3.0`, `codebook: mcg`, `keep_mxfp4_per_layer: 64`.

## Credit

**The hybrid quant method is [brandonmusic](https://huggingface.co/brandonmusic)’s TR3 recipe**, first published as [`brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid`](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid): keep the highest round-trip-error experts in native 4-bit, EXL3-TR3 **K3 mcg** on the tail, same numeric core (`encode_tr3_v31.py`). This checkpoint ports that method onto DeepSeek-V4.1-Flash. If you use these weights, credit **brandonmusic** for the hybrid quant.

Also:

- DeepSeek-AI for the base model.
- EXL3 trellis / mcg as used in that TR3 pipeline.

This pack is **not** [`bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) (uniform ~3.51 bpw Pollard on all routed experts). Different method, different BPW mix.

## Intended runtime (4× DGX Spark)

EXL3 GEMM requires **N % 128 == 0**. Uniform TP4 on 2304-wide experts is `2304/4 = 576` (illegal). Options that are legal:

- **Plugin expert-parallel:** 96 full-width 2304 experts/rank (this lab’s hybrid path).
- **Uneven TP4 slice** `512/640/640/512` (Tony / cuda-exl3 128-aligned split).
- **TP3:** `2304/3 = 768` (legal); needs V4.1 TP3 patches (virtual heads 64→72, etc.).

## License

MIT, same as the DeepSeek-V4.1-Flash release.
