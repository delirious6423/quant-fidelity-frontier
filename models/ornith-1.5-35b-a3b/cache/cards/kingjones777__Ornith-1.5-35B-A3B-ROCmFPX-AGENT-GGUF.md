---
license: apache-2.0
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
  - ornith-ai/Ornith-1.5-35B-A3B-GGUF
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - mtp
  - speculative-decoding
  - gguf
  - rocmfp4
  - rocmfpx
  - strix-halo
  - gfx1151
  - amd
  - rocm
  - quantized
  - ornith
  - multimodal
  - agent
---

# Ornith-1.5-35B-A3B — AGENT — ROCmFP4 / ROCmFPX + MTP GGUF

**The 8-bit AGENT tier** of `ornith-ai/Ornith-1.5-35B-A3B` — ftype 115 `Q8_0_ROCMFPX_AGENT`, which keeps more attention-output tensors at higher precision for tool-calling and speculative-draft acceptance.

Built for **AMD Strix Halo (gfx1151)** — Ryzen AI MAX+ 395, 128 GB unified memory, ROCm 7.2.4 — using the [ROCmFPX](https://github.com/charlie12345/ROCmFPX) llama.cpp fork, which adds AMD-native FP4/FP8 tensor types that mainline llama.cpp does not have.

> ⚠️ These files require a **ROCmFPX-capable llama.cpp build**. They will **not** load in
> stock llama.cpp / Ollama / LM Studio — the `Q4_0_ROCMFP4_*` and `Q*_0_ROCMFPX*` tensor
> types are not in mainline.

## Variants in this repo

| file | ftype | size | BPW | `token_embd` | `output.weight` | decode | decode +MTP |
|---|---|---|---|---|---|---|---|
| `Ornith-1.5-35B-A3B-Q6_0_ROCMFPX_AGENT.gguf` | 114 | 30.89 GiB | 7.47 | q8_0 | **q8_0** | **48.56 t/s** | 43.22 t/s (0.89×) |
| `Ornith-1.5-35B-A3B-Q8_0_ROCMFPX.gguf` | 111 | 34.20 GiB | 8.27 | q8_0 | **q8_0** | **45.74 t/s** | 43.17 t/s (0.94×) |
| `Ornith-1.5-35B-A3B-Q8_0_ROCMFPX_AGENT.gguf` | 115 | 34.74 GiB | 8.40 | q8_0 | **q8_0** | **45.69 t/s** | 41.12 t/s (0.90×) |

**Which to pick:** the AGENT tiers when tool-calling fidelity matters more than speed. ⚠️ `Q6_0_ROCMFPX_AGENT` is **not** a middle rung — it lands at 7.47 BPW and performs like the 8-bits (48.56 vs 45.74 t/s). Choose it for precision, not size. For general use the 4-bit **COHERENT** repo is **1.32× faster** at **54% the size**.

## Head protection — verified in the file, not assumed

`tie_word_embeddings` is **false** on this model, so `output.weight` is a real standalone tensor and `--output-tensor-type` genuinely bites. Every artifact here was re-opened after quantization and its header read back:

| ftype | `token_embd.weight` | `output.weight` |
|---|---|---|
| 102 | q6_K | **q6_K** |
| 114 | q8_0 | **q8_0** |
| 111 | q8_0 | **q8_0** |
| 115 | q8_0 | **q8_0** |

⚠️ For contrast, the other public ROCmFP4 build of this model (`julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF`, ftype 106) ships **Q5_K token embeddings and a 4-bit `output.weight`**, while its card states "FP16 embedding/norm preservation". We read both headers with two independent parsers. Head protection is not implied by an ftype name — it has to be requested and then verified in the file.

## Measured — not estimated

Hardware: **AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151)**, 128 GB unified, ROCm 7.2.4. Idle box, **2 warm-ups discarded**, **median of 5**, 300 tokens, identical prompt across every sample.

| quant | run 1 | run 2 | run 3 | run 4 | run 5 | median |
|---|---|---|---|---|---|---|
| 114 | 48.45 | 48.60 | 48.57 | 48.56 | 47.41 | **48.56** |
| 111 | 45.76 | 45.74 | 45.74 | 45.74 | 45.74 | **45.74** |
| 115 | 45.69 | 45.83 | 45.71 | 45.65 | 45.63 | **45.69** |

### Speculative decoding (MTP)

This model ships its **draft head inside the base weights** — `qwen35moe.nextn_predict_layers = 1`, tensors under `blk.40.nextn.*`. Enable it with `--spec-type draft-mtp --spec-draft-ngl 999` and **no `--model-draft`**.

⚠️ Do **not** pass `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` as a draft model. It is the same 753 tensors with 32 bytes of extra metadata — loading it as a drafter loads a **second full 35B**.

⛔ **MTP measured a net loss on every tier here** (0.89–0.94×). The flags are documented so you can turn it on; we are not selling it as faster. Best n-max on the 4-bit was **3** (56.05 t/s) — note that n=2 had *higher* acceptance (0.927 vs 0.913) and was **slower**, so rank on t/s, not acceptance.

## Source — byte-verified, not re-converted

Quantized from `ornith-ai/Ornith-1.5-35B-A3B-GGUF` → `Ornith-1.5-35B-BF16.gguf`, **71,066,994,240 bytes**, sha256 `a3ee48dd8f05d10f529aa8ca8b9e080082c38910909638d226001b74e3307591`. The vision projector `mmproj-Ornith-1.5-35B-BF16.gguf` (**902,822,016 bytes**, sha256 `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837`) is included.

Both were byte-verified against the Hub before quantization. **No re-conversion from safetensors.**

## Verification

Every artifact: loaded at `-ngl 999 -c 4096 -fit off -fa on`, **3/3 correctness** (17×23 → 391, capital of Japan → Tokyo, days in 2024 → 366) asserted on both `content` and `reasoning_content` with `finish_reason=stop`, and **4/4 vision** on a four-quadrant colour image via the mmproj.

File sizes were checked against `--dry-run` projections: the header delta is a **constant 10.48 MiB** across all artifacts (spread 0.005 MiB), which is the signature of complete, untruncated files.

⚠️ **Bandwidth note:** this is a **256-expert MoE with ~3B active**. Effective bandwidth must be computed against the *active* weight (~1.72 GB at 4.58 BPW), not the 20.3 GB file. At 60.34 t/s that is ~104 GB/s — the file size is not the bus.
