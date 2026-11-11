---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: qwegn
pipeline_tag: text-generation
tags:
  - qwegn
  - apple-silicon
  - metal
  - quantized
  - shvq
  - moe
  - mla
  - long-context
---

# GLM-5.3-Flash · `SHVQ4_G32 + SHVQ2_G64 + SHVQ3_G64` · Qwegn native pack

A native weight pack for the **[Qwegn inference engine](https://github.com/qwegnai/qwegn)**, a Swift + Metal 4 runtime for Apple Silicon. Spherical Hadamard Vector Quantization (`SHVQ`, `arXiv:2611.09501`) compresses all **321.32 B parameters** into **100.63 GiB (`2.69 bits/param`, `-33.74 GiB` / `-25.1%` vs `HQ4+HQ3`)**, allowing all `288/288` experts per layer to reside **100% in unified memory on a 128 GB Mac** with **zero SSD expert misses, zero OS swap, `36.2 tok/s` median decode (`38.4 tok/s` peak), and `44.0 tok/s` speculative decode**.

<p align="center"><img src="media/qwegn_model_glm53_flash.png" alt="GLM-5.3-Flash SHVQ4/2/3 on Qwegn" width="960"></p>

<p align="center"><img src="media/qwegn_lsa_scaling.png" alt="GLM-5.3-Flash scaling on Qwegn" width="960"></p>

> This pack uses Qwegn's own `QWP1` container and loads only in Qwegn (`v0.14.0+`). It is not a safetensors, GGUF or MLX checkpoint.

## At a glance

| Property | Value |
| :--- | :--- |
| Upstream model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) @ `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| Parameters | 321.32 B total (~18 B active) |
| Architecture | 45-layer hybrid MoE (3 dense + 42 MoE × 288 experts, 8 routed + 1 shared active), 11 NoPE MLA (`512-d` latent KV + Key-Pool(4)) + 34 KDA linear attention, 4-stream Sinkhorn-20 mHC, NextN MTP (Layer 45), 24-block vision tower |
| Pack size | **100.63 GiB** (`108,051,337,284` bytes; `-33.74 GiB` vs `134.37 GiB` `HQ4_G32 + HQ3_G64`) |
| Weights | `SHVQ4_G32` + `SHVQ2_G64` + `SHVQ3_G64` + `BF16`, **2.69 bits/param** |
| UMA residency (128 GB Mac) | **100% resident** (`288/288` experts/layer, `99.56 GiB` RSS, `0` SSD misses) |
| Native context | 202,752 tokens; MLA Key-Pool(4) sparse indexer (`k = 2048`) |

## Performance on Qwegn (`v0.14.1`)

Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core GPU, 585 GB/s), `llmprobe v0.6.17` & quiet-GPU `16K–1M` scaling, `--kv-mode q4 --temperature 0`.

| Metric | Result |
| :--- | :---: |
| Prefill (`~4.2K` prompt) | **618 tok/s** |
| Cold TTFT / Warm prefix-cache TTFT | 230–366 ms / **18 ms (`147.1×`)** |
| Decode throughput (novel reasoning) | **36.2 tok/s median** (`26.7–38.4 tok/s`) |
| Speculative decode (NextN + prompt lookup) | **44.0 tok/s** (`1.67×`, `3.69 tok/step`) |
| Long-context MLA `exactFlat` vs `e-means` (`QW_MLA_EMEANS=1`, `64K` / `160K`) | `24.26 →` **25.61 tok/s** (`+5.6%`) / `22.23 →` **26.63 tok/s** (**`+19.8%`, `3/3` needle**) |
| `1M` (`1,048,576` tok) UMA residency (`3,696 B/tok` `q4` MLA KV) | **100% resident** (`102.27 GiB` plan / `107.18 GiB` RSS, `0` SSD misses) |
| 4-stream concurrent aggregate | 30.8 tok/s |
| Multi-row verify determinism | **5 / 5 bit-exact SHA-256** |
| Structured JSON fidelity (`marker` / `nested-schema`) | **3 / 3 · 3 / 3 (`100%`)** |
| Graduate math (`AIME 2025`, `16K` budget) | **2 / 2 (`0` wrong answers across `core` eval)** |

## Mixed Spherical Hadamard Vector Quantization (`SHVQ`)

**2.69 bits per parameter overall.** Each weight group is rotated by an orthonormal Walsh–Hadamard transform and vector-quantized in 4D spherical coordinates via capacity-balanced deep `e-means` codebooks (`arXiv:2611.09501`), with asymmetric precision protecting sensitive projection classes:

| Tensor class | Format | Effective bpw | Share of parameters |
| :--- | :--- | :---: | :---: |
| Routed experts — `gate_proj` & `up_proj` (`24,768` planes) | `SHVQ2_G64` | 2.25 bpw | **64.66%** (`207.77 B`) |
| Routed experts — `down_proj` (`12,384` planes) | `SHVQ3_G64` | 3.25 bpw | **32.33%** (`103.88 B`) |
| Dense layers, NoPE MLA, KDA, shared experts, NextN MTP, `lm_head` (`462` planes) | `SHVQ4_G32` | 4.50 bpw | **2.58%** (`8.30 B`) |
| Token embeddings, vision tower, router gates & norms (`1,156` tensors) | `BF16` | 16.00 bpw | **0.43%** (`1.37 B`) |

### Why this layout works on Apple Silicon

- **Eliminates NVMe paging on 128 GB Macs:** Compressing routed experts from `3.50 bpw` (`HQ3_G64`, `134.37 GiB`) to `2.25 bpw` (`gate_up`) + `3.25 bpw` (`down_proj`) saves **`33.74 GiB`**, bringing the entire 321B pack inside 128 GB unified memory (`99.56 GiB` RSS) while `QWResidencyAdmission` automatically clamps server `PrefixCache` warm RAM to safe post-load headroom.
- **Threadgroup SRAM codebook staging (`ShvqTgLut<2|3>`):** Dense decode, greedy argmax, and routed MoE kernels cooperatively stage the `4 KiB` (`SHVQ2`) and `2 KiB` (`SHVQ3`) codebooks into `threadgroup` SRAM once per threadgroup, eliminating divergent `constant`-cache serialization on Apple AGX GPUs.
- **Asymmetric `down_proj` protection:** Keeping `down_proj` at `SHVQ3_G64` (`3.25 bpw`) and all dense/MLA/KDA/shared/MTP weights at `SHVQ4_G32` (`4.50 bpw` direct 32 KiB half-LUT) preserves residual-stream injection fidelity (`2/2` AIME 2025, `3/3` schema JSON).

## Memory footprint & residency flags

| Configuration (`16K` `q4` KV) | CPU RSS | Metal Wiring | Notes |
| :--- | :---: | :---: | :--- |
| Default (`--no-load-vision`) | `99.56 GiB` | `107.0 GiB` | `100%` weights UMA-resident; `BF16` `model.visual.*` (`1.05 GiB`) unmapped via `adviseDontNeed()` |

 steady-state decode executes with **0 heap and 0 Metal buffer allocations**.

## Quick start

Install Qwegn (`v0.14.0+`) from [https://github.com/qwegnai/qwegn](https://github.com/qwegnai/qwegn), then:

```bash
# OpenAI / Anthropic / Ollama server on 128 GB Apple Silicon
qwg serve ~/.qwg/models/GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  --kv-mode q4 --context-tokens 32768 --no-load-vision

# Single-shot CLI run
qwg run ~/.qwg/models/GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  --kv-mode q4 --prompt "Derive the complexity of Sinkhorn-Knopp matrix balancing."
```

## Requirements

- macOS 26+ on Apple Silicon (M5 Max 128 GB recommended for 100% UMA residency; Macs with smaller unified memory automatically engage Qwegn's `ColdEventPipeline` NVMe expert streaming)
- **Qwegn 0.14.0 or newer** (required for `SHVQ{2,3,4}` Metal kernels)

## Files

| File | Contents |
| :--- | :--- |
| `model.qwp` | Weights in Qwegn's `QWP1` container (`100.63 GiB`, `38,770` tensors) |
| `model.qwp.sha256` | SHA-256 of `model.qwp` |
| `README.md` | This model card |
| `media/` | Per-model Qwegn infographics (SVG + PNG) |
| `LICENSE` | Upstream MIT license |
| `assets/` | Upstream `config.json`, `tokenizer.json`, `chat_template.jinja`, `README.md` and `LICENSE` (`manifest.json` lists SHA-256 digests) |

## License and attribution

MIT, inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). Quantized by the Qwegn team; all credit for the base model belongs to its original authors.
