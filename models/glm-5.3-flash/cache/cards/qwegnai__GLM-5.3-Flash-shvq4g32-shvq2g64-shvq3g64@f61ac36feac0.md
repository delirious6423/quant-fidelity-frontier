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

A native weight pack for the **[Qwegn inference engine](https://github.com/qwegnai/qwegn)**, a Swift + Metal 4 runtime for Apple Silicon. Spherical Hadamard Vector Quantization (`SHVQ`, `arXiv:2611.09501`) compresses all **321.32 B parameters** into **100.63 GiB (`2.69 bits/param`, `-33.74 GiB` / `-25.1%` vs `HQ4+HQ3`)**, allowing all `288/288` experts per layer to reside **100% in unified memory on a 128 GB Mac** with **zero SSD expert misses, zero OS swap, `1,014 tok/s` prefill @4K (`1,006` @5.2K · `825–869 tok/s` server), `46.0 tok/s` median decode (`46.4 tok/s` peak), `50.7–53.6 tok/s` exact NextN speculative decode (`1.98 tok/step`), `81/100` HumanEval+MBPP (`92%` HE), and `99.74%` `llmprobe` fidelity**.

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
| UMA residency (128 GB Mac) | **100% resident** (`288/288` experts/layer, `97.84 GiB` RSS serial / `100.14 GiB` MTP-1, `0` SSD misses) |
| Native context | 202,752 tokens (`1M` UMA-capable); MLA Key-Pool(4) sparse indexer (`k = 2048`) + `e-means-4096` |

## Performance on Qwegn (`v0.17.1–v0.20.5`)

Apple M5 Max (`Mac17,6`, 128 GB unified memory, 40-core Apple10 NAX GPU, 585 GB/s), `paired_perf --strict`, `llmprobe v0.6.16–v0.6.17` (`--full` `1.49M` tokens + `100`-task `code`), & quiet-GPU `15K–242.5K+` scaling, `--kv-mode q4 --temperature 0`.

| Metric | Result |
| :--- | :---: |
| Prefill (`4K` / `5.2K` CLI / `9.4K` `serve` / `65.6K` `serve`) | **1,014 tok/s** @4K / **1,005.9** @5.2K (`1,061.3` GPU) / **825.3–869.2** (`serve`) / **780 tok/s** (@65.6K) |
| Cold TTFT / Warm prefix-cache TTFT | 180–339 ms / **10–18 ms (`96–193.4×`)** |
| Decode throughput (novel reasoning) | **46.0 tok/s median** (`44.64` CLI – **46.4 tok/s peak** `qwg serve`) |
| Speculative decode (exact NextN MTP-1) | **53.6 tok/s predictable** · **50.7 tok/s peak** (`45.6–47.3` novel `serve` · **51.19 tok/s** CLI, `1.98 tok/step`) |
| Sustained 21-minute thermal stability (`llmprobe --full`) | **Steady `+5.4%` (`41.1 → 43.3 tok/s`)** (`>= 75%` UMA auto scratch eviction) |
| Long-context `15K` / `60K` / `242.5K` (`4,096` NAX chunk + `e-means`) | **834.2 / 46.54** (`15K`) · **706.7 / 32.38** (`60K`) · **743.0 / 44.98 tok/s** (**`242.5K`, `3/3` needle**) |
| `1M` (`1,048,576` tok) UMA residency (`3,696 B/tok` `q4` MLA KV) | **100% resident** (`102.27 GiB` plan / `107.18 GiB` RSS, `0` SSD misses) |
| API conformance / Model capability (`llmprobe --full`) | **184 / 184 (`100%`)** · **37 / 37 (`100%`, `strong`)** |
| Engine fidelity (`llmprobe --full`) | **99.74%** (`36/36` correctness · `3/3` greedy determinism · `36/36` logprob argmax) |
| Coding (`100`-task HumanEval + MBPP, `--reasoning off`) | **81 / 100 (`81.0%`: `46/50` `92.0%` HumanEval · `35/50` `70.0%` MBPP)** |
| Agentic execution / Graduate math (`AIME 2025`) | **66 / 88 (`75.0%` agentic)** · **2 / 2 `AIME 2025` · 3 / 3 nested schema JSON** |

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
| Serial (`--no-load-vision`) | `97.84 GiB` | `102.89 GiB` | `fixed_scratch = 4.31 GiB` (`-2.65 GiB` in `v0.17.0`); `288/288` experts resident; `model.visual.*` (`1.05 GiB`) unmapped |
| Exact NextN MTP-1 (`--mtp-drafts 1`) | `100.14 GiB` | `105.65 GiB` | `fixed_scratch = 4.48 GiB`; 1-draft CPU `embed_tokens` path (`0` VNODE faults, `p95 = 39 ms`) |

Steady-state decode executes with **0 heap and 0 Metal buffer allocations**.

## Quick start

Install Qwegn (`v0.20.5+` recommended, `v0.17.1+` minimum) from [https://github.com/qwegnai/qwegn](https://github.com/qwegnai/qwegn), then:

```bash
# Download the pack
qwg pull qwegnai/GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64

# OpenAI / Anthropic / Ollama server on 128 GB Apple Silicon
qwg serve ~/.qwg/models/GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  --kv-mode q4 --context-tokens 32768 --no-load-vision

# Single-shot CLI run
qwg run ~/.qwg/models/GLM-5.3-Flash-shvq4g32-shvq2g64-shvq3g64 \
  --kv-mode q4 --prompt "Derive the complexity of Sinkhorn-Knopp matrix balancing."
```

## Requirements

- macOS 26+ on Apple Silicon (M5 Max 128 GB recommended for 100% UMA residency; Macs with smaller unified memory automatically engage Qwegn's `ColdEventPipeline` NVMe expert streaming)
- **Qwegn 0.14.0 or newer** (`v0.17.1+` recommended for generalized `SHVQ`/`q4` NoPE MLA NAX acceleration, `>= 75%` UMA scratch eviction, and parallel tool calling)

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
