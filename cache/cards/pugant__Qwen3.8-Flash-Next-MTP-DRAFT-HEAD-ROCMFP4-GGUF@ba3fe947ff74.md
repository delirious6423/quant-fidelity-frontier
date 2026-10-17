---
library_name: llama.cpp
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - gguf
  - mtp
  - nextn
  - speculative-decoding
  - draft-model
  - rocmfp4
  - rocmfpx
  - strix-halo
  - qwen4exp
pipeline_tag: text-generation
quantized_by: pugant
version: '1.0'
date: 2026-09-10
---

# Qwen3.8-Flash-Next MTP draft head — head-only ROCmFP4 requant (GGUF)

The jointly-trained MTP (NextN) draft head of **Qwen/Qwen3.8-Flash-Next** (arch `qwen4exp`), requantized **head-only** for ROCmFPX engines: the draft LM head (`output.weight`) is converted to **Q4_0_ROCMFP4** (644.14 → 341.02 MiB), while **all other 33 tensors are byte-identical** to the community Q8_0 extraction of the official head.

This is the sidecar served in our production engine since 2026-09-10.

> **About the filename**: the file is named `…-MTP-Q8_0-HEAD-ROCMFP4.gguf` because the base
> file type is Q8_0 — only the head tensor is ROCmFP4. The per-tensor types are the
> authoritative ones (same convention as our model cards).

## Why a matched-quant draft head

The draft head re-reads its **entire LM head** (248k vocab × 2560) for **every drafted
token** — a per-token matmul over the full vocabulary. On a bandwidth-limited APU that
read dominates the draft traffic. Quantizing the drafter *like the target* has a
measurable side benefit beyond the byte saving: the two quantized models are wrong in the
same places, so **acceptance goes up**, not down (first observed by
[agentionai](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST-GGUF)
on the same hardware class).

## Measured

Radeon 8060S (gfx1151), Vulkan, target = our
[ROCmFP4 STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-Flash-Next-ROCMFP4_STRIX_LEAN-GGUF)
build, `draft-mtp` n-max 6 / p-min 0.75, interleaved A/B in a production-style container.
Full protocol in the [lab wave-6 note](https://github.com/pugant/strix-nebulosa/blob/main/docs/experiments/2026-09-10-vulkan-optimization-wave6.md).

| sidecar | decode t/s | acceptance |
|---|---:|---:|
| Q8_0 head (community) | 23.92 | 0.652 |
| **this head** | **24.99 (+4.5%)** | **0.667** |

Greedy fidelity: the engine release this head ships in passed the 12-prompt battery
**bit-identical** to the pinned canon (sha `6cc4f583…`) — the head-only requant does not
change generated text in the greedy regime.

## Runtime

The `Q4_0_ROCMFP4` tensor type needs a **ROCmFPX-capable build** — stock llama.cpp cannot
load this file. Recommended runtime: our lab build
([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) — full source,
Dockerfile and guides; upstream format definition:
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX).

```bash
llama-server -m Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-*.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q8_0-HEAD-ROCMFP4.gguf \
  --spec-type draft-mtp --spec-draft-ngl all \
  --spec-draft-n-max 6 --spec-draft-p-min 0.75
```

Pairs naturally with the LEAN FP4 target above (same quant family on both sides of the
speculative pair — that is the point of the matched quant).

## Provenance & reproduction

- **Base**: the community Q8_0 extraction of the official head
  ([quimmedes/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/quimmedes/Qwen3.8-Flash-Next-MTP-GGUF)),
  itself converted from the official checkpoint via the llama.cpp NextN export
  ([PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836) lineage).
- **One tensor converted** (`output.weight` Q8_0 → Q4_0_ROCMFP4); the 33 non-head tensors
  are sha256-identical to the base; all metadata values unchanged (file size 3.82 GB).
- **Reproduce** (native quantizer flags, no patches needed — quantize flags go first):

  ```bash
  llama-quantize --allow-requantize --pure \
    --output-tensor-type Q4_0_ROCMFP4 \
    mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    Qwen3.8-Flash-Next-MTP-Q8_0-HEAD-ROCMFP4.gguf Q8_0
  ```

  Script with dry-run pre-flight and stat checks:
  [`scripts/w6t3-drafter-head-requant.sh`](https://github.com/pugant/strix-nebulosa/blob/main/rocmfpx/scripts/w6t3-drafter-head-requant.sh)
  in the lab snapshot; it regenerates this artifact bit-identically.

- **sha256**: `fa30a89fa06cd1bcc40810dcf1fed73d69c5ff33afb2a08b76333c6f9629ea34`

## Credits & license

- **Qwen team** — the MTP head itself, trained jointly with the target model.
- **[quimmedes](https://huggingface.co/quimmedes)** — the Q8_0 GGUF extraction this file
  derives from, and the cafe-llama.cpp fork that runs it.
- **[charlie12345](https://github.com/charlie12345)** (ROCmFP4 format / ROCmFPX) and
  **[ciru-ai](https://github.com/ciru-ai)** (ROCmFPX) — the tensor types.
- **[danielhanchen](https://github.com/danielhanchen)** (qwen4exp support,
  [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)) and **JJJYmmm**
  (MTP draft graph, [PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739)).
- **[agentionai](https://huggingface.co/agentionai)** — the matched-quant observation.

Derivative of Qwen/Qwen3.8-Flash-Next, distributed under the **Qwen Community License
1.0** (see `LICENSE`; note the license's Model-as-a-Service clause before commercial
serving). Requantization and measurements by **pugant**; work assisted by GLM by z.ai.

---

> Everything here is experimental and provided as-is, at your own risk.
