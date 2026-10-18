---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags: [gguf, rocmfpx, strix-halo, mixed-precision, quantization]
---

# DeepSeek-V4-Flash-0731 — ROCmFPX mixed precision for Strix Halo

**File: `ds4-0731-gu106down-v3.gguf` — 98.29 GB, single file, no sidecars. 2.766 bits per weight**, computed all-in: whole file (header and embedded codebooks included) over the artifact's 284.33B parameters. The published reference for this checkpoint is 2.88 bpw at the same 82/92 score.

A mixed-precision GGUF of DeepSeek-V4-Flash-0731 (660B MoE) sized for the AMD
Strix Halo **96 GiB single-allocation window**: it loads **unsplit on the iGPU**
of a 128 GB Strix Halo box (Ryzen AI MAX+ 395 / Radeon 8060S) with stock
parameters, leaving headroom for a speculative-decode draft. All codebooks are
embedded in the GGUF as KV — download one file and serve.

## Quality

| suite | score |
|---|---|
| COMPSEC-17 (think budget 15488) | **17/17** |
| full ds4-eval 92 (COMPSEC + AIME2025 + GPQA-Diamond + SuperGPQA) | **82/92** |

At **2.766 bpw** overall — ~4% fewer bits per weight than the 2.88-bpw published
reference that scores the same 82/92.

82/92 equals the published reference artifact for this checkpoint — with one
important difference: **this model's calibration holds out all 92 eval items**
(the reference's calibration set contained 75 of them). Scores were produced with
the reference's own grader (`ds4-eval --regrade-trace`, self-check 92/92) at the
matched thinking budget, serving on dflash with fused decode off.

## Serving configs (measured, Radeon 8060S iGPU, 128 GB)

Two supported configurations — pick per workload. The speed lever is
`--ds4-expert-top-k 4` (4 instead of the model-default 6 routed experts per
token) plus fused decode; it costs measured quality, stated below, never
hidden:

| config | flags | decode | COMPSEC-17 |
|---|---|---|---|
| **quality** (default) | model-default top-k 6, fused decode off | 18.1 tok/s | **17/17** |
| **throughput** | `--ds4-expert-top-k 4 --ds4-fused-decode` | **22.3 tok/s** (+23%) | 16/17 |

The 82/92 full-eval score was measured at the quality config. The DSpark
speculative drafter does NOT currently help this artifact (accept rate
0.51–0.67 vs 0.837 on the reference it was calibrated for — net slower);
draft–target co-calibration for mix-quantized targets is an open item.

## Bit allocation

Every tensor's format is chosen by a measured allocator (layer-output damage
under real MoE routing, one knapsack over the whole file under the 98.4e9-byte
ceiling) — nothing is hand-picked. The result:

| family | format | bpw |
|---|---|---|
| routed gate/up experts (all 43 layers) | Q2_1_ROCMFP2_MIX (106) — learned per-expert codebooks | 2.50 |
| routed down experts, 15 layers | Q2_1_ROCMFP2_MIX (106) | 2.50 |
| routed down experts, 28 layers | Q3_1_ROCMFP3_MIX (105) — adaptive | 3.50 |
| attention / dense / shared expert | Q4_0_ROCMFP4_FAST (101) | 4.25 |
| norms / router / scaffolding | F32 / 101 passthrough | — |

The artifact is gated before release: a registered golden probe (COMPSEC ≥16
bar, positive-control validated) must pass at the byte ceiling.

## Serving

Needs a dflash build with qtype-106 down-expert (gumix surface=2) registration
and the mix-qtype monolithic-load fallback — branch `feat/qtype106-down-surface`
of `GeometricAGI/lucebox-hub`.

```
dflash_server ds4-0731-gu106down-v3.gguf --target-device hip:0 --max-ctx 8192
```

On Strix Halo use stock settings (auto-UMA on). If a discrete GPU is present,
mask it (`HIP_VISIBLE_DEVICES=<iGPU index>`) or pick the iGPU explicitly. A
healthy load prints `registered 101 qtype-106` and `registered 28 qtype-105`,
then serves with fused decode off.

---

<sub>The repo also carries `ds4-0731-gu106-v1.gguf` (+ its `.gumix.bin`
sidecar), an earlier 102.32 GB revision kept for reproducibility — superseded
by v3 on size, quality, and deployability.</sub>
