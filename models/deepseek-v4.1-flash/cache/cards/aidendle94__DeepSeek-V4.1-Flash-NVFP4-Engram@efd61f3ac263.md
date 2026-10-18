---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - deepseek
  - engram
  - nvfp4
  - vllm
  - dgx-spark
library_name: vllm
---

# DeepSeek-V4.1-Flash — NVFP4 Engram variant

DeepSeek-V4.1-Flash with its two Engram n-gram tables (layers 1 and 14, 384M rows × 256 each) re-quantized from
fp8 e4m3 + UE8M0 block-32 scales to **NVFP4**: e2m1 values packed two per byte, one fp8 e4m3 scale per 16 elements and one
fp32 global scale per table (`value = e2m1 × e4m3_scale × global`, `global = amax / (448 × 6)`, the ModelOpt convention).
Every other tensor is byte-identical to the original checkpoint. The tables shrink from 203 GB to 111 GB
(the whole repository from ~510 GB to ~418 GB).

| | fp8 Engram (original) | NVFP4 Engram (this repo) |
|---|---|---|
| `layers.{1,14}.engram.embed.weight` | F8_E4M3 [rows, 256] | U8 [rows, 128] (e2m1 pairs, low nibble = even element) |
| `layers.{1,14}.engram.embed.scale` | F8_E8M0 [rows, 8] (per 32) | F8_E4M3 [rows, 16] (per 16) |
| global scale | — | `config.json: engram_quant_global_scale[layer]` |
| `config.json` | — | `engram_quant: "nvfp4"`, `engram_quant_block_size: 16` |

## Quantization report

| table | rows | tensor amax | global scale | relative RMS error vs fp8 | scale underflows | shard size |
|---|---|---|---|---|---|---|
| layer 1 | 384,006,168 | 11.0 | 0.004092 | **8.99 %** (first upload: 9.49 %) | 0 of 6,144,098,688 blocks | 55.45 GB (was 101.4 GB) |
| layer 14 | 384,016,682 | 24.0 | 0.008929 | **9.06 %** (first upload: 9.64 %) | 0 of 6,144,266,912 blocks | 55.46 GB (was 101.4 GB) |

Relative RMS error is `‖deq − fp8‖₂ / ‖fp8‖₂` over the whole table (fp8 rows dequantized with their UE8M0 scales as the
reference). Quantization ran on one GB10 (2M rows per step; ~14 min per table with round-to-nearest, ~48 min with the per-column GPTQ feedback used for this upload — the tool now has the lazy-batch update).

## Running it

The variant needs a runtime that reads the NVFP4 row format. Our vLLM stack for 4× DGX Spark does
(`aidendle94/sparkrun-vllm-dsv41-gb10:production-1.1` plus the `dsv41-dcp2` patch set stage 10, Engram tables on disk with
`DSV41_ENGRAM_DISK=1`); the recipe, launcher and patch scripts are described in the blog post “Serving DeepSeek-V4.1-Flash on 4× DGX Spark” (aidendle94). Row decode is a
16-entry lookup times the two scales, done on the CPU staging path exactly like the fp8 rows; nothing else in the model
changes. The stock HBM-resident Engram path does not read this format.

## Measured (4× DGX Spark, vLLM, 2026-09-16)

Served with the production configuration (TP4, DCP4, DSpark k=5, 8K prefill chunks, 6 GiB KV per rank, 500K context) and
the variant's rows read straight from the checkpoint shards (over NFS on three of the four nodes for this test):

| gate | fp8 Engram (production) | NVFP4 Engram (this repo) |
|---|---|---|
| teacher-forced scoring of 40 fp8-model completions (8,345 tokens) | mean logprob −0.112 · argmax 97.5 % (the model scoring its own outputs) | **−0.111 · 97.5 %** (GPTQ+MSE); −0.110 · 97.5 % (first upload, RTN) |
| needle at 32K / 128K | found / found | found / found (20.7 s / 76.3 s to first token in the test window) |
| tool call, thinking mode | pass | pass |
| 4-way soak | clean | clean |
| decode counting / code / prose | 90.7 / 63.5 / 31.3 tok/s | 87.3 / 66.4 / 27.1 tok/s (rows over NFS in this test) |
| KV pool | 3,418,145 tokens | 3,418,145 tokens |

The 9.5 % relative RMS error at row level does not reach the next-token distribution: agreement with the fp8 model is
identical to the fp8 model's own batch-noise floor. The decode difference in this table is the test setup (Engram rows
fetched over NFS instead of node-local NVMe), not the format; row bytes per lookup halve.


## Rounding method: GPTQ with an MSE block scale

The first upload used plain round-to-nearest with the block's maximum mapped to 6 ("amax" scale). The current shards use:

- **per-block MSE scale search** — for every block of 16 the scale is the best of 8 candidates in `[0.75, 1] × amax/6`
  (after e4m3 rounding), chosen by the block's squared error; about a third of the blocks pick a smaller scale;
- **GPTQ / OBQ error feedback** with the Hessian proxy of the Engram's output projection: rows are consumed only through
  `wkv`, so for a row of hash column `c` the error that reaches the model is `‖W_c e‖`, with `W_c` the 256-column slice of
  `wkv`. Columns are quantized in order with the residual fed forward through the upper Cholesky factor of
  `(W_cᵀ W_c + 0.01·mean(diag)·I)⁻¹` (one factor per hash column, 24 per layer, no calibration data).

Offline comparison on 300K sampled rows of layer 1 (`tools/nvfp4_gptq_study.py`):

| method | error through `wkv` | raw RMS |
|---|---|---|
| RTN, amax scale (first upload) | 8.96 % | 9.49 % |
| RTN, MSE scale | 8.40 % | 8.88 % |
| GPTQ, amax scale | 8.80 % | 9.63 % |
| **GPTQ, MSE scale (current)** | **8.23 %** | 8.99 % |

`wkv`'s per-column Gram matrices are nearly isotropic (median max/min diagonal ratio 1.2), which is why error feedback adds
little on top of the scale search; the remaining error is the e2m1 grid itself. Full-table numbers are in
`engram-nvfp4.json` (`method`, `rel_rms_error`).

## How it was made

`tools/engram_nvfp4.py` (in the recipe repository): two passes per table on one GB10 — pass 1 finds the tensor amax,
pass 2 quantizes 2M rows at a time (fp8 → fp32 → per-16 amax → e4m3 scale → e2m1 round-to-nearest → pack) and writes the
new shard, then copies the shard's other tensors verbatim, writes the index and config, and links the untouched files.
Quantization error and scale-underflow counts are in `engram-nvfp4.json`.

## Credits

Base model: DeepSeek-AI. Engram-on-disk serving: tonyd2wild and Kai. Variant: Aiden Le.
