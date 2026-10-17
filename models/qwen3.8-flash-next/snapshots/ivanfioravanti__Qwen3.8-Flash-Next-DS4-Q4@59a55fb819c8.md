---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- ds4
- apple-metal
- quantized
- qwen
- imatrix
---

# Qwen3.8-Flash-Next DS4 Q4 (imatrix)

DS4 quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
built from the official BF16 checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` with an importance-matrix
calibrated Q4_K routed-expert recipe.

The current release is the **qwen4exp-schema** structure: standard llama.cpp
GGUF tensor naming that the DS4 engine (Metal graph) loads directly, with the
51B-value PLE n-gram table kept in the external sidecar so it never has to be
memory-resident.

The main GGUF includes the 48 trunk layers and one **embedded MTP block**.
This is the only model version distributed here: use the same file for
ordinary decoding or enable speculative decoding with `--mtp`. The external
PLE table is required in both cases.

## Quantization recipe

| tensor group | format |
|---|---|
| routed expert gate/up (48 trunk layers × 512 experts) | **Q4_K, imatrix-quantized** |
| routed expert down | **MXFP4** (640-wide rows cannot take a 256-block type; the same down treatment as the canonical Q4K recipe) |
| dense GDN/QSA projections, shared experts, output | Q8_0 |
| embedding, router, hyper-connections, norms, control tensors | BF16 / F32 |
| PLE n-gram rows (external sidecar, 51B values) | Q4_1, read on demand from the sidecar file — never resident |

The importance matrix is the pinned `unsloth/Qwen3.8-Flash-Next-GGUF`
calibration (`revision c8b5954a88c2775c54692593eda40ea041d3176`, imatrix
SHA-256 `a5863123db1ca458727e738955befb7bfc199520aa2bee3a30142a1aff925415d`,
dataset `unsloth_calibration_Qwen3.8-Flash-Next.txt`, 45 chunks of 18432).
24 zero-count part-experts (16 gate/up, 8 down) use the deterministic
per-expert input-column weight-energy fallback; all other entries are
normalized per expert.

Note: the Hub sidebar's auto-detected quantization label keys off the PLE
sidecar (genuinely Q4_1); the routed experts are Q4_K/MXFP4 as listed above.

## Files

| file | size | role |
|---|---|---|
| `Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf` | 74.9 GB (69.7 GiB) | **main model with embedded MTP**; supports both ordinary and speculative decoding |
| `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` | 32.0 GB (29.8 GiB) | required external PLE table (keep on a fast local SSD) |

## Running

Requires the DS4 runtime from
[`ivanfioravanti/ds4`](https://github.com/ivanfioravanti/ds4),
branch `qwen3.8-flash-next` (`make`, run from the repo root).

```sh
# Download the main model and the required PLE table.
QWEN_DS4_DIR="$HOME/models/qwen38-ds4-q4"
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 \
  --local-dir "$QWEN_DS4_DIR"

# Speculative decoding using the MTP block inside the main GGUF.
./ds4 \
  -m "$QWEN_DS4_DIR/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf" \
  --ple "$QWEN_DS4_DIR/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
  --mtp --metal --power 100 --mtp-draft 7 --mtp-timing

# Ordinary decoding uses the same GGUF, with MTP disabled.
./ds4 \
  -m "$QWEN_DS4_DIR/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf" \
  --ple "$QWEN_DS4_DIR/Qwen3.8-Flash-Next-PLE-Q4_1.gguf" \
  --metal --power 100
```

Pass the MTP-inclusive GGUF as `-m` / `--model` and enable its embedded
block with `--mtp`. Do not pass this file through `--mtp-model`: that
external-support option is not supported for Qwen3.8.

`--mtp-timing` reports draft acceptance. The current Qwen graph uses one
draft token per verification cycle; `--mtp-draft 7` is accepted but does
not enable seven-token draft chains. Add `--temp 0` for greedy decoding,
or `--mtp-exact-sampling` to preserve the target distribution when sampling.

`--ple` is required because the main model carries no inline n-gram table.
The sidecar mapping is CPU-only and never enters Metal residency.
The conversion tool that produced these files from the fast-pack artifacts
is `gguf-tools/qwen4_pack_to_qwen4exp.py` in the same repository
(`--ple-external`, `--mtp`).

## Requirements

- Apple Silicon with Metal (developed and measured on an M3 Ultra, 512 GB).
- Resident model weights occupy approximately **69.7 GiB**, including MTP,
  plus the context-dependent KV cache and graph buffers. Use a 128 GB or
  larger Mac for headroom.
- The PLE sidecar stays on disk and is demand-paged (32.0 GB / 29.8 GiB, fast local
  SSD recommended).

## Measured on the reference M3 Ultra

| context | prefill | plain decode | MTP decode |
|---:|---:|---:|---:|
| 4K–16K | 1071–1171 tok/s | 45.4–45.6 tok/s | 55–56 tok/s |
| 64K | 1158 tok/s | 44.8 tok/s | 51.7 tok/s |
| 128K | 1136 tok/s | 44.1 tok/s | 53.9 tok/s |
| 262K (native window) | 1063 tok/s | 42.2 tok/s | 45.2 tok/s |

Short-context structured prompts (e.g. math) reach ~66 tok/s with MTP at
~98% draft acceptance; the numbers above are prose-continuation workloads.
MTP preserves exact sampling with `--mtp-exact-sampling`.

## Quality

Scored against an exact-checkpoint local BF16 reference (100-case fixture,
24-token greedy continuations, top-20 logprobs; 85 tokenization-aligned
cases): target-logprob MAE **0.0431**, greedy first-token match 74/85,
top-1 agreement 96.6%, average target NLL 0.213. For reference on the same
harness, the ggml-org Q8_0 build scores MAE 0.0211 / top-1 98.7%, and the
previous Q4_0-routed release scored MAE 0.0430 / 96.5% — the imatrix Q4_K
recipe buys back the down-projection precision budget as MXFP4 while
holding overall drift flat. Methodology lives in the repository's
`QWEN38_FLASH_NEXT.md` and `OUT/qwen38-eval/` reports.

## Attribution and license

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(thanks to the Qwen team), with the imatrix calibration from
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
and the DS4 inference engine by the
[ds4](https://github.com/ivanfioravanti/ds4) contributors (engine:
Tarjei Mandt). This derivative inherits the **Qwen Community License 1.0**
of the base model — the full text is in [`LICENSE`](LICENSE) and the usual
license conditions (attribution, branding, and the MaaS / AI Work Assistant
clauses) apply as they do to the original.
