---
license: mit
base_model: zai-org/GLM-5.3-Flash
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - oq
  - oq2e
  - quantized
  - 2-bit
  - moe
  - mtp
  - imatrix
  - apple-silicon
  - glm
---

# GLM-5.3-Flash-oQ2e-mtp (MLX)

2-bit **imatrix-calibrated** oQ quantization (`oQ2e`) of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), built locally on a
Mac Studio M4 Max (128 GB) with [oMLX](https://github.com/jundot/omlx)'s oQ quantizer. The
MTP (nextn) head is preserved **and calibrated** like the backbone.

| | |
|---|---|
| Source | `zai-org/GLM-5.3-Flash` (FP8, e4m3, 128x128 block scales), 328.37 GB |
| Format | MLX safetensors, `model_type: glm5_next`, 3057 tensors |
| Size on disk | 110.3 GB (103 GiB), 21 shards |
| Level | oQ level 2, **enhanced (oQe)** — imatrix-calibrated, target 2.8 bpw / hard cap 3.0 |
| Non-quantized | `bfloat16` |
| MTP head | preserved, 59 tensors (`language_model.mtp.0.*`), with imatrix entries |
| Built | 2026-10-01, 33.7 min wall clock, one calibration round, **no calibration proxy** |

## Quantization layout (actual per-tensor config)

| Bit width | Scope |
|---|---|
| **2 bit** (default, `group_size 64`, `affine`) | trunk routed experts (`mlp.switch_mlp.*`, 288 experts / top-8, layers 3-44) |
| **4 bit** (3 tensors) | the MTP head's routed experts (`mtp.0.block.mlp.switch_mlp.{gate,up,down}_proj`) |
| **8 bit** (554 tensors) | attention projections incl. indexer + `embed_q`/`unembed_out` (`q/k/v/o_proj`, `q_a/q_b`, `kv_a_proj_with_mqa`, `b_proj`, `g_a/g_b`, `forget_gate.*`, `indexer.*`), `shared_experts.*`, dense layers 0-2, `lm_head`, `embed_tokens` |

So the routed experts carry the 2-bit budget (that is where oQ2 spends it) while attention,
shared experts and the head stay protected — plus the MTP head's experts are boosted to
4 bit, which is cheap because the head is a single layer.

## Calibration (oQe imatrix)

| | |
|---|---|
| Dataset | oMLX `oqe_code_multilingual` (code/en/zh/ja/ko/tool-calling/reasoning) |
| Budget | 128 samples x 512 tokens, micro-batch 12, **1 round** (coverage sufficient) |
| Collection | **streamed** layer-by-layer from the FP8 checkpoint (`load_kind: streaming`) |
| Entries applied | 682 (`oq_imatrix_report.json`: 671 entries) |
| Uncalibrated | **none** (`missing: []`, `mismatched: []`) |
| Expert coverage | 38 688 / 38 688 active, 0 zero-count, min_count 35 (required >= 16) |

The imatrix is collected from the real FP8 weights. A uniform 4-bit calibration proxy
(~181 GB resident) does not fit a 128 GB machine (`0.75 * capacity` = 92.8 GB), which is why
the streaming path was needed — see [jundot/omlx#4173](https://github.com/jundot/omlx/issues/4173)
and [jundot/omlx#4174](https://github.com/jundot/omlx/pull/4174) (CI green; it also fixes two
glm5_next MTP-head/imatrix bugs found while verifying: the resident head pass is skipped for
this family, and `embed_q` capture is lost for prefixed collector installs).

## Verification

* loads on a 128 GB M4 Max with MoE expert offload:
  `wrapped 42 layers at 82.5% residency (expert tables: 95.13 GB total, 78.61 GB resident)`,
  `actual: 87.43 GB` (fully resident without offload: ~103 GB)
* MTP active in generation (`MTP[0] ... accept=8/9 (88.9%)` on a short probe)
* clean output/thinking split (`finish=stop`, `content='17 x 23 = 391'`, thinking in
  `reasoning_content`), `cached_tokens: 0` on the first run
* no keys in the wrong namespace and no loader shape errors (the `preserve_mtp` regressions
  reported in jundot/omlx#3845 are not present in this build)

No systematic benchmark was run. Short probes on the machine above measured low-to-mid tens
of tokens/s depending on prompt and warm-up; prefill is dominated by reading the ~95 GB of
expert tables from disk (offload on).

## Usage (oMLX)

Place this folder in your oMLX model directory, reload the model list, and set the per-model
settings that GLM-5.3-Flash needs on a <=128 GB machine:

```
moe_expert_offload_enabled = true
moe_expert_offload_resident_fraction = 0.825
mtp_enabled = true
max_context_window = 262144
model_type_override = "vlm"
```

Without the offload setting the model loads fully resident (~103 GB), leaving no KV headroom
on a 128 GB machine.

## How it was built

```
quantize_oq_streaming(
    model_path=<zai-org/GLM-5.3-Flash, FP8>,
    output_path=.../GLM-5.3-Flash-oQ2e-mtp,
    oq_level=2, group_size=64, dtype="bfloat16",
    enhanced=True,             # oQe imatrix
    preserve_mtp=True,         # keep + calibrate the nextn head
    stream_calibration=True,   # layer-by-layer instead of the RAM proxy
    imatrix_num_samples=128, imatrix_seq_length=512,
)
```

An already-quantized MLX/GGUF build cannot be used as an oQ source — `validate_quantizable()`
accepts only native FP8/MXFP8 (or unquantized) checkpoints, so the official FP8 release is
the source.

## Provenance and license

* Base model: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT —
  this derivative is published under the same license, with attribution to zai-org.
* Quantized with oMLX (`omlx/oq.py`); the streaming calibration used here is in
  [jundot/omlx#4174](https://github.com/jundot/omlx/pull/4174).
* `oq_imatrix_report.json` is the calibration report of this exact run (entries, coverage,
  sample budget, load kind).

*Weights are quantized, not retrained: quality tracks the base model within the usual 2-bit /
target-2.8-bpw limits. Test before production use.*
