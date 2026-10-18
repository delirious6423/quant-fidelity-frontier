---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: transformers
tags: [auto-round, mxfp4, engram, model-free, deepseek]
---

# DeepSeek-V4.1-Flash — D1 (Engram tables FP8→MXFP4), auto-round one-command

Same artifact class as `DeepSeek-V4.1-Flash-D1-engram-mxfp4` (only the two engram
embedding tables requantized FP8→MXFP4), but produced **end-to-end by a single
auto-round `--model_free` command** (with the DeepSeek-V4.1 source-format patch set).

Checkpoint size: **412.0 GB** (vs 510.3 GB official, -19.3%). All 96,081 non-engram
tensors verified **byte-identical** to the official checkpoint; engram tables are
int8-packed E2M1 `[R,128]` + E8M0 `[R,8]` under their original names, rel_err ≈ 0.12
vs the FP8 source (4-bit E2M1 theoretical band).

## Quantization command

```bash
AR_MODEL_FREE_SHARD_PARALLELISM=1 \
auto-round --model_name deepseek-ai/DeepSeek-V4.1-Flash \
  --model_free --scheme BF16 \
  --layer_config '{
    "engram.embed":    {"bits": 4, "data_type": "mx_fp4", "group_size": 32},
    "indexer.wq_b":    {"bits": 8, "data_type": "mx_fp8", "group_size": 32},
    "compressor":      {"bits": 16},
    "indexer":         {"bits": 16},
    "shared_experts":  {"bits": 8, "data_type": "mx_fp8", "group_size": 32},
    "ffn.experts":     {"bits": 4, "data_type": "mx_fp4", "group_size": 32},
    "attn":            {"bits": 8, "data_type": "mx_fp8", "group_size": 32},
    "engram.wkv":      {"bits": 8, "data_type": "mx_fp8", "group_size": 32},
    "main_proj":       {"bits": 8, "data_type": "mx_fp8", "group_size": 32}}' \
  --output_dir <out>
# afterwards: restore the official config.json (this repo's config.json is already official)
```

Notes on the recipe (learned the hard way):
- `BF16` default keeps every unlisted family in full precision; FP8/MXFP4 source families
  listed with their *same* target format pass through **byte-identically** (no requant).
- `layer_config` patterns are substring matches, first match wins: `ffn.experts` (not
  `experts`) must be used for routed experts so it cannot shadow `shared_experts`;
  `compressor`/`indexer` are BF16 and must be pinned to `bits:16` before `attn`.
- `AR_MODEL_FREE_SHARD_PARALLELISM=1` is required for this model: parallel shard workers
  share one GPU and can OOM the giant-table (384M-row) quantization worker.

## Inference (vLLM)

Identical to `DeepSeek-V4.1-Flash-D1-engram-mxfp4`: vLLM `main` (≥ PR #56201) +
`dsv41-quant-plugin` engram patch, official `config.json` (FP8 declaration) kept:

```bash
DSV41_ENGRAM_DTYPE=fp4 NCCL_NVLS_ENABLE=0 \
vllm serve <this-model-dir> \
  --tensor-parallel-size 4 --max-model-len 8192 --language-model-only \
  --gpu-memory-utilization 0.90 --port 8100
```

## Provenance

- auto-round: patched copy at main `@141e4c99` (+270 lines in
  `auto_round/utils/model_free_utils.py`: V4.1 source normalize/restore, blockwise FP8
  passthrough, chunked MXFP for giant tables, `should_skip` yields to explicit
  `layer_config`).
- vLLM main nightly (`0.1.1.dev39+g46d2b23ac`).
- Reference accuracy (same format, script-produced twin): gsm8k 92.19/92.27 vs 92.87
  baseline (n=1319, raw 5-shot, thinking off).
