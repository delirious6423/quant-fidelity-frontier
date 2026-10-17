---
library_name: vllm
license: other
license_name: qwen-research
base_model:
- Qwen/Qwen3.8-Flash-Next
- halt95/Qwen3.8-Flash-Next-W4A16-Merlin
tags:
- vllm
- int4
- ple
- quantized
- base_model:quantized:Qwen/Qwen3.8-Flash-Next
- base_model:quantized:halt95/Qwen3.8-Flash-Next-W4A16-Merlin
---

# Qwen3.8-Flash-Next W4A16 + INT4-PLE + FP8-KV

**All credit for the underlying quantization work belongs to
[halt95](https://huggingface.co/halt95)**, author of the base
`Qwen3.8-Flash-Next-W4A16-Merlin` checkpoint. This repository is that
checkpoint repackaged with two additions: the 51B-parameter n-gram PLE
embedding table converted from FP8 to INT4, and a calibrated FP8 KV
cache. The W4A16 model body, the MTP head, and every other tensor are
halt95's work, carried over unchanged. If this model is useful to you, go like and
credit the original. For model capabilities and documentation see the
[base model card](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin).

The INT4 PLE repack halves the pinned-host table from 47.7 GB to
26.8 GB; serving with `VLLM_PLE_CPU_OFFLOAD=1` requires ~32 GB of free
host RAM. The FP8 KV cache roughly doubles the token pool per GiB of
VRAM and unlocks the model's native 262,144-token context.

>[!NOTE]
>To serve this model you **must** use a vLLM build that carries the
	int4 PLE plugin and the `qwen4_exp` model support. The
[`zankich/vllm`](https://github.com/zankich/vllm) fork integrates both,
	and [`zankich/vllm-openai`](https://hub.docker.com/r/zankich/vllm-openai/tags)
	is the prebuilt serve image from that fork. Stock vLLM has neither
	the architecture nor the plugin.

## Running the model

```bash
docker run -d --gpus all -p 8000:8000 \
  -v /path/to/Qwen3.8-Flash-Next-W4A16-Merlin-INT4PLE-FP8KV:/model:ro \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_QSA_KV_SCALES=/model/qsa_kv_scales_int4ple_2026-09-23_margin1.20.json \
  zankich/vllm-openai:latest \
  /model \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- `VLLM_PLE_CPU_OFFLOAD=1` is required: the 26.8 GB int4 PLE table is
  pinned in host RAM and looked up over UVA.
- `--tensor-parallel-size 4 --enable-expert-parallel`: the model serves
  across 4 GPUs with expert parallelism on the MoE layers.
- `--quantization compressed-tensors`: loads the W4A16 body.
- `--kv-cache-dtype fp8_e4m3` with `VLLM_QSA_KV_SCALES` pointing at the
  sidecar: activates the calibrated FP8 KV cache. The sidecar carries
  per-layer static K/V scales (12 QSA layers, calibration margin 1.20).
- Vision is supported; the multimodal path serves alongside text.
- The reasoning/tool parsers make the model's think tags and tool calls
  parse correctly in chat completions; the chat template ships inside
  the checkpoint.

Optional, see the fork's documentation: `--speculative-config` for MTP
speculative decoding, and the tiered KV offload connector
(`--kv-transfer-config` with `TieringOffloadingSpec`).

## Summary

| Property | Value |
|---|---|
| Base model | `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` (W4A16 GPTQ, compressed-tensors) |
| PLE n-gram table | INT4, group-32 symmetric, repacked from halt95's FP8 table |
| KV cache | FP8 E4M3, per-layer static scales in `qsa_kv_scales_int4ple_2026-09-23_margin1.20.json` (halt95's original calibration ships alongside as `qsa_kv_scales_262k.json`) |
| MTP head | bf16, unchanged |
| Size | 95 GB total, 40 files |
| Host memory | ~32 GB free RAM required for the pinned int4 table |

## INT4 PLE layout contract

`text_config.ple_embedding_dtype: "int4"`. Words: int32 `[rows, 20]`, 8
little-endian nibbles per word (element `i` of each 8-group in bits
`4i`). Codes stored bias+8: `value = (nibble - 8) * scale`. Scales:
fp16 `[rows, 5]`, one per group of 32 columns, with the FP8 source's
global scale folded in (decode approximates `fp8_source *
global_scale`). Round-trip residual bounded by half a quant step.
`ple-int4-manifest.json` records per-shard SHA-256 digests and the
packer's self-check residuals for the table repack.

## Measured quality

MMLU-Pro (lm-eval, chat template, 5-shot, 700 questions): **0.8486 ±
0.0133**. The published omlx 4-bit measurement of the base model is
0.843 on a 300-question subset. With the FP8 KV cache active:
**0.875 ± 0.031** (112-question deterministic subset, calibration
margin 1.20, clean tier) against 0.8839 for the same subset at the
default KV dtype — parity within noise.
