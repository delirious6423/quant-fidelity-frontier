---
license: mit
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: Avifenesh
tags:
  - nvfp4
  - fp4
  - quantized
  - mxfp4
  - moe
  - agentic
  - conversational
  - speculative-decoding
  - dspark
  - blackwell
  - deepseek_v4
---

# DeepSeek-V4-Flash-0731 — NVFP4 (tiyuvta serving artifact)

NVFP4 quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) —
the official 0731 release (284B MoE / 13B active, 1M context, DSpark speculative-decoding
module included, MIT). **Lossless cast**: 0731's routed experts ship MXFP4-QAT from
DeepSeek; this artifact re-encodes their block scales into the modelopt NVFP4 layout
(per-16 E4M3 × F32 `weight_scale_2`) with **bit-identical dequantized values, verified on
every tensor** — 33,024/33,024 expert projections, 8,657,043,456/8,657,043,456 source
per-32 blocks (= 17,314,086,912 output per-16 blocks) decode bit-equal to the source
(per-tensor cast log with sha256 of every decoded tensor in this repo:
`cast_mxfp4_to_nvfp4.log`). Attention/shared/dense tensors and the DSpark drafter are
**byte-identical** to the source. First public NVFP4 of the 0731 checkpoint.

## Provenance

| | |
|---|---|
| Base | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (MIT) |
| Operation | MXFP4 → NVFP4 lossless scale re-encode, trunk routed experts only (exclude: `*.attn.*`, `*.ffn.shared_experts.*`, `head`, `mtp.*`) |
| Tooling | `dsv4_mint`, the tiyuvta serving engine's mint tool (CPU-only), commit `46b9c97364`; the recipe mirrors [nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)'s cast of the preview checkpoint |
| Verified | source shards sha256 == HF LFS oids at the pinned revision; full tensor census on source (72,317) and mint (105,341) — names/dtypes/shapes/bytes derived from config math, zero mismatches; per-tensor dequant **bit-identity** vs source on all 33,024 cast projections (refusal on the first non-representable value — none occurred); pass-through tensors byte-compared |
| Reference code | `inference/` (incl. `kernel.py`) + `encoding/` + tokenizer + `generation_config.json` + `LICENSE` shipped **unmodified** from the 0731 repo |

### Why this cast is exactly lossless

MXFP4 stores one E8M0 scale per 32 weights: `s = 2^(b-127)`, always a power of two.
NVFP4 stores `e4m3(scale) × f32 weight_scale_2` per 16 weights. Choosing
`weight_scale_2 = 2^(floor(log2(max s)) - 8)` per tensor puts every quotient
`s / weight_scale_2` at an exact power of two inside e4m3's exactly-representable range;
the e2m1 weight codes are copied verbatim (each group of 32 becomes two identical groups
of 16). Zero rounding anywhere; the mint refuses (rather than rounds) on any value that
cannot be represented exactly. This is the same operation NVIDIA's modelopt performed on
the preview checkpoint (their `cast_mxfp4_to_nvfp4.log`: 17.7e9/17.7e9 blocks lossless).

### `input_scale` is intentionally absent

NVIDIA's preview artifact carries a calibrated F32 `input_scale` per expert projection
(measured: 814 distinct values across 33,024 tensors — activation statistics, not a
constant). A lossless cast has no calibration step, so this artifact **omits**
`input_scale`. Engines that quantize activations dynamically (the reference
`inference/kernel.py` law — per-128 dynamic act quant — and the vLLM/SGLang NVFP4 paths
for this family) do not read it. Static-activation paths (e.g. some TRT-LLM configs)
would need calibration — use NVIDIA's preview artifact or calibrate downstream if that
is your deployment.

## What's inside / how it differs from the base

| class | container | vs base |
|---|---|---|
| trunk routed experts (43 layers × 256 experts × w1/w2/w3) | NVFP4: `weight` U8 e2m1-pairs, `weight_scale` F8_E4M3 per-16, `weight_scale_2` F32 | scales re-encoded (bit-identical decode); weight bytes verbatim |
| DSpark drafter (`mtp.0..2`, 3 fused MoE blocks incl. 768 experts) | OCP MXFP4 + FP8 + BF16, as shipped | byte-identical |
| attention / shared experts / indexer linears | FP8 E4M3 + 128×128 UE8M0 block scales | byte-identical |
| embed / head / norms / hc / router / tid2eid | BF16 / F32 / I64 | byte-identical |

72,317 → 105,341 tensors (each cast projection gains `weight_scale_2`; `scale` becomes
`weight_scale`); 48 shards, same grouping as the base; `model.safetensors.index.json`
regenerated with exact sizes; `hf_quant_config.json` added (modelopt schema — NVFP4 /
group 16 on `layers.*.ffn.experts`, exclude list above).

## How to run

These are the base repo's official engine commands with the model id swapped. The
artifact replicates the NVFP4 layout of nvidia's preview artifact (same
`hf_quant_config.json` schema, same tensor recipe, minus the calibrated `input_scale`).
We have not yet gated engine loads of this artifact ourselves — validate on your stack
before production use.

vLLM (DSpark speculative decoding is one flag):

```bash
vllm serve tiyuvta/DeepSeek-V4-Flash-0731-NVFP4 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

SGLang: `--speculative-algorithm DSPARK` (target and draft weights both come from this
checkpoint — do not set a separate draft model path).

llama.cpp: consumes the drafter as a separate extracted GGUF — see
[am17an/DeepseekV4-Flash-20260731-DSpark](https://huggingface.co/am17an/DeepseekV4-Flash-20260731-DSpark)
(it is the `mtp.*` subtree of this same checkpoint, which is byte-identical here).

## DSpark drafter

The drafter is **fused in-repo** under `mtp.*` (three full window-attention MoE blocks,
`dspark_block_size 5`, rank-256 markov head, confidence head) and is byte-identical to
the base checkpoint — no separate drafter repo needed. Wiring per engine above.

## Chat template / reasoning effort

No Jinja template; the `encoding/` folder (shipped unmodified) is the official
message-encoding reference. 0731's `reasoning_effort` ladder is `low` / `high` / `max`
(default `low` = no prefix — note this differs from the preview, where `high` was the
no-op). Sampling per DeepSeek's evals: `temperature 1.0, top_p 0.95`.

## Evaluation

Quality numbers below are **DeepSeek's own published table for the base checkpoint**
(README of deepseek-ai/DeepSeek-V4-Flash-0731; agentic tasks, `max` reasoning effort,
temperature 1.0, top_p 0.95). This artifact's dequantized weights are bit-identical to
that checkpoint's, so the quantization adds zero eval delta by construction; we ran no
re-evals and publish none.

| Benchmark | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash (Preview) | DeepSeek-V4-Pro (Preview) |
| :--- | :---: | :---: | :---: |
| Terminal Bench 2.1 | 82.7 | 61.8 | 72.1 |
| NL2Repo | 54.2 | 39.4 | 38.5 |
| Cybergym | 76.7 | 38.7 | 52.7 |
| DeepSWE | 54.4 | 7.3 | 12.8 |
| Toolathlon-Verified | 70.3 | 49.7 | 55.9 |
| Agents' Last Exam | 25.2 | 15.8 | 16.5 |
| AutomationBench Public | 25.1 | 10.8 | 12.8 |

(Do not mix these numbers with preview-lineage artifacts: nvidia's NVFP4 quantizes the
*preview* checkpoint, whose scores are the middle column.)

Our own verification gates on this artifact (receipts in `cast_mxfp4_to_nvfp4.log` and
`sha256-manifest.txt`):

- source census: 72,317/72,317 tensors match the config-derived expectation, zero extras
- cast: 33,024/33,024 projections, 8,657,043,456/8,657,043,456 blocks decode
  bit-identical to source (E8M0 NaN scan over all 35,328 expert scale tensors: zero)
- mint census: 105,341/105,341 tensors, `hf_quant_config.json` declaration matches
- every non-cast tensor byte-identical to source

## sha256 manifest

`sha256-manifest.txt` in this repo lists sha256 for all 48 output shards plus the
ancillary files.

## License

MIT, inherited from the base model —
[LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE)
(copied verbatim into this repo). This is a derivative work of a third-party model;
DeepSeek-V4-Flash-0731 is a model by DeepSeek-AI.

## Citations

- DeepSeek-V4 Technical Report: [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- DSpark (speculative decoding module): [arXiv:2607.05147](https://arxiv.org/abs/2607.05147)
