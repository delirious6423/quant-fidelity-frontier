---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
pipeline_tag: text-generation
library_name: vllm
tags:
  - deepseek
  - int4
  - w4a16
  - compressed-tensors
  - rocm
  - gfx1030
---

# DeepSeek-V4-Flash-0731 INT4-W4A16

> **Work in progress.** This card and the checkpoint are still being validated.

Routed experts of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
transcoded from FP4 to INT4, for AMD RDNA2 (gfx1030), which has no FP4 or FP8
conversion instructions. 169.9 GiB across 34 shards.

## Requirements

**This checkpoint does not run on stock vLLM.** It needs a vLLM build carrying
the RDNA2 DeepSeek-V4 patches, and `--dtype float16`.

Target configuration is 8 gfx1030 dies across 2 nodes (PP=2 x TP=4):

```bash
vllm serve yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code
```

`--kv-cache-dtype` may be omitted; it resolves to `fp8_ds_mla`, the only layout
these backends implement.

## Quantization

| | |
|---|---|
| Method | compressed-tensors, `pack-quantized` |
| Scheme | INT4 symmetric, group size 32, weight-only (W4A16) |
| Scope | routed experts only (`w1`/`w2`/`w3`/`gate_proj`/`up_proj`/`down_proj`/`gate_up_proj`) |
| Untouched | attention, shared experts, router gates, norms, embeddings, `lm_head`, MTP heads |

The source weights are already 4-bit (FP4 e2m1 with UE8M0 block scales), so this
is a transcode rather than a fresh quantization: each expert is dequantized
bit-exactly and re-encoded as symmetric INT4. Tensors stored as FP8 elsewhere in
the model are converted to fp16 bit-exactly.

## Fidelity

Every one of the 35,328 transcoded expert tensors was compared against the
dequantized source:

| | SNR (dB) |
|---|---|
| minimum | 22.71 |
| 1st percentile | 23.15 |
| median | 25.69 |

No range violations. All 390 FP8 to fp16 conversions are bit-exact.

## Verification

Run end to end on 4 gfx1030 dies (one Radeon Pro W6800X Duo pair, 128 GiB) with
CPU offload, `--dtype float16`, TP=4:

- Coherent greedy generation and correct answers through the chat template.
- Perplexity 2.42 (prose) and 1.72 (code) on held-out text.
- 3,011-token prompt exercising the sparse attention indexer, summarized correctly.

The 8-die, 2-node configuration above is the deployment target and has not been
run yet.

## Limitations

- fp16 only; bf16 is untested here and the RDNA2 kernels this targets are fp16.
- The router, shared experts and attention stay at full precision, so the memory
  saving is confined to the routed experts.
- Quality has been checked for coherence and perplexity, not with a benchmark
  suite; treat published scores for the base model as an upper bound.
