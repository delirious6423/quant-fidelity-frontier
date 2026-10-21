---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
tags:
- compressed-tensors
- quantized
- nvfp4
- vllm
- moe
- multimodal
pipeline_tag: image-text-to-text
library_name: vllm
---

# GLM-5.3-Flash — NVFP4 (routed experts)

NVFP4 on the routed experts of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), in
compressed-tensors format, verified end to end on 4x NVIDIA H100 80 GB.

**191.0 GiB** on disk, down from **305.8 GiB** — the difference between a model that needs
five 80 GB GPUs and one that fits on three.

## What is quantized

| modules | count | scheme |
|---|---|---|
| `mlp.experts.*.{gate,up,down}_proj`, layers 3-44 | 36 288 | NVFP4 — FP4 E2M1 weights, an FP8 E4M3 scale per 16 values, one FP32 global scale per module |
| shared experts, the three dense MLP blocks, MLA projections, KDA linear-attention projections, the DSA indexer, `mlp.gate`, the vision tower, embeddings, `lm_head` | 764 | bf16 |

The layer selection mirrors [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4),
including leaving the MTP block (layer 45) alone.

**Weight-only (NVFP4A16).** There is no `input_global_scale` in this checkpoint. Calibrating one
means running the model, and no released `transformers` implements `Glm5NextForConditionalGeneration`
— so a locally calibrated activation scale was not obtainable. vLLM handles this explicitly:
`CompressedTensorsW4A4Nvfp4MoEMethod(..., use_a16=(input_quant is None))`. On H100 it makes no
observable difference, because vLLM's MoE oracle falls through the SM100-only FlashInfer/CUTLASS
kernels to Marlin and runs any NVFP4 MoE as W4A16 regardless. On Blackwell this build runs W4A16
where a calibrated one would run W4A4.

## The source is already FP8

`zai-org/GLM-5.3-Flash` ships quantized: the routed experts, shared experts, dense MLP blocks and MLA projections
carry a DeepSeek-style 128x128 block `weight_scale_inv`, while the KDA linear-attention paths, the
indexer and the vision tower are bf16. This build folds those block scales back into the weights and
re-quantizes the experts from bf16; everything else is written out as plain bf16.

## Verified against the reference conversion

Quantizing the same source tensors and comparing to RedHatAI's published checkpoint:

| module | packed bytes identical | cos(this, source) | cos(RedHat, source) |
|---|---|---|---|
| `layers.3.mlp.experts.0.gate_proj` | 96.30 % | 0.995908 | 0.995942 |
| `layers.4.mlp.experts.7.down_proj` | 100.00 % | 0.995730 | 0.995730 |

The 100 % byte match confirms the nibble packing order, the block-scale layout and the global-scale
convention. The residual 3.7 % on the other module are rounding-boundary flips from a global scale
that differs by 0.45 %; reconstruction fidelity is identical.

## Why there is no FP8 build alongside this one

The upstream release is already FP8, and already in the better configuration: `quant_method: fp8`,
`activation_scheme: **dynamic**`. Of its 305.8 GiB, **292.8 GiB are FP8 weights**, 0.1 GiB are the
128x128 block scales, and only **12.9 GiB are bf16** — the KDA linear-attention projections, the DSA
indexer, the vision tower and the embeddings. Quantizing that remainder too would save 6.5 GiB (2 %),
would not improve the activation scaling because it is already per-token dynamic, and would still not
fit on four 80 GB GPUs. NVFP4 is the conversion that changes what this model runs on.

## Measured on 4x H100 80 GB

| build | size | GSM8K chat | AI2D vision | GSM8K lm-eval | ARC-C | tok/s @1 | tok/s @32 |
|---|---|---|---|---|---|---|---|
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (source, FP8) | 305.8 GiB | 98.8 | n/a | n/a | n/a | n/a | n/a |
| **this build** (NVFP4) | 191.0 GiB | 98.4 | 84.6 | 90.4 | 69.2 | 136 | 1509 |

The source does not fit on four 80 GB GPUs (328 GB of weights). Its GSM8K row was measured with
vLLM's per-GPU CPU offload; the wall clock there was 637 s against this build's 73 s, so its
throughput is not comparable and no AI2D number is given — at 28 prompt tokens/s under offload the
500-image run would have taken over 14 hours.

Serving the source across two nodes to get a fair comparison does not work with the image the model
ships: `--tensor-parallel-size 8` spanning both nodes hangs after the Ray placement group is created,
and `--pipeline-parallel-size 2` fails weight loading with
`KeyError: 'layers.11.self_attn.fused_qkv_a_proj.weight'` — `Glm5Next` has the pipeline-parallel
scaffolding (`make_layers`, `IntermediateTensors`) but its `load_weights` does not filter by pipeline
stage. The Ray cluster itself was fine: 8 GPUs across two nodes over 400 Gb/s NDR InfiniBand.

Vision was checked qualitatively instead: both the source and this build describe the same two test
images correctly and in comparable detail.

## Serving

`Glm5NextForConditionalGeneration` is not in any released vLLM or transformers. Use the image the
model's authors ship:

```bash
docker run --gpus all -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm/vllm-openai:glm53-flash mbehr90/GLM-5.3-Flash-nvfp4 \
    --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.85
```

