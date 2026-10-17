---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - w4a16
  - 4-bit
  - moe
  - quantized
  - modelopt
  - sglang
---

# Qwen3.8-Flash-Next-NVFP4-W4A16

Built with Qwen.

Weight-only NVFP4 quantization of the routed MoE experts of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Activations stay at 16-bit.

## What is quantized

| | |
|---|---|
| Scheme | NVFP4, 4-bit weights, group size 16, **BF16 activations (W4A16)** |
| Scope | routed experts of the 48 text MoE layers — 120.8B params |
| Tool | NVIDIA TensorRT Model Optimizer 0.46.0 |
| Calibration | data-free (`algorithm: max`) |
| Size | 169 GiB, down from 336 GiB BF16 |

Everything else stays in BF16: attention, Gated DeltaNet linear attention,
QSA indexer, shared experts, routers, hyper-connections, PLE n-gram tables,
MTP head, vision tower, and `lm_head`.

### Parameter breakdown

The model is 48 text layers, hidden size 2560, 512 experts per layer with
top-10 routing and `moe_intermediate_size` 640. The routed experts are stored
as fused 3-D tensors (`mlp.experts.gate_up_proj` `(512, 1280, 2560)` and
`mlp.experts.down_proj` `(512, 2560, 640)`) and account for 120.8B of the
~125B main-model parameters — 96.6% of it.

Separately, the per-layer-embedding (PLE) n-gram tables are 128 shards of
`(2500012, 160)`, or 51.2B parameters / 97.7 GiB in BF16. They are **not**
quantized here and are copied through byte-for-byte from the source
checkpoint.

The parameter count shown in the sidebar is derived from the packed tensors
and understates the real total.

## Serving

**SGLang** has day-0 support for `qwen4_exp` and can serve this architecture
today. **vLLM** supports the architecture from v0.29.0
([vllm-project/vllm#53896](https://github.com/vllm-project/vllm/pull/53896)).

Single-GPU serving needs the 97.7 GiB PLE table offloaded to host RAM.
vLLM v0.30.0 adds a UVA offload for it
([vllm-project/vllm#54371](https://github.com/vllm-project/vllm/pull/54371)):
set `VLLM_PLE_CPU_OFFLOAD=1`, or `cpu_offload=true` in `--engram-config`. With PLE offloaded, the on-GPU
weights are roughly 60 GiB.

**We have not validated serving end to end yet.** Measured configurations —
GPU, tensor-parallel size, achievable `max-model-len`, and required
engine flags — will be published here once we have run them.

## Reproduction

Quantized with `modelopt`'s `NVFP4_EXPERTS_ONLY_CFG`, modified to disable the
expert input quantizers so activations remain 16-bit. The model is loaded
with accelerate offload so that `export_hf_checkpoint` takes modelopt's
streaming writer path, `mtq.compress` is applied before export, and the PLE
tables are excluded from the export and restored afterwards from the source
shards. This keeps peak host memory low enough to run the whole job on a
single 96 GB GPU node with 160 GB of RAM.

## Run it on Meshive

Meshive is a task-first GPU cloud. Bring your task — training, inference, or rendering — and run it on RTX 5090 and RTX PRO GPUs at up to 70% less.

This checkpoint was produced on a single [Meshive GPU Pod](https://meshive.ai/pods?utm_source=huggingface&utm_medium=model_card&utm_campaign=Qwen3.8-Flash-Next-NVFP4-W4A16) with one RTX PRO 6000 Blackwell (96 GB).

## License

Released under the Qwen Community License (see `LICENSE`). Built with Qwen.