---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
quantized_by: LibertAIDAI
tags:
  - nvfp4
  - modelopt
  - quantization
  - blackwell
  - vision
  - mtp
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-NVFP4

NVFP4 post-training quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with NVIDIA ModelOpt. This is the safetensors source checkpoint. If you want ready-to-run llama.cpp files, use [Qwen3.8-27B-NVFP4-GGUF](https://huggingface.co/LibertAIDAI/Qwen3.8-27B-NVFP4-GGUF) or the [MTP variant](https://huggingface.co/LibertAIDAI/Qwen3.8-27B-NVFP4-MTP-GGUF), both converted from this checkpoint.

We publish it because no recipe-transparent NVFP4 source existed for this model. The exact recipe is below so the result can be audited or reproduced.

## What is quantized

NVFP4, group size 16, weights and activations, on the language-model MLP linears only:

- `model.language_model.layers.{0..63}.mlp.{gate,up,down}_proj`

That is 192 quantized linears, 64 layers by three projections. This follows NVIDIA's recipe for dense models rather than quantizing everything reachable.

Kept in BF16:

- `model.language_model.layers.*.self_attn.*`, the 16 full-attention layers
- `model.language_model.layers.*.linear_attn.*`, the 48 Gated-DeltaNet layers
- `model.visual.*`, the entire vision tower
- `mtp.*`, the multi-token-prediction block
- `lm_head.weight` and `model.language_model.embed_tokens.weight`
- all norms and the Mamba `conv1d` weights

KV cache quantization is not enabled, so this checkpoint carries no `k_scale` or `v_scale` calibration artifacts.

## Recipe

- Tool: NVIDIA ModelOpt 0.45.0
- Stack: `transformers 5.10.1`, `torch 2.11.0+cu128`
- Calibration: 512 samples from [`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail) 3.0.0, sequence length 512, `max` algorithm
- Config: `mtq.NVFP4_DEFAULT_CFG` with disable rules appended for `*self_attn*`, `*linear_attn*`, `*visual*`, `*vision*`, `*embed*`, `*lm_head*` and `*mtp*`
- Hardware: single RTX 5090 (32 GB, sm_120) with accelerate CPU offload, since the BF16 source is 55.6 GB
- Wall clock: 18.7 minutes, of which 17 minutes was calibration

## The MTP block

`transformers` declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` for the `qwen3_5` architecture, so the MTP weights are dropped silently when the model is loaded and any straightforward export omits them entirely, without raising an error.

The 15 `mtp.*` tensors here were re-injected unquantized in BF16 after export, written to `model-mtp.safetensors` and referenced from `model.safetensors.index.json`. Keep that file alongside the main shards. If it goes missing, downstream tooling will load the model without speculative decoding and give no warning.

## Caveats

- Calibration was English news text, matching NVIDIA's published choice. Multilingual and code-heavy workloads were not measured against a held-out set.
- The vision tower is BF16 and was not separately evaluated. Vision quality should track the upstream BF16 release closely.
- This checkpoint was validated by converting it to GGUF and benchmarking there. It has not been tested under vLLM or TensorRT-LLM.

## License and attribution

Inherits Apache 2.0 from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). All rights, responsibilities and acceptable-use policies of the upstream license apply.

Quantization performed by [LibertAI](https://libertai.io).
