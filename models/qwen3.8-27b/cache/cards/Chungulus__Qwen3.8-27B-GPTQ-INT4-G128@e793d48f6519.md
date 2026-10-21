---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - quantized
  - gptq-int4-g128
---

# Qwen3.8-27B GPTQ-INT4-G128

This repository is a quantized deployment variant of the official Qwen/Qwen3.8-27B model. It is not a fine-tune.

## Provenance and configuration

- Exact parent revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Quantizer: GPTQModel
- Quantizer revision: `7df3d1e837cb5552512bf99b8729c334d0b2c010`
- Quantization configuration: `{"bits": 4, "calibration_batch_size": 1, "calibration_corpus_sha256": "7ac4e3f0092bd5acfdeaec88f19a8c829458b0716c10ab9e81cda6f8b064b675", "calibration_samples": 128, "calibration_sequence_length": 2048, "calibration_sort": true, "checkpoint_format": "gptq", "desc_act": true, "format": "gptq", "group_size": 128, "lm_head": false, "meta": {"act_group_aware": false, "auto_forward_data_parallel": true, "dense_vram_strategy": "exclusive", "dense_vram_strategy_devices": null, "fallback": {"smooth": null, "strategy": "rtn", "threshold": "0.5%"}, "gc_mode": "interval", "gptaq": null, "hessian": {"chunk_bytes": null, "chunk_size": null, "staging_dtype": "float32"}, "mock_quantization": false, "moe_vram_strategy": "exclusive", "moe_vram_strategy_devices": null, "mse": 0.0, "offload_to_disk": false, "offload_to_disk_path": null, "pack_impl": "cpu", "wait_for_submodule_finalizers": false}, "method": "gptq", "pack_dtype": "int32", "quant_method": "gptq", "sym": true}`
- Artifact size: 18.238 GiB
- Supported runtime: GPTQModel 7.3.4+
- Tested hardware: NVIDIA GeForce RTX 4070 12 GB (Ada, SM89), WSL2 Ubuntu 24.04

## Calibration and architecture handling

The calibration corpus is original CC0-1.0 material authored for this project. It contains
216 text/tool samples and 8 vision samples
across prose, code, debugging, math, science, structured output, tools, multilingual, and
long-context categories. This build used calibration/core.jsonl:128.

- Tensor exclusions: `["*.input_layernorm", "*.post_attention_layernorm", "*.self_attn.q_norm", "*.self_attn.k_norm", "*.linear_attn.norm", "*.linear_attn.conv1d", "*.linear_attn.in_proj_a", "*.linear_attn.in_proj_b"]`
- Vision: True (deterministic image generation gate)
- Tool use: True (schema-valid XML tool-call gate)
- MTP: True for preservation of the 15 external tensors; no local
  speculative drafter/acceptance runtime is claimed.
- Processor/chat template: exact pinned-parent copy verified.

## Numerical validation

Full-vocabulary held-out comparison used 512 token positions.

| Metric | Result |
|---|---:|
| Mean KL | 0.02518878 |
| Median KL | 0.00990394 |
| P95 KL | 0.08463629 |
| Perplexity delta | +4.0345% |
| Top-1 agreement | 0.935547 |
| Top-5 set overlap | 0.912109 |
| Logit cosine similarity | 0.99503607 |

## Performance smoke

- Execution mode: RTX 4070 + CPU offload/auto map
- Device map summary: `{"cpu": 52, "cuda:0": 16}`
- Prompt/decode: 512 / 8 tokens
- Median TTFT: 48.3485 s
- Median prefill: 10.134 tok/s
- Median decode: 0.075 tok/s
- Cold/warm load: 153.769 / 141.985 s
- Peak physical device memory: 11.684 GiB
- PyTorch logical allocator peak (diagnostic, may exceed physical VRAM under WSL): 15.577 GiB
- Peak build RAM / VRAM: 17.538 / 11.180 GiB

## Limitations

The RTX 4070 validation may use CPU offload and is not a pure-GPU throughput claim unless the
performance mode explicitly says so. Preserved MTP weights do not establish speculative decoding
support. Compatibility and quality are established only for the exact revision and runtime above.
Machine-readable structural, generation, vision, tools, numerical, and performance reports are
stored under `validation/` in this repository. Remote verification evidence is retained by the
build worker and tied to the exact repository revision.
