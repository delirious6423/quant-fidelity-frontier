---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - quantized
  - exl3
---

# Qwen3.8-27B EXL3

This repository is a quantized deployment variant of the official Qwen/Qwen3.8-27B model. It is not a fine-tune.

## Provenance and configuration

- Exact parent revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Quantizer: GPTQModel EXL3
- Quantizer revision: `7df3d1e837cb5552512bf99b8729c334d0b2c010`
- Quantization configuration: `{"bits": 4.0, "calibration_batch_size": 1, "calibration_corpus_sha256": "7ac4e3f0092bd5acfdeaec88f19a8c829458b0716c10ab9e81cda6f8b064b675", "calibration_samples": 128, "calibration_sequence_length": 2048, "calibration_sort": true, "checkpoint_format": "exl3", "codebook": "mcg", "desc_act": false, "format": "exl3", "group_size": -1, "head_bits": 6.0, "lm_head": false, "meta": {"auto_forward_data_parallel": true, "dense_vram_strategy": "exclusive", "dense_vram_strategy_devices": null, "fallback": {"smooth": null, "strategy": "rtn", "threshold": "0.5%"}, "gc_mode": "interval", "moe_vram_strategy": "exclusive", "moe_vram_strategy_devices": null, "offload_to_disk": false, "offload_to_disk_path": null, "pack_impl": "cpu", "wait_for_submodule_finalizers": false}, "method": "exl3", "out_scales": "auto", "pack_dtype": "int32", "quant_method": "exl3"}`
- Artifact size: 17.798 GiB
- Supported runtime: GPTQModel 7.3.4+ / EXL3 CUDA kernel
- Tested hardware: NVIDIA GeForce RTX 4070 12 GB (Ada, SM89), WSL2 Ubuntu 24.04

## Calibration and architecture handling

The calibration corpus is original CC0-1.0 material authored for this project. It contains
216 text/tool samples and 8 vision samples
across prose, code, debugging, math, science, structured output, tools, multilingual, and
long-context categories. This build used calibration/core.jsonl:128.

- Tensor exclusions: `[]`
- Vision: True (deterministic image generation gate)
- Tool use: True (schema-valid XML tool-call gate)
- MTP: True for preservation of the 15 external tensors; no local
  speculative drafter/acceptance runtime is claimed.
- Processor/chat template: exact pinned-parent copy verified.

## Numerical validation

Full-vocabulary held-out comparison used 512 token positions.

| Metric | Result |
|---|---:|
| Mean KL | 0.00759337 |
| Median KL | 0.00308530 |
| P95 KL | 0.02567411 |
| Perplexity delta | -1.2346% |
| Top-1 agreement | 0.943359 |
| Top-5 set overlap | 0.943359 |
| Logit cosine similarity | 0.99754137 |

## Performance smoke

- Execution mode: RTX 4070 + CPU offload/auto map
- Device map summary: `{"cpu": 51, "cuda:0": 17}`
- Prompt/decode: 512 / 8 tokens
- Median TTFT: 45.4428 s
- Median prefill: 11.389 tok/s
- Median decode: 0.156 tok/s
- Cold/warm load: 62.684 / 50.418 s
- Peak physical device memory: 11.605 GiB
- PyTorch logical allocator peak (diagnostic, may exceed physical VRAM under WSL): 17.200 GiB
- Peak build RAM / VRAM: 17.469 / 10.123 GiB

## Limitations

The RTX 4070 validation may use CPU offload and is not a pure-GPU throughput claim unless the
performance mode explicitly says so. Preserved MTP weights do not establish speculative decoding
support. Compatibility and quality are established only for the exact revision and runtime above.
Machine-readable structural, generation, vision, tools, numerical, and performance reports are
stored under `validation/` in this repository. Remote verification evidence is retained by the
build worker and tied to the exact repository revision.
