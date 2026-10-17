---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
- nvidia/Qwen3.8-Flash-Next-NVFP4
- Qwen/Qwen3.8-Flash-Next
tags:
- pruning
- moe
- reap
- nvfp4
- quantized
---

# Qwen3.8-Flash-Next-NVFP4-REAP-k448

**REAP-pruned derivative of [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)**:
64 of 512 routed experts pruned per MoE layer (keep k=448), selected by REAP
(Router-weighted Expert Activation Pruning) saliency.

## What was pruned

| | Original | Pruned |
|---|---|---|
| Routed experts / layer | 512 | 448 (48 layers) |
| Total parameters | 180.0B | 164.9B |
| Checkpoint size | 123.6 GiB | 115.6 GiB (**−7.9 GiB**) |
| Routed-expert REAP mass | 100% | 92.65% retained |

Untouched: MTP module, PLE n-gram table, vision tower, attention, shared experts,
embeddings/lm_head, and the NVFP4 quantization format — the checkpoint loads anywhere
the original does (ModelOpt / TRT-LLM / vLLM NVFP4 paths), with `num_experts=448`.

## How experts were selected

- **Metric**: REAP saliency (mean of ‖f‖·g over tokens where the expert is selected —
  routing-weighted activation norms), computed per layer.
- **Calibration data**: 16.49M tokens of **real production traffic of this exact model** —
  681 LLM calls from an agentic deployment trace, including the multimodal path
  (626 unique images routed through the vision tower, 7,383 image references).
- **Protections**: 28 dead experts (zero activation across calibration) pruned first;
  5 "super experts" (reference `get_super_expert_indices` rule) force-kept.
- **Routers**: router weight columns subset to the kept experts (bit-exact rows of the
  original); kept expert weights are bit-identical to the source (only renumbered 0..447).

Per-layer cost: mean 7.35% of REAP mass removed (worst layer 15: 9.11%) for a 12.5%
expert-count reduction.

## Provenance

- Base model: `Qwen/Qwen3.8-Flash-Next` (Qwen Team), NVFP4 quantization by NVIDIA ModelOpt
  (snapshot `fc694b54fb0174e0913e6adf86691ef85a4ead47`).
- Pruning method: REAP — Router-weighted Expert Activation Pruning (Cerebras), applied
  with reference-semantics tooling; mask + calibration states archived by the authors
  of this derivative.
- Licenses: NVIDIA Open Model License + Qwen Community License 1.0 (inherited from the
  base model; see original model cards for full terms).

## Caveats

- Pruned at 48% calibration coverage (16.49M of 34.2M available tokens); full-corpus
  calibration may shift borderline expert rankings.
- QA diff vs the unpruned NVFP4 baseline on held-out tasks is recommended before
  production use.
- **MTP module**: preserved verbatim (its 512-expert bank, router, and embeddings
  are untouched and bit-identical to the source). MTP speculative decoding works on
  **vLLM >= 0.30.0** (which upstreamed the mixed-precision block-FP8 MTP draft fix,
  PR #55513). Because this checkpoint's main model has 448 routed experts while the
  MTP bank keeps the original 512, the MTP draft's expert count must be overridden
  at runtime: put the provided [`sitecustomize.py`](./sitecustomize.py) directory on
  `PYTHONPATH` and serve with `VLLM_MTP_NUM_EXPERTS=512`:

  ```sh
  VLLM_MTP_NUM_EXPERTS=512 PYTHONPATH=/path/to/sitecustomize-dir \
    vllm serve lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448 \
    --quantization modelopt \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
  ```

  The hook is a no-op unless the env var is set, and self-disables on vLLM versions
  where it is not needed. vLLM >= 0.30 also loads this checkpoint's FP8 PLE n-gram
  table natively; on vLLM 0.29 the same file patches PLE embedding method selection
  for non-MTP serving.
