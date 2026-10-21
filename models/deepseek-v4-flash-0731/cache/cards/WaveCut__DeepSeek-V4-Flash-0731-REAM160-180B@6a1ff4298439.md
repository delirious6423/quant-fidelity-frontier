---
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- deepseek-v4
- moe
- fp4
- fp8
---

<!-- ream-line-status -->
> [!WARNING]
> **Line status (2026-08-12): use with caution.** This first-generation REAM release passed its structural and smoke checks at publication time, but the cascaded descendants built from it (REAM128, REAM96) later showed severe multi-turn degradation, and the whole REAM line is being rebuilt with a revised methodology (single-step prune-first from the base model, longer multi-turn calibration). A REAM144 replacement targeting the same memory tier is planned; this repo may be superseded.

# DeepSeek V4 Flash 0731 REAM160-180B

This measured release reduces the 43-layer routed MoE from 256 to 160 experts while preserving top-k 6 routing. It contains approximately 180.445B logical parameters.

## Lineage and precision

- Source: `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Build code revision: `7d69b8490736789d47be25e6d7ccf5a8effefa16`.
- Routed expert weights: source-native E2M1 FP4, block 32, with E8M0 scales.
- Attention projections and their scales remain in the source FP8 layout and are copied byte-for-byte where unchanged.
- REAM weight accumulation is deterministic FP32; BF16 is transient compute, not the primary checkpoint format.
- MTP and DSpark tensors and metadata are absent.
- Checkpoint: 100,797,983,212 bytes; committed `NATIVE_READY.json` SHA-256 `68fc46c987891e6e763611612262b8fae0cbd3de7de300efa169b5701bfaaaec`.

## REAM methodology

REAM160 is a calibration-guided, training-free expert merge: it does not apply gradient updates or add training examples to the checkpoint.

Calibration composition: 3,072 samples x 512 tokens.

| Slice | Samples | Share | Pinned sources and focus |
| --- | ---: | ---: | --- |
| Code | 614 | 20.0% | `sahil2801/CodeAlpaca-20k` and `bigcode/self-oss-instruct-sc2-exec-filter-50k`; instruction following and executable code |
| Math | 307 | 10.0% | `openai/gsm8k` and `AI-MO/NuminaMath-CoT`; arithmetic and mathematical reasoning |
| Agent/tool | 1,075 | 35.0% | `NousResearch/hermes-function-calling-v1`; function selection and structured tool calls |
| Russian/Cyrillic | 1,076 | 35.0% | `ZeroAgency/ru-big-russian-dataset`; Russian and Cyrillic language coverage |

A disjoint 512-sample heldout split was excluded from calibration.

At each layer, REAP saliency is the routed activation norm multiplied by router probability. The 160 highest-saliency source experts become centroids; the remaining source experts are assigned by their mean output and router-gate similarity, with group capacity 32. Each group is merged by saliency-weighted deterministic FP32 accumulation, router rows are remapped, and top-k 6 routing is preserved. Calibration therefore determines which source experts are retained and combined for the code, math, agent/tool, and Russian/Cyrillic workload mix; it is not fine-tuning.

## Measured validation

- Hardware: 4x NVIDIA L40 44 GB.
- Wall time: 26.555 hours.
- Supervisor-accounted merge/finalization RunPod cost: $73.92.
- Comparative heldout NLL/PPL: not measured for this release.

Semantic suite: 100/100 cases passed; ru_dialogue=pass, code=pass, math=pass, agent_tool=pass, think=pass, no_think=pass.

## Loading and runtime compatibility

The configuration and tokenizer can be inspected with Transformers without remote code:

```python
from transformers import AutoConfig, AutoTokenizer

repo_id = "WaveCut/DeepSeek-V4-Flash-0731-REAM160-180B"
config = AutoConfig.from_pretrained(repo_id, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False)
print(config.num_hidden_layers, config.n_routed_experts, config.num_experts_per_tok)
```

End-to-end generation was validated with the REAM160 native expert-parallel loader on the hardware reported above. Stock `AutoModelForCausalLM`, vLLM, and SGLang loading is not claimed for this packed E2M1/E8M0 checkpoint; a runtime must implement the 160-expert DeepSeek V4 layout and its native FP4/FP8 storage.

## Limitations

- MTP and DSpark are intentionally absent.
- Comparative heldout NLL/PPL was not measured for this release.
- Stock AutoModelForCausalLM, vLLM, and SGLang loading was not validated for the packed E2M1/E8M0 checkpoint.
