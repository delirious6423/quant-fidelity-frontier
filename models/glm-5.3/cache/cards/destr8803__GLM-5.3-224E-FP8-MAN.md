---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3
tags:
- glm
- mixture-of-experts
- fp8
- expert-pruning
- man
---

# GLM-5.3 224E FP8 MAN

This is an independently pruned FP8 checkpoint derived from
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3). It retains 224 of
the original 256 routed experts in every sparse layer while keeping top-8
routing unchanged. Here, **224E means 224 routed experts per sparse layer; it
does not mean 224 billion parameters**.

The checkpoint is not an official Z.ai release. The original tokenizer, chat
template, dense weights, attention weights, shared experts, and license are
retained. Only the routed-expert inventory and corresponding router outputs are
rewritten according to the keep map.

## Pruning Method

Experts were selected with Mean Activation Norm (MAN):

```text
S(1,0,1) = mean(l2(unweighted_expert_output)) over tokens routed to the expert
```

For each sparse layer, the 224 experts with the largest MAN scores are retained.
No retraining, fine-tuning, merging, or weight averaging is applied. Seven
activation-outlier candidates in layer 77 were protected with the conservative
guardrail `max_abs > max(global_p99.5, global_max / 10)`. The MTP layer reuses
the layer-77 keep IDs.

This reduces the routed-expert count by 12.5%, from 256 to 224, while preserving
top-8 routing.

## Calibration Data

The deterministic calibration subset contains 2,848 samples and 1,983,722
tokens, with approximately 500,000 tokens per axis:

| Axis | Samples | Tokens |
|---|---:|---:|
| General | 1,128 | 495,244 |
| Code | 912 | 500,387 |
| Agentic | 648 | 497,302 |
| Reasoning | 160 | 490,789 |

The source pool used public data from:

- `allenai/c4` (`en`)
- `theblackcat102/evol-codealpaca-v1`
- `open-r1/Mixture-of-Thoughts` (`code`, `math`, and `science`)
- `glaiveai/glaive-function-calling-v2`
- `SWE-bench/SWE-smith-trajectories` (`tool`)

The calibration data is not included in this model repository. The subset was
selected with seed 42, without packing or truncation, and scored across eight
Gaudi2 ranks. Its SHA-256 is
`27bdf438f05019d65655c2f0db7ef0b7f994d3c2b250abe66244f30af986a372`.

## Validation

The materialized checkpoint passed deep structural and tensor validation:

| Check | Result |
|---|---:|
| Safetensors shards | 141 |
| Indexed tensors checked | 104,037 / 104,037 |
| Indexed tensor payload | 663,759,907,008 bytes |
| Routed experts per sparse layer | 224 |
| Functional runtime checks | 4 / 4 |
| Behavior checks across low/high/max reasoning effort | 12 / 12 |
| GPQA Diamond deterministic 40-question subset | 39 / 40 (97.5%) |

The GPQA run used the public OpenAI simple-evals GPQA Diamond CSV with one
deterministic permutation per question, seed 0, `temperature=1.0`, `top_p=0.95`,
and a maximum output of 120,000 tokens in a 131,072-token context. The clustered
bootstrap 95% interval was 92.5%-100%. This is a 40-question evaluation, not a
claim of performance on the complete benchmark or under the unpublished Z.ai
evaluation protocol.

The checkpoint was runtime-qualified with vLLM on eight Intel Gaudi2 devices
using tensor parallelism 8, expert parallelism 8, BF16 KV cache, and a
131,072-token maximum context. In that environment it produced 19.72 tokens/s
single-stream and 237.03 aggregate tokens/s across 16 concurrent 1,024-token
generations. These throughput figures are hardware- and configuration-specific.

See [`prune-report.json`](./prune-report.json) for hashes and machine-readable
provenance.

## Usage

Use a recent Transformers or vLLM release with GLM-5.3, FP8 MoE, and the target
accelerator supported. Follow the upstream
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) chat-template and
serving guidance. In particular, GLM-5.3 supports `reasoning_effort` values
`low`, `high`, and `max`; benchmark validation used the default `max` behavior.

Deployment has been validated on Gaudi2 with TP8/EP8. CUDA and other backends
were not qualified as part of this release.

## Limitations

- This is a 12.5% expert-pruned derivative and may regress on tasks not covered
  by the calibration and evaluation suites.
- Calibration was task-agnostic but English-heavy; no explicit Spanish
  calibration data was used.
- GPQA coverage is limited to a deterministic 40-question subset.
- Long-context, multilingual, safety, coding-agent, and tool-use behavior has
  not been exhaustively benchmarked.
- The original GLM-5.3 license applies. Review `LICENSE` before use.

## Method References

- Unified expert-pruning formulation, MAN, and MSAN:
  https://arxiv.org/abs/2606.15716v1
- Reference implementation: https://github.com/ZongfangLiu/unified-expert-pruning
- Activation-outlier motivation: https://arxiv.org/abs/2507.23279v3

For the base model architecture, capabilities, citation, and intended use, see
the upstream [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) model
card.
