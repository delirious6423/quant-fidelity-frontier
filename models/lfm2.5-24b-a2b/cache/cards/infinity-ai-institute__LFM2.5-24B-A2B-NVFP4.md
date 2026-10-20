---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- nvfp4
- modelopt
- vllm
- moe
- liquid
- lfm2
---

# LFM2.5-24B-A2B NVFP4 (experts)

NVFP4 quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for Blackwell GPUs.
Only the routed MoE experts are quantized (FP4 E2M1 values with one FP8 E4M3 scale per 16 values, for weights and
activations). Attention, the short-convolution blocks, the router, the two dense layers, embeddings and the LM head stay
BF16. Size: 14 GB (BF16: 45 GB).

## Serve with vLLM

```bash
vllm serve infinity-ai-institute/LFM2.5-24B-A2B-NVFP4 --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser lfm2
```

The model always thinks before answering. Give agentic and tool-calling requests a generous `max_tokens` (we used 8192
per agent turn): with small limits most turns end inside the reasoning.

## Quality and speed

One NVIDIA B300, vLLM 0.29.1rc1, both models served the same way. Deltas are paired per item against BF16, with 95%
bootstrap intervals.

| | BF16 | NVFP4 (this) | Paired delta |
|---|---|---|---|
| BFCL-style tool calling, 800 items | 0.334 | 0.329 | -0.5 pt [-3.0, +2.1] |
| tau2-style agent tasks, 60 tasks x 2 trials, pass | 0.525 | 0.517 | -0.8 pt [-8.3, +6.7] |
| tau2-style, database end state correct | 0.725 | 0.708 | -1.7 pt [-10.0, +6.7] |
| tau2-style, required information communicated | 0.667 | 0.642 | -2.5 pt [-10.8, +5.8] |
| GSM8K (strict match) | 0.774 | 0.767 | |
| Output throughput, 16 concurrent, 1024 in / 512 out | 2959 tok/s | 4655 tok/s | 1.57x |
| Time per output token | 5.27 ms | 3.31 ms | |
| Time to first token | 74 ms | 69 ms | |

The tool-calling and agent numbers come from our internal benchmark harness. Its tau2-style tasks are our own airline,
retail and telecom tasks (not the upstream tau2-bench set), and the simulated customer is played by the model under test.
None of the deltas is distinguishable from zero. For scale, two identical greedy runs of BF16 disagree on 11 of 60 tau2
tasks.

We also measured exact full-vocabulary KL divergence from BF16 on 29,646 tokens of BF16's own greedy continuations of
64 held-out agentic contexts (SWE-agent, OpenHands and tool-use traces, 12K to 30K tokens of context each). That
measure picks the recipe:

| Recipe | KL (nats/token) | Top-1 agreement |
|---|---|---|
| BF16 run on a different kernel path (noise floor) | 0.0023 | 98.5% |
| ModelOpt max calibration (default) | 0.0245 | 95.0% |
| ModelOpt MSE weight-scale search, FP8 scale sweep (this checkpoint) | 0.0217 | 95.3% |

## How it was made

NVIDIA Model Optimizer (ModelOpt) 0.48.0rc1, `NVFP4_EXPERTS_ONLY_CFG` with the MSE weight-scale search
(`fp8_scale_sweep`) and max calibration for activations. Calibration data is agentic: 32 sequences of 8,192 tokens from
SWE-agent trajectories (nebius/SWE-agent-trajectories) and tool-calling and interactive-agent conversations
(nvidia/Nemotron-Agentic-v1), rendered with the model's chat template including tool schemas. The evaluation contexts
above were held out from calibration.

## License

Same license as the base model (LFM Open License v1.0, `LICENSE`). Modified from the original: the expert weights are
quantized to NVFP4 and `config.json` carries the quantization config; all other weights are unchanged.
