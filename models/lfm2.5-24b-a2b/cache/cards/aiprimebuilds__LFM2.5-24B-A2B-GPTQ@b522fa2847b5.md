---
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model:quantized: LiquidAI/LFM2.5-24B-A2B
author: aiprimebuilds
language:
- en
tags:
- gptq
- 4-bit
- int4
- moe
- lfm2_moe
- lfm2.5
- liquid
- vllm
- quantized
---

# LFM2.5-24B-A2B-GPTQ

4-bit GPTQ build of [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), built to serve an office of agents from one RTX 4000 Ada 20 GB. Also tested on a 16 GB RTX 4080 SUPER.

![At a glance](figures/glance.svg)

![Accuracy: GPTQ vs BF16](figures/accuracy.svg)

Within noise everywhere except GPQA Diamond: 11 fewer correct out of 198.

![Under load](figures/load.svg)

The RTX 4000 runs all 8 users at once, even with 32k prompts. The 4080 SUPER generates faster but fits only about two long prompts at a time; the rest wait.

![Agent users per card](figures/agents.svg)

**RTX 4000 Ada: 4 users comfortably, 6 with slowdowns. 4080 SUPER: 1–2.** The limit is how many conversation histories fit in the KV cache (KV cache tokens ÷ average session size).

## Run it

Tested on vLLM 0.31.0 with these flags:

| Flag | Why |
|---|---|
| `--quantization gptq_marlin` | fast 4-bit kernels |
| `--kv-cache-dtype fp8` | halves KV cache memory; every GPTQ result above uses it |
| `--max-model-len 131072` (`65536` on 16 GB) | what fits next to the weights |
| `--gpu-memory-utilization 0.95` | gives vLLM 95% of VRAM, so more is left for KV cache |
| `--max-num-seqs 8` | up to 8 requests at once |
| `--reasoning-parser qwen3` | puts the `<think>` block in `reasoning_content` |
| `--enable-auto-tool-choice --tool-call-parser lfm2` | OpenAI-style tool calls |

<details>
<summary><b>Details</b></summary>

- **Fidelity:** wikitext-2 perplexity 15.69 vs 13.81 for BF16. KL divergence 0.365 mean. 78% top-token agreement.
- **KV cache:** every GPTQ number uses an FP8 KV cache. The BF16 reference uses BF16.
- **Tool calls:** 96–98% valid. On vLLM 0.31.0, streaming sometimes cuts off a call's arguments.
- **Sampling:** the server default is `temperature 0.1`. For agents, use about 0.4.

</details>

**Next:** a 3-seed Red Hat-style evaluation.

LFM Open License v1.0 · [shipleybuilds.ai](https://shipleybuilds.ai)
