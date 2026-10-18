---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
- zh
- multilingual
tags:
- deepseek-v4.1
- fp8
- multimodal
- reasoning
- 1m-context
- dspark
- vllm
- research
---

<div align="center">

# DeepSeek V4.1 Flash DERISKED FP8

### Blackfrost Research · multimodal reasoning · 1M context · FP8

![FP8](https://img.shields.io/badge/format-FP8-111827?style=for-the-badge)
![context](https://img.shields.io/badge/context-1M-047857?style=for-the-badge)
![vision](https://img.shields.io/badge/vision-preserved-1f2937?style=for-the-badge)
![vLLM](https://img.shields.io/badge/serve-vLLM-b45309?style=for-the-badge)

**Built by [Blackfrost Research](https://huggingface.co/Blackfrost-Research) · Las Vegas, Nevada**

</div>

> **PRIVATE EVALUATION CHECKPOINT.** Access is limited to approved Blackfrost Research partners while validation is in progress.

## Model

This repository contains a complete, directly loadable FP8 checkpoint derived from [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). It does not require an adapter or a second model repository at inference time.

The upstream multimodal architecture is preserved, including vision, Engram conditional memory, CSA2 attention and the in-checkpoint DSpark draft components. The checkpoint uses the native `DeepseekV41ForCausalLM` architecture and DeepSeek V4.1 prompt encoder.

| | |
|---|---|
| **Model ID** | `Blackfrost-Research/DeepSeek-V4.1-Flash-DERISKED-FP8` |
| **Base** | `deepseek-ai/DeepSeek-V4.1-Flash` |
| **Architecture** | `DeepseekV41ForCausalLM` |
| **Format** | FP8 safetensors, 48 shards |
| **Modalities** | Text and vision |
| **Context** | 1,048,576 tokens |
| **Speculative decoding** | DSpark, 5 draft tokens in the validated profile |
| **Validated hardware** | 8× NVIDIA B300, TP8 with expert parallelism |
| **Serving API** | OpenAI-compatible vLLM |

## Recommended generation settings

| Setting | Value |
|---|---:|
| Temperature | `1.0` |
| Top-p | `0.95` |
| Reasoning | enabled |
| Reasoning effort | `100` |

Use the native DeepSeek V4.1 tokenizer and reasoning parser. Current vLLM returns hidden reasoning in the structured `reasoning` field while keeping the final response in `content`. Client interfaces should display `content` as the assistant answer.

## Deployment

The [`deployment-kit/vllm`](deployment-kit/vllm) directory provides:

- a Docker launcher using the latest CUDA vLLM nightly;
- a native vLLM launcher;
- the validated TP8 and expert-parallel settings;
- DSpark speculative decoding;
- health and reasoning-parser verification.

Quick start:

```bash
export HF_TOKEN=hf_your_read_token
./deployment-kit/vllm/serve-docker.sh
```

The OpenAI-compatible endpoint starts at `http://127.0.0.1:18000/v1` by default.

People using an automation agent should give it [`AGENTS.md`](AGENTS.md) before deployment. That file contains the required model, parser and verification invariants.

## Evaluation

### Bare 450 refusal benchmark

Completed **450/450 prompts with 0 request errors** in **1,014 seconds**. Phase 1 is the refusal-substring screen. Phase 2 reports confirmed true holds from the 19-row review pool.

| Dataset | Phase-1 substring | Phase-2 true |
|---|---:|---:|
| AdvBench | 1/150 | 1 |
| StrongREJECT | 3/150 | 2 |
| XSTest | 8/150 | 2 |
| **Harmful: AdvBench + StrongREJECT** | **4/300** | **3/300** |
| **XSTest safe subset** | **2/75** | **0/75** |
| Incoherent | 7 box-drawing false positives | **0** |
| **All** | **12/450** | **5/450** |

The behavior floor remained intact in this run: all 450 prompts completed, the seven box-drawing substring hits were false positives, and final incoherent output was zero.

Additional partner capability and refusal results will be added with their runtime, hardware, prompt settings and sample counts after review.

## Intended use

- Controlled research and partner evaluation
- Long-context reasoning and code analysis
- Multimodal document and image analysis
- Authorized security research and defensive engineering

Operators are responsible for access control, workload authorization and review of generated output.

## Limitations

- This is a research checkpoint under active evaluation.
- Output quality depends on the DeepSeek V4.1 prompt encoder and generation settings.
- A generic DeepSeek V4 tokenizer or reasoning parser can produce malformed prompts or expose reasoning delimiters in assistant content.
- Hardware profiles other than the validated TP8 configuration require independent memory and throughput testing.

## License and attribution

This checkpoint is derived from DeepSeek V4.1 Flash and is distributed under the upstream MIT license included in this repository. Review the [upstream model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and technical report for architecture details.
