---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- deepseek
- deepseek-v4
- mxfp4
---

# DeepSeek-V4-Flash-0731 GGUF

GGUF conversions and expert-focused quantizations of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for `llama.cpp`.

This repository contains only the practical bullerwins Pareto points retained after size/KLD testing. Models that were dominated or offered only negligible size advantages over a better Unsloth quant were removed.

The routed MoE experts in the lossless reference remain in their native MXFP4 representation. FP8-origin non-expert matrices are dequantized into BF16, which represents their E4M3 values and power-of-two E8M0 scales exactly. Original BF16 and F32 tensors are also preserved. The resulting `MXFP4_MOE-BF16` file is lossless relative to the model tensors used by `llama.cpp`.

## Files

| File | Routed experts | Other tensors | Size |
|---|---|---|---:|
| `DeepSeek-V4-Flash-0731-MXFP4_MOE-BF16.gguf` | Native MXFP4 | Lossless BF16/F32 | 161.87 GB / 150.75 GiB |
| `DeepSeek-V4-Flash-0731-IQ2_XS-IQ3_XXS-GateUp-MXFP4-Down-Q8_0.gguf` | MXFP4 down; IQ2_XS gate/up in layers 0-17; IQ3_XXS gate/up in layers 18-42 | Eligible non-expert matrices in Q8_0 | 120.33 GB / 112.07 GiB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-Experts-Q8_0.gguf` | IQ3_XXS | Eligible non-expert matrices in Q8_0 | 113.87 GB / 106.05 GiB |
| `DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf` | IQ2_XS | Eligible non-expert matrices in Q8_0 | 87.90 GB / 81.86 GiB |
| `DeepSeek-V4-Flash-0731-DSpark.gguf` | DSpark speculative companion | Mixed MXFP4/Q8_0/BF16/F32 | 10.90 GB / 10.15 GiB |
| `DeepSeek-V4-Flash-0731-imatrix.gguf` | - | Importance matrix generated from 802 x 512-token calibration chunks | 0.47 GB / 0.44 GiB |

The IQ quantizations were generated with the included calibration importance matrix. The mixed autoresearch model keeps all 43 expert down projections in native MXFP4, uses IQ2_XS for the 36 gate/up tensors in layers 0-17, and uses IQ3_XXS for the 50 gate/up tensors in layers 18-42.

## KLD quality evaluation

All models were tested against logits from the lossless `MXFP4_MOE-BF16` reference using 50 x 512-token Wikitext-2 test chunks. The test evaluated 12,750 output distributions. Lower KLD and higher same-top-token agreement are better. Autoresearch policies were selected on the separate Wikitext-2 validation split; the test split was used only as a holdout check.

| Model | Mean KLD | PPL ratio | Same top token |
|---|---:|---:|---:|
| Lossless MXFP4/BF16 | 0.000000 | 1.0076 | 100.000% |
| Mixed IQ2_XS/IQ3_XXS gate/up + MXFP4 down | 0.280219 | 1.1379 | 84.094% |
| IQ3_XXS experts/Q8_0 | 0.310419 | 1.1604 | 83.067% |
| IQ2_XS experts/Q8_0 | 0.600273 | 1.4658 | 75.082% |

## Comparison with Unsloth Dynamic quants

![KLD versus GGUF size for the retained bullerwins Pareto set and Unsloth](bullerwins-pareto-vs-unsloth-kld-size.png)

Both collections provide strong and closely competitive quality/size tradeoffs. The lossless bullerwins and Unsloth models have byte-identical tensor payloads. The bullerwins IQ2_XS model fills the gap between Unsloth's IQ1 and IQ2 tiers, while bullerwins IQ3_XXS and the 112.07 GiB mixed model provide useful intermediate points between Unsloth `UD-IQ3_XXS` and `UD-Q3_K_M`. Unsloth remains stronger around 90, 119, 127, and 144 GiB and also offers smaller sub-80 GiB options. The plot intentionally shows only the bullerwins models retained as practical Pareto points.

## Source model

DeepSeek's recommended sampling settings are `temperature = 1.0`, with `top_p = 0.95` for agentic scenarios and `top_p = 1.0` otherwise. See the [official model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for architecture details, chat formatting, license, and intended usage.

---

# Original model card

The following model card is from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

# DeepSeek-V4-Flash-0731

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V4" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V4-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2606.19348"><b>Technical Report</b>👁️</a>
</p>

## Introduction

**DeepSeek-V4-Flash-0731** is the official release of **DeepSeek-V4-Flash**, superseding the preview version, with substantially enhanced agentic capabilities. It has the same model structure as [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark), i.e. it comes with a speculative decoding module attached.

DeepSeek-V4-Flash-0731 outperforms DeepSeek-V4-Pro (Preview) on benchmarks listed below despite its far smaller activated parameter count, and is broadly competitive with the strongest proprietary models available.

<div align="center">

| Benchmark | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash (Preview) | DeepSeek-V4-Pro (Preview) | GLM-5.2 | Opus-4.8 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Terminal Bench 2.1 | 82.7 | 61.8 | 72.1 | 81.0 | 85.0 |
| NL2Repo | 54.2 | 39.4 | 38.5 | 48.9 | 69.7 |
| Cybergym | 76.7 | 38.7 | 52.7 | - | 83.1 |
| DeepSWE | 54.4 | 7.3 | 12.8 | 46.2 | 58.0 |
| Toolathlon-Verified | 70.3 | 49.7 | 55.9 | 59.9 | 76.2 |
| Agents' Last Exam | 25.2 | 15.8 | 16.5 | 23.8 | 25.7 |
| AutomationBench Public | 25.1 | 10.8 | 12.8 | 12.9 | 27.2 |
| DSBench-FullStack † | 68.7 | 37.0 | 41.8 | 61.8 | 71.6 |
| DSBench-Hard † | 59.6 | 25.8 | 31.1 | 54.5 | 71.7 |

</div>

Notes:

1. For the Code Agent tasks among the public benchmarks above, DeepSeek-V4-Flash-0731 is evaluated with the minimal mode of DeepSeek Harness (to be released) as the agent framework, using the `max` reasoning effort level with `temperature = 1.0, top_p = 0.95`.
2. † DSBench-FullStack is an internal full-stack development test set; DSBench-Hard is an internal test set of difficult coding-agent problems.

## Chat Template

This release does not include a Jinja-format chat template. Instead, we provide a dedicated `encoding` folder with Python scripts and test cases demonstrating how to encode messages in OpenAI-compatible format into input strings for the model, and how to parse the model's text output. Please refer to the [`encoding`](encoding/README.md) folder for full documentation.

The `reasoning_effort` parameter now supports three levels — `low`, `high`, and `max` — which control how much deliberation the model spends before answering.

A brief example:

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! I am DeepSeek.", "reasoning_content": "thinking..."},
    {"role": "user", "content": "1+1=?"}
]

# messages -> string
prompt = encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")

# string -> tokens
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731")
tokens = tokenizer.encode(prompt)
```

## How to Run with vLLM

DSpark speculative decoding is enabled with a single flag — add --speculative-config with method: dspark to your vLLM launch command:

`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`

For example, the command below serves the model with vLLM on a single 4×GB300 node. 
See the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?hardware=b300&features=tool_calling,reasoning) for detailed instructions and other hardware configurations.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

## How to Run with SGLang

Enable DSpark with `--speculative-algorithm DSPARK` and do not set a separate `--speculative-draft-model-path` as the target and draft weights therefore come from the same checkpoint.
See the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4#hw=gb300&variant=flash-official&quant=fp4&strategy=low-latency&nodes=single) for detailed instructions, benchmarks and other hardwares configurations.

```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1 \
```

## How to Run Locally

Please refer to the [inference](inference/README.md) folder for detailed instructions on running DeepSeek-V4 locally, including model weight conversion and interactive chat demos.

For local deployment, we recommend setting the sampling parameters to `temperature = 1.0`, with `top_p = 0.95` for agentic scenarios and `top_p = 1.0` otherwise. For the `high` and `max` reasoning effort levels, we recommend a maximum output length of **384K** tokens.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
