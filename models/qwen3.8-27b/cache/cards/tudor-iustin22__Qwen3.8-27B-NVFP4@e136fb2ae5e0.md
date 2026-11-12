---
language:
- en
- zh
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp4
- fp8
- vllm
- sglang
- modelopt
- blackwell
- nvidia-gb10
- nvidia-gb300
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
provider: Qwen
name: tudor-iustin22/Qwen3.8-27B-NVFP4
readme: https://huggingface.co/tudor-iustin22/Qwen3.8-27B-NVFP4/blob/main/README.md
---

<div align="center">

# Qwen3.8-27B-NVFP4

**NVFP4 quantization of Qwen3.8-27B, built for NVIDIA GB10 and GB300 superchips and optimized for the Blackwell architecture**

[![X](https://img.shields.io/badge/Follow-@tudoriustin__-000000?logo=x&logoColor=white)](https://x.com/tudoriustin_)
[![Model Optimizer](https://img.shields.io/badge/Method-NVIDIA%20Model%20Optimizer-76b900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/Model-Optimizer)
[![Base model](https://img.shields.io/badge/Base-Qwen3.8--27B-6d28d9)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](https://choosealicense.com/licenses/apache-2.0/)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-tudoriustin__-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/tudoriustin_)

</div>

> **Follow [@tudoriustin_](https://x.com/tudoriustin_) on X** for more quantizations and updates.

## About the Model

Qwen3.8-27B-NVFP4 is a mixed NVFP4/FP8 quantization of Qwen3.8-27B, made specifically for the **NVIDIA GB10 and GB300 superchips** and optimized for the **Blackwell architecture**. It sharply reduces weight storage and accelerator memory requirements relative to 16-bit precision while staying close to the BF16 baseline on reasoning, coding, agentic tasks, long-context recall, instruction following and multimodal understanding. It was quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) and is ready for commercial or non-commercial use.

## Model Details

| | |
| --- | --- |
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Architecture** | Qwen3_5 (`Qwen3_5ForConditionalGeneration`), 27B parameters |
| **Modalities** | Text / Image / Video in, Text out |
| **Context length** | Up to 262K tokens |
| **Quantization** | Mixed NVFP4 / FP8 |
| **KV cache** | FP8 (`fp8_e4m3`) |
| **Target hardware** | NVIDIA GB10 and GB300 superchips (Blackwell) |
| **Test hardware** | NVIDIA Grace Blackwell GB300 |
| **Runtime engines** | vLLM, SGLang |
| **Preferred OS** | Linux |
| **Developer** | Tudor Iustin |
| **License** | Apache 2.0 |

## Method

**Scope.** NVFP4 quantization is applied to the MLP layers and the language model head (`lm_head`). FP8 quantization is applied to the self-attention and linear-attention layers. The KV cache is stored in FP8.

**Framework.** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt), NVIDIA's library for post-training quantization on Blackwell GPUs.

**Procedure.** Quantization was performed with NVIDIA Model Optimizer. The NVFP4 layers were calibrated on 2,048 samples from [Nemotron-Post-Training-Dataset-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) using the [NVIDIA Model Optimizer Local-Hessian calibration algorithm](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.config.html#modelopt.torch.quantization.config.LocalHessianCalibConfig).

## Serving the Model

### vLLM

Start the docker image `vllm/vllm-openai:nightly` and run:

```bash
vllm serve tudor-iustin22/Qwen3.8-27B-NVFP4 \
    --port 8000 \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --mm-encoder-tp-mode data \
    --seed 0 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill
```

### SGLang

Start the docker image `lmsysorg/sglang:dev` and run:

```bash
sglang serve \
  --trust-remote-code \
  --model-path tudor-iustin22/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --host 0.0.0.0 \
  --port 30000 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

For more details, see the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B).

## Evaluation

### Setup

We evaluate on GPQA Diamond, Terminal-Bench, AA-LCR, MMMU-Pro, SciCode and IFBench, covering text-based reasoning, agentic terminal tasks, long-context recall, multimodal reasoning, scientific coding and instruction following. All evaluations used `temperature=1.0`, `top_p=0.95` and a vLLM context limit of 262,144 tokens. All benchmarks except Terminal-Bench 2.1 used `max_new_tokens=65536`; Terminal-Bench 2.1 did not set an explicit generation-token limit.

### Results

| Benchmark | Qwen3.8-27B (BF16) | Qwen3.8-27B-NVFP4 |
| --- | :---: | :---: |
| GPQA Diamond | **88.92** | 88.01 |
| Terminal-Bench | **75.56** | 74.02 |
| AA-LCR | 72.63 | **73.38** |
| MMMU-Pro | **75.14** | 74.86 |
| SciCode | 47.93 | **48.41** |
| IFBench | **80.07** | 78.93 |

Bold marks the higher score in each row.

## Acknowledgements

This work builds on [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team, [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) for inference.

## Citation

```bibtex
@misc{iustin2026qwen38nvfp4,
  title  = {Qwen3.8-27B-NVFP4: NVFP4 Quantization with NVIDIA Model Optimizer for GB10 and GB300},
  author = {Tudor Iustin},
  year   = {2026},
  url    = {https://huggingface.co/tudor-iustin22/Qwen3.8-27B-NVFP4}
}
```

---

<div align="center">

**More quants coming. Follow [@tudoriustin_](https://x.com/tudoriustin_) on X.**

</div>
