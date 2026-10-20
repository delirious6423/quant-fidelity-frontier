---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
license: mit
library_name: exllamav3
pipeline_tag: text-generation
tags:
- exl3
- quantized
- ornith
- moe
quantized_by: ultimatechris
---

# Ornith-1.5-35B-A3B EXL3 4.0 bpw HQ

EXL3 quantization of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), generated with [ExLlamaV3 1.4.2](https://github.com/turboderp-org/exllamav3) and the high-quality (`-hq`) recipe. This is a community quantization by `ultimatechris`, not an official Ornith release.

## Quantization

- Target: 4.0 bpw with the high-quality (`-hq`) strategy
- Artifact size: 18.47 GiB
- Conversion hardware: one NVIDIA A100 40GB

The large expert matrices use the target bitrate. Selected shared, mixer, attention, and output projections use higher bitrates where required by the HQ recipe. See `quantization_config.json` for the exact tensor layout.

## Base-vs-quantization check

A small teacher-forced comparison scored the same 60 target tokens from five fixed text continuations with the original BF16 base and this EXL3 artifact:

- Renormalized KL on the shared finite vocabulary support: `0.01702`
- Next-token top-1 agreement: `93.3%`
- NLL delta (EXL3 minus base): `-0.0381` nats/token

No meaningful degradation was observed in this sample. This is a limited diagnostic, not a broad benchmark; the slightly negative NLL delta should not be interpreted as evidence that the quantization improves the base model.

## Runtime

Use an ExLlamaV3-compatible frontend such as [TabbyAPI](https://github.com/theroyallab/tabbyAPI). This repository contains the tokenizer, chat template, configuration, and custom-code files needed by the base model. Runtime speed and maximum context depend on the frontend, GPU, context length, and batch settings.

Read the [base model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) before deployment.

## Runtime notes

I benchmarked this on a 4090 at a few context lengths. At short context fp16 KV is
a touch faster, but the 6-bit cache wins once you go long, around 12% faster at
128k. Decode barely drops with context (~105-115 tok/s out to 128k) and 128k only
needs about 18GB, so long context is cheap here. MTP and n-gram speculative decoding
both came out slower, and batching didn't really help, so I just run single stream
with the 6-bit cache. Build the cache before `model.load()` and stay on
exllamav3 1.4.2 / torch 2.10.0+cu128, torch 2.8 crashed on load.

## Attribution and limitations

The base model was developed and released by Ornith AI. This quantization retains the base model's capabilities, limitations, and usage requirements. Please follow the base model's license and safety guidance.

## EXL3 attribution

The EXL3 format and conversion tooling are provided by [TurboDerp's ExLlamaV3](https://github.com/turboderp-org/exllamav3).
