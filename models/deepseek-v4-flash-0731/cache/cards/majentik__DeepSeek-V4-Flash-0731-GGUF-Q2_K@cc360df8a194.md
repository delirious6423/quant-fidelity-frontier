---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - gguf
  - llama.cpp
  - deepseek
  - deepseek_v4
  - moe
  - quantized
library_name: gguf
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-GGUF-Q2_K

GGUF **Q2_K** weight-quantized variant of
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for use with any
llama.cpp-compatible runtime (llama.cpp, LM Studio, Ollama via Modelfile).

## Provenance

- Source: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (MIT).
- The source checkpoint ships FP8 (e4m3, 128x128 block) weights; it was
  dequantized to a BF16 GGUF with llama.cpp `convert_hf_to_gguf.py`, then
  quantized to Q2_K with `llama-quantize` (llama.cpp b10310).
- Architecture: `deepseek_v4` (supported in llama.cpp b10310+).

## Smoke gate

Before upload this file passed a deterministic coherence gate
(greedy 64-token generation judged for emptiness, repetition loops,
multi-script gibberish, and special-token debris). Verdict: `ok`.

## Usage

```bash
llama-completion -no-cnv -m DeepSeek-V4-Flash-0731-Q2_K.gguf -p "Hello" -n 128
# or serve:
llama-server -m DeepSeek-V4-Flash-0731-Q2_K.gguf --port 8080
```

## Evaluation

_Text benchmarks are not run for this pack: it is a GGUF artifact evaluated via the llama.cpp runtime, and the eval harness uses `mlx_lm`. Quality gating is the deterministic coherence smoke (see "Smoke gate" above), which this tier passed before upload._

## Available tiers

| Repo | Quant |
| --- | --- |
| [majentik/DeepSeek-V4-Flash-0731-GGUF-Q2_K](https://huggingface.co/majentik/DeepSeek-V4-Flash-0731-GGUF-Q2_K) | Q2_K |

IQ4_XS and Q3_K_M tiers were quantized but **held back**: they could not be
verified by the local smoke gate (142GB / 126GB weights exceed the 128GB RAM
of the release machine), and unverified quants are not published. Note the
source checkpoint is already a ~4.4 BPW FP8/MXFP4 hybrid, so tiers at or above
Q4_K_M would inflate rather than compress and are intentionally not offered.
