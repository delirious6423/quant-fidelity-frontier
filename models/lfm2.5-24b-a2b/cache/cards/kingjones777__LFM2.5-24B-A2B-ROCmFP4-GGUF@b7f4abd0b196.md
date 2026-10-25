---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - rocm
  - amd
  - rocmfp4
  - rocmfpx
  - gfx1151
  - strix-halo
  - amd-strix-halo
  - ryzen-ai-max
  - ryzen-ai-max-395
  - radeon-8060s
  - lfm2
  - liquid-ai
  - quantized
---

# LFM2.5-24B-A2B (LEAN) — ROCmFP4 for AMD Strix Halo (gfx1151)

A 4-bit ROCmFP4 quantisation of **LiquidAI/LFM2.5-24B-A2B**, a mixture-of-experts model with 64 experts (top-4), 40 layers (30 double-gated short-convolution layers plus 10 grouped-query attention layers), 2 dense leading layers, hidden size 2048, a 128K vocabulary with tied embeddings, and a 131K-token native context. The model ships no MTP head.

## Files

| File | Size |
|---|---:|
| `LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.gguf` | 13,578,191,040 bytes |

## Why it was built from the vendor QAD file

When the vendor ships a quantization-aware (QAD) 4-bit file, building ROCmFP4 from it beats building from BF16: the same LEAN recipe applied to BF16 reads perplexity 16.0320 (1.044 times the stock Q4_K_M file), while built from the QAD file it reads 14.2330 (0.927 times).

## How it was built

```sh
llama-quantize --allow-requantize LFM2.5-24B-A2B-QAD-Q4_0.gguf scratch-QAD-F16.gguf F16 16
gufo-quant --protect tools/gufo-quant/protect/lfm2moe-v0.txt --manifest LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.manifest.json --time \
  scratch-QAD-F16.gguf LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.gguf LEAN 16
```

Result: 428 tensors, 4.53 bits per weight. Every expert, attention, short-convolution projection and dense feed-forward matrix is Q4_0_ROCMFP4; the token embedding (which is also the tied output head) is Q6_K; routers, norms, `exp_probs_b` and short-convolution state stay F32. The `lfm2moe-v0` protection preset is a single rule: `=token_embd.weight` at Q6_K.

## Quality

Perplexity on wikitext-2 test (512-token context, 320 chunks) with llama.cpp (ROCmFPX build); ratio against the stock Q4_K_M file, lower is better:

| Build | Perplexity | Ratio vs stock |
|---|---|---:|
| Vendor QAD-Q4_0 | 13.9774 | 0.910 |
| **This file (LEAN from the QAD file)** | **14.2330** | **0.927** |
| Stock Q4_K_M | 15.3536 | 1.000 |
| LEAN from BF16 | 16.0320 | 1.044 |

On Super ROC itself the served file reads 14.5655 on the decode path and 14.5154 on batched prefill, against 14.5560 on llama.cpp (ROCmFPX build) with the same file and text.

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, no drafter.

| Measurement | Result |
|---|---:|
| Prompt processing, 1518-token prompt | 2459.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 97.8 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 101.8 tok/s (101.7-101.8) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 6/7 |

## Run it

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.gguf --served-model-name lfm2.5 --context 32768
```

## Notes

- The model ships no MTP head; DSpark drafters run exact (matching drafter-off text) but at about half the speed of drafter-off, so serve without a drafter.
- No DSpark drafter was measured for the 24B model.
- Tool calls in the model's own pythonic format work, including required, named, parallel and streaming calls; the prompt cache works (a repeated 2,412-token prompt reaches the first token in 2.2 ms).

These files use ROCmFP4 tensor types (100 and above), which stock llama.cpp does not load. They were measured with Super ROC (https://github.com/kingjones30/superroc).
