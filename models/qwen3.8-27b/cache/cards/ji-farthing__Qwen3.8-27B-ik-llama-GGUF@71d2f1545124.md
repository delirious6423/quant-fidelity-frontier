---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - gguf
  - ik_llama
  - qwen35
  - imatrix
  - text-only
---

*Updated 08-15-2026: replaced IQ4_NL with IQ4_KS; added IQ3_KT*

**HuggingFace's download widget does not recognize _KS and _KT ggufs ----->**

**Download links in table below**

# Qwen3.8-27B, ik_llama.cpp GGUF

Text-only GGUF conversions of Qwen3.8-27B, quantized with an importance matrix calibrated for agentic coding rather than general text.

Qwen3.8-27B ships as a vision-language model. These files carry the text tower and MTP block only. The vision tower is not converted and is not present, so these will not process images or video. Vision requires the original weights with a runtime that supports them.

## Files

| File | Quant | Download | GPU-resident weights | Host-side embedding |
| --- | --- | ---: | ---: | ---: |
| [`Qwen3.8-27B-ik_llama-IQ4_KS.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-27B-ik-llama-GGUF/resolve/main/Qwen3.8-27B-ik_llama-IQ4_KS.gguf?download=true) | IQ4_KS | 13.91 GiB | 13.27 GiB | 0.63 GiB |
| [`Qwen3.8-27B-ik_llama-IQ3_KT.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-27B-ik-llama-GGUF/resolve/main/Qwen3.8-27B-ik_llama-IQ3_KT.gguf?download=true) | IQ3_KT | 10.89 GiB | 9.62 GiB | 1.26 GiB |
| [`Qwen3.8-27B-ik_llama-IQ3_XXS.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-27B-ik-llama-GGUF/resolve/main/Qwen3.8-27B-ik_llama-IQ3_XXS.gguf?download=true) | IQ3_XXS | 10.12 GiB | 9.60 GiB | 0.51 GiB |
| [`qwen38-27b-agentic.imatrix`](https://huggingface.co/ji-farthing/Qwen3.8-27B-ik-llama-GGUF/resolve/main/qwen38-27b-agentic.imatrix?download=true) | n/a | 13.0 MiB | n/a | n/a |

**The quantization selector on this page lists only IQ3_XXS, and the "Use this model" snippets are all pinned to it.** That is a display limit, not a statement about the other files. The Hub builds that list from `general.file_type`, and it recognizes the values in upstream llama.cpp's enum. IQ3_XXS is 23 and is recognized. IQ4_KS is 145 and IQ3_KT is 152, both ik_llama-specific and outside that enum, so neither appears. Use the links above, or the Files tab.

`token_embd` is not a layer, so `-ngl 99` does not move it to the GPU. Host-side embedding stays in system memory rather than VRAM and no flag is needed. Its precision therefore costs VRAM nothing but adds download size. IQ3_KT carries it at q8_0; the other two carry it at the file's own precision.

All three carry the NextN MTP block, so speculative decoding needs no separate draft model. The imatrix is published alongside the weights.

## Calibration / imatrix

The importance matrix was computed over 296 chunks at 512 tokens, 151,552 tokens in total. The corpus was built to match this specific deployment distribution rather than general capability.

| Class | Share of tokens |
| --- | ---: |
| Agentic and technical English | 45% |
| Code | 30% |
| General English prose | 15% |
| Chinese prose | 10% |

The code and technical portion includes source code from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), which is the runtime these files target. The remainder is agent session transcripts, project documentation, and English and Chinese encyclopedia prose. The Chinese share is not meant to tune the model toward Chinese; it exercises the activations a Chinese-capable model includes, so that the quantizer does not treat them as unimportant. Proportions are measured in tokens rather than characters.

Calibration reaches every matmul in the 64 transformer layers, 496 tensors. The NextN draft head is not exercised by a plain forward pass, so it carries no importance data.

The MTP block is therefore held at q5_K in both IQ3 files, by design. An uncalibrated tensor quantized at very low precision degrades unpredictably, and in the draft head this shows up directly as a lower acceptance rate. Holding the block at q5_K costs about 150 MB on a 10 GB file and keeps speculative decoding working at the smaller quants.

## Runtime

These target [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). Two of the three cannot load anywhere else:

- **IQ4_KS** contains `IQ4_KS`, `IQ5_K` and `Q6_K` tensors. The first two are ik_llama-specific and upstream llama.cpp with its `gguf-py` tooling does not recognize them.
- **IQ3_KT** is a trellis quantization. That family exists only in ik_llama.
- **IQ3_XXS** contains only standard types. It may load elsewhere, but it has not been tested on upstream llama.cpp.

ik_llama activates the in-model MTP block with `--spec-type mtp:...`.

### Allocations, measured

These come from load logs and let you compute a fit for your own card. They follow from the architecture and the cache type, not from which of the three files you use.

| Allocation | Size |
| --- | --- |
| KV cache at `-ctk q4_0 -ctv q4_0` | 149.63 MiB, plus 19.125 MiB for every 1,024 tokens of context |
| MTP draft KV cache | 1.125 MiB for every 1,024 tokens of context |

Enabling the MTP block also allocates a second compute buffer the same size as the main one. `--ctx-size-draft` does not shrink it, because the draft context follows `n_ctx`.

Thinking is on by default and `reasoning_effort` is a per-request parameter that defaults to `xhigh`. Draft acceptance measured 0.917 with thinking on against 0.723 with it off, on one file and one build, so turning thinking off also lowers the value of the MTP block.

## Validation

Measured on a single RTX 4070 with 12 GB VRAM, 64 GB system RAM, ik_llama.cpp at commit `43afea46c`.

### Perplexity against the unquantized conversion

Reference is the BF16 conversion of the released weights. Both corpora are held out and share no text with the calibration corpus.

| File | Held-out English | vs BF16 | Held-out code | vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 (reference) | 6.6738 | n/a | 1.9750 | n/a |
| IQ4_KS | 6.7204 | +0.7% | 1.9862 | +0.6% |
| IQ3_KT | 7.0688 | +5.9% | 2.0486 | +3.7% |
| IQ3_XXS | 7.1235 | +6.7% | 2.0837 | +5.5% |

40 chunks of English and 54 of code, 512 tokens each, every file measured with the same binary entirely on CPU so that no arm mixes CPU and GPU kernels differently from another. Repeat runs reproduced exactly, confirming the measurement is deterministic.

IQ3_KT and IQ3_XXS occupy almost identical GPU memory, so that pair is a like-for-like comparison of the quantization scheme rather than of the bit budget. The trellis file is better on both corpora, and the gap is larger on code.

## Conversion notes

Source: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Converted with `convert_hf_to_gguf.py` from llama.cpp at commit `4c1a0af40`, unmodified, using `--outtype bf16`. The converter drops the vision tower without a flag: 1199 tensors in the source index, 866 in the GGUF.

Quantized with `llama-quantize` from ik_llama.cpp at commit `43afea46c`.

IQ3_KT uses a hand-written recipe rather than the type's defaults:

```
blk\.64\.=q5_K
blk\..*\.(ffn_down|ffn_gate|ffn_up|attn_qkv|attn_q|attn_k|attn_v|attn_output|attn_gate|ssm_out|ssm_alpha|ssm_beta)\.weight=iq3_kt
token_embd\.weight=q8_0
output\.weight=iq3_kt
```

Qwen3.8-27B loads and runs on ik_llama.cpp main with no source changes. No patched build is needed to use these files.
