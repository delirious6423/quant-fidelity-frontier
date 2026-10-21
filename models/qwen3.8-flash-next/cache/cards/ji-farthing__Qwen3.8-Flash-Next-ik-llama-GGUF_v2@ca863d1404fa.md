---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - gguf
  - ik_llama
  - qwen4exp
  - imatrix
  - text-only
---

**Hugging Face's download widget does not recognize _KT GGUFs. Download links are in the table below.**

# Qwen3.8-Flash-Next, ik_llama.cpp IQ4_KT_v2 GGUF

For the recipe comparison, see the [main model card](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF#choosing-a-file).

This repository retains IQ4_KT_v2 for existing downloads. It uses less GPU memory for model tensors than v3 under the documented offload configuration. Its measured KL divergence is higher. The `_v2` suffix does not identify the latest release.

Date: 2026-09-22

Text-only GGUF conversion of Qwen3.8-Flash-Next, quantized with an importance matrix made from an agentic coding corpus.

This version uses IQ4_KT for the 160-column PLE table and IQ3_KT for the routed and shared expert down projections (`ffn_down_exps`, `ffn_down_shexp`). The previous release used IQ4_NL for tensors whose row widths were not multiples of 256. The v2 file is 80.72 GiB. The original IQ4_KT file was 88.42 GiB.

**These replacement files require [PR #2481](https://github.com/ikawrakow/ik_llama.cpp/pull/2481), merged into `main` 2026-09-23 as `cb860a9`, or any later `main`. Earlier heads store IQ4_KT tails in a different order and read these bytes incorrectly.**

This revision changes the IQ4_KT tail byte order. IQ3_KT tails keep their previous layout. The conversion is a lossless byte permutation. It preserves the quantized values and file size. It does not use a new importance matrix. Download both replacement shards together. Use the SHA-256 values below to distinguish this revision from the earlier files with the same names.

## Files

| File | Quant | Download |
| --- | --- | ---: |
| [`Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2/resolve/main/Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf?download=true) | IQ4_KT_v2 | 44.41 GiB |
| [`Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00002-of-00002.gguf`](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF_v2/resolve/main/Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00002-of-00002.gguf?download=true) | IQ4_KT_v2 | 36.31 GiB |
| IQ4_KT_v2 total | | 80.72 GiB |

Download both shards into the same directory. Pass the first shard to `llama-server`.

The PLE table uses IQ4_KT. Routed expert gate/up tensors use IQ4_KT. `ffn_down_exps` and `ffn_down_shexp` use IQ3_KT. Token embeddings use Q8_0. The four-column PLE convolution uses F16.

## MTP draft head

The [main repository](https://huggingface.co/ji-farthing/Qwen3.8-Flash-Next-ik-llama-GGUF#mtp-draft-head) provides the existing MTP draft head. This repository contains only the v2 main model. MTP is optional.

## Runtime

These files need [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) `main` at `cb860a9` or later. For multi-turn chat, use `391a95f` ([PR #2524](https://github.com/ikawrakow/ik_llama.cpp/pull/2524)) or later, or pass `"preserve_thinking": true` in `chat_template_kwargs`. Builds from [PR #2382](https://github.com/ikawrakow/ik_llama.cpp/pull/2382) up to PR #2524 force `preserve_thinking` off unless the request sets it. With it off, whether forced or set to `false` by the client, the chat template drops the reasoning of earlier turns, and each new user message makes the server process the previous exchange again. The generation checks below used this command. Metal, Vulkan, ARM, and multi-GPU execution are not validated.

```bash
GGML_CUDA_NO_PINNED=1 llama-server \
   -m Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf \
   --defer-ple -ngl 49 -ot 'ffn_.*_exps=CPU' \
   -c 32768 -b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q8_0 --jinja
```

## Validation

Measured on one RTX 4070, 64 GB RAM, Core i7-11700K.

### KL divergence against the BF16 source

Both test texts used ten chunks of 2048 tokens. The code text contains CPython and PostgreSQL source. All quantized runs used the same BF16 references and runtime library contents. Reference perplexities were 3.3316 for English and 1.7218 for code.

| File | Size | English mean KLD | Same top-1 | Code mean KLD | Same top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous IQ3_KT with IQ4_NL fallbacks | 80.22 GiB | 0.151454 | 85.054% | 0.134990 | 92.131% |
| Previous IQ4_KT with IQ4_NL fallbacks | 88.42 GiB | 0.132548 | 85.738% | 0.124749 | 92.414% |
| IQ4_KT_v2 | 80.72 GiB | 0.132805 | 86.276% | 0.124558 | 93.216% |

Same top-1 is the share of positions where the quantized model's most likely token matches the BF16 reference. On these samples, v2 has approximately equal divergence to the previous IQ4_KT near the previous IQ3_KT size. Its measured top-1 agreement is higher than both. Other workloads were not measured.

The quality runs used the earlier revisions `563ede2` and `8eb6566`. On the PR #2481 head `15fd2c0`, a CUDA graph matrix passed 560 cases and 1,124,880 comparisons with a scalar reference, and the CPU and CUDA quantizer checks passed. On commit `1d4abbe`, the converted main GGUF completed all four standard prompts, and all four texts matched the earlier results. The only differences between `1d4abbe` and `15fd2c0` are a variable name and a condition in CPU kernels that KT types do not reach. Text agreement does not establish correctness of generated code.

The lossless conversion changed 320,994,816 IQ4 tail rows across 98 tensors. Every changed row passed an inverse byte comparison. IQ3 tensors and full IQ4 blocks remain unchanged. Every payload byte in all 1,224 output tensors was compared with the converted single file after splitting.

## Calibration


The importance matrix was computed over 700 chunks of 2048 tokens. The corpus was built to match an agentic coding deployment rather than general capability: 45% agentic and technical English, 30% code (including all of the `ik_llama` and `llama.cpp` source code), 15% general English prose, 10% Chinese prose, measured in tokens. The Chinese share keeps the quantizer from giving those channels low importance. It does not tune the model toward Chinese.

## Conversion notes

Source: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

The BF16 conversion used `convert_hf_to_gguf.py` from llama.cpp at `f8dbcd6`. This GGUF was quantized at `bd7c321` with the `qwen4exp-agentic-v3.imatrix` available in the previous repository. That quantizer also quantized tensors in slabs to reduce memory use. Slab quantization is not part of PR #2481. The original GGUF is preserved. This revision changes the order of its IQ4 tail bytes.

Quantization options:

```text
--token-embedding-type q8_0
--custom-q "ffn_down_exps=iq3_kt,ffn_down_shexp=iq3_kt"
iq4_kt
```

The four-column PLE convolution is F16, because its rows are too narrow for a KT type. Tensor counts are 713 IQ4_KT, 96 IQ3_KT, 25 Q8_0, one Q6_K, 388 F32, and one F16.

SHA-256 of the converted single file and the replacement shards:

```text
d67e94de26809ef9c86f275430beff8af29e18d8bf2a2aba4f76a7526701a315  Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2.gguf
38f3285c5a4768896d729d9b1b21ed5c9cdd73fa6860c4bce7ef364b135063cc  Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00001-of-00002.gguf
d704c2551ca87e34134a12cd9a2fc9c854223a9ec75c8ee22e481a99efcea827  Qwen3.8-Flash-Next-ik_llama-IQ4_KT_v2-00002-of-00002.gguf
```
