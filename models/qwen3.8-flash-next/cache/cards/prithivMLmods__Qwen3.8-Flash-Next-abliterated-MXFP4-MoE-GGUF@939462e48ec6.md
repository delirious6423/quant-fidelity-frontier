---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
language:
- en
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen4_exp
- llama-cpp
- llama-cli
- mtp
- speculative-decoding
- qwen4_exp
- microscaling
- text-generation-inference
- moe
- Qwen3.8
- mxfp4_moe
- open-weight
- mmproj
- nvidia-blackwell
- rtx-pro-6000
- rtx-5090
- SM120
- GB202
- B200
- B300
- FP4
- blackwell
- kv-cache
- abliterated
- heretic
- uncensored
- decensored
- instruction-following
- multimodal
---

# **Qwen3.8-Flash-Next-abliterated-MXFP4-MoE-GGUF**

> **Qwen3.8-Flash-Next** is an open-weight multimodal Mixture-of-Experts (MoE) model and experimental architectural preview for Qwen4, combining 125B core parameters (6B activated per token across 10 routed and 1 shared expert out of 512) with 51B offload-friendly n-gram embedding parameters and a 4B multi-token prediction (MTP) head alongside a vision encoder. Across its 48 layers, it introduces a hybrid attention layout interleaving Gated DeltaNet with block-level **Qwen Sparse Attention (QSA)** to cut long-context latency, multi-branch **Gated Residual** streams for training stability, and a native 262,144-token context window extensible to 1M tokens via YaRN. Supporting text, image, and hour-scale video inputs with configurable `<think>` reasoning (`enable_thinking`, `preserve_thinking`, and `reasoning_effort`), it outperforms larger and proprietary peers across agentic coding, computer use, and multimodal benchmarks—including DeepSWE 1.1 (58.7), SWE-bench Pro (62.5), SWE-bench Multilingual (81.0), AndroidWorld (84.5), GPQA Diamond (91.7), LiveCodeBench v6 (91.9), and MathVision (95.7 with code interpreter)—and is released under the **Qwen Community 1.0** license for deployment via `Transformers`, `vLLM`, `SGLang`, and `TokenSpeed`.

> [!NOTE]
> MTP — Speculative Decoding: Yes

> [!NOTE]
> Multi-Token Prediction (MTP) is a speculative decoding technique that accelerates local inference by predicting multiple future tokens in a single forward pass. Unlike traditional speculative decoding, it does not require a separate, smaller "draft" model; the MTP prediction heads act as the drafter. In this repository, the MTP weights are shipped as a **separate GGUF file** **[Qwen3.8-Flash-Next-MTP.BF16.gguf](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-abliterated-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next-MTP.BF16.gguf)** rather than embedded in the main model file. Load it alongside the main model GGUF to enable speculative decoding. The main model runs standalone without it.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| Qwen3.8-Flash-Next-MTP.BF16.gguf | BF16 (MTP) | 7.77 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-abliterated-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next-MTP.BF16.gguf) | Standalone Multi-Token Prediction (MTP) file in BF16. Loaded alongside the main model to enable speculative decoding. Not a standalone model. |
| Qwen3.8-Flash-Next-abliterated.MXFP4_MOE.gguf | MXFP4_MOE | 124 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-abliterated-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next-abliterated.MXFP4_MOE.gguf) | **MXFP4_MOE**: Highly optimized for **NVIDIA Blackwell** hardware accelerators **(e.g., RTX Pro 6000, RTX 50-series, B200, B300)**, offering native hardware-accelerated FP4 Tensor Core speeds. Main model file; pair with the MTP file for speculative decoding. |
| Qwen3.8-Flash-Next.mmproj-bf16.gguf | mmproj-bf16 | 908 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-abliterated-MXFP4-MoE-GGUF/blob/main/Qwen3.8-Flash-Next.mmproj-bf16.gguf) | Multimodal projection file in BF16 format. Used for vision/image capabilities. |


## Run Inference (llama.cpp)

```bash
/path/to/llama.cpp/build/bin/llama-cli \
  -m /path/to/models/Qwen3.8-Flash-Next-abliterated.MXFP4_MOE.gguf \
  -md /path/to/models/Qwen3.8-Flash-Next-MTP.BF16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl auto \
  -c 8192 \
  --cache-type-k q8_0 \
  --seed 42 \
  -n 8192 \
  -st \
  -p "A, B and C jointly thought of engaging themselves in a business venture. It was agreed that A would invest Rs. 6500 for 6 months, B, Rs. 8400 for 5 months and C, Rs. 10,000 for 3 months. A wants to be the working member for which, he was to receive 5% of the profits. The profit earned was Rs. 7400. Calculate the share of B in the profit."
```

## Output (llama.cpp)

<div style="
    background: rgba(255, 255, 0, 0.25); /* light yellow background */
    padding: 16px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 0, 0.5); /* light yellow border */
    margin: 16px 0;
">

<details>
  <summary>Expand the dropdown to view the full output.</summary>

> A, B and C jointly thought of engaging themselves in a business venture. It was agreed that A would invest Rs. 6500 for 6 months, B, Rs. 8400 for 5 months and C, Rs. 10,000 for 3 months. A wants to be the working member for which, he was to receive 5% of the profits. The profit earned was Rs. 7400. Calculate the share of B in the profit.

[Start thinking]

We need solve math problem. Need provide final answer. Let's analyze fully.

Problem: A, B, C jointly thought engaging themselves in business venture. Agreed A invest Rs 6500 for 6 months, B Rs 8400 for 5 months, C Rs 10000 for 3 months. A wants to be working member for which, he was to receive 5% of profits. Profit earned Rs 7400. Calculate share of B in profit.

Typical partnership with working member receives 5% profit before distribution? Usually A gets 5% of profit as remuneration, remaining profit distributed in ratio of investments*months.

Compute remaining profit = 7400 - 5% of 7400 = 7400 - 370 = 7030.

Investment-month ratio:
A = 6500*6 = 39000
B = 8400*5 = 42000
C = 10000*3 = 30000
Total = 111000. Simplify? divide 1000? 39:42:30 divide 3 = 13:14:10.

B share remaining = 14/(13+14+10=37) * 7030. 7030/37 =? 37*190 = 7030? 37*190=7030 yes. B = 14*190=2660. A share = 13*190=2470 plus 370 working = 2840. C = 10*190=1900. Sum 7400. B = Rs 2660.

Need answer concise maybe steps. Check ambiguity: "A wants to be the working member for which, he was to receive 5% of the profits." Could mean he receives 5% of total profit, and rest shared as capital ratio. Yes.

Could there be alternative: first divide profit 7400 in ratio of investments and months, then A gets 5% of profit? No working member allowance before distribution typical. If A gets 5% profit, rest 95% distributed.

Let's consider if B's share includes any working? No only A.

Need final with formula. Maybe note if A's 5% deducted before sharing. Let's craft final. Ensure no mistakes: 5% of 7400 = 370. Remaining 7030. Ratios. B share = 2660. Final.
[End thinking]

A’s working-member share = 5% of Rs. 7400 = Rs. 370.

Remaining profit to be divided among A, B, C in the ratio of their investments × time:

- A: 6500 × 6 = 39000
- B: 8400 × 5 = 42000
- C: 10000 × 3 = 30000

Ratio = 39000 : 42000 : 30000 = 13 : 14 : 10

Total remaining profit = Rs. 7030.

B’s share = \(\frac{14}{37} \times 7030 = 2660\)

**Share of B = Rs. 2660**

[ Prompt: 168.9 t/s | Generation: 141.0 t/s ]

Exiting...

</details>
</div>

> [Prompt: 168.9 t/s | Generation: 141.0 t/s] - Generated with full GPU offloading on 2x NVIDIA RTX PRO 6000 Blackwell Server Edition using MTP (draft-n-max: 2). GGUF: `Qwen3.8-Flash-Next-abliterated.MXFP4_MOE.gguf` | Draft MTP: `Qwen3.8-Flash-Next-MTP.BF16.gguf`

## llama.cpp

LLM inference in C/C++ — [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

Quantized using llama.cpp by [prithivMLmods](https://huggingface.co/prithivMLmods).