---
quantized_by: bartowski
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
---

## Llamacpp imatrix Quantizations of Qwen3.8-27B by Qwen

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b10896">b10896</a> for quantization.

Original model: https://huggingface.co/Qwen/Qwen3.8-27B

**Model details:**
- Parameter count: 28B
- Input support: text, image (with mmproj file) - [details](#multimodal)
- Speculative decoding: yes (MTP) - [details](#mtp)
- imatrix: yes - [details](#imatrix)
- Perplexity/KLD measured: yes - [details](#perplexity)

[How to run](#how-to-run)

## Prompt format

```
<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.

{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

**Don't know which to choose?** Grab [Q4_K_M](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) (17.44GB) - usually a good mix of size and performance. Download instructions available [here](#downloading-using-the-hugging-face-cli)

## Updates 11-09-2026

Re-uploaded using my new tensor-layout!

Here's some graphs showing the relative performance between the previous files and the new ones:

![image](https://cdn-uploads.huggingface.co/production/uploads/6435718aaaef013d1aec3b8b/1jjkNZtNxHKFCDS5v42UZ.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/6435718aaaef013d1aec3b8b/NzB3GyXwTbxNtComDzXEg.png)

You'll note not everything is 100% perfect, but overall everything gets shoved to the ideal quandrant, which is great!

Note that `Q2_K_L`, `Q3_K_XL`, and `Q5_K_L` are no longer being created and so were not updated, they still exist in this repo. If I see a need to introduce them again, I'll consider it, but I think the curve is already pretty nice.

## Available files:

| Filename | Quant type | File Size | Split | bpw | PPL | Mean KLD | Same top-p | Notes |
| -------- | ---------- | --------- | ----- | --- | --- | -------- | ---------- | ----- |
| [Qwen3.8-27B-Q8_0.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf) | Q8_0 | 29.12GB | false | 8.38 | 6.75 | 0.0009 | 98.7% |  |
| [Qwen3.8-27B-Q6_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q6_K_L.gguf) | Q6_K_L | 24.96GB | false | 7.19 | 6.74 | 0.0024 | 97.7% | *recommended* |
| [Qwen3.8-27B-Q6_K.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q6_K.gguf) | Q6_K | 23.86GB | false | 6.87 | 6.74 | 0.0030 | 97.4% | *recommended* |
| [Qwen3.8-27B-Q6_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q6_K_S.gguf) | Q6_K_S | 22.86GB | false | 6.58 | 6.74 | 0.0036 | 97.4% | *recommended* |
| [Qwen3.8-27B-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q5_K_M.gguf) | Q5_K_M | 20.92GB | false | 6.03 | 6.75 | 0.0053 | 96.9% | *recommended* |
| [Qwen3.8-27B-Q5_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q5_K_S.gguf) | Q5_K_S | 19.57GB | false | 5.63 | 6.76 | 0.0060 | 96.4% | *recommended* |
| [Qwen3.8-27B-Q4_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_L.gguf) | Q4_K_L | 18.82GB | false | 5.42 | 6.76 | 0.0106 | 95.5% | *recommended* |
| [Qwen3.8-27B-Q4_1.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_1.gguf) | Q4_1 | 17.83GB | false | 5.13 | 6.76 | 0.0158 | 94.5% |  |
| [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) | Q4_K_M | 17.44GB | false | 5.02 | 6.76 | 0.0139 | 95.0% | *recommended* |
| [Qwen3.8-27B-IQ4_NL.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ4_NL.gguf) | IQ4_NL | 17.44GB | false | 5.02 | 6.76 | 0.0150 | 94.8% |  |
| [Qwen3.8-27B-Q4_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_S.gguf) | Q4_K_S | 16.36GB | false | 4.71 | 6.77 | 0.0156 | 94.8% | *recommended* |
| [Qwen3.8-27B-Q4_0.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_0.gguf) | Q4_0 | 16.35GB | false | 4.71 | 6.86 | 0.0265 | 92.8% |  |
| [Qwen3.8-27B-IQ4_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ4_XS.gguf) | IQ4_XS | 15.48GB | false | 4.46 | 6.78 | 0.0188 | 94.1% | *recommended* |
| [Qwen3.8-27B-IQ3_M.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ3_M.gguf) | IQ3_M | 14.86GB | false | 4.28 | 6.92 | 0.0406 | 91.1% |  |
| [Qwen3.8-27B-Q3_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q3_K_L.gguf) | Q3_K_L | 14.12GB | false | 4.07 | 6.90 | 0.0432 | 90.7% |  |
| [Qwen3.8-27B-Q3_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q3_K_M.gguf) | Q3_K_M | 13.40GB | false | 3.86 | 7.03 | 0.0564 | 89.8% |  |
| [Qwen3.8-27B-IQ3_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ3_XS.gguf) | IQ3_XS | 12.80GB | false | 3.69 | 7.00 | 0.0585 | 89.4% |  |
| [Qwen3.8-27B-Q3_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q3_K_S.gguf) | Q3_K_S | 12.74GB | false | 3.67 | 7.08 | 0.0648 | 89.0% |  |
| [Qwen3.8-27B-IQ3_XXS.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ3_XXS.gguf) | IQ3_XXS | 12.32GB | false | 3.55 | 7.06 | 0.0739 | 88.5% |  |
| [Qwen3.8-27B-Q2_K.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q2_K.gguf) | Q2_K | 10.82GB | false | 3.12 | 7.44 | 0.1612 | 83.5% |  |
| [Qwen3.8-27B-IQ2_M.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ2_M.gguf) | IQ2_M | 10.52GB | false | 3.03 | 7.53 | 0.1494 | 83.9% |  |
| [Qwen3.8-27B-IQ2_S.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ2_S.gguf) | IQ2_S | 9.68GB | false | 2.79 | 7.73 | 0.2090 | 81.0% |  |
| [Qwen3.8-27B-IQ2_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ2_XS.gguf) | IQ2_XS | 9.09GB | false | 2.62 | 8.02 | 0.2403 | 79.7% |  |
| [Qwen3.8-27B-IQ2_XXS.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-IQ2_XXS.gguf) | IQ2_XXS | 8.88GB | false | 2.56 | 8.41 | 0.2834 | 78.0% |  |

Download a specific file:

```
hf download bartowski/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-Q4_K_M.gguf" --local-dir ./
```

## Downloading using the Hugging Face CLI

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have the Hugging Face CLI installed:

```
pip install -U "huggingface_hub[cli]"
```

Download a specific file:

```
hf download bartowski/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-Q4_K_M.gguf" --local-dir ./
```

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/Qwen3.8-27B-GGUF:Q4_K_M
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b10896 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## Multimodal

This model supports image input. Alongside the quants, this repo includes the multimodal projector files [mmproj-Qwen3.8-27B-f16.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/mmproj-Qwen3.8-27B-f16.gguf) and [mmproj-Qwen3.8-27B-bf16.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/mmproj-Qwen3.8-27B-bf16.gguf), which pair with any quant above.

llama.cpp downloads the mmproj automatically when using `-hf` as shown above; if you're loading files manually, pass it with `--mmproj`.

## MTP

This model has MTP (Multi-Token Prediction) layers, and they are included in these quants

MTP layers act as a built-in draft model, letting llama.cpp run speculative decoding for faster generation. To use them, add the following flag to your llama.cpp command:

```
--spec-type draft-mtp
```

Note: the MTP layers are stored at Q4_0 in the imatrix quants (except for the Q8_0 quant), since imatrix calibration does not exercise them. Q4_0 is chosen for its speed which massively benefits MTP performance.

## Per-tensor layouts

Some of these files were built with a layout computed for this model instead of llama.cpp's standard one-size-fits-all rules. A Q4_K_M is still mostly Q4_K; the extra precision goes to the weights this particular model is most sensitive to. The S, M or L in a name says how much of the model stays at the base precision: about 90 % for S, 70 % for M and 50 % for L. An `_L` name is simply the large size of its family. Q4_K_L is to Q4_K_M what Q4_K_M is to Q4_K_S; Q6_K_S, Q6_K and Q6_K_L are the small, medium and large sizes of Q6_K, with Q6_K_L about halfway to Q8_0. In earlier releases an `_L` name meant the embedding and output weights were kept at Q8_0; in these files it means the larger size of the base type. There is no size target, so each file's bits per weight is reported rather than promised.

The layout each of these files was built with is published in the [`layouts/`](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/main/layouts) folder: `<file>.tensor-types.txt` is the exact `--tensor-type-file` given to `llama-quantize`, and `<file>.layout.json` records how it was computed, including the generator version, the llama.cpp release and the commit, so any of them can be rebuilt.

Checked on this model before any of these files were released: Q6_K reached 0.96×, Q4_K_M 0.94×, Q3_K_M 0.79× and IQ2_XXS 0.78× the KL divergence of the standard layout at the same file size.

These files replace an earlier release of this model that used the standard layout. Every file of both releases was measured the same way, against the same bf16 reference on the same text, and the full side-by-side is in the "Compared with the previous release" fold below.

<details>
<summary>Layout details</summary>

Files built from a computed layout:

| Quant | Size | Body bits/weight | File bits/weight | Body kept at base type |
| ----- | ---- | ---------------- | ---------------- | ---------------------- |
| Q6_K_L | 24.96GB | 7.41 | 7.19 | 50 % |
| Q6_K | 23.86GB | 7.04 | 6.87 | 70 % |
| Q6_K_S | 22.86GB | 6.71 | 6.58 | 90 % |
| Q5_K_M | 20.92GB | 6.13 | 6.03 | 70 % |
| Q5_K_S | 19.57GB | 5.69 | 5.63 | 90 % |
| Q4_K_L | 18.82GB | 5.49 | 5.42 | 50 % |
| Q4_K_M | 17.44GB | 5.04 | 5.02 | 70 % |
| IQ4_NL | 17.44GB | 5.04 | 5.02 | 70 % |
| Q4_K_S | 16.36GB | 4.69 | 4.71 | 90 % |
| IQ4_XS | 15.48GB | 4.41 | 4.46 | 90 % |
| IQ3_M | 14.86GB | 4.25 | 4.28 | 50 % |
| Q3_K_L | 14.12GB | 4.00 | 4.07 | 50 % |
| Q3_K_M | 13.40GB | 3.77 | 3.86 | 70 % |
| IQ3_XS | 12.80GB | 3.57 | 3.69 | 90 % |
| Q3_K_S | 12.74GB | 3.55 | 3.67 | 90 % |
| IQ3_XXS | 12.32GB | 3.41 | 3.55 | 70 % |
| Q2_K | 10.82GB | 2.98 | 3.12 | 70 % |
| IQ2_M | 10.52GB | 2.88 | 3.03 | 70 % |
| IQ2_S | 9.68GB | 2.60 | 2.79 | 70 % |
| IQ2_XS | 9.09GB | 2.40 | 2.62 | 90 % |
| IQ2_XXS | 8.88GB | 2.34 | 2.56 | 70 % |

Checked on this model: the computed layout against the standard one, measured by KL divergence against the unquantized model. The ratio compares each computed file with the standard ladder read at that file's own size, so it can differ from the two KLD columns when the two files differ in size.

| Quant | Computed layout KLD | Standard layout KLD | Ratio at equal size | Size vs standard file |
| ----- | ------------------- | ------------------- | ------------------- | --------------------- |
| Q6_K | 0.0030 ± 0.0001 | 0.0031 ± 0.0001 | 0.96× | +1.7 % |
| Q4_K_M | 0.0139 ± 0.0004 | 0.0129 ± 0.0002 | 0.94× | −1.9 % |
| Q3_K_M | 0.0564 ± 0.0009 | 0.0466 ± 0.0008 | 0.79× | −8.2 % |
| IQ2_XXS | 0.2834 ± 0.0033 | 0.3037 ± 0.0034 | 0.78× | −5.4 % |

How it works: the base type is a floor for every body tensor and a fixed share of the body bytes stays at it (90 % for S, 70 % for M, 50 % for L); the remaining bytes go where a cross-model sensitivity prior, measured by KL divergence against the unquantized model, says they buy the most quality. The embedding and output tensors are sized by their share of the file: a small table is kept at Q8_0, a large one follows the file's bitrate. A K-quant and the IQ quant with the same base bitrate (Q3_K_S and IQ3_XS, Q3_K_M and IQ3_S, Q3_K_L and IQ3_M) come out at about the same size; the IQ file is the GPU-oriented twin.

</details>

<details>
<summary>Compared with the previous release</summary>

The previous release of this model used llama.cpp's standard layout; these files use the computed one. Both sets were measured against the same bf16 reference on `wiki.test.raw` over 100 chunks, so the numbers are directly comparable. "Same top-p" is the share of tokens where the quantized model's most likely next token matches the bf16 model's. Q4_0, Q4_1 and Q8_0 are left out because the layout does not touch them. The previous release's Q2_K_L, Q3_K_XL and Q5_K_L have no counterpart here: an `_L` name now means the large size of its family, so those names are no longer built. Q6_K_S is new in this release, the small size of Q6_K, and has no previous counterpart.

| Quant | Previous size | New size | Size change | Previous KLD | New KLD | Previous same top-p | New same top-p |
| ----- | ------------- | -------- | ----------- | ------------ | ------- | ------------------- | -------------- |
| Q6_K_L | 24.08GB | 24.96GB | +3.7 % | 0.0027 ± 0.0001 | 0.0024 ± 0.0001 | 97.64 ± 0.10 % | 97.70 ± 0.09 % |
| Q6_K | 23.46GB | 23.86GB | +1.7 % | 0.0031 ± 0.0001 | 0.0030 ± 0.0001 | 97.49 ± 0.10 % | 97.39 ± 0.10 % |
| Q6_K_S | — | 22.86GB | — | — | 0.0036 ± 0.0002 | — | 97.35 ± 0.10 % |
| Q5_K_L | 21.54GB | — | — | 0.0050 ± 0.0002 | — | 96.92 ± 0.11 % | — |
| Q5_K_M | 20.75GB | 20.92GB | +0.8 % | 0.0054 ± 0.0001 | 0.0053 ± 0.0002 | 96.71 ± 0.11 % | 96.87 ± 0.11 % |
| Q5_K_S | 19.68GB | 19.57GB | −0.6 % | 0.0065 ± 0.0001 | 0.0060 ± 0.0002 | 96.25 ± 0.12 % | 96.40 ± 0.12 % |
| Q4_K_L | 18.72GB | 18.82GB | +0.6 % | 0.0124 ± 0.0003 | 0.0106 ± 0.0002 | 95.00 ± 0.14 % | 95.45 ± 0.13 % |
| Q4_K_M | 17.77GB | 17.44GB | −1.9 % | 0.0129 ± 0.0002 | 0.0139 ± 0.0004 | 95.02 ± 0.14 % | 95.00 ± 0.14 % |
| IQ4_NL | 16.33GB | 17.44GB | +6.8 % | 0.0178 ± 0.0003 | 0.0150 ± 0.0004 | 94.22 ± 0.15 % | 94.78 ± 0.14 % |
| Q3_K_XL | 16.39GB | — | — | 0.0426 ± 0.0007 | — | 91.01 ± 0.18 % | — |
| Q4_K_S | 16.71GB | 16.36GB | −2.1 % | 0.0165 ± 0.0003 | 0.0156 ± 0.0003 | 94.18 ± 0.15 % | 94.77 ± 0.14 % |
| IQ4_XS | 15.57GB | 15.48GB | −0.6 % | 0.0196 ± 0.0005 | 0.0188 ± 0.0005 | 94.03 ± 0.15 % | 94.14 ± 0.15 % |
| IQ3_M | 13.90GB | 14.86GB | +6.9 % | 0.0566 ± 0.0008 | 0.0406 ± 0.0006 | 89.66 ± 0.19 % | 91.06 ± 0.18 % |
| Q3_K_L | 15.28GB | 14.12GB | −7.6 % | 0.0437 ± 0.0007 | 0.0432 ± 0.0007 | 90.82 ± 0.18 % | 90.71 ± 0.18 % |
| Q3_K_M | 14.61GB | 13.40GB | −8.2 % | 0.0466 ± 0.0008 | 0.0564 ± 0.0009 | 90.57 ± 0.18 % | 89.81 ± 0.19 % |
| Q2_K_L | 13.08GB | — | — | 0.1322 ± 0.0018 | — | 84.37 ± 0.23 % | — |
| IQ3_XS | 13.33GB | 12.80GB | −4.0 % | 0.0624 ± 0.0009 | 0.0585 ± 0.0009 | 89.18 ± 0.20 % | 89.41 ± 0.19 % |
| Q3_K_S | 13.72GB | 12.74GB | −7.1 % | 0.0686 ± 0.0010 | 0.0648 ± 0.0010 | 88.17 ± 0.20 % | 88.95 ± 0.20 % |
| IQ3_XXS | 12.63GB | 12.32GB | −2.4 % | 0.0850 ± 0.0012 | 0.0739 ± 0.0011 | 87.34 ± 0.21 % | 88.51 ± 0.20 % |
| Q2_K | 11.84GB | 10.82GB | −8.6 % | 0.1357 ± 0.0019 | 0.1612 ± 0.0022 | 83.98 ± 0.23 % | 83.53 ± 0.23 % |
| IQ2_M | 10.87GB | 10.52GB | −3.2 % | 0.1551 ± 0.0021 | 0.1494 ± 0.0021 | 83.06 ± 0.23 % | 83.91 ± 0.23 % |
| IQ2_S | 10.30GB | 9.68GB | −5.9 % | 0.2166 ± 0.0026 | 0.2090 ± 0.0027 | 79.78 ± 0.25 % | 80.98 ± 0.25 % |
| IQ2_XS | 9.99GB | 9.09GB | −9.0 % | 0.2293 ± 0.0028 | 0.2403 ± 0.0028 | 79.53 ± 0.25 % | 79.70 ± 0.25 % |
| IQ2_XXS | 9.39GB | 8.88GB | −5.4 % | 0.3037 ± 0.0034 | 0.2834 ± 0.0033 | 76.29 ± 0.27 % | 78.04 ± 0.26 % |

</details>

## Perplexity

Perplexity and KL divergence measured against the bf16 model on `wiki.test.raw` over 100 chunks.

<details>
<summary>Click to view full KL-divergence statistics</summary>

- Reference model: bf16
- Dataset: wiki.test.raw
- Chunks: 100
- Base PPL: 6.744 ± 0.10317
- llama.cpp build: b10896

### Summary

| Quant | PPL(Q) | PPL(Q)/PPL(base) | PPL(Q)-PPL(base) | Mean KLD | Mean Δp | RMS Δp | Same top-p |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Q8_0 | 6.75142 ± 0.103316 | 1.00149 ± 0.000351 | 0.010043 ± 0.002378 | 0.000921 ± 6.7e-05 | -0.02 ± 0.006% | 0.977 ± 0.106% | 98.745 ± 0.07% |
| Q6_K_L | 6.73971 ± 0.103022 | 0.999752 ± 0.000524 | -0.001674 ± 0.003535 | 0.002399 ± 7.4e-05 | -0.027 ± 0.009% | 1.382 ± 0.058% | 97.698 ± 0.094% |
| Q6_K | 6.73999 ± 0.102975 | 0.999794 ± 0.000568 | -0.001387 ± 0.003832 | 0.002984 ± 0.000106 | -0.044 ± 0.01% | 1.523 ± 0.066% | 97.388 ± 0.1% |
| Q6_K_S | 6.74361 ± 0.103111 | 1.00033 ± 0.00061 | 0.002233 ± 0.004113 | 0.00357 ± 0.000153 | -0.031 ± 0.01% | 1.599 ± 0.054% | 97.353 ± 0.101% |
| Q5_K_M | 6.75391 ± 0.103262 | 1.00186 ± 0.000762 | 0.012531 ± 0.00514 | 0.005287 ± 0.00017 | -0.091 ± 0.013% | 2.103 ± 0.081% | 96.871 ± 0.109% |
| Q5_K_S | 6.76193 ± 0.103379 | 1.00305 ± 0.000822 | 0.020546 ± 0.005547 | 0.006012 ± 0.000195 | -0.12 ± 0.014% | 2.199 ± 0.069% | 96.4 ± 0.117% |
| Q4_K_L | 6.76195 ± 0.103389 | 1.00305 ± 0.001104 | 0.02057 ± 0.007443 | 0.010581 ± 0.000197 | -0.099 ± 0.018% | 2.841 ± 0.067% | 95.451 ± 0.13% |
| Q4_1 | 6.76072 ± 0.103288 | 1.00287 ± 0.001291 | 0.019337 ± 0.008695 | 0.015787 ± 0.000324 | -0.113 ± 0.022% | 3.483 ± 0.085% | 94.51 ± 0.143% |
| Q4_K_M | 6.76446 ± 0.103309 | 1.00342 ± 0.001235 | 0.023082 ± 0.008317 | 0.013878 ± 0.000366 | -0.153 ± 0.02% | 3.254 ± 0.081% | 94.996 ± 0.137% |
| IQ4_NL | 6.76445 ± 0.103307 | 1.00342 ± 0.001293 | 0.023072 ± 0.008703 | 0.014962 ± 0.000371 | -0.172 ± 0.021% | 3.411 ± 0.079% | 94.784 ± 0.139% |
| Q4_K_S | 6.77046 ± 0.103353 | 1.00431 ± 0.001274 | 0.029081 ± 0.008578 | 0.015601 ± 0.000295 | -0.215 ± 0.022% | 3.472 ± 0.08% | 94.773 ± 0.139% |
| Q4_0 | 6.86425 ± 0.105544 | 1.01823 ± 0.001735 | 0.122867 ± 0.011843 | 0.026524 ± 0.000451 | -0.266 ± 0.028% | 4.536 ± 0.101% | 92.784 ± 0.162% |
| IQ4_XS | 6.78031 ± 0.103638 | 1.00578 ± 0.001437 | 0.03893 ± 0.009676 | 0.018841 ± 0.000471 | -0.208 ± 0.024% | 3.793 ± 0.079% | 94.141 ± 0.147% |
| IQ3_M | 6.92355 ± 0.105518 | 1.02702 ± 0.002104 | 0.182168 ± 0.014211 | 0.040647 ± 0.000622 | -0.847 ± 0.036% | 5.783 ± 0.11% | 91.055 ± 0.179% |
| Q3_K_L | 6.90279 ± 0.106406 | 1.02394 ± 0.002157 | 0.161412 ± 0.014737 | 0.043241 ± 0.000706 | -0.355 ± 0.036% | 5.685 ± 0.101% | 90.714 ± 0.182% |
| Q3_K_M | 7.03035 ± 0.109156 | 1.04286 ± 0.002543 | 0.288969 ± 0.01779 | 0.056425 ± 0.00088 | -0.609 ± 0.041% | 6.584 ± 0.116% | 89.812 ± 0.189% |
| IQ3_XS | 6.99622 ± 0.10673 | 1.0378 ± 0.002569 | 0.254844 ± 0.017398 | 0.058524 ± 0.00086 | -1.006 ± 0.042% | 6.846 ± 0.114% | 89.408 ± 0.193% |
| Q3_K_S | 7.08464 ± 0.110429 | 1.05092 ± 0.002747 | 0.343256 ± 0.019408 | 0.064792 ± 0.000983 | -0.6 ± 0.044% | 7.094 ± 0.126% | 88.953 ± 0.196% |
| IQ3_XXS | 7.05932 ± 0.107866 | 1.04716 ± 0.002906 | 0.317942 ± 0.019746 | 0.07388 ± 0.00114 | -1.251 ± 0.048% | 7.79 ± 0.134% | 88.51 ± 0.2% |
| Q2_K | 7.43849 ± 0.114518 | 1.10341 ± 0.004572 | 0.697107 ± 0.031501 | 0.161216 ± 0.002239 | -2.38 ± 0.07% | 11.408 ± 0.163% | 83.525 ± 0.232% |
| IQ2_M | 7.52818 ± 0.116097 | 1.11671 ± 0.004474 | 0.786799 ± 0.031377 | 0.149377 ± 0.002056 | -2.608 ± 0.067% | 11.051 ± 0.153% | 83.906 ± 0.23% |
| IQ2_S | 7.72954 ± 0.118009 | 1.14658 ± 0.005361 | 0.988156 ± 0.036929 | 0.209045 ± 0.002695 | -3.736 ± 0.08% | 13.257 ± 0.164% | 80.976 ± 0.246% |
| IQ2_XS | 8.025 ± 0.122332 | 1.19041 ± 0.005935 | 1.28362 ± 0.04144 | 0.24027 ± 0.002841 | -4.846 ± 0.086% | 14.572 ± 0.165% | 79.698 ± 0.252% |
| IQ2_XXS | 8.41042 ± 0.130004 | 1.24758 ± 0.006905 | 1.66904 ± 0.049626 | 0.283397 ± 0.003313 | -5.486 ± 0.092% | 15.735 ± 0.171% | 78.039 ± 0.259% |

### KLD percentile ladder

| Quant | Max | 99.9% | 99% | 95% | 90% | Median | 10% | 5% | 1% | 0.1% | Min |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Q8_0 | 1.0809 | 0.040699 | 0.00706 | 0.002394 | 0.001448 | 0.000334 | 9e-06 | 2e-06 | -1e-06 | -8e-06 | -3e-05 |
| Q6_K_L | 0.973111 | 0.092973 | 0.01931 | 0.006956 | 0.004484 | 0.001173 | 2.3e-05 | 5e-06 | 0 | -4e-06 | -3.2e-05 |
| Q6_K | 1.92698 | 0.105185 | 0.024555 | 0.00898 | 0.005684 | 0.00142 | 3.2e-05 | 7e-06 | 0 | -3e-06 | -3.9e-05 |
| Q6_K_S | 3.01015 | 0.139894 | 0.030559 | 0.010436 | 0.006639 | 0.001579 | 3.7e-05 | 9e-06 | 0 | -3e-06 | -2e-05 |
| Q5_K_M | 2.84051 | 0.179695 | 0.051284 | 0.016652 | 0.010222 | 0.002319 | 5.9e-05 | 1.4e-05 | 1e-06 | -2e-06 | -3.5e-05 |
| Q5_K_S | 4.19186 | 0.204707 | 0.058225 | 0.019891 | 0.011923 | 0.002638 | 6.7e-05 | 1.7e-05 | 2e-06 | -1e-06 | -1.3e-05 |
| Q4_K_L | 2.23146 | 0.353113 | 0.102686 | 0.036515 | 0.02187 | 0.004622 | 0.000122 | 3e-05 | 3e-06 | -1e-06 | -1.1e-05 |
| Q4_1 | 4.48169 | 0.535538 | 0.15125 | 0.053299 | 0.032149 | 0.006836 | 0.000176 | 4.3e-05 | 5e-06 | -0 | -8e-06 |
| Q4_K_M | 6.50884 | 0.52521 | 0.142121 | 0.045992 | 0.027477 | 0.005795 | 0.000152 | 3.7e-05 | 3e-06 | -0 | -1.1e-05 |
| IQ4_NL | 5.86888 | 0.572585 | 0.148268 | 0.049484 | 0.030007 | 0.006205 | 0.00016 | 4.1e-05 | 4e-06 | -1e-06 | -1e-05 |
| Q4_K_S | 2.95896 | 0.534154 | 0.159605 | 0.054462 | 0.032237 | 0.006696 | 0.000178 | 4.5e-05 | 4e-06 | 0 | -6e-06 |
| Q4_0 | 4.82004 | 0.873295 | 0.263285 | 0.089089 | 0.054593 | 0.011859 | 0.000302 | 7.4e-05 | 8e-06 | 1e-06 | -1e-05 |
| IQ4_XS | 7.31304 | 0.646303 | 0.189096 | 0.064107 | 0.037538 | 0.007868 | 0.000206 | 5.5e-05 | 5e-06 | -0 | -2.5e-05 |
| IQ3_M | 3.71947 | 1.19807 | 0.405264 | 0.139634 | 0.084763 | 0.018335 | 0.000582 | 0.00015 | 1.7e-05 | 2e-06 | -1e-06 |
| Q3_K_L | 6.37739 | 1.27356 | 0.427362 | 0.15179 | 0.090328 | 0.019294 | 0.000469 | 0.000124 | 1.1e-05 | 1e-06 | -7e-06 |
| Q3_K_M | 6.22563 | 1.74283 | 0.582466 | 0.193088 | 0.116458 | 0.025449 | 0.000631 | 0.000151 | 1.6e-05 | 2e-06 | -2e-06 |
| IQ3_XS | 6.70819 | 1.54722 | 0.583696 | 0.205317 | 0.12211 | 0.026238 | 0.000848 | 0.000228 | 2.8e-05 | 4e-06 | -6e-06 |
| Q3_K_S | 6.36858 | 2.01934 | 0.657329 | 0.222889 | 0.135785 | 0.029351 | 0.000742 | 0.000188 | 1.9e-05 | 3e-06 | -0 |
| IQ3_XXS | 6.38787 | 2.27993 | 0.733495 | 0.2525 | 0.156451 | 0.033092 | 0.000938 | 0.000241 | 2.6e-05 | 4e-06 | -4e-06 |
| Q2_K | 8.55553 | 4.72795 | 1.66833 | 0.554146 | 0.340936 | 0.072722 | 0.00214 | 0.000494 | 5.3e-05 | 8e-06 | -0 |
| IQ2_M | 11.0817 | 4.06564 | 1.48056 | 0.513973 | 0.316133 | 0.068237 | 0.002261 | 0.000603 | 6.7e-05 | 1.1e-05 | -0 |
| IQ2_S | 11.4424 | 5.39947 | 1.96664 | 0.727937 | 0.448283 | 0.099357 | 0.003653 | 0.000949 | 0.000119 | 1.9e-05 | 4e-06 |
| IQ2_XS | 10.1463 | 5.58995 | 2.10155 | 0.839563 | 0.520093 | 0.123041 | 0.005352 | 0.001388 | 0.000189 | 2.8e-05 | 5e-06 |
| IQ2_XXS | 11.2416 | 6.63126 | 2.46692 | 0.972586 | 0.613898 | 0.147177 | 0.007022 | 0.001842 | 0.000237 | 3.7e-05 | 3e-06 |

</details>

Also available as [perplexity.md](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/perplexity.md).

## imatrix

All quants made using imatrix option with dataset from [here](https://gist.github.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d). The imatrix is available here: [Qwen3.8-27B-imatrix.gguf](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-imatrix.gguf).

## ARM/AVX information

llama.cpp automatically "repacks" weights into an interleaved layout at load time for faster inference on ARM and AVX machines - details in [this PR](https://github.com/ggml-org/llama.cpp/pull/9921). This once required downloading special Q4_0_4_4/4_8/8_8 files; those are long gone. Online repacking now covers Q4_0, IQ4_NL, and most K-quants, so no special quant choice is needed for CPU inference.

## Which file should I choose?

<details>
  <summary>Click here for details</summary>

An older (early 2024) but still useful write-up with charts comparing quant performances is provided by Artefact2 [here](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

The first thing to figure out is how big a model you can run. To do this, you'll need to figure out how much RAM and/or VRAM you have.

If you want your model running as FAST as possible, you'll want to fit the whole thing on your GPU's VRAM. Aim for a quant with a file size 1-2GB smaller than your GPU's total VRAM.

If you want the absolute maximum quality, add both your system RAM and your GPU's VRAM together, then similarly grab a quant with a file size 1-2GB Smaller than that total.

Hugging Face can also do this math for you: add your hardware in your [Local Apps settings](https://huggingface.co/settings/local-apps) and the model page will show which files fit.

Next, you'll need to decide if you want to use an 'I-quant' or a 'K-quant'.

If you don't want to think too much, grab one of the K-quants. These are in format 'QX_K_X', like Q5_K_M.

If you want to get more into the weeds, you can check out this extremely useful feature chart:

[llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)

But basically, if you're aiming for below Q4, and you're running cuBLAS (Nvidia) or rocBLAS (AMD), you should look towards the I-quants. These are in format IQX_X, like IQ3_M. These are newer and offer better performance for their size.

These I-quants can also be used on CPU, but will be slower than their K-quant equivalent, so speed vs performance is a tradeoff you'll have to decide.

</details>

## Credits

Thank you kalomaze and Dampf for assistance in creating the imatrix calibration dataset.

Thank you ZeroWw for the inspiration to experiment with embed/output.

Want to support my work? Visit my ko-fi page here: https://ko-fi.com/bartowski