---
quantized_by: bartowski
pipeline_tag: text-generation
base_model_relation: quantized
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
license: other
tags:
- liquid
- lfm2.5
- edge
- moe
base_model: LiquidAI/LFM2.5-24B-A2B
license_name: lfm1.0
license_link: LICENSE
---

## Llamacpp imatrix Quantizations of LFM2.5-24B-A2B by LiquidAI

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b11606">b11606</a> for quantization.

Original model: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B

**Model details:**
- Parameter count: 24B (source checkpoint)
- Input support: text
- Speculative decoding: no
- imatrix: yes - [details](#imatrix)

[How to run](#how-to-run)

## Prompt format

```
<|startoftext|><|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

<details><summary>Prompt format with tool definitions</summary>

```
<|startoftext|><|im_start|>system
{system_prompt}
List of tools: [{"type": "function", "function": {"name": "get_stock_price", "description": "Get the current stock price", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL, GOOG"}}, "required": ["symbol"]}}}]<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

</details>

**Don't know which to choose?** Grab [Q4_K_M](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_M.gguf) (15.35GB) - usually a good mix of size and performance. Download instructions available [here](#downloading-using-the-hugging-face-cli)

## Available files:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [LFM2.5-24B-A2B-bf16.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-bf16.gguf) | bf16 | 47.96GB | false | Full BF16 weights. |
| [LFM2.5-24B-A2B-Q8_0.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q8_0.gguf) | Q8_0 | 25.49GB | false | Extremely high quality, generally unneeded but max available quant. |
| [LFM2.5-24B-A2B-Q6_K_L.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q6_K_L.gguf) | Q6_K_L | 22.23GB | false | The large size of Q6_K, about halfway to Q8_0 in size. Very high quality, near perfect, *recommended*. |
| [LFM2.5-24B-A2B-Q6_K.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q6_K.gguf) | Q6_K | 21.15GB | false | Very high quality, near perfect, *recommended*. |
| [LFM2.5-24B-A2B-Q6_K_S.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q6_K_S.gguf) | Q6_K_S | 20.18GB | false | Very high quality, near perfect, a little smaller than Q6_K with almost all of the model at Q6_K precision, *recommended*. |
| [LFM2.5-24B-A2B-Q5_K_M.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_K_M.gguf) | Q5_K_M | 18.50GB | false | High quality, *recommended*. |
| [LFM2.5-24B-A2B-Q5_K_S.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q5_K_S.gguf) | Q5_K_S | 17.19GB | false | High quality, *recommended*. |
| [LFM2.5-24B-A2B-Q4_K_L.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_L.gguf) | Q4_K_L | 16.37GB | false | The large size of Q4_K, between Q4_K_M and Q5_K_S: more of the most sensitive weights kept at higher precision, *recommended*. |
| [LFM2.5-24B-A2B-Q4_K_M.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_M.gguf) | Q4_K_M | 15.35GB | false | Good quality, default size for most use cases, *recommended*. |
| [LFM2.5-24B-A2B-IQ4_NL.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ4_NL.gguf) | IQ4_NL | 15.18GB | false | Similar to IQ4_XS, but slightly larger. |
| [LFM2.5-24B-A2B-Q4_1.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_1.gguf) | Q4_1 | 15.07GB | false | Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon. |
| [LFM2.5-24B-A2B-Q4_K_S.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_K_S.gguf) | Q4_K_S | 14.27GB | false | Slightly lower quality with more space savings, *recommended*. |
| [LFM2.5-24B-A2B-Q4_0.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q4_0.gguf) | Q4_0 | 13.65GB | false | Legacy format, kept for compatibility with older tools. |
| [LFM2.5-24B-A2B-IQ4_XS.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ4_XS.gguf) | IQ4_XS | 13.38GB | false | Decent quality, smaller than Q4_K_S with similar performance, *recommended*. |
| [LFM2.5-24B-A2B-IQ3_M.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_M.gguf) | IQ3_M | 12.59GB | false | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| [LFM2.5-24B-A2B-Q3_K_L.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_L.gguf) | Q3_K_L | 12.24GB | false | Lower quality but usable, good for low RAM availability. |
| [LFM2.5-24B-A2B-Q3_K_M.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_M.gguf) | Q3_K_M | 11.55GB | false | Low quality. |
| [LFM2.5-24B-A2B-Q3_K_S.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q3_K_S.gguf) | Q3_K_S | 10.98GB | false | Low quality, *not* recommended. |
| [LFM2.5-24B-A2B-IQ3_XS.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_XS.gguf) | IQ3_XS | 10.88GB | false | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| [LFM2.5-24B-A2B-IQ3_XXS.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ3_XXS.gguf) | IQ3_XXS | 10.33GB | false | Lower quality, new method with decent performance, comparable to Q3 quants. |
| [LFM2.5-24B-A2B-Q2_K.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-Q2_K.gguf) | Q2_K | 9.18GB | false | Very low quality but surprisingly usable. |
| [LFM2.5-24B-A2B-IQ2_M.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_M.gguf) | IQ2_M | 8.78GB | false | Relatively low quality, uses SOTA techniques to be surprisingly usable. |
| [LFM2.5-24B-A2B-IQ2_S.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_S.gguf) | IQ2_S | 7.96GB | false | Low quality, uses SOTA techniques to be usable. |
| [LFM2.5-24B-A2B-IQ2_XS.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-IQ2_XS.gguf) | IQ2_XS | 7.49GB | false | Low quality, uses SOTA techniques to be usable. |

Download a specific file:

```
hf download bartowski/LFM2.5-24B-A2B-GGUF --include "LFM2.5-24B-A2B-Q4_K_M.gguf" --local-dir ./
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
hf download bartowski/LFM2.5-24B-A2B-GGUF --include "LFM2.5-24B-A2B-Q4_K_M.gguf" --local-dir ./
```

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/LFM2.5-24B-A2B-GGUF:Q4_K_M
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b11606 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## Per-tensor layouts

Some of these files were built with a layout computed for this model instead of llama.cpp's standard one-size-fits-all rules. A Q4_K_M is still mostly Q4_K; the extra precision goes to the weights this particular model is most sensitive to. The S, M or L in a name says how much of the model stays at the base precision: about 90 % for S, 70 % for M and 50 % for L. An `_L` name is simply the large size of its family. Q4_K_L is to Q4_K_M what Q4_K_M is to Q4_K_S; Q6_K_S, Q6_K and Q6_K_L are the small, medium and large sizes of Q6_K, with Q6_K_L about halfway to Q8_0. In earlier releases an `_L` name meant the embedding and output weights were kept at Q8_0; in these files it means the larger size of the base type. There is no size target, so each file's bits per weight is reported rather than promised.

The layout each of these files was built with is published in the [`layouts/`](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/tree/main/layouts) folder: `<file>.tensor-types.txt` is the exact `--tensor-type-file` given to `llama-quantize`, and `<file>.layout.json` records how it was computed, including the generator version, the llama.cpp release and the commit, so any of them can be rebuilt. The code that computed them is public at [`quantization-config`](https://github.com/bartowski1182/quantization-config); its tag `key-3bf8b43e20a68f0b` is the exact snapshot these files record. The method is described in [this write-up](https://huggingface.co/blog/bartowski/per-tensor-layout-maps-for-gguf-quantization).

Checked on this model before any of these files were released: Q4_K_M reached 0.75×, Q3_K_M 0.62× and IQ2_XS 0.92× the KL divergence of the standard layout at the same file size.

<details>
<summary>Layout details</summary>

Files built from a computed layout:

| Quant | Size | Body bits/weight | File bits/weight | Body kept at base type |
| ----- | ---- | ---------------- | ---------------- | ---------------------- |
| Q6_K_L | 22.23GB | 7.40 | 7.42 | 50 % |
| Q6_K | 21.15GB | 7.04 | 7.06 | 70 % |
| Q6_K_S | 20.18GB | 6.71 | 6.73 | 91 % |
| Q5_K_M | 18.50GB | 6.14 | 6.17 | 70 % |
| Q5_K_S | 17.19GB | 5.70 | 5.74 | 90 % |
| Q4_K_L | 16.37GB | 5.42 | 5.46 | 50 % |
| Q4_K_M | 15.35GB | 5.08 | 5.12 | 70 % |
| IQ4_NL | 15.18GB | 5.02 | 5.07 | 70 % |
| Q4_K_S | 14.27GB | 4.71 | 4.76 | 90 % |
| IQ4_XS | 13.38GB | 4.41 | 4.46 | 90 % |
| IQ3_M | 12.59GB | 4.15 | 4.20 | 50 % |
| Q3_K_L | 12.24GB | 4.03 | 4.09 | 50 % |
| Q3_K_M | 11.55GB | 3.79 | 3.86 | 70 % |
| Q3_K_S | 10.98GB | 3.60 | 3.66 | 90 % |
| IQ3_XS | 10.88GB | 3.57 | 3.63 | 90 % |
| IQ3_XXS | 10.33GB | 3.38 | 3.45 | 70 % |
| Q2_K | 9.18GB | 2.99 | 3.06 | 70 % |
| IQ2_M | 8.78GB | 2.86 | 2.93 | 70 % |
| IQ2_S | 7.96GB | 2.58 | 2.66 | 70 % |
| IQ2_XS | 7.49GB | 2.42 | 2.50 | 90 % |

Checked on this model: the computed layout against the standard one, measured by KL divergence against the unquantized model. The ratio compares each computed file with the standard ladder read at that file's own size, so it can differ from the two KLD columns when the two files differ in size.

| Quant | Computed layout KLD | Standard layout KLD | Ratio at equal size | Size vs standard file |
| ----- | ------------------- | ------------------- | ------------------- | --------------------- |
| Q4_K_M | 0.1296 ± 0.0016 | 0.2031 ± 0.0022 | 0.75× | +5.5 % |
| Q3_K_M | 0.2276 ± 0.0023 | 0.4134 ± 0.0036 | 0.62× | +5.5 % |
| IQ2_XS | 0.5838 ± 0.0046 | 0.6562 ± 0.0049 | 0.92× | +4.1 % |

How it works: the base type is a floor for every body tensor and a fixed share of the body bytes stays at it (90 % for S, 70 % for M, 50 % for L); the remaining bytes go where a cross-model sensitivity prior, measured by KL divergence against the unquantized model, says they buy the most quality. The embedding and output tensors are sized by their share of the file: a small table is kept at Q8_0, a large one follows the file's bitrate. A K-quant and the IQ quant with the same base bitrate (Q3_K_S and IQ3_XS, Q3_K_M and IQ3_S, Q3_K_L and IQ3_M) come out at about the same size; the IQ file is the GPU-oriented twin.

</details>

## imatrix

All quants made using imatrix option, with a calibration corpus rendered through this model's own chat template. The corpus pairs plain prose with tool-calling and reasoning conversations ([corpus source data](https://gist.github.com/bartowski1182/e26453c0404e24eb317543ec5360f87a)), encoded exactly as this model sees them at inference and processed with `--parse-special`, so chat-format special tokens contribute to the importance matrix. The corpus rendered for this model is included in this repo: [LFM2.5-24B-A2B-calibration-v6.txt](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-calibration-v6.txt). The imatrix is available here: [LFM2.5-24B-A2B-imatrix.gguf](https://huggingface.co/bartowski/LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B-imatrix.gguf).

<details>
<summary>Calibration render details</summary>

```json
{
  "recipe": "calibration-v6",
  "encoder": "chat_template",
  "library_versions": {
    "transformers": "5.9.0",
    "tokenizers": "0.22.2",
    "tiktoken": "0.14.0",
    "blobfile": "3.3.0",
    "huggingface_hub": "1.31.0"
  },
  "chunk_size": 512,
  "prose_chunks": 216,
  "tool_chunks": 271,
  "total_chunks": 487,
  "n_conversations": 137,
  "conversation_token_lengths": [
    369,
    1392,
    638,
    1111,
    849,
    1134,
    2302,
    373,
    982,
    921,
    834,
    1840,
    675,
    721,
    2158,
    983,
    931,
    802,
    548,
    478,
    1173,
    768,
    1070,
    1002,
    1506,
    1269,
    806,
    463,
    755,
    1209,
    1267,
    537,
    1013,
    826,
    1046,
    1406,
    1357,
    957,
    285,
    1672,
    1087,
    852,
    1140,
    1560,
    1709,
    1052,
    1367,
    508,
    2181,
    875,
    2435,
    528,
    751,
    519,
    858,
    506,
    2016,
    490,
    920,
    880,
    955,
    979,
    682,
    756,
    629,
    1092,
    721,
    1235,
    1625,
    648,
    315,
    880,
    2803,
    2575,
    517,
    497,
    783,
    562,
    1029,
    908,
    882,
    589,
    776,
    791,
    1071,
    1224,
    1087,
    1133,
    625,
    444,
    2219,
    640,
    854,
    1372,
    1730,
    1002,
    392,
    1090,
    1045,
    1421,
    1504,
    1371,
    618,
    552,
    801,
    2533,
    532,
    343,
    555,
    971,
    859,
    1062,
    568,
    351,
    311,
    2069,
    492,
    852,
    1465,
    1457,
    2110,
    2136,
    440,
    743,
    642,
    598,
    988,
    784,
    656,
    1113,
    624,
    527,
    1149,
    736,
    717,
    1049,
    1181
  ]
}
```

</details>

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

Want to support my work? Visit my ko-fi page here: https://ko-fi.com/bartowski
