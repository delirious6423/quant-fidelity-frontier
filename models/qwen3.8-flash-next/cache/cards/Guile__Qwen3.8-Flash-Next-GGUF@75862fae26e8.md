---
quantized_by: bartowski
pipeline_tag: image-text-to-text
license: other
license_link: LICENSE
base_model_relation: quantized
license_name: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
---

## Llamacpp imatrix Quantizations of Qwen3.8-Flash-Next by Qwen

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b10665">b10665</a> for quantization.

Original model: https://huggingface.co/Qwen/Qwen3.8-Flash-Next

**Model details:**
- Parameter count: 180B
- Input support: text, image (with mmproj file) - [details](#multimodal)
- Speculative decoding: no
- imatrix: yes - [details](#imatrix)

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

**Don't know which to choose?** Grab [Q4_K_M](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_K_M) (119.60GB) - usually a good mix of size and performance. Download instructions available [here](#downloading-using-the-hugging-face-cli)

## Available files:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [Qwen3.8-Flash-Next-bf16.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-bf16) | bf16 | 354.03GB | true | Full BF16 weights. |
| [Qwen3.8-Flash-Next-Q8_0.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q8_0) | Q8_0 | 188.26GB | true | Extremely high quality, generally unneeded but max available quant. |
| [Qwen3.8-Flash-Next-Q6_K.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q6_K) | Q6_K | 168.11GB | true | Very high quality, near perfect, *recommended*. |
| [Qwen3.8-Flash-Next-Q5_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q5_K_L) | Q5_K_L | 151.06GB | true | Uses Q8_0 for embed and output weights. High quality, *recommended*. |
| [Qwen3.8-Flash-Next-Q4_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_K_L) | Q4_K_L | 139.27GB | true | Uses Q8_0 for embed and output weights. Good quality, *recommended*. |
| [Qwen3.8-Flash-Next-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q5_K_M) | Q5_K_M | 134.67GB | true | High quality, *recommended*. |
| [Qwen3.8-Flash-Next-Q5_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q5_K_S) | Q5_K_S | 128.10GB | true | High quality, *recommended*. |
| [Qwen3.8-Flash-Next-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_K_M) | Q4_K_M | 119.60GB | true | Good quality, default size for most use cases, *recommended*. |
| [Qwen3.8-Flash-Next-Q3_K_XL.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q3_K_XL) | Q3_K_XL | 119.35GB | true | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability. |
| [Qwen3.8-Flash-Next-Q4_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_K_S) | Q4_K_S | 113.14GB | true | Slightly lower quality with more space savings, *recommended*. |
| [Qwen3.8-Flash-Next-Q4_1.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_1) | Q4_1 | 111.26GB | true | Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon. |
| [Qwen3.8-Flash-Next-Q2_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q2_K_L) | Q2_K_L | 107.15GB | true | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable. |
| [Qwen3.8-Flash-Next-Q4_0.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q4_0) | Q4_0 | 100.60GB | true | Legacy format, kept for compatibility with older tools. |
| [Qwen3.8-Flash-Next-IQ4_NL.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ4_NL) | IQ4_NL | 100.29GB | true | Similar to IQ4_XS, but slightly larger. |
| [Qwen3.8-Flash-Next-IQ4_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ4_XS) | IQ4_XS | 97.68GB | true | Decent quality, smaller than Q4_K_S with similar performance, *recommended*. |
| [Qwen3.8-Flash-Next-IQ3_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ3_M) | IQ3_M | 93.23GB | true | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| [Qwen3.8-Flash-Next-Q3_K_L.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q3_K_L) | Q3_K_L | 93.19GB | true | Lower quality but usable, good for low RAM availability. |
| [Qwen3.8-Flash-Next-Q3_K_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q3_K_M) | Q3_K_M | 91.96GB | true | Low quality. |
| [Qwen3.8-Flash-Next-IQ3_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ3_XS) | IQ3_XS | 91.95GB | true | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| [Qwen3.8-Flash-Next-Q3_K_S.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q3_K_S) | Q3_K_S | 89.44GB | true | Low quality, not recommended. |
| [Qwen3.8-Flash-Next-IQ3_XXS.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ3_XXS) | IQ3_XXS | 88.02GB | true | Lower quality, new method with decent performance, comparable to Q3 quants. |
| [Qwen3.8-Flash-Next-Q2_K.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-Q2_K) | Q2_K | 80.93GB | true | Very low quality but surprisingly usable. |
| [Qwen3.8-Flash-Next-IQ2_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ2_M) | IQ2_M | 80.37GB | true | Relatively low quality, uses SOTA techniques to be surprisingly usable. |
| [Qwen3.8-Flash-Next-IQ2_S.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ2_S) | IQ2_S | 77.83GB | true | Low quality, uses SOTA techniques to be usable. |
| [Qwen3.8-Flash-Next-IQ2_XS.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ2_XS) | IQ2_XS | 77.74GB | true | Low quality, uses SOTA techniques to be usable. |
| [Qwen3.8-Flash-Next-IQ2_XXS.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ2_XXS) | IQ2_XXS | 75.19GB | true | Very low quality, uses SOTA techniques to be usable. |
| [Qwen3.8-Flash-Next-IQ1_M.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ1_M) | IQ1_M | 72.01GB | true | Extremely low quality, *not* recommended. |
| [Qwen3.8-Flash-Next-IQ1_S.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ1_S) | IQ1_S | 70.10GB | true | Extremely low quality, *not* recommended. |

Download a specific file:

```
hf download bartowski/Qwen3.8-Flash-Next-GGUF --include "Qwen3.8-Flash-Next-Q4_K_M/*" --local-dir ./
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
hf download bartowski/Qwen3.8-Flash-Next-GGUF --include "Qwen3.8-Flash-Next-Q4_K_M/*" --local-dir ./
```

The files marked `true` in the Split column above are stored as multiple parts in a folder. To download all the parts to a local folder, run:

```
hf download bartowski/Qwen3.8-Flash-Next-GGUF --include "Qwen3.8-Flash-Next-Q8_0/*" --local-dir ./
```

You can either specify a new local-dir (Qwen3.8-Flash-Next-Q8_0) or download them all in place (./)

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/Qwen3.8-Flash-Next-GGUF:Q4_K_M
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b10665 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## Multimodal

This model supports image input. Alongside the quants, this repo includes the multimodal projector files [mmproj-Qwen3.8-Flash-Next-f16.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/blob/main/mmproj-Qwen3.8-Flash-Next-f16.gguf) and [mmproj-Qwen3.8-Flash-Next-bf16.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/blob/main/mmproj-Qwen3.8-Flash-Next-bf16.gguf), which pair with any quant above.

llama.cpp downloads the mmproj automatically when using `-hf` as shown above; if you're loading files manually, pass it with `--mmproj`.

## imatrix

All quants made using imatrix option, with a calibration corpus rendered through this model's own chat template. The corpus pairs plain prose with tool-calling and reasoning conversations ([corpus source data](https://gist.github.com/bartowski1182/e26453c0404e24eb317543ec5360f87a)), encoded exactly as this model sees them at inference and processed with `--parse-special`, so chat-format special tokens contribute to the importance matrix. The corpus rendered for this model is included in this repo: [Qwen3.8-Flash-Next-calibration-v6.txt](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/blob/main/Qwen3.8-Flash-Next-calibration-v6.txt). The imatrix is available here: [Qwen3.8-Flash-Next-imatrix.gguf](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/blob/main/Qwen3.8-Flash-Next-imatrix.gguf).

<details>
<summary>Calibration render details</summary>

```json
{
  "generator": "auto_quant_v2 calibration renderer",
  "recipe": "calibration-v6",
  "model": "Qwen3.8-Flash-Next",
  "encoder": "chat_template",
  "chunk_size": 512,
  "prose_chunks": 214,
  "tool_chunks": 369,
  "total_chunks": 583,
  "tool_chunk_fraction": 0.633,
  "n_conversations": 137,
  "extension_convs_used": 0,
  "conversation_token_lengths": [
    604,
    1667,
    1297,
    1559,
    1185,
    1404,
    3220,
    866,
    1267,
    1462,
    1107,
    2136,
    921,
    1293,
    2829,
    1313,
    1160,
    1026,
    777,
    758,
    1402,
    1098,
    1447,
    1242,
    1912,
    1538,
    1694,
    948,
    1455,
    1686,
    1609,
    1260,
    1308,
    1077,
    1060,
    1752,
    1691,
    1215,
    517,
    1950,
    1451,
    1166,
    1438,
    2042,
    2130,
    1355,
    1649,
    947,
    2996,
    1142,
    2908,
    827,
    1064,
    1010,
    947,
    738,
    2530,
    955,
    1188,
    1127,
    1271,
    1211,
    966,
    1259,
    1223,
    1628,
    953,
    1592,
    2166,
    881,
    365,
    1162,
    3375,
    2876,
    749,
    982,
    1072,
    1113,
    1338,
    1122,
    1191,
    835,
    1250,
    1111,
    1331,
    1586,
    1443,
    2106,
    913,
    698,
    2806,
    682,
    1443,
    1715,
    1987,
    1243,
    683,
    1396,
    1168,
    1769,
    1884,
    1755,
    857,
    1055,
    1077,
    2896,
    780,
    774,
    803,
    1452,
    1092,
    1617,
    808,
    392,
    361,
    2666,
    1058,
    1179,
    1870,
    2079,
    2735,
    2717,
    856,
    1020,
    880,
    999,
    1279,
    1039,
    882,
    1397,
    870,
    750,
    1783,
    1067,
    957,
    1356,
    1519
  ],
  "warnings": []
}
```

</details>

## Embed/output weights

Some of these quants (Q3_K_XL, Q4_K_L etc) are the standard quantization method with the embeddings and output weights quantized to Q8_0 instead of what they would normally default to.

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
