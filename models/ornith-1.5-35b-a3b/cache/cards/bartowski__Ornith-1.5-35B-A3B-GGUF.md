---
quantized_by: bartowski
pipeline_tag: image-text-to-text
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
license: mit
---

## Llamacpp imatrix Quantizations of Ornith-1.5-35B-A3B by ornith-ai

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b10472">b10472</a> for quantization.

Original model: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B

**Model details:**
- Parameter count: 36B
- Input support: text, image (with mmproj file) - [details](#multimodal)
- Speculative decoding: yes (MTP) - [details](#mtp)
- imatrix: yes - [details](#imatrix)

[How to run](#how-to-run)

## Prompt format

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

**Don't know which to choose?** Grab [Q4_K_M](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_K_M.gguf) (21.86GB) - usually a good mix of size and performance. Download instructions available [here](#downloading-using-the-hugging-face-cli)

## Available files:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [Ornith-1.5-35B-A3B-bf16.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/tree/main/Ornith-1.5-35B-A3B-bf16) | bf16 | 71.07GB | true | Full BF16 weights. |
| [Ornith-1.5-35B-A3B-Q8_0.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q8_0.gguf) | Q8_0 | 37.81GB | false | Extremely high quality, generally unneeded but max available quant. |
| [Ornith-1.5-35B-A3B-Q6_K_L.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q6_K_L.gguf) | Q6_K_L | 30.77GB | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, *recommended*. |
| [Ornith-1.5-35B-A3B-Q6_K.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q6_K.gguf) | Q6_K | 30.53GB | false | Very high quality, near perfect, *recommended*. |
| [Ornith-1.5-35B-A3B-Q5_K_L.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q5_K_L.gguf) | Q5_K_L | 25.81GB | false | Uses Q8_0 for embed and output weights. High quality, *recommended*. |
| [Ornith-1.5-35B-A3B-Q5_K_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q5_K_M.gguf) | Q5_K_M | 25.49GB | false | High quality, *recommended*. |
| [Ornith-1.5-35B-A3B-Q5_K_S.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q5_K_S.gguf) | Q5_K_S | 24.63GB | false | High quality, *recommended*. |
| [Ornith-1.5-35B-A3B-Q4_1.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_1.gguf) | Q4_1 | 22.45GB | false | Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon. |
| [Ornith-1.5-35B-A3B-Q4_K_L.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_K_L.gguf) | Q4_K_L | 22.24GB | false | Uses Q8_0 for embed and output weights. Good quality, *recommended*. |
| [Ornith-1.5-35B-A3B-Q4_K_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_K_M.gguf) | Q4_K_M | 21.86GB | false | Good quality, default size for most use cases, *recommended*. |
| [Ornith-1.5-35B-A3B-Q4_K_S.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_K_S.gguf) | Q4_K_S | 21.07GB | false | Slightly lower quality with more space savings, *recommended*. |
| [Ornith-1.5-35B-A3B-Q4_0.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q4_0.gguf) | Q4_0 | 20.42GB | false | Legacy format, kept for compatibility with older tools. |
| [Ornith-1.5-35B-A3B-IQ4_NL.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ4_NL.gguf) | IQ4_NL | 20.33GB | false | Similar to IQ4_XS, but slightly larger. |
| [Ornith-1.5-35B-A3B-IQ4_XS.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ4_XS.gguf) | IQ4_XS | 19.28GB | false | Decent quality, smaller than Q4_K_S with similar performance, *recommended*. |
| [Ornith-1.5-35B-A3B-Q3_K_XL.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q3_K_XL.gguf) | Q3_K_XL | 17.80GB | false | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability. |
| [Ornith-1.5-35B-A3B-IQ3_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ3_M.gguf) | IQ3_M | 17.37GB | false | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| [Ornith-1.5-35B-A3B-Q3_K_L.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q3_K_L.gguf) | Q3_K_L | 17.36GB | false | Lower quality but usable, good for low RAM availability. |
| [Ornith-1.5-35B-A3B-Q3_K_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q3_K_M.gguf) | Q3_K_M | 16.70GB | false | Low quality. |
| [Ornith-1.5-35B-A3B-IQ3_XS.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ3_XS.gguf) | IQ3_XS | 16.69GB | false | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| [Ornith-1.5-35B-A3B-Q3_K_S.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q3_K_S.gguf) | Q3_K_S | 15.98GB | false | Low quality, not recommended. |
| [Ornith-1.5-35B-A3B-IQ3_XXS.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ3_XXS.gguf) | IQ3_XXS | 15.34GB | false | Lower quality, new method with decent performance, comparable to Q3 quants. |
| [Ornith-1.5-35B-A3B-Q2_K_L.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q2_K_L.gguf) | Q2_K_L | 13.58GB | false | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable. |
| [Ornith-1.5-35B-A3B-Q2_K.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-Q2_K.gguf) | Q2_K | 13.09GB | false | Very low quality but surprisingly usable. |
| [Ornith-1.5-35B-A3B-IQ2_M.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ2_M.gguf) | IQ2_M | 12.54GB | false | Relatively low quality, uses SOTA techniques to be surprisingly usable. |
| [Ornith-1.5-35B-A3B-IQ2_S.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ2_S.gguf) | IQ2_S | 11.49GB | false | Low quality, uses SOTA techniques to be usable. |
| [Ornith-1.5-35B-A3B-IQ2_XS.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ2_XS.gguf) | IQ2_XS | 11.27GB | false | Low quality, uses SOTA techniques to be usable. |
| [Ornith-1.5-35B-A3B-IQ2_XXS.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-IQ2_XXS.gguf) | IQ2_XXS | 10.26GB | false | Very low quality, uses SOTA techniques to be usable. |

Download a specific file:

```
hf download bartowski/Ornith-1.5-35B-A3B-GGUF --include "Ornith-1.5-35B-A3B-Q4_K_M.gguf" --local-dir ./
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
hf download bartowski/Ornith-1.5-35B-A3B-GGUF --include "Ornith-1.5-35B-A3B-Q4_K_M.gguf" --local-dir ./
```

The files marked `true` in the Split column above are stored as multiple parts in a folder. To download all the parts to a local folder, run:

```
hf download bartowski/Ornith-1.5-35B-A3B-GGUF --include "Ornith-1.5-35B-A3B-bf16/*" --local-dir ./
```

You can either specify a new local-dir (Ornith-1.5-35B-A3B-bf16) or download them all in place (./)

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/Ornith-1.5-35B-A3B-GGUF:Q4_K_M
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b10472 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## Multimodal

This model supports image input. Alongside the quants, this repo includes the multimodal projector files [mmproj-Ornith-1.5-35B-A3B-f16.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/mmproj-Ornith-1.5-35B-A3B-f16.gguf) and [mmproj-Ornith-1.5-35B-A3B-bf16.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/mmproj-Ornith-1.5-35B-A3B-bf16.gguf), which pair with any quant above.

llama.cpp downloads the mmproj automatically when using `-hf` as shown above; if you're loading files manually, pass it with `--mmproj`.

## MTP

This model has MTP (Multi-Token Prediction) layers, and they are included in these quants

MTP layers act as a built-in draft model, letting llama.cpp run speculative decoding for faster generation. To use them, add the following flag to your llama.cpp command:

```
--spec-type draft-mtp
```

Note: the MTP layers are stored at Q4_0 in the imatrix quants (except for the Q8_0 quant), since imatrix calibration does not exercise them. Q4_0 is chosen for its speed which massively benefits MTP performance.

## imatrix

All quants made using imatrix option, with a calibration corpus rendered through this model's own chat template. The corpus pairs plain prose with tool-calling and reasoning conversations ([corpus source data](https://gist.github.com/bartowski1182/e26453c0404e24eb317543ec5360f87a)), encoded exactly as this model sees them at inference and processed with `--parse-special`, so chat-format special tokens contribute to the importance matrix. The corpus rendered for this model is included in this repo: [Ornith-1.5-35B-A3B-calibration-v6.txt](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-calibration-v6.txt). The imatrix is available here: [Ornith-1.5-35B-A3B-imatrix.gguf](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/blob/main/Ornith-1.5-35B-A3B-imatrix.gguf).

<details>
<summary>Calibration render details</summary>

```json
{
  "generator": "auto_quant_v2 calibration renderer",
  "recipe": "calibration-v6",
  "model": "Ornith-1.5-35B-A3B",
  "encoder": "chat_template",
  "chunk_size": 512,
  "prose_chunks": 214,
  "tool_chunks": 359,
  "total_chunks": 573,
  "tool_chunk_fraction": 0.627,
  "n_conversations": 137,
  "extension_convs_used": 0,
  "conversation_token_lengths": [
    566,
    1629,
    1259,
    1521,
    1147,
    1366,
    3182,
    828,
    1229,
    1424,
    1069,
    2098,
    883,
    1255,
    2791,
    1275,
    1122,
    988,
    739,
    720,
    1364,
    1060,
    1409,
    1204,
    1874,
    1500,
    1656,
    910,
    1417,
    1648,
    1571,
    1222,
    1270,
    1039,
    1018,
    1714,
    1653,
    1177,
    479,
    1912,
    1413,
    1128,
    1400,
    2004,
    2092,
    1317,
    1611,
    909,
    2958,
    1104,
    2870,
    789,
    1026,
    972,
    905,
    700,
    2492,
    917,
    1150,
    1089,
    1233,
    1173,
    928,
    1221,
    1185,
    1590,
    915,
    1554,
    2128,
    843,
    323,
    1124,
    3337,
    2838,
    711,
    944,
    1034,
    1075,
    1300,
    1084,
    1153,
    797,
    1212,
    1073,
    1293,
    1548,
    1405,
    2068,
    875,
    660,
    2768,
    640,
    1405,
    1677,
    1949,
    1205,
    645,
    1358,
    1126,
    1731,
    1846,
    1717,
    819,
    1017,
    1039,
    2858,
    742,
    736,
    765,
    1414,
    1054,
    1579,
    770,
    350,
    319,
    2628,
    1020,
    1141,
    1832,
    2041,
    2697,
    2679,
    818,
    982,
    842,
    961,
    1241,
    1001,
    844,
    1359,
    832,
    712,
    1745,
    1029,
    919,
    1318,
    1481
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
