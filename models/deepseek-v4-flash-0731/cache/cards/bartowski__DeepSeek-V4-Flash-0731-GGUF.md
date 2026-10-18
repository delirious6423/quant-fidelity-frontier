---
quantized_by: bartowski
pipeline_tag: text-generation
license: mit
base_model_relation: quantized
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
---

## Llamacpp Quantizations of DeepSeek-V4-Flash-0731 by deepseek-ai

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b10173">b10173</a> for quantization.

Original model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

This model is in MXFP4 and as such has only been provided in MXFP4 format!

Other sizes may be provided after some investigation.

**Model details:**
- Parameter count: 284B
- Input support: text
- MTP: yes - [details](#mtp-dspark)
- imatrix: no (for now)

[How to run](#how-to-run)

## Prompt format

No chat template specified so default is used. This may be incorrect, check original model card for details.

```
{system_prompt}<｜User｜>{prompt}<｜Assistant｜></think>
```

## Download the MXFP4 files:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [DeepSeek-V4-Flash-0731-MXFP4.gguf](https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF/tree/main/DeepSeek-V4-Flash-0731-MXFP4) | MXFP4 | 156.38GB | true | Original quality. |

## Downloading using the Hugging Face CLI

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have the Hugging Face CLI installed:

```
pip install -U "huggingface_hub[cli]"
```

The files marked `true` in the Split column above are stored as multiple parts in a folder. To download all the parts to a local folder, run:

```
hf download bartowski/DeepSeek-V4-Flash-0731-GGUF --include "DeepSeek-V4-Flash-0731-MXFP4/*" --local-dir ./
```

You can either specify a new local-dir (DeepSeek-V4-Flash-0731-MXFP4) or download them all in place (./)

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/DeepSeek-V4-Flash-0731:MXFP4
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b10173 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## MTP (DSpark)

This model has MTP (Multi-Token Prediction) weights in the form of a DSpark draft model. Unlike other MTP implementations, these are not included in the quants themselves - they are provided as a separate file in this repo: [dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf](https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf)

DSpark acts as a draft model, letting llama.cpp run speculative decoding for faster generation. To use it, add the following flag to your llama.cpp command:

```
--spec-type draft-dspark
```

When running with `-hf` as shown above, llama.cpp downloads the DSpark file automatically alongside the model. If you're downloading files manually instead, also grab the dspark file and pass it with `-md /path/to/dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf`.

Note: the DSpark file keeps the original model's MXFP4 expert weights unchanged (remaining tensors at Q8_0 or higher).

## Credits

Thank you kalomaze and Dampf for assistance in creating the imatrix calibration dataset.

Thank you ZeroWw for the inspiration to experiment with embed/output.

Want to support my work? Visit my ko-fi page here: https://ko-fi.com/bartowski