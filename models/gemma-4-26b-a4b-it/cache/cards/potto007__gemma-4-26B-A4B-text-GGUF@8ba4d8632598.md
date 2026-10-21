---
quantized_by: potto007
pipeline_tag: image-text-to-text
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
---

## Llamacpp Quantizations of gemma-4-26B-A4B-it by google

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b9352">b9352</a> for quantization.

Vision tower (VLM) stripped for lighter local operation. Stripped using forked version of REAP (https://github.com/potto007/reap/tree/gemma4-reap-support).

Original model: https://huggingface.co/google/gemma-4-26B-A4B-it

Run them in your choice of tools:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [ramalama](https://github.com/containers/ramalama)
- [LM Studio](https://lmstudio.ai/)
- [koboldcpp](https://github.com/LostRuins/koboldcpp)
- [Jan AI](https://www.jan.ai/)
- [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui)
- [LoLLMs](https://github.com/ParisNeo/lollms)

## Prompt format

```
<bos><|turn>system
{system_prompt}<turn|>
<|turn>user
{prompt}<turn|>
<|turn>model
<|channel>thought
<channel|>
```

## Download a file (not the whole branch) from below:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [gemma-4-26B-A4B-text-f16.gguf](https://huggingface.co/potto007/gemma-4-26B-A4B-text-GGUF/blob/main/gemma-4-26B-A4B-text-f16.gguf) | bf16 | 50.5GB | true | Full BF16 weights. |
| [gemma-4-26B-A4B-text-Q8_0.gguf](https://huggingface.co/potto007/gemma-4-26B-A4B-text-GGUF/blob/main/gemma-4-26B-A4B-text-Q8_0.gguf) | Q8_0 | 26.9GB | false | Extremely high quality, generally unneeded but max available quant. |
| [gemma-4-26B-A4B-text-Q6_K.gguf](https://huggingface.co/potto007/gemma-4-26B-A4B-text-GGUF/blob/main/gemma-4-26B-A4B-text-Q6_K.gguf) | Q6_K | 22.8GB | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, *recommended*. |

## Downloading using huggingface-cli

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have hugginface-cli installed:

```
pip install -U "huggingface_hub[cli]"
```

Then, you can target the specific file you want:

```
huggingface-cli download potto007/gemma-4-26B-A4B-text-GGUF --include "gemma-4-26B-A4B-text-Q8_0.gguf" --local-dir ./
```

You can either specify a new local-dir (gemma-4-26B-A4B-it-Q8_0) or download them all in place (./)

</details>
