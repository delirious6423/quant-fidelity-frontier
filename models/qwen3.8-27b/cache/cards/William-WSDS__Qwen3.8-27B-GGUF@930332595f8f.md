---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- embraos
---

# Qwen3.8-27B, GGUF for embraOS

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to GGUF and quantised to Q4_K_M with llama.cpp 1.0.0. Nothing else was done to the model: no fine-tuning, no importance matrix, no change to its chat template.

These are the two files [embraOS](https://github.com/Ward-Software-Defined-Systems/embraOS) downloads when this model is installed as a native model (experimental): a model that runs on the machine's own processor, served by the llama.cpp that is built into the OS image. The OS pins each file by the sha256 below. They are ordinary GGUF files, and llama.cpp 1.0.0 runs them.

| File | What it is | Bytes | sha256 |
|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | The model, Q4_K_M | 17,005,351,136 | `4fc2219c0f0335e04b92ea446ce9f9cb6584370ac35cc751e340829bdebffa80` |
| `mmproj-Qwen3.8-27B-BF16.gguf` | What the model reads images through (its projector), BF16 | 931,145,952 | `e43a597863a21bfa48b0fbd4553a771ae4117e25bb172e66f1dbc3fc6d037131` |

## Where they come from

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, licensed Apache-2.0. Every weight file of that revision was downloaded whole and its sha256 checked against the hub's before the conversion.

## How they were made

With llama.cpp at its tag `v1.0.0`, from the directory that holds the revision above:

```
python3 convert_hf_to_gguf.py Qwen3.8-27B --outtype bf16 --outfile Qwen3.8-27B-BF16.gguf
python3 convert_hf_to_gguf.py Qwen3.8-27B --mmproj --outtype bf16 --outfile mmproj-Qwen3.8-27B-BF16.gguf
llama-quantize Qwen3.8-27B-BF16.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

## What is in them

- `Qwen3.8-27B-Q4_K_M.gguf`: 770 tensors, 343 Q4_K (11,289 MiB), 67 Q6_K (4,908 MiB), 360 F32 (10 MiB). A context window of 262,144 tokens, as the model was trained for.
- The converter of this version also writes the model's block for predicting several tokens at once (`qwen35.nextn_predict_layers = 1`, 251 MiB of the file). llama.cpp 1.0.0 does not use it, and says so when it loads the model ("model has unused tensor").
- `mmproj-Qwen3.8-27B-BF16.gguf`: 334 tensors, 110 BF16 (870 MiB), 224 F32 (18 MiB), projector type `qwen3vl_merger`. llama.cpp 1.0.0 reads images through it, and no video.

## Use with llama.cpp

```
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-BF16.gguf --image-min-tokens 1024 --image-max-tokens 1024
```

The server itself asks for at least 1,024 tokens an image for these models; embraOS holds an image to exactly that. Sampling and the thinking switch are as [Qwen's model card](https://huggingface.co/Qwen/Qwen3.8-27B) gives them.

## Licence

Apache-2.0, as the original. `LICENSE` is Qwen's file, unchanged. What was changed is the format of the weights: they were converted to GGUF, and the model's were quantised to Q4_K_M.
