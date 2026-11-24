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
| `mmproj-Qwen3.8-27B-F16.gguf` | What the model reads images through (its projector), F16 | 927,607,008 | `cb0a71fe9b84af2552fc6260acf8e9773a29d115c683aba01c72a7ac537b212b` |

## Where they come from

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, licensed Apache-2.0. Every weight file of that revision was downloaded whole and its sha256 checked against the hub's before the conversion.

## How they were made

With llama.cpp at its tag `v1.0.0`, from the directory that holds the revision above:

```
python3 convert_hf_to_gguf.py Qwen3.8-27B --outtype bf16 --outfile Qwen3.8-27B-BF16.gguf
llama-quantize Qwen3.8-27B-BF16.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
python3 convert_hf_to_gguf.py Qwen3.8-27B --mmproj --outtype f16 --outfile mmproj-Qwen3.8-27B-F16.gguf
```

## What is in them

- `Qwen3.8-27B-Q4_K_M.gguf`: 770 tensors, 343 Q4_K (11,289 MiB), 67 Q6_K (4,908 MiB), 360 F32 (10 MiB). A context window of 262,144 tokens, as the model was trained for.
- The converter of this version also writes the model's block for predicting several tokens at once (`qwen35.nextn_predict_layers = 1`, 251 MiB of the file). llama.cpp 1.0.0 does not use it, and says so when it loads the model ("model has unused tensor").
- `mmproj-Qwen3.8-27B-F16.gguf`: 334 tensors, 112 F16 (873 MiB), 222 F32 (12 MiB), projector type `qwen3vl_merger`. llama.cpp 1.0.0 reads images through it, and no video.
- The projector is F16, and not BF16 as Qwen stores the weights. llama.cpp 1.0.0 has vector code for F16 on Arm and on x86 alike, and for BF16 on x86 alone: from a BF16 file the encoding of one image took 11 to 14 minutes on four cores of an Apple Silicon machine. F16 cannot hold the smallest values of BF16: 166,773 of the projector's 460,730,096 values (0.04 %) differ between the two formats, by 3e-8 at most. The BF16 file this repository carried first is in its history, at revision `930332595f8f72f27ce2bd9ae5dfc7aea4e877da`.

## Use with llama.cpp

```
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-F16.gguf --image-min-tokens 1024 --image-max-tokens 1024
```

The server itself asks for at least 1,024 tokens an image for these models; embraOS holds an image to exactly that. Sampling and the thinking switch are as [Qwen's model card](https://huggingface.co/Qwen/Qwen3.8-27B) gives them.

## Licence

Apache-2.0, as the original. `LICENSE` is Qwen's file, unchanged. What was changed is the format of the weights: they were converted to GGUF, the model's were quantised to Q4_K_M, and the projector's are stored as F16.
