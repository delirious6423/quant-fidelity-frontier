---
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
library_name: colibri
tags:
  - colibri
  - qwen3.8
  - qwen3.8-flash-next
  - moe
  - i4
  - fp8
---

# Qwen3.8-Flash-Next (125B, 6B active) for Colibri, int4 g64

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) converted for the [Colibri](https://github.com/JustVugg/colibri) runtime (`c/qwen38.c`). The routed experts are quantized to int4 (group size 64). Everything else is kept as in the source snapshot.

> **Status: smoke-tested only.** The model loads and gives sensible answers in short chats on the setup described below. Output quality has **not** been compared against the source model, and no benchmark has been run.
>
> **Note:** Fresh upload. External download validation has not yet been completed. If you encounter missing files, corrupted shards, or download issues, please open a discussion in the Community tab.

## What is in this repository

| Path | Content |
|---|---|
| `model-*.safetensors` (131 shards) | Source weights in BF16 and FP8 (E4M3): dense layers, vision tower, MTP head. |
| `experts-int4g64/` | Routed experts of the 48 text layers, int4, group size 64. 2.64 MiB per expert (the FP8 original is 4.69 MiB). **Required by Colibri.** |
| `config.json`, tokenizer files, `chat_template.jinja`, `generation_config.json` | As in the source model. |
| `.coli_usage`, `.coli_analysis.json` | Expert-usage history Colibri uses to warm-start its GPU expert tier. Produced from the converter's own prompts. |
| `LICENSE` | License file. See also the license of the source model. |

Total size: about 254 GB. Keep the directory layout as is: Colibri looks for `experts-int4g64/` next to the shards. Hugging Face lists only BF16, F8_E4M3 and I64 tensors because the int4 data lives in that folder, not in the safetensors files.

## Model

- 125B parameters, 6B active per token (the source model also counts 51B of n-gram embedding and a 4B MTP head).
- 48 layers: Gated DeltaNet plus Qwen Sparse Attention (QSA), one QSA layer every four (12 in total). Mixture of experts, gated residual streams, MTP head, vision tower.
- Native context: 262,144 tokens (`max_position_embeddings`).
- Tensor names use the prefix `model.language_model`, for example `model.language_model.layers.{i}.linear_attn.in_proj_qkv` and `model.language_model.layers.{i}.self_attn.indexer.index_qk_proj`. The MTP head is under `mtp.*` and the vision tower under `model.visual.*`.

## Running it

```
python coli web --model <path-to-this-repo> --cap 512 --ctx 20000
```

This starts an OpenAI-compatible API on `http://127.0.0.1:8000/v1`. `--cap` is the number of experts cached in RAM per layer. A full cache at 512 per layer needs about 63 GiB (2.64 MiB x 512 x 48). With less RAM, lower `--cap`: the remaining experts are read from disk, which is slower. Colibri uses the MTP head for speculative decoding.

## Tested on

One machine: Windows laptop, Intel i9, 64 GB RAM, NVIDIA RTX 5060 Laptop (8 GB VRAM), Colibri release build with Vulkan. <!-- TODO: add the exact Colibri release tag or commit -->

- RSS after load: about 11.8 GB.
- With `--cap 512` and 49 GB of RAM available, the expert cache held 369 experts per layer.

## Known limitations

- **Quality and speed are not measured.** No perplexity or benchmark against the source model, and no published tokens-per-second.
- **High `--ctx` on 8 GB VRAM (release build).** At `--ctx 60000` Colibri kept only part of the K/V cache on the GPU and read the prompt one row at a time (a 3.7k-token prompt took 775 s). Setting `COLI_VK_KV_DEVICE_ROWS=20000` restored multi-row prompt chunks in our test. We have not yet measured the end-to-end time or the cost of attention once the context is really full. See Colibri's `ENVIRONMENT.md`.
- Other hardware, and real contexts above 20k tokens, are untested.

## Links

- Engine: