---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: text-generation
library_name: ds4
tags:
  - gguf
  - deepseek
  - moe
  - sidecar
  - vision
  - speculative-decoding
  - lecore
---

# DeepSeek-V4.1-Flash-1017 — sidecars (vision tower, DSpark drafter, llama-layout skeleton)

![cover](cover.jpg)

The three small companion files the **ds4 / ds4-server** runtime loads next to a lecore-quantized
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) build.
**None of these is a standalone model** — the weights themselves live in the 99–107 GB VQ GGUFs
listed at the bottom, and only the ds4 runtime can load those.

| file | size | what it is | how it is loaded |
|---|---:|---|---|
| `DeepSeek-V4.1-Flash.vision.gguf` | 0.90 GiB | Vision tower sidecar: 32 ViT blocks, width 1024, patch 14, downsample 3, ≤1024 image tokens per image. Architecture `deepseek4`, `sidecar=vision`, 306 tensors. Emitted from the unquantized model; shared by every V4.1-Flash build, abliterated or not. | `ds4-server --vision-gguf <file>` (already the default in `run_ds41.ps1`) |
| `DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal.dspark.llama.gguf` | 5.05 MiB | DSpark speculative drafter: 3 draft towers × 128 experts, top-3, block 5, target layers 37/38/39, confidence head. Weights are the pqhaz-abliterated draft towers, VQ 8×1024 codewords. | picked up from the model directory next to the main GGUF (drafter armed automatically) |
| `DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal.llama.gguf` | 605.8 MiB | Dense-skeleton emit in llama.cpp GGUF layout: all non-expert tensors (attention q_a/q_b/kv/output_a/output_b, shared-expert FFN, indexer, MTP main_proj, token embedding, output head) with the full `deepseek41.*` metadata, 922 tensors. **The routed experts are not in it** — they stay in the 103 GB VQ file. | metadata/type-pass sidecar; not a loadable model on its own |

## Use

```powershell
git clone https://huggingface.co/staccs/ds4-winhip      # patched ds4 runtime + build scripts
cd ds4-winhip
.\hip\win\run_ds41.ps1 -Gguf D:\models\ds41\DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal.gguf `
                       -Engram D:\models\ds41\deepseek-engram
```

`run_ds41.ps1` points `--vision-gguf` at `DeepSeek-V4.1-Flash.vision.gguf` inside the same model
directory, so drop this file next to the main GGUF (or pass `-Extra "--vision-gguf,<path>"`).

Measured DSpark acceptance on the pqhaz build (Radeon 8060S, 128 GB unified): 1.0–3.3 accepted
tokens per 5-token verification block depending on prompt mix; with any repetition penalty set the
drafter is disabled by the runtime and you get plain decode. Vision encoder loads additively — the
image tower is ~1 GiB of weights and does not touch the text path.

## Engram n-gram tables

The main GGUFs do **not** contain DeepSeek's Engram tables. They are the last two shards of the
upstream repo, used unchanged (~203 GB, read from disk per token):

- [model-00047-of-00048.safetensors](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model-00047-of-00048.safetensors) — 101.5 GB
- [model-00048-of-00048.safetensors](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model-00048-of-00048.safetensors) — 101.5 GB

Drop both in one directory and pass `--engram-dir <dir>` (or `-Engram <dir>`); the server maps
them, it does not copy them into GPU memory.

## Main weights

| build | size | PPL (WikiText-2, 2k) | top-1 vs source |
|---|---:|---:|---:|
| [pqhaz abliterated, vq11cal](https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal-GGUF) | 106.8 GB | 2.78 | 80.4% |
| [dealignai abliterated, vq11cal](https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-abliterated-dealignai-lecore-vq11cal-GGUF) | 106.8 GB | 2.82 | 80.2% |
| [mrfakename abliterated, vq11cal](https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-abliterated-mrfakename-lecore-vq11cal-GGUF) | 106.8 GB | 2.80 | 80.3% |
| [rumplestilzken abliterated, vq11cal](https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-abliterated-rumplestilzken-lecore-vq11cal-GGUF) | 106.8 GB | 2.82 | 80.5% |
| [base model, vq10sh12](https://huggingface.co/staccs/DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-GGUF) | 99.1 GB | 3.615 | 74.5% |

## License

MIT, following the base model. The vision tower and Engram tables are DeepSeek's own weights,
redistributed unchanged.
