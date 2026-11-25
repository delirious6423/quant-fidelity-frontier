---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
- gguf
- jucee
- apple-silicon
- metal
- vision
---

# DeepSeek V4.1 Flash for Jucee

Mirror of the DeepSeek V4.1 Flash GGUF files from [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf), hosted under `jucee` for reliable downloads in [Jucee](https://github.com/bitnautlabs/jucee).

| File | On disk | Resident weights | Runs on |
|---|---|---|---|
| `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` | 341 GiB | 152 GiB *(Engram on disk)* | one 192/256 GB Mac, SSD streaming on 128 GB, or two Macs |
| `DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf` … `-00009-of-00009.gguf` | 467 GiB joined | 278 GiB *(Engram on disk)* | two Macs together (or 512 GB Mac) |
| `DeepSeek-V4.1-Flash-Vision.gguf` | 0.9 GiB | 0.9 GiB | Vision sidecar |
| `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | 7.4 GiB | 7.4 GiB | DSpark speculative drafter (mirror of `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`) |

The 9 MXFP4 shards are joined automatically into `DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf` by Jucee after download.

## License & Attribution

Base model: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) by DeepSeek, licensed under the **MIT License** (see [`LICENSE`](./LICENSE)).
Original GGUF conversions by [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) (weights and vision) and [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (DSpark Q8_0 drafter).
Jucee is Apache 2.0. Jucee is a trademark of Bitnaut, LLC.
