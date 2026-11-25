---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- gguf
- jucee
- apple-silicon
- metal
- vision
---

# Qwen3.8 Flash Next for Jucee

Mirror of the Qwen3.8 Flash Next GGUF files for reliable downloads in [Jucee](https://github.com/bitnautlabs/jucee).

| File | On disk | Resident weights | Runs on | Source |
|---|---|---|---|---|
| `Qwen3.8-Flash-Next-Q4.gguf` | 165 GiB | 69.7 GiB *(95.4 GiB n-grams on disk)* | one 96/128 GB Mac or larger | Mirror of `antirez/qwen3.8-flash-next-gguf` |
| `Qwen3.8-Flash-Next-Q2.gguf` | 137 GiB | 41.7 GiB *(95.4 GiB n-grams on disk)* | one 64 GB Mac or larger | Mirror of `antirez/qwen3.8-flash-next-gguf` |
| `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` | 0.6 GiB | 0.6 GiB | Vision sidecar | Mirror of `ggml-org/Qwen3.8-Flash-Next-GGUF` |

## License & Attribution

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, licensed under the **Qwen Community License 1.0** — see [`LICENSE`](./LICENSE).
Original GGUF conversions by [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf) and [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF).
Jucee is Apache 2.0. Jucee is a trademark of Bitnaut, LLC.
