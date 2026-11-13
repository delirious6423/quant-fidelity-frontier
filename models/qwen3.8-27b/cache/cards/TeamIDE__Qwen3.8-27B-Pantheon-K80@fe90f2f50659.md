---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags: [pantheon, pantheon-local, k80, gguf]
---

# Qwen3.8-27B — Pantheon K80 build (Q8_0)

The file Pantheon Local installs for **NVIDIA Tesla K80** boxes (two boards, 4 GPUs).

- `Qwen3.8-27B-Q8_0.gguf` — **unmodified** Q8_0 GGUF by [unsloth](https://huggingface.co/unsloth)
  ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)) of
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team. 29,047,086,048 bytes,
  sha256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`. Text only (no vision).

**License:** Apache License 2.0 (see `LICENSE`), the license of the original model:
https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE. Model © the Qwen team, Alibaba Cloud.
Quantization by unsloth. Redistributed unchanged.

Measured on Pantheon, 4 K80 GPUs on two boards, one user, MTP off (greedy): decode 4.1 tok/s on
short prompts. Checked against the llama.cpp CPU reference (Q8_0).
Pantheon: https://git.teamide.dev/David/Pantheon
