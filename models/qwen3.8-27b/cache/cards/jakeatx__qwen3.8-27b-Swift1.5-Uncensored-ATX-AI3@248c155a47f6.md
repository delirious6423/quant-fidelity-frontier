---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3_5
- mtp
- speculative-decoding
- llamampere
- ampere
---

# Swift 1.5 Qwen3.8-27B Uncensored, ATX AI3 (3.0 bits per weight on card, with MTP, 12 GB cards)

Qwen3.8-27B Swift 1.5, compressed by llamAmpere to fit a 12 GB GPU with long context and MTP speculative decoding on.

Files in this repo: `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf` plus lsa-qwen38-27b-d2048.gguf. Keep them all in one folder.

## Small cards: 8 / 10 / 12 GB

| Card | File | Context | MTP | Peak VRAM (measured) |
| :--- | :--- | ---: | :---: | ---: |
| 12 GB | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf` | 262,144 | On | 11,544 MiB |
| 10 GB | [`qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2`](https://huggingface.co/jakeatx/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2) (2.3 bpw) | 212,992 | On | — |
| 8 GB | `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2.1.gguf` | 204,800 | On | 7,506 MiB |

**Quality: drift from full precision (KLD, lower is better).** All rows use the same 16,200 positions (coding, agentic and RAG at ~12K ctx), scored against Swift 1.5 BF16. The Unsloth rows are Unsloth's exact UD per-tensor recipes and imatrix applied to Swift 1.5, so every row shares one teacher.

| File | Size | KLD | Top-1 |
| :--- | ---: | ---: | ---: |
| **ATX 12 GB** | 11.24 GB | **0.0270** | **95.98%** |
| Unsloth UD-IQ3_XXS recipe | 10.93 GB | 0.0396 | 95.33% |
| Unsloth UD-Q2_K_XL recipe | 9.95 GB | 0.0591 | 94.00% |
| **ATX AI2 2.3 bpw (10 GB)** | 9.14 GB | **0.0593** | 94.06% |
| Mirai S | — | 0.0745 | — |

The 12 GB file has 54% less drift than UD-Q2_K_XL. The 2.3 bpw file matches UD-Q2_K_XL in a file 19% smaller.

**Shorter answers.** On LiveCodeBench v6 (the same 100 problems), the 12 GB file used **33% fewer tokens** than UD-Q2_K_XL: median 4,608 vs 8,932 per answer, 95% range 22–44% fewer. It was shorter on 74 of 100 problems.

**Coding score, one seed (directional, not statistically significant).** LiveCodeBench v6, strict grader: ATX 12 GB 89%, UD-Q2_K_XL 87% (run on llamAmpere), Qwen BF16 published 90.3%.

**Fit.** With standard attention, UD-Q2_K_XL cannot run full context with MTP on a 12 GB card (estimate). The ATX 12 GB file runs the full 262K context with MTP.

**Running the small-card files** requires **llamAmpere 0.7 (coming, not yet released)**: LSA attention + host-offloaded KV. The `--preset` commands below will work once 0.7 ships. Download the sidecar files into the same folder as the model:
```bash
# 12 GB
./llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf --preset 12dense
# 8 GB
./llama-server -m qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2.1.gguf --preset 8dense
```
Sidecars: `lsa-qwen38-27b-d2048.gguf` (LSA index map, both sizes). For 8 GB only: `mtp-dm.gguf`, `head-ait2.gguf` (draft overlays) and `head-exact-f16.gguf` (exact rescoring head).

SHA-256:
- `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI3.gguf` d28a3c00f53ce95566df6478b9733d570e8143d5cca669ff057465f1e26812f5
- `qwen3.8-27b-Swift1.5-Uncensored-ATX-AI2.1.gguf` cfb2ece5c79ba94cb3880f5edaab023c655e99c1797c4438e9e97c7b8e74aad7
- `lsa-qwen38-27b-d2048.gguf` 5489a4f2f6e9169c57db7344afdebdca243c3d438d8e70197c52a6c95244aa68
- `mtp-dm.gguf` 4a8e9a5cfb6b1395088bd88b26400b504e47a10fad5a109e4cbee3276f9d5cae
- `head-ait2.gguf` 673687fe741bf65c34ebdc73c53248de60f82184cd3a0138b320acfe3d811e5d
- `head-exact-f16.gguf` f812690e5925605edafb2f364e1b40ef89756856c7b6e16ba1e4515bc80d1227

