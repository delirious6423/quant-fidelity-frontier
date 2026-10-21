---
license: other
license_name: qwen-research-terms-derivative
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags:
  - l0xre
  - gguf
  - speculative-decoding
  - 27b
  - 12gb-vram
language:
  - en
pipeline_tag: text-generation
---

# L0xRE-27b-Low

**A 27-billion-parameter model built for a small GPU-memory footprint — 8.62 GB on disk, running on any RTX 30/40/50 card with at least 12 GB of VRAM.** Paired with the [L0xRE BeeLLama Low runtime](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low), it serves as a fast, smart, small agentic coding assistant on your own machine.

“27b” is the parameter count, not the download size. “Low” describes the memory-focused model build. Context length depends on the card, runtime profile, and other applications.

**Start here: [the runtime README](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low#readme) — one installer command per OS.** The official release with every package, checksum, and certification receipt is [beellama-v0.4.7-r6](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-r6).

## Download these two files

| File | What it does | Download size |
| --- | --- | ---: |
| [L0xRE-27b-Low.gguf](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/main/L0xRE-27b-Low.gguf) | The main model. This is the recommended model file. | 8.62 GB |
| [Qwen3.8-27B-DFlash2-Q4_K_M.gguf](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Recommended drafter: the runtime uses it to propose tokens that the main model verifies (DFlash2 speculative decoding). | 1.14 GB |

For the recommended setup, download both files: **9.76 GB in total**, plus the runtime package. The installer fetches and verifies both automatically. The main model can also run without a drafter, with speculative decoding disabled.

The smaller `Qwen3.8-27B-DFlash2-Q2_K.gguf` file is an optional 705 MB drafter for memory-constrained experiments. Start with the recommended Q4_K_M drafter to match the published certified configurations.

## Install and run

The guided installer detects your GPU, picks the right certified or starter package, verifies every SHA-256, downloads these files, and prints a ready-to-run start command:

```bash
# Linux / WSL — see the runtime README for the fetch command
bash install-l0xre.sh
```

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-l0xre.ps1
```

When ready, the server's health endpoint is `http://127.0.0.1:8080/health` and its API base is `http://127.0.0.1:8080/v1`. For a compatible chat client or coding agent, use that API base and select `L0xRE-27b-Low` from the server's `/v1/models` endpoint.

**This model requires the L0xRE BeeLLama Low runtime.** Direct use with stock llama.cpp, Ollama, LM Studio, Transformers, or other inference engines is outside this release's supported setup. The GGUF extension alone does not establish compatibility.

## GPU support and context

| GPU / OS | Certified configuration | Status |
| --- | --- | --- |
| RTX 30 series ≥12 GB / Linux | 80K context, KVarN4/4 KV, DFlash2 N7 — measured on RTX 3060 | **Certified** |
| RTX 40 series ≥12 GB / Linux | 80K context — certified on RTX 4090; starter profile on other cards | **Certified** (4090) / Starter |
| RTX 5090 32 GB / Linux or Windows | 80K context, CUDA 13.0 on Windows | **Certified** |
| Other RTX 50 series ≥12 GB | 32K starter profile | Starter |
| RTX 30/40 series / Windows | 80K profile; RTX 4070 Ti measured 68/116 t/s | Starter |

**Certified** routes passed correctness, capacity, soak, and consumer-probe gates on the exact card. **Starter** routes ship the best measured configuration so far, clearly labeled. See the [architecture status](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/PARITY.md) for details.

## Measured speed (frozen Bench.sh protocol)

| Hardware | Prose | Code | Cold prefill |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, Linux | 37 t/s | 65 t/s | 566 t/s (10K) |
| RTX 4090 24 GB, Linux | 104 t/s | 167 t/s | 2,034 t/s (10K) |
| RTX 5090 32 GB, Linux | 131 t/s | 229 t/s | 3,527 t/s |
| RTX 5090 32 GB, Windows | 137 t/s | 230 t/s | 3,611 t/s |
| RTX 4070 Ti 12 GB, Windows (starter) | 68 t/s | 116 t/s | — |

These are workload-specific certified measurements with published receipts, not guaranteed speeds for every prompt. Quality scores from other checkpoints in the model family are not presented as an evaluation of this exact file. No external leaderboard result is claimed here.

## File integrity

Download sizes use decimal GB.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8,619,127,680 | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| `Qwen3.8-27B-DFlash2-Q2_K.gguf` | 705,430,880 | `e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e` |

## Related checkpoint

[L0xRE-27b-Low-MTP](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate, larger checkpoint with an additional prediction block. Its file hash, runtime setup and qualification are distinct. Use the main model file on this page for the recommended drafter-based setup.

## Model origin and license

L0xRE-27b-Low is a quantized derivative of **Qwen3.8-27B**, with low-bit projections and higher-precision embeddings/output tensors. The model's configured native context capacity is 262,144 tokens; supported and tested serving contexts are determined by the runtime profiles above.

Model weights remain under the base model's research/community terms: [license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). The L0xRE runtime tooling ships under the BeeLlama license with third-party notices in its packages. Model licensing and runtime licensing are separate.
