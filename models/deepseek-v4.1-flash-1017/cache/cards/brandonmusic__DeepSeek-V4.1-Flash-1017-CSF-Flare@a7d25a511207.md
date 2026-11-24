---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
license: mit
pipeline_tag: image-text-to-text
tags:
- flare
- hyperflare
- mxfp4
- csf
- nvfp4
- fp8
- moe
- tensor-parallel
- gb202
- conversational
- vision
---

# DeepSeek V4.1 Flash 1017 CSF for Flare

Built on the work of DeepSeek AI, NVIDIA, Local Inference Lab, Luke Alonso, Martin Vit (`voipmonitor`), and the Hyperflare, FlashInfer, and b12x contributors. See [Credits](#credits) and [`CREDITS.md`](CREDITS.md).

This repository publishes a prepared **Hyperflare (Flare)** runtime package for **DeepSeek V4.1 Flash 1017**, built for four-GPU tensor-parallel serving on NVIDIA RTX PRO 6000 Blackwell (SM120 / `gb202`) GPUs. Flare is a native Rust inference runtime (`NVIDIA/hyperflare-inference`). Each hardware revision contains prepared weights, compiled CUDA kernels, captured execution graphs, and signed package metadata. Loading runs directly on the GPU with no Python and no runtime compilation. Transformers, vLLM, SGLang, and llama.cpp cannot load this format.

The model handles text and images, supports up to **32 concurrent requests** and **1,048,576 tokens** per request, and serves an OpenAI-compatible API under the model name `DeepSeek-V4.1-Flash`.

## Results

Measured on 2026-11-24 on four RTX PRO 6000 Blackwell GPUs (96 GB each; two Max-Q cards at 300 W and two Workstation Edition cards at 350 W) with Engram tables in host RAM, seven MTP draft tokens per round with adaptive verification (`--mtp-draft-tokens 7 --mtp-adaptive-verification --mtp-max-verification-token-budget 128 --mtp-adaptive-verification-alpha 0.6`), and default sampling (`temperature=1.0`, `top_p=1.0`).

| Measurement | Result |
| --- | --- |
| Cold prefill, 8K-token prompt (C1) | 23,498 tok/s (0.35 s TTFT) |
| Cold prefill, 64K-token prompt (C1) | 22,038 tok/s (2.92 s TTFT) |
| Cold prefill, 128K-token prompt (C1) | 20,789 tok/s (6.18 s TTFT) |
| Sustained decode, 1 request | 375.2 tok/s |
| Sustained decode, 2 requests (aggregate) | 556.7 tok/s (278.4 tok/s/user) |
| Sustained decode, 4 requests (aggregate) | 858.7 tok/s (214.7 tok/s/user) |
| Sustained decode, 8 requests (aggregate) | 1,114.9 tok/s (139.4 tok/s/user) |
| Sustained decode, 16 requests (aggregate) | 1,518.4 tok/s (94.9 tok/s/user) |
| Sustained decode, 32 requests (aggregate) | 2,138.4 tok/s (66.8 tok/s/user) |
| Sustained decode draft acceptance rate | 52.5 % (`0.525`) |
| Single-stream coding task (HTML Tetris, `thinking: false`, 5 runs) | 788.4 tok/s median / 821.4 tok/s best (122.0 steps/s, 888.6 tok/s 1 s peak) |
| Single-stream coding task (HTML Tetris, `thinking: true`, 3 runs) | 569.2 tok/s median / 589.4 tok/s best (119.6 steps/s, 828.4 tok/s 1 s peak) |
| Shared GPU KV cache pool (4× 96 GB dedicated GPUs) | 72.9 GiB / 21,102,592 tokens (21.10M tokens; 1,997 snapshot slots) |
| Startup time from NVMe (Engram in host RAM) | 112–113 s (53.5 s weight load, 41.7 s Engram RAM load) |

The [`gb202` platform card](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-CSF-Flare/blob/gb202/README.md) covers installation, launch commands, prepared settings, and serving flags.

## The weights and preparation

The package is prepared from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), a mixture-of-experts text and image model released under the **MIT license**. The architecture has 40 layers, hidden width 5,120, 64 attention heads, 384 routed experts plus one shared expert, top-6 routing, 129,280 vocabulary entries, a 1,048,576-token context window, three MTP draft layers, a vision tower, and 188.8 GiB of Engram embedding tables.

Key weight and runtime preparation properties in this `gb202` package:

- **Lossless CSF expert block scales:** Routed MXFP4 expert weights retain their exact 32×32 E8M0 block scales stored with Local Inference Lab's lossless CSF encoding (`row-base-offset1-u24-exceptions/1`), compressing 46,080 scale tensors from 16,986,931,200 bytes to 2,316,453,540 bytes with bit-exact SHA-256 verification against every original source shard. Keeping CSF scales resident leaves **72.9 GiB (21,102,592 tokens)** of GPU KV cache across four 96 GB cards.
- **Attention and KV cache:** Multi-head latent attention runs in BF16 compute (`--mla-compute-mode bf16`) with FP8 compressed KV cache storage.
- **Speculative decoding and sampling heads:** Prepared in MTP-only mode (`--mtp-only --mtp-draft-tokens 7 --speculative-token-capacity 128 --speculative-graph-budget 64`) with vocabulary-parallel sampling (`--vocabulary-parallel-sampling`), NVFP4 draft head (`--draft-head-quantization nvfp4`), and FP8 Markov head (`--markov-head-quantization fp8`).
- **SM120 PCIe TP4 `RO=1` collectives and kernels:** Prepared with `B12X_PCIE_X8_RANK=2` asymmetric `rot_rsag64` (`0/33/33/33` split) runtime-tuned collectives, split-lagged mHC overlapped with PCIe AllReduce, prequantized compact W4A8 routed experts, fused Q-RoPE TC-GEMV, and 8,192-token prefill chunks.

Each hardware revision is a self-contained, signed TP4 package consisting of `model.safetensors` (signed metadata primary), content-addressed companion files under `weights/` and `deployments/`, and `publisher.json`. Keep all files together under their original names.

## Pick a platform

| Revision | Hardware | Package size | Platform setup |
| --- | --- | --- | --- |
| `gb202` | Linux x86-64; 4× RTX PRO 6000 Blackwell 96 GB (SM120) on one host | 508.8 GB (473.9 GiB, 197 files) | [gb202 setup](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-CSF-Flare/blob/gb202/README.md) |

The `main` revision contains this model card, `CREDITS.md`, `LICENSE`, `NOTICE`, and `publisher.json`. Download the `gb202` revision to run the model.

## Flare fundamentals

### Install Flare

This package uses the standard DeepSeek V4.1 Flash Flare runtime format and requires no custom Flare executable. Use a released Flare `v0.2.3` or newer build:

- **Container:** `ghcr.io/local-inference-lab/flare:release` (`v0.2.3` or newer) with the NVIDIA Container Toolkit.
- **Native host binary:** `flare` (`v0.2.3` or newer) installed via the signed `flare-inference` APT package or built from [`NVIDIA/hyperflare-inference`](https://github.com/NVIDIA/hyperflare-inference).

### Download and verify

Download with `flare download` or the Hugging Face `hf` CLI, naming `--revision gb202`. Before loading, Flare verifies the package signature against `publisher.json` and prompts for publisher and model-license acceptance. The signature covers `model.safetensors` and every companion file under `weights/` and `deployments/`.

### Serve

Rank 0 exposes the OpenAI-compatible API at `/v1` and health at `/health` under model name `DeepSeek-V4.1-Flash`.

- **Concurrency and context:** 32 concurrent requests, 1,048,576-token per-request limit, 8,192-token prefill chunks, and 7-token MTP speculative decoding with adaptive verification.
- **Multimodal input:** Text and images through OpenAI `image_url` items (PNG, JPEG, or static WebP), up to 4 concurrent image-bearing requests and 4,096 visual tokens per request.
- **Sampling defaults:** Temperature 1.0, top-p 1.0, top-k disabled, min-p 0, repetition penalty 1.0 (request range `1.0–1.1`), zero presence/frequency penalties, and request-controlled seed.
- **Engram tables:** Stored on local NVMe by default (`--embedding-table-storage disk`) or kept in host RAM (`--embedding-table-storage ram`) on hosts with at least 320 GiB of system memory.

## Validation

All checks below passed on 2026-11-24 on four RTX PRO 6000 Blackwell GPUs with zero server or CUDA errors:

- **Core liveness and multimodal checks (`13/13`):** health, served model names, arithmetic (`17 × 23 = 391`), prime generation, classification, function tool call, JSON mode, vision image reading (`4871`), multi-turn prefix retention, sampling penalties, SSE streaming, long-context needle (`HALVES-CHECK` 6/6 exact answers and first-token match), and repeated/unique prefix-cache stress (28/28).
- **Multi-turn and tool-calling suite (`agentic_suite.py`, 3 trials per case at default sampling):** **3/3** pass on all 7 cases (chat two turns, Responses API two turns, chat tool round trip, Responses API tool round trip, Responses API tool error result, Responses API free-text custom tool, chat parallel tool calls), with tool calls made in **3/3** trials on all 5 tool cases.
- **DeepSeek chat and Responses edge checks (`5/5`):** assistant-ended conversation, empty user message, thinking disabled, system prompt, and Responses API assistant-ended turn.
- **Server and CUDA error audit:** 0 panics, aborts, segfaults, or CUDA runtime errors.

**Known evaluation status:** BFCL v4 and comprehensive accuracy benchmarks have not yet been run on this core.

## Credits

- **Base model:** DeepSeek V4.1 Flash 1017 by DeepSeek AI ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, MIT License). Weights, vision tower, MTP draft layers, Engram tables, tokenizer, and chat template are theirs.
- **Serving runtime:** Hyperflare (Flare), [`NVIDIA/hyperflare-inference`](https://github.com/NVIDIA/hyperflare-inference) (Apache-2.0), developed by the NVIDIA Hyperflare team and community contributors, including Luke Alonso, Martin Vit, Jason Cook, and collaborators.
- **Recipes, CSF encoding, and release layout:** Local Inference Lab ([`local-inference-lab/hyperflare-recipes`](https://github.com/local-inference-lab/hyperflare-recipes), Apache-2.0; lossless CSF MXFP4 block-scale encoding; `llm-inference-bench v0.4.29`; and the multi-branch Flare Hugging Face layout from [`local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE`](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE)).
- **SM120 kernels and PCIe TP4 optimizations:** b12x and FlashInfer by Luke Alonso and Local Inference Lab ([`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer), Apache-2.0), with Martin Vit (`voipmonitor`) for split-lagged mHC overlapped with PCIe AllReduce, prequantized compact W4A8 MoE, fused RoPE registration, BF16 TC-GEMV `split_k=5,10`, and SM120 PCIe TP4 `RO=1` collective improvements (`local-inference-lab/flashinfer#136`, `NVIDIA/hyperflare-inference#491`, `local-inference-lab/hyperflare-recipes#19`).
- **Package preparation and publication:** Brandon M. Music.

## License

The model weights and this prepared package are distributed under the **MIT License** (`LICENSE`), matching [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). See [`NOTICE`](NOTICE) for component attributions and licenses.
