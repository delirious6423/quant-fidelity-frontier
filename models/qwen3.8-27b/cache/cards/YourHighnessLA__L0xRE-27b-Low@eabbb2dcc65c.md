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

**27B parameters. 72 tok/s code. 128K context and vision on a 12 GB GPU.**

L0xRE-27b-Low is a memory-focused, quantized Qwen3.8-27B derivative for a local coding assistant or AI agent. With the [L0xRE BeeLLama Low runtime](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low), every model layer runs on the GPU, without CPU layer offload. The main model is **8.62 GB on disk**; “27b” is the parameter count, not the download size.

The **72 tok/s** headline is code decode on an **RTX 3060 12 GB, native Linux, headless**. Speeds depend on the card and workload; measurements below identify both.

**Current runtime: [L0xRE v0.1.3.2](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1.3.2), released November 5, 2026.** This updates the runtime and launchers, not the model weights in this repository. Start with the [runtime README](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low#readme) or the installation commands below.

## What changed in the runtime

- **v0.1.3:** faster decode on the RTX 3060 and much faster prompt reuse, including time to first token, follow-up turns and cached system prompts. The release gates measured about **21% faster code decode** and **45% lower follow-up time to first token** on the RTX 3060 versus v0.1.2.
- **v0.1.3.1:** 12 GB cards that also drive a display use a **1856 MiB desktop arena**, leaving GPU memory for a browser, messaging apps and an agent. Short decode is unchanged; the smaller arena trades some deep-context speed for headroom.
- **v0.1.3.2:** **`doctor`** explains slow performance, including GPU memory paging, competing GPU programs and driver-reported heat or hardware slowdowns. On Windows and WSL, a background watch warns when GPU memory pages into system RAM while L0xRE is answering, and reports when it recovers. It only warns; it does not change settings or restart the server. Use `--no-gpu-watch` to turn the watch off.

The v0.1.3.2 patch leaves runtime binaries, presets, arenas, model files and selftest references identical to v0.1.3.1. The [v0.1.3 speed results](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V013-RELEASE.md) still apply. See the [v0.1.3.2 patch notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V0132-RELEASE.md) for diagnostics and upgrade details.

## Install and run

You need an **NVIDIA RTX 30, 40 or 50 series GPU with at least 12 GB VRAM**, a recent NVIDIA driver and about **20 GB free disk space**. The download is about 11 GB with the runtime and vision support. The guided installer detects the GPU, selects the package and memory profile, and verifies the downloaded files against their SHA-256 pins.

**Windows 10 or 11**

Run these commands in a terminal:

```bat
curl.exe -LO https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.cmd
.\install.cmd -Vision
```

Then double-click **`start.cmd`** in the install folder printed by setup. The installer bundles Python and the runtime DLLs; no PowerShell execution-policy flag is needed.

**Linux or WSL2**

Ubuntu 24.04 or compatible, with Python 3.12+, `curl`, `zstd` and `numactl`:

```bash
curl -fL -o install-l0xre.sh https://raw.githubusercontent.com/seanyourhighness/L0xRE-BeeLLama-Low/main/install/install.sh
bash install-l0xre.sh --vision
~/L0xRE/start.sh
```

The vision option downloads the release-pinned **Q5_K-MIX projector** separately; it is not included in the main GGUF file. Omit `-Vision` / `--vision` for a text-only installation. See [installer options](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/install/README.md).

**Connect a client or agent**

- API base: `http://127.0.0.1:8080/v1`
- Model name: `L0xRE-27b-Low`
- Health check: `http://127.0.0.1:8080/health`

Use any client that accepts a custom OpenAI-compatible endpoint. You can also give your agent the [agent installation guide](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/AGENT-INSTALL.md).

**Update or diagnose**

Stop the server before updating. On Windows, run **`update.cmd`** in the install folder. On Linux or WSL, run **`bash install-l0xre.sh --update`** using the installer downloaded above. The installer supports rollback to the retained previous runtime.

If performance is slow, double-click **`doctor.cmd`** in the Windows runtime folder, or run **`./doctor`** there on Linux / WSL; the generated **`start.sh --doctor`** command also works. If Windows is paging GPU memory, close competing GPU applications to free VRAM. Full speed returns without restarting L0xRE once memory is available. A few GB of shared GPU memory in Task Manager can be normal pinned system memory, not evidence of paging by itself.

**This model requires the L0xRE BeeLLama Low runtime.** Direct use with stock llama.cpp, Ollama, LM Studio, Transformers or other inference engines is outside this release's supported setup. The GGUF extension alone does not establish compatibility.

## Model files

The installer downloads and verifies the recommended pair automatically:

| File | Purpose | Download size |
| --- | --- | ---: |
| [L0xRE-27b-Low.gguf](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/main/L0xRE-27b-Low.gguf) | Main model | 8.62 GB |
| [Qwen3.8-27B-DFlash2-Q4_K_M.gguf](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low/resolve/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Recommended DFlash2 drafter, which proposes tokens for the main model to verify | 1.14 GB |

Together these are **9.76 GB**, before the runtime and optional vision projector. Sizes use decimal GB. The main model can also run without a drafter, with speculative decoding disabled.

The included `Qwen3.8-27B-DFlash2-Q2_K.gguf` is an optional **705 MB** drafter for memory-constrained experiments, not the recommended release configuration.

## GPU support and context

| Runtime package | Memory profiles and context | Measurement scope |
| --- | --- | --- |
| Linux SM86, RTX 30 series | 12 GB / 128K | RTX 3060, native Linux |
| Linux SM89, RTX 40 series | 12 GB / 128K; 16 GB / 256K | RTX 4090, native Linux |
| Linux / WSL SM120, RTX 50 series | 12 GB / 128K; 16 GB / 256K | RTX 5090, WSL2; no separate bare-metal Linux speed claim |
| Native Windows SM120, RTX 50 series | 12 GB / 128K; 16 GB / 256K | RTX 5090, native Windows |
| Native Windows SM86 / SM89, RTX 30 / 40 series | SM86: 12 GB / 128K; SM89: also 16 GB / 256K | Release packages are build-verified, not formally hardware-qualified; separate RTX 4070 Ti desktop measurements appear below |

128K means **131,072 tokens**; 256K means **262,144 tokens**. SM86 has no 16 GB profile. The installer selects 16 GB where the package supports it and detected memory is at least 16,000 MiB; otherwise it selects 12 GB.

The 4090 and 5090 measurements establish memory envelopes on 24 GB and 32 GB cards, not certification or speed guarantees for every smaller card of the same architecture. Actual free VRAM must cover the preset's measured requirement plus **64 MiB**. The launcher offers Retry, an explicitly accepted Start slower option where a validated smaller setting exists, or Quit; it does not silently downgrade. See [qualification details and known limits](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/DETAILS.md).

## Measured performance

**12 GB cards in use**

| Card and platform | Story / chat | Code | 128K prompt prefill | Decode at 128K |
| --- | ---: | ---: | ---: | ---: |
| RTX 3060 12 GB, native Linux, headless | ~40 tok/s | ~72 tok/s | ~400 tok/s | 35.9 tok/s |
| RTX 4070 Ti 12 GB, Windows 11, driving a display with desktop apps open | ~67 tok/s | ~100–118 tok/s | ~920 tok/s | ~53 tok/s |

The 3060 short-prompt results come from repeated `bench.sh` runs; its full-context selftest is one run with a 130,048-token prompt and 1,023 generated tokens. The 4070 Ti results are individual desktop runs using the v0.1.3.1 1856 MiB default arena, not a clean benchmark or formal release qualification. The code range includes different output lengths.

**v0.1.3 release gates, 12 GB profile**

| Measured card and platform | Narrative decode | Code decode | 10K prefill | Short first token | Follow-up first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 3060, native Linux | 39.98 tok/s | 71.64 tok/s | 685.63 tok/s | 494 ms | 873 ms |
| RTX 4090, native Linux | 88.97 tok/s | 152.21 tok/s | 2,350.18 tok/s | 293 ms | 548 ms |
| RTX 5090, WSL2 | 114.61 tok/s | 198.66 tok/s | 3,230.36 tok/s | 376 ms | 399 ms |
| RTX 5090, native Windows | 124.22 tok/s | 208.93 tok/s | 3,425.48 tok/s | 159 ms | 264 ms |

These are workload-specific measurements, not forecasts for every prompt or compatible GPU. The first-token workloads in the release gates differ from the short `bench.sh` prompt used for the ~0.23 s 3060 figure in the runtime benchmarks. Cold 10K prefill is not the same workload as reading a full 128K prompt.

On the same RTX 4070 Ti at a 2048 MiB arena, L0xRE measured **54.1 tok/s decode at 128K**, versus **29.8 tok/s** for Ternary Bonsai 2 27B MTP: about **1.8× faster**. This is a single-run comparison; Bonsai's streamed token IDs were incomplete, so its timing is server-reported rather than token-by-token verified. Bonsai was faster on the short narrative prompt on this card (72.0 versus 68.4 tok/s).

Sources, workload definitions and comparison settings: [benchmarks](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/BENCHMARKS.md) and [full v0.1.3 release gates](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V013-RELEASE.md). The accepted WSL SM120 16 GB exception is **2.43% lower cold 10K prefill** than v0.1.2; cached and follow-up turns are faster on that profile.

## Quality

The v0.1.3 release gates scored **120/150** on the **Club-3090 8-pack**, thinking off, temperature 0, on the RTX 3060 / native Linux 12 GB profile, unchanged from its previous baseline. Scores vary by platform, profile and thinking mode; the complete table is in the [release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V013-RELEASE.md). Thinking is off by default. This is a release evaluation, not an external leaderboard claim.

## File integrity

These model files are unchanged by runtime v0.1.3.2:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8,619,127,680 | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| `Qwen3.8-27B-DFlash2-Q2_K.gguf` | 705,430,880 | `e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e` |

The optional vision projector is pinned separately in the [installer catalog](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/install/catalog.json); use the installer to select the release's qualified projector.

## Related checkpoint

[L0xRE-27b-Low-MTP](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate, larger checkpoint with an additional prediction block. Its file hash, runtime setup and qualification are distinct. Use the main model file on this page for the recommended DFlash2 setup.

## Model origin and license

L0xRE-27b-Low is a quantized derivative of **Qwen3.8-27B**, with low-bit projections and higher-precision embeddings/output tensors. Its configured native context capacity is 262,144 tokens; supported serving contexts are determined by the runtime profiles above.

Model weights remain under the base model's research/community terms: [license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). Runtime licensing is separate: the L0xRE runtime tooling ships under the BeeLlama license with [third-party notices](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/THIRD-PARTY-NOTICES.md) and [component licenses](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/tree/main/licenses).
