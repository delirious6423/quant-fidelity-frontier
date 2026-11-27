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

> **V2 is the default model.** [L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low) is the model used for the current v0.1.6.5 sparse runtime and its 256K / 12 GB profile. This card documents the original V1 weights and their v0.1.3 measurements. Use V2 for new installs; V1 remains available for rollback.

**27B parameters. 72 tok/s code at v0.1.3. 128K context and vision on a 12 GB GPU; the current sparse runtime with V2 reaches 256K.**

L0xRE-27b-Low is a memory-focused, quantized Qwen3.8-27B derivative for a local coding assistant or AI agent. With the [L0xRE BeeLLama Low runtime](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low), every model layer runs on the GPU, without CPU layer offload. The main model is **8.62 GB on disk**; “27b” is the parameter count, not the download size.

The **72 tok/s** headline is code decode on an **RTX 3060 12 GB, native Linux, headless**, measured on the v0.1.3 runtime. Speeds depend on the card and workload; measurements below identify both.

> **Runtime: L0xRE v0.1.6.5 (Sparse Defaults) or later.** The [repository home page](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low) always names the current release and the install/update steps, so start there rather than at a pinned release. Runtime updates do not change the V1 weights in this repository.

## What changed in the runtime

The runtime has moved well past the v0.1.3 build this card was measured on. **v0.1.6 made sparse attention the default above 8K tokens and raised the 12 GB profile to 262,144 tokens; v0.1.6.5 promotes the current sparse defaults.** Below 8K the model is fully dense and logit-identical to the dense path. Above 8K each query attends to a sink, a local window, and the most relevant earlier tokens selected by a learned D-LSA map, with exact-match retrieval links for repeated records. See the [v0.1.6.5 release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V0165-RELEASE.md) and the [v0.1.6 notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V016-RELEASE.md).

Sparse attention and the 256K / 12 GB profile are measured with **L0xRE-V2-27b-Low**, not the V1 weights on this page. V1 remains useful for rollback and for reproducing the v0.1.3 measurements below; use V2 for a new 256K installation.

The earlier v0.1.3 work (faster decode, much faster prompt reuse, the 1856 MiB desktop arena and `doctor`) remains in the runtime; see the [v0.1.3 release notes](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/V013-RELEASE.md).

## Sparse attention and 256K context

The current runtime serves **262,144 tokens on the 12 GB profile**. The measured v0.1.6 results with V2 + DFlash2 at temperature 0:

| Test | Dense | v0.1.6 sparse |
| --- | ---: | ---: |
| Needle recall, 8 records, 16K–192K (RTX 3060 and RTX 5090) | 224/224 on each card | 223/224 on each card |
| Held-out retrieval, two prompt sets, RTX 5090 | 269/288 | 268/288 |
| Responses below 8K tokens | — | bit-identical to the dense path |
| Prefill cost | — | RTX 3060 −0.2…+0.1%; RTX 4090 −0.4…−0.1%; RTX 5090 within 1.6% |

v0.1.6.5's defaults are 256-token tiles, top-k 16,384, an 11,648-token sink, pooled prefill, a generation anchor, and CLOCK unit-cache gather v3. Its lab run (F1 + DFlash2 + s2 template) measured KL versus dense at 0.0107 for 42–78K, 0.0140 for 128–170K, and 0.0263 for 180–200K, with recall 72/72. Known limits: one sequence at a time (`-np 1`), a 1-in-862 confident-disagreement rate with dense, and neighbour copying at counting positions. The sparse map is `qwen38-dlsa-d2048.l0xmap`, installed next to the model.

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

The vision option downloads the release's qualified projector separately; v0.1.6.5 uses the **Q8_0 projector**. It is not included in the main GGUF file. Omit `-Vision` / `--vision` for a text-only installation. See [installer options](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/install/README.md).

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

The current v0.1.6.5 runtime ships six packages (Linux and native Windows, SM86 / SM89 / SM120). **All use the 12 GB profile, including cards with more VRAM, and serve 262,144 tokens with sparse attention.** V1 was measured at 128K; the 256K sparse profile is qualified with V2.

| Runtime package | Current v0.1.6.5 profile | V1 measurement scope |
| --- | --- | --- |
| Linux SM86, RTX 30 series | 12 GB / 256K | RTX 3060, native Linux; V1 measured at 128K |
| Linux SM89, RTX 40 series | 12 GB / 256K | RTX 4090, native Linux; V1 measured at 128K (v0.1.3 also offered a 16 GB / 256K profile) |
| Linux / WSL SM120, RTX 50 series | 12 GB / 256K | RTX 5090, WSL2; V1 measured at 128K (v0.1.3 also offered a 16 GB / 256K profile) |
| Native Windows SM120, RTX 50 series | 12 GB / 256K | RTX 5090, native Windows; V1 measured at 128K (v0.1.3 also offered a 16 GB / 256K profile) |
| Native Windows SM86 / SM89, RTX 30 / 40 series | 12 GB / 256K | Release packages are build-verified, not formally hardware-qualified |

256K means **262,144 tokens**; 128K means **131,072 tokens**. The current release uses one 12 GB profile across the package set. V1's original 16 GB profiles are historical; see the v0.1.3 release notes.

The 4090 and 5090 measurements establish memory envelopes on 24 GB and 32 GB cards, not certification or speed guarantees for every smaller card of the same architecture. Actual free VRAM must cover the preset's measured requirement plus **64 MiB**. The launcher offers Retry, an explicitly accepted Start slower option where a validated smaller setting exists, or Quit; it does not silently downgrade. See [qualification details and known limits](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/docs/DETAILS.md).

## Measured performance

**V1 at v0.1.3, 12 GB profile (historical).** For current v0.1.6.5 numbers with V2, see the [V2 card](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low#speed).

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

These V1 model files are unchanged by runtime updates:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `L0xRE-27b-Low.gguf` | 8,619,127,680 | `b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1,143,006,816 | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| `Qwen3.8-27B-DFlash2-Q2_K.gguf` | 705,430,880 | `e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e` |

The optional vision projector is pinned separately in the [installer catalog](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/install/catalog.json); use the installer to select the release's qualified projector.

## Related checkpoints

[L0xRE-V2-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low) is the current default: same size and speed, 24% closer to BF16 overall, and the model used for the v0.1.6.5 sparse 256K configuration. [L0xRE-27b-Low-MTP](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low-MTP) is a separate, larger checkpoint with an additional prediction block. Its file hash, runtime setup and qualification are distinct. Use the V2 file for a new install, or the main V1 file on this page for the original DFlash2 setup.

## Model origin and license

L0xRE-27b-Low is a quantized derivative of **Qwen3.8-27B**, with low-bit projections and higher-precision embeddings/output tensors. Its configured native context capacity is 262,144 tokens; supported serving contexts are determined by the runtime profiles above.

Model weights remain under the base model's research/community terms: [license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). Runtime licensing is separate: the L0xRE runtime tooling ships under the BeeLlama license with [third-party notices](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/blob/main/THIRD-PARTY-NOTICES.md) and [component licenses](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/tree/main/licenses).
