---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- 2-bit
- long-context
- 27b
- mac
- hopper
- blackwell
- turing
- ada
- ampere
---

# Kintsugi-27B

**Qwen3.8-27B at 2.47 bits per weight, with the full 262,144-token context on one 12 GB GPU.** Level with full-precision
BF16 on agent tasks, perfect recall at 245k tokens, and it keeps its instructions deep into long agent sessions.
Apache-2.0, fully open.

## Runs on

| Hardware | Status | Get it |
|---|---|---|
| **NVIDIA RTX 20 / 30 / 40 / 50** (12 GB or more) | ✅ Supported | [Windows zip / Linux tarball](https://github.com/professorpalmer/Kintsugi-27B/releases/latest) |
| **NVIDIA H100 / H200** (Hopper) | ✅ Supported (v1.1.2) | [Windows zip / Linux tarball](https://github.com/professorpalmer/Kintsugi-27B/releases/latest) |
| **Apple Silicon Macs** | ✅ Supported: tested on an M5 Max (48 GB); 16 GB Macs and M1-M4 untested | [mac/ kit](https://github.com/professorpalmer/Kintsugi-27B/tree/main/mac) |
| **AMD Radeon (ROCm)** | 🧪 Experimental test build for RDNA4 (R9700 / RX 9070 / 9070 XT), Linux: testers wanted | [AMD test build](https://github.com/professorpalmer/Kintsugi-27B/releases/tag/amd-test-1) |
| NVIDIA B200, DGX Spark (GB10) | Runs the engine's portable code (compiled once at first start, about 1.5 min); untested | same bundle |
| Intel GPUs | Not supported yet | |

Checked per architecture: RTX 5070 and H100 match our RTX 4070's quality (KL within 0.03 % / 0.02 %), with 24 / 24
needles and 20 / 20 exact code copies at 245k tokens on both; the Mac matches the CUDA engine's KL and finds 24 / 24
needles at 259k ([cross-architecture](https://github.com/professorpalmer/Kintsugi-27B/blob/main/research/cross-architecture/README.md)).

![Kintsugi-27B stat card](kintsugi-card.png)

![Kintsugi-27B v1.1.2 highlights](images/kintsugi-27b-highlights.png)

![Persistence at depth](images/persistence-at-depth.png)

![Kintsugi-27B vs Qwen3.8-27B BF16 on BFCL multi-turn tool use](images/kintsugi-vs-bf16-bfcl.png)

## Quick start

**Windows / Linux (NVIDIA):**
1. Download the bundle for your system from [GitHub Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest)
   (v1.1.2: Windows zip 679 MB, Linux tarball 786 MB).
2. Download `Kintsugi-27B-12G.gguf` from this page into the bundle's `models` folder.
3. Start it: Windows `powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait`, Linux `./start-server.sh --wait`.
   Then open http://127.0.0.1:8080/ or point an OpenAI-compatible app or coding harness at `http://127.0.0.1:8080/v1`.

Needs NVIDIA driver 580+ on Windows, 570+ on Linux (glibc 2.34+: Ubuntu 22.04 / 24.04). The launcher checks your card
and driver and tells you if something is missing. The model file is the same for every bundle version: for a new bundle,
download only the bundle.

**Mac (Apple Silicon):** `git clone https://github.com/professorpalmer/Kintsugi-27B.git`, then `cd Kintsugi-27B/mac`,
`./run.sh install` (downloads and converts the model), then `./run.sh chat` or `./run.sh serve`.
Full guide: [mac/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/mac/README.md).

## Results

| Test | Result |
|---|---|
| AppWorld, agent tasks (fresh seeds 102 + 103) | 294 / 336, level with full-precision BF16 (293) |
| BFCL v3 multi-turn tool use, 800 entries | 505 / 800; full-precision BF16 461 (one run each; setup differs, [details](https://github.com/professorpalmer/Kintsugi-27B/blob/main/benchmarks/bfcl/README.md)) |
| HumanEval (medium thinking) | 161 / 164 |
| Needles + exact code copies at 245k tokens | 24 / 24 and 20 / 20 |
| Standing system-prompt rules after up to 237k tokens of its own agent trace | all 5 kept, no decay; not worse than BF16 |
| Fidelity to BF16 (public quant-challenge golden set) | top-1 90.8 %, mean KL 0.058, PPL ratio 1.035 |
| Decode on an RTX 4070 at 64k / 117k / 200k / 245k (MTP + lookup) | 67 / 59 / 49 / 31 tokens per second |
| Restore a saved 245k-token conversation | 2.8 s |
| Coding harnesses: Claude Code, OpenAI Codex, Pi, OpenCode | 5 / 6 tasks in each, no malformed tool calls ([settings](https://github.com/professorpalmer/Kintsugi-27B/blob/main/harnesses/README.md)) |
| Images (optional projector) | 19 / 20 on our image set, the same as stock llama.cpp with the same files |

How each number was measured: [benchmarks/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/benchmarks/README.md),
including a head-to-head with Inkling 2 Mini on the same 12 GB card. The persistence test sees only large gaps
([the study](https://github.com/professorpalmer/Kintsugi-27B/blob/main/research/persistence-at-depth/README.md)).

## Optional add-ons

<details>
<summary><b>Images: the vision projector</b> (bundle v1.1.2+)</summary>

Download `mmproj-Qwen3.8-27B-Q8_0.gguf` (629,247,008 bytes, sha256
`2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb`) from
[ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (linked, not mirrored here) into the
bundle's `models` folder, and the launcher turns images on (chat requests in OpenAI and Anthropic formats, and tool
results such as screenshots).

- **cpu** (the default on cards under 16 GB): no VRAM cost, every K/V cell stays in VRAM; 5-25 s per image.
- **gpu** (the default from 16 GB): about 1.3 GB of VRAM; 1-3 s per image. On a 12 GB card it leaves about 61,000 of
  144,000 K/V cells in VRAM.
- With images on, drafting is MTP + lookup: the DFlash2 drafter does not work with images yet.
- Our checks: 19 / 20 on our image set in both modes (stock llama.cpp with the same files: 19 / 20), images after a
  117k-token conversation 4 / 4, the same text output with the projector loaded (50 / 50 prompts). The built-in web page
  is not tested with images.
- Credits: the Qwen3.8-27B vision encoder by the Qwen team (Apache-2.0), converted to GGUF by ggml-org with llama.cpp's
  convert_hf_to_gguf.py (MIT).

</details>

<details>
<summary><b>Faster decode: the DFlash2 drafter</b> (bundle v1.1+)</summary>

`dflash-Qwen3.8-27B-Q4_0.gguf` (1.1 GB, sha256 `0a994382adf17c720c9e5d4292f57344f3b445f058d3e9d5b0f601ae7d6ca3c4`) is a
block drafter for speculative decoding. Put it in the bundle's `models` folder: the launcher uses it automatically
(draft 7) and falls back to the model's own MTP drafter when it is not there. Measured on an RTX 4070 with the shipped
settings: about 25-35 % faster decode at 4k-64k tokens of context, about 23 % at 117k, and the same speed at 245k (the
drafter takes about 0.8 GB of VRAM, so fewer K/V cells stay in VRAM). On RTX 30 / 40 cards the greedy answers do not
change: speculative decoding checks every drafted token with the model. On RTX 20 (Turing) cards, drafting can round a
near tie the other way (MTP the same); the fidelity is the same.

**Source and licence.** This file is the Q4_0 GGUF by ggml-org
(https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF, revision cc54a8a3955ee1a099011952d0ef2eab24a7eab3), converted from
Inco AI's DFlash2 for Qwen3.8-27B (https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), unchanged. Licence: Apache-2.0
(`LICENSE-dflash-drafter`). The current file on ggml-org's page is 192 bytes larger: a later metadata-only update. We
keep the cc54a8a3 bytes, because our tests ran on them.

**Citation** (the drafter's card asks for both):

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

</details>

## Good to know

- **Engine:** the file needs our engine (a llama.cpp fork with EXL3 trellis kernels), which is in every bundle. Stock
  llama.cpp, Ollama and LM Studio cannot load it.
- **Chat template:** the bundle carries a fixed chat template. The template inside the .gguf is the original one, which
  refuses a system message after the first turn.
- **"7B params" on this page:** Hugging Face counts the stored elements of the GGUF file. Our 2-bit and 3-bit projections
  are stored as packed trellis code words (several weights per stored element), so the page shows about 7B. The model
  has all 27 billion weights of Qwen3.8-27B; nothing is pruned. Format details:
  [model/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/model/README.md).
- **File:** `Kintsugi-27B-12G.gguf`, 10,546,687,168 bytes, sha256
  `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`.
- **Issues and pull requests:** welcome on [GitHub](https://github.com/professorpalmer/Kintsugi-27B).

## Built on

The **L0xRE-V2 bit allocation** by YourHighnessLA (used unchanged), the **Escha W2 format** by Escha Labs, and the
**EXL3 trellis codec** by turboderp (exllamav3, after QTIP). Our work: our own fits of that layout, the head, the engine
and serving stack (262k context on 12 GB), and task gates on every candidate map: our own maps with lower KL failed
recall, long-answer or agent tests, so they did not ship
([the study](https://github.com/professorpalmer/Kintsugi-27B/blob/main/research/bit-allocation/README.md)).

Thanks to the Qwen team, turboderp (exllamav3), the QTIP authors, Escha Labs, YourHighnessLA (L0xRE), StillDeadcode
(quant challenge), Ajay9o9, Aldehir Rojas (llama.cpp #32381), the llama.cpp / ggml contributors and many more. Full
list: [CREDITS.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/CREDITS.md).