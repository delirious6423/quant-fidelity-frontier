---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- 2-bit
- long-context
---

# Kintsugi-27B

![Kintsugi-27B stat card](kintsugi-card.png)

Kintsugi-27B is Qwen3.8-27B (27 billion weights) at 2.47 bits per weight. It runs on one 12 GB NVIDIA card (RTX 20, 30
or 40 series) with the full 262,144-token context. You can use it for anything, under Apache-2.0.

Built on: the **L0xRE-V2 bit allocation** by YourHighnessLA (used unchanged), the **Escha W2 format** by Escha Labs,
and the **EXL3 trellis codec** by turboderp (exllamav3, after QTIP). Our work: our own fits of that layout, the head,
the engine and serving stack (262k context on 12 GB), and task gates on every candidate map. Our own maps with lower KL
failed recall, long-answer or agent tests, so they did not ship
([the study](https://github.com/professorpalmer/Kintsugi-27B/blob/main/research/bit-allocation/README.md)).

**Run it:**

1. Get the latest Windows bundle (engine, launcher, saved-chat layer) from
   [GitHub Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest) (v1.1, Windows and Linux).
2. Download `Kintsugi-27B-12G.gguf` from this page into the `models` folder of the bundle.

The model file is the same for every bundle version. When a new bundle comes out, download only the bundle. The setup
steps, licences and full credits are in the [GitHub repo](https://github.com/professorpalmer/Kintsugi-27B). Issues
and pull requests are welcome there.

- **Systems:** Windows (driver 580+) and Linux x86_64 (driver 570+, glibc 2.34+: Ubuntu 22.04 / 24.04), bundle v1.1. Both on the [Releases page](https://github.com/professorpalmer/Kintsugi-27B/releases/latest).
- **Cards:** RTX 20 / 30 / 40 with 12 GB or more. Checked on Turing, Ampere and Ada cards. RTX 50 is not built.
- **Driver:** NVIDIA 580 or newer (the bundle carries the CUDA 13 runtime). The launcher stops with a message if the
  driver is too old.
- **Coding harnesses:** tested with Claude Code, OpenAI Codex, Pi and OpenCode (6 coding tasks each, no malformed tool
  calls). Use bundle v1.0.2 or later. The settings are in
  [harnesses/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/harnesses/README.md).
- **Chat template:** the bundle carries a fixed chat template. The template inside the .gguf is the original one, which
  refuses a system message after the first turn.

| Test | Result |
|---|---|
| HumanEval (medium thinking) | 161 / 164 |
| AppWorld test_normal (agent tasks, 2 seeds) | 151 / 168 on each seed |
| Needles + exact code copies at 245k tokens | 24 / 24 and 20 / 20 |
| Fidelity to BF16 (public quant-challenge golden set, shipped settings) | top-1 90.8 %, mean KL 0.058, PPL ratio 1.035 |
| Decode on an RTX 4070 at 64k / 117k / 200k / 245k (MTP + lookup, mean of 2 runs) | 67 / 59 / 49 / 31 tokens per second |
| Decode with the optional DFlash2 drafter, RTX 4070, shipped settings | about 25-35 % faster at 4k-64k, about 23 % at 117k, the same at 245k |
| Restore a saved 245k-token conversation | 2.8 s |

How we measured each number:
[benchmarks/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/benchmarks/README.md). The same page
has a head-to-head with Inkling 2 Mini on the same 12 GB card
([section](https://github.com/professorpalmer/Kintsugi-27B/blob/main/benchmarks/README.md#head-to-head-inkling-2-mini-on-the-same-12-gb-card)).

The engine is a llama.cpp fork with EXL3 trellis kernels. The file needs that engine, which is in the bundle. Stock
llama.cpp, Ollama and LM Studio cannot load it.

**About the parameter count on this page:** Hugging Face counts the elements of the tensors in the file and shows about
7B. The coded tensors store packed 2-bit and 3-bit codes, so their element count is much smaller than the number of
weights they hold. The model has 27 billion weights. Format details are in
[model/README.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/model/README.md).

File: `Kintsugi-27B-12G.gguf`, 10,546,687,168 bytes, sha256 `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`.

### Optional: the DFlash2 drafter (faster decode)

`dflash-Qwen3.8-27B-Q4_0.gguf` (1.1 GB, sha256 `0a994382adf17c720c9e5d4292f57344f3b445f058d3e9d5b0f601ae7d6ca3c4`) is a
block drafter for speculative decoding. Put it in the bundle's `models` folder: bundle v1.1 and later use it
automatically (draft 7) and fall back to the model's own MTP drafter when it is not there. Measured on an RTX 4070 with
the shipped settings: about 25-35 % faster decode at 4k-64k tokens of context, about 23 % at 117k, and the same speed at
245k (the drafter takes about 0.8 GB of VRAM, so fewer K/V cells stay in VRAM). On RTX 30 / 40 cards the greedy answers
do not change: speculative decoding checks every drafted token with the model. On RTX 20 (Turing) cards, drafting can
round a near tie the other way (MTP the same); the fidelity is the same.

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

Built on the work of the Qwen team, turboderp (exllamav3), the QTIP authors, Escha Labs, YourHighnessLA (L0xRE),
StillDeadcode (quant challenge), Ajay9o9, the llama.cpp / ggml contributors and many more. See
[CREDITS.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/CREDITS.md).
