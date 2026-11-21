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
   [GitHub Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest) (now v1.0.3).
2. Download `Kintsugi-27B-12G.gguf` from this page into the `models` folder of the bundle.

The model file is the same for every bundle version. When a new bundle comes out, download only the bundle. The setup
steps, licences and full credits are in the [GitHub repo](https://github.com/professorpalmer/Kintsugi-27B). Issues
and pull requests are welcome there.

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
| Decode on an RTX 4070 at 64k / 117k / 200k / 245k (v1.0.3, mean of 2 runs) | 67 / 59 / 49 / 31 tokens per second |
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

Built on the work of the Qwen team, turboderp (exllamav3), the QTIP authors, Escha Labs, YourHighnessLA (L0xRE),
StillDeadcode (quant challenge), Ajay9o9, the llama.cpp / ggml contributors and many more. See
[CREDITS.md](https://github.com/professorpalmer/Kintsugi-27B/blob/main/CREDITS.md).
