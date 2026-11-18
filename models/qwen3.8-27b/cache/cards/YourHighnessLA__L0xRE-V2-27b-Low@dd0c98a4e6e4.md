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
  - agent
language:
  - en
pipeline_tag: text-generation
---

# L0xRE-V2-27b-Low

**A capable 27B agent on a 12 GB GPU. Same size and speed as L0xRE-27b-Low, closer to the full model.**

L0xRE-V2-27b-Low is the second-generation weight set for L0xRE Low, a quantized Qwen3.8-27B derivative for local coding assistants and AI agents. The file size is unchanged (**8.62 GB**, every layer on the GPU), as is the speed. V2 is a refit with better calibration and a re-balanced mix of 2-bit and 3-bit codes, and it is measurably closer to the BF16 model, especially on code, tool calls and agent traces.

> **Runs on [L0xRE v0.1.5](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/l0xre-v0.1.5)**, the default model there. Install or update with the [runtime README](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low#install). Updating downloads V2 next to v1, which is kept for rollback. v0.1.4 does not fast-path four of V2's projection shapes, so stay on [L0xRE-27b-Low](https://huggingface.co/YourHighnessLA/L0xRE-27b-Low) if you remain on v0.1.4.

## What improved

KLD against BF16 Qwen3.8-27B, on gen-masked positions of the public [quant-challenge golden set](https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge). Lower is better. **Holdout** is 200 documents frozen before any fitting and scored once, for this release.

| Split | L0xRE-27b-Low (v1) | L0xRE-V2-27b-Low | Change |
| --- | ---: | ---: | ---: |
| Holdout, 200 docs: overall | 0.1223 | **0.0928** | **−24.2%** [−20.4, −26.9] |
| Holdout: agentic mean (code, tools, reasoning, math) | 0.0701 | **0.0497** | **−29.1%** |
| Holdout: code | 0.0752 | **0.0499** | **−33.7%** |
| Holdout: agent and SWE traces | 0.0596 | **0.0383** | **−35.8%** |
| Development, 116 docs: overall | 0.1151 | **0.0833** | **−27.7%** |

V2 has lower KLD on 190 of the 200 holdout documents. Its weakest case is a thinking-off prompt that keeps the xhigh reasoning instruction with an empty think block, where a few chat and tool documents score worse. v0.1.5's thinking-off serving mode does not render that prompt.

## Benchmarks

[club-3090](https://github.com/club-3090) 8-pack N150 (150 tasks: tool calls, instruction following, structured output, data extraction, reasoning/math, bug finding, Hermes agent and CLI tasks):

| Mode | v1 (B02-A2 weights) | **V2** |
| --- | ---: | ---: |
| Thinking off, RTX 3060 | 119 | **121** |
| Thinking on, medium effort, 8K reasoning budget, ×2 (pass@1 / pass@3) | — | **127/133, 125/135** |

For comparison, v1 scored 127/133 with thinking on at **xhigh** effort and no reasoning budget, which took about twice as long per run.

## Speed

RTX 3060 12 GB, native Linux, headless, DFlash2 drafter, `bench.sh`. Paired against v1 weights on the same build and flags:

| Metric | V2 | Versus v1 |
| --- | ---: | ---: |
| Narrative decode | ~45 tok/s | within 2% |
| Code decode | ~78 tok/s | within 2% |
| 10K prompt prefill | ~720 tok/s | within 1% |
| 90K prompt prefill | ~480 tok/s | within 1% |
| Decode at 90K depth (3 runs each) | 61.3 tok/s | v1 50.2 |

Decode at depth depends mostly on how many tokens the drafter gets accepted, so single runs vary widely. The per-step compute cost is the same for both weight sets.

## Model files

| File | Purpose | Size |
| --- | --- | ---: |
| `L0xRE-V2-27b-Low.gguf` | Main model, sha256 `8eb326fddc73abae8ab26ae1c33fdeafd9d773c2702836a9fed77339bc537459` | 8.62 GB |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Recommended DFlash2 drafter (unchanged from v1), sha256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` | 1.14 GB |

**This model requires the [L0xRE BeeLLama Low runtime](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low), v0.1.5 or later.** The API model name is still `L0xRE-27b-Low`, so existing client settings keep working. Stock llama.cpp, Ollama, LM Studio, Transformers and other engines are not supported. The GGUF extension alone does not mean it is compatible.
