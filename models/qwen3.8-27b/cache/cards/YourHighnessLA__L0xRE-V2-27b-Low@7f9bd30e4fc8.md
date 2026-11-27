---
license: other
license_name: qwen-research-terms-derivative
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags:
  - l0xre
  - gguf
  - speculative-decoding
  - sparse-attention
  - long-context
  - 27b
  - 12gb-vram
  - agent
language:
  - en
pipeline_tag: text-generation
---

# L0xRE-V2-27b-Low

**A capable 27B agent on a 12 GB GPU. 256K context with sparse attention, the same size and speed as L0xRE-27b-Low, and closer to the full model.**

L0xRE-V2-27b-Low is the second-generation weight set for L0xRE Low, a quantized Qwen3.8-27B derivative for local coding assistants and AI agents. The file size is unchanged (**8.62 GB**, every layer on the GPU), as is the speed. V2 is a refit with better calibration and a re-balanced mix of 2-bit and 3-bit codes, and it is measurably closer to the BF16 model, especially on code, tool calls and agent traces.

> **Runtime: L0xRE v0.1.6.5 (Sparse Defaults) or later.** This is the default model for the sparse runtime. The [repository home page](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low) always names the current release and the install/update steps, so start there rather than at a pinned release. Runtime updates do not change the weights in this repository.

## Sparse attention and 256K context

v0.1.6 made sparse attention the default above 8K tokens and raised the 12 GB profile to **262,144 tokens**. v0.1.6.5 promotes the current defaults. The same 12 GB card now serves twice the context, with retrieval close to dense in the measured tests.

- **Below 8K tokens:** fully dense. Outputs are logit-identical to the dense path, not merely close.
- **Above 8K tokens:** each query attends to a sink, a local window, and the most relevant earlier tokens selected by a learned D-LSA map (`qwen38-dlsa-d2048.l0xmap`, installed next to the model). v0.1.6.5 defaults are 256-token tiles, top-k 16,384, an 11,648-token sink, pooled prefill, a generation anchor, and CLOCK unit-cache gather v3.
- **Exact-match retrieval links:** a rare token sequence seen earlier hard-links the current block back to those earlier blocks. This targets the main failure of learned sparse attention, picking the wrong one of several similar records. Generation uses the plain sparse selection, so decode speed is unchanged.
- **Dense fallback:** `L0XRE_SPARSE=0` disables sparse attention. Dense mode is not the default above 8K; it mis-handles some agent transcripts past about 30K tokens.

### Measured sparse quality

These are the v0.1.6 release measurements with L0xRE-V2-27b-Low + DFlash2 at temperature 0:

| Test | Dense | v0.1.6 sparse |
| --- | ---: | ---: |
| Needle recall, 8 records in real agent transcripts, 16K–192K (RTX 3060 and RTX 5090) | 224/224 on each card | 223/224 on each card |
| Held-out retrieval (code, paths, paraphrased questions), two prompt sets, RTX 5090 | 269/288 | 268/288 |
| Responses below 8K tokens | — | bit-identical to the dense path (logit hashes) |
| Prefill cost, same runtime without the links | — | RTX 3060 −0.2…+0.1%; RTX 4090 −0.4…−0.1%; RTX 5090 within 1.6% |

v0.1.6.5's lab quality run (F1 + DFlash2 + s2 template) measured KL versus dense on real Hermes trajectories at **0.0107** for 42–78K, **0.0140** for 128–170K, and **0.0263** for 180–200K, with recall **72/72**, matching dense. This is an internal lab result, not an external leaderboard.

Known limits: one sequence at a time (`-np 1`); exact-match links can copy a neighbour at counting positions (for example a chapter number after "CHAPTER"); confident disagreements with dense attention remain at the plain-sparse rate of **1 in 862 positions**; tree verify runs inside the resident window up to 8K, and deeper verify uses the 8-row chain.

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

Club-3090 8-pack N150 (150 tasks: tool calls, instruction following, structured output, data extraction, reasoning/math, bug finding, Hermes agent and CLI tasks):

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

### v0.1.6.5 sparse runtime check

On the same RTX 3060 12 GB, the v0.1.6.5 release kit booted at **11,175 MiB / 12,288 MiB** with the full 262,144-token sparse context and Q8_0 vision. `bench.sh` measured **55.73 tok/s** narrative (transformer essay) and **100.17 tok/s** code (Python quicksort), with **839 tok/s** prefill at 10K and **688 tok/s** prefill at 90K. These use the v0.1.6.5 sparse defaults and are not directly comparable with the paired agent-prompt table above.

## Model files

| File | Purpose | Size |
| --- | --- | ---: |
| `L0xRE-V2-27b-Low.gguf` | Main model, sha256 `8eb326fddc73abae8ab26ae1c33fdeafd9d773c2702836a9fed77339bc537459` | 8.62 GB |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Recommended DFlash2 drafter (unchanged from v1), sha256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` | 1.14 GB |
| `qwen38-dlsa-d2048.l0xmap` | D-LSA sparse-attention map installed next to the model (v0.1.6+) | 6 MB |

**This model requires the [L0xRE BeeLLama Low runtime](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low), v0.1.5 or later. The current v0.1.6.5 release adds the sparse defaults described above.** The API model name is still `L0xRE-27b-Low`, so existing client settings keep working. Stock llama.cpp, Ollama, LM Studio, Transformers and other engines are not supported. The GGUF extension alone does not mean it is compatible.
