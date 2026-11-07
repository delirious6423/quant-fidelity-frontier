---
license: cc-by-sa-4.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
- it
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- quantized
- trellis
- 2-bit
- qwen3_5
- gated-deltanet
- ik_llama.cpp
---
# Qwen3.8-27B-TOQ-v1

**Qwen3.8-27B with 2-bit trellis weights: 6.17 GiB on the GPU.**
TOQ (*Trellis-Operator Quantization*) is a trellis quantizer of the QTIP / EXL3 family followed by a short
distillation from the original bf16 model. This is **not** an official Qwen release; the base model is
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).

> **This file does not run on stock llama.cpp, Ollama, LM Studio or vLLM.** It needs the
> [mango-toq](https://github.com/Federico-Anastasi/mango-toq) engine (a fork of ik_llama.cpp), or
> [Mango Code](https://github.com/Federico-Anastasi/mango-code), a desktop app that installs engine and model together.

| | |
|---|---|
| file | `Qwen3.8-27B-TOQ-v1.gguf`, 9,489,952,128 bytes (8.84 GiB) |
| weights on the GPU | 6.17 GiB: 64 layers at 2 bits per weight, output head at 3 bits; plus 176 MiB (f32) for the distillation correction |
| in system RAM | the token embedding table, bf16 (about 2.5 GB) |
| architecture | Qwen3.8 hybrid: 64 layers, 48 Gated DeltaNet + 16 full attention (`qwen35` in the GGUF) |
| context | 262,144 tokens native; tested up to 120k |
| multi-token prediction | the MTP layer is included, quantized the same way (101.5 MiB instead of 810 MiB in bf16) |
| images | the GGUF holds the language model only; vision works with the original Qwen3.8 projector, `mmproj-F16.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

## How it was made

Three parts. The quantization method is not ours; the distillation and the engine are.

1. **Trellis quantization.** Method: QTIP (Tseng et al., arXiv:2406.11235) as implemented in EXL3 by turboderp; the
   code is our own implementation. Weights are rotated with a block Hadamard transform (input side only) and coded as
   a path through a trellis with a computed codebook ("mul1"), chosen with BlockLDLQ against layer Hessians measured
   on 250 rows × 2,048 tokens of EXL3's calibration mix. Our tiles are 64×4 (EXL3: 16×16), the state is 14 bits
   (EXL3: 16) and the scales are per row, so the file is not EXL3-compatible.
2. **Distillation.** The scales of every quantized linear layer, plus a rank-32 low-rank correction on the 64 MLP gate
   projections, were trained against the bf16 model's logits (KL divergence) on 0.39M tokens (48 steps × 4 rows ×
   2,048 tokens). The trellis codes are unchanged. The correction is stored in the GGUF as tensors and applied on top
   of the quantized weight (W·x + B·(A·x)).
3. **Engine.** [mango-toq](https://github.com/Federico-Anastasi/mango-toq), branch `mango`: the trellis weight types on
   CUDA, CPU (AVX2) and Vulkan; an int8 tensor-core GEMM for prompt processing; prompt processing on the main GPU with
   the other GPUs' weights streamed from pinned RAM; CUDA graphs on Pascal and Turing; multi-token prediction; an
   on-disk cache of conversations.

## Quality

### MMLU, answered directly

1,548 questions of the MMLU validation split, forced answer `Answer: X`, temperature 0, reasoning off; same questions
and prompt for every model. TOQ-v1 ran on mango-toq, the other models on PrismML's llama.cpp fork. A control run of a
reference model on mango-toq has not been done.

| model | size | MMLU |
|---|---|---|
| Qwen3.8-27B UD-IQ4_XS | 13.3 GiB file | 82.56% |
| **Qwen3.8-27B-TOQ-v1** | 6.17 GiB on the GPU | **77.78% ± 1.06** |
| TOQ, same weights before distillation | 6.17 GiB on the GPU | 78.29% ± 1.05 |
| Ternary Bonsai 2 27B (PrismML) | 5.27 GiB on the GPU | 77.91% |
| Qwen3.8-27B UD-IQ2_XXS | ~6.4 GiB | 72.87% |

Paired on the same questions (right only for TOQ-v1 / right only for the other model): Bonsai 2 94 / 96 (tie);
UD-IQ2_XXS 174 / 98 (better, p < 0.001); UD-IQ4_XS 55 / 129 (worse); before distillation 51 / 59 (tie). The
distillation did not change MMLU. On size: all of Bonsai 2's language-model weights take 5.93 GB, including a ternary
token table; this file is 9.49 GB because its token table stays bf16 (in system RAM).

### Distance from the original

KL divergence to the bf16 model on held-out rows of the distillation sources, 4 rows × 2,048 tokens per domain.

| domain | before distillation | TOQ-v1 |
|---|---|---|
| web text | 0.1404 | 0.1187 |
| code | 0.2621 | 0.2199 |
| science and math | 0.1562 | 0.1272 |
| agentic traces | 1.7001 | 1.1265 |
| mean | 0.5647 | 0.3981 |

Four rows per domain is a small control set, drawn from the same sources as the training text. On agentic traces the
model stays about ten times farther from bf16 than on web text.

### Perplexity and KL divergence against Qwen3.8 at 4 bits

Both models read the same text, the first 150 chunks of 512 tokens of the wikitext-2 test set, on the same engine
(mango-toq, CUDA, f16 KV cache). The reference is [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
UD-IQ4_XS (13.3 GiB), not the bf16 model: the KL divergence says how far TOQ is from a good 4-bit file.

| model | PPL wikitext-2 (c=512, 150 chunks) | mean KLD vs UD-IQ4_XS | same top token |
|---|---|---|---|
| Qwen3.8-27B UD-IQ4_XS (reference) | 6.763 ± 0.085 | — | — |
| **Qwen3.8-27B-TOQ-v1** | **7.501 ± 0.095** (+10.9%) | **0.181 ± 0.002** | **81.8%** |

KL divergence distribution: median 0.088, 90th percentile 0.389, 99th percentile 1.735.

The quantizer's calibration text (EXL3's mix, which includes Wikipedia articles) shares no text with the wikitext-2
test set: 4 of its 330,000 distinct 8-word sequences also appear in the test set.

## Speed

Measured with Mango Code's benchmark: the model started as the app starts it, real text cut to the exact number of
tokens, temperature 0, 256 tokens generated, no prompt cache, q8_0 KV cache, multi-token prediction with 2 draft
tokens. CUDA build. Tokens per second.

| GPUs | context | prompt processing 1k / 32k / 64k | generation 1k / 32k / 64k |
|---|---|---|---|
| RTX 3060 Ti + GTX 1660 SUPER (vision encoder on a GTX 1080) | 120k | 567 / 649 / 533 | 27.5 / 26.5 / 23.2 |

Earlier measurements, same quantized weights before the distillation correction was added to the file:

| setup | prompt processing | generation |
|---|---|---|
| RTX 3060 Ti alone, short context, no MTP | 929 (512-token prompt) | 35.7 |
| same, MTP with 2 draft tokens | — | 46.5 (77% of drafts accepted) |
| RTX 3060 Ti + GTX 1660 SUPER, 120k prompt, MTP with 2 draft tokens | 393 | 21.1 |

There are no speed measurements for other GPUs, nor for the Vulkan and CPU builds.

## Requirements

- **NVIDIA GPU (CUDA 12 build)**: the measured path. Tested on RTX 3060 Ti (sm_86), GTX 1660 SUPER (sm_75) and GTX
  1080 (sm_61). The build also targets sm_70, sm_80 and sm_89 (with PTX for newer GPUs), untested.
- **Vulkan and CPU (AVX2) builds**: supported, less tested (reduced shapes and a software Vulkan).
- **GPU memory**: 6.17 GiB of weights plus the context. One 8 GB card holds about 14k tokens of context (RTX 3060 Ti);
  RTX 3060 Ti + GTX 1660 SUPER hold 120k. A 6 GB card alone does not fit the weights.
- **Disk**: 8.84 GiB for the model, 0.93 GB more for the vision projector.
- **OS**: the prebuilt engine is Windows 10/11 x64. On Linux the engine builds from source, untested.
- On Windows, set the NVIDIA option "CUDA – Sysmem Fallback Policy" to "Prefer No Sysmem Fallback": otherwise running
  out of GPU memory does not fail, it slows generation down 10–30×.

## How to run

**With Mango Code**: install the app; at first run it detects your GPUs, downloads the matching engine build and the
model with checksums, and starts it with tuned arguments, image support included.

**By hand** (Windows):

1. From the [mango-toq release `toq-35e7256`](https://github.com/Federico-Anastasi/mango-toq/releases/tag/toq-35e7256)
   download `toq-35e7256-win-cuda-12.zip` (or `-vulkan`, `-cpu`) and unzip it. The CUDA zip does not include the CUDA
   runtime: if the CUDA 12 toolkit is not installed, unzip llama.cpp's
   [`cudart-llama-bin-win-cuda-12.4-x64.zip`](https://github.com/ggml-org/llama.cpp/releases/download/b12110/cudart-llama-bin-win-cuda-12.4-x64.zip)
   into the same folder.
2. Download `Qwen3.8-27B-TOQ-v1.gguf` and `chat_template.jinja` from this repository.
3. Start the server:

```powershell
.\llama-server.exe -m Qwen3.8-27B-TOQ-v1.gguf --jinja --chat-template-file chat_template.jinja `
    -ngl 999 -fa 1 -wgt 1 -ctk q8_0 -ctv q8_0 -c 12288 --host 127.0.0.1 --port 8080
# images:                  --mmproj mmproj-F16.gguf --image-min-tokens 1024
# multi-token prediction:  --spec-type mtp:n_max=2,p_min=0.0
```

The distillation correction is inside the GGUF: do **not** add `--lora` (it would be applied twice). `-wgt 1` keeps
the compute buffer small, which matters on 8 GB cards. `chat_template.jinja` is the Qwen3.8 template with one line
changed: a system message in the middle of a conversation is rendered instead of rejected; the agentic part of the
distillation text was rendered with it. The server exposes an OpenAI-compatible API at `/v1`.

## Training data (distillation only)

0.39M tokens in 2,048-token rows, mixed 40% web, 25% code, 25% science and math, 10% agentic. The text itself is not
redistributed.

| domain | sources |
|---|---|
| web | HuggingFaceFW/fineweb-edu (ODC-By 1.0) |
| code | 102 files from 19 public repositories (CPython, Django, NumPy, Flask, Requests, Rust, Tokio, ripgrep, Linux, Redis, curl, git, SQLite, LevelDB, Go, Node.js, React, Vue, MDN learning-area), each under its own license; 36 files from the author's own working folders, 8 of which are copies of third-party code (3 from opencode, 2 from llama.cpp, 1 from vLLM, 2 scripts from Anthropic's document skills, under a proprietary license) |
| science and math | EleutherAI/proof-pile-2, arXiv subset (the articles keep their own licenses); HuggingFaceTB/finemath (ODC-By 1.0) |
| agentic | nebius/SWE-rebench-openhands-trajectories (CC BY 4.0); NousResearch/hermes-function-calling-v1 (Apache-2.0) |

Decontamination: code, science and agentic documents sharing three or more 8-word sequences with MMLU questions
(validation, dev and test) were dropped (5 science documents); web documents sharing three or more 8-word sequences
with wikitext-2 test were dropped.

## Limitations

- **Not evaluated as an agent.** MMLU is a multiple-choice test; it does not say whether the model completes
  multi-step tool-use tasks. On agentic traces the model is far from the original (KL table above): expect more
  tool-call format errors and shorter reliable horizons than with the bf16 model.
- Context tested up to 120k tokens, on two GPUs only.
- Vulkan and CPU builds: less tested, no speed numbers. Images tested only with the CUDA build.
- Generation at temperature 0 is not bit-for-bit repeatable across runs on CUDA. With multi-token prediction the text
  can diverge from the same prompt without it after some tokens: verification uses different kernels (perplexity at
  batch 1 and batch 3: 12.5423 and 12.5426).
- On Pascal and Turing GPUs the trellis decoding, not memory bandwidth, limits generation speed.
- The engine is a fork with a single maintainer and is not part of upstream ik_llama.cpp or llama.cpp. No tool to
  convert other models to this format is released.
- The model inherits the limitations and safety properties of Qwen3.8-27B and adds quantization error. Do not let an
  agent with unrestricted shell access work on files you cannot lose.

## Files

| file | |
|---|---|
| `Qwen3.8-27B-TOQ-v1.gguf` | 9,489,952,128 bytes, sha256 `f435afc543f84b8cbc278dc7059599034b6ef957973300dde0b28f52db584a24` |
| `chat_template.jinja` | chat template (see How to run) |
| `LICENSE`, `NOTICE` | licenses and attributions |
| `SHA256SUMS` | checksums |

## License

The TOQ-v1 weights are released under **CC BY-SA 4.0**. They are a modified version of Qwen3.8-27B (Apache-2.0,
copyright the Qwen Team): the Apache-2.0 license and notices of the base model are kept in `LICENSE` and `NOTICE`, and
the changes are listed in `NOTICE`. "Qwen" names the original model; this is not an official Qwen release. The
distillation sources keep their own licenses (list above and in `NOTICE`). The mango-toq engine is MIT, like
ik_llama.cpp and llama.cpp; Mango Code is AGPL-3.0-or-later.

## Credits

- Qwen Team, Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- Tseng, Chee, Sun, Kuleshov, De Sa, *QTIP: Quantization with Trellises and Incoherence Processing*, arXiv:2406.11235
- turboderp, ExLlamaV3 (EXL3): https://github.com/turboderp-org/exllamav3
- Iwan Kawrakow and contributors, ik_llama.cpp; Georgi Gerganov and contributors, llama.cpp: the base of the engine
- Unsloth, the GGUF conversion of the Qwen3.8 vision projector: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- Trofimova et al., *OpenHands Trajectories with Qwen3-Coder-480B-A35B-Instruct* (Nebius, 2025)
- Hugging Face (fineweb-edu, finemath), EleutherAI (proof-pile-2), Nous Research (hermes-function-calling-v1)
- Comparison model: Ternary Bonsai 2 27B, PrismML
