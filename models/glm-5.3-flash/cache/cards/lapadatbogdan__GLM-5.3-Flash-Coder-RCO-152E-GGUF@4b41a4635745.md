---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - expert-pruning
  - rco
  - code
  - agentic
  - llama.cpp
---

# GLM-5.3-Flash-Coder-RCO-152E-GGUF

> **New release (October 2026).** This repository used to hold REAP144, a 144-expert version calibrated on Python
> agentic sessions only. REAP144 looped on general questions and on other programming languages and could get simple
> arithmetic wrong. It is replaced by **RCO-152E**: 152 of 288 experts per layer, chosen by optimization on a mix of
> agentic sessions, code in 21 languages, math and general text, then healed. The old REAP144 files were removed; they
> remain available in the repository history.

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B MoE) with **152 of 288 experts per layer**,
built for coding and agentic use on a home PC. **321B-A18B becomes ~170B-A17B.**

Recipe, mask, patches and evaluation harness: [github.com/lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E](https://github.com/lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E)

## What changed from REAP144

| | REAP144 (old) | RCO-152E (this release) |
|---|---|---|
| Loop test: problems in 30 hard instruction-following runs | 22 | **4** (0 repetition loops) |
| 17 x 23, 1234 x 5678, Fibonacci, small executed code task, tool call | arithmetic errors | **pass** at every quant (Q3_K missed Fibonacci once in one run) |
| Experts per layer | 144 | 152 (incl. 8 generalists) |
| Calibration | Python agentic sessions | agentic sessions + code in 21 languages + math + 15% general text |

## Files

| Quant | Size | Fits (GPU VRAM + RAM) | KLD to Q8 |
|---|---|---|---|
| Q8_0 | 180.4 GB (4 parts) | 24 + 192, multi-GPU | 0 |
| Q4_K_M | 102.6 GB (3 parts) | 24 + 96, 32 + 96, 128 GB unified | 0.024 |
| Q3_K | 75.1 GB (2 parts) | 24 + 64, 32 + 64 | 0.045 |
| IQ2_XXS (GSQ) | ~47 GB | 24 + 32 | in progress |

KLD: llama.cpp `llama-perplexity --kl-divergence` against the Q8_0 of this model on held-out code and agentic text.
Same top token as Q8: Q4_K_M 94.5%, Q3_K 92.5%.

A 2-bit version for 24 GB VRAM + 32 GB RAM is in progress (GSQ vector quantization of the experts + quantization-aware
healing) and will be added here when it passes the same tests.

## Results (Q8_0, no special settings)

Against the full GLM-5.3-Flash (Q8), same prompts, default sampling:

| Benchmark | RCO-152E | Full model |
|---|---|---|
| AIME 2025 (20) | 19 | 20 |
| MATH-500 subset (6) | 6 | 6 |
| LiveCodeBench subset (6) | 6 | 6 |
| IFEval (100) | 71 | 83 |

Agentic results (51 DeepSWE tasks, same harness as the full model) will be added here when the run completes.

Held-out KL divergence to the full model (top-20 logits, 256 windows): agentic 0.209, code 0.193, math 0.089,
general text 0.594.

## Recommended settings

- **Q8_0, code and agentic use:** works without special settings.
- **Q4_K_M and Q3_K, code:** ban the "hmm" tokens (see below). Without it, these files can loop on harder exercises.
- **Long prose (essays, poems) and non-English prompts:** the model can fall into repetition loops; this is where
  pruning hurts most. A presence penalty of 1.0 reduces it. This model is not meant for creative writing or other
  languages.

### The "hmm" loop at Q4_K_M and Q3_K

On harder coding exercises the quantized files can get stuck in their reasoning: the model hesitates ("hmm"),
repeats it ("hmm, hmm, hmm, ...") until the token limit and never writes the code. Once one attempt has looped, the
next attempts with that reasoning in the history loop again within a few thousand characters.

Aider Polyglot, up to 10 attempts per exercise, exercises on which a file looped (runs in progress):

| File | Exercises with a loop |
|---|---|
| Q8_0 | 0 of 19 |
| Q4_K_M | 2 of 15 |
| Q3_K | 4 of 14 |

Banning the three "hmm" tokens of this tokenizer removes it. With the ban, Q3_K solved all 3 exercises it had looped
on (2 of them on the first attempt). Q4_K_M loops the same way and should get the same setting.

- **llama.cpp** (applies to every request): add
  `--logit-bias 86776-100 --logit-bias 79380-100 --logit-bias 87459-100` to `llama-server`.
- **Per request** (OpenAI-compatible API, also Maya): `"logit_bias": {"86776": -100, "79380": -100, "87459": -100}`.

Token ids: 86776 = " hmm", 79380 = "Hmm", 87459 = " Hmm". What did not fix this loop in our tests: a presence penalty
(the reasoning no longer repeats but does not end either) and a reasoning budget of 16K tokens (helped on 1 of 3
exercises).

## Run

Split files load directly; point the runtime at the first part.

```bash
pip install -U huggingface_hub
hf download lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF --include "*Q3_K*" --local-dir glm53-rco-q3k
```

- **llama.cpp:** `llama-server -m glm53-rco-q3k/GLM-5.3-Flash-Coder-RCO-152E-Q3_K-00001-of-00002.gguf --jinja -c 131072 --logit-bias 86776-100 --logit-bias 79380-100 --logit-bias 87459-100`
  with expert tensors offloaded to CPU as needed (`--n-cpu-moe`).
- **Maya** (made for GLM-5.3-Flash, hottest experts on the GPU, the rest in RAM):
  `git clone https://github.com/mw00/project-maya && cd project-maya && ./maya.sh --gguf-dir ../glm53-rco-q3k --no-vision`
  (add `--backend hip` on AMD).

## How it was made

1. Calibration mix: agentic coding sessions written by the full model, code in 21 languages, math, and 15% general
   text, all outside the evaluation sets.
2. Expert selection with RCO (optimization of the expert mask against the full model's outputs), 152 experts per
   layer. Against a REAP-style mask of the same size: same code and agentic KL, 22% lower KL on general text, and far
   fewer loops.
3. Healing: LoRA on attention and the shared expert, distilled from the full model's top-20 logits (96 steps).
4. imatrix on the same mix, then Q4_K_M and Q3_K.

## Limits

Built and tested for code and agentic use. Pruning removes knowledge outside that domain: creative writing and
non-English prompts are the weak spots. Text only: no vision projector (mmproj) is included.

## Acknowledgements

- RCO and GSQ: IST Austria (ISTA-DASLab).
- REAP (Lasby et al., 2025) for the expert-saliency score used in earlier versions.
- This project was built with the help of Claude (Anthropic) as a coding and research assistant.
  The decisions are the author's.

## License and credits

MIT, like the base model. Base model by Z.ai. Runtimes: llama.cpp, Maya.

Contact: lapadat.bogdanv@gmail.com or [LinkedIn](https://www.linkedin.com/in/bogdan-lapadat-57baa08/)
