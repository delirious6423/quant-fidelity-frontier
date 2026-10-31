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

> **Update (October 29, 2026): Q3_K_M and Q4_K_M republished with an anti-loop fine-tune merged in.** A small
> LoRA trained with verifiable-reward RL (GRPO with distillation from the full model) on multi-language coding tasks
> removes the reasoning loops of the quantized files. Measured on Q3_K_M with the fine-tune applied as an adapter (40 held-out coding tasks x 2, and the Aider
> "hmm" test, without any logit bias). The republished files contain the same fine-tune merged into the weights:
>
> | Q3_K_M | Before | After |
> |---|---|---|
> | Held-out coding tasks: looping runs | 11 of 80 | **0 of 80** |
> | Held-out coding tasks: solved | 46 of 80 | **51 of 80** |
> | Aider "hmm" test: exercises with a loop | 4 of 14 | **0 of 14** (14 of 14 solved) |
>
> The `--logit-bias` "hmm" ban below is no longer needed for these files. Q8_0 is unchanged: it does not show the
> loop. The previous Q3_K_M / Q4_K_M files remain available in the repository history.

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
| 17 x 23, 1234 x 5678, Fibonacci, small executed code task, tool call | arithmetic errors | **pass** at every quant (Q3_K_M missed Fibonacci once in one run) |
| Experts per layer | 144 | 152 (incl. 8 generalists) |
| Calibration | Python agentic sessions | agentic sessions + code in 21 languages + math + 15% general text |

## Files

| Quant | Size | Fits (GPU VRAM + RAM) | KLD to Q8 (October 22 files) |
|---|---|---|---|
| Q8_0 | 180.4 GB (4 parts) | 24 + 192, multi-GPU | 0 |
| Q4_K_M | 102.6 GB (3 parts) | 24 + 96, 32 + 96, 128 GB unified | 0.024 |
| Q3_K_M | 75.1 GB (2 parts) | 24 + 64, 32 + 64 | 0.045 |
| IQ2_XXS (GSQ) | 47.4 GB (1 file) | 24 + 32 | 0.121 |
| IQ2_XXS-downIQ2_S (GSQ) | 50.8 GB (2 parts) | 24 + 32 | 0.115 |

KLD: llama.cpp `llama-perplexity --kl-divergence` against the Q8_0 of this model on held-out code and agentic text.
Same top token as Q8: Q4_K_M 94.5%, Q3_K_M 92.5%, IQ2_XXS-downIQ2_S 88.3%, IQ2_XXS 88.1%.

**Re-uploaded on 25 Oct 2026.** The files uploaded on 22 Oct were truncated during our release step (the second half of the tensors was zeros), so they produced garbage output. If you downloaded before 25 Oct, please download again and check the SHA256 below. Thanks to Martin (discussion #2) for the report.

SHA256 (`sha256sum -c` format):

```
5fac2bbfa70dc9227ab5c6878b752f41c0a30873c22276f27cbd1fd2b9001a3f  GLM-5.3-Flash-Coder-RCO-152E-Q3_K_M-00001-of-00002.gguf
23366624c40943a7a837c24a4efceb18c804aaba3652665df1f564b46eb1e88f  GLM-5.3-Flash-Coder-RCO-152E-Q3_K_M-00002-of-00002.gguf
58322a98c38933d55489b43224dbcff55550939b74f2d84e8e0e8cd124bcb5c0  GLM-5.3-Flash-Coder-RCO-152E-Q4_K_M-00001-of-00003.gguf
459ed452103fe06c4f3ce7725b8c7db228ad766f9b01fc10f72674b45988194b  GLM-5.3-Flash-Coder-RCO-152E-Q4_K_M-00002-of-00003.gguf
6b085048bfcd568e1ed934cc75411b218d97d97c1102acb280ea1e8c35dba8ca  GLM-5.3-Flash-Coder-RCO-152E-Q4_K_M-00003-of-00003.gguf
16b0410cd2af42264813ac209687cafb36d2ab14cc3f74c14150ce0b1ff89a82  GLM-5.3-Flash-Coder-RCO-152E-Q8_0-00001-of-00004.gguf
e5a815e3bbc62c347d452dedd88db0e48204f078ea11814158e59ac9a11f3756  GLM-5.3-Flash-Coder-RCO-152E-Q8_0-00002-of-00004.gguf
6a77d3328a9eab2c7944d4be5a66952f856ed3d9706be9d76996ce3ac675e992  GLM-5.3-Flash-Coder-RCO-152E-Q8_0-00003-of-00004.gguf
1bd0db0cf92039950cafbead6f7243bf54b6aee5439f704d58c4aac0481caba7  GLM-5.3-Flash-Coder-RCO-152E-Q8_0-00004-of-00004.gguf
4fa2efdfde5565ca135753c4b34838e85b032e24af17ba12ccb8ba3dc4fce3f7  GLM-5.3-Flash-Coder-RCO-152E-IQ2_XXS.gguf
5d920a285e9ef7f8ac76bf814de29da2ad2568c3bb7341f0e820938b5fc593a2  GLM-5.3-Flash-Coder-RCO-152E-IQ2_XXS-downIQ2_S-00001-of-00002.gguf
44c59644d9e37b503e19bde769e96f55f8c35589b300ac63809a784fbeb48a29  GLM-5.3-Flash-Coder-RCO-152E-IQ2_XXS-downIQ2_S-00002-of-00002.gguf
```

The two 2-bit files (24 GB VRAM + 32 GB RAM) are described in the next section.

## 2-bit files: IQ2_XXS and IQ2_XXS-downIQ2_S (October 31, 2026)

Made for **24 GB VRAM + 32 GB RAM**. The routed experts are 2-bit (IQ2_XXS, improved with GSQ vector quantization);
attention, the shared expert and the dense layers keep the Q4_K_M types (Q4_K, Q6_K, Q8_0). Both files contain the
anti-loop fine-tune merged into the weights, like the republished Q3_K_M and Q4_K_M: only the attention and
shared-expert tensors changed, the 2-bit experts are the same as before.

- `IQ2_XXS`: every routed expert in IQ2_XXS. 47.4 GB, one file.
- `IQ2_XXS-downIQ2_S`: the expert down projections in IQ2_S instead. 50.8 GB, 2 parts. Lower KLD; take it if it
  fits.

Measured with the fine-tune applied as an adapter (40 held-out coding tasks x 2, no logit bias):

| | IQ2_XXS before | IQ2_XXS after | IQ2_XXS-downIQ2_S before | IQ2_XXS-downIQ2_S after |
|---|---|---|---|---|
| Held-out coding tasks: looping runs | 19 of 79 | **2 of 78** | 25 of 80 | **1 of 80** |
| Held-out coding tasks: solved | 44 of 79 | 43 of 78 | 37 of 80 | **47 of 80** |

Aider "hmm" test (8 exercises, up to 10 attempts, no logit bias): before the fine-tune the IQ2_XXS file solved 3 of 8
and looped on 6 of 8. After: IQ2_XXS solved 5 of 8 (one exercise still looped and did not finish in the time limit),
IQ2_XXS-downIQ2_S solved 6 of 8 without a loop.

The uploaded files were then tested exactly as published (no adapter, same 40 tasks, first sample of each task with the
same seed as above):

| First sample per task | Before | Fine-tune as adapter | Published file (merged) |
|---|---|---|---|
| IQ2_XXS: looping runs | 12 of 40 | 0 of 38 | **1 of 39** |
| IQ2_XXS: solved | 21 of 40 | 20 of 38 | **22 of 39** |
| IQ2_XXS-downIQ2_S: looping runs | 13 of 40 | 1 of 40 | **1 of 38** |
| IQ2_XXS-downIQ2_S: solved | 19 of 40 | 24 of 40 | **21 of 38** |

Runs still in progress when a test was stopped are not counted (none of them had been flagged as a loop). Both files
answer 17 x 23 = 391. The "hmm" ban is not needed for these files.

```bash
hf download lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF --include "*IQ2_XXS-downIQ2_S*" --local-dir glm53-rco-iq2
llama-server -m glm53-rco-iq2/GLM-5.3-Flash-Coder-RCO-152E-IQ2_XXS-downIQ2_S-00001-of-00002.gguf --jinja -c 131072 --n-cpu-moe N
```

Set `--n-cpu-moe N` (number of layers whose experts stay in RAM) to the lowest value that fits your VRAM. For the single-file IQ2_XXS use
`--include "*IQ2_XXS.gguf"`.

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
- **Q4_K_M and Q3_K_M, code:** ban the "hmm" tokens (see below). Without it, these files can loop on harder exercises.
- **Long prose (essays, poems) and non-English prompts:** the model can fall into repetition loops; this is where
  pruning hurts most. A presence penalty of 1.0 reduces it. This model is not meant for creative writing or other
  languages.

### The "hmm" loop at Q4_K_M and Q3_K_M (files before October 29)

This section describes the files published on October 22. The republished Q3_K_M and Q4_K_M no longer need the ban.


On harder coding exercises the quantized files can get stuck in their reasoning: the model hesitates ("hmm"),
repeats it ("hmm, hmm, hmm, ...") until the token limit and never writes the code. Once one attempt has looped, the
next attempts with that reasoning in the history loop again within a few thousand characters.

Aider Polyglot, up to 10 attempts per exercise, exercises on which a file looped (runs in progress):

| File | Exercises with a loop |
|---|---|
| Q8_0 | 0 of 19 |
| Q4_K_M | 2 of 15 |
| Q3_K_M | 4 of 14 |

Banning the three "hmm" tokens of this tokenizer removes it. With the ban, Q3_K_M solved all 3 exercises it had looped
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
hf download lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E-GGUF --include "*Q3_K_M*" --local-dir glm53-rco-q3k
```

- **llama.cpp:** `llama-server -m glm53-rco-q3k/GLM-5.3-Flash-Coder-RCO-152E-Q3_K_M-00001-of-00002.gguf --jinja -c 131072 --logit-bias 86776-100 --logit-bias 79380-100 --logit-bias 87459-100`
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
4. imatrix on the same mix, then Q4_K_M and Q3_K_M.

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
