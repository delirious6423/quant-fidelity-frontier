---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- fr
datasets:
- HumanjavaEnterprises/bilingual-canadian-eval
library_name: gguf
tags:
- gguf
- llama.cpp
- qwen3.8
- bilingual
- french
- canada
---
# Qwen3.8-27B Bilingual Canadian

Qwen3.8-27B for English and French, in files that run in unmodified stock llama.cpp on one GPU. v2 is calibrated on
English, French and Québec French text and checked against the full-size original on both languages at every step.

The 4-bit file (15.7 GiB) scores within about a point of the original on Québec French benchmarks run with the official
COLE protocol (QFrCoLA 0.803 vs 0.802, QFrBLiMP 0.895 vs 0.895, QFrCoRE 0.894 vs 0.895, QFrCoRT 0.901 vs 0.912), on
maths (GSM8K 0.966 vs 0.964) and on instruction following in both languages (IFEval EN 0.813 vs 0.821, FR 0.754 vs
0.738). Its next-token predictions drift from the original by the same small amount in English, French and Québec
French (KL divergence 0.0093 / 0.0097 / 0.0093). Long French answers keep their accents and stay in French, with up to
24k tokens of French context, or 8k of English, in front of them. Text in, text out; the 4-bit file uses about 16.4 GB of GPU memory at an 8k context.

*Unofficial build of Qwen3.8-27B by Humanjava Enterprises; not made or endorsed by the Qwen team.*

| | |
|---|---|
| **Guide** (setup, full results, issues) | [github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian) |
| **Ollama** | `ollama run vveerrgg/qwen3.8-27b-bilingual-canadian` |
| **Evaluation set** (per-file scores and task answers behind the numbers) | [HumanjavaEnterprises/bilingual-canadian-eval](https://huggingface.co/datasets/HumanjavaEnterprises/bilingual-canadian-eval) (config `v2`) |
| **Original model** (full precision, with vision) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Results** | [EVALS.md](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/blob/main/EVALS.md) |

## Files

| File | Size | Use |
|---|---|---|
| `qwen3-8-27b-bilingual-canadian-v2-Q4_K_M.gguf` (**default**) | 15.69 GiB | one GPU with 24 GB (measured: 16.4 GB in Ollama at an 8k context), with room for a longer context |
| `qwen3-8-27b-bilingual-canadian-v2-Q3_K_M.gguf` | 12.61 GiB | one GPU with 16 GB (measured: 13.6 GB in Ollama at an 8k context); close to the 4-bit file on every task, a little further from the original on text (KL 0.036-0.039) |
| `qwen3-8-27b-bilingual-canadian-v1-Q2_K_mix.gguf` (v1) | 9.15 GiB | the v1 file, kept for 12 GB GPUs; see the v1 notes in EVALS.md |

Standard llama.cpp K-quant mixes made with an importance matrix from English, French and Québec French text. Every
byte is a stock GGUF type, so any llama.cpp build that runs Qwen3.8 runs these files; no fork or patch needed. Verify
the download with `sha256sum -c SHA256SUMS`.

We tested a 2-bit file of v2 too and did not ship it: it starts repeating itself in long answers (2.3% repeated
phrases vs 0.3% for the original). The v1 file above is the 2-bit option, tuned to avoid that.

## Quick start

```bash
ollama run vveerrgg/qwen3.8-27b-bilingual-canadian
# or
ollama run hf.co/HumanjavaEnterprises/Qwen3.8-27B-Bilingual-Canadian-GGUF:Q4_K_M
# or llama.cpp
llama-server -hf HumanjavaEnterprises/Qwen3.8-27B-Bilingual-Canadian-GGUF:Q4_K_M -ngl 99 -c 8192 -fa on --reasoning-budget 0
```

Ollama needs a recent release (tested on 0.40.16); older releases cannot load the Qwen3.8 architecture.

This repo also carries Ollama `template`, `system` and `params` files so the `hf.co/` path gets the chat format
(thinking off) and stop tokens.

## Versions

| Version | Date | Change |
|---|---|---|
| **v2** (current) | 2026-11-25 | 4-bit and 3-bit files calibrated on English, French and Québec French; within about a point of the original on COLE (official protocol), GSM8K and IFEval EN/FR. |
| v1 | 2026-11-17 | First public release: a 9.15 GiB 2-bit-class file, calibrated and trained on English and French. |

Each version is a git tag (`v1`, `v2`) on the Hugging Face repo and the guide repo, a pinned Ollama tag, and a config in the
evaluation set. Published numbers are never changed silently: a change gets a date and a one-line note.

## Prompt format

Qwen3 chat format (ChatML), **thinking off**: the assistant turn starts with an empty think block. All numbers here
were measured that way, at temperature 0. Stop on `<|im_end|>` (also `<|im_start|>`, `<|endoftext|>`).

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

No special system prompt is needed. Ask in English or French; it answers in the language you use.

## Results

Full tables: [EVALS.md](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/blob/main/EVALS.md).

| | Original BF16 | v2 4-bit | v2 3-bit | v1 (2-bit) |
|---|---|---|---|---|
| COLE QFrCoLA, test (official protocol) | 0.802 | 0.803 | 0.803 | 0.780 |
| COLE QFrBLiMP, dev (official protocol) | 0.895 | 0.895 | 0.879 | 0.855 |
| COLE QFrCoRE, test (official protocol) | 0.895 | 0.894 | 0.885 | 0.864 |
| COLE QFrCoRT, test (official protocol) | 0.912 | 0.901 | 0.895 | 0.865 |
| GSM8K (5-shot, flexible) | 0.964 | 0.967 | 0.964 | 0.954 |
| MGSM EN / FR | 0.956 / 0.884 | 0.960 / 0.872 | 0.960 / 0.876 | 0.964 / 0.892 |
| IFEval EN / FR (prompt, strict) | 0.821 / 0.738 | 0.813 / 0.754 | 0.808 / 0.736 | 0.806 / 0.726 |
| KL divergence vs original, EN / FR / QC (lower is closer) | 0 | 0.0093 / 0.0097 / 0.0093 | 0.036 / 0.039 / 0.038 | 0.091 / 0.093 / 0.091 |

COLE rows use the COLE authors' own prompts and answer format (the tool-call path), on the splits shown. MGSM, GSM8K
and IFEval use lm-evaluation-harness through llama-server. The v1 KL figures were measured on the v1 text sets.

## How we measured (and why your numbers may differ)

- Stock `llama-server`, one GPU, the model's own chat template (`--jinja`), thinking off, temperature 0, seed 0.
- COLE: the official COLE prompts and answer parsing, tool-call mode, one request at a time.
- Other tasks: lm-evaluation-harness, chat-completions endpoint, `--apply_chat_template`.
- **Harness settings move scores**, sometimes by many points: the chat template, stop tokens, system prompt, thinking
  mode and sampling all matter. For example, a GGUF whose end-of-turn token is not marked as a stop token prints that
  token at the end of every answer, and instruction-following checks then fail. Use the bundled template and the
  settings above to reproduce our numbers, and compare models under the same harness.

## How it was built

- **Base:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), text weights only.
- **Calibration:** an importance matrix from English, French and Québec French text, including text written by the
  original model. No fine-tuning in v2: the weights are the original's, quantized. Open models only.
- **Measured against the original:** each file was scored on held-out English, French and Québec French text and on
  the task evals above before release.
- **Export:** stock llama.cpp `llama-quantize`, standard K-quant mixes. The files load in an unmodified llama.cpp.

## Related work

- [COLE](https://github.com/GRAAL-Research/COLE) (Université Laval, GRAIL lab; [paper](https://arxiv.org/abs/2510.05046)):
  the French benchmark, with Québec-specific tests, used for the COLE numbers here.
  [QFrCoLA](https://arxiv.org/abs/2508.16867) and [QFrBLiMP](https://arxiv.org/abs/2509.25664) are its Québec grammar
  tests.
- [QuebecLLM](https://huggingface.co/QuebecLLM) (Concordia CLaC lab and Université Laval, LREC 2026;
  [paper](https://arxiv.org/abs/2510.22747)): the first open Québec French language models, adapters for base models
  continued on Québec French text and measured on COLE.
- [Chocolatine-2-14B-Instruct](https://huggingface.co/jpacifico/Chocolatine-2-14B-Instruct-v2.0.3): an openly licensed
  English and French instruct model with GGUF files, scored by the COLE authors at full precision.

What this build adds: its Québec French scores were measured on the exact file you download, with the COLE authors'
own protocol, and every answer is published in the evaluation set so anyone can check them.

## Build on it

This is a foundation, not a finished product. If you are building for French-speaking Canadians (a Québec clinic, a
bilingual help desk, a francophone community outside Québec), you can start here instead of from a model that treats
French as an afterthought: tune it on your own domain, test it in both languages, and run it on hardware you own.
If you learn something along the way (Québec text you have the rights to share, an evaluation in your field, a fix),
the [GitHub repository](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian) takes it back through
issues and pull requests, so the next version is better for everyone who builds on it.

## Limitations

| Limitation | What to do |
|---|---|
| Québec civic and legal detail is unreliable: on 16 questions about Bill 96 it gets 7 right. It confuses specific rules (for example, when the government may use a language other than French with new arrivals, or what changed for English-language CEGEPs). Civic basics we checked (official languages, the bilingual province) are correct. | Check civic and legal facts against official sources, and tell us when it slips: [Model got something wrong](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/issues/new?template=wrong-answer.yml). |
| The 4-bit file needs a 24 GB GPU; the 3-bit file a 16 GB one. | On a 12 GB GPU use the v1 file (9.15 GiB). |
| Text only: the original's image and video input is not included. | Use the original model for vision. |
| Quantization still costs a little: QFrCoRT is 0.011 below the original on the 4-bit file, and the 3-bit file trails the 4-bit by up to 0.016 on QFrBLiMP. | Use the 4-bit file, or the original, where those points matter. |

## Feedback and security

Wrong answers, bugs and docs: [issues on the guide](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/issues/new/choose). Security problems, privately:
[GitHub private reporting](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/security/advisories/new) or security@humanjava.com. See
[SECURITY.md](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/blob/main/SECURITY.md).

## Licence and credits

Model weights: Apache-2.0, same as the base model ([LICENSE](LICENSE); changes stated in [NOTICE](NOTICE)). Docs in the guide repo: Apache-2.0.

Base model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team (Apache-2.0). Runtime used for
every number here: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Comparison file:
[Unsloth](https://huggingface.co/unsloth) UD-Q2_K_XL. French held-out text: FineWeb-2 fra_Latn test split (ODC-By).
Benchmarks: Belebele, IFEval, French IFEval, GSM8K, MGSM, COLE, run with lm-evaluation-harness.

Built by [Humanjava Enterprises](https://humanjava.com).
