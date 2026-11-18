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

A 9.15 GiB build of Qwen3.8-27B that works in English and French, made to run in unmodified stock llama.cpp on one
GPU. It was calibrated and trained on English and French and measured against the original at every step.

On French web text, and on a separate Québec French judge, it stays as close to the original model as Unsloth's
UD-Q2_K_XL file of the same size, a hair closer (KL divergence 0.0928 vs 0.0935 French, 0.0908 vs 0.0913 Québec; lower
is closer); in English it drifts more (0.0908 vs 0.0790). On Québec French grammar judgements (COLE QFrCoLA) it scores
0.762 (Unsloth 0.756, original 0.800); on GSM8K maths 0.954 (Unsloth 0.953). Text in, text out; uses about 10.5 GB of
GPU memory at an 8k context.

*Unofficial build of Qwen3.8-27B by Humanjava Enterprises; not made or endorsed by the Qwen team.*

| | |
|---|---|
| **Guide** (setup, full results, issues) | [github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian) |
| **Ollama** | `ollama run vveerrgg/qwen3.8-27b-bilingual-canadian` |
| **Evaluation set** (per-file scores and task answers behind the numbers) | [HumanjavaEnterprises/bilingual-canadian-eval](https://huggingface.co/datasets/HumanjavaEnterprises/bilingual-canadian-eval) (config `v1`) |
| **Original model** (full precision, with vision) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Results** | [EVALS.md](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/blob/main/EVALS.md) |

## Files

| File | Size | Use |
|---|---|---|
| `qwen3-8-27b-bilingual-canadian-v1-Q2_K_mix.gguf` (**default**) | 9.15 GiB (9370 MiB) | one GPU with 12 GB or more (measured: 10.5 GB in Ollama at an 8k context); 16 GB leaves room for a long context |

A mix of llama.cpp's standard 2- to 4-bit block types, chosen tensor by tensor. Every byte is a stock GGUF type, so
any llama.cpp build that runs Qwen3.8 runs this file; no fork or patch needed. Verify the download with
`sha256sum -c SHA256SUMS` from the Hugging Face repo.

## Quick start

```bash
ollama run vveerrgg/qwen3.8-27b-bilingual-canadian
# or
ollama run hf.co/HumanjavaEnterprises/Qwen3.8-27B-Bilingual-Canadian-GGUF:Q2_K_mix
# or llama.cpp
llama-server -hf HumanjavaEnterprises/Qwen3.8-27B-Bilingual-Canadian-GGUF:Q2_K_mix -ngl 99 -c 8192 -fa on --reasoning-budget 0
```

Ollama needs a recent release (tested on 0.40.16); older releases cannot load the Qwen3.8 architecture.

This repo also carries Ollama `template`, `system` and `params` files so the `hf.co/` path gets the chat format
(thinking off) and stop tokens.

## Versions

| Version | Date | Change |
|---|---|---|
| **v1** (current) | 2026-11-17| First public release. |

Each version is a git tag (`v1`, ...) on the Hugging Face repo and the guide repo, a pinned Ollama tag, and a config in the
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

Full tables: [EVALS.md](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/blob/main/EVALS.md). Columns: original BF16 / this build / Unsloth UD-Q2_K_XL (9.15 GiB).

| | Original | This build | Unsloth |
|---|---|---|---|
| French text, KL divergence vs original (lower is closer) | 0 | 0.0928| 0.0935|
| Québec French text, KL divergence vs original | 0 | 0.0908| 0.0913|
| English text, KL divergence vs original | 0 | 0.0908| 0.0790|
| Belebele FR / EN | 0.949/ 0.962| 0.947/ 0.961| 0.936/ 0.959|
| IFEval FR / EN (prompt, strict) | 0.738/ 0.821| 0.726/ 0.806| 0.732/ 0.830|
| MGSM FR / EN | 0.884/ 0.956| 0.892/ 0.964| 0.888/ 0.956|
| GSM8K (5-shot) | 0.964| 0.954| 0.953|
| COLE QFrCoLA / QFrBLiMP (Québec French grammar) | 0.800/ 0.895| 0.762/ 0.839| 0.756/ 0.879|

## How it was built

- **Base:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), text weights only.
- **Calibration and training:** English and French text, including text written by the original model; no new
  licensed datasets. Open models only.
- **Measured against the original:** every candidate file was scored on held-out English and French text and kept
  only if it moved closer to the BF16 original.
- **Export:** bit-exact to stock llama.cpp GGUF types. The file loads in an unmodified llama.cpp.

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

What this build adds: its Québec French scores were measured on the exact file you download, and every answer is
published in the evaluation set so anyone can check them.

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
| English drifts further from the original than Unsloth's same-size file (0.0908 vs 0.0790). French and Québec French are level, a hair closer. | For English-only work, Unsloth's UD-Q2_K_XL or a larger quant of the original is the better pick. |
| French extractive question answering (copying a span from a passage: FQuAD, PIAF) is weaker than Unsloth's (.557 vs .618 F1). | Ask for a short answer in your own words, or check quoted spans against the source. Use a larger quant where exact extraction matters. |
| It sometimes ends its turn without answering: 2 of 1,319 GSM8K items come back empty (earlier builds: up to 1 in 11). | Retry, or put few-shot examples in one user message; single questions are rarely affected. |
| It can describe Québec's Bill 96 inaccurately: in testing it called the bill "the Charter of the French Language" (it amends that Charter) and, in French, used loaded wording about immigrants. Civic basics we checked (official languages, the bilingual province) are correct. | Check civic and legal facts against official sources, and tell us when it slips: [Model got something wrong](https://github.com/HumanjavaEnterprises/qwen3.8-27b-bilingual-canadian/issues/new?template=wrong-answer.yml). |
| Text only: the original's image and video input is not included. | Use the original model for vision. |
| A 2-bit-class file loses some quality against the original on every task (see Results). | Use a larger quant of the original if your GPU has room. |

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
