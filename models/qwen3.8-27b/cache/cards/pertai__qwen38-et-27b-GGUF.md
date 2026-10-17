---
language: [et]
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [estonian, gguf, ollama]
---

# qwen38-et-27b GGUF — merged Estonian model, Q6_K

Fully merged (base + CPT + skills), single-file GGUF, Q6_K (21 GB).

**Q6 on purpose:** in our test Q4_K_M brought back orthographic errors that
training had fixed, while Q6 did not. We did not repeat this across several
quantizers or seeds, so we cannot say for certain that quantization itself was
the cause rather than something else in the conversion chain. The practical
advice stands either way: after continued pretraining, check the packaging step
separately.

> ⚠️ **About the 86.1% figure.** The 200-task Estonian set behind it was
> consulted after every training round and used to choose the next training
> batch, which makes it a **development set, not a held-out test**. The number
> is optimistically biased by an unknown amount and is **not** comparable to
> scores other models report elsewhere. No independent blind evaluation has been
> done. Details and three measurement corrections:
> [PARANDUSED.md](https://github.com/pertlomp/qwen38-et/blob/main/PARANDUSED.md).

**Measured results:** 86.1% on the 200-task Estonian development set (151
auto-scored); held-out fiction perplexity −31% vs base (22.2 → 15.4) after 110M
tokens of continued pretraining; HumanEval 85.4%; EstQA reading F1 93.6.
The perplexity figure is the most defensible of these, because nothing was
tuned against it.

Ollama:
```
ollama create eesti -f Modelfile   # Modelfile in this repo
ollama run eesti
```
Use think=false. Method & eval: github.com/pertlomp/qwen38-et

Eesti keeles: liidetud täismudel, üks fail, Q6_K. Kasuta think=false.
**NB:** 86,1% on arendusmõõt, mitte sõltumatu testitulemus; vt PARANDUSED.md.
