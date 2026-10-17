---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - trohrbaugh/Qwen3.8-27B-heretic-ara
tags:
  - gguf
  - qwen3.8
  - heretic
  - abliterated
  - uncensored
  - vocabulary-pruning
  - p1m
pipeline_tag: text-generation
---

# Qwen3.8-27B RVN Heretic Abliterated Uncensored — Q6_K P1M

This is a P1M vocabulary-pruned version of `RVN-Q6_K.gguf` from
[`0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF).
The upstream RVN model is based on Qwen3.8-27B and the ARA abliteration by
[`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara).

P1M keeps ASCII, common math and typography symbols, special tokens, all 256
byte-fallback tokens, and partial UTF-8 fragments. The vocabulary was reduced
from 248,320 to 129,272 entries. The tokenizer metadata and vocabulary matrices
were updated; all other model tensors were preserved byte-for-byte. No weights
were retrained or requantized. Non-ASCII text outside the P1M keep-set remains
representable through byte fallback, with a higher token cost.

## File

| File | Quantization | Size |
|---|---|---:|
| `RVN-Q6_K-P1M.gguf` | Q6_K | 21.08 GB |

## Use with llama.cpp

```sh
llama-cli -m RVN-Q6_K-P1M.gguf -p "Write a short Python function to sort a list."
```

This inherits the upstream model's reduced safety guardrails and Apache-2.0
license. See the upstream model card and the Qwen3.8-27B and Heretic-ARA model
cards for full method details, evaluations, and attribution.
