---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
language:
- it
- en
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- gemma4
- moe
- pruned
---

# Gemma 4 26B-A4B-it, expert-pruned (88/128), GGUF

An expert-pruned version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
built to run in less memory while keeping good quality in **Italian** (and English).

Please refer to the [original model page](https://huggingface.co/google/gemma-4-26B-A4B-it) for the
model's architecture, intended use, capabilities, chat template and limitations.

## What was changed

- **Expert pruning.** Each of the 30 MoE layers originally has 128 experts. The 40 least-used experts per
  layer were removed (88 kept), ranked by their routing mass on a calibration set made mostly of Italian
  text, with some English: conversations, tool use and general prose. The router still selects 8 experts
  per token, so compute per token is unchanged; only memory goes down (about 18B total parameters, about
  4B active).
- **Source weights.** The kept experts were cut directly from the quantization-aware-trained Q4_0 weights
  of [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF).
  Nothing else was retrained.

## Files

| file | size | notes |
|---|---:|---|
| `gemma-4-26B-A4B-it-pruned88-Q4_0.gguf` | 10.2 GB | pruned only, no requantization (QAT Q4_0 weights as-is) |
| `gemma-4-26B-A4B-it-pruned88-IQ3_M.gguf` | 8.9 GB | expert weights requantized to IQ3_M with an Italian-weighted importance matrix; attention, dense layers and embeddings kept at the original Q4_0 |

## Usage

```bash
llama-server -m gemma-4-26B-A4B-it-pruned88-IQ3_M.gguf --jinja -ngl 99 -fa on -c 49152
```

On machines with little memory, `-np 1` and `--cache-type-k q8_0 --cache-type-v q8_0` reduce the
context memory further.

## Limitations

Removing experts removes whatever knowledge they held. Italian and English were the focus of the
calibration; other languages and very specialised domains may be noticeably weaker than in the original
model. All limitations of the original model apply.

## License

Apache 2.0, as the original model.
