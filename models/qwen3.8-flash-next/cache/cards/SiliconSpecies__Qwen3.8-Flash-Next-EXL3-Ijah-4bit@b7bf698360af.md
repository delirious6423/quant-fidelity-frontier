---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- exl3
- ijah
- apple-silicon
- metal
- moe
---

# Qwen3.8-Flash-Next EXL3 for Ijah (4-bit experts)

> **The preferred model is the T6 pack, [SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah).** Its KL divergence from the bf16
> model is 1.5-2x lower. This repo is the faster 4-bit-expert variant of it: ~9% faster decode, 15 GB smaller,
> 13 GB less peak memory, at a measurable cost in fidelity.

It is the T6 pack with one change: the routed experts are Sushi's 4-bit EXL3 experts instead of our 5-bit ones. The
trunk, final mixer, MTP head, embeddings and tokenizer are the T6 pack's tensors, byte for byte. For the
[Ijah](https://github.com/publicExcess/ijah) engine only (Apple Silicon, M5). Base model only: there is no Swift-1.5 4-bit pack.

## What is in it

| part | format |
|---|---|
| routed experts (48 layers x 512) | EXL3 K4, 4.03 bpw with scales, MCG codebook: [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`), encoded by Sushi; tensors re-laid out on 16-byte offsets, values unchanged |
| trunk Linears (GDN, attention + indexer, shared experts, HC down-mixes) | EXL3 K6, Ijah's T6 trunk |
| HC up-mixes, PLE key/value projections | EXL3 K8 |
| final hyper-connection mixer | bf16, from the base |
| MTP head and experts, embeddings, lm_head, routers, norms, vision tower, tokenizer | as in the T6 pack (from beamster/Qwen3.8-Flash-Next-Sushi-4bpw) |
| n-gram table (`ngram_table.bin`, 102.4 GB) | **not in this repo**: [SiliconSpecies/Qwen3.8-Flash-Next-ngram-table](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-ngram-table) |

## Against the T6 pack

| pack | KLD B | KLD C | KLD A | top-1 B | decode tok/s (DFlash2) | cold prefill 10K tok/s | weights | peak wired |
|---|---|---|---|---|---|---|---|---|
| [T6](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah) (preferred) | 0.0092 | 0.0148 | 0.0332 | 97.5% | 122.3 [113.8, 130.4] | 1,867 | 82.5 GB | 95 GB |
| **this pack (4-bit experts)** | 0.0185 | 0.0236 | 0.0513 | 96.6% | 133.0 [123.1, 142.2] | 2,210 | 67.4 GB | 82 GB |

KLD: whole model vs the bf16 teacher, full vocabulary, teacher-forced (lower is closer). B: 16 reasoning traces
(2,048 tokens); C: 4 windows of 32,768 tokens (code, prose); A: wikitext-2, 16 x 512. Speed: one Apple M5 Max (40-core
GPU), 128 GB; one streaming OpenAI-compatible client, 6 prompts x 3 seeds, 1,024 tokens each, T 1.0 / top-p 0.95 /
top-k 20, medium thinking, DFlash2 drafter on for both rows; mean [95% CI]. Prefill: cold 10K prompt, median.

## Requirements

An Apple **M5**-family Mac with ~85 GB of free unified memory (a 128 GB Mac); disk: 67.4 GB for this repo +
102.4 GB for the n-gram table.

## Download and run

```sh
hf download SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah-4bit --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah-4bit
hf download SiliconSpecies/Qwen3.8-Flash-Next-ngram-table ngram_table.bin --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah-4bit   # 102.4 GB; or hard-link another Ijah pack's copy
git clone https://github.com/publicExcess/ijah && cd ijah && make
tools/serve.sh start -- --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah-4bit --model-id Qwen3.8-Flash-Next-native-4bit
# or, with the DFlash2 drafter the decode numbers used (optional; see the Ijah README):
hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir ~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2
tools/serve.sh start -- --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah-4bit --model-id Qwen3.8-Flash-Next-native-4bit --mtp 5 --drafter dflash2 \
  --drafter-path ~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2
```

## Licences and credits

- **Model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`de4b8e4`), Qwen Community License 1.0 (`LICENSE`).
- **Routed-expert quantization:** [Sushi](https://github.com/beamivalice/sushi) by Theinruj Toranavikrai and David
  Dalcu (MIT), as published in [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`), with thanks.
  Its notice is below and in `LICENSE-SUSHI-MIT`.
- **Format:** EXL3 trellis, [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3) (MIT). **Engine:**
  [Ijah](https://github.com/publicExcess/ijah). Trunk, mixer and packing: SiliconSpecies (as in the T6 pack).
- Modified files: the `model-exl3-L*` expert shards (re-laid out), all `model-t6-*` files (our trunk), `config.json`,
  `model.safetensors.index.json`.

### Sushi licence notice

```text
MIT License

Copyright (c) 2026 Theinruj Toranavikrai
Copyright (c) 2026 David Dalcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
