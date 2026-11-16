---
license: apache-2.0
base_model:
- unsloth/Qwen3.8-27B-GGUF
- Qwen/Qwen3.8-27B
language:
- en
tags:
- gguf
- qwen3.5
- llama.cpp
- quantization
- vocabulary-pruned
- tokenizer
- unsloth
- amd
- vulkan
- rdna1
- local-llm
---

<p align="center">
  <img src="monoglot-logo.png" alt="Monoglot" width="300">
</p>

# Monoglot 27B, second cut: Unsloth's weights with the vocabulary taken off

**One language, on purpose.** This replaces the first Monoglot 27B, and it replaces it because I measured
my own work and it lost. The weights here are **Unsloth's UD-IQ2_S**, not mine. The surgery is mine. Their
stock 2-bit measures better than the 3-bit quant I built by hand, on my own gate, at a smaller file size.

| File | Bytes | PPL, wikitext-2, 8 x 512 |
|---|---:|---:|
| **This file's source, unsloth/Qwen3.8-27B UD-IQ2_S** | **8,422,719,488** | **7.5478** |
| My Q3-DOWN-XS (the first Monoglot 27B) | 8,480,272,384 | 8.2587 |

Same binary, same frozen gate, same chunks. Their 2-bit wins by 8.7% perplexity while being 68 MB smaller,
and it uses no custom recipe at all.

## Credit

- **The weights: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).** UD-IQ2_S from their Dynamic
  v3.1 line. The quantisation is entirely theirs; I did not touch a weight, retrain anything or re-quantise.
- **The idea and the first public build: [fakezeta](https://huggingface.co/fakezeta)**, whose
  [`Condensed-P2`](https://huggingface.co/fakezeta/Qwen3.8-27B-UD-Q3_K_XL-Condensed-P2) release is what
  showed a vocabulary prune was worth doing.
- **The tool: [`bsaleh03/ASCII-Condensed-prune-tools`](https://github.com/bsaleh03/ASCII-Condensed-prune-tools).**
  Policies, keep-set construction and the in-quantised-space row gather are his work. That repository has
  **no licence**, so none of its code is redistributed here.
- **[@TeksEdge](https://x.com/teksedge/status/2122118832441708600)** for the post that surfaced it.
- **Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, Apache-2.0.

What is mine is the prune, the measurements and the correction below. The first entry in this series was my
own quant, and this one exists because the numbers said theirs is better.

## What was cut

Policy **P1G**: ASCII, plus math and typography symbols, Greek, box-drawing and status glyphs, units and
currency. Specials, all 256 byte-fallback tokens and partial-UTF-8 fragments always survive.

| Tensor | Type | Before | After |
|---|---|---:|---:|
| `output.weight` | Q4_K | 715 MB | 374 MB |
| `token_embd.weight` | Q2_K | 417 MB | 218 MB |
| **File** | UD-IQ2_S | **8,422,719,488 B** | **7,876,946,592 B** |

545 MB off, 6.5%. The two vocabulary tensors were 13.5% of the source file.

## Measured

`llama.cpp` `3ba1ce0` Vulkan, RX 5700 XT (8 GB, RDNA1), `-ngl 999 -fa on -t 6 --cpu-mask 0x3f`,
`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, three reps, card empty before each run.

| llama-bench | result |
|---|---:|
| pp512 | 170.3 ± 1.56 t/s |
| tg128 | 23.57 ± 0.03 t/s |

At the 16K production window it depends on the KV type, and this is the part worth reading:

| ctx 16384 | prefill | decode | VRAM | GTT |
|---|---:|---:|---:|---:|
| KV q8_0 | **28.9 t/s** | **10.8 t/s** | 7.91 GiB | 0.65 GiB |
| **KV q4_0** | 153.2 t/s | **21.9 t/s** | 7.83 GiB | 0.48 GiB |

At q8_0 this file misses the fast path by about 7 MB and falls off a cliff. With `-ctk q4_0 -ctv q4_0` it
clears it. On this hybrid model the KV cost scales with the attention layers only, and 8-chunk gate
measurements put q4_0 KV at PPL +0.58%, which is inside the chunk noise. **Use q4_0 KV.** Running it at q8_0
at 16K will look broken rather than slow.

```bash
llama-server -m Qwen3.8-27B-UD-IQ2_S-Monoglot-P1G-130K.gguf \
  -c 16384 -ngl 999 -ctk q4_0 -ctv q4_0 -fa on
```

## What it costs

| Text | Source | This file | Ratio |
|---|---:|---:|---:|
| English prose | 30 | 30 | 1.00x |
| Python | 41 | 41 | 1.00x |
| Math with `∀ ∑ π ≥ ✓` | 27 | 27 | 1.00x |
| French, accented | 18 | 31 | 1.72x |
| Japanese | 15 | 40 | 2.67x |
| Chinese | 15 | 50 | 3.33x |
| Russian | 17 | 93 | 5.47x |

English, code and math tokenise one for one. Anything outside the kept set falls back to bytes. If you need
multilingual output, use UD-IQ2_S unpruned.

## Correctness checks that did pass

- `verify_prune.py --policy P1G`: all 849 non-vocabulary tensors byte-identical, gathered vocab rows match
  their source rows, keep-set exactly the policy, every metadata key preserved.
- Greedy 64-token continuations byte-identical to the source for an English prompt and a code prompt.
- A `tools` request on both files returns `finish_reason: tool_calls`, `read_file`, `{"path":"/etc/hostname"}`.

Identical greedy text is the reason this file can claim the source's 7.5478: it requires both the prompt
tokeniser and the output-head rows to stay aligned with their token strings.

## The correction this file is, in plain terms

The first Monoglot 27B was my own hand-built Q3-DOWN-XS, and its card quotes "86.5% of control" on
perplexity. That number was true against the control the ladder was built on, which was **Unsloth UD-IQ3_XXS
at 11,913,559,104 bytes**. Unsloth re-cut every quant for Dynamic v3.1 on 2026-11-11 and that exact artifact
no longer exists: their UD-IQ3_XXS now serves at 10.93 GB. So the ladder was ranking candidates against a
reference that had been replaced, and nothing in my own numbers changed to reveal it.

Against a current v3.1 file of the same nominal size, my 3-bit is behind their 2-bit. That is the finding,
and it is the reason this repository exists rather than a quieter edit to the old one.

## What I did not measure

- **No task benchmark on this pair.** The perplexity gate is the comparison here. An earlier task harness
  scored my Q3 at 21/26 and their old IQ2_S at 20/26, and that one-point gap was inside that harness's noise,
  which is exactly why the sharper instrument decided this.
- **No perplexity on the pruned file.** PPL is not comparable across a vocabulary change: the same text
  tokenises differently and the logit spaces differ in size. The claim is that the prune is transparent, and
  the greedy-identity and tensor-byte checks are what support it.
- **One machine, one build, one card.** RDNA1, Vulkan, ctx 16384, KV q4_0.
- **Text only.** No `mmproj` here; the prune is text-vocabulary only.

## Files and verification

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-UD-IQ2_S-Monoglot-P1G-130K.gguf` | 7,876,946,592 | `1ce313ce26c7189bca02eff889f4feb17a53d1d4a890a4b744322342fa138d09` |

## Licence

Apache-2.0, following Qwen3.8-27B. Weights derive from Unsloth's UD-IQ2_S quant of Qwen3.8-27B. The pruning
tool is linked, not included, because it carries no licence.
