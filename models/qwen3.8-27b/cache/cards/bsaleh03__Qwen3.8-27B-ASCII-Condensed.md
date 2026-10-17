---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - qwen3
  - long-context
  - vocabulary-pruning
  - llama.cpp
---

# Qwen3.8-27B-ASCII-Condensed

**Qwen3.8-27B with an ASCII-only vocabulary, for long context on a single 16 GB GPU.**

Up to **135,168 tokens** of context on an RTX 5070 Ti (16 GB).

The same quantization unpruned reaches 114,688 on the same card, so the
vocabulary prune is worth about **+20,000 tokens (1.18x)**.

The vocabulary is reduced from 248,320 to 129,006 rows. **No weights were
retrained, requantized, or otherwise altered** — 864 of 866 tensors are
bit-identical to the base quant, and the two vocabulary tensors are exact
row-subsets of theirs.

## What this is

Built from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
`UD-IQ4_XS` by removing non-ASCII vocabulary rows from `token_embd` and
`output`, then rewriting the tokenizer to match.

Embeddings are a *gather*, and the output head is a GEMV over the whole
vocabulary on every token. Shrinking both frees VRAM that becomes KV cache,
which is what buys the context.

Made with [https://github.com/bsaleh03/ASCII-Condensed-prune-tools](https://github.com/bsaleh03/ASCII-Condensed-prune-tools)

**What it is not:** a finetune, a distillation, or a requantization.

## Quickstart

```bash
llama-cli -m Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 \
  -ot "token_embd.weight=CPU" \
  -c 135168 -ub 128 -b 512
```

`-ot "token_embd.weight=CPU"` keeps the embedding table in system RAM. It is a
pure gather, so this costs ~1% of decode speed and frees its full footprint.

## Configuration

Maximum context measured on a 16 GB card (headless), with a real 15K-token
prompt — a context that merely *allocates* is not a usable context.

| `-ctk` / `-ctv` | bytes/token | max context |
|---|---|---|
| `q4_0` / `q4_0` | 18,432 | **135,168** |
| `q5_1` / `q4_0`* | 21,504 | 118,784 |
| `q8_0` / `q4_0`* | 26,624 | 98,304 |
| `q8_0` / `q8_0` | 34,816 | 77,824 |

### Throughput

Measured at `q4_0`/`q4_0`, `-c 135168`, on an RTX 5070 Ti:

| prompt depth | prefill | decode |
|---|---|---|
| 15K | 1,501 t/s | 47.3 t/s |
| 90K | 930 t/s | 33.1 t/s |

Decode slows as the KV cache fills — budget for the depth you actually work at,
not the shallow-context figure.

\* Mixed K/V types need a patched llama.cpp. **On stock builds `-ctk` and
`-ctv` must match** — mismatched types silently disable flash attention and
cost ~37× prefill speed, with no warning. Use `q4_0/q4_0` or `q8_0/q8_0`
unless you know your build supports mixed types.

### MTP (speculative decoding)

The model carries its MTP head, giving ~1.8× decode at reduced context:

| `-ctk`/`-ctv` | max context with MTP |
|---|---|
| `q4_0`/`q4_0` | 77,824 |
| `q8_0`/`q8_0` | 49,152 |

```bash
--spec-type draft-mtp --spec-draft-n-max 3 -ctkd q4_0 -ctvd q4_0
```

The draft model keeps its own KV cache and does **not** inherit `-ctk`/`-ctv`
— it defaults to f16, so pass `-ctkd`/`-ctvd` explicitly.

## Language support

The vocabulary is **ASCII-only**. All 256 byte-fallback tokens are retained,
so nothing is unrepresentable — but non-ASCII text costs more tokens:

| Text | Cost |
|---|---|
| English, code, ASCII punctuation | unchanged |
| Unaccented Latin script | unchanged |
| Accented Latin (é, ñ, ü, ç …) | ~2 tokens per accented character |
| CJK, Cyrillic, Arabic, Thai … | ~3 tokens per character |

Nothing breaks; it is a token-efficiency tax proportional to how non-ASCII
your text is. If you work in a non-English language, this build is not for you.

## Verification

- 864 / 866 tensors **byte-identical** to the base quant (BLAKE2b compared)
- Surviving vocabulary rows are **bit-exact** copies — the tables were
  row-gathered in quantized space, with no dequantize/requantize step
- All 276 special tokens and all 256 byte-fallback tokens retained
- Merge rules filtered so every surviving rule has surviving parents
- `bos`/`eos`/`pad` ids remapped; chat template unchanged
- On ASCII text the tokenizer produces **identical output to the base model**,
  token for token

## Known limitations

- **Text only.** No `mmproj` is provided and the vision pathway is untested.
- Non-ASCII input costs extra tokens (see above).
- Context ceilings are for a 16 GB card. More VRAM shifts them all upward.
- Mixed `-ctk`/`-ctv` requires a patched llama.cpp (see Configuration).

## Credits and license

- **Qwen** for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **unsloth** for the [GGUF quantization](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF);
  this build's weights are theirs, unaltered

Apache 2.0, inherited from the base model; `LICENSE` included.

**Modifications from the base:** vocabulary reduced 248,320 → 129,006 rows;
`token_embd.weight` and `output.weight` row-gathered to the retained set;
`tokenizer.ggml.tokens`, `token_type`, `merges` and special-token ids
rewritten to match. No other tensor was modified.
