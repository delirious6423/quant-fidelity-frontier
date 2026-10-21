---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
  - qwen
  - qwen3
  - unsloth
  - vocabulary-pruning
  - ascii
language:
  - en
pipeline_tag: text-generation
library_name: gguf
---

# Qwen3.8-27B-UD-IQ3_XXS-ASCII-Condensed

A GGUF build of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), starting from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s `UD-IQ3_XXS` quant, with the vocabulary pruned down to ASCII-representable tokens only. All surviving weights are bit-identical to the source quant. This is a vocabulary edit, not a re-quantization. **It's about 500MB smaller in terms of VRAM for what it's worth if you really need to squeeze in and are VRAM poor.** In terms of what happens if the model wants to output a lot of unicode checkmarks or think in chinese, well, those logits won't be in the list for the sampler to pick and use so single byte composites will represent them and more tokens will be used. And maybe it'll make the attention less effective.

## What this is

The base `UD-IQ3_XXS` GGUF ships with Qwen's full multilingual BPE vocabulary (248,320 tokens), which includes coverage for CJK, Cyrillic, Arabic, emoji, and dozens of other scripts. If your use case only ever needs English/ASCII text, most of that vocabulary is dead weight: every one of those tokens has a corresponding row in the embedding and output-projection tensors, and those two tensors are large enough (`5120 x 248320`) to matter even at 2-3 bits per weight.

This operation was done with a custom perl tool at: [http://superkuh.com/ascii-vocab-only-surgery-for-qwen3827b.pl.txt](http://superkuh.com/ascii-vocab-only-surgery-for-qwen3827b.pl.txt)

This build removes every vocabulary entry that isn't ASCII-safe, while leaving every remaining weight untouched:
| | Original | ASCII-Condensed |
|---|---|---|
| Vocabulary size | 248,320 | 128,190 |
| Merge rules | 247,587 | 128,775 |
| `token_embd.weight` rows | 248,320 | 128,190 |
| `output.weight` rows | 248,320 | 128,190 |
| Quantization | UD-IQ3_XXS | UD-IQ3_XXS (unchanged) |

| Text | Cost |
|---|---|
| English, code, ASCII punctuation | unchanged |
| Unaccented Latin script | unchanged |
| Accented Latin (é, ñ, ü, ç …) | ~2 tokens per accented character |
| CJK, Cyrillic, Arabic, Thai … | ~3 tokens per character |

## Credits

- Base model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team (Alibaba).
- Vocabulary-pruning technique: same general approach as [bsaleh03/Qwen3.8-27B-ASCII-Condensed](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed).
- Source GGUF/quantization: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (`UD-IQ3_XXS`) by Unsloth.

## How it was built

This is a surgical edit performed directly on the GGUF container, entirely in quantized space. No tensor is ever dequantized or requantized:

1. **Token survival rule.** Every token in `tokenizer.ggml.tokens` is decoded from its GPT2-style byte-level-BPE surface form back to the raw byte(s) it represents. A token is kept if:
   - it's a `CONTROL`-type token (special/chat-template tokens like `<|im_start|>`), or
   - it decodes to exactly one raw byte (the atomic byte-fallback tokens. These guarantee llama.cpp's tokenizer can always represent *any* input byte, even non-ASCII ones, by falling back to raw bytes when needed), or
   - it decodes to a byte sequence that's entirely ASCII (< 0x80).

   Every other token i.e. any multi-byte merged token whose decoded bytes include a non-ASCII byte is dropped.

2. **Row-gather the two vocabulary tensors.** `token_embd.weight` and `output.weight` each have one row per vocabulary entry. For every surviving token, its row of raw quantized bytes (whatever block format the tensor already uses `Q2_K` and `Q4_K` respectively in this file) is copied byte-for-byte into the new tensor, in the same relative order. Dropped rows are simply skipped. Every surviving weight is therefore bit-identical to the source file; nothing is recomputed.

3. **Filter the BPE merge table.** A merge rule is kept only if both of its parent tokens survived step 1. This keeps the tokenizer's merge search consistent with the pruned vocabulary.

4. **Remap special-token IDs.** `bos_token_id`, `eos_token_id`, `padding_token_id`, and friends are forced to survive step 1 regardless of type, then rewritten to point at their new (post-pruning) positions.

5. **Copy everything else unchanged.** All 864 other tensors (attention, FFN, SSM, and norm weights across all 65 layers) are copied byte-for-byte from the source file, untouched. Tensor offsets are recomputed to account for the two shrunk tensors, but no other tensor's data is modified in any way.

The result is a fully valid, standalone GGUF file. Not a diff or patch against the original.

## Usage

This is a normal GGUF file; anything that loads `unsloth/Qwen3.8-27B-GGUF` (llama.cpp, llama-cpp-python, LM Studio, Ollama via a Modelfile, etc.) will load this one the same way:

```bash
llama-cli -m Qwen3.8-27B-UD-IQ3_XXS-ASCII-Condensed.gguf \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 \
  -c 135168 -ub 128 -b 512
```


## Limitations

- **No non-ASCII output.** Because the vocabulary no longer contains tokens for non-ASCII characters (accented Latin letters, CJK, Cyrillic, Greek, emoji, etc.), the model cannot produce these in its output. Where the base model would have used a dedicated token, this model can only reconstruct the underlying bytes via the surviving single-byte tokens, if it reconstructs them at all. In practice, expect garbled or refused output for non-English scripts.
- **Non-ASCII input is not well supported either.** Prompts containing non-ASCII text will tokenize inefficiently (falling back to per-byte tokens) and may not be understood as intended, since the model was never trained against that fragmented tokenization.
- **Not a re-quantization.** This process does not change quantization quality, inference speed, or (beyond the two resized tensors) memory footprint of the transformer blocks. The savings are limited to the embedding and output-projection tensors' vocabulary dimension.
- Intended for **English/ASCII-only use cases** (chat, code, structured text generation) where the multilingual vocabulary was never going to be used.


## License

Apache 2.0, inherited from the base Qwen3.8-27B license. See the base model's repo for full license terms.
