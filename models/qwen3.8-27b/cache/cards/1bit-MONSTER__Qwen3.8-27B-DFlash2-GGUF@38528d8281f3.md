---
license: apache-2.0
base_model:
- z-lab/Qwen3.8-27B-DFlash2
- Qwen/Qwen3.8-27B
tags:
- gguf
- dflash2
- speculative-decoding
- draft-model
- qwen3.8
---

# Qwen3.8-27B DFlash2 drafters — GGUF

GGUF conversions of the DFlash 2 draft model for `Qwen/Qwen3.8-27B`
([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), Apache-2.0,
a mirror of `incoai/Qwen3.8-27B-DFlash2`). This is not a standalone language model: it drafts
blocks of tokens that a Qwen3.8-27B target verifies, so it only runs inside a speculative-decoding
server next to a Qwen3.8-27B GGUF. The weights are the original drafter's, converted and quantized;
nothing was retrained.

## Contents

| file | type | size |
|---|---|---|
| `Qwen3.8-27B-DFlash2-q8_0.gguf` | Q8_0 | 2.06 GB |
| `Qwen3.8-27B-DFlash2-q4k.gguf` | Q4_K | 1.21 GB |

Both carry the drafter's metadata: `dflash.block_size = 8` (seven drafted tokens per verify step)
and `dflash.selector_top_k = 16`. The Q8_0 file is what the drafter's own README recommends over
BF16; the Q4_K file is the smaller alternative.

## How they were made

```bash
hf download Qwen/Qwen3.8-27B --include "*.json" "*.jinja" "merges.txt" "vocab.json" --local-dir Qwen3.8-27B-hf
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir DFlash2
python convert_hf_to_gguf.py DFlash2 --target-model-dir Qwen3.8-27B-hf --outtype bf16 --outfile Qwen3.8-27B-DFlash2-bf16.gguf
llama-quantize Qwen3.8-27B-DFlash2-bf16.gguf Qwen3.8-27B-DFlash2-q8_0.gguf Q8_0
llama-quantize Qwen3.8-27B-DFlash2-bf16.gguf Qwen3.8-27B-DFlash2-q4k.gguf Q4_K
```

with the [1bit-MONSTER/llama.cpp](https://github.com/1bit-MONSTER/llama.cpp) fork (branch
`1bit/hrx-vulkan-patched`), whose DFlash2 support is the upstream port
(`ggml-org/llama.cpp#27816`) plus the fork's HRX runtime changes.

## Running it

With the [1bit engine](https://github.com/1bit-MONSTER/engine) (`1bit serve`), next to any
Qwen3.8-27B GGUF, for example the
[`UD-Q4_K_XL` bundle](https://huggingface.co/1bit-MONSTER/Qwen3.8-27B-UD-Q4_K_XL-MTP):

```bash
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device hrx --dflash Qwen3.8-27B-DFlash2-q8_0.gguf
```

`--mtp` and `--dflash` are exclusive. The engine keeps the drafter's whole 8-token blocks
(`--spec-draft-p-min 0`, set by its built-in recipe for this model); raising the threshold throws
blocks away and slows prose. With plain `llama.cpp` (`llama-server`) use the fork and
`--spec-type draft-dflash -md Qwen3.8-27B-DFlash2-q8_0.gguf`.

When to pick it over the in-file MTP head: DFlash2 drafts longer blocks, so it pays most on
code and structured output; the MTP head (`--mtp`) stays the better default for short chat
turns. A drafted token is kept only when the target model agrees, so the output is the target's own.

## License

Apache-2.0, as the drafter it converts (see `LICENSE`). Qwen3.8-27B itself is Apache-2.0.
