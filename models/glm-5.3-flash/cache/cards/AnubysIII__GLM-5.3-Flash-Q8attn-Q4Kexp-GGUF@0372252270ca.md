---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
  - gguf
  - llama.cpp
  - moe
  - mtp
  - cpu-offload
pipeline_tag: image-text-to-text
---

# GLM-5.3-Flash: Q8 attention and shared expert, Q4_K routed experts (GGUF)

A GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total, about 17B active, 288 routed experts plus 1 shared), made for one consumer GPU plus a lot of system RAM.

- **Q8_0:** attention, the linear-attention blocks, the indexer, the shared expert, the 3 dense FFN layers, the MTP head, embeddings and output.
- **Q4_K:** routed experts only (`ffn_{gate,up,down}_exps`).

That comes to 185 GB (4.6 bits per weight), against 341 GB for Q8_0. With `-ngl 99 -ncmoe 99`, the Q8 part (about 11 GB) sits on the GPU and the experts run from RAM. So the Q8 attention costs almost nothing in speed: decode speed is set by the Q4 experts and your memory bandwidth.

## Files

The model is split into `GLM-5.3-Flash-Q8attn-Q4Kexp-0000N-of-00004.gguf`. Point llama.cpp at the first part.

`mmproj-F16.gguf` is the vision projector, from Unsloth, unchanged.

## How it was made

From Unsloth's Q8_0 GGUF, with Unsloth's imatrix:

```
llama-quantize --allow-requantize --imatrix imatrix_unsloth.gguf_file --tensor-type "ffn_.*_exps=q4_K" \
  <Unsloth Q8_0, first part>.gguf GLM-5.3-Flash-Q8attn-Q4Kexp.gguf Q8_0
```

The types listed above are read from the shipped file. Built with llama.cpp `f3363593f` (October 2026). You need that build or newer, for GLM5-Next and MTP support.

## Quality

Perplexity on wikitext-2 (test set, 40 chunks of 512 tokens): **2.647 ± 0.049**. The Q8_0 source was deleted before we measured, so there's no Q8_0 number to compare against.

## Speed (measured)

One RTX 3080 20 GB, 2x EPYC 7402 (Rome, AVX2, 16 DDR4 channels), `-ngl 99 -ncmoe 99 -t 40 -fa on`:

| | tok/s |
| --- | --- |
| generation (llama-bench tg) | 9.5 |
| server, code, plain | 8.9 |
| server, code, MTP `--spec-type draft-mtp --spec-draft-n-max 2` | **11.7** |
| prompt, 4096 tokens, `-b 8192 -ub 4096` | 140 |
| prompt, 512 tokens, ub 2048 | 23 |

Use a big ubatch for prompts. With MTP n=2 and `-b 4096 -ub 2048`, the server used 19.25 GB of the 20 GB card.

Vision works with the mmproj.

## Running it

```
llama-server -m GLM-5.3-Flash-Q8attn-Q4Kexp-00001-of-00004.gguf -ngl 99 -ncmoe 99 -t 40 -fa on \
  -b 4096 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 2 --mmproj mmproj-F16.gguf --jinja
```

You need about 190 GB of RAM plus your context. Thinking stays on even with `enable_thinking=false`.

## Credits and license

- Base model: Z.ai GLM-5.3-Flash, MIT.
- Q8_0 source, imatrix and mmproj: [Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
- [llama.cpp](https://github.com/ggml-org/llama.cpp).
- This quant: AnubysIII (Sovereign Games LLC). MIT.
