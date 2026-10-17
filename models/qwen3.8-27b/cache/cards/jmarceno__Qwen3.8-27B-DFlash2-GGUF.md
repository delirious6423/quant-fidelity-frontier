---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-27B
  - incoai/Qwen3.8-27B-DFlash2
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - draft-model
  - dflash2
  - block-diffusion
  - vision
---

# Qwen3.8-27B-DFlash2-GGUF

GGUF conversion of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
(the DFlash 2 block-diffusion drafter, also mirrored as
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2))
for **llama.cpp's `draft-dflash` speculative decoding**.

This is a **draft model, not a standalone language model**. It drafts a whole
block of tokens in one pass and lets a Qwen3.8-27B target verify them. It has
no token embeddings and no output projection of its own: the draft decoder
consumes the target's embeddings/lm-head plus hidden states extracted from five
target layers, so llama.cpp only runs it together with a compatible target.

Decoding stays lossless in the same sense as the original: greedy output
matches the target exactly, sampling preserves the target's distribution — the
drafter only changes how many target forward passes are needed.

> **2026-09-27: image prompts now work.** The files were re-uploaded with one
> added metadata key, `dflash.rope.dimension_sections = [64, 0, 0, 0]`. Before
> that, any request containing an image (target loaded with `--mmproj`) failed
> with `decode() failed: failed to process speculative batch`. Tensors are
> unchanged and text-only drafting is identical, so re-download only if you
> serve images — or patch a local copy in place of the download with
> `add_dflash_mrope_sections.py`. See [Images / vision](#images--vision).

## Files

| file | size | sha256 | notes |
| --- | --- | --- | --- |
| `Qwen3.8-27B-DFlash2-Q4_K_M-swa.gguf` | 1.09 GiB | `76fd50b90c65b254a24b1a8f5771574e2d13908f0c3bdb271aee223e30573eaf` | **recommended default**; measured same/better decode than Q8_0 at depth and ~0.9 GiB smaller |
| `Qwen3.8-27B-DFlash2-Q8_0-swa.gguf` | 1.96 GiB | `c13f57570c5a7695a5b1d3e02ecd57afb4793122d609779475fc3a9761890baf` | higher-precision alternative |
| `Qwen3.8-27B-DFlash2-F16-swa.gguf` | 3.86 GiB | `0d43457bcbad64e981ac839eeddc4e03dff9f544f1497e6eae937db04b902014` | full-precision GGUF, requantize from this instead of the safetensors |
| `convert_dflash2_to_gguf.py` | — | — | the conversion script used for all three files |
| `add_dflash_mrope_sections.py` | — | — | adds the M-RoPE key to a drafter converted before 2026-09-27 |

Previous (text-only) uploads, still in the repo history: Q4_K_M
`9f3b6d6de16a8fcdf34b719418ba34f9e2e51785d0a74a2bf4dfc16eabf40e86`, Q8_0
`d6ba93a5072569ee20928d26bac28b8b7dbe3b503e5833ac3504c8e6ae50d580`, F16
`6c8c725f30cee36c2f2154f76687dfd57f1a7beb5ffcc8e1487a0223fc54bd4a`.

All three files are the `-swa` conversion: they carry the drafter's trained
sliding-window metadata (window 2048, per the checkpoint config). An F16
elementwise operand aborts llama.cpp's CUDA binary-broadcast path
(`binbcast.cu: GGML_ASSERT(nb10 % sizeof(src1_t))`), so **norm weights and the
two-tap conv `*_base` kernels are stored F32**; the script keeps that rule.

## Requirements

- llama.cpp **with DFlash 2 drafter support** (`dflash` arch + the selector in
  `src/models/dflash.cpp`, i.e. the `draft-dflash` spec type). Tested against
  `ggml-org/llama.cpp` master `56381e407c0ccfb3a6f71e668a27a901001d22ce`
  (`build 0.4.0-dev`).
- A **Qwen3.8-27B** target GGUF with the same tokenizer (the drafter reuses the
  target's embeddings and output projection, so the tokenizer must match; the
  draft file's tokenizer metadata is copied from the target and its
  `mask_token_id` is set from the drafter config: **248070**).

## Usage

```bash
llama-server \
  -m Qwen3.8-27B-IQ4_XS.gguf \
  -c 131072 --parallel 1 \
  --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M-swa.gguf \
  --spec-draft-n-max 7 \
  --spec-draft-p-min 0 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  -fa on
```

Notes:

- `--spec-draft-n-max 7` is the trained depth: DFlash 2's block size is 8
  (anchor + 7 draft tokens). The engine clamps larger values and warns.
- The drafter is a normal offloadable model; on a tight-VRAM dual-GPU box put it
  on the card with room (`--spec-draft-device CUDA1`) — but if the target's
  shared tensors span several devices, give the drafter those devices too
  (`--spec-draft-device CUDA0,CUDA1`), otherwise the scheduler rejects the
  shared `output.weight`/`token_embd`.
- Use `--parallel 1` unless you need concurrency: llama.cpp allocates recurrent
  (hybrid GDN) state per slot, so an unused 4-slot server wastes several GiB.
- The drafter's KV cache is capped by its 2048-token sliding window, so it adds
  only tens of MiB regardless of the target context length.
- Vision works: add the target's `--mmproj` as usual (see below).

## Images / vision

Qwen3.8-27B uses M-RoPE. An image of *n* tokens advances the text position by
only max(h, w), so the text after an image resumes past the end of what the
drafter's KV has seen. A drafter declared with plain NeoX RoPE requires strictly
consecutive positions and rejects that jump, which killed every image request.

These files declare degenerate M-RoPE sections `[head_dim/2, 0, 0, 0]`. llama.cpp
then treats the drafter as M-RoPE on the temporal dimension only: text rotation
is exactly the NeoX rotation, the draft KV accepts the forward jump, and the
image rows themselves are skipped from draft injection (the draft conditions on
the target's hidden states of the surrounding text). The upstream DFlash
converter writes the same key for M-RoPE targets.

Checked on the setup below: greedy output and draft counts identical to the old
files on text; a generated colour-square image is answered correctly under
`draft-dflash` with the projector on GPU or CPU. With a 512x512 image plus
120,517 prompt tokens, vision on GPU0 and a 131,072 reservation, this drafter
decoded **41.6 tok/s** after the prefill vs **30.4 tok/s** for the built-in MTP
head (6 tokens) on the same engine, at 450.7 tok/s prefill. That engine carries
local FlashAttention patches (in-kernel q4_0 K/V for the 8-row verify), so
absolute numbers on stock llama.cpp will be lower; the DFlash-over-MTP ranking
matches the table below.

## Measured performance

2× RTX 3060 12 GB, Qwen3.8-27B target (IQ4_XS), 131,072-token reservation,
target KV `q4_0/q4_0`, `-b 256 -ub 128`, `--parallel 1`, thinking on. "Exercised
context" is the prompt length actually processed, reusing a shared prefix
between steps.

| exercised ctx | decode | acceptance | mean accepted/step | prefill |
| --- | --- | --- | --- | --- |
| ~58K | **36.3 tok/s** | 49.7% | 4.45 | 447 tok/s |
| ~90K | **30.7 tok/s** | 48.8% | 4.38 | 341 tok/s |

Same-prompt A/B against the target's built-in MTP (7-token) head:

| engine | ~58K | ~90K | ~120K |
| --- | --- | --- | --- |
| built-in MTP | 20.8–24.3 tok/s | 17.6 | n/a |
| this drafter, Q4_K_M | **28.5** | **25.9** | **23.0** |
| this drafter, Q8_0 | 26.5 | 23.6 | n/a |

The drafter runs **24–34% faster than MTP at depth** even though its raw
acceptance is lower (36–50% vs 40–51%): one parallel draft pass replaces several
sequential MTP passes, so its per-step cost is much smaller (e.g. 149 ms vs
192 ms at ~90K).

Observed acceptance varies with prompt/context (roughly 28–60% across our
prompts); block drafting holds throughput up better than depth-reduced MTP as
context grows. For reference, the upstream model card reports acceptance length
5.46 / 5.28 / 4.39 / 4.79 / 4.10 (GSM8K / MATH-500 / HumanEval / MBPP /
MT-Bench) on H200 + SGLang.

### KV precision

On the above hardware, target KV precision had **no measurable effect on decode
step time** at 120K+ context (q8_0/q4_0, q4_0/q4_0 and q4_0/q8_0 all landed
within ~2% at 58/90/120K), only on VRAM: `q4_0/q4_0` is the smallest and just as
fast, and `f16/f16` (~8 GiB of KV at 131K) does not fit on 12 GB cards at all.
The draft's own KV can be `q4_0` too.

## Conversion

`convert_dflash2_to_gguf.py` reads the original **BF16 safetensors** directly
(no dequantize/requantize round trip) and writes the `dflash`-arch GGUF:

```bash
pip install gguf numpy torch
python3 convert_dflash2_to_gguf.py \
  --src-dir ./Qwen3.8-27B-DFlash2 \
  --target-gguf ./Qwen3.8-27B-IQ4_XS.gguf \
  --out ./Qwen3.8-27B-DFlash2-F16-swa.gguf

# optional: quantize for serving (the M-RoPE key survives quantization)
llama-quantize ./Qwen3.8-27B-DFlash2-F16-swa.gguf \
               ./Qwen3.8-27B-DFlash2-Q4_K_M-swa.gguf Q4_K_M
```

The `--target-gguf` argument only supplies the tokenizer (shared with the
target); `--no-swa` omits the sliding-window metadata (not recommended: the
draft KV then grows with context, ~40 KB/token).

Metadata written: `dflash.block_size 8`, `conv_kernel_size 2`,
`conv_group_size 16`, `selector_rank 256`, `selector_top_k 16`,
`target_layers [5, 19, 33, 47, 61]`, sliding window 2048 with an all-true
pattern, rope theta 1e7, `rope.dimension_sections [64, 0, 0, 0]` (M-RoPE
target, since 2026-09-27), and the target tokenizer with
`tokenizer.ggml.mask_token_id 248070`.

## Changelog

- **2026-09-27** — Re-uploaded all three GGUFs with
  `dflash.rope.dimension_sections = [64, 0, 0, 0]` so image prompts work under
  `draft-dflash`; tensors unchanged. The converter writes the key; added
  `add_dflash_mrope_sections.py` for older local conversions.
- **2026-09-15** — Initial upload (Q4_K_M / Q8_0 / F16, `-swa`).

## Credits

- [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
  — the original bf16 drafter (Inco AI); also at
  [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).
  Apache-2.0.
- [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
  — the W4A16 requant whose reported file layout helped validate this conversion.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the DFlash/DFlash2 drafter
  implementation these files target.
- Conversion, GGUF packaging and measurements: this repository's authors.

## Citation

If you use DFlash 2, cite the original work:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```

```bibtex
@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
