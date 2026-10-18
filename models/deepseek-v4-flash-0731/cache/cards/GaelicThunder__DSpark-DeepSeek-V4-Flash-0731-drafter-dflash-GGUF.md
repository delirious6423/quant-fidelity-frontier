---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- gguf
- speculative-decoding
- dspark
- dflash
- llama.cpp
- deepseek
---

# DSpark drafter for DeepSeek-V4-Flash-0731 — mainline llama.cpp format (`dflash` arch)

**TL;DR**: the official DSpark speculative drafter for DeepSeek-V4-Flash-0731, converted from
the ds4-fork GGUF format (`deepseek4-dspark` arch) to **mainline llama.cpp** format (`dflash`
arch + DSpark markov/confidence heads). As far as I know this is the first publicly available
DSpark drafter that loads on upstream llama.cpp (`--spec-type draft-dspark`). Numerical content
is byte-identical to the source checkpoint (bleysg's 0731 drafter, Q2_K experts / Q8_0 attention):
pure rename of arch string, KV metadata keys and tensor names, plus the tokenizer block copied
from the target model (the fork format carries none) and `tokenizer.ggml.mask_token_id` set to
the DSpark noise token (128799).

## Usage

```bash
llama-server -m DeepSeek-V4-Flash-0731-<your-quant>.gguf -ngl 99 \
  --spec-type draft-dspark \
  --spec-draft-model DSpark-drafter-0731-mainline-dflash.gguf \
  --spec-draft-ngl 99 --spec-draft-n-max 5
```

`--spec-draft-n-max 5` matters: the drafter was trained with `block_size=5` and the default
(3) wastes 2 draft positions per block.

## Measured on GB10 (ASUS Ascent GX10, 121 GB unified, sm_121a, single box)

Target model: Unsloth UD-IQ3_XXS (97 GB), full GPU offload, temperature 0.

| workload | no spec | with DSpark | accept rate |
|---|---|---|---|
| code generation (512 tok) | 16.6 t/s | **30.5-31.5 t/s** | ~50%, mean accepted len 3.5/5 |
| literary prose (512 tok) | ~16.6 t/s | 20.2 t/s | ~25% |
| 256k ctx, KV q8_0, 2 expert layers on CPU, measured AFTER a real 227k-token prefill | — | 25.3 t/s | — |

Two practical notes from the same session: (1) speculative decoding composes badly with heavy
CPU expert offload — the k-token verify batch touches ~k× distinct experts on the CPU side, so
keep offload to 1-2 layers if you need it at all; (2) llama.cpp CUDA copies weights into device
buffers (it is not an mmap view), so leave real headroom on unified-memory boxes.

## Which target quant to pair it with (tested on 121 GB unified)

Unsloth's published KLD vs Q8 (lower = closer to full quality), mapped onto what actually
runs on a 121 GB unified-memory box together with this 6.5 GB drafter:

| target quant | size | KLD | measured here |
|---|---|---|---|
| **UD-IQ3_XXS** | 97 GB | 0.2403 | **the sweet spot**: the only quant that fits fully on GPU with the drafter → 31 t/s code / 20 t/s prose; 25.3 t/s code at 256k ctx (KV q8_0 + last 2 expert layers on CPU), verified after a real 227k-token prefill |
| UD-IQ3_S | 108 GiB | ~0.17 | not viable with the drafter: needs 5-6 expert layers on CPU, which inverts the speculative gain (k-token verify batches touch ~k× distinct experts on the CPU path) |
| UD-Q3_K_XL | 120 GB | 0.1062 | runs at 9 t/s with the experts of the last 7 layers on CPU, **without** speculation (spec measured harmful at that offload depth) — quality-max option, not a daily driver |
| UD-IQ4_XS | 128 GB | 0.0747 | ruled out by the same fit physics |

Extra datapoint: splicing Q3_K_XL expert tensors into the IQ3_XXS base (same Unsloth imatrix,
bytes directly transplantable) recovers only ~25% of the quality gap when swapping 21/43
expert layers — first or last, nearly the same — i.e. the quantization divergence is spread
almost uniformly across layers. There is no free intermediate quant on this hardware.

## Conversion

`dspark-drafter-convert.py` (included) does the whole thing in ~1 minute with gguf-py only —
no requantization. Per-layer tensors map 1:1 (`dspark.N.*` → `blk.N.*`, names identical);
globals: `main_proj`→`fc`, `main_norm`→`enc.output_norm`, `norm`→`output_norm`,
`hc_head_*`→`output_hc_*` (the only non-obvious rename); KV: `dflash.block_size=5`,
`dflash.target_layers=[40,41,42]`, hparams copied from the target model's metadata
(the drafter stages are full DSV4 blocks with the same dimensions). The script asserts
tensor count, shapes and swap coverage, and is trivially adaptable to future DSpark
checkpoints in the fork format.

Source checkpoint: bleysg's DSpark-drafter-Q2K-Q8-0731 (ds4 fork format). All credit for the
drafter training goes to the DeepSeek team; conversion only here. MIT, same as the base model.
