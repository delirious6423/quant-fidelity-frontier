---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- qwen
- qwen3.5
- unsloth
- dynamic-quantization
- apple-silicon
library_name: mlx
---

# Qwen3.8-27B — Unsloth Dynamic v3.0 allocation, native MLX

Apple MLX quant of `Qwen/Qwen3.8-27B` (text tower, `qwen3_5` hybrid arch:
transformer + Gated-DeltaNet + MTP) using the **per-layer bit allocation of
Unsloth's Dynamic v3.0 `Q3_K_XL` GGUF**, applied to **clean fp16 weights**
(no double quantization).

**3.95 bits/weight · 12.4 GB · loads and runs in oMLX / mlx-lm 0.31.3+**

## Why this model

- The quant recipe is not a fixed bit-width: each tensor keeps the bit-width
  the unsloth v3.0 imatrix calibration chose for it in the `Q3_K_XL` GGUF
  (2/3-bit where robust, 5/6/8-bit where sensitive), re-encoded in MLX's
  affine group-quant format.
- It was quantized once, from the fp16 checkpoint — not derived from the
  already-quantized GGUF, so there is no compounding quantization error.

## Bit allocation (per tensor, group_size 64, affine)

| Bits | Tensors | What |
|---|---|---|
| 2 | 24 | robust GDN/attention projections (from Q2/IQ2 source layers) |
| 3 | 157 | MLP gate/up, GDN in-proj (from Q3/IQ3 source layers) |
| 4 | 194 | bulk MLP down, attention, GDN (from Q4/IQ4 source layers) |
| 5 | 26 | sensitive projections, lm_head (from Q5_K source layers) |
| 6 | 1 | lm_head (from Q6_K source layer) |
| 8 | 96 | conv1d + high-value projections (from Q8_0 source layers) |
| bf16 | — | norms, A_log, dt_bias, F16/F32 source tensors |

## Verification (M1 Pro 32 GB, wikitext-2, 64×512-token samples, seed 123)

- **Perplexity: 12.724 ± 0.212**
- **Decode: 10.17 t/s**, prefill 49.2 t/s (batch 1)
- **Peak memory: 13.8 GB**

For reference, the same checkpoint with the published Unsloth Dynamic 2.0
static per-tensor table (5.27 bpw, 17 GB) scores 13.445 ± 0.231 at 8.23 t/s.
The v3.0 allocation is strictly better on every axis at 4.6 GB smaller.

## Usage

```bash
pip install mlx-lm
python -m mlx_lm.generate --model jclyons52/Qwen3.8-27B-Unsloth-Dynamic-v3-mlx \
  --prompt "Explain quantization in one sentence."
```

Serve:

```bash
python -m mlx_lm.server --model jclyons52/Qwen3.8-27B-Unsloth-Dynamic-v3-mlx
```

oMLX: add the repo id `jclyons52/Qwen3.8-27B-Unsloth-Dynamic-v3-mlx`.

## How it was built

1. Downloaded `Qwen/Qwen3.8-27B` fp16 (18 shards, ~54 GB).
2. Parsed the official `unsloth/Qwen3.8-27B-GGUF` `Q3_K_XL` file's per-tensor
   quant types (IQ2/IQ3/IQ4/IQ5/Q6/Q8 mixes) and mapped each to an MLX affine
   bit-width.
3. `mlx_lm.convert` with that per-tensor predicate, `dtype=bfloat16`,
   `group_size=64` — text-only (vision tower and MTP head are not included;
   mlx-lm 0.31.3's `qwen3_5` model strips them; text generation unaffected).

Reproducible with the scripts in [jclyons52/ud2mlx](https://github.com/jclyons52/ud2mlx):
`convert_gguf_to_mlx.py --gguf <Q3_K_XL.gguf> --match-source-quants --hf-path ./Qwen3.8-27B`.

## Caveats

- Text-only: this is the language-model tower of the multimodal checkpoint.
- `config.json` uses `model_type: qwen3_5` (loads with mlx-lm 0.31.3+).
- MTP/speculative head not included.
