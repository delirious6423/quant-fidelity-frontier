---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- mlx
- mlx-vlm
- qwen3.8
- unsloth
- gguf-to-mlx
- mixed-precision
- apple-silicon
---

![](./banner.jpg)

# Qwen3.8-27B · Unsloth UD-IQ4_XS → MLX

[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) **`UD-IQ4_XS`** (Unsloth Dynamic), converted to an **MLX** checkpoint for Apple Silicon. Text + vision. Runs with `mlx-lm`, `mlx-vlm` and LM Studio's MLX engine.

The interesting part is *how* the conversion was done. Unsloth's dynamic quant picks a precision per tensor using an importance matrix, from IQ2_XS on some early MLP layers up to Q8_0 on the gating projections. This checkpoint keeps that allocation. Each tensor's MLX bit width follows the type Unsloth chose for it, so you aren't getting one flat 4-bit everywhere.

| | this repo | `lmstudio-community/Qwen3.8-27B-MLX-4bit` (made from bf16) |
|---|---|---|
| Source | Unsloth UD-IQ4_XS GGUF + mmproj-F16 | original bf16 safetensors |
| Text quantization | mixed 4 / 5 / 6 / 8-bit affine, gs 64 | uniform 4-bit affine, gs 64 |
| Size on disk | 16.45 GB (text ≈ 15.5 GB + bf16 vision tower) | 16.05 GB |
| wikitext-2 PPL (16 × 2048 tok) | **6.4428** | 6.4349 |
| Peak memory (2k-token forward) | 19.1 GB | 18.7 GB |
| Chat template | **Unsloth's fixed template** (from the GGUF) | upstream |

Perplexity comes within **0.12 %** of a direct bf16 → 4-bit MLX conversion, even though these weights went through two rounds of quantization. Measured on an Apple Silicon Mac with 24 GB RAM. Swap and memory pressure will depend on your machine.

> ⚡ **Native MTP drafter:** [**Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp**](https://huggingface.co/Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp) (0.35 GB). It is the model's own multi-token-prediction head, converted from the same GGUF. With mlx-vlm speculative decoding it gave **1.7–2.0× faster** generation on code and math, and 1.2× on free-form prose, with token-identical greedy output. See the usage section below.

## Usage

```bash
pip install -U mlx-lm mlx-vlm

# text
mlx_lm.generate --model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX \
  --prompt "Explain gated delta networks in three sentences." --max-tokens 400

# image + text
mlx_vlm.generate --model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX \
  --image photo.jpg --prompt "What is in this picture?" --max-tokens 200
```

```bash
# faster: speculative decoding with the native MTP head (mlx-vlm)
mlx_vlm generate --model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX \
  --draft-model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp --draft-kind mtp \
  --prompt "Write a quicksort in Rust." --max-tokens 512 --temperature 0
```

```python
from mlx_lm import load, generate
model, tok = load("Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX")
prompt = tok.apply_chat_template([{"role": "user", "content": "Hi!"}], add_generation_prompt=True)
print(generate(model, tok, prompt=prompt, max_tokens=256))
```

**LM Studio:** download it from the in-app model search, or drop the folder into `~/.lmstudio/models/<publisher>/<name>/`. You need about 24 GB of unified memory or more.

Recommended sampling, from the GGUF metadata: `temperature=1.0, top_p=0.95, top_k=20`. Qwen3.8 is a reasoning model, so give it room with `max_tokens`.

## How the bits were allocated

When you re-quantize, the extra error depends almost entirely on the *target* bit width, not on the source type. Measured relative error was about 0.09 at 4-bit, 0.045 at 5-bit, 0.023 at 6-bit and 0.008 at 8-bit. Errors add in quadrature, so the cheap way to stay faithful is to spend bits where Unsloth already did.

| GGUF type (Unsloth's choice) | tensors | → MLX |
|---|---:|---|
| IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q2_K, Q3_K | 82 | 4-bit |
| IQ4_XS, IQ4_NL, Q4_K | 267 | 4-bit |
| Q5_K | 51 | 5-bit |
| Q6_K | 2 | 6-bit |
| Q8_0 (`in_proj_a` / `in_proj_b`) | 96 | 8-bit |
| F32 (norms, `A_log`, `dt_bias`, conv1d) | — | bf16 (`A_log` fp32) |
| vision tower (mmproj F16) | 333 | bf16, lossless |

The ≤3-bit sources are *up*-cast to 4-bit on purpose. Re-quantizing an IQ2 codebook to 2-bit or 3-bit affine would add 0.2–0.4 relative error on top of an already lossy tensor. Every tensor's GGUF type, MLX bits and measured requant error are listed in [`conversion_stats.json`](./conversion_stats.json).

## Conversion details

The converter, [`conversion/gguf2mlx.py`](./conversion/gguf2mlx.py), streams the GGUF one tensor at a time: it dequantizes with `gguf.quants`, undoes llama.cpp's layout changes, casts to bf16 and calls `mx.quantize`. It never holds the full bf16 model in RAM, so the whole 27B conversion takes about 3 minutes and 7 GB RSS. These are the llama.cpp `qwen35` layout changes it reverses. Each was found and checked numerically against an MLX checkpoint built from bf16:

- **Linear-attention value heads are re-tiled.** With 16 K-heads and 48 V-heads, ggml stores V head `r·16 + k` where HF/MLX expects `k·3 + r`. This is undone on `in_proj_qkv` (V rows), `in_proj_z`, `in_proj_a/b`, `A_log`, `dt_bias`, `conv1d` (V channels) and `out_proj` (columns).
- `ssm_a` holds `-exp(A_log)`, so the converter takes `log(-x)`.
- `ssm_conv1d` is `(C, K)`; MLX expects `(C, K, 1)`.
- RMSNorm weights are already stored as `1 + w`, which is what MLX stores too, so they need no change.
- The vision `patch_embd` is stored as two temporal slices. They are stacked into MLX's channels-last Conv3d `(O, T, H, W, C)`.
- The MTP (multi-token-prediction) block `blk.64` is not part of this checkpoint. mlx-vlm would treat `mtp.*` keys as an unsanitized checkpoint and re-add `+1` to the norms. It is shipped instead as a standalone `qwen3_5_mtp` drafter in [the `-mtp` repo](https://huggingface.co/Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp), produced with `--mtp-out`.

Validation: all 1,184 converted tensors (851 text + 333 vision) were compared with `lmstudio-community/Qwen3.8-27B-MLX-4bit`. Correlation is above 0.97 for tensors from ≥4-bit GGUF types and above 0.92 for Unsloth's IQ2/IQ3 tensors, where the source itself is that lossy. The vision tower is at 0.995 or higher. A layout bug would push these to about 0. The GGUF's embedded vocabulary also matches `tokenizer.json` exactly.

To reproduce:

```bash
python conversion/gguf2mlx.py \
  --gguf Qwen3.8-27B-UD-IQ4_XS.gguf --mmproj mmproj-F16.gguf \
  --hf-config config.json --tokenizer-dir <dir with Qwen3.8 tokenizer.json> \
  --out Qwen3.8-27B-UD-IQ4_XS-MLX --mtp-out Qwen3.8-27B-UD-IQ4_XS-MLX-mtp
```

## Caveats

- This is a *re-quantization of a quantized model*. On a 16×2k wikitext window it matches a direct bf16 → MLX 4-bit conversion. It has not been run through task benchmarks, so treat it as experimental. If you only care about quality per GB, a direct conversion from bf16 is still the principled choice. This repo is for people who want Unsloth's dynamic allocation and fixed chat template on MLX.
- Tokenizer files come from the official Qwen3.8 release. The chat template is Unsloth's, taken from the GGUF.

## Credits

Original model by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). Dynamic GGUF quantization and template fixes by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Runtime by [MLX](https://github.com/ml-explore/mlx).
