---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-vlm
- speculative-decoding
- mtp
- qwen3.8
- unsloth
- gguf-to-mlx
---

![](./banner.jpg)

# Qwen3.8-27B · native MTP drafter (from Unsloth UD-IQ4_XS)

This is Qwen3.8-27B's built-in **multi-token-prediction (MTP)** head, converted from the `blk.64` / `nextn.*` tensors of [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-IQ4_XS`. It is packaged as a standalone **mlx-vlm `qwen3_5_mtp` speculative-decoding drafter** for the main checkpoint, [**Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX**](https://huggingface.co/Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX).

It is 0.35 GB, and in greedy decoding the output is **token-for-token identical** to the model without the drafter. It is just faster:

| prompt (256 new tokens, greedy) | baseline | + MTP drafter | speedup | drafts accepted |
|---|---:|---:|---:|---:|
| math word problem | 14.1 tok/s | **27.7 tok/s** | **1.96×** | 94.9 % |
| Python function | 14.4 tok/s | **24.8 tok/s** | **1.72×** | 93.3 % |
| open-ended Chinese prose | 14.3 tok/s | 16.8 tok/s | 1.18× | 47.1 % |

Measured on an Apple Silicon Mac with 24 GB RAM, mlx-vlm 0.7.4 / mlx 0.32.3, block size 3 (2 drafted tokens per round). Peak memory was 17.6 GB with both models loaded, about 1.4 GB more than without the drafter. Structured text such as code and math gains the most. Free-form prose gains the least.

## Usage

The main model goes in `--model` and this repo goes in `--draft-model`:

```bash
pip install -U mlx-vlm

mlx_vlm generate \
  --model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX \
  --draft-model Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp --draft-kind mtp \
  --prompt "Solve step by step: what is 17 * 23 + 19?" --max-tokens 512 --temperature 0
```

```python
from mlx_vlm import load, generate
from mlx_vlm.speculative.drafters import load_drafter

model, processor = load("Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX")
drafter, kind = load_drafter("Steven10429/Qwen3.8-27B-UD-IQ4_XS-MLX-mtp")  # kind == "mtp"
msgs = [{"role": "user", "content": "Write a haiku about unified memory."}]
prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print(generate(model, processor, prompt, max_tokens=256, temperature=0.0,
               draft_model=drafter, draft_kind=kind).text)
```

The drafter reuses the target model's `embed_tokens` and `lm_head`, so it only works together with the main checkpoint above. `mlx-lm` has no MTP drafter support, so use `mlx-vlm` for this.

## Why a separate repo?

When a Qwen3.5-family checkpoint contains `mtp.*` keys, mlx-vlm decides it has not been sanitized and adds `+1` to the norms again. Shipping MTP inside the main model would break it. The mlx-vlm convention, used by `mlx_vlm.convert --mtp` and `mlx_vlm.split_mtp`, is a separate `<model>-mtp` drafter folder with `model_type: qwen3_5_mtp`, and that is what this repo is.

## Conversion & validation

- Source tensors: `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` plus one full-attention decoder layer. Unsloth kept them at **Q6_K / Q8_0**, and they map to MLX **6-bit / 8-bit** affine (group size 64). The norms stay in bf16.
- No layout changes were needed beyond renaming. The GGUF norms are already `1 + w`, which is what MLX stores. `eh_proj` concatenates `[embedding, hidden]` in the same order as HF `mtp.fc`, and there is no head permutation.
- Checked against the **official bf16 `mtp.*` tensors of `Qwen/Qwen3.8-27B`**, which were range-downloaded so the 54 GB checkpoint wasn't needed. Every matrix is within 0.05 relative error: Unsloth Q6_K is about 0.018 and the MLX requant adds about 0.023 in quadrature. The norms match exactly, up to bf16 rounding.
- Greedy speculative output was token-identical to plain decoding on all 3 benchmark prompts.
- Converter: [`conversion/gguf2mlx.py`](./conversion/gguf2mlx.py) (`--mtp-out`). Per-tensor stats: [`conversion_stats.json`](./conversion_stats.json).

## Credits

Model by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). GGUF by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). MTP speculative decoding runtime by [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).
