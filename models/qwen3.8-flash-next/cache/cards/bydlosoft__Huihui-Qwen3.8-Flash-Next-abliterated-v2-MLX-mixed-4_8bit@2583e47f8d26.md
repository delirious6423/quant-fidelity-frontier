---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - mlx
  - mlx-vlm
  - qwen4_exp
  - moe
  - abliterated
  - uncensored
---

# Huihui Qwen3.8-Flash-Next abliterated v2, MLX mixed 4/8-bit

An MLX build of [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF), an abliterated version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (about 180B parameters, 6B active per token, 262,144-token context). It loads with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.12 or later and needs about 110 GB of unified memory.

| Part | Size |
|---|---|
| Weights, 100 files | 74.2 GB |
| n-gram embedding table as a row store (`ple-q4.rows`, read from disk row by row) | 32.0 GB |
| MTP drafter, 8-bit (`mtp-drafter/`) | 2.8 GB |

## How it was built

The abliterated model is published as GGUF only. This build starts from the official BF16 weights and replaces the tensors that abliteration changed:

1. The base weights were converted with llama.cpp's converter and compared, tensor by tensor, with the abliterated `Q8_0` GGUF. Exactly 96 tensors differ: the attention output projections (12), the linear-attention output projections (36) and the shared-expert down projections (48), each by 3–4.5%, more than 93% of it along one direction. Every other tensor stored at 8 bits in both files is byte-identical; sampled routed experts differ only by 8-bit rounding.
2. The 96 tensors were dequantized from the GGUF, with the converter's column reordering inverted, and written in place of the base tensors. Converted forward and quantized to `Q8_0` again, all 96 reproduce the published GGUF tensors byte for byte.
3. The result was quantized with a mixed recipe, the one that [PipeNetwork](https://github.com/PipeNetwork/qwen38-flash-next-mlx) measured at +1.3% perplexity against BF16 (uniform 4-bit: +20.6%):

| Weights | Precision |
|---|---|
| Routed experts | 4-bit, group 64 |
| n-gram embedding table | 4-bit, group 32 |
| All other quantizable weights, including the 96 abliterated tensors | 8-bit, group 64 |
| MoE router, residual write gates, linear-attention `in_proj_a`/`in_proj_b`, sparse-attention indexer, vision encoder | BF16 |

All quantization scales and biases are BF16. (An earlier internal build had float32 scales on the 96 abliterated tensors; in MLX that turns the whole residual stream into float32 and makes decoding about 3.5 times slower.)

## Evaluation

| Check | Result |
|---|---|
| Perplexity, wikitext-2 test, the 145 windows of 2,048 tokens from PipeNetwork's evaluation | 4.5502, 95% CI [4.229, 4.882] |
| PipeNetwork's numbers for the base model on the same windows | BF16 4.4708, mixed 4/8-bit 4.5286 |
| Retrieval of 3 facts placed at 10%, 50% and 90% of a wikitext-2 document | 3 of 3 at 16k, 32k, 64k, 128k, 192k and 258k tokens |

The perplexity difference from the base model's mixed build (+0.5%) is consistent with the small change that abliteration makes.

## Usage

```bash
pip install "mlx-vlm>=0.7.12"
python -m mlx_vlm.generate --model bydlosoft/Huihui-Qwen3.8-Flash-Next-abliterated-v2-MLX-mixed-4_8bit \
  --prompt "Explain mixture-of-experts routing in two paragraphs." --max-tokens 512
```

## Abliteration

Abliteration removes the direction in the model's activations that is associated with refusals, so this model answers requests that the original model declines. It is not otherwise trained or aligned further. You are responsible for how you use it and for complying with the license and applicable law.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-Flash-Next and its license.
- [huihui-ai](https://huggingface.co/huihui-ai) for the abliterated v2 model.
- [PipeNetwork](https://github.com/PipeNetwork/qwen38-flash-next-mlx) for the parity fixes, the quantization study and the evaluation windows.
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for the `qwen4_exp` implementation, the row-store n-gram table and the MTP splitter.

License: [Qwen Community License 1.0](LICENSE), as for the base model.
