---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- qwen3_5
- conversational
---


# Qwen3.8-27B — MLX, 8-bit (group size 64, affine)

This model was converted to MLX format (8-bit (group size 64, affine) quantization) from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using
[mlx-lm](https://github.com/ml-explore/mlx-lm).
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B)
for more details on the model.
Weights: ~29 GB (8.501 effective bits/weight) on disk (bf16 MLX artifact: ~54 GB; the original
checkpoint is ~55 GB including the vision encoder and MTP drafter, which MLX
conversions drop).

**This conversion is text-only.** mlx-lm's `qwen3_5` implementation strips the
base model's vision encoder and multi-token-prediction drafter; this artifact
does not accept images or videos. For vision use, run the original checkpoint
with Transformers, vLLM, or SGLang.

**Quantization changes the numerics.** The base card's benchmark numbers do not
transfer to this artifact as-is.
Measured on this build: wikitext-2 (test) perplexity **6.9447**, vs **6.9352** for the bf16 conversion in the identical harness (+0.14% relative, effectively lossless; 4-bit: 7.0871) — disjoint 2048-token windows, no overlap, single token stream. Decode ~18 tok/s on an M5 Max MacBook Pro at 28.9 GB peak memory (4-bit: ~33 tok/s at 15.5 GB; bf16: ~10 tok/s at 54 GB).

## Model description (inherited from the base card)

Qwen3.8-27B is the compact dense model of the Qwen3.8 generation, built on the
Qwen3.5 architecture: 27B parameters, 64 layers in a 3:1 hybrid of Gated
DeltaNet linear attention and gated full attention (24Q/4KV, head dim 256),
vocab 248,320, context 262,144 native. Thinking mode on by default with
`reasoning_effort` control (`xhigh` default / `medium` / `low`) and
`preserve_thinking`. See the
[base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for details and
benchmarks.

## Use with mlx-lm

```bash
pip install -U mlx-lm
mlx_lm.generate --model malekoo/Qwen3.8-27B-MLX-8bit --prompt "Explain KV caches briefly." --max-tokens 2048
```

OpenAI-compatible server: `mlx_lm.server --model malekoo/Qwen3.8-27B-MLX-8bit --port 8080`.
Thinking control via `chat_template_kwargs`: `enable_thinking`,
`preserve_thinking`, `reasoning_effort` (`xhigh`/`medium`/`low`).

## Recommended sampling (from the base card)

- Thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
  `presence_penalty=0.0`, `repetition_penalty=1.0`
- Instruct (non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`,
  `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`

## Conversion provenance

- Source: `Qwen/Qwen3.8-27B` (bf16 safetensors, released 2026-08-14)
- Quantized with `mlx_lm.convert -q --q-bits 8 --q-group-size 64`
- Toolchain: mlx `0.32.1.dev20260814+3d23f7d87`, mlx-lm at ml-explore/mlx-lm main `254d153`

## License and attribution

Apache-2.0, inherited from the base model. Copyright the Qwen team; this
repository is a quantized conversion and claims no additional rights.

## Citation

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
