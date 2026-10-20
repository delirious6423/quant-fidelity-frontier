---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
language: [en, ar, zh, fr, de, ja, ko, es, pt, it, vi, th, id, hi, ru, pl]
tags: [liquid, lfm2.5, lfm2_moe, moe, mlx, mlx-lm, apple-silicon, 4-bit]
---

# LFM2.5-24B-A2B — MLX (Q4_K_S weights, repacked)

An MLX build of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for Apple silicon, made
from [bartowski's imatrix Q4_K_S GGUF](https://huggingface.co/bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF). The quantized
blocks were **repacked into MLX's format, not re-quantized**: the weights are the GGUF's, up to the bf16 rounding of
the per-block scales that MLX stores. Model, tokenizer and chat template are Liquid AI's; this repository is a format
conversion only.

> **Changed files** (LFM Open License §4(b)): the `model-*.safetensors` shards, `model.safetensors.index.json` and
> `config.json` (which adds MLX's `quantization` map) were produced by this conversion. `tokenizer.json`,
> `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` and `LICENSE` are Liquid AI's files,
> unchanged.

## Requirements

- **mlx-lm ≥ 0.32.0** (it pulls mlx ≥ 0.32.2). Older mlx-lm releases (≤ 0.31.3) route this model's experts with a
  softmax instead of its sigmoid gating: it still writes fluent text, but it is not computing what it was trained to
  compute (see [Known routing bug](#known-routing-bug-in-other-mlx-runtimes)).
- About 16 GB of unified memory (peak 15.9 GB in the test below): 32 GB Macs or larger are comfortable, 24 GB is tight.

## Usage

```bash
pip install -U "mlx-lm>=0.32.0"
mlx_lm.generate --model notfakerv/LFM2.5-24B-A2B-MLX-Q4_K_S --prompt "Why is the sky blue?" --max-tokens 1024
```

```python
from mlx_lm import load, generate

model, tokenizer = load("notfakerv/LFM2.5-24B-A2B-MLX-Q4_K_S")
messages = [{"role": "user", "content": "Why is the sky blue?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024))
```

The model reasons in a `<think>` block before it answers. Liquid AI's `generation_config.json` sets temperature 0.1,
top_k 50 and repetition_penalty 1.05.

## What is in the files

| Tensors | Source GGUF | MLX here |
| --- | --- | --- |
| Routed experts, layers 2–38 | Q4_K | affine 4-bit, group 32 (scale `d·sc`, bias `−dmin·m`) |
| Attention, short-conv in/out projections, dense FFN (layers 0–1), token embedding (tied LM head) | Q8_0 | affine 8-bit, group 32 (`q+128`, bias `−128·d`) |
| Layer-39 experts: up and down / gate | Q8_0 / Q6_K | 8-bit, group 32; the Q6_K tensor is re-quantized (MLX has no group size 16) |
| Router, RMSNorm weights, short-conv taps | F32 holding bf16 values | bf16, unquantized (exact) |
| Expert bias | F32 | fp32 |

14.7 GiB (15.8 GB) in three shards.

## Fidelity checks

- Every quantized tensor was compared with gguf-py's dequantization of the source file. All 221 lie inside the bound
  implied by bf16 rounding of each block's scale and bias (worst case 0.31 of the bound); a layout error would show up
  at the size of the weights themselves. Each block's bf16 scale/bias pair is chosen by least squares over the
  neighbouring bf16 values. Residual error (rms), relative to the source format's own quantization noise: 4.4 % for
  the Q4_K experts, about 28 % for the Q8_0 tensors (bf16-level precision of the weights), 34 % for the single
  re-quantized Q6_K tensor. With fp32 scales the repack reproduces the GGUF values bit for bit.
- Against an independent fp32 NumPy forward pass of the GGUF (Hugging Face `Lfm2Moe` semantics) on a 31-token chat
  prompt: top-1 agreement 84 % over prompt positions and 100 % over decode positions, mean KL 0.06 nats per position.
  That is this model's bf16 noise floor: near-tied expert choices flip under tiny perturbations. In fp32, MLX gives
  identical results for any prefill chunking; in bf16, chunking alone moved the first-token top log-probability by up
  to 0.13.
- Perplexity on 8,191 tokens of English technical text: 5.956 with bf16 activations, 5.990 with fp32 activations, so
  running in bf16 costs nothing measurable.

## Speed (Apple M4 Max)

- mlx-lm 0.32.0, greedy decoding: 149 tokens/s.
- LM Studio 0.4.26 (MLX runtime 1.13.1, one parallel slot): 148 tokens/s decode, about 2,150 tokens/s prompt
  processing on a 17.6k-token prompt.

## Known routing bug in other MLX runtimes

LFM2-MoE picks its top-4 experts by `sigmoid(router logits) + expert_bias`, then mixes them with the **unbiased**
sigmoid scores, normalized to sum to 1. That is what Hugging Face transformers, llama.cpp and mlx-lm ≥ 0.32.0 do.
As of 2026-10-15 the following route with a softmax and add the expert bias into the mixing weights instead:

- mlx-lm ≤ 0.31.3;
- mlx-vlm up to at least 0.7.8 (`mlx_vlm/models/lfm2/language.py`, which `lfm2_moe` reuses);
- LM Studio's MLX runtime 1.13.1 whenever the model is loaded with more than one parallel slot (the default), because
  that path runs LM Studio's mlx-vlm fork. Load it with one slot (`lms load <model> --parallel 1`, or Max Concurrent
  Predictions = 1) to get the correct mlx-lm path.

Measured on this model against the fp32 reference: mean KL 1.14 nats per token with the softmax routing, 0.06 with
the correct routing.

## Provenance

- Base model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), revision
  `1769516ac588c4c558089c94f27d96b0dc67b183` (tokenizer, chat template, generation config and license taken from it).
- Source quantization: [bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF](https://huggingface.co/bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF),
  `LiquidAI_LFM2.5-24B-A2B-Q4_K_S.gguf` (sha256 `8e5362ed49c2dbeb8210732b3e1727e06587a2cdb0e1ee0e5d7f5a437608a286`,
  repo revision `dea419016c4d1079f0c014867df9df688dcb6188`), importance matrix from bartowski's calibration set v6.
- Conversion: a custom GGUF-to-MLX repacker, 2026-10-15.

## License

[LFM Open License v1.0](LICENSE). Note its Section 5: commercial use is licensed only to entities with less than
US$10 million in annual revenue. All credit for the model goes to Liquid AI, and for the quantization to bartowski.
