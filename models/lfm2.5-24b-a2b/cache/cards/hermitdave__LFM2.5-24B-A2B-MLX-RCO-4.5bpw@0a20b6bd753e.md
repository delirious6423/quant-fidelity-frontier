---
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
tags:
- mlx
- apple-silicon
- text-generation
- lfm2_moe
- moe
- quantization
- rco
- mixed-precision
- 4-bit
---

# LFM2.5-24B-A2B-MLX-RCO-4.5bpw

**Upstream model**: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)

[Liquid AI](https://liquid.ai)'s LFM2.5-24B-A2B (24 B total parameters, ~2 B
active per token) converted to MLX and quantized with **RCO** —
rate-constrained optimization, a budget-constrained mixed-precision allocator
([ISTA-DASLab/RCO](https://github.com/ISTA-DASLab/RCO), arXiv:2605.00649) —
ported to MLX by the `rco-mlx` pipeline. The bit budget is spent where the
measured per-tensor reconstruction error says it matters, under an exact
bits-per-weight constraint rather than a penalty.

## Architecture

| | |
|---|---|
| Total / active parameters | 23.97 B / ~2 B |
| Layers | 40 (10 full-attention, 30 short-conv) |
| Hidden size | 2048 |
| Attention | 32 heads / 8 KV heads |
| MoE | 64 experts, top-4, moe_intermediate_size 1536 |
| Dense layers | 2 |
| Vocab / context | 128000 / 131072 |
| Routing | sigmoid + expert_bias, norm_topk_prob |
| License | LFM Open License v1.0 (see `LICENSE`) |

## Quantization

| | |
|---|---|
| Method | RCO mixed-precision (streaming RTN error DB + budget DP) |
| Format | MLX affine, group size 64, scales/biases **bfloat16** |
| Default width | 4-bit |
| Per-tensor overrides | 220 entries |
| Routed experts | 4-bit x101, 6-bit x13 |
| Other quantized tensors | 8-bit x106 |
| Kept BF16 | embeddings (`embed_tokens`, tied output head), norms, MoE router gate, expert bias, depthwise conv |
| On-disk size | **14.90 GB** |
| Effective bpw | **4.972** (over 23.97 B source params) |

No public imatrix exists for this checkpoint, so the allocation uses the plain
RTN error database; an imatrix-weighted build would be the next quality step.

## Validation

WikiText-2 test (BOS-prefixed rows; the tokenizer omits `<|startoftext|>`,
which degenerates this model without it). KL and top-1 are measured against the
BF16 reference on identical 32x512 rows; PPL is paired per row on 256x512
(130,816 tokens).

| Metric | BF16 | This quant |
|---|---|---|
| KL(ref‖quant) ↓ | — | **0.1532 ± 0.0056** |
| Top-1 agreement ↑ | — | **82.0 %** |
| Perplexity ↓ | 20.61 | **20.82 ± 0.55** |

Zero-shot task recovery (5 tasks x 400 examples, identical prompts, paired 2x2):

| Task | BF16 acc | Quant acc | Recovery |
|---|---|---|---|
| arc_challenge | 0.4400 | 0.4450 | 1.011 |
| arc_easy | 0.7400 | 0.7050 | 0.953 |
| hellaswag | 0.3975 | 0.3975 | 1.000 |
| winogrande | 0.4975 | 0.4800 | 0.965 |
| mmlu | 0.4075 | 0.4275 | 1.049 |
| **Mean recovery** | | | **0.9956 (TASK-LOSSLESS)** |

Throughput (M3 Max 64 GB, mlx-lm, ~1K prompt, 256 decode tokens, >=2 runs):
prefill 990.5 tok/s, decode **90.9 tok/s** (BF16 reference: 665.9 / 53.7).

Peak memory (mlx-lm, M3 Max 64 GB): **17.1 GB** (BF16 reference: 50.1 GB).
Savings are in storage and I/O; MLX dequantizes at runtime.

Validated with `mlx-lm` 0.32.2 / MLX 0.32.3 on macOS (Apple Silicon). Mixed-bit
checkpoints use per-module widths from `config.json["quantization"]`; any
mlx-lm that loads `lfm2_moe` will honour them.

## Usage

```bash
python3 -m mlx_lm generate \
  --model hermitdave/LFM2.5-24B-A2B-MLX-RCO-4.5bpw \
  --prompt "Explain why the sky is blue in two sentences."
```

```python
from mlx_lm import load, generate
model, tokenizer = load("hermitdave/LFM2.5-24B-A2B-MLX-RCO-4.5bpw")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt, max_tokens=128))
```

## Citation

```bibtex
@misc{liquidai_lfm25_24b_a2b,
  title  = {LFM2.5-24B-A2B},
  author = {Liquid AI},
  url    = {https://huggingface.co/LiquidAI/LFM2.5-24B-A2B}
}
```

## Attribution

This MLX quantization was prepared by **Hermit Dave**, produced with
**Hermes Agent** (Nous Research). RCO is © ISTA-DASLab (Michael Helcig,
Dan Alistarh), Apache-2.0; the optimizer, manifold primitives and grouping logic
in the port follow [ISTA-DASLab/RCO](https://github.com/ISTA-DASLab/RCO).
