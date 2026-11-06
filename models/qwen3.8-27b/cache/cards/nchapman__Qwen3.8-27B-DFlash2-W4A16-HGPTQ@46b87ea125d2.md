---
license: apache-2.0
library_name: compressed-tensors
pipeline_tag: text-generation
base_model:
  - incoai/Qwen3.8-27B-DFlash2
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - dflash2
  - speculative-decoding
  - block-diffusion
  - draft-model
  - compressed-tensors
  - w4a16
  - gptq
  - hadamard
---

# Qwen3.8-27B-DFlash2-W4A16-HGPTQ

4-bit (`W4A16`, group size 64) **Hadamard + Cholesky GPTQ (`HGPTQ`)** quantization of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (the 5-layer block-diffusion speculative drafter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and [`nchapman/Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ)).

## Summary & RTX 3090 Impact

In BF16, `incoai/Qwen3.8-27B-DFlash2` is **3,670.5 MiB (`3.58 GiB`)** and streams **3.18 GiB** of decoder-layer linear weights on every 8-token draft step. Applying deterministic block-64 Sylvester Walsh-Hadamard rotation (`H_64`) + `actorder="static"` Cholesky GPTQ (`group_size=64`) across all 45 decoder-layer linears (`layers.0..4.{self_attn,mlp,attention_conv,mlp_conv}`) while keeping the 5-tap conditioning projection (`fc.weight`) and sparse bilinear `candidate_selector` in BF16:

- **Shrinks the drafter by 2.74×**: **3,670.5 MiB (`3.58 GiB`) → 1,338.9 MiB (`1.31 GiB`)**, freeing **2.27 GiB of VRAM** for KV cache on a 24 GB GPU.
- **Cuts draft-step memory bandwidth by 3.76×**: Streams **843 MiB** of W4 decoder-layer weights instead of **3,175 MiB** of BF16 weights (`~0.9 ms` vs `~3.4 ms` memory-bandwidth floor per draft step on an RTX 3090).
- **Preserves 100% of BF16 Top-16 Candidate Recall (`53.88%` vs `53.88%`) and 99.8% of Block Acceptance Length $\tau$ (`2.535` vs `2.540`)**.

## Evaluation (`probe-v4`, 2,048 8-Token Draft Blocks = 14,336 Drafted Tokens)

| `Qwen3.8-27B-DFlash2` Arm | Checkpoint Size | Unary KL vs BF16 ↓ | Median KL ↓ | Tail `p99` KL ↓ | Unary Top-1 vs BF16 ↑ | Selector Top-1 vs BF16 ↑ | Verifier Accept $\tau$ ↑ | Recall@16 ↑ | Pos-1 / Pos-2 / Pos-3 Acc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `BF16_Reference` ([`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)) | 3,670.5 MiB | 0.00000 | 0.00000 | 0.0000 | 100.00% | 100.00% | **2.540** | **53.88%** | 56.0% / 38.5% / 29.4% |
| `W4_RTN g64-h0` (unrotated RTN) | 1,338.9 MiB | 0.05452 | 0.02412 | 0.4949 | 88.57% | 87.60% | 2.543 | 53.66% | 55.7% / 38.4% / 29.1% |
| `W4_RTN g64-h64` (data-free Hadamard RTN) | 1,338.9 MiB | 0.08127 | 0.05444 | 0.5160 | 88.51% | 87.44% | 2.518 | 53.52% | 56.0% / 38.9% / 29.4% |
| **`W4A16-HGPTQ` (this checkpoint, 45 linears)** | **1,338.9 MiB** | **0.02309 (−57.6%)** | **0.01074** | **0.1876 (−62.1%)** | **92.63%** | **92.03%** | **2.535** | **53.88%** | **56.5% / 38.9% / 29.4%** |

## Tensor Allocation

- **Quantized (`W4A16-HGPTQ`, `group_size=64`, `hadamard_block_size=64`, `actorder="static"`, 45 Linear layers)**:
  - `layers.{0..4}.self_attn.{q_proj,k_proj,v_proj,o_proj}` (20 matrices; `k_proj` and `v_proj` calibrated jointly over both fused 5-tap `target_hidden` context states and draft-block `hidden_states`)
  - `layers.{0..4}.mlp.{gate_proj,up_proj,down_proj}` (15 matrices)
  - `layers.{0..4}.{attention_conv,mlp_conv}.kernel_projection` (10 matrices)
- **Preserved in BF16**:
  - `fc.weight` (`[5120, 25600]`, 5-tap target hidden state fusion projection)
  - `candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection.weight}`
  - `layers.{0..4}.{attention_conv,mlp_conv}.base_kernel` (`[2, 2, 5120]`) and all RMSNorm weights (`hidden_norm`, `norm`, `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`)

## Attribution & Citation

Derived from [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0). Please cite the original DFlash 2 and DFlash works:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
