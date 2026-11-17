---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
tags:
- mlx
- macaq
- mxfp4
- 4-bit
- quantized
datasets:
- Salesforce/wikitext
- allenai/c4
---

# Qwen3.8-27B — MACaQ W4A16 (MXFP4) for MLX

A 4-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made with **MACaQ**
([arXiv 2611.10649](https://arxiv.org/abs/2611.10649)), a calibrated MXFP4 post-training quantization method.
It was produced with [macaq-mlx](https://github.com/rgarcia-fx/macaq-mlx), an independent native-MLX implementation.

- **Format:** MXFP4 weights (E2M1 values with an E8M0 scale per 32 weights) and **bf16 activations** (W4A16). It runs on MLX's stock `quantized_matmul(mode="mxfp4")`.
- **Size:** **18.1 GB**, versus 54.7 GB for bf16. All 400 decoder linear layers are 4.25 bits per weight; embeddings, lm_head and norms stay bf16.
- **Text-only:** the base checkpoint's vision tower is not included.

## Quality

Measured against the bf16 model with the same harness and the same eval tokens for every row.

| Model | Size | KLD WikiText-2 ↓ | KLD C4 ↓ | Top-1 agreement WikiText-2 / C4 | PPL WikiText-2 | PPL C4 |
|---|---|---|---|---|---|---|
| bf16 | 54.7 GB | 0 | 0 | 1 / 1 | 7.0532 | 9.4876 |
| **This model** | **18.1 GB** | **0.0277** | **0.0306** | **0.926 / 0.929** | **7.1565** | **9.6032** |
| `mlx_lm.convert -q` (affine 4-bit, group size 64) | ≈18.9 GB | 0.0402 | 0.0401 | 0.909 / 0.918 | 7.1690 | 9.7063 |
| `mlx_lm.convert -q --q-mode mxfp4` | ≈18.1 GB | 0.0606 | 0.0574 | 0.888 / 0.891 | 7.2023 | 9.8595 |

- **KLD** is the mean token-level KL(bf16 ‖ quantized) over the full 248k-token vocabulary, on 16×512 positions each of WikiText-2 test and C4 validation.
- **PPL** is over the full WikiText-2 test set and 128 random 2048-token C4 validation windows.
- **Calibration was on the train splits of the same two corpora,** so the evals are held-out but in-domain.

## Usage

The weights are MXFP4 codes of a *transformed* weight matrix, so they need the small loader included here
(`macaq_model.py`, which depends only on `mlx` and `mlx-lm`). It is loaded via `trust_remote_code`:

```bash
pip install -U mlx-lm
mlx_lm.generate --model mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx --trust-remote-code \
  --prompt "Explain why the sky is blue." --max-tokens 256
mlx_lm.chat   --model mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx --trust-remote-code
mlx_lm.server --model mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx --trust-remote-code
```

```python
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Qwen3.8-27B-MACaQ-W4A16-mlx", trust_remote_code=True)
messages = [{"role": "user", "content": "Write a haiku about quantization."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt, max_tokens=256))
```

Loading without `trust_remote_code` fails **on purpose**. A plain mxfp4 load would drop the per-channel scale
and rotation, and the output would be garbage. Please read `macaq_model.py` before trusting it; it's about 100 lines.

**Speed:** about 26–31 tok/s greedy decode on an M5 Max (bf16: about 9 tok/s), with 23.6 GB peak memory. The online transform adds 3–9% per linear layer at batch 1 compared with plain mxfp4.

## How it was made

Each linear $y = x W^\top$ is rewritten as $y = \big((x \odot t) B\big) \big((W \oslash t) B\big)^\top$:
- $t$ is a per-input-channel scale.
- $B$ is a block-diagonal normalized 32×32 Hadamard matrix.
- Only $(W \oslash t) B$ is quantized; $x \odot t$ and $B$ are applied online.

MXFP4 codes are then chosen to minimize a layer-wise Hessian-weighted output error, in two steps:
1. **Exponent screen** over $e_0 + \lbrace +1, 0, -1 \rbrace$ per 32-block, where $e_0 = \lfloor \log_2 \mathrm{amax} \rfloor - 2$ is the standard MXFP4 exponent.
2. **Flip-Refinement**, coordinate descent over $\pm 1$ code moves.

| Setting | Value |
|---|---|
| Channel scale | $t_j = \sigma_j(W)^{\alpha} / \mathrm{rms}_j(x)^{1-\alpha}$, **$\alpha = 1.0$**, clamped to $[10^{-4}, 10^{4}]$, stored as bf16 |
| Rotation | 32-block Hadamard |
| Ridge | $\lambda = 1.0 \cdot \mathrm{mean}(\mathrm{diag}(H))$ |
| Flip-Refinement | $K = 512$ iterations per row |
| Calibration | 262,144 tokens: 64 × 2048 WikiText-2 train plus 64 × 2048 C4 train (shard 0, shuffled), interleaved. Each block was calibrated on activations of the **unquantized** model. |
| Quantized | All 400 decoder linears. Every block: MLP gate/up/down. Full-attention blocks: q/k/v/o. Gated-DeltaNet blocks: in_proj_qkv/in_proj_z/out_proj. |
| bf16 | Embeddings, lm_head, norms, GDN conv1d and gating parameters, 48-wide `in_proj_a`/`in_proj_b` |
| Hardware / time | Apple M5 Max (128 GB): about 24 min calibration plus 23 min quantization |

`macaq_provenance.json` records the full recipe, the calibration token-id sha256, sha256s of the base-model and checkpoint shards and the code files, and the software versions (MLX 0.32.4, mlx-lm 0.32.2).

### Differences from the MACaQ paper

- **W4A16, not W4A4.** The paper also quantizes activations, for FP4 tensor cores. Apple GPUs have no FP4 matmul, so that would only add error. In our emulation it more than doubled KLD.
- **$\alpha = 1.0$ rather than 0.75,** since only weight balance matters without activation quantization.
- **Per-block calibration on bf16 activations,** not sequential calibration on the quantized model.
- **Rounding and flip threshold:** round-half-to-even E2M1 rounding (MLX's convention), and a relative flip-acceptance threshold.

The [macaq-mlx write-up](https://github.com/rgarcia-fx/macaq-mlx/blob/main/docs/WRITEUP.md) has the ablations behind these choices, the W4A4 results and the mixed-precision study.

## Limitations

- **Text only.**
- **Calibration domain:** only English Wikipedia and web text. Code, chat and multilingual quality were not measured beyond a short greedy-generation check (6 prompts, all correct).
- **No downstream benchmarks yet.** KLD differences under about 3% are within eval noise.

## License and citation

- **Model:** Apache-2.0, same as the base model.
- **Calibration data:** WikiText-2 (CC BY-SA 3.0) and C4 (ODC-BY), train splits only.

```bibtex
@misc{muller2026macaq,
  title         = {{MACaQ}: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations},
  author        = {Muller, Lorenz K. and Bich, Philippe and Boretti, Chiara and Chang, Hyun-Min and Zhuang, Jiawei and Cavigelli, Lukas},
  year          = {2026},
  eprint        = {2611.10649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.10649}
}
```
