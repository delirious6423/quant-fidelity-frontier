---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- compressed-tensors
- vllm
---

# Qwen3.8-27B-W2.2A16-HGPTQ-RCO

Ultra-compact mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.2A16`, group size 128, `H_512` Walsh–Hadamard rotation, symmetric mid-riser `code_offset=0.5` 2-bit grid) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built with Riemannian Co-design (`HGPTQ-RCO`). It compresses the 51.8 GiB BF16 model down to **11.06 GiB** (`7.00 GiB` pure text backbone; 4.68× smaller; `10.59 GiB` text-only) for ultra-long-context **16 GB GPU** serving or headless **12 GB GPU** text-only serving (`--language-model-only`).

- **VRAM & GPU fit:** `11.06 GiB` full VLM+MTP (`10.59 GiB` text-only; `7.00 GiB` pure text backbone, saving `40.69 GiB` vs. BF16); for general 16 GB multimodal + MTP serving, prefer [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) at `12.20 GiB`.
- **Quantization recipe:** Upgrades `103 / 400` backbone linears (`16.5%` parameter budget, fused `vLLM` groups) to `W4_g128` (`H_512`), quantizes the remaining `297 / 400` backbone linears to symmetric mid-riser `W2_g128` (`code_offset=0.5`, codes `{-1.5, -0.5, +0.5, +1.5}`), keeps Vision and Native MTP transformer blocks in `W4_g64` (`lm_head` and `mtp.fc` in `W8_g64`), and polishes `W2` scales + FP32-master `RMSNorm`/GDN gates from clean GPTQ weights via end-to-end KL QAT.
- **Fidelity vs. BF16:** `0.3187` weighted KL (`0.4900` mean KL) on `probe-v4`, `0.3809` mean KL (`0.5330` weighted KL) on the `probe-guard` holdout, `96.84%` / `69.84%` isolated/E2E MTP top-1 agreement, and `96.69%` / `88.08%` isolated/E2E vision top-1 agreement (`99.26%` / `86.67%` on `vision-guard-v2`).
- **Downstream EvalScope (Full `N=1,205`, greedy `temp=0.0`, `enable_thinking=False`):** **`78.00%` 3-benchmark mean** (`72.09%` IFEval prompt-strict / `80.20%` MATH-500 / `81.71%` HumanEval+) in strict `2/4/8`-bit at `7.00 GiB` pure text backbone.

| Variant | Backbone W4 Budget | Pure Text Backbone | Full VLM+MTP Size | Text-Only Size | Target GPU Tier |
|---|---:|---:|---:|---:|---|
| **`Qwen3.8-27B-W2.2A16-HGPTQ-RCO` (this repo)** | **`16.5%` `W4` (`103/400`)** | **`7.00 GiB`** | **`11.06 GiB`** | **`10.59 GiB`** | **Headless `12 GB` (text-only) / ultra-long-context `16 GB`** |
| [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) | `36.75%` `W4` (`177/400`) | `8.15 GiB` | `12.20 GiB` | `11.74 GiB` | Recommended `16 GB` GPU (full VLM + MTP + CUDA graphs) |
| [`Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) | `100%` `W4` (`400/400`) | `12.08 GiB` | `16.14 GiB` | `15.67 GiB` | `24 GB` GPU (RTX 3090 / 4090 / 5090) |

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-27B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-27B`) | **`Qwen3.8-27B-W2.2A16-HGPTQ-RCO`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,563,006,776 B`) | **`11.06 GiB` (`11,870,854,952 B`)** | **`4.68×` compression** (`7.00 GiB` pure text backbone; `10.59 GiB` text-only) |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.3187` / `0.4900`** | 378 prompts, 852k tokens (PPL `8.5332 → 4.1388`; `math` `0.2089`, `chat` `0.2031`, `if` `0.1948`, `science` `0.2080`) |
| **General & Multilingual Holdout (`probe-guard`)** (token mean KL / weighted KL) ↓ | `0.0000` | **`0.3809` / `0.5330`** | 76 prompts, 23k tokens (PPL `61.0805 → 36.6476`; `no_robots` `0.2572`, 9-lang Belebele macro KL `0.5330`) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.9579`) | **`96.84%` Isolated Top-1 (`NLL 2.9741`)** | Isolated MTP KL `0.0087` (end-to-end Top-1 `69.84%`, KL `0.7578`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`96.69%` Isolated Top-1 · `88.08%` E2E Top-1** | Isolated ViT KL `0.0067`, E2E KL `0.4552` (`vision-guard-v2` Iso Top-1 `99.26%`, E2E Top-1 `86.67%`, KL `0.5523`) |

### End-to-End EvalScope Benchmarks (`N=1,205`, greedy `temp=0.0`, `enable_thinking=False`)

| Model / Checkpoint | Text Backbone (`bpw`) | IFEval (`541`) | MATH-500 (`500`) | HumanEval+ (`164`) | **3-Bench Mean** |
|---|---:|---:|---:|---:|---:|
| BF16 Base (`Qwen/Qwen3.8-27B`) | `45.36 GiB` (`16.00`) | `82.62%` | `84.80%` | `93.29%` | **`86.90%`** |
| **`W2.2A16-HGPTQ-RCO` V2 (strict `2/4/8`-bit, this repo)** | **`7.00 GiB` (`2.47`)** | **`72.09%`** | **`80.20%`** | **`81.71%`** | **`78.00%`** |

## What Is Quantized

- **Quantized (`519` linear layers total)**:
  - **Language Backbone (`400` linears across `256` fused `vLLM` groups, `H_512`, `group_size=128`)**:
    - **`103` linears in `W4A16-HGPTQ` (`w4_g128`)**: Top `16.5%` parameter budget selected via non-myopic Gumbel-Softmax Riemannian Co-design, calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`297` linears in `W2A16-HGPTQ` (`w2_g128_sym`)**: Remaining backbone groups quantized with symmetric mid-riser 2-bit (`code_offset=0.5`, codes `{-1.5, -0.5, +0.5, +1.5}`) `H_512` Cholesky GPTQ + 3-pass `H-CD` and polished end-to-end from clean GPTQ weights via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 5120]`) and `mtp.fc` (`[5120, 10240]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`).
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`, `group_size=64`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj`.
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel Cholesky GPTQ + `head_dim=16` `v_vision_fc2`; other Vision linears use `group_size=64` + `head_dim=128`).
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Zero-patch `vLLM` + Hugging Face `compressed-tensors` compatibility**:
  - Uses standard `pack-quantized` int32 shards with `w2_g128_sym` (`num_bits: 2`, `code_offset: 0.5`), `w4_g128` / `group_0` (`num_bits: 4`), `vision_fc2_channel` (`num_bits: 4`), and `w8_head` (`num_bits: 8`) config groups, plus bundled `vllm_warp_quant.py` auto-loader.
  - Serve with `VLLM_WARP_QUANT_W2_MARLIN=1` for full-speed Machete/Marlin CUDA-graph kernels (lossless `uint4b8` repack of the `code_offset=0.5` grid), or default mode for minimum true 2-bit VRAM (`11.06 GiB`).
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone linears use `head_dim=512` Sylvester Walsh–Hadamard rotation (`v`); Vision and MTP use compatible block sizes (`v_vision`, `v_vision_fc2`).
