---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: mlx
tags:
- mlx
- qwen3_5
- awq
- imatrix
- mtp
---

# Qwen3.8-27B-UD-Q5_K_XL-AWQ-MTP-mlx

`Q5_K_XL` quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for
[MLX](https://github.com/ml-explore/mlx) on Apple silicon, with AWQ imatrix
pre-scaling and the **MTP head kept inline**.

`AWQ` here means activation-aware **pre-scaling applied before** MLX affine
quantization — the same sense as `mlx-community/*-4bit-AWQ`. The stored format is
standard MLX affine, **not** AutoAWQ/GPTQ packing, so this will not load in vLLM or
`autoawq`. `MTP` means the multi-token-prediction head ships inside the main shards
rather than as a separate drafter.

| | |
|---|---|
| On disk | **24.65 GB** |
| **Effective bits per weight** | **7.10 bpw** |
| Parameters | 27.78 B |
| Tensors / shards | 2195 / 5 |
| Base quant | 5-bit affine, group size 64 |
| Per-layer overrides | 370 |
| MTP tensors | 15 (inline, BF16, unquantized) |

## Read the BPW, not the name

**This model is 7.10 bpw, not 5 bpw.** The `Q5_K_XL` label follows the
Unsloth naming convention, where `--q-bits` sets the *base* class of a mixed-precision
recipe rather than the average. Higher-precision classes then sit above that base and
dominate the file. The real distribution:

| Class | GB | Share | bpw |
|---|---:|---:|---:|
| 8-bit quantized                  |     9.78 |  39.7% | 8.00 |
| 5-bit quantized                  |     7.13 |  28.9% | 5.00 |
| 6-bit quantized                  |     4.28 |  17.4% | 6.00 |
| quant sidecars (scales + biases) |     1.68 |   6.8% | — |
| embeddings / head / norms (BF16) |     0.93 |   3.8% | 16.00 |
| MTP head (BF16, unquantized)     |     0.85 |   3.4% | 16.00 |

Sidecars are the per-group scales and biases required by affine quantization at
group size 64; they carry no parameters but do occupy space.

### Which weight gets which width

Every quantized tensor, by module and resolved bit width (370 explicit
per-layer overrides; anything not overridden inherits the 5-bit top-level default):

| Module | Scope | Width |
|---|---|---|
| `embed_tokens` | single tensor | **8-bit** |
| `linear_attn.in_proj_a` | 48 layers | **8-bit** |
| `linear_attn.in_proj_b` | 48 layers | **8-bit** |
| `linear_attn.in_proj_qkv` | 48 layers | **8-bit** |
| `linear_attn.in_proj_z` | 48 layers | **8-bit** |
| `linear_attn.out_proj` | 48 layers | **8-bit** |
| `lm_head` | single tensor | **8-bit** |
| `self_attn.k_proj` | 16 layers | **8-bit** |
| `self_attn.o_proj` | 16 layers | **8-bit** |
| `self_attn.q_proj` | 16 layers | **8-bit** |
| `self_attn.v_proj` | 16 layers | **8-bit** |
| `mlp.down_proj` | 64 layers | **6-bit** |
| `mlp.gate_proj` | 64 layers | **5-bit** *(top-level default)* |
| `mlp.up_proj` | 64 layers | **5-bit** *(top-level default)* |
| RMSNorms, biases, attention sinks, `mtp.*` | — | BF16 (not quantized) |

Attention `o_proj` and GDN `out_proj` are pinned high because AWQ cannot correct them
(no preceding norm to fold into), so they carry uncompensated quantization error and are
given extra width instead.

Size it against your RAM using the **24.65 GB** figure plus KV-cache headroom,
not against what `Q5_K_XL` implies elsewhere.

## Sources

| Component | Value |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Converter | [mlx-node](https://github.com/mlx-node/mlx-node) v0.0.10 + bounded-memory AWQ patch |
| Quantization recipe | `unsloth` — mlx-node's port of the Unsloth Dynamic 2.0 class map (legacy affine) |
| Recipe reference | [Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx](https://huggingface.co/Brooooooklyn/Qwen3.6-27B-UD-Q4_K_XL-mlx) |
| Calibration imatrix | AtomicChat, via [Qwen3.8-27B discussion #65](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65) ([collection](https://huggingface.co/collections/AtomicChat/qwen-38-27b-6a7c86fe00e317b78f572767)) |
| imatrix details | `qwen3.8-27b-atomicchat-imatrix.gguf` — GGUF v3, 10,027 chunks x 512 tokens |
| AWQ ratio | 0.5 |

The `unsloth` recipe is mlx-node's reimplementation of Unsloth's per-layer class map.
It is **not** an Unsloth-produced artifact and carries none of Unsloth's calibrated
NVFP4 global scales, W4A4/W8A8 activation execution, or calibrated FP8 KV-cache scales.
No numerical or performance parity with upstream Unsloth is claimed.

The imatrix was produced by AtomicChat and published alongside their Qwen3.8-27B
quantization comparison; its author describes it as calibrated on their public corpora.
Inside the GGUF the corpus is recorded only as the producer's local path
(`/eval/calib_train.txt`), so the exact composition is not asserted here — see the
linked discussion and collection for their methodology.

## How it was built

```bash
mlx convert \
  --input  <Qwen/Qwen3.8-27B snapshot> \
  --output Qwen3.8-27B-UD-Q5_K_XL-AWQ-MTP-mlx \
  --model-type qwen3_5 \
  --quantize \
  --q-bits 5 \
  --q-group-size 64 \
  --q-recipe unsloth \
  --imatrix-path qwen3.8-27b-atomicchat-imatrix.gguf \
  --q-mtp off
```

`--q-mtp off` retains the MTP tensors **inline in the main shards, unquantized**. There
is no `mtp-drafter/` directory and no `mtp.safetensors` sidecar; loaders that expect a
split drafter will not find one here.

AWQ pre-scaling amplifies important weight columns using the imatrix and folds the
inverse into the preceding norm, so the reparametrization is output-preserving. It runs
on MLP gate/up/down, attention q/k/v, and GatedDeltaNet `in_proj_*`. Attention `o_proj`
and GDN `out_proj` are deliberately excluded — their inputs come from attention output,
not from a norm, so there is no norm to fold into.

## Architecture

| | |
|---|---|
| Model type | `qwen3_5_text` |
| Hidden size | 5120 |
| Intermediate size | 17408 |
| Layers | 64 (48 GatedDeltaNet + 16 full-attention, interval 4) |
| Attention heads | 24 (KV 4, head dim 256) |
| Vocab | 248,320 |
| Max positions | 262,144 |
| Tied embeddings | False |

## Usage

Requires an MLX runtime with `qwen3_5` support (hybrid GatedDeltaNet + full attention).
Stock `mlx-lm` does not implement this architecture.

```bash
mlx chat --model Qwen3.8-27B-UD-Q5_K_XL-AWQ-MTP-mlx
```

## Performance

Measured on **Apple M3 Max (30-core GPU)** under oMLX, tokens/sec:

| Context | Prompt processing | Token generation |
|---|---:|---:|
| 1k | 157.0 | **18.2** |
| 4k | 160.1 | **17.6** |

All three variants, same machine:

| Variant | GB | bpw | TG @1k | TG @4k |
|---|---:|---:|---:|---:|
| Q4_K_XL | 20.89 | 6.02 | 19.6 | 16.7 |
| Q5_K_XL | 24.65 | 7.10 | 18.2 | 17.6 |
| Q6_K_XL | 27.50 | 7.92 | 16.3 | 14.0 |

Prompt processing is compute-bound and sits at 154–160 tok/s regardless of quant width.
Token generation is memory-bandwidth-bound and orders by file size at 1k. At 4k the Q4
and Q5 figures cross over; these are single runs taken about a minute apart on one
machine, so differences of ~1 tok/s are within run-to-run and thermal noise and should
not be read as a real ordering.

## Verification status

Structure, quantization metadata, and BPW in this card were measured directly from the
safetensors headers and `config.json` of these files. The model loads and generates —
the throughput above is from a real run. **Output quality has not been benchmarked**:
no perplexity, and no downstream eval against the BF16 base or other quantizations.
