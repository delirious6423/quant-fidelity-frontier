---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: 6block
license: other
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen4exp
- moe
- quantized
- layered-quantization
---

# Qwen3.8-Flash-Next GGUF

GGUF quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), made with llama.cpp.

Chinese version: [README_zh.md](https://huggingface.co/6block/Qwen3.8-Flash-Next-GGUF/blob/main/README_zh.md)

180B total parameters. A MoE core of 125B (48 layers, 512 experts, top-10 plus one
shared expert) plus a 51B **PLE** n-gram lookup table. Hybrid attention: Gated
DeltaNet linear attention interleaved with Qwen Sparse Attention (QSA) blocks,
wrapped in hyper-connections. Context length up to 262K. The one MTP draft layer
in the checkpoint is excluded at conversion time.

Quantized from the official **BF16** weights. Every tier is imatrix-calibrated and
ships as 8 shards.

## Quantizations

| Quant | Size | Shards | BPW | PPL (wikitext-2) |
|---|---|---|---|---|
| BF16 master | 329.7 GiB | 8 | 15.73 | **1.7164 ± 0.04795** |
| `IQ4_XS` | 118.8 GiB | 8 | 5.67 | 1.8378 ± 0.05554 |
| `IQ3_XXS` | 107.4 GiB | 8 | 5.13 | 2.0528 ± 0.06396 |
| `IQ2_XS-layered` | 83.5 GiB | 8 | 3.98 | 2.7833 ± 0.10058 |
| `IQ1_M-layered` | 76.4 GiB | 8 | 3.65 | 4.4105 ± 0.19322 |

The master row is not a file in this repo. It is listed so the numbers above have
a reference point — the BF16 master fits on the PPL machine, so tiers are compared
against a real baseline.

**The two low tiers use layered quantization** (`IQ2_XS-layered`, `IQ1_M-layered`):
the 51B PLE n-gram table tolerates heavy compression (measured: `Q8_0` → `Q4_0`
costs nothing), so it is packed at `Q4_0`, and the space saved is spent on the
expert gate/up projections instead — which improves PPL by ~14% at a *smaller*
file size than the uniform tier. High tiers stay uniform since they already sit
close to the master.

## What is protected

The layers that would hurt most under low-bit compression are kept at higher
precision:

| Tensors | Type | Reason |
|---|---|---|
| `ffn_gate_inp` / `ffn_gate_inp_shexp` | F32 | MoE router; compressing it routes to the wrong experts |
| `ffn_*_shexp` | Q8_0 | the shared expert runs on every token |
| `ssm_a` / `ssm_conv1d` / `ssm_dt` / `ssm_alpha` / `ssm_beta` / `ssm_norm` / `ssm_out` | F32 | Gated DeltaNet linear-attention state; low bit-width destroys long-range recall |
| `hc_attn_*` / `hc_ffn_*` / `output_hc_*` | F32 | hyper-connection streams, every layer |
| `ple_key` / `ple_value` / `ple_conv1d` | Q8_0 | PLE projection layers |
| `per_layer_token_embd` (51B n-gram table) | Q4_0 | n-gram counts, highly compression-tolerant |
| `token_embd` / `output` | Q6_K | a global type would otherwise squeeze these hard |

With 512 experts the expert layers dominate the file, so protecting everything
else is cheap.

## Usage

```bash
# Point at the first shard; llama.cpp finds the rest on its own.
llama-cli -m Qwen3.8-Flash-Next-IQ4_XS-00001-of-00008.gguf
```

Do **not** pass `-ngl` manually — llama.cpp fits layers to free VRAM by itself,
and any explicit `-ngl` aborts that fitting. Leave it off.

Every tier ships as 8 shards. Download all 8 into one directory — you only ever
name `-00001-of-00008` on the command line.


## About the PPL numbers

wikitext-2 test, `n_ctx=512`, 12 chunks, every tier through the exact same
command. **These numbers are only comparable within this table.** Do not compare
them against PPL figures published by other repos.

The BF16 master row is the measured baseline. `IQ2_XS-layered` / `IQ1_M-layered`
use layered quantization (see above) — at the same size they beat the uniform
tiers by a wide margin.

## License

[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE),
inherited from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — this is **not** MIT.
Quantized by 6block.
