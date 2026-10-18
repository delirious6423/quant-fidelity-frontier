---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: qtum
license: mit
license_link: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- glm
- moe
- quantized
---

# GLM-5.3-Flash GGUF

GGUF quantizations of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), made with llama.cpp.

Chinese version: [README_zh.md](https://huggingface.co/qtum/GLM-5.3-Flash-GGUF/blob/main/README_zh.md)

320B total parameters, 18B active. 45 layers with hybrid attention: 34 KDA
linear-attention layers interleaved with 11 DSA sparse-attention layers, built on
MLA, wrapped in Manifold-Constrained Hyper-Connections (mHC). 288 MoE experts
with top-8 routing plus a shared expert. Context length up to 1M.

Quantized from the official **BF16** weights. Every tier is imatrix-calibrated and
ships as 15 shards.

## Quantizations

| Tier | Size | Shards | BPW | PPL |
|---|---|---|---|---|
| master (BF16, not in this repo) | 583 GiB | 15 | 16.00 | **6.6974 ± 0.34871** |
| `IQ4_XS` | 155 GiB | 15 | 4.27 | 7.1537 ± 0.37538 |
| `IQ3_XXS` | 112 GiB | 15 | 3.09 | 8.2485 ± 0.41877 |
| `IQ2_XS` | 85 GiB | 15 | 2.35 | 20.2651 ± 1.10727 |
| `IQ1_M` | 65 GiB | 15 | 1.80 | 73.9234 ± 4.73259 |

The master row is not a file in this repo. It is listed so the numbers above have
a reference point — and this time the BF16 master **did** fit on the machine used
for the PPL measurements, so the tiers are compared against a real baseline.

The layers that would hurt most under low-bit compression are protected
(measured: on IQ4_XS this costs a few GiB over the bare tier):

| Tensors | Type | Reason |
|---|---|---|
| `mlp.gate.weight` | F32 | MoE router; compressing it routes to the wrong experts |
| `mlp.shared_experts.*` | Q8_0 | the shared expert runs on every token |
| `hc_attn_*` / `hc_ffn_*` | F32 | hyper-connection streams, every layer |
| `self_attn.A_log` / `k_conv1d` / `dt_bias` | F32 | KDA linear-attention state; low bit-width destroys long-range recall |
| `mlp.gate_proj` / `up_proj` / `down_proj` | Q8_0 | the 3 dense MLP layers before the MoE block |
| `token_embd` / `output` | Q6_K | a global type would otherwise squeeze these hard |

With 288 experts the expert layers dominate the file, so protecting everything
else is cheap.

One `NextN` (MTP) layer exists in the checkpoint but is excluded at conversion
time via `--no-mtp`; these files carry the main model's tensors only.

## Usage

```bash
# Point at the first shard; llama.cpp finds the rest on its own.
# Leave -c off as well on the first run — the same fitting that picks the layer
# split will also reduce the context size if that is what it takes to fit.
llama-cli -m GLM-5.3-Flash-IQ4_XS-00001-of-00015.gguf
```

Do **not** pass `-ngl` manually. llama.cpp fits layers to free VRAM by itself,
and any explicit `-ngl` value — including `0` — aborts that fitting:

```
common_fit_params: failed to fit params to free device memory:
                   n_gpu_layers already set by user to 99, abort
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

Leave `-ngl` off and let the fitting run.

Every tier ships as 15 shards (the master is 15 shards of ~39 GiB each for the
first 14 and ~9 GiB for the last). Download all 15 into one directory — you only
ever name `-00001-of-00015` on the command line.


## About the PPL numbers

wikitext-2 test, `n_ctx=512`, 12 chunks, every tier through the exact same
command, measured on 7× H100 80GB. **These numbers are only comparable within
this table.** Do not compare them against PPL figures published by other repos —
different corpora and chunk counts make absolute values meaningless across
setups.

The BF16 master row is the measured baseline (it fits in RAM on this box, unlike
the 4.5 TB monsters). Note `IQ2_XS` and `IQ1_M` degrade sharply — they exist for
when you must fit in a small footprint; prefer `IQ4_XS` or `IQ3_XXS`.

## License

MIT, inherited from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — see
[LICENSE](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE) for terms.
Quantized by qtum.
