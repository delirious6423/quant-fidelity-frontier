---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
quantized_by: qtum
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- deepseek
- moe
- quantized
---

# DeepSeek-V4.1-Flash-1017 GGUF

GGUF quantizations of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), made with llama.cpp.

Chinese version: [README_zh.md](https://huggingface.co/qtum/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/README_zh.md)

## What this model is, in one paragraph

763 G parameters total, of which **196.6 G are the Engram tables** — two giant
n-gram lookup tables (`per_layer_token_embd`, 256 x 768,022,850) that occupy 26%
of the whole model. The routed experts ship as FP8 in the checkpoint and land in
MXFP4 after conversion: 384 of them per layer, 6 active per token, across 40
layers. Everything else is small. This shape drives every choice below.

**These files are text-only.** The upstream checkpoint also carries a 32-layer
vision tower (`vision_config`, patch size 14, `image_token_id`), and llama.cpp
does not support it for this architecture yet — a multimodal projector would be a
separate `mmproj` file. None of these GGUFs contain a single vision tensor, so
image input is not available here. Use the original weights if you need vision.

## Why there is no Q8_0 / Q6_K / Q5_K_M / Q4_K_M

The routed experts are already FP8 (`quantization_config.quant_method = fp8`,
`weight_block_size = [32, 32]`, `expert_dtype = fp4`). Re-quantizing them upward
only inflates the file — it cannot recover precision the upstream already
discarded. Quantizing the rest (attention, shared expert, embeddings) to a high
tier sounds lossless, but on this model those tensors are a small slice of the
total, so a "near-lossless" build ends up **larger than the original checkpoint**
while changing almost nothing about the part that dominates quality.

If you want near-lossless, use the original weights directly. An intermediate
tier buys you nothing on this model.

## What happened to the Engram tables

They are stored at **IQ4_NL**, and that is the single most important detail here.
They are 196.6 G parameters, so:

| Engram type | Table size |
|---|---|
| BF16 (master) | 366.2 GiB |
| Q8_0 | 194.6 GiB |
| Q6_K | 148.8 GiB |
| Q5_K | 125.9 GiB |
| IQ4_NL | 103.0 GiB |

Keeping them at Q8_0 would add ~92 GiB to *every single tier* for no measurable
gain. IQ4_NL is safe for this particular tensor because it is a **lookup table**
(`GET_ROWS` — one row fetched per token), so its quantization error never
accumulates through a matmul the way expert weights would. It cannot be split out
into a separate downloadable file either: the runtime creates it as a
*required* tensor, so a GGUF without it will not even load in `llama-quantize`.

## Quantizations

| Tier | Size | Shards | PPL (wikitext-2) |
|---|---|---|---|
| BF16 master (reference, not in this repo) | 650.9 GiB | — | **1.7914 ± 0.06318** |
| `IQ1_S` | 212.9 GiB | 5 | 3.1912 ± 0.12236 |
| `IQ1_M` | 224.3 GiB | 5 | 2.7648 ± 0.10085 |
| `IQ2_XXS` | 243.2 GiB | 6 | 2.2878 ± 0.08231 |
| `IQ2_XS` | 258.4 GiB | 6 | 2.1610 ± 0.07753 |

The master row is not a file in this repo. It is listed so the numbers above have
a reference point: its expert layers are the factory MXFP4 weights with no further
quantization, which is the accuracy ceiling for this model.

Every tier is imatrix-calibrated and the non-expert layers are protected:

| Tensors | Type | Reason |
|---|---|---|
| `attn_*` | Q8_0 | source weights are already FP8; compressing further gains nothing |
| `ffn_*_shexp` | Q8_0 | the shared expert runs on every token |
| `indexer*` / `*compressor*` | Q8_0 | sparse-attention index; compressing it picks the wrong KV |
| `engram_kv` | Q8_0 | real matmul projection feeding the Engram lookup (only 2 tensors, 0.58 GiB total) |
| `ffn_gate_inp` | F32 | router; compressing it routes to the wrong experts |
| `engram_gate` | F32 | element-wise multiply, not a matmul |
| `hc_*` | F32 | per-layer control signals, negligible size |
| `per_layer_token_embd` | IQ4_NL | the Engram tables, see above |
| `token_embd` / `output` | Q6_K | a global type would otherwise squeeze these hard |

## Usage

```bash
llama-cli -hf qtum/DeepSeek-V4.1-Flash-1017-GGUF:IQ1_S -ngl 99 -c 8192

# or point at the files yourself
llama-cli -m DeepSeek-V4.1-Flash-1017-IQ1_S-00001-of-0000N.gguf -ngl 99 -c 8192
```

For sharded tiers, download all shards and point at `-00001-of-*`; llama.cpp finds
the rest on its own.

The sizes above are well past 8x80 GB of VRAM, and about 103 GiB of every tier is
just the Engram table. Keep that one tensor in system RAM:

```bash
llama-cli -m ...-00001-of-0000N.gguf -ngl 99 -ot per_layer_token_embd=CPU -c 8192
```

`CPU` is the ggml-cpu buffer type name. If you typo it, llama.cpp exits with
`unknown buffer type` and prints the valid list.

### DSpark speculative decoding

The base model ships a DSpark speculative-decoding module (the `mtp.*` weights,
7.9 GiB in BF16). This repo exports it separately at 7.4 GiB, one draft file
serving every tier.

Its expert layers are factory MXFP4, exactly like the main model — so they are
kept as-is and everything else is Q8_0. Quantizing the whole draft to Q8_0 would
*inflate* it to 14.1 GiB: MXFP4 is 4.25 bpw and Q8_0 is 8.5 bpw, so re-quantizing
upward only doubles the bytes without recovering any precision.

```bash
llama-cli -m DeepSeek-V4.1-Flash-1017-IQ1_S-00001-of-0000N.gguf \
  -md dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --spec-type draft-dspark -ngl 99 -ngld 99 -c 2048
```

If a tier ships as a single file, drop the `-0000N-of-0000M` suffix.

Two things worth knowing, both measured in-house on 8xH100:

- **`-md` on its own does not turn it on.** Without `--spec-type draft-dspark` the
  draft file loads, the log prints `no implementations specified for speculative
  decoding`, and the slot reports `"speculative":false` — you get plain decoding at
  full speed and no error. `--spec-default` does not help either; it only enables
  `ngram-mod` and never touches the draft model.
- **One GPU at a time, for now.** Split across 2 or more GPUs it aborts on
  `GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)`. That is a llama.cpp
  scheduler limit, not a problem with these files.

## About the PPL numbers

Measured in-house with one fixed setup: wikitext-2 test, `n_ctx=512`, 12 chunks,
`-ngl 99`, `-ot per_layer_token_embd=CPU`, and the master baseline run through the
exact same command. **These numbers are only comparable within this table.** Do not
compare them against PPL figures published by other repos — different corpora,
chunk counts and mixed-precision maps make absolute values meaningless across setups.

## License

MIT, inherited from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Quantized by qtum.
