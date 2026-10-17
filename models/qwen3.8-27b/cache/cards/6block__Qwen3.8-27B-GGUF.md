---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: 6block
license: apache-2.0
language:
- en
- zh
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- qwen
- multimodal
- vision
- quantized
- imatrix
---

# Qwen3.8-27B GGUF

GGUF quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made with llama.cpp.

Chinese version: [README_zh.md](https://huggingface.co/6block/Qwen3.8-27B-GGUF/blob/main/README_zh.md)

A 27B dense vision-language model (27.8B including the vision tower). 64 decoder
layers in a hybrid stack: `full_attention_interval=4` means every fourth layer is
grouped-query full attention (16 layers) and the rest are Gated DeltaNet linear
attention (48 layers). Vocab 248320 with untied embeddings. Context length 262144
natively, extensible further with YaRN.

Text, image and video in; text out.

## What is in this repo

| File group | What it is |
|---|---|
| `Qwen3.8-27B-<tier>.gguf` | text backbone, one file per tier |
| `mmproj-Qwen3.8-27B-*.gguf` | vision projector — **required for image/video input** |
| `mtp-Qwen3.8-27B-*.gguf` | MTP head, usable as a speculative-decoding draft |
| `imatrix.gguf` | the calibration matrix every tier was built with |

The mmproj file is separate by design: llama.cpp loads the text backbone and the
vision projector as two files. Download one tier plus one mmproj. Without the
mmproj the model still runs, but text-only.

The `mmproj-` and `mtp-` prefixes are what recent llama.cpp builds match on to
tell the three file kinds apart, so `-hf 6block/Qwen3.8-27B-GGUF` picks a backbone rather than a
projector, and `--spec-type mtp` finds the draft on its own.

The draft is larger than an MTP head alone would suggest: a standalone draft
needs its own embedding and output projection, and with vocab 248320 × hidden
5120 untied those two copies dominate the file. The MTP block itself is small.

## Quantizations

| Tier | Size | BPW | PPL |
|---|---|---|---|
| master (BF16, not in this repo) | 50.11 GiB | 16.00 | **6.7856 ± 0.07388** |
| `Q8_0` | 26.12 GiB | 8.34 | 6.7900 ± 0.07391 |
| `Q6_K` | 21.01 GiB | 6.71 | 6.7951 ± 0.07403 |
| `Q5_K_M` | 18.70 GiB | 5.97 | 6.8010 ± 0.07406 |
| `Q5_K_S` | 18.21 GiB | 5.82 | 6.7973 ± 0.07396 |
| `Q4_1` | 16.89 GiB | 5.40 | 6.8105 ± 0.07411 |
| `Q4_K_M` | 16.53 GiB | 5.28 | 6.8103 ± 0.07414 |
| `IQ4_NL` | 15.87 GiB | 5.07 | 6.8491 ± 0.07484 |
| `Q4_K_S` | 15.68 GiB | 5.01 | 6.8291 ± 0.07440 |
| `Q4_0` | 15.62 GiB | 4.99 | 6.8749 ± 0.07520 |
| `IQ4_XS` | 15.28 GiB | 4.88 | 6.8529 ± 0.07487 |
| `Q3_K_M` | 13.23 GiB | 4.23 | 7.1020 ± 0.07902 |
| `IQ3_M` | 12.56 GiB | 4.01 | 7.0306 ± 0.07531 |
| `Q3_K_S` | 12.16 GiB | 3.88 | 7.2176 ± 0.08074 |
| `IQ3_XXS` | 11.60 GiB | 3.70 | 7.1994 ± 0.07848 |
| `Q2_K` | 11.11 GiB | 3.55 | 7.4700 ± 0.08388 |
| `IQ2_M` | 10.50 GiB | 3.35 | 7.5101 ± 0.08176 |
| `IQ2_XS` | 9.86 GiB | 3.15 | 7.9668 ± 0.08803 |
| `IQ2_XXS` | 9.30 GiB | 2.97 | 8.5638 ± 0.09560 |

## Tiers a smaller file already matches

The ladder is published in full for coverage, which means a few tiers end up
larger than another tier that measured no worse on this model:

| Tier | Size / PPL | Smaller, PPL no higher |
|---|---|---|
| `Q5_K_M` | 18.70 GiB / 6.8010 | `Q5_K_S` — 18.21 GiB / 6.7973 |
| `Q4_1` | 16.89 GiB / 6.8105 | `Q4_K_M` — 16.53 GiB / 6.8103 |
| `IQ4_NL` | 15.87 GiB / 6.8491 | `Q4_K_S` — 15.68 GiB / 6.8291 |
| `Q4_0` | 15.62 GiB / 6.8749 | `IQ4_XS` — 15.28 GiB / 6.8529 |
| `Q3_K_M` | 13.23 GiB / 7.1020 | `IQ3_M` — 12.56 GiB / 7.0306 |
| `Q3_K_S` | 12.16 GiB / 7.2176 | `IQ3_XXS` — 11.60 GiB / 7.1994 |

This is a size argument, not a quality one. The size gap is real; the PPL side is
a tie — `Q5_K_M` and `Q5_K_S` differ by 0.004 against a ±0.074 error bar, and
`Q4_1` vs `Q4_K_M` by 0.0002. Nothing here says the left column is broken.

`Q4_0` and `Q4_1` are the pre-K-quant formats and `IQ4_NL` targets backends
without full K-quant support, so they are kept for runtimes that need them. If
yours loads either, take the right-hand column.

The master row is not a file in this repo. It is listed so the numbers above have
a reference point.

## KV cache is cheaper than the layer count suggests

Only the 16 full-attention layers keep a growing KV cache; the 48 linear-attention
layers hold a fixed-size recurrent state that does not grow with context. That
works out to roughly 64 KiB per token, so 32k context costs about 2 GiB — much
less than a 64-layer all-attention model of this size would need.

Budget roughly `tier size + 0.9 GiB (mmproj) + KV cache` when picking a tier.

## Calibration and protection

Every tier is imatrix-calibrated (`n_ctx=512`, mixed code / English / Chinese),
and `imatrix.gguf` is published here so the calibration is auditable and reusable.
The layers that suffer most under low-bit compression are protected:

| Tensors | Type | Reason |
|---|---|---|
| `ssm_alpha` / `ssm_beta` | F32 | the linear-attention state gates; 0.09 GiB in total, so full precision is essentially free |
| `attn_q` / `attn_k` / `attn_v` / `attn_output` | Q8_0 | the 16 full-attention layers — the only ones whose KV cache grows with context |
| `token_embd` / `output` | Q6_K (Q4_K below 4-bit) | see below |

Deliberately **not** pinned: `ssm_out`, `attn_qkv` and `attn_gate` take the tier
type like any other projection. Together they are 20.6% of the backbone, so an
8-bit floor under all three flattens the ladder: a Q4_K_M built that way came out
at 22.97 GiB / 7.33 bpw against the shipped 16.53 GiB / 5.28 bpw, with five tiers
landing within 7 GiB of each other.

Both builds were measured on the same command. The heavy one scored **6.8161**
against **6.8103** shipped — a tie inside the ±0.074 error bar. The extra 6.44 GiB
bought nothing. `ssm_out` reads like state but is the output projection of the
linear-attention block, the counterpart of `attn_output`; llama.cpp's own
quantizer gives it no special treatment either.

`token_embd` and `output` are 9.4% of the backbone on their own — vocab 248320 ×
hidden 5120, and the two are untied so there are two copies. Pinning both to Q6_K
at every tier would push the low tiers well above their nominal bit-width (IQ2_XS
would land near 3.0 bpw effective), so at sub-4-bit tiers they drop to Q4_K
instead. The BPW column above is the effective figure, protection included.

## Usage

```bash
# Text only
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192

# With vision — pass the projector alongside the backbone
llama-mtmd-cli -m Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
  --image photo.jpg -p "Describe this image."
```

Do **not** pass `-ngl` manually. llama.cpp fits layers to free VRAM by itself;
setting `-ngl` explicitly aborts that fitting (`n_gpu_layers already set by user`)
and can silently fall back to CPU.

Speculative decoding with the MTP head. `--spec-type draft-mtp` is required, not
optional: with `-md` alone the draft is loaded but never used, because the
speculative type stays unset and the MTP path is gated on it.

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -md mtp-Qwen3.8-27B-Q8_0.gguf --spec-type draft-mtp
```

Pulling straight from this repo, the draft is discovered from the `mtp-` prefix
and the type is inferred, so `--spec-type` can be omitted:

```bash
llama-server -hf 6block/Qwen3.8-27B-GGUF --spec-draft-hf 6block/Qwen3.8-27B-GGUF
```

## About the PPL numbers

Measured in-house with one fixed setup, master baseline run through the exact
same command. **These numbers are only comparable within this table.** Do not
compare them against PPL figures published by other repos — different corpora
and chunk counts make absolute values meaningless across setups.

## License

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Quantized by 6block.
