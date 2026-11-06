---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
quantized_by: YogiQAI
tags:
  - gguf
  - imatrix
  - mixture-of-experts
  - glm
  - glm5-next
  - vision
  - dgx-spark
  - llama.cpp
language:
  - en
  - zh
---

# YogiQ GLM-5.3-Flash (GGUF)

Two GGUF quantizations of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) made from Z.ai's
official FP8 weights, with bits placed by a direct block-by-block sensitivity measurement on this model.
Full write-up, including the sensitivity scan and a negative result on entropy-coding the codebook indices:
**[Fitting GLM-5.3-Flash on one DGX Spark: measured bit placement](https://yogiq.ai/research/glm-5-3-flash-on-one-dgx-spark)**.

| Tier | Size | Fits | Perplexity 20×2048 | Perplexity 30×512 | vs BF16 |
|---|---:|---|---:|---:|---:|
| **Spark** | 108.3 GiB (116.3 GB) | one NVIDIA DGX Spark (121 GiB), 16K context + vision, 5 GiB spare | **3.354 ± 0.047** | **2.765 ± 0.060** | +23% |
| **Prime** | 118.0 GiB (126.7 GB) | ~125 GiB or more: two Sparks, 192 GB Mac Studio, 4×32 GB GPUs | 3.253 ± 0.045 | 2.702 ± 0.059 | +20% |

Perplexity is on wikitext-2 (test split), lower is better. BF16 reference on the matched 30×512 setting: 2.2474.
**Spark is, to our measurement, the lowest-perplexity GLM-5.3-Flash GGUF that fits one DGX Spark** (comparison below).

Both files are plain GGUF for stock [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit 330f018 or later,
architecture `glm5-next`) and anything built on it. Vision works with the included F16 projector. The MTP draft
layer is not included (no speculative drafting from these files).

## Files

| File | Size | SHA256 |
|---|---:|---|
| `Spark/YogiQ-GLM-5.3-Flash-Spark-IQ2_S-00001-of-00003.gguf` | 44.95 GB | `b65773bf…ecdd7e` |
| `Spark/YogiQ-GLM-5.3-Flash-Spark-IQ2_S-00002-of-00003.gguf` | 44.36 GB | `4407d194…90809b` |
| `Spark/YogiQ-GLM-5.3-Flash-Spark-IQ2_S-00003-of-00003.gguf` | 27.02 GB | `342fc9d1…bc285` |
| `Prime/YogiQ-GLM-5.3-Flash-Prime-IQ3_XXS-00001-of-00003.gguf` | 44.84 GB | `d1bab6bf…3bc225` |
| `Prime/YogiQ-GLM-5.3-Flash-Prime-IQ3_XXS-00002-of-00003.gguf` | 44.42 GB | `b6417d8d…8ded1d` |
| `Prime/YogiQ-GLM-5.3-Flash-Prime-IQ3_XXS-00003-of-00003.gguf` | 37.42 GB | `ce6a8340…acb6d1` |

The type token in each name is the base type of the mix (Spark: IQ2_S gate/up experts; Prime: uniform IQ3_XXS experts); the
exact per-tensor types are in the table below.
| `vision/mmproj-GLM-5.3-Flash-F16.gguf` (vision tower + projector, F16, from the official weights) | 1.13 GB | `de1eb189…50928f` |

Full checksums are in `SHA256SUMS`. Point llama.cpp at the first shard of a tier; it loads the rest. Files were split
with `llama-gguf-split --split-max-size 45G` from a single quantizer output.

## What is in them

GLM-5.3-Flash is 320B parameters, about 18B active per token: 45 layers (3 dense + 42 MoE), 288 routed experts +
1 shared expert per MoE layer, top-8 routing, 34 KDA linear-attention layers and 11 DSA/MLA attention layers.
95% of the parameters are the three routed-expert tensors per MoE layer, so the recipe is mostly about those.

| Tensors | Spark | Prime |
|---|---|---|
| Routed experts, gate and up, MoE layers 24–37 | **IQ3_XXS** | IQ3_XXS |
| Routed experts, gate and up, all other MoE layers | IQ2_S | IQ3_XXS |
| Routed experts, down (all layers) | IQ3_XXS | IQ3_XXS |
| Attention projections (KDA, MLA/DSA), shared expert, dense layers 0–2, embeddings | Q6_K | Q8_0 |
| Output projection | Q8_0 | Q8_0 |
| Router, indexer, KDA gates, MLA k_b / v_b / kv_a, hyper-connection (mHC) weights, norms (~1.1 GiB, left unquantized by llama-quantize) | BF16 / F32 | BF16 / F32 |

Bytes by type in the Spark file: IQ3_XXS 60.3 GiB, IQ2_S 40.4, Q6_K 6.0, BF16 0.7, Q8_0 0.6, F32 0.4.
Overall 2.97 bits per weight (Spark), 3.23 (Prime). Calibration: Unsloth's importance matrix for GLM-5.3-Flash
(blocks 0–44). Quantized with `llama-quantize` from the BF16 GGUF produced by llama.cpp's converter.

## How the bits were placed

The all-3-bit file (Prime) is 10 GiB too big for one DGX Spark. The uniform 2-bit/3-bit mix that fits (IQ2_S gate/up,
IQ3_XXS down, 106 GiB) scores 3.539. To find where the extra bits matter, we took that file and swapped in the
3-bit gate/up expert tensors from Prime one 7-layer block at a time (raw tensor copy, no requantization), measuring
perplexity each time. Each block costs +1.9 GiB.

| MoE layers upgraded to IQ3_XXS gate/up | PPL 20×2048 | gain |
|---|---:|---:|
| 3–9 | 3.482 | −0.057 |
| 10–16 | 3.478 | −0.061 |
| 17–23 | 3.497 | −0.042 |
| 24–30 | 3.470 | −0.069 |
| **31–37** | **3.439** | **−0.100** |
| 38–44 | 3.505 | −0.034 |

The best block is worth three times the worst for the same bytes, and the gains add up within noise (±0.05).
Spark upgrades the two best blocks (24–37). The ~2 GiB that pays for the second block comes from taking the
non-expert core from Q8_0 to Q6_K, which is near-lossless because the core is only ~5% of the model.

## Comparison with other GLM-5.3-Flash GGUFs at this size

All rows measured by us on one x86 machine (CPU, 20 threads), the same wikitext-2 text and settings, llama.cpp
commit 330f018, except Maya-M and the 512-context figure of the uniform mix, which needed a newer master build
(186eab1) with `--no-repack`; a cross-check of the uniform mix on both builds agreed. ± is the tool's reported
standard error. The Maya-M and GSQ-RCO figures are **our re-measurements**, not their authors' published numbers;
those authors evaluate differently (zero-shot task accuracy and KL divergence on their own text) and target
different hardware.

| File | Size | PPL 20×2048 | PPL 30×512 | vs BF16 |
|---|---:|---:|---:|---:|
| YogiQ Prime (needs ~125 GiB) | 118.0 GiB | 3.253 ± 0.045 | 2.702 ± 0.059 | +20% |
| **YogiQ Spark** | 108.3 GiB | **3.354 ± 0.047** | **2.765 ± 0.060** | **+23%** |
| our one-block variant (31–37 only, not released) | 108.1 GiB | 3.439 ± 0.048 | 2.871 ± 0.064 | +28% |
| [Maya-M](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (peasantsmith) | 108.0 GiB | 3.509 ± 0.049 | 2.971 ± 0.067 | +32% |
| uniform IQ2_S/IQ3_XXS mix (ours, not released) | 106.2 GiB | 3.539 ± 0.050 | 2.983 ± 0.067 | +33% |
| [GSQ-RCO 3.0-bit](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) (pfeifferj) | 109.4 GiB | 3.715 ± 0.054 | 3.177 ± 0.073 | +41% |

Maya-M keeps gate/up experts at IQ2_S in every MoE layer and upgrades only the down tensors (IQ3_S) in the first
and last four MoE layers; GSQ-RCO places its extra bits in MoE layers 3–8. Our scan puts them in layers 24–37.

## Limitations

- Wikitext-2 perplexity only: no coding benchmark, no KL divergence against FP8, no zero-shot tasks.
- Unsloth's own GGUFs at this size were not measured.
- The BF16 reference (2.2474 at 30×512) is taken from [another contributor's card](https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF)
  at identical settings; we could not run BF16 ourselves.
- 20 and 30 chunks per figure, one run each; the ordering of the middle scan blocks is within noise, the two ends are not.
- The sensitivity scan is at 7-layer block granularity and covers gate/up tensors only.
- The MTP draft layer is not included, so no speculative drafting from these files.
- We claim only what is in the tables above.

## Running

Serve with an OpenAI-compatible API and vision (what we run on the Spark, 16K context, ~14.6 tokens/s decode):

```bash
llama-server -m Spark/YogiQ-GLM-5.3-Flash-Spark-IQ2_S-00001-of-00003.gguf \
  --mmproj vision/mmproj-GLM-5.3-Flash-F16.gguf \
  -ngl 99 -fa auto -c 16384 -np 1 --jinja --host 0.0.0.0 --port 8090 --alias glm-5.3-flash
```

Memory: Spark tier with 16K context and the vision projector leaves ~5 GiB free of 121 GiB on a DGX Spark.
Prime does not load on one Spark; do not try, it can hang the box.

## Reproducing the numbers

Perplexity, wikitext-2 raw test split, two settings (use `-b 512` at context 512; the default batch gives
unreliable numbers for this architecture):

```bash
llama-perplexity -m MODEL.gguf -ngl 99 -fa auto -c 2048 -b 2048 --chunks 20 -f wiki.test.raw --no-warmup
llama-perplexity -m MODEL.gguf -ngl 99 -fa auto -c 512  -b 512  --chunks 30 -f wiki.test.raw --no-warmup
```

Spark tier recipe (from the BF16 GGUF converted with `--no-mtp`; rules are regexes on tensor names, first match wins):

```bash
llama-quantize --imatrix imatrix_unsloth.gguf_file \
  --tensor-type "blk\.[0-2]\.ffn_=q6_k" \
  --tensor-type "blk\.(2[4-9]|3[0-7])\.ffn_(gate|up)_exps=iq3_xxs" \
  --tensor-type "ffn_(gate|up)_exps=iq2_s" \
  --tensor-type "ffn_down_exps=iq3_xxs" \
  --tensor-type "_shexp=q6_k" --tensor-type "attn_=q6_k" --tensor-type "ssm_=q6_k" \
  --tensor-type "indexer=q6_k" --tensor-type "ffn_gate_inp=q6_k" \
  --token-embedding-type q6_k --output-tensor-type q8_0 \
  GLM-5.3-Flash-BF16.gguf YogiQ-GLM-5.3-Flash-Spark.gguf IQ2_XXS 16
```

Prime: the same with a single expert rule `ffn_(gate|up|down)_exps=iq3_xxs` and no Q6_K rules.

Full write-up: [yogiq.ai/research/glm-5-3-flash-on-one-dgx-spark](https://yogiq.ai/research/glm-5-3-flash-on-one-dgx-spark).

## Credits and license

- Model: [Z.ai](https://huggingface.co/zai-org), GLM-5.3-Flash, MIT license. These files are released under the same license.
- Calibration data: [Unsloth](https://huggingface.co/unsloth)'s importance matrix for GLM-5.3-Flash.
- Format, quantizers and kernels: [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml contributors.
- Compared works: [peasantsmith / Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) and
  [pfeifferj / GSQ-RCO](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF).

These are quantizations, not fine-tunes: no weights were trained, pruned or edited, apart from the omitted MTP
draft layer. Quantized and measured by **YogiQ AI Labs Pvt Ltd**, 5–6 November 2026. Contact: info@yogiq.ai.
