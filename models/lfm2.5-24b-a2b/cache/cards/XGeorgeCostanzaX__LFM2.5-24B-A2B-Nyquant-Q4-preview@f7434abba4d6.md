---
library_name: gguf
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- quantized
- imatrix
- moe
- lfm2
---

<div align="center">

# LFM2.5-24B-A2B &middot; Nyquant Q4

**Importance-matrix, per-tensor 4-bit-class GGUFs of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), 12 to 14 GB**

`preview` &nbsp;|&nbsp; 3 files &nbsp;|&nbsp; 24B total, about 2B active (MoE) &nbsp;|&nbsp; llama.cpp

</div>

Routed experts sit at about 4 bits; the more sensitive tensors (attention, short-conv, dense FFN, embeddings) are kept at higher precision. All bit allocations were chosen with an importance matrix.

> **Read this first.** This is a preview. A plain imatrix quant of the same size scored a lower KL divergence than our recipe in our earlier head-to-head, by 9.5% to 15.8% (details and chart below). We publish the comparison, including where we lose, so you can decide. The three released files are not yet scored on the full KLD protocol; see [Known limitations](#known-limitations).

## Which file should I download?

| File | Size | Bits per weight | Wiki PPL<sup>1</sup> | Measured on a P100<sup>2</sup> decode / prefill | Pick it when |
|---|---:|---:|---:|---:|---|
| `LFM2.5-24B-A2B-mix-ar4match-preview.gguf` | 13.88 GB | 4.63 | 10.98 | 67.8 / 518.5 tok/s | You want the highest-precision file of the three and have about 16 GB of VRAM |
| `LFM2.5-24B-A2B-mix-iq4xs-preview.gguf` | 13.09 GB | 4.37 | 11.04 | 63.4 / 579.3 tok/s | A middle size; faster prefill than ar4match on the P100 |
| `LFM2.5-24B-A2B-mix-F12000-preview.gguf` | 12.00 GB | 4.01 | 11.29 | 63.8 / 576.0 tok/s | You need the smallest file and accept a higher perplexity |

<sub>1. WikiText perplexity, 60 windows of 2,048 tokens, tokens 1,024 to 2,046 scored (61,380 tokens), llama.cpp `035094c7f`. This is a sanity check on the exact released files; it is not the chat/web/code panel, and the BF16 reference and public rivals are not scored on it. Lower is better.<br>
2. Tesla P100 16 GB at 125 W, `llama-bench`, f16 KV, flash attention on, tg128 at depth 0 and pp512, mean of 5 runs (SD about 0.31, 1.64, 0.22 tok/s for decode). Details in [Speed](#speed-approximate).</sub>

## Quality vs other quants

The panel below compares our recipe against standard llama.cpp quants and a plain imatrix control on one protocol. **It was measured on earlier builds of the same recipe (the "D2" panel), not on the three released hashes**: the released files were rebuilt and we have not proven that the weights are identical. It is the best matched-protocol evidence we have today, and it is labelled as such everywhere.

![KL divergence vs file size for our AR4 recipe, plain imatrix controls and standard quants](assets/chart-kld-vs-size.png)

**How to read it.** Lower and further left is better. At each of our three sizes the plain imatrix control (blue squares) sits below our recipe (orange), with intervals that barely separate at 13.88 GB and overlap at the two smaller sizes: the controls beat us by 9.5% to 15.8% on KLD. Against the standard quants we are lower than Q4_K_M, QAD-Q4_0 and IQ4_XS at similar or smaller sizes, and Q4_K_M with an imatrix (14.52 GB) is the closest standard rival. We have not yet scored public third-party GGUFs (unsloth, bartowski and others) on this protocol, so they are absent from the chart rather than losing or winning on it.

![Perplexity on held-out chat, web and code text for the same quants](assets/chart-ppl.png)

**How to read it.** Perplexity differences between these quants are small, and each panel uses its own axis, so a visually large gap can be a few hundredths. Our recipe is lower than the imatrix controls on web text at all three sizes, and level or slightly higher on chat and code at two of three sizes. KLD, which measures drift from the BF16 distribution directly, is the stricter test and is where we lose.

### Full table (earlier builds, same protocol)

| Quant | Size | KLD vs BF16 [95% CI] | PPL chat | PPL web | PPL code |
|---|---:|---|---:|---:|---:|
| Q4_K_M | 14.52 GB | 0.0710 [0.0662, 0.0768] | 3.72 | 18.03 | 1.63 |
| Q4_K_M with imatrix | 14.52 GB | 0.0588 [0.0552, 0.0629] | 3.71 | 17.95 | 1.63 |
| **Nyquant AR4** (size-matched to mix-ar4match) | 13.88 GB | 0.0415 [0.0385, 0.0449] | 3.68 | 16.58 | 1.62 |
| Plain imatrix control, same size | 13.87 GB | 0.0358 [0.0335, 0.0384] | 3.72 | 16.84 | 1.61 |
| QAD-Q4_0 (official) | 13.57 GB | 0.0689 [0.0645, 0.0735] | 3.63 | 16.33 | 1.61 |
| **Nyquant AR4-F13080** (size-matched to mix-iq4xs) | 13.09 GB | 0.0438 [0.0406, 0.0472] | 3.73 | 16.22 | 1.63 |
| Plain imatrix control, same size | 13.08 GB | 0.0392 [0.0366, 0.0422] | 3.71 | 16.85 | 1.62 |
| IQ4_XS | 12.83 GB | 0.0708 [0.0663, 0.0758] | 3.71 | 17.75 | 1.63 |
| **Nyquant AR4-F12000** (size-matched to mix-F12000) | 12.00 GB | 0.0615 [0.0574, 0.0657] | 3.79 | 16.45 | 1.65 |
| Plain imatrix control, same size | 11.99 GB | 0.0560 [0.0525, 0.0599] | 3.76 | 17.09 | 1.63 |
| IQ3_S | 10.42 GB | 0.1524 [0.1442, 0.1619] | 3.92 | 16.81 | 1.67 |

<sub>KLD is the mean over three corpora (held-out chat, web, code), 180 windows in total, with 95% bootstrap intervals (2,000 draws). Sizes are the sizes of those earlier files. BF16 reference scores are not part of this panel.</sub>

## Speed (approximate)

Most of this table is a projection, not a benchmark. Ranges are planning bands, not confidence intervals.

![Approximate decode speed per GPU](assets/chart-speed-approx.png)

| GPU | Decode, approx. tok/s (ar4match) | Decode, approx. (iq4xs) | Decode, approx. (F12000) | Prefill pp512, approx. tok/s (ar4match) |
|---|---:|---:|---:|---:|
| RTX 3060 12 GB | needs offload/partial | needs offload/partial | needs offload/partial | needs offload/partial |
| RTX 3090 24 GB | ~60–210 | ~60–220 | ~70–230 | ~900–2300 |
| RTX 4090 24 GB | ~70–230 | ~70–240 | ~70–250 | ~2200–5400 |
| RTX 5090 32 GB | ~120–410 | ~130–430 | ~130–440 | ~2900–6900 |
| V100 32 GB | ~60–210 | ~60–210 | ~60–220 | ~700–1800 |
| A100 80 GB | ~130–450 | ~140–460 | ~140–480 | ~2100–5100 |
| H100 80 GB SXM | ~230–780 | ~240–800 | ~250–830 | ~6800–16400 |
| P100 16 GB (125 W) | ~50–80 | ~50–80 | ~50–80 | ~400–500 |
| P40 24 GB | ~20–80 | ~20–80 | ~20–80 | ~600–1000 |
| Apple M4 Pro 20 GPU cores, 32 GB | ~10–60 | ~10–60 | ~20–60 | ~200–500 |
| Apple M4 Max 40 GPU cores, 48 GB | ~30–120 | ~30–130 | ~40–130 | ~400–1100 |

<sub>**Measured (not approximate):** Tesla P100 at 125 W, `llama-bench`, n=5: decode 67.8 / 63.4 / 63.8 tok/s and pp512 518.5 / 579.3 / 576.0 tok/s for ar4match / iq4xs / F12000. A later runtime build measured 93.2 tok/s native decode on the same P100 for ar4match (n=3), so the P100 figures above are conservative. A Tesla P40 over an HTTP server (512-token prompt, 128 generated, n=3) measured 81.0 tok/s for ar4match.</sub>

**Method note, approx.** Decode is memory-bandwidth divided by the bytes read per token (all shared weights plus 4 of 64 experts, plus a 20 KiB-per-token KV term), times an efficiency band calibrated on our P100 and P40 timings; moving that efficiency to other GPU architectures is an assumption. Prefill (pp512) scales the P100 calibration by nominal FP16 compute with a deliberately wide 0.25 to 0.60 transfer factor. Apple figures use a published M4 Max control on a similar-size MoE and core-count scaling. Decode assumes a 4K context, f16 KV, one user and the whole model on the GPU, so the RTX 3060 12 GB needs CPU offload (not estimated, as is CPU-only). Both ends of every range are rounded down. Only the P100 and P40 rows come from running these files; every other row is approx., not measured.

## Quick start

```bash
# llama.cpp (recent build with the lfm2moe architecture)
llama-cli -m LFM2.5-24B-A2B-mix-ar4match-preview.gguf -ngl 99 -c 8192 -fa on -cnv

# OpenAI-compatible server, larger context
llama-server -m LFM2.5-24B-A2B-mix-ar4match-preview.gguf -ngl 99 -c 32768 -fa on --port 8080
```

- Use the chat template embedded in the GGUF. The model supports a context of up to 131,072 tokens; KV cache memory grows with context (about 20 KiB per token at f16), so lower `-c` if you run out of VRAM.
- If a model does not fit in VRAM, lower `-ngl` or use `--n-cpu-moe` to keep some experts on the CPU; expect a large slowdown.
- Older llama.cpp builds take `-fa` without a value.
- Ollama and LM Studio: we have not tested these files there. Any runtime built on a llama.cpp recent enough to know `lfm2moe` should load them.

## How we measured

| | Earlier-build panel (charts, full table) | Released files |
|---|---|---|
| Teacher | BF16 LFM2.5-24B-A2B | BF16 (not yet scored on chat/web/code) |
| Windows | 60 windows of 2,048 tokens per corpus, 3 corpora (held-out chat, web, code): 180 | WikiText, 60 windows of 2,048 |
| Scored tokens | positions 1,024 to 2,046 of each window | positions 1,024 to 2,046: 61,380 tokens |
| Statistic | KLD vs BF16, mean over corpora, 95% bootstrap CI (2,000 draws, seed 0); perplexity per corpus | perplexity with the llama.cpp-printed uncertainty (0.132 / 0.132 / 0.135), which is **not** a bootstrap CI |
| Speed | see above | P100 125 W, n=5, runtime `035094c7f` |

## Known limitations

- **Preview.** Not a drop-in replacement for the original or for a stock Q4_K_M without your own checks.
- **The released hashes lack the full quality panel.** For each released file we have only four chat windows and the WikiText perplexity; the CPU teacher run that would produce the 180-window panel stopped on a token-alignment problem. The charts show earlier builds of the same recipe.
- **We lose to a plain imatrix quant at equal size** in the earlier panel, on KLD (15.8%, 11.5% and 9.5% higher KLD for the 13.88, 13.09 and 12.00 GB recipes). If KLD is your priority, an imatrix Q4 of the same size is at least as good by that measure.
- **No public third-party rivals scored yet** (unsloth, bartowski and others), and no BF16 perplexity row.
- Tool-calling and long-context quality vs BF16 have not been measured.
- Most speeds are approximations (see above).

## License and attribution

The [LFM Open License v1.0](LICENSE). These files are modified versions of the original LiquidAI/LFM2.5-24B-A2B weights; see [NOTICE](NOTICE) and [LICENSE](LICENSE). This is an independent derivative and is not affiliated with or endorsed by Liquid AI. Base model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

## Changelog

- **2026-10-22:** Card refresh: charts, quality table with intervals from the earlier-build panel, WikiText perplexity and P100 speeds for the released files, approximate speeds for other GPUs, quick start and limitations. No change to the model files.
- **Earlier:** initial early-preview upload (three GGUF files).
