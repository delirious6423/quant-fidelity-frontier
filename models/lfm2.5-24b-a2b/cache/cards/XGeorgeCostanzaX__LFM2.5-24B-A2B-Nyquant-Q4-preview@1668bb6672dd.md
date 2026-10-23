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

> **Read this first.** These are previews, and they are scored against public GGUFs on one protocol. The public **unsloth UD-Q4_K_XL** (14.87 GB) has a lower macro KLD (0.0310) than our best file (0.0362, 13.88 GB), and is lower on all three corpora, at 1.00 GB more. The bytes are not equal, so we make no win or loss claim either way. Plain-imatrix rebuilds of our recipe score the same as these files, so no replacement build is queued from this evidence.

## Which file should I download?

| File | Size | Bits per weight | Macro KLD vs BF16 [95% CI] | P100 decode / prefill (measured) | Pick it when |
|---|---:|---:|---|---:|---|
| `LFM2.5-24B-A2B-mix-ar4match-preview.gguf` | 13.88 GB | 4.63 | 0.0362 [0.0338, 0.0388] | 67.8 / 518.5 tok/s | You want the lowest KLD of our three files and have about 16 GB of VRAM |
| `LFM2.5-24B-A2B-mix-iq4xs-preview.gguf` | 13.09 GB | 4.37 | 0.0396 [0.0367, 0.0427] | 63.4 / 579.3 tok/s | A middle size, faster prefill than ar4match on the P100 |
| `LFM2.5-24B-A2B-mix-F12000-preview.gguf` | 12.00 GB | 4.01 | 0.0560 [0.0526, 0.0598] | 63.8 / 576.0 tok/s | You need the smallest file and accept the highest KLD of the three |

<sub>KLD is against BF16 on the fixed 180-window protocol, mean of 3 corpora; see [Quality](#quality-measured-on-the-released-files). Speeds: Tesla P100 16 GB at 125 W, `llama-bench`, f16 KV, flash attention on, tg128 at depth 0 and pp512, mean of 5 runs. Other GPUs are approximate; see [Speed](#speed-approximate).</sub>

## Quality, measured on the released files

All rows below were scored on the same fixed protocol, with a BF16 teacher and the same runtime, on exactly the files you can download (identities and SHA-256 in the [appendix](#appendix-identities-and-earlier-panel)). Lower is better everywhere.

![KLD against file size for the three released files and public GGUFs](assets/chart-released-kld-vs-size.png)

**How to read it.** Lower and further left is better. Among our files, KLD falls as size grows, from the 12.00 GB file to the 13.88 GB file. Public **unsloth UD-Q4_K_XL** sits lower than all three of our files but is 1.00 GB larger than our largest. **unsloth UD-Q3_K_XL** is smaller than all of ours (10.94 GB) and has a higher KLD. **QAD-Q4_0** (Liquid AI) is larger than our 13.09 GB file and has a higher KLD than every file of ours. None of these pairs has equal bytes, so this is a trade-off picture, not an equal-size ranking.

![Per-corpus KLD and perplexity for the same six files](assets/chart-released-per-corpus.png)

**How to read it.** The same ordering holds corpus by corpus. Web text carries most of the KLD for every file; code is the easiest. Perplexity moves very little between files (look at the axis ticks), so KLD is the more sensitive measure.

### KLD per corpus, arithmetic macro and geometric mean

| Model | Bytes | GB | KLD chat | KLD web | KLD code | Arithmetic macro [95% CI] | Geometric mean (3 corpora) |
|---|---:|---:|---:|---:|---:|---|---:|
| unsloth UD-Q4_K_XL | 14,876,901,632 | 14.87 | 0.0193 | 0.0583 | 0.0156 | 0.0310 [0.0289, 0.0335] | 0.0260 |
| **Nyquant preview ar4match** | 13,873,184,832 | 13.88 | 0.0235 | 0.0662 | 0.0189 | 0.0362 [0.0338, 0.0388] | 0.0309 |
| QAD-Q4_0 (Liquid AI) | 13,578,190,880 | 13.57 | 0.0428 | 0.1375 | 0.0320 | 0.0708 [0.0665, 0.0754] | 0.0574 |
| **Nyquant preview iq4xs** | 13,080,461,248 | 13.09 | 0.0252 | 0.0731 | 0.0206 | 0.0396 [0.0367, 0.0427] | 0.0336 |
| **Nyquant preview F12000** | 11,996,757,952 | 12.00 | 0.0387 | 0.1008 | 0.0287 | 0.0560 [0.0526, 0.0598] | 0.0482 |
| unsloth UD-Q3_K_XL | 10,947,100,928 | 10.94 | 0.0554 | 0.1411 | 0.0390 | 0.0785 [0.0739, 0.0834] | 0.0673 |

<sub>Per-corpus values have no interval here; the macro has a 95% bootstrap interval (2,000 draws, seed 0, interval endpoints rounded outward). Geometric mean is computed from the three per-corpus values and has no interval. Our values round up and public rows round down.</sub>

### Perplexity per corpus

| Model | PPL chat | PPL web | PPL code |
|---|---:|---:|---:|
| unsloth UD-Q4_K_XL | 3.71 | 17.23 | 1.62 |
| **Nyquant preview ar4match** | 3.73 | 16.97 | 1.62 |
| QAD-Q4_0 (Liquid AI) | 3.65 | 16.74 | 1.61 |
| **Nyquant preview iq4xs** | 3.71 | 16.93 | 1.63 |
| **Nyquant preview F12000** | 3.77 | 17.13 | 1.64 |
| unsloth UD-Q3_K_XL | 3.78 | 17.05 | 1.63 |

### Public rows not shown

Four more public GGUFs were scored once (bartowski IQ4_XS and Q3_K_L, ggml-org Q4_K_M, mradermacher i1-IQ4_XS). Their runs used a start-of-text setting that differs from the reference and inflates the result, so they are withheld until a corrected rerun. They are pending, not omitted because of the outcome.

## Two products per size (coming)

From the next release each size ships as **-Chat** (decided by an on-policy chat and tool-call panel) and **-Balanced** (decided by the geometric mean of the per-corpus KLDs, with no corpus regressing). The three files here are single previews that are not yet decided by either rule: there is no on-policy chat panel for this model yet, and the Balanced rule is defined over four corpora while we have measured three. We therefore show per-corpus values and both macros, and claim nothing against either rule.

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

<sub>**Measured (not approximate):** Tesla P100 at 125 W, `llama-bench`, n=5: decode 67.8 / 63.4 / 63.8 tok/s and pp512 518.5 / 579.3 / 576.0 tok/s for ar4match / iq4xs / F12000. Our P100 kernel branch (not yet public) measured **97.3 tok/s** single-user decode for ar4match on one P100 at 125 W, against 71.4 tok/s for the unmodified build in the same session (five alternating A/B/A repeats); four-user throughput (126.3 tok/s) and pp512 (about 540 tok/s) were unchanged, and its outputs were bit-identical to the unmodified build (KLD 0). The P100 figures above are therefore conservative for a tuned runtime. A Tesla P40 over an HTTP server (512-token prompt, 128 generated, n=3) measured 81.0 tok/s for ar4match.</sub>

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

| | |
|---|---|
| Teacher | BF16 LFM2.5-24B-A2B, SHA-256 `55103fda18def5210dafac2d1f1125afc93baf604dc24cf0edf9e9aee0d576ec` |
| Runtime | llama.cpp commit `035094c7f12003af148772bf9ebb549f59fd6b73`, one A100 SXM4 80 GB per run |
| Text | held-out chat, web and code; 60 windows of 2,048 tokens per corpus (180 in total), positions 1,024 to 2,046 scored |
| Statistic | native KLD vs BF16, mean over corpora with a 95% bootstrap interval (2,000 draws, seed 0); perplexity from the same positions |
| Start-of-text | forced to the reference setting for every row shown; rows that could not be were withheld |
| Token arrays | chat `b738a385...`, code `117c07ef...`, web `960000eb...` (SHA-256 prefixes) |
| Speed | see [Speed](#speed-approximate): P100 at 125 W measured, n=5; other GPUs approximate |

## Known limitations

- **Preview.** Not a drop-in replacement for the original or for a stock quant without your own checks.
- **Public rows lead at larger size.** unsloth UD-Q4_K_XL scores lower KLD than all three of our files, with more bytes. We have not run an equal-byte comparison.
- **No better build yet.** Plain-imatrix rebuilds of our recipe tie these files to within 1e-17 in paired KLD, so none replaces a preview. We do not have a build that beats the previews.
- **Chat and tool-use quality are not measured.** There is no on-policy panel and no tool-calling comparison for this model.
- **Four public rows are pending** a corrected rerun (see above), and third-party GGUFs beyond these are not scored.
- **Perplexity differences are small** and should not be over-read.
- Most speeds are approximations.

## License and attribution

The [LFM Open License v1.0](LICENSE). These files are modified versions of the original LiquidAI/LFM2.5-24B-A2B weights; see [NOTICE](NOTICE) and [LICENSE](LICENSE). This is an independent derivative and is not affiliated with or endorsed by Liquid AI. Base model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).


## Appendix: identities and earlier panel

<details>
<summary>File identities (bytes and SHA-256)</summary>

| File | Bytes | SHA-256 |
|---|---:|---|
| unsloth UD-Q4_K_XL | 14,876,901,632 | `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8` |
| ar4match | 13,873,184,832 | `307001728853b9f18993596e6da49ca7655a506fdf85fa2471344300b287fb3c` |
| QAD-Q4_0 (Liquid AI) | 13,578,190,880 | `d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5` |
| iq4xs | 13,080,461,248 | `4c1fd7f51abc6fbd1512b8042deeba08cf37bad67112b26657db1f3637c8e54a` |
| F12000 | 11,996,757,952 | `a2b215c64eb9fb735eca2138d346828c35a853fad555cbf520d33feb295a63f0` |
| unsloth UD-Q3_K_XL | 10,947,100,928 | `d54834c785c7e265f1cb7375e47aa6f55a0b7cfab75816254c4d2e0c82fd5efc` |

</details>

<details>
<summary>Earlier panel (historical D2 builds, kept for the record)</summary>

An earlier version of this card showed a panel measured on earlier builds, not on the released hashes. It is kept here unchanged. In that panel an AutoRound-based "AR4" recipe lost to plain imatrix quants of the same size. The released files scored in this card (0.0362, 0.0396 and 0.0560 macro KLD) are closer to the plain-imatrix rows of that panel than to its AR4 rows, and the paired rebuilds tie them, so treat the released files as plain-imatrix allocations, not as the AR4 recipe.

![Historical KLD vs size](assets/chart-kld-vs-size.png)

![Historical perplexity](assets/chart-ppl.png)

</details>

## Changelog

- **2026-10-23:** P100 note updated with the tuned-runtime measurement (97.3 tok/s single-user decode for ar4match).
- **Next:** replaced the earlier-build quality panel with scores on the released files, per corpus and as two macros, against public GGUFs; added file identities; moved the earlier panel to the appendix. No change to the model files.
- **2026-10-22:** card refresh with charts, rounding corrections, approximate speeds and quick start.
- **Earlier:** initial early-preview upload (three GGUF files).
