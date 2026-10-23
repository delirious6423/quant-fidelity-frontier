---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- llama.cpp
- qwen3_5_moe
- imatrix
---

# Ornith-1.5-35B-A3B-GGUF

GGUF quantizations of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), made with an importance matrix, plus a KL-divergence comparison with the other published GGUFs of this model.

## Files

| File | Size | Mean KLD ja / en | Mix | Use when |
|---|---|---|---|---|
| `Ornith-1.5-35B-A3B-Q4_K_M-hq.gguf` | 22.4 GB | 0.0339 / 0.0357 | Experts as in Q4_K_M | Best quality here; about the same as APEX I-Quality (23.7 GB) |
| `Ornith-1.5-35B-A3B-IQ4_XS-exps-hq.gguf` | 20.0 GB | 0.0405 / 0.0429 | Experts IQ4_XS | Lower KLD than every published Q4_K_M, at about 2 GB less |
| `Ornith-1.5-35B-A3B-IQ3_S-exps-down4-hq.gguf` | 18.1 GB | 0.0706 / 0.0906 | Experts gate/up IQ3_S, down Q4_K | Around 18 GB |
| `Ornith-1.5-35B-A3B-IQ3_S-exps-hq.gguf` | 16.7 GB | 0.0927 / 0.1056 | Experts IQ3_S | Around 17 GB |
| `Ornith-1.5-35B-A3B-IQ3_XXS-exps-down3-hq.gguf` | 15.7 GB | 0.1008 / 0.1104 | Experts gate/up IQ3_XXS, down IQ3_S | Around 16 GB; mean KLD close to IQ3_S-exps-hq with a lower tail |
| `Ornith-1.5-35B-A3B-IQ2_S-exps-down3-hq.gguf` | 14.4 GB | 0.1523 / 0.1710 | Experts gate/up IQ2_S, down IQ3_S | Around 14 GB |
| `Ornith-1.5-35B-A3B-IQ2_XS-exps-down3-hq.gguf` | 13.2 GB | 0.2071 / 0.1969 | Experts gate/up IQ2_XS, down IQ3_XXS | Around 13 GB |
| `imatrix.gguf` | | | | Importance matrix used for all files |
| `tensor-types-*.txt` | | | | `--tensor-type-file` rules for each file |

All files are Q4_K_M with two kinds of changes. The `-hq` part raises the tensors outside the routed experts to Q8_0: attention, the linear-attention (SSM) projections, the shared expert, and the MTP projection. The `-exps` part changes the routed expert tensors (`ffn_{gate,up,down}_exps`) in layers 0-39. The token embedding and the output keep the Q4_K_M types (Q4_K and Q6_K). All files keep the built-in MTP (NextN) layer, so they work with `--spec-type draft-mtp`. For image input, use the vision projector from the official repo ([`mmproj-Ornith-1.5-35B-BF16.gguf`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)).

## Quality

KL divergence (KLD) of the next-token distribution from BF16; lower is closer to the original. The other published GGUFs of this model are measured the same way. Rows marked "not uploaded" are our own mixes without `-hq`, kept to show what it adds.

| | Size | Mean KLD ja / en | 99.9% KLD ja / en | Same top token ja / en | PPL ratio ja / en |
|---|---|---|---|---|---|
| Q8_0 (reference, not uploaded) | 37.8 GB | 0.0132 / 0.0200 | 0.458 / 0.967 | 94.7% / 94.1% | 1.001 / 1.004 |
| [APEX-MTP I-Quality](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) | 23.7 GB | 0.0328 / 0.0402 | 1.310 / 1.435 | 91.5% / 91.1% | 1.009 / 1.004 |
| **Q4_K_M-hq** | 22.4 GB | 0.0339 / 0.0357 | 1.228 / 1.568 | 91.7% / 92.4% | 1.001 / 1.001 |
| [bartowski Q4_K_M](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) | 21.9 GB | 0.0444 / 0.0620 | 1.482 / 2.958 | 90.6% / 89.9% | 0.998 / 1.041 |
| [mradermacher MTP i1-Q4_K_M](https://huggingface.co/mradermacher/Ornith-1.5-35B-A3B-MTP-i1-GGUF) | 21.7 GB | 0.0533 / 0.0606 | 1.644 / 1.627 | 89.2% / 89.5% | 1.012 / 1.015 |
| Q4_K_M (not uploaded) | 21.7 GB | 0.0534 / 0.0738 | 1.368 / 2.621 | 89.1% / 89.4% | 1.007 / 1.057 |
| [official Q4_K_M](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) | 21.7 GB | 0.0565 / 0.0663 | 1.530 / 1.858 | 89.1% / 89.3% | 1.004 / 1.034 |
| **IQ4_XS-exps-hq** | 20.0 GB | 0.0405 / 0.0429 | 1.115 / 1.064 | 91.1% / 92.1% | 1.008 / 0.987 |
| IQ4_XS-exps (not uploaded) | 19.3 GB | 0.0600 / 0.0708 | 1.659 / 1.922 | 88.9% / 89.2% | 1.014 / 1.030 |
| bartowski IQ4_XS | 19.3 GB | 0.0579 / 0.0721 | 1.330 / 2.420 | 89.2% / 89.8% | 1.015 / 1.024 |
| mradermacher MTP i1-IQ4_XS | 19.2 GB | 0.0680 / 0.0741 | 1.628 / 1.694 | 88.0% / 88.4% | 1.015 / 1.004 |
| **IQ3_S-exps-down4-hq** | 18.1 GB | 0.0706 / 0.0906 | 1.786 / 3.383 | 88.3% / 88.5% | 1.040 / 1.074 |
| IQ3_S-exps-down4 (not uploaded) | 17.5 GB | 0.0874 / 0.1322 | 2.130 / 4.271 | 86.4% / 86.5% | 1.042 / 1.126 |
| APEX-MTP I-Compact | 17.4 GB | 0.0889 / 0.1318 | 2.352 / 4.144 | 86.5% / 86.3% | 1.012 / 1.093 |
| bartowski IQ3_M | 17.4 GB | 0.1097 / 0.1357 | 2.667 / 4.541 | 85.7% / 86.3% | 1.058 / 1.097 |
| bartowski Q3_K_L | 17.4 GB | 0.1248 / 0.1474 | 2.678 / 4.053 | 84.8% / 85.1% | 1.084 / 1.076 |
| **IQ3_S-exps-hq** | 16.7 GB | 0.0927 / 0.1056 | 2.499 / 3.725 | 86.9% / 87.5% | 1.059 / 1.084 |
| IQ3_S-exps (not uploaded) | 16.1 GB | 0.1134 / 0.1538 | 2.464 / 4.161 | 84.9% / 85.4% | 1.063 / 1.145 |
| **IQ3_XXS-exps-down3-hq** | 15.7 GB | 0.1008 / 0.1104 | 2.210 / 2.839 | 86.4% / 87.0% | 1.052 / 1.027 |
| mradermacher MTP i1-IQ3_S | 15.6 GB | 0.1590 / 0.1839 | 3.890 / 4.860 | 83.5% / 83.3% | 1.091 / 1.133 |
| **IQ2_S-exps-down3-hq** | 14.4 GB | 0.1523 / 0.1710 | 3.336 / 5.176 | 82.9% / 83.3% | 1.092 / 1.133 |
| APEX-MTP I-Mini | 14.4 GB | 0.2437 / 0.2971 | 4.336 / 7.321 | 79.7% / 78.5% | 1.210 / 1.285 |
| bartowski Q2_K_L | 13.6 GB | 0.2460 / 0.2431 | 4.847 / 6.432 | 78.7% / 80.1% | 1.156 / 1.172 |
| **IQ2_XS-exps-down3-hq** | 13.2 GB | 0.2071 / 0.1969 | 4.118 / 4.431 | 79.6% / 81.9% | 1.096 / 1.103 |
| bartowski IQ2_M | 12.5 GB | 0.2483 / 0.3052 | 4.524 / 7.589 | 78.7% / 78.4% | 1.145 / 1.258 |
| bartowski IQ2_XXS | 10.3 GB | 0.4464 / 0.3606 | 6.225 / 6.656 | 71.8% / 76.2% | 1.345 / 1.221 |

- Each `-hq` file has a lower mean KLD, in both languages, than every published file of the same size or smaller. Q4_K_M-hq and IQ4_XS-exps-hq are also below every published Q4_K_M; the only published file below them is APEX I-Quality, 0.0011 lower than Q4_K_M-hq in Japanese at 1.3 GB more.
- Q8_0 for the non-expert tensors costs 0.6-0.7 GB and lowers mean KLD by 18-52%, as the not-uploaded rows show. Each routed expert runs for 8 of 256 slots, while the non-expert tensors run for every token, so their error weighs more per byte.
- In the 16-18 GB range, IQ3_S-exps-hq (16.7 GB) is 26% (ja) and 28% (en) below bartowski Q3_K_L (17.4 GB). IQ3_XXS-exps-down3-hq is another 1 GB smaller and still below Q3_K_L, with a lower tail than IQ3_S-exps-hq.
- Below 15 GB, IQ2_S-exps-down3-hq and IQ2_XS-exps-down3-hq are 16-42% below the files of similar size.
- Even Q8_0 is further from BF16 here than usual (mean KLD 0.013-0.020; BF16 perplexity is about 8.4 on both texts), so the absolute KLD numbers are higher than for many other models.
- Our imatrix and bartowski's give nearly the same result: IQ3_S-exps-hq made with bartowski's imatrix measured 0.0949 / 0.1001, against 0.0927 / 0.1056 with ours. The gains above come from the mix, not the calibration data.

## How these were made

- llama.cpp `9b56ccf` (CPU build). The BF16 source is the official [`Ornith-1.5-35B-BF16.gguf`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF), which includes the MTP layer.
- Imatrix: computed on Q8_0 over 367 chunks of 512 tokens from [bartowski's calibration_datav3](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8) and [TFMC's ja/en C4 set](https://huggingface.co/datasets/TFMC/imatrix-dataset-for-japanese-llm). About a third of the chunks are wrapped in the model's chat format, including `<think>` and tool-call round trips, so that the chat and tool markers are part of the activations. The MTP layer is not run during imatrix collection.
- Mixed files: Q4_K_M plus the rules in `tensor-types-<name>.txt`, matched in order, first match wins. The expert rules cover layers 0-39 only, so the experts of the MTP layer (`blk.40`) keep the Q4_K_M types.

```bash
llama-quantize --imatrix imatrix.gguf --tensor-type-file tensor-types-IQ3_S-exps-hq.txt \
  Ornith-1.5-35B-BF16.gguf Ornith-1.5-35B-A3B-IQ3_S-exps-hq.gguf Q4_K_M
```

- KLD: `llama-perplexity --kl-divergence` against BF16 logits, context 1024, on Japanese Wikipedia (wiki40b-ja test, 16 chunks) and English Wikipedia (wikitext-2 test, 8 chunks). Neither text is in the calibration data. The other repos' files were downloaded in October 2026 and measured with the same build and settings.

## License and credits

MIT License, the same as the original model.
