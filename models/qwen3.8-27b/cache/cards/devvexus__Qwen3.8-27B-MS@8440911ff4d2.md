---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- quantized
- llama.cpp
- ms
- madshift
---

# Same size as unsloth. 1.6-2.3x lower quality loss.

**MS (Mad Shift)** quants of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), for the [ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp) fork.

- **Accuracy:** 1.61-2.31x lower KL divergence than unsloth's UD3 file of the same size (1): 1.61-1.63x for MS2.7-MS3.5, 1.90-2.31x for MS3.7-MS5.
- **Top-1:** the most likely next token matches the original BF16 model's 93.8-98.7% of the time, vs 91.8-97.9% for unsloth's UD3 file of the same size: 22-35% fewer mismatches (1).
- **vs EXL3:** 1.81-2.50x lower KL divergence than turboderp's EXL3 files at the same size (the better of the standard and SCP file at each size) for MS2.7-MS4.3; MS5 (19.41 GB) is lower than every EXL3 file measured (largest 18.77 GB) (5).
- **Speed:** generation about the same as unsloth (98-100% on an RTX Spark, 97% on an RTX 4080 SUPER) (2)(4). MTP doubles it: 2.1x faster than without MTP, MS4.3 32.4 tokens/s, MS2.7 46.6 tokens/s on an RTX Spark (3); MS2.7 101 vs 97 tokens/s for unsloth's same-size file with their separate MTP file on an RTX 4080 SUPER (4).
- **Memory:** every file no larger than its unsloth file, except MS4.3 (+0.09 GB) (1). MTP adds about 0.9-1.0 GiB, about as much as the MTP block inside unsloth's files (their separate MTP file adds 1.6-1.7 GiB) (3).
- **MTP:** our own MS-coded MTP file, 0.30 GB; one file serves every size (it borrows the loaded file's token table and output head). Against the MTP block inside unsloth's files, MS2.7 is 16% faster with less GPU memory, and MS4.3 is about as fast, with 274 MiB more GPU memory (3).

<sub>(1) KL divergence from the original BF16 model, mean over four English texts; top-1 = how often the quantized model's most likely next token is the BF16 model's. UD3 = Unsloth Dynamic 3, unsloth's files at revision `4ca72078` (August 2026); their sizes without their 351 MB MTP block. (2) Chat benchmark without MTP on an RTX Spark, MS and unsloth files both on ms-llama.cpp. (3) RTX Spark (GB10), 20 chat prompts x 256 tokens, greedy, one session on the released build; unsloth's files with the MTP block inside the file and with their separate MTP file, also on ms-llama.cpp; GPU memory from nvidia-smi. (4) RTX 4080 SUPER 16 GB, same benchmark, x64 release build, medians of 3; unsloth's UD-IQ3_XXS with their separate MTP file reaches 97.1 tokens/s there (the built-in block was not measured there). (5) turboderp/Qwen3.8-27B-exl3, standard and SCP branches, scored in exllamav3 against the same BF16 logits and tokens; EXL3 values between files read on a log scale; exllamav3 built from source for Windows on ARM. See *Other trellis quants*.</sub>

![KL divergence vs size](charts/qwen3.8-27b_kld.png)

![Top-1 agreement vs size](charts/qwen3.8-27b_top1.png)

![MTP: one MS file for every size vs unsloth's](charts/qwen3.8-27b_mtp.png)

**MS (Mad Shift)** was created by **Matthew Priester** ([GitHub](https://github.com/devvexus), [ORCID](https://orcid.org/0009-0000-0032-5292)). If you use these files or build on the format, please credit *MS (Mad Shift) by Matthew Priester*.

## Run it now

MS files need the MS fork of llama.cpp, [ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp): stock llama.cpp, Ollama and LM Studio cannot load them.

**1. Install the engine** (Windows, NVIDIA GPU; once; PowerShell):

```powershell
$d="$env:LOCALAPPDATA\ms-llama"; $z=if($env:PROCESSOR_ARCHITECTURE -eq "ARM64"){"arm64-cuda"}else{"x64-cuda12"}; ni -ItemType Directory -Force $d|Out-Null; curl.exe -fL -o "$d\ms.zip" "https://github.com/devvexus/ms-llama.cpp/releases/latest/download/ms-llama-win-$z.zip"; if($?){tar -xf "$d\ms.zip" -C $d}
```

Linux and macOS: build from source, see the [ms-llama.cpp README](https://github.com/devvexus/ms-llama.cpp#readme).

**2. Run the file that fits your GPU.** It downloads once, then serves a chat page and an API on http://localhost:8080.

16 GB (RTX 4070 Ti Super, 4080, 5070 Ti, 5080): **MS2.7**, 10.6 GB

```powershell
& "$env:LOCALAPPDATA\ms-llama\llama-server.exe" -hf devvexus/Qwen3.8-27B-MS:MS2.7 --spec-type draft-mtp -c 32768 -np 1 --port 8080
```

24 GB (RTX 3090, 4090): **MS4.3**, 17.3 GB

```powershell
& "$env:LOCALAPPDATA\ms-llama\llama-server.exe" -hf devvexus/Qwen3.8-27B-MS:MS4.3 --spec-type draft-mtp -c 32768 -np 1 --port 8080
```

32 GB or more (RTX 5090, RTX Spark, DGX Spark): **MS5**, 19.4 GB

```powershell
& "$env:LOCALAPPDATA\ms-llama\llama-server.exe" -hf devvexus/Qwen3.8-27B-MS:MS5 --spec-type draft-mtp -c 32768 -np 1 --port 8080
```

`-c 32768` gives a 32k-token context; raise it if your GPU has room. Set it yourself, as with stock llama.cpp: on Windows the automatic sizing can overestimate a display GPU's free memory, and a model that spills into system RAM runs very slowly. `-np 1` serves one chat at a time: every extra parallel slot needs its own copy of the model's recurrent state, which can push layers off the GPU and slow generation down a lot. Leaving out `--spec-type draft-mtp` saves about 1 GB. Any other file by name: `-hf devvexus/Qwen3.8-27B-MS --hf-file MS3.7/Qwen3.8-27B-MS3.7.gguf` (the tags `:MS3` and `:MS4` would also match MS3.5 and MS4.3). A .gguf you downloaded yourself: `-m <file>.gguf -md mtp-Qwen3.8-27B-MS.gguf --spec-type draft-mtp`. After the first run, you can add `--offline` so the server starts without checking Hugging Face; that check can take a few minutes. On Windows the first download may print 'failed to create symlink ... switching to degraded mode'; it is harmless: the Hugging Face cache just can't use symlinks, so updated files may take extra disk space.

**Images (optional).** The vision encoder, `vision/mmproj-Qwen3.8-27B-F16.gguf` (0.93 GB, F16), lets the model read images. `-hf` does not load it by itself: add `--mmproj-url` with its address, or `--mmproj <file>` for a downloaded copy. It works with every MS file and with MTP, and adds about 1.1-1.4 GiB of GPU memory when on (nothing when off). The vision encoder is not MS-encoded and was not part of our evaluation; MS-encoding it is future work. For example:

```powershell
& "$env:LOCALAPPDATA\ms-llama\llama-server.exe" -hf devvexus/Qwen3.8-27B-MS:MS2.7 --spec-type draft-mtp -c 32768 -np 1 --port 8080 --mmproj-url https://huggingface.co/devvexus/Qwen3.8-27B-MS/resolve/main/vision/mmproj-Qwen3.8-27B-F16.gguf
```

## Use it in your apps

While the server runs:

- **Claude Code:** `$env:ANTHROPIC_BASE_URL="http://localhost:8080"; $env:ANTHROPIC_AUTH_TOKEN="local"; claude` Claude Code's own prompt is large (about 24k tokens in our test with Claude Code 2.1.325), so give the server more context for it: `-c 49152` instead of `-c 32768` (with `-np 1`, MS2.7 still fits a 16 GB card whose GPU isn't busy with other apps).
- **OpenAI-compatible apps:** base URL `http://localhost:8080/v1`, any API key.
- **Apps that run llama.cpp themselves:** point them at `%LOCALAPPDATA%\ms-llama`.
- Ollama and LM Studio can't load MS files.

## Files

| MS file | Size | KLD | UD3 file (size) | UD3 KLD | MS is | Top-1, MS / UD3 |
|---|---|---|---|---|---|---|
| **MS5** | 19.41 GB | 0.001531 | UD-Q5_K_M (19.42 GB) | 0.003539 | **2.31x** closer | 98.66% / 97.95% |
| **MS4.3** | 17.30 GB (+0.09) | 0.002984 | UD-Q4_K_XL (17.21 GB) | 0.006457 | **2.16x** closer | 98.20% / 97.28% |
| **MS4** | 16.11 GB | 0.004417 | UD-Q4_K_M (16.11 GB) | 0.008994 | **2.04x** closer | 97.78% / 96.79% |
| **MS3.7** | 15.00 GB | 0.006226 | UD-Q4_K_S (15.01 GB) | 0.011821 | **1.90x** closer | 97.41% / 96.34% |
| **MS3.5** | 13.89 GB | 0.010001 | UD-IQ4_XS (13.90 GB) | 0.016138 | **1.61x** closer | 96.57% / 95.59% |
| **MS3** | 12.79 GB | 0.015699 | UD-Q3_K_XL (12.80 GB) | 0.025538 | **1.63x** closer | 95.91% / 94.49% |
| **MS2.7** | 10.58 GB | 0.034064 | UD-IQ3_XXS (10.58 GB) | 0.055020 | **1.62x** closer | 93.77% / 91.81% |

Each quant is in its own folder: `<quant>/Qwen3.8-27B-<quant>.gguf`; the MTP file `mtp-Qwen3.8-27B-MS.gguf` (0.30 GB) is in the top folder; the optional vision encoder is `vision/mmproj-Qwen3.8-27B-F16.gguf` (0.93 GB, F16, see *Images*). Sizes in GB (10^9 bytes). Compared against Unsloth Dynamic 3 (UD3): unsloth's files at revision `4ca72078` (August 2026), their sizes without their 351 MB MTP block (see *How the comparison was made*). Every MS file is no larger than its UD3 file except MS4.3 (0.09 GB larger). The tensors MS does not code (input embedding table, norms, small SSM tensors) are unsloth's, copied unchanged; see *How these files were made*. Checksums: `SHA256SUMS`.

<details>
<summary>Per-text KL divergence and every UD3 file</summary>

KL divergence per text and top-1 agreement (how often the quantized model's most likely next token is the BF16 model's):

| MS file | wiki | web | code | math | Top-1 |
|---|---|---|---|---|---|
| MS5 | 0.00173 | 0.00129 | 0.00192 | 0.00119 | 98.66% |
| MS4.3 | 0.00311 | 0.00241 | 0.00392 | 0.00250 | 98.20% |
| MS4 | 0.00449 | 0.00363 | 0.00630 | 0.00325 | 97.78% |
| MS3.7 | 0.00628 | 0.00515 | 0.00880 | 0.00468 | 97.41% |
| MS3.5 | 0.01060 | 0.00826 | 0.01309 | 0.00806 | 96.57% |
| MS3 | 0.01658 | 0.01372 | 0.02177 | 0.01072 | 95.91% |
| MS2.7 | 0.03671 | 0.03305 | 0.04373 | 0.02277 | 93.77% |

unsloth's UD3 files on the same eval (revision `4ca72078`, August 2026):

| UD3 file | Size (without MTP) | Mean KLD | wiki | web | code | math | Top-1 |
|---|---|---|---|---|---|---|---|
| [UD-Q5_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q5_K_M.gguf) | 19.77 (19.42) | 0.003539 | 0.00336 | 0.00281 | 0.00465 | 0.00334 | 97.95% |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) | 17.56 (17.21) | 0.006457 | 0.00630 | 0.00535 | 0.00857 | 0.00562 | 97.28% |
| [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | 16.46 (16.11) | 0.008994 | 0.00892 | 0.00757 | 0.01233 | 0.00715 | 96.79% |
| [UD-Q4_K_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_S.gguf) | 15.36 (15.01) | 0.011821 | 0.01223 | 0.01032 | 0.01572 | 0.00902 | 96.34% |
| [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf) | 14.25 (13.90) | 0.016138 | 0.01834 | 0.01428 | 0.02104 | 0.01089 | 95.59% |
| [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q3_K_XL.gguf) | 13.15 (12.80) | 0.025538 | 0.02711 | 0.02310 | 0.03457 | 0.01738 | 94.49% |
| [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_XXS.gguf) | 10.93 (10.58) | 0.055020 | 0.06289 | 0.05482 | 0.06638 | 0.03599 | 91.81% |

</details>

## Memory per file

Memory needed to run each file fully on a GPU (or in RAM on CPU), approximate, with the default batch size. The context cache grows with the context length; a smaller `-ub` lowers the compute buffers. With MTP on, add about 1 GB.

| Quant | File size | Memory at 8k context | Memory at 32k context | Free memory to have (32k) |
|---|---|---|---|---|
| **MS5** | 19.41 GB | ~21.6 GB | ~23.2 GB | 32 GB |
| **MS4.3** | 17.30 GB | ~19.5 GB | ~21.1 GB | 24 GB |
| **MS4** | 16.11 GB | ~18.3 GB | ~19.9 GB | 24 GB |
| **MS3.7** | 15.00 GB | ~17.2 GB | ~18.8 GB | 20 GB |
| **MS3.5** | 13.89 GB | ~16.1 GB | ~17.7 GB | 20 GB |
| **MS3** | 12.79 GB | ~15.0 GB | ~16.6 GB | 20 GB |
| **MS2.7** | 10.58 GB | ~12.8 GB | ~14.4 GB | 16 GB |

## MTP (faster generation)

| Model + MTP | tokens/s | speedup | drafts accepted | extra GPU memory |
|---|---|---|---|---|
| MS4.3, no MTP | 15.2 | - | - | - |
| **MS4.3 + ours** | **32.4** | 2.12x | 62.9% | +1.05 GiB |
| UD-Q4_K_XL, no MTP | 15.3 | - | - | - |
| UD-Q4_K_XL + MTP block inside the file | 31.6 | 2.07x | 64.5% | +0.90 GiB |
| UD-Q4_K_XL + unsloth's separate MTP file | 33.8 | 2.22x | 63.8% | +1.62 GiB |
| MS2.7, no MTP | 22.5 | - | - | - |
| **MS2.7 + ours** | **46.6** | 2.07x | 65.9% | +0.90 GiB |
| UD-IQ3_XXS, no MTP | 22.7 | - | - | - |
| UD-IQ3_XXS + MTP block inside the file | 40.3 | 1.77x | 62.0% | +0.93 GiB |
| UD-IQ3_XXS + unsloth's separate MTP file | 42.1 | 1.85x | 62.0% | +1.68 GiB |

With MTP on, MS4.3 runs at 102% of unsloth's speed with the MTP block inside their file and 96% with their separate MTP file; MS2.7 at 116% and 111%. One file, `mtp-Qwen3.8-27B-MS.gguf` (0.30 GB), serves every tier: it holds only the MTP block (from Qwen's original MTP weights, MS-coded; no unsloth tensors) and borrows the loaded tier's token table and output head, while unsloth's `mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB) carries its own 3-bit copies. The loaded tier checks every drafted token: the output follows the output without MTP almost everywhere (99.9% same top token for MS4.3). Setup as in footnote (3), draft length 3, on the released ms-llama.cpp build; GiB = 1024 MiB. All ten runs: one session, single runs, unsloth and MS alternating.

On an RTX 4080 SUPER (16 GB), same benchmark, x64 release build, medians of 3, with unsloth's separate MTP file (the built-in block was not measured there):

| Model + MTP file | tokens/s without MTP | with MTP | speedup | drafts accepted | extra GPU memory |
|---|---|---|---|---|---|
| **MS2.7 + ours** | 51.1 | **101.2** | 1.98x | 65.7% | +0.91 GiB |
| UD-IQ3_XXS + unsloth's separate MTP file | 53.0 | 97.1 | 1.83x | 62.3% | +1.67 GiB |

MS2.7 runs at 104% of unsloth's speed with their separate MTP file there.

## Speed

Without MTP, on an NVIDIA RTX Spark N1X laptop GPU (GB10, Blackwell, 6144 cores), Windows 11 on ARM; MS and unsloth files both on ms-llama.cpp, each MS file against the UD3 file of the same size:

| MS file | unsloth UD3 file | Generation, tokens/s | Prompt processing, tokens/s |
|---|---|---|---|
| **MS4.3** | UD-Q4_K_XL | 14.74 vs 14.80 (**1.00x**) | 918 vs 970 (**0.95x**) |
| **MS4** | UD-Q4_K_M | 15.75 vs 15.91 (**0.99x**) | 918 vs 999 (**0.92x**) |
| **MS3** | UD-Q3_K_XL | 19.00 vs 19.10 (**0.99x**) | 949 vs 953 (**1.00x**) |
| **MS2.7** | UD-IQ3_XXS | 21.36 vs 21.84 (**0.98x**) | 977 vs 919 (**1.06x**) |
| **MS2.7**, RTX 4080 SUPER | UD-IQ3_XXS | 51.13 vs 52.95 (**0.97x**) | 2018 vs 2301 (**0.88x**) |

- **Generation:** about the same as unsloth (98-100%) on the RTX Spark, 97% on the RTX 4080 SUPER, in the chat benchmark of *MTP* (20 chat prompts x 256 tokens, greedy; the MS2.7 pairs are medians of 3 runs). With MTP on, see *MTP*.
- **Prompt processing:** 0.92x to 1.06x on the RTX Spark, `llama-bench -p 512 -n 0 -ngl 99`, MS and unsloth files alternating in one session (median of 3). At the larger files this is a kernel limit on that machine. RTX 4080 SUPER: 0.88x with the default settings (f16 sums on GeForce), median of 3.
- **Sums:** Quality numbers use f32 sums; GeForce cards default to f16 sums, which read prompts faster and measured the same mean KLD (Qwen3.5-2B MS4, wiki text: 0.003233 with either; top-1 97.25% vs 97.26%).

## Other trellis quants

ik_llama.cpp has its own trellis quant types (the KT types). We measured Thireus's Qwen3.8-27B KT files the same way; at about the same size, MS files have lower KL divergence:

| File | File size | Mean KLD | Top-1 | Nearest MS files (size, difference, mean KLD) |
|---|---|---|---|---|
| [IQ2_KT](https://huggingface.co/Thireus/Qwen3.8-27B-THIREUS-IQ2_KT-SPECIAL_SPLIT/tree/5cc56d32f9b02045d579a7412158a787f80cfa2c) | 7.18 | 0.232825 | 81.41% | none within 1 GB |
| [IQ3_KT](https://huggingface.co/Thireus/Qwen3.8-27B-THIREUS-IQ3_KT-SPECIAL_SPLIT/tree/2e08ef42995273e297ed4c5d6348b857b386a039) | 10.54 | 0.061664 | 90.68% | MS2.7 (10.58, +0.04 GB, 0.034064) |
| [IQ4_KT](https://huggingface.co/Thireus/Qwen3.8-27B-THIREUS-IQ4_KT-SPECIAL_SPLIT/tree/6fb037a2e7704d9652847b74632206ae9169d311) | 13.49 | 0.020679 | 94.59% | MS3 (12.79, -0.70 GB, 0.015699); MS3.5 (13.89, +0.41 GB, 0.010001) |

KT files on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) `fcf7b3647` against the same BF16 reference logits; its own BF16 run scores 0.000285 against them (32 wiki chunks).

[EXL3](https://github.com/turboderp-org/exllamav3) (exllamav3) is a trellis format for GPUs. We measured turboderp's Qwen3.8-27B EXL3 files, the standard branches and the newer SCP branches, against the same BF16 reference logits and tokens; at about the same size, MS files have lower KL divergence:

| File | Size | Mean KLD | Top-1 | Nearest MS files (size, difference, mean KLD) |
|---|---|---|---|---|
| [SCP_2.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/98f2f8dfab4cb73ed1ae3428a1671483ab05c490) | 9.17 | 0.155958 | 86.38% | none within 1 GB |
| [2.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/5fce94d8233690102891329ac3dfca626ece4807) | 9.64 | 0.148783 | 86.77% | MS2.7 (10.58, +0.94 GB, 0.034064) |
| [SCP_2.20bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/08ade2fd2935ee6ef07ee3d10e29176cced7ac4f) | 9.77 | 0.120146 | 88.10% | MS2.7 (10.58, +0.81 GB, 0.034064) |
| [2.50bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/0cd1091272291d008e06976acb81e25946cdc767) | 11.16 | 0.077013 | 90.59% | MS2.7 (10.58, -0.58 GB, 0.034064) |
| [SCP_3.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/550c12cf3cad8d787c70ec7db38a5682fee8c886) | 12.37 | 0.039869 | 93.31% | MS3 (12.79, +0.42 GB, 0.015699) |
| [3.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/6fe61ad620abfe97c5b49f9722c2bceeea4ccc28) | 12.69 | 0.036295 | 93.78% | MS3 (12.79, +0.10 GB, 0.015699) |
| [3.50bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/8351c54ef11a63e38bca978655086d7998b8a7f9) | 14.20 | 0.021292 | 95.14% | MS3.5 (13.89, -0.31 GB, 0.010001); MS3.7 (15.00, +0.80 GB, 0.006226) |
| [SCP_4.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/387b844556856493893ed14f72bca8b2aee5b8b4) | 15.57 | 0.010626 | 96.49% | MS3.7 (15.00, -0.57 GB, 0.006226); MS4 (16.11, +0.54 GB, 0.004417) |
| [4.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/113cf7ab958054860e43fb7f3063b1af19171095) | 15.73 | 0.010195 | 96.63% | MS3.7 (15.00, -0.73 GB, 0.006226); MS4 (16.11, +0.38 GB, 0.004417) |
| [5.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/a35e75a73baee51da709329d19294245cbeeb5d8) | 18.77 | 0.002976 | 98.16% | MS5 (19.41, +0.64 GB, 0.001531) |
| [SCP_5.00bpw](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/b9e2c51f4e901c7ed547d21b204fb2df81ec4262) | 18.77 | 0.003109 | 98.13% | MS5 (19.41, +0.64 GB, 0.001531) |

EXL3 sizes are the text model without the vision tower or MTP layers, like the MS sizes. Standard files on exllamav3 v0.0.28, SCP files (made with 1.6.7) on 1.6.7. exllamav3 was built from source for Windows on ARM; the scores did not change across three exllamav3 versions (3.00bpw wiki: 0.039926, 0.039883, 0.039888). exllamav3 with the BF16 weights scores 0.000376 against llama.cpp's BF16 logits (mean of the four texts); the EXL3 values include it.

## How the comparison was made

KL divergence (KLD) and top-1 agreement against the BF16 model, measured with `llama-perplexity --kl-divergence` (512-token context, 128 chunks per text, the second half of each chunk scored: 255 tokens per chunk, 32,640 per text) on four English texts that were not used to make the files: **wiki** (Wikipedia articles), **web** (FineWeb-Edu), **code** (real code patches from SWE-Gym) and **math** (MATH test problems and solutions). The headline number is the mean over the four texts; lower KLD is better. Top-1 agreement is how often the quantized model's most likely next token is the BF16 model's. The texts, with their sources and licenses, are published as [devvexus/ms-eval-texts](https://huggingface.co/datasets/devvexus/ms-eval-texts) (the math text as a script that rebuilds it).

**Why KL divergence:** it compares the quantized model's whole next-token distribution with the original model's at every position of held-out text. It is not a task score: it measures how closely a file follows the original model on held-out text.

Compared against Unsloth Dynamic 3 (UD3): [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at revision `4ca72078` (August 2026), for every tier, measured on stock llama.cpp [`36a0e2ffc`](https://github.com/ggml-org/llama.cpp/commit/36a0e2ffc0b352b774f8d71c43668b75b3061ef5). MS files: measured on [ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp) [`6085028c3`](https://github.com/devvexus/ms-llama.cpp/commit/6085028c3305e6ea8acf0c1600d89ca01d63a9aa) with `GGML_CUDA_MMMS_ACC=f32`. Both against the same BF16 reference logits. All numbers are our own measurements.

**Sizes are compared without the MTP block.** Qwen3.8-27B ships a multi-token-prediction (MTP) layer used only for speculative decoding. unsloth's files each include it (351 MB); MS model files do not (MTP is one separate file, see *MTP*). unsloth's own charts plot size without MTP, and so do ours: every MS file is no larger than the unsloth file it is compared with, after removing that file's MTP block, except MS4.3 (0.09 GB larger than the August UD-Q4_K_XL).

A few text passages dominate their text's mean for every quant (for example drawing code inside a math solution).

**Check it yourself:** any English text works. Make reference logits with the BF16 model once, then compare any quant against them (MS files with the fork's build, other GGUFs with the same build or stock llama.cpp):

```bash
./build/bin/llama-perplexity -m Qwen3.8-27B-BF16.gguf -f text.txt -c 512 --kl-divergence-base base.kld
./build/bin/llama-perplexity -m Qwen3.8-27B-MS4.3.gguf -f text.txt -c 512 --kl-divergence-base base.kld --kl-divergence
```

## What MS is

**MS (Mad Shift)** is a GGUF weight format for llama.cpp. MS uses the bitshift-trellis construction and the 1MAD decoding function described in **QTIP**
([Tseng, Sun, Hou and De Sa, NeurIPS 2024](https://arxiv.org/abs/2406.11235)).

Every large matrix is stored trellis-coded after a random Hadamard rotation (undone at run time); a tensor with k bits
(k = 1..6) costs k + 0.0625 bits per weight. The fork has CPU and CUDA kernels for it. The files are made with the bitshift-trellis
construction described in QTIP + our own calibration-aware, layer-by-layer conversion.

**Names:** N in MSN is a size class, not the average bits per weight.

## How these files were made

- **Conversion:** the bitshift-trellis construction described in QTIP + our own calibration-aware, layer-by-layer conversion.
- **Make your own MS files:** these files are made with the MS converter, open source (Apache 2.0): [mad-shift-converter](https://github.com/devvexus/mad-shift-converter). A lightweight MS encoder for ms-llama.cpp's `llama-quantize`, built for fast conversion, is in development, so that anyone can make MS files of most models on their own hardware.
- **Output head:** split in two, the most frequent tokens at higher precision.
- **MTP file:** the MTP block only, made from Qwen's original MTP weights (see *MTP*).
- **Vision encoder:** `vision/mmproj-Qwen3.8-27B-F16.gguf`, converted from Qwen's original weights with llama.cpp's `convert_hf_to_gguf.py` (`--mmproj --outtype f16`); not MS-encoded, no unsloth tensors.
- **Taken unchanged from unsloth's files:** every tensor that is not MS-coded (the input embedding table, the norms and the small SSM and convolution tensors) comes from unsloth's GGUFs ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), Apache 2.0): the input embedding table from UD-Q4_K_M (MS2.7: UD-Q3_K_XL), every other one from unsloth's GGUF of the matching size. Each file names its sources in its `msffn.token_embd_source` and `msffn.copy_source` metadata.

<details>
<summary>Source file and revision of the unsloth tensors in each MS file</summary>

| MS file | Input embedding table from | Other non-MS tensors from | unsloth revision |
|---|---|---|---|
| MS5 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-Q5_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q5_K_M.gguf) | `4ca72078` |
| MS4.3 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/ce331efc48d5722b4b8041f72793e530eae18d9b/Qwen3.8-27B-UD-Q4_K_XL.gguf) | `4ca72078`, `ce331efc` |
| MS4 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | `4ca72078` |
| MS3.7 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-Q4_K_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_S.gguf) | `4ca72078` |
| MS3.5 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf) | `4ca72078` |
| MS3 | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q3_K_XL.gguf) | `4ca72078` |
| MS2.7 | [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q3_K_XL.gguf) | [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_XXS.gguf) | `4ca72078` |

</details>

## Limitations

- The vision encoder is not MS-encoded and was not part of our evaluation (see *Images*).
- MTP comes as a separate file (see *MTP*).
- Needs the MS fork; GPU (CUDA) and CPU (x86 and ARM) are supported there.
- Accuracy was measured on English text; the split output head keeps rarely used tokens (much of the non-English vocabulary) at lower precision.

## Citation

**Technical report:** Priester, M. (2026). MS (Mad Shift): Trellis-Coded GGUF Quantization for llama.cpp (Version 1.0). Zenodo. [https://doi.org/10.5281/zenodo.24145285](https://doi.org/10.5281/zenodo.24145285)

Please cite MS (Mad Shift). MS uses the bitshift-trellis construction and the 1MAD decoding function described in QTIP; please cite QTIP for them:

```bibtex
@misc{priester2026madshift,
  author       = {Matthew Priester},
  title        = {{MS (Mad Shift)}: trellis-coded GGUF quantization for llama.cpp},
  year         = {2026},
  howpublished = {\url{https://github.com/devvexus/ms-llama.cpp}}
}
@inproceedings{tseng2024qtip,
  title     = {{QTIP}: Quantization with Trellises and Incoherence Processing},
  author    = {Tseng, Albert and Sun, Qingyao and Hou, David and De Sa, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

## License

The model files are licensed under Apache 2.0, like the base model
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Copyright 2026 Alibaba Cloud): see `LICENSE` and `NOTICE`.
They are modified (quantized) versions of Qwen3.8-27B and contain tensors copied from unsloth's GGUFs (Apache 2.0; see
*How these files were made*). The MS engine, [ms-llama.cpp](https://github.com/devvexus/ms-llama.cpp), is MIT-licensed. MS (Mad Shift) was
created by Matthew Priester; please keep the `NOTICE` file when you redistribute these files. Checksums: `SHA256SUMS`.

This project is not affiliated with or endorsed by ggml-org/llama.cpp, Qwen/Alibaba, unsloth or NVIDIA. Other product
names (Claude Code, Ollama, LM Studio and others) are trademarks of their owners, named only to describe compatibility.
