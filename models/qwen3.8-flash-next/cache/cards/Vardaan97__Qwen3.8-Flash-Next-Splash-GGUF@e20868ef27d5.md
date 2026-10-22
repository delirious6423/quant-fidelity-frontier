---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: image-text-to-text
quantized_by: Vardaan97
tags:
- gguf
- imatrix
- qwen4exp
- moe
- splash
- apple-silicon
- dflash2
- speculative-decoding
- vision
- mixed-precision
---

# Qwen3.8-Flash-Next · Splash GGUF

**Two GGUF builds of Qwen3.8-Flash-Next for [Splash](https://github.com/incoai/splash) on 128 GB Apple-silicon Macs: a 4-bit and a 5-bit, each with a choice of PLE table.**

- **Both fit a 128 GB Mac.** The Hub's hardware panel lists each build once, at the size it needs in memory.
- **Tested.** All four folders installed from this repository and passed a smoke test on stock Splash 1.4.0 (results below).
- **Splash sets itself up.** It adds its DFlash2 draft for speculative decoding by itself, and vision is on by default.

## Quick start

```bash
brew install incoai/tap/splash
splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Q4-PLE8
```

Swap the part after `:` for any folder in the table below.

- **First start.** Splash downloads that folder and the vision projector (0.91 GB) from this repository. Then it fetches the DFlash2 draft it releases for Qwen3.8-Flash-Next, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (1.21 GB). Later starts reuse the downloads.
- **No `--draft-model` is needed.** Splash reads the model family from the GGUF and picks the matching draft.
- **Text only:** add `--language-only`.
- **Coding agents:** once Splash prints `Ready`, run `splash opencode`, `splash claude` or `splash codex` in another terminal.

## Which one to pick

| Folder (`:VARIANT`) | Body | PLE table | In memory (Hub panel size) | Download | Splash peak | KLD vs BF16 | Same top-1 % |
|---|---|---|---:|---:|---:|---:|---:|
| **`Q4-PLE8`** (default) | UD-Q4_K_XL | Q8_0 | 82.5 GB | 136.9 GB | 80.9 GiB | 0.021037 | 94.807 |
| `Q4-PLE4` | UD-Q4_K_XL | IQ4_NL | 82.5 GB | 111.3 GB | 80.9 GiB | 0.022283 | 94.658 |
| **`Q5-PLE8`** | Q5 (Q5_K experts) | Q8_0 | 92.6 GB | 147.0 GB | 90.3 GiB | **0.013842** | **95.741** |
| `Q5-PLE4` | Q5 (Q5_K experts) | IQ4_NL | 92.6 GB | 121.4 GB | 90.3 GiB | 0.015382 | 95.558 |

- **Our default is `Q4-PLE8`.** It has the fastest decode and the smallest memory footprint.
- **For the least error vs BF16, pick `Q5-PLE8`.** Its KLD is about a third lower than the default's, and it needs 10 GB more memory.
- **PLE8 or PLE4.** The table choice changes the download and slightly changes quality, but not the memory needed. The Q8_0 table (PLE8) is 54.4 GB and the IQ4_NL table (PLE4) is 28.8 GB. Pick PLE4 for a smaller download, or PLE8 for slightly lower error.
- **The tags are approximate.** The Hub lists the Q4 build as `UD-Q4_K_XL` and the Q5 build as `Q5_K_M`, the nearest standard names for these mixed-precision bodies.

### How to read the Hub's hardware panel

- **The panel lists one entry per body,** at its in-memory size: 82.5 GB (`UD-Q4_K_XL`) and 92.6 GB (`Q5_K_M`). On a 128 GB Mac it marks both as fitting.
- **The PLE table isn't in that size, because it isn't held in memory.** Splash reads the few rows each token needs straight from the SSD through a small cache. The table still downloads: the Download column above gives the full size.
- **The table file in each folder is named without a quant tag** (`…-ple8-table-…` or `…-ple4-table-…`), so the panel counts the body alone. The PLE4 folders' body files are byte-identical to the PLE8 folders'. They're named without a tag too, so the panel doesn't list them a second time.
- **Every measurement on this card is from one M5 Max 128 GB with the GPU wired limit at 110,592 MB** (`iogpu.wired_limit_mb`). A Mac at the default limit is untested.

## How the Q5 body was made

The Q5 body is our Balanced body, trimmed to fit the panel's 128 GB line without re-quantizing anything:

- **The swap.** 15 of Balanced's Q8_0 expert down projections were replaced, tensor for tensor, by the Q5_1 down projections of our Quality body. Those are layers 20, 23, 31, 32 and 35–45.
- **What stays Q8_0.** The down projections of layers 2, 4, 30, 46 and 47 keep Q8_0, as do the same layers in unsloth's own UD-Q4_K_XL.
- **Result.** The body shrank from 96.5 GB to 92.6 GB. Its KLD vs BF16 (PLE4 table) went from 0.014940 to 0.015382.

We also built the same trim on unsloth's UD-Q5_K_XL, replacing its Q8_0 down projections with unsloth's own Q5_1 tensors from UD-Q4_K_XL. It scored within noise of this body (0.013957 with the PLE8 table, 0.015431 with PLE4), so we publish one Q5 build. The untrimmed UD-Q5_K_XL (KLD 0.012496, 103.9 GB in memory) stays in [its own repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF).

## Speed with Splash and DFlash2 (M5 Max 128 GB)

The Q4 body's tensors are byte-identical to the build timed below; only the file layout changed.

| Run | Q4-PLE8 |
|---|---:|
| Stock Splash 1.4.0, 2026-10-21, 3 launches: 1 stream, tok/s per stream | 135.9–139.1 |
| Stock Splash 1.4.0: 4 concurrent streams, total tok/s | 183.2–186.9 |
| Stock Splash 1.4.0: agent-style turns, 1 stream, tok/s per stream | 150.3–150.6 |

**Settings for every run:**
- 48 greedy prompts per concurrency level, covering code, math, multilingual, prose, reasoning and tool JSON. Zero errors.
- 176K context, int8 KV, fp32 state, SSD cache off, text only.

**The Q5 body hasn't been timed yet.** For reference, the untrimmed UD-Q5_K_XL decoded 125.7 tok/s per stream in the same kind of run, against 134.0 for Q4 ([receipt](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF/blob/main/evidence/dflash2-speed-2026-10-20.md)).

### Install from this repository with Splash 1.4.0

<!-- E2E-140 -->
**Passed, 2026-10-23.** This ran on stock Splash 1.4.0 (`1482f6bb`) on an M5 Max 128 GB with the wired limit at 110,592 MB.

- **How it ran:** exactly as a user runs it, `splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:FOLDER`, with Splash's defaults (vision on, automatic memory and context) plus `--max-cache-disk 0`.
- **Which files:** byte-identical files on a staging branch of this repository, before they moved to `main`. Every SHA-256 matches [sha256sums.txt](sha256sums.txt).
- **Downloads:** none. The Hugging Face cache already held these exact files, matched by SHA-256, and the DFlash2 draft.
- **Context:** in every launch, Splash's automatic plan held the model's full 262,144-token window, with vision and the DFlash2 draft loaded.

| Folder | Parts Splash selected | Process start to `Ready` | Served context | Splash Metal peak | Checks passed |
|---|---:|---:|---:|---:|---|
| `Q4-PLE8` | 5 | 110 s | 256K | 80.9 GiB | 4/4 |
| `Q4-PLE4` | 5 | 89 s | 256K | 80.9 GiB | 4/4 |
| `Q5-PLE8` | 8 | 119 s | 256K | 90.3 GiB | 4/4 |
| `Q5-PLE4` | 8 | 101 s | 256K | 90.3 GiB | 4/4 |

The checks:

- 17 × 23 answered 391.
- A code planted in an ~8K-token prompt was retrieved.
- Two `get_weather` calls (Paris and Tokyo, celsius) came back as parsed tool calls.
- A synthetic receipt image was read into a JSON schema with all 6 fields correct.

The time to `Ready` includes loading the weights. Peak memory is Splash's Metal allocation high-water mark after the checks, not process RSS.

## Distance from BF16 (KLD and perplexity)

Each variant was measured against the same BF16 reference on neutral text: 87 × 4096-token chunks, 178,089 scored tokens, base PPL 4.045181. The scorer and its arguments are the same for every row.

| Variant | Mean KLD | SE | Same top-1 % | PPL(Q)/PPL(base) |
|---|---:|---:|---:|---:|
| `Q5-PLE8` | 0.013842 | 0.000131 | 95.741 | 1.002088 |
| `Q5-PLE4` | 0.015382 | 0.000123 | 95.558 | 1.001249 |
| `Q4-PLE8` | 0.021037 | 0.000163 | 94.807 | 1.004707 |
| `Q4-PLE4` | 0.022283 | 0.000172 | 94.658 | 1.003870 |
| *Untrimmed UD-Q5_K_XL (its own repository)* | 0.012496 | 0.000108 | 95.956 | 1.002015 |
| *Untrimmed Balanced, IQ4_NL table (older layout)* | 0.014940 | — | 95.609 | 1.000898 |

- KLD measures distance from BF16 on text. It isn't a task score.
- Logs are in [kld/v5/](kld/v5/). Method and earlier variants: [kld/](kld/KLD-TABLE.md).

## Agent tasks (earlier build, native MTP, 2026-10-18/19)

Sixteen graded tasks from public benchmarks, one trial each, ran on the earlier Splash build with native MTP.

| Variant | Strict | Lenient |
|---|---|---|
| Quality-PLE8 | 8/16 | 9/16 |
| UD-Q4_K_XL-PLE8 (now `Q4-PLE8`, same tensors) | 7/16 | 8/16 |
| Balanced-PLE8 (the untrimmed Q5 body) | 8/16 | 8/16 |
| Max-PLE8 | 7/16 | 8/16 |

A one-task difference is within noise. The trimmed Q5 body was not in that sweep. The full table, task list and method are on the [UD-Q4_K_XL-PLE8 card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF).

## Files

| Path | Notes |
|---|---|
| `Q4-PLE8/` | 4 body parts (`…-UD-Q4_K_XL-0000K-of-00005.gguf`) + the Q8_0 table as part 5 |
| `Q4-PLE4/` | The same 4 body parts, untagged (`…-Q4-body-…`), + the IQ4_NL table as part 5 |
| `Q5-PLE8/` | 7 body parts (`…-Splash-Q5_K_M-0000K-of-00008.gguf`) + the Q8_0 table as part 8 |
| `Q5-PLE4/` | The same 7 body parts, untagged (`…-Q5-body-…`), + the IQ4_NL table as part 8 |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | unsloth's BF16 vision projector, byte-identical. Splash loads it for every folder |
| `sha256sums.txt` | SHA-256 of every GGUF |
| `provenance/` | Quantization plans, recipes and logs of the Q4, Balanced and Quality bodies |
| `imatrix_unsloth.gguf_file` | unsloth's importance matrix, for provenance. Splash doesn't download it |
| `kld/`, `MEASUREMENTS.md`, `UD-EVIDENCE.md` | Measurement receipts |

- **The table is always the last part of a folder.** Splash orders a split GGUF's parts by file name, so the table's name sorts after the body's.
- **The Q8_0 table is byte-identical across builds.** It's the same tensor as in unsloth's UD-Q5_K_XL.

**Older layouts.** Splash keeps a model at the revision it installed, and `--revision` pins one:
- **Until 2026-10-23,** this repository held the folders `UD-Q4_K_XL-PLE8`, `UD-Q5_K_XL`, `Quality-PLE8` and `Balanced-PLE4`. They're at `--revision 7e18f53420a2181cad81b79980662a51e1bd33ee`, and in the per-body repositories:
  - [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF)
  - [UD-Q5_K_XL](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF)
  - [Quality](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF)
  - [Balanced](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF)
  - [Max](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF)
- **The layout for the earlier native-MTP build** is at `--revision 294f1434444d74d57413c209cebcabb4db20fcb5`.

## Limitations

- **Coverage.** Every measurement is from one M5 Max 128 GB with a raised wired limit. It covers short prompts, timed concurrency runs, an install smoke test and a 16-task agent sweep on the earlier build. Images and PDFs beyond a smoke check, full-window reasoning and other hardware are untested.
- **Repeatability.** Even greedy decoding can produce different text across repeated and batched requests. For MoE targets, the rows Splash verifies follow a timing-based budget.
- **Memory.** Splash plans from the Mac's whole GPU budget. Close other memory-heavy applications, or cap Splash with `--max-memory`.

## Credits

- [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): the UD-Q4_K_XL body, the importance matrix, the Q8_0 table and the vision projector.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the base model and its license.
- The [Splash contributors](https://github.com/incoai/splash): the runtime and the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).

## Licence

These files are derivatives of Qwen3.8-Flash-Next and fall under the **Qwen Community License 1.0**. The full terms are in [LICENSE](LICENSE).
