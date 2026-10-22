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

**Four GGUF builds of Qwen3.8-Flash-Next for [Splash](https://github.com/incoai/splash) on 128 GB Apple-silicon Macs.** Each sits in its own folder with its own quant tag, so the Hub's file panel lists them separately. All four installed from this repository and passed a smoke test on stock Splash 1.4.0 (results below). Splash adds its DFlash2 draft for speculative decoding by itself, and vision is on by default.

## Quick start

```bash
brew install incoai/tap/splash
splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:UD-Q4_K_XL-PLE8
```

Swap the part after `:` for any folder in the table below.

- **First start.** Splash downloads that folder's shards and the vision projector (0.91 GB) from this repository. Then it fetches the DFlash2 draft it releases for Qwen3.8-Flash-Next, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (1.21 GB). Later starts reuse the downloads.
- **No `--draft-model` is needed.** Splash reads the model family from the GGUF and picks the matching draft.
- **Text only:** add `--language-only`.
- **Coding agents:** once Splash prints `Ready`, run `splash opencode`, `splash claude` or `splash codex` in another terminal.

## Which one to pick

| Folder (`:VARIANT`) | Hub tag | What it is | Download GB | Resident body GB | KLD vs BF16 | Same top-1 % |
|---|---|---|---:|---:|---:|---:|
| **`UD-Q4_K_XL-PLE8`** (default) | `UD-Q4_K_XL` | unsloth's UD-Q4_K_XL body, unchanged, with a Q8_0 PLE table in place of its IQ4_NL table | 136.9 | 82.5 | 0.021037 | 94.807 |
| `UD-Q5_K_XL` | `UD-Q5_K_XL` | unsloth's UD-Q5_K_XL. The only change is 6 zero bytes of padding on its metadata-only part 1 | 158.3 | 103.9 | **0.012496** | **95.956** |
| `Quality-PLE8` | `Q5_K_XL` | Custom body: experts mostly Q6_K/Q5_K, down projections Q5_1/Q8_0; Q8_0 table | 152.8 | 98.4 | 0.012613 | 95.928 |
| `Balanced-PLE4` | `Q5_K_M` | Custom body: experts Q5_K, down projections Q5_1/Q8_0; IQ4_NL table. Our smallest download | 125.3 | 96.5 | 0.014940 | 95.609 |

- **Our default is UD-Q4_K_XL-PLE8.** It has the fastest decode and the smallest resident body. Its agent-task scores were within noise of the custom bodies on the earlier build.
- **For the least error vs BF16, pick UD-Q5_K_XL or Quality-PLE8.** Their KLD is a statistical tie, and both are about 40 % below the default's.
- **The tags are approximate.** `Q5_K_XL` and `Q5_K_M` are the nearest standard names for mixed-precision bodies of about 6.1–6.3 bits per weight. Folder names give the PLE table precision: PLE8 = Q8_0, PLE4 = IQ4_NL.
- **More variants:** the Quality, Balanced and Max bodies with every table option (PLE8, PLE5, PLE4) are in their own repositories:
  - [Quality](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF)
  - [Balanced](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF)
  - [Max](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF)
  - [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF)
  - [UD-Q5_K_XL](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF)

### How to read the Hub's size and memory marks

- **The panel shows each folder's download size, PLE table included.** The table is 54.4 GB at Q8_0 and 28.8 GB at IQ4_NL.
- **Splash doesn't hold the table in memory.** It reads the few rows each token needs straight from the SSD through a small cache. So the memory a variant needs is its **resident body** (table above), plus the KV cache and the runtime's working memory.
- **That is why every variant shows "probably too large" on a 128 GB Mac**, even though each one runs on one.
- **Every measurement on this card is from one M5 Max 128 GB with the GPU wired limit at 110,592 MB** (`iogpu.wired_limit_mb`). A Mac at the default limit is untested.

## Speed with Splash and DFlash2 (M5 Max 128 GB)

| Run | UD-Q4_K_XL-PLE8 | UD-Q5_K_XL |
|---|---:|---:|
| Stock Splash 1.4.0, 2026-10-21, 3 launches: 1 stream, tok/s per stream | 135.9–139.1 | — |
| Stock Splash 1.4.0: 4 concurrent streams, total tok/s | 183.2–186.9 | — |
| Stock Splash 1.4.0: agent-style turns, 1 stream, tok/s per stream | 150.3–150.6 | — |
| Upstream main just before 1.4.0 (`e92c00c8`), 2026-10-20: 1 stream, tok/s per stream | 134.0 | 125.7 |
| Same run: 4 concurrent streams, total tok/s | 174.7 | 155.3 |

**Settings for every run:**
- 48 greedy prompts per concurrency level, covering code, math, multilingual, prose, reasoning and tool JSON. Zero errors.
- 176K context, int8 KV, fp32 state, SSD cache off, text only.

The custom bodies haven't been timed with DFlash2. On the earlier native-MTP build they decoded at 82–87 tok/s in the agent sweep, against 96 for UD-Q4_K_XL-PLE8. [Speed receipt](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF/blob/main/evidence/dflash2-speed-2026-10-20.md).

### Install from this repository with Splash 1.4.0

<!-- E2E-140 -->
**Passed, 2026-10-22.** This ran on stock Splash 1.4.0 (`1482f6bb`) on an M5 Max 128 GB with the wired limit at 110,592 MB.

- **How it ran:** exactly as a user runs it, `splash serve --model REPO:FOLDER`, with Splash's defaults (vision on, automatic memory and context) plus `--max-cache-disk 0`.
- **What it downloaded:** the Hugging Face cache already held these exact files, matched by SHA-256, so the large shards weren't downloaded again. On the first start, Splash fetched the 1.21 GB DFlash2 draft from its own repository by itself, and nothing else.
- **Context:** its automatic plan held the model's full 262,144-token window.

| Model id | Process start to `Ready` | Served context | Vision | Splash Metal peak | Checks passed |
|---|---:|---:|---|---:|---|
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:UD-Q4_K_XL-PLE8` | 183 s (incl. draft download) | 256K | on | 80.9 GiB | 4/4 |
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:UD-Q5_K_XL` | 117 s | 256K | on | 100.8 GiB | 4/4 |
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Quality-PLE8` | 118 s | 256K | on | 96.0 GiB | 4/4 |
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Balanced-PLE4` | 107 s | 256K | on | 93.9 GiB | 4/4 |

The checks:

- 17 × 23 answered 391.
- A code planted in an ~8K-token prompt was retrieved.
- Two `get_weather` calls (Paris and Tokyo, celsius) came back as parsed tool calls.
- A synthetic receipt image was read into a JSON schema with all 6 fields correct.

The time to `Ready` includes loading the weights; the first start also includes the draft download. Peak memory is Splash's Metal allocation high-water mark after the checks, not process RSS.

## Distance from BF16 (KLD and perplexity)

Each variant was measured against the same BF16 reference on neutral text: 87 × 4096-token chunks, 178,089 scored tokens, base PPL 4.045181.

| Variant | Mean KLD | SE | PPL(Q)/PPL(base) |
|---|---:|---|---:|
| UD-Q5_K_XL | 0.012496 | 0.000108 (scorer) | 1.002015 |
| Quality-PLE8 | 0.012613 | 0.000436 (chunk) | 1.000487 |
| Balanced-PLE4 | 0.014940 | 0.000501 (chunk) | 1.000898 |
| UD-Q4_K_XL-PLE8 | 0.021037 | 0.000163 (scorer) | 1.004707 |
| *Original UD-Q4_K_XL, IQ4_NL table (not in this repo)* | 0.022195 | 0.000715 (chunk) | 1.004633 |

- The scorer SE and the chunk SE are different estimates; don't treat them as matched error bars.
- KLD measures distance from BF16 on text. It isn't a task score.
- Logs, method and the paired comparisons are in [kld/](kld/KLD-TABLE.md). Every other variant is on its own repository's card.

## Agent tasks (earlier build, native MTP, 2026-10-18/19)

Sixteen graded tasks from public benchmarks, one trial each, ran on the earlier Splash build with native MTP.

| Variant | Strict | Lenient |
|---|---|---|
| Quality-PLE8 | 8/16 | 9/16 |
| UD-Q4_K_XL-PLE8 | 7/16 | 8/16 |
| Balanced-PLE8 | 8/16 | 8/16 |
| Max-PLE8 | 7/16 | 8/16 |

A one-task difference is within noise. UD-Q5_K_XL and Balanced-PLE4 were not in that sweep. The full table, task list and method are on the [UD-Q4_K_XL-PLE8 card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF).

## Files

| Path | Notes |
|---|---|
| `UD-Q4_K_XL-PLE8/` | 4 parts. Same bytes as the [UD-Q4_K_XL-PLE8 repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF) |
| `UD-Q5_K_XL/` | 6 parts. Same bytes as the [UD-Q5_K_XL repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF) |
| `Quality-PLE8/` | 8 parts. Same bytes as the [Quality repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE8) |
| `Balanced-PLE4/` | 8 parts. Same bytes as the [Balanced repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE4); the files here are tagged `Q5_K_M` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | unsloth's BF16 vision projector, byte-identical. Splash loads it for every folder |
| `sha256sums.txt` | SHA-256 of every GGUF |
| `imatrix_unsloth.gguf_file` | unsloth's importance matrix, for provenance. Splash doesn't download it |
| `kld/`, `MEASUREMENTS.md`, `UD-EVIDENCE.md` | Measurement receipts |

**Older layout.** Until 2026-10-22 this repository held the `Balanced-PLE4`, `Balanced-PLE8` and `Quality-PLE4` folders and an MTP head, for an earlier Splash build.
- `Balanced-PLE4` is still here under the same folder name.
- `Balanced-PLE8` and `Quality-PLE4` are in the Balanced and Quality repositories. Use `--model Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF:Balanced-PLE8` or `--model Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF:Quality-PLE4`.
- `--revision 294f1434444d74d57413c209cebcabb4db20fcb5` keeps the old layout.

## Limitations

- **Coverage.** Every measurement is from one M5 Max 128 GB with a raised wired limit. It covers short prompts, timed concurrency runs and a 16-task agent sweep on the earlier build. Images and PDFs beyond a smoke check, full-window reasoning and other hardware are untested.
- **Repeatability.** Even greedy decoding can produce different text across repeated and batched requests. For MoE targets, the rows Splash verifies follow a timing-based budget.
- **Memory.** Splash plans from the Mac's whole GPU budget. Close other memory-heavy applications, or cap Splash with `--max-memory`.

## Credits

- [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): the UD bodies, the importance matrix and the vision projector.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the base model and its license.
- The [Splash contributors](https://github.com/incoai/splash): the runtime and the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).

## Licence

These files are derivatives of Qwen3.8-Flash-Next and fall under the **Qwen Community License 1.0**. The full terms are in [LICENSE](LICENSE).
