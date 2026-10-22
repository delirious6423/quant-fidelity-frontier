---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: image-text-to-text
quantized_by: unsloth
tags:
- gguf
- imatrix
- qwen4exp
- moe
- splash
- apple-silicon
- dflash2
- vision
- mixed-precision
---

# Qwen3.8-Flash-Next · Splash · UD-Q5_K_XL

**unsloth's UD-Q5_K_XL, with six padding bytes so Splash loads it.** These are [unsloth's files](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q5_K_XL), unchanged except that **six zero bytes are appended to shard 1**. Shard 1 holds metadata only, no tensors, and in unsloth's upload it ends 6 bytes before its 32-byte-aligned data offset. Splash 1.4.0 refuses that with `GGUF data section is truncated`. Padded, it grows from 10,946,618 to 10,946,624 bytes. Shards 2–6 and the vision projector are byte-identical to unsloth's. **Nothing was re-quantized**: the per-layer embedding (PLE) table in shard 3 is already Q8_0 in unsloth's UD-Q5_K_XL. See [upstream discussion #86](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/86). Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Quick start

Splash 1.4.0 or newer, a 128 GB Apple-silicon Mac, macOS 26.4 or later:

```bash
brew install incoai/tap/splash
splash serve --model Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF:UD-Q5_K_XL
```

The first start downloads the six shards (158.3 GB) and the vision projector (0.91 GB) from this repository. Splash then adds the DFlash2 draft it releases for Qwen3.8-Flash-Next, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (1.21 GB), by itself. You don't need to pass `--draft-model`, and this repository ships no draft. Later starts reuse the downloads. Splash picks the draft from the model family it reads in the GGUF, not from the repository name, so the draft that matches unsloth's files also matches these.

- **Vision is on by default.** It serves images and PDFs through the projector. Add `--language-only` to serve text only.
- **Memory.** The model needs about 102 GiB of the GPU budget. Splash plans from the Mac's whole GPU budget, so close other memory-heavy applications; `--max-memory` caps Splash instead. The PLE table (54.4 GB) stays on the SSD, and Splash streams it through a row cache.
- **Coding agents.** Once it prints `Ready`, run `splash opencode`, `splash claude` or `splash codex` from another terminal.

## At a glance

| | |
|---|---|
| **What** | unsloth's UD-Q5_K_XL (6 shards) plus unsloth's BF16 vision projector. The DFlash2 draft comes from Splash's own release. |
| **For** | 128 GB Apple-silicon Macs. |
| **Quality vs BF16** | Mean KLD **0.012496**, same top-1 **95.956 %**: the lowest of the Flash-Next GGUFs we measured, below UD-Q4_K_XL with a Q8_0 table (0.021037). |
| **Speed** | Short prompts, one stream: **125.7 tok/s**, against 134.0 for UD-Q4_K_XL with a Q8_0 table in the same run. |
| **Download** | 159.2 GB: shards 158.286 GB + projector 0.908 GB, plus the 1.21 GB draft from its own repository. |
| **Resident body** | 103.875 GB of tensors in memory. The 54.400 GB PLE table streams from the SSD. |

## Benchmarks (M5 Max 128 GB)

### Distance from BF16 (KLD and perplexity)

Measured 2026-10-19 against a BF16 reference on neutral text: 87 × 4096-token chunks, 178,089 scored tokens, base PPL 4.045181.

| Variant | Mean KLD | Reported SE | Same top-1 % | PPL(Q)/PPL(base) |
|---|---:|---:|---:|---:|
| **UD-Q5_K_XL (this repo)** | **0.012496** | 0.000108, scorer | **95.956** | 1.002015 |
| UD-Q4_K_XL with a Q8_0 table | 0.021037 | 0.000163, scorer | 94.807 | 1.004707 |
| Quality-PLE8 | 0.012613 | 0.000436, chunk | 95.928 | 1.000487 |
| Balanced-PLE8 | 0.013270 | 0.000433, chunk | 95.844 | 1.001775 |
| Max-PLE8 | 0.012537 | 0.000422, chunk | 95.918 | 1.001522 |

- UD-Q5 PPL is 4.053330 ± 0.021747; the top-1 SE is 0.047 percentage points.
- The scorer SE and the chunk SE are different estimates, so don't compare them as if they were matched error bars.
- KLD measures distance from BF16 on text. It doesn't establish reasoning or tool-use accuracy.

Sources: [UD-Q5 scorer summary](evidence/KLD-summary.txt), [KLD comparison](evidence/KLD-comparison.md), [Q4 card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF#neutral-kld-and-perplexity).

### Speed with DFlash2, 2026-10-20

These ran on upstream Splash at `e92c00c8`, the main branch just before 1.4.0, with the same DFlash2 draft. The model came from a local copy of these shards, text only. Settings: 176K context, int8 KV, fp32 state, SSD cache off. Each concurrency level ran 48 greedy prompts (code, math, multilingual, prose, reasoning, tool JSON) with **zero errors**. UD-Q4_K_XL with a Q8_0 table ran first, in the same session.

| Measure | UD-Q5_K_XL | UD-Q4_K_XL, Q8_0 table |
|---|---:|---:|
| Launch to ready | 20.1 s | 75.2 s (cold file cache) |
| Served context | 180,224 | 180,224 |
| 1 stream: tok/s per stream | 125.7 | 134.0 |
| 2 concurrent: total tok/s | 120.9 | 144.2 |
| 4 concurrent: total tok/s | 155.3 | 174.7 |
| 74K-token cold prompt: time to first token | 36.6 s | 35.7 s |
| 74K-token cold prompt: decode tok/s | 105.6 | 105.9 |

- Q5 is 5–6 % slower than Q4 for one stream and 11–16 % slower in total with 2–4 concurrent streams.
- With one stream, each step accepted 2.57 draft tokens for Q5 and 2.62 for Q4.
- There was one launch per model and no confidence interval. [Receipt](evidence/dflash2-speed-2026-10-20.md).

### Install from this repository with Splash 1.4.0

<!-- E2E-140 -->
**Passed, 2026-10-22.** This ran on stock Splash 1.4.0 (`1482f6bb`) on an M5 Max 128 GB with the wired limit at 110,592 MB.

- **How it ran:** exactly as a user runs it, `splash serve --model REPO:FOLDER`, with Splash's defaults (vision on, automatic memory and context) plus `--max-cache-disk 0`.
- **What it downloaded:** the Hugging Face cache already held these exact files, matched by SHA-256, so the large shards weren't downloaded again. On the first start, Splash fetched the 1.21 GB DFlash2 draft from its own repository by itself, and nothing else.
- **Context:** its automatic plan held the model's full 262,144-token window.

| Model id | Process start to `Ready` | Served context | Vision | Splash Metal peak | Checks passed |
|---|---:|---:|---|---:|---|
| `Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF:UD-Q5_K_XL` | 104 s | 256K | on | 100.8 GiB | 4/4 |
| `Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF:UD-Q5_K_XL` `--language-only` | 103 s | 256K | off | 99.9 GiB | 3/3 (text only) |
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:UD-Q5_K_XL` | 117 s | 256K | on | 100.8 GiB | 4/4 |

The checks:

- 17 × 23 answered 391.
- A code planted in an ~8K-token prompt was retrieved.
- Two `get_weather` calls (Paris and Tokyo, celsius) came back as parsed tool calls.
- A synthetic receipt image was read into a JSON schema with all 6 fields correct.

The time to `Ready` includes loading the weights; the first start also includes the draft download. Peak memory is Splash's Metal allocation high-water mark after the checks, not process RSS.

The same files are also the `UD-Q5_K_XL` folder of the family repository [Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF), which lists every variant.

## Files and integrity

| Path | Bytes | Same as unsloth's |
|---|---:|---|
| `UD-Q5_K_XL/…-00001-of-00006.gguf` | 10,946,624 | No: +6 zero bytes, metadata only |
| `UD-Q5_K_XL/…-00002…00006-of-00006.gguf` | 158,275,460,032 in total | Yes, byte-identical |
| `mmproj-BF16.gguf` | 907,542,944 | Yes, byte-identical |

The shards keep unsloth's names under `UD-Q5_K_XL/`, so `:UD-Q5_K_XL` resolves the whole split set and the Hub shows the 5-bit label. Splash takes the projector from any root file named `mmproj`. [sha256sums.txt](sha256sums.txt) lists every payload. [Tensor inventory](evidence/tensor-inventory.json).

When Splash accepts a tensor-free part that ends at its header, unsloth's own folder will load as well: `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q5_K_XL`. The output is the same, since only the padding differs.

## Limitations

- Every measurement is from one M5 Max 128 GB, with the GPU wired limit at 110,592 MB (`iogpu.wired_limit_mb`). On a Mac with the default limit, Splash sizes a smaller budget. It may then hold less context, or stream more of the model from the SSD. We haven't tested that.
- The checks above cover short answers, tool calls, an 8K-token retrieval, one synthetic receipt image and timed prompts. They don't validate reasoning over the whole context, broad agent accuracy, PDFs or other hardware.
- With greedy decoding, batched and repeated requests can still produce different text. Splash notes this for MoE targets: the rows it verifies follow a timing-based budget.

## Credits

[unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) made the quantization and the vision projector. [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) made the base model and owns the license. The [Splash contributors](https://github.com/incoai/splash) made the runtime and the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). The BF16 reference logits and neutral evaluation text come from the datasets linked in the [Q4 card's KLD evidence](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/kld).

## Licence

These files are derivatives of Qwen3.8-Flash-Next and fall under the **Qwen Community License 1.0**. The full terms are in [LICENSE](LICENSE).
