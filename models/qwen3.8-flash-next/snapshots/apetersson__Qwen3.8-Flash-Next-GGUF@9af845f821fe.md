---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- qwen
- quantized
- vision
- video
---

# Qwen3.8-Flash-Next: shared BF16 n-grams

**Q5 targets 128GB systems; IQ4 targets 96GB systems.** On a discrete **96GB RTX PRO 6000 Blackwell**, Q5 also passed at **262,144 context with vision and MTP**, using **90.51 GiB including driver reservation**. IQ4 passed a **1,048,576-context allocation and short-generation check** at **87.85 GiB**. These CUDA configurations use separate host RAM for the BF16 n-gram table. Both standalone trunks reference the same original BF16 n-gram table on SSD. Q5 retains the calibrated mixed-precision trunk; the new UD-IQ4_XS-BF16-Ngrams recipe preserves Unsloth's IQ4 trunk and replaces its quantized n-grams with the original BF16 table.

## Run locally with llama.cpp

Build the shared-ngram runtime using [GUIDE.md](GUIDE.md), then run the Q5 variant for 128GB systems from this package directory:

```bash
/path/to/llama-server \
  --model ./Q5_K_M-BF16-Ngrams/Qwen3.8-Flash-Next-Q5_K_M.gguf \
  --mmproj ./shared/mmproj-Qwen3.8-Flash-Next-f16.gguf \
  --model-draft ./shared/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 \
  --fit off --lazy-mode on -ngl 99 -fa on --jinja --parallel 1 \
  --ctx-size 262144 --batch-size 512 --ubatch-size 128 --no-context-shift \
  --host 127.0.0.1 --port 18080
```

For IQ4 on 96GB systems, replace only the model argument with:

```bash
--model ./UD-IQ4_XS-BF16-Ngrams/Qwen3.8-Flash-Next-UD-IQ4_XS.gguf
```

The trunk's `qwen4exp.ngram_file` metadata resolves the table relative to the trunk, independent of the shell's working directory. Both recipes use the **same physical BF16 file**. No n-gram payload is stored in the new IQ4 directory. Keep `--lazy-mode on` so the table is gathered on the CPU from its file mapping rather than fully loaded into GPU memory.

This package uses a **custom packaging/runtime extension**. Stock llama.cpp does not resolve this reference, and `-hf` does not automatically download the referenced dependency. Use the complete local directory and the patched build. Download both your chosen trunk directory and `shared/`, preserving their relative paths:

```bash
hf download apetersson/Qwen3.8-Flash-Next-GGUF --local-dir Qwen3.8-Flash-Next-GGUF --include "*.md" "*LICENSE" "*.patch" "*.json" "Q5_K_M-BF16-Ngrams/*" "shared/*"
```

For IQ4, replace `Q5_K_M-BF16-Ngrams/*` with `UD-IQ4_XS-BF16-Ngrams/*`.

## Files and deduplication

| Artifact | GiB | Role |
|---|---:|---|
| `Q5_K_M-BF16-Ngrams/Qwen3.8-Flash-Next-Q5_K_M.gguf` | 76.24 | Standalone Q5 trunk |
| `UD-IQ4_XS-BF16-Ngrams/Qwen3.8-Flash-Next-UD-IQ4_XS.gguf` | 60.42 | Standalone IQ4 trunk |
| `shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf` | 95.37 | Single shared original BF16 n-gram table |
| `shared/mmproj-Qwen3.8-Flash-Next-f16.gguf` | 0.84 | Shared vision projector |
| `shared/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.60 | Shared MTP head |

All three shared artifacts now live in `shared/`. Both trunks reference `../shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf`. The shared loader imports the table tensor independently of its retained legacy split metadata.

The active artifacts for **both** recipes occupy approximately **235.47 GiB**, counting the table, MTP and projector once. The original IQ4 source shards and obsolete Q5 split trunk have been removed after verification.

## Run on RTX PRO 6000 Blackwell (96GB)

Tested on [Runpod](https://runpod.io?ref=bjuhotnb) on 2026-09-08 with one RTX PRO 6000 Blackwell Server Edition, 97,887 MiB total VRAM, 32 allocated vCPUs and 188GB host RAM. Build the pinned CUDA runtime using [GUIDE.md](GUIDE.md#linuxcuda-build-tested-on-rtx-pro-6000-blackwell). The same combined patch and model files work on CUDA; no additional source changes were needed.

**Q5 at 256K:** run the first command in this README with your CUDA-built `llama-server`. It passed exact text checks, an image-input check, and 2K/8K prompt benchmarks with all trunk layers on the GPU, default F16 KV, vision and MTP depth 2. A smaller context was not necessary. To use IQ4 at 256K, change only the model path as shown above.

**IQ4 at 1M:** from the model package directory, run:

```bash
/path/to/llama-server \
  --model ./UD-IQ4_XS-BF16-Ngrams/Qwen3.8-Flash-Next-UD-IQ4_XS.gguf \
  --mmproj ./shared/mmproj-Qwen3.8-Flash-Next-f16.gguf \
  --model-draft ./shared/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 \
  --fit off --lazy-mode on -ngl 99 -fa on --jinja --parallel 1 \
  --ctx-size 1048576 --cache-type-k q8_0 --cache-type-v q8_0 \
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 \
  --batch-size 512 --ubatch-size 128 --no-context-shift \
  --host 127.0.0.1 --port 18080
```

This 1M configuration loaded in 78.42 seconds and passed both short exact-answer tests, with the projector and MTP loaded. The CUDA test evaluated actual prompts up to 261,888 tokens at native context; it did not evaluate a million-token prompt. Main-model KV is Q8 in the 1M command; the separate MTP draft cache retains its default F16 precision.

### Measured VRAM

| Quant | Allocated context | Main KV | Vision / MTP | Peak VRAM including reservation | Minimum free VRAM |
|---|---:|---|---|---:|---:|
| Q5_K_M | 262,144 | F16 | Both | 90.51 GiB | 5.09 GiB |
| UD-IQ4_XS | 262,144 | F16 | Both | 74.76 GiB | 20.84 GiB |
| UD-IQ4_XS | 1,048,576 | Q8_0 | Both loaded | 87.85 GiB | 7.74 GiB |

Peaks are sampled, not inferred from GGUF sizes. NVML byte counters were sampled every 100ms throughout Q5 and the IQ4 1M check, and during the final portion of the IQ4 native-context sweep; earlier IQ4 cases used two-second `nvidia-smi` samples. The Q5 peak separates into **89.88495 GiB allocated + 0.62036 GiB driver reservation**. The BF16 n-gram file mapping is CPU-side and excluded from VRAM. These numbers are discrete GPU memory, not combined host-plus-GPU memory.

### Reference performance from the feasibility test

These runs establish that the packaged models load and generate on the card. They were not tuned for maximum throughput; the rates below are reference observations, not performance records or expected hardware limits.

One slot, batch 512, micro-batch 128, 262,144 allocated context, F16 KV, MTP depth 2, projector loaded, temperature 0 and seed 1234. Prompts contain public llama.cpp source/documentation truncated to the exact token counts below. Every request reports zero reused prompt tokens. PP means prompt processing; TG means generated tokens, including accepted MTP drafts.

| Quant | Input tokens | Output tokens | PP tokens/s | TG tokens/s |
|---|---:|---:|---:|---:|
| Q5_K_M | 2,048 | 256 | 48.10 | 22.00 |
| Q5_K_M | 8,192 | 256 | 103.72 | 26.95 |
| UD-IQ4_XS | 2,048 | 256 | 310.62 | 76.12 |
| UD-IQ4_XS | 4,096 | 256 | 564.52 | 79.92 |
| UD-IQ4_XS | 8,192 | 256 | 600.70 | 84.58 |
| UD-IQ4_XS | 32,768 | 256 | 550.16 | 85.08 |
| UD-IQ4_XS | 102,400 | 256 | 656.24 | 64.10 |
| UD-IQ4_XS | 261,888 | 255 | 504.63 | 29.60 |

The last row processed every input token and stopped at the context boundary after 255 output tokens. Its response reports `truncated: true`; the harness's strict 256-output assertion therefore failed. The table uses actual output counts and timings. Both quants separately passed exact text and solid-red-image checks.

**Host memory and storage matter.** These models use a network-mounted volume in this test. Before each main sweep, we sequentially read the BF16 table to warm the OS file cache; this is independent of prompt/KV caching. IQ4's initial cold-network 2K run reached only **32.59 PP / 14.40 TG tokens/s**, compared with **310.62 / 76.12** after warming. During Q5, cgroup memory reached approximately **187.76GB of its 188GB limit** despite the warmup. Its slower results were measured under host-memory pressure and are not an isolated comparison of Q5 versus IQ4 GPU kernels. Use fast local storage and budget host RAM for useful n-gram caching; a 96GB GPU does not replace that host-memory requirement.

[Machine-readable CUDA results](cuda-tests.json) records the pinned release/runtime, measured timings, token counts, validation flags and memory readings.

### Provisional context capacity on this 96GB GPU

The following are **memory-based estimates to try**, not measured maximum contexts. They use the observed CUDA allocations, leave approximately **2 GiB free**, and round down to 32K increments. **K = 1,024 tokens; 1M = 1,048,576 tokens.** One slot, batch 512 and micro-batch 128; BF16 n-grams remain CPU-side.

| Mode | Q5 with F16 KV | Q5 with Q8 KV | IQ4 with F16 KV | IQ4 with Q8 KV |
|---|---:|---:|---:|---:|
| Vision + MTP | ~320K | ~544K | ~736K | 1M |
| MTP only | ~352K | ~608K | ~768K | 1M |
| Vision only | ~416K | ~736K | ~864K | 1M |
| Neither | ~448K | ~800K | ~896K | 1M |

The main model's attention and indexer caches cost **8.25 GiB per 262,144 tokens in F16**, or approximately **4.38 GiB in Q8_0**. MTP adds approximately **2.67 GiB of weights/compute plus 0.5 GiB per 262,144 tokens** for its default F16 cache. Vision adds approximately **1.08 GiB of weights/reserved compute** in the tested configuration. The projection also allows roughly **0.75 GiB of additional compute memory per 262,144 tokens**, inferred from the two measured IQ4 configurations; actual allocation growth can differ.

Reducing main KV precision buys the most context; removing MTP frees more memory than removing vision. To disable vision, omit `--mmproj`. To disable MTP, omit its four arguments listed below under “Images, video and MTP”. Beyond native 262,144 context, configure context extension such as YaRN; memory fit alone does not establish long-context quality. Image/video tokens share the context budget and larger visual inputs can require extra workspace. The table is capped at 1M; higher limits were not investigated. Only **Q5 256K with both features** and **IQ4 256K/1M with both loaded** were tested on CUDA, as detailed above.

## Shared-loader smoke tests

Both quants passed with **1,048,576 allocated context, Q8 KV, 4x YaRN, MTP depth 2 and the vision projector loaded** on the M1 Ultra. Each returned the exact checkpoint string and `323` for `17 * 19`, temperature 0, seed 1234. These are short text smoke tests, not million-token retrieval or visual-input tests. The runtime also rejected missing tables, wrong tensors, absolute paths and paths outside the package directory. See [shared-ngram-tests.json](shared-ngram-tests.json).

## Historical Q5 performance

These results used the previous standard two-shard Q5 layout, whose tensor payloads are unchanged in the standalone Q5 recipe. Measured on an **Apple M1 Ultra, 128 GiB unified memory**, with Metal, vision and MTP depth 2, one slot, batch 512 and micro-batch 128. Temperature 0; seed 1234; prompt caching disabled.

| Input tokens | Output tokens | Prefill tokens/s | Generation tokens/s |
|---:|---:|---:|---:|
| 2,048 | 256 | 176.4 | 27.7 |
| 8,192 | 256 | 175.8 | 31.8 |
| 261,888 | 23 | 100.7 | 12.3 |
| 1,048,320 | 23 | 53.68 | 4.05 |

The 2K/8K figures average two runs from the internal SSD. The native 256K near-capacity test used the packaged model on the external SSD: **43.4 minutes**, **3/3 checkpoint codes recovered**, **no truncation or context shifting**, and **95.5 GiB peak system wired memory**. Swap did not grow. Generation rates reflect the output lengths shown. [Test details and memory records](Q5_K_M-BF16-Ngrams/context-tests.json).

The 1M run used Q8 KV caches and 4× YaRN on the external SSD. It completed in **325.6 minutes (5 hours 26 minutes)** and recovered **3/3 checkpoint codes**, evaluating every input token with no prompt-cache reuse, truncation or context shifting. Peak system wired memory was **110.33 GiB**. Swap rose from **7.03 GiB to 10.96 GiB**. The HF weights upload ran concurrently, so these throughput figures include competing upload I/O.

## Images, video and MTP

- **Images and video:** the included projector enables visual input. Clients must send image/video content, not just a local path. The tested llama.cpp runtime defaults to **4 FPS** for video.
- **Audio:** not supported by this package.
- **MTP:** the quick start enables the shared Q8 head at draft depth 2. To disable it, omit `--model-draft`, `--spec-type draft-mtp`, `--spec-draft-n-max 2` and `-ngld 99`.
- **Text only:** omit `--mmproj` and its path from the quick-start command.

## iMatrix and credits

An **importance matrix (iMatrix)** captures activation statistics from a calibration corpus. llama.cpp uses them to weight quantization error toward channels exercised by that corpus. It is used when quantizing the model and is not needed for inference.

The IQ4 trunk comes from [Unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). It is preserved byte-for-byte, excluding the removed quantized n-gram tensor. This is a new BF16 n-gram recipe, not a claim that Unsloth published this exact combination.

This custom Q5 trunk was quantized locally from BF16 using **[bartowski's published iMatrix](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/blob/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-imatrix.gguf)**: **926 entries, 582 processed chunks of 512 tokens**. The calibration-v6 corpus combines prose, tool calling and reasoning, rendered with the model's chat template. Credit to **bartowski**, and to **kalomaze and Dampf** for their calibration-dataset contributions. [Corpus source](https://gist.github.com/bartowski1182/e26453c0404e24eb317543ec5360f87a), [upstream calibration description and credits](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF#imatrix), and [pinned provenance](GUIDE.md#calibration-and-conversion-provenance).

- **[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):** original model, tokenizer, BF16 n-grams and vision weights. The F16 projector was converted locally from those weights.
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf):** the shared Q8_0 MTP head, included unchanged.
- **[llama.cpp/ggml contributors](https://github.com/ggml-org/llama.cpp):** GGUF, iMatrix, conversion, quantization and inference tooling, including [PR 28243](https://github.com/ggml-org/llama.cpp/pull/28243) for the bundled runtime patch.

## License

Model weights: **[Qwen Community 1.0](LICENSE)**. Bundled llama.cpp patch: **[MIT](LLAMA-CPP-LICENSE)**.

## Support me

If this release is useful to you, support my work by signing up through my [Runpod affiliate link](https://runpod.io?ref=bjuhotnb). You’ll receive signup credit, and I may earn a commission.
