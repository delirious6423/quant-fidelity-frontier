---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
quantized_by: bitlamas
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: llama.cpp
tags:
- gguf
- llama.cpp
- imatrix
- qwen4exp
- strix-halo
- ryzen-ai-max
- radeon-8060s
- amd
- windows
- vulkan
- unified-memory
- long-context
---

# Qwen3.8-Flash-Next GGUF

llama.cpp quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), sized for a 128 GB AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S) on Windows: the whole transformer on the GPU at 262k context.

## Why this size, on Windows

A 128 GB Strix Halo on Windows can give its GPU memory in one of two ways:
- **A fixed carve-out of 96 GB.** The GPU gets 96 GB; Windows, your apps and everything else share the other 32 GB.
- **Dynamic allocation**, with the shared-memory slider at its 124 GB maximum. Memory stays shared, and one process can put about 80 GiB on the GPU.

llama.cpp treats this model's 28.8 GB n-gram table as an input-layer lookup: it stays in system RAM and is never offloaded, in either mode. With the carve-out, that table has to share the remaining 32 GB with Windows and your apps, and the paging that follows makes the whole desktop worse. With dynamic allocation, the table streams from the file in system RAM while the transformer sits on the GPU. That is the mode that works, and it leaves about 80 GiB for the model.

unsloth's UD-Q4_K_XL (111.3 GB) does not fit in that unless you page or move expert layers to the CPU, and the next size down, UD-IQ4_XS, is a measurably worse model. Nobody ships the size in between, so I made it.

## Q4_K_XL-IQ4NL

This is unsloth's UD-Q4_K_XL with one change. Its 48 expert down projections (`ffn_down_exps`) are IQ4_NL, not Q5_1, quantized from Qwen's BF16 weights with unsloth's importance matrix. That takes the file from 111.3 GB to 102.5 GB.

Why the down projections: `ffn_down_exps` is 640 values wide. That is not a multiple of 256, so no 256-block K- or I-quant can touch it. Q5_1 and IQ4_NL are the 32-block options, and IQ4_NL saves 8.9 GB.

| Quant | Size | Mixture (other / gate / up / down) | PPL | KLD vs UD-Q4_K_XL | Same top token | Down error vs BF16 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| UD-Q4_K_XL (unsloth, reference) | 111.3 GB | Q8_0 / Q4_K / Q4_K / Q5_1 | 2.9167 ± 0.0365 | — | — | — |
| **Q4_K_XL-IQ4NL (this repo)** | **102.5 GB** | Q8_0 / Q4_K / Q4_K / **IQ4_NL** | **2.9092 ± 0.0362** | 0.0304 | 94.40 % | **0.00490** |
| UD-IQ4_XS (unsloth) | 93.7 GB | — | 2.998 ± 0.038 | 0.089 | 90.75 % | — |

Notes on the table:
- **PPL** is perplexity on the wikitext-2 test set, 24 chunks of 2048 tokens, unsloth's llama.cpp `b10715`; lower is better. **KLD** is how far this file's token probabilities are from UD-Q4_K_XL's; lower is closer. **Same top token** is how often both pick the same most likely token. **Down error** is the imatrix-weighted squared error of the 48 down projections against Qwen's BF16 originals; lower is better.
- **"Other"** covers attention, linear attention, hyper-connections, the shared expert, the embeddings and the output, 5.5 GB at Q8_0. The 28.8 GB n-gram table is IQ4_NL in all three. "—" means not measured.
- **Same run:** this file and the reference row come from one run, so their PPLs compare directly: equal within the error bars. An earlier all-GPU run measured the reference at 2.9117 ± 0.0363, and UD-IQ4_XS was scored against that run. The difference is the kernel path, not the file.
- **Raw logs** for every number: [logs/](logs/).

## Run it (128 GB Strix Halo, Windows, Vulkan)

Mainline llama.cpp has the MTP draft (`--spec-type draft-mtp`) since PR #29761 (2026-10-01), but it can only load unsloth's **non-shared** head, `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`. The smaller `shared-` head borrows the model's embeddings and output, and only unsloth's builds can load it; on mainline it fails with `tensor 'token_embd.weight' not found`. The command below uses the non-shared head, measured on mainline `b11681`.

```
llama-server -m Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00001-of-00004.gguf \
  -c 262144 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -np 1 \
  -lm mmap --lazy-mode on --cache-ram 0 --ctx-checkpoints 8 --jinja \
  -md mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 128 --spec-ngram-mod-n-match 48
```

- **Memory:** use dynamic memory with the shared-memory slider at its 124 GB maximum, which gives one process ~80 GiB on the GPU. This command lands at 78.3–78.6 GiB on mainline `b11681` (78.2 GiB on `b11160-mix` with the shared head). If it dies at warm-up, `-ub 256` frees ~2.2 GiB.
- **The n-gram table:** `-lm mmap --lazy-mode on` streams the 28.8 GB table from the file. `mlock` cannot pin a region that big on Windows.
- **Speculative decoding:** the draft head is Qwen's MTP weights as exported by unsloth: `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.79 GB, any build with MTP) or `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (1.91 GB, unsloth builds only). It adds ~50 % speed on fresh text; the n-gram drafter adds ~13 % on edits.
- **Prompt quality:** every stock build since `b11160`, mainline and unsloth, reads the prompt through an int8 matmul on this GPU (PR #27952), and none has a switch to turn it off. On this file it raises the KLD from 0.030 to 0.038. Generation is not affected. See Measurements, Engines.
- **Vision:** unsloth's `mmproj` works with this file. Run the projector on the CPU with `--no-mmproj-offload`.
- **Real use:** decode medians are **28.4 / 20.1 / 17.0 tok/s** under 25k, at 25–50k and at 50–100k context, on `b11160-mix` with int8 coopmat off (see Measurements).

## Download

```
hf download bitlamas/Qwen3.8-Flash-Next-GGUF --include "Q4_K_XL-IQ4NL/*" --local-dir ./
```

| File | Bytes | sha256 |
| :--- | :--- | :--- |
| `Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00001-of-00004.gguf` | 10,947,232 | 9975fc8a0fbb4ef70a9ff667588e967d2932d48906dd67e6e5dd58b0a21bc331 |
| `Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00002-of-00004.gguf` | 47,447,858,336 | e27d8e842be942edc50d52f2e4c8dbd2db1265675b8e3e543437ec58dd5b2692 |
| `Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00003-of-00004.gguf` | 44,552,691,904 | 1b4597c34629cab398f3c046f1bb570a6684002a543964bc991f9d4b05108cd2 |
| `Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00004-of-00004.gguf` | 10,462,690,720 | ff88c6b7de4c2715a03731657cfdba08fb9d8a11e463217888c500da63db5021 |

Total 102,474,188,192 bytes (95.43 GiB). `SHA256SUMS` has the same hashes.

The MTP draft head and the vision projector are unsloth's:

```
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" "mmproj-F16.gguf" --local-dir ./
```

<details>
<summary>Recipe</summary>

`method/` has the scripts (plain Python, no numpy). `METHOD.md` has the exact commands. You do not need to download the 360 GB of BF16 weights.

1. `bf16_downs_fetch.py` fetches only the 48 BF16 down tensors from `Qwen/Qwen3.8-Flash-Next` with HTTP range requests, about 80.5 GB, into one GGUF. It resumes after an interruption and spot-checks every layer against the existing quant.
2. `llama-quantize --imatrix imatrix_unsloth.gguf_file --tensor-type ffn_down_exps=iq4_nl downs-bf16.gguf downs-iq4nl.gguf IQ4_NL`.
   The explicit `--tensor-type` is required. With the `IQ4_NL` file type alone, llama-quantize picks q5_K for the downs, hits the 256-block rule, and silently falls back to q5_1. Read the log.
3. `gguf_splice.py` writes the new shard set. Every tensor is copied byte for byte except the 48 replacements.
4. `downs_vs_bf16.py` is the BF16-anchored error comparison.
5. `gguf_tensor_map.py` reads a GGUF's real composition from its header alone, from a local file or a Hugging Face URL. Judge a quant by this map, never by its name.

The same recipe works on any tensor whose width is not a multiple of 256, on any model.
</details>

<details>
<summary>Measurements: the first build (DN4) against this one, KV cache, engines</summary>

Full write-up: [NOTES.md](NOTES.md).

**DN4 against this build (DN4B in my notes).** DN4 rounded twice: BF16 to Q5_1 (unsloth), then Q5_1 to IQ4_NL. This build rounds once, from BF16. Across 19,200 sampled rows in all 48 layers, the error against BF16 is:

| | DN4 | this build | change |
| :--- | :--- | :--- | :--- |
| squared error | 0.00689 | 0.00583 | −15.4 % |
| imatrix-weighted squared error | 0.00667 | 0.00490 | **−26.5 %** |

This build is better on the weighted measure in every one of the 48 layers, by 11 to 58 %.

DN4 measured PPL 2.9165 ± 0.0363 and KLD 0.0282 against UD-Q4_K_XL. Its KLD is lower because DN4 was made *from* the reference's own Q5_1 values. A KL divergence against a quantized reference cannot rank two quants whose rounding comes from different places. The BF16 error can.

**KV cache:** measured on DN4, which has the same cache path. Each figure is the KLD cost against an f16 cache at 2048 tokens:
- q8_0 on both sides costs 0.012;
- q4_0 on both sides costs 0.026;
- q4_0 keys with q8_0 values cost 0.019.

None changes decode speed. q4_0 saves 1.5 GiB at 262k.

**Engines:** one conversation grown to 262k:

| engine | KLD vs UD-Q4_K_XL | prefill, first 33k | at ~131k | at ~262k | decode at 262k |
| :--- | :--- | :--- | :--- | :--- | :--- |
| unsloth `b10715` | 0.0304 | 128 tok/s | 78 | 57 | 14.4 |
| unsloth `b11160-mix`, int8 coopmat on (as shipped) | — | 204 | 97 | 71 | 15.8 |
| `b11160-mix` with int8 coopmat off (my build) | 0.0301 | 159 | 87 | 67 | 15.8 |

The int8 coopmat matmul (PR #27952) changes how the prompt is read, not generation. Scored against the same build with it off, it moves the KLD by 0.047 at 512 tokens and by 0.027 at 16k. I run it off; upstream has no switch. `GGML_VK_DISABLE_COOPMAT` and `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` do not reach it.

KLD vs UD-Q4_K_XL on the standard 24 × 2048 run:

| engine | KLD | same top token |
| :--- | :--- | :--- |
| mainline `b11681`, as shipped | 0.0383 | 93.71 % |
| unsloth `b11669-mix`, as shipped | 0.0383 | 93.71 % |
| mainline `b11681` with int8 coopmat off (my build) | 0.0301 | 94.62 % |

My build changes one condition in `ggml-vulkan.cpp` so that the int8 path is off unless `GGML_VK_ENABLE_COOPMAT_INT=1` is set: [method/coopmat-int-exact-default-b11681.patch](method/coopmat-int-exact-default-b11681.patch), applied with `git apply` to mainline tag `b11681`, then a normal Vulkan build. On mainline `b11681` it also decodes as fast as `b11160-mix` or faster, and prefills 7–9 % faster.

One warning when benchmarking: this machine drops its power limit to ~15 W while nobody uses it. Keep a Remote Desktop session attended, or disable Modern Standby.
</details>

## Credits and license

Qwen made the model. It is under the Qwen Community License 1.0, included as `LICENSE`, as the license requires for derivatives.

unsloth made UD-Q4_K_XL, which is every tensor in this file but 48. They also made the importance matrix used for those 48, and the MTP head export. Their choice to keep the small tensors at Q8_0 is what makes this model's 4-bit quants good.

I built and measured this on my own machine. I am not affiliated with Qwen or unsloth.

## Updates

- **2026-10-17:**
  - Run it: mainline llama.cpp now has the MTP draft, with the non-shared head only. The command and the download use that head.
  - The int8 prompt matmul in every stock build since `b11160` costs 0.008 KLD on this file, and no build switch turns it off.
  - New: `method/coopmat-int-exact-default-b11681.patch`, the one-condition patch that turns the int8 path off.
  - No model files changed.
- **2026-10-09:**
  - Renamed from `bitlamas/Qwen3.8-Flash-Next-Q4_K_XL-DN4`; the old links redirect.
  - The files are now `Q4_K_XL-IQ4NL/…`.
  - Shard 1's header now names the right quantizer and credits unsloth and Qwen in the standard GGUF fields. No weights changed, since shard 1 holds no tensors.
- **2026-10-07:**
  - The expert down projections are rebuilt from Qwen's BF16 weights, no longer requantized from unsloth's Q5_1.
  - This cuts their imatrix-weighted error against BF16 by 26.5 %.
  - The first build (DN4) is still at [revision 374ee57](https://huggingface.co/bitlamas/Qwen3.8-Flash-Next-GGUF/tree/374ee5752a4e5dd2b98ca76eafaec6e3b7af3105).
- **2026-09-05:** first upload (DN4).
