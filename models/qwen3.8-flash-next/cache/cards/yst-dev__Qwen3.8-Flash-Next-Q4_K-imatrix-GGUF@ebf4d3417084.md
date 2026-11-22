---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - imatrix
  - qwen4exp
---

# Qwen3.8-Flash-Next: calibrated Q4_K GGUF (imatrix)

A 4-bit GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), made by YST. The routed experts
are Q4_K, quantised with an importance matrix. On held-out text it stays closer to a Q8_0 reference than the widely used
UD-IQ4_XS file (mean KL 0.032 against 0.056). It is also 12.5 GB bigger, which explains part of the gain.

This is the file behind our Flash-Next serving work on AMD cards in [Gary](https://github.com/msw-yst/gary) (branch
`flashnext-amd`, docs in `docs/flashnext/`).

Text only. The vision encoder and the MTP drafter are not in this repository (see [Files](#files)).

## Files

| File | Size (bytes) | What it is |
|---|---|---|
| `Qwen3.8-Flash-Next-YST-Q4K-imat-00001-of-00004.gguf` | 10,945,888 | metadata and tokenizer (no tensors) |
| `Qwen3.8-Flash-Next-YST-Q4K-imat-00002-of-00004.gguf` | 43,134,439,040 | weights |
| `Qwen3.8-Flash-Next-YST-Q4K-imat-00003-of-00004.gguf` | 34,266,801,056 | weights |
| `Qwen3.8-Flash-Next-YST-Q4K-imat-00004-of-00004.gguf` | 28,800,138,432 | the per-layer n-gram embedding table (IQ4_NL), byte-identical to Unsloth's UD-IQ4_XS shard 4 |
| `chat_template.jinja` | 8,952 | the Qwen chat template from revision `de4b8e4d`, without `{% generation %}` tags (see [How to run](#how-to-run)) |
| `LICENSE` | 3,235 | Qwen Community License 1.0, verbatim |
| `SHA256SUMS` | - | sha256 of every file above |

The four GGUF shards total 106.2 GB (98.9 GiB). Load shard 1 and the others are found by name, so keep the four file names
unchanged and in one folder.

Not included:
- **MTP drafter.** We did not build one; our convert ran with `--no-mtp`. Unsloth's
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  (revision `f5d175de`, sha256 `58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1`) has the same architecture
  metadata. We kept it beside this file but have not measured it with llama.cpp. Gary's setup builds its own drafter from the
  BF16 checkpoint.
- **Vision encoder (mmproj).** Use a published `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, for example ISTA-DASLab's.

## How it was made

1. **Source.** BF16 safetensors of `Qwen/Qwen3.8-Flash-Next` at revision `812d96e10a0dd82b69d749e68693698cf6c36daa`.
2. **Q8_0 intermediate.** The checkpoint was converted with llama.cpp's `convert_hf_to_gguf.py` (llama.cpp b11870, commit
   `74bca3950`) to Q8_0 with `--no-mtp`. The n-gram embedding table was not converted; we took Unsloth's IQ4_NL table (shard 4
   above) instead. The metadata matches Unsloth's file key for key.
3. **Importance matrix.** We ran `llama-imatrix -c 512` on the Q8_0 over 252 KB of calibration text, 140 chunks of 512 tokens
   (71,680 tokens). The text is English technical prose and code:
   - 72 KB of alternating prose and code excerpts from the llama.cpp repository;
   - the first 90 KB of llama.cpp's `docs/*.md`;
   - the first 90 KB of a Python source corpus (open-source inference-server code).

   There is no chat, multilingual or reasoning data in it.
4. **Quantise.** We ran `llama-quantize --allow-requantize --keep-split --imatrix ... --tensor-type-file types.txt <Q8_0> <out> Q4_K_M`.
   The input is the Q8_0 intermediate, not the BF16. A per-tensor type map sets the types:
   - **Routed experts:** gate and up are **Q4_K** in all 48 layers. The down matrix is 640 wide, which no 256-block type fits, so
     llama.cpp falls back to **Q5_0** in all 48 layers.
   - **Every other quantised tensor** has the type the UD-IQ4_XS file gives it, so the comparison below changes only the experts:
     - `output` Q6_K and `token_embd` IQ4_XS;
     - attention and GDN projections are mostly Q6_K or Q8_0;
     - shared-expert down is Q8_0.
   - **Totals:** 1224 tensors = F32 388, Q8_0 476, Q6_K 213, Q4_K 96, Q5_0 48, F16 1, IQ4_XS 1, IQ4_NL 1. The experts average
     4.83 bpw (73.0 GB), the other tensors 7.15 bpw (4.4 GB) and the n-gram table 4.50 bpw (28.8 GB).

Exact commands, with paths replaced by placeholders, are in the Gary repository at `docs/flashnext/QUANT.md`.

## Quality: KL against Q8_0 on held-out text

**Method:**
- **Reference:** our Q8_0 intermediate. It is not BF16, and every file below shares the same n-gram table.
- **Text:** 140 KB taken from the same two sources as the calibration text (llama.cpp docs and Python source), from the bytes
  right after the parts used for calibration. 1.7 % of its 64-byte windows also occur in the calibration text.
- **Run:** `llama-perplexity -c 512 --chunks 40` (20,480 tokens), CPU, with `--kl-divergence-base` from the Q8_0 and then
  `--kl-divergence` for each file. The settings were the same for every file.

| File | Size | Mean KL vs Q8_0 | 99 % KL | Same top token | PPL (Q8_0: 2.140) |
|---|---|---|---|---|---|
| Unsloth UD-IQ4_XS (Dynamic v3.1) | 93.7 GB | 0.0559 +- 0.0018 | 0.765 | 93.6 % | 2.088 |
| **This file (calibrated Q4_K)** | 106.2 GB | **0.0322 +- 0.0010** | **0.502** | **95.2 %** | 2.156 |
| Our calibrated IQ4_XS (not published) | 98.7 GB | 0.0366 +- 0.0013 | 0.512 | 94.8 % | 2.150 |
| AtomicChat IQ4_XS, 3.84 bpw | 84.9 GB | 0.2459 +- 0.0068 | 3.368 | 87.5 % | 2.353 |

How to read this:
- **Size.** This file is bigger than the UD-IQ4_XS, which explains part of the gain. We did not build a calibrated file at the
  UD-IQ4_XS size.
- **Domain.** The calibration and test text share a domain (technical English and code), and that may favour the calibrated
  files.
- **Proxy.** KL against Q8_0 is a proxy for quality. We have not run task benchmarks (coding, agents, tool calling) on this file.
  A lower PPL than the reference, as in the first row, is not a sign of higher quality.

## How to run

### Gary (Strata fork, AMD ROCm)
[github.com/msw-yst/gary](https://github.com/msw-yst/gary), branch `flashnext-amd`. `docs/flashnext/README.md` covers the
build, the launch commands and the measured layouts. The short version:
- Put the four shards in one folder and serve it through the Unsloth Flash-Next family (`FAMILY=unsloth MODEL=UD-IQ4_XS
  MODEL_PATH=<folder>`). That entry picks the folder's only `-00001-of-` file.
- Pack it with `tools/iq_pack.py --gguf <shard 1> --out <pack> --compat-bf16`, as setup does.
- Use this repository's `chat_template.jinja`. The template embedded in the GGUF carries Hugging Face's `{% generation %}` tags,
  which the server's jinja2 rejects.

### llama.cpp
The `qwen4exp` architecture is in upstream llama.cpp. We ran this model's Q8_0 intermediate (same layout and the same n-gram
shard) through `llama-imatrix` and `llama-perplexity` of llama.cpp b11870 on CPU. We scored this file on CPU with
`llama-perplexity`. We have **not** tested `llama-server` / `llama-cli` chat, GPU offload or MTP drafting in llama.cpp with this
file. A starting point (untested as written):

```sh
llama-server -m Qwen3.8-Flash-Next-YST-Q4K-imat-00001-of-00004.gguf \
  --jinja --chat-template-file chat_template.jinja -c 32768 -ngl 99 --temp 1.0 --top-p 0.95 --top-k 20
```

The GGUF stores Qwen's sampling defaults (temperature 1.0, top-p 0.95, top-k 20).

## Hardware notes
- **Memory.** About 106 GB of weights plus the KV cache and runtime. The n-gram table (28.8 GB) is a lookup table: an engine can
  keep it in host RAM.
- **Our test machine:** 2x RX 7900 XTX (24 GB) and 4x Instinct MI50 (16 GB), one EPYC 7642, 256 GB DDR4, ROCm 7.2.
  - With Gary's MI50 helper caches, all 24,576 experts fit on the GPUs (each expert is 2.97 MB in this file).
  - Decode speed matched the UD-IQ4_XS file within noise: 50.2 vs 49.4 tok/s at 1 user under tensor parallel 2.
  - On a single GPU, the bigger experts mean more of them stay on the CPU than with the UD-IQ4_XS.

## Limitations
- KL against a Q8_0 reference only, on one domain of text, with 20k tokens. No BF16 reference and no task-level evaluation.
- The calibration text is small (72k tokens) and covers only technical English and code. Other languages and chat or reasoning
  traffic are not represented.
- Requantised from Q8_0, not quantised straight from BF16.
- The n-gram table and the non-expert type choices come from Unsloth's UD-IQ4_XS; only the routed experts are our calibration.
- Text only in this repository; the MTP drafter and vision encoder come from elsewhere (see [Files](#files)).
- Tested on AMD (ROCm) with Gary and on CPU with llama.cpp; other backends not tested.

## Licence
- **Licence.** The weights are a derivative of Qwen3.8-Flash-Next and are distributed under the **Qwen Community License 1.0**.
  Its full text, with Qwen's copyright notice, is in [LICENSE](LICENSE) and must ship with any copy or derivative of these files.
- **Two conditions in that licence:**
  - Commercial products or services with more than 100 million monthly active users or US$20 million monthly revenue must
    prominently display the model name.
  - Anyone running a Model-as-a-Service or AI Work Assistant business needs a separate licence from Qwen before any commercial
    use.

  Read the licence itself; this summary is not legal advice.
- **Gary.** Gary's MIT licence covers the engine code, not these weights.

## Attribution
- **Base model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by Qwen. Copyright (c) 2026 Qwen.
- **Shard 4:** the n-gram embedding table (shard 4) and the per-tensor types of the non-expert tensors are from Unsloth's
  [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ4_XS (revision `f5d175de`).
- **Tooling:** quantised with [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- **Made by** [YST](https://yst.dev).
