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
- mtp
---

# Qwen3.8-Flash-Next · Splash GGUF

GGUF quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) built for
[Splash](https://github.com/incoai/splash) on Apple silicon Macs with 128 GB of unified memory.

- Every variant keeps attention, Gated-DeltaNet, shared experts, hyper-connections, embedding and output head at
  Q8_0 and spends the remaining memory on the routed experts.
- The 51.2 B-parameter per-layer n-gram (PLE) table sits alone in part 3 of each variant and is **streamed from the
  SSD** by Splash (`--ple-ssd-cache`). It takes no resident memory, so it can be kept at 8 bits (**PLE8**, Q8_0,
  54.4 GB) or 4 bits (**PLE4**, IQ4_NL, 28.8 GB) at the same memory and speed.
- Every variant keeps the full **262,144-token** context window on a 128 GB Mac.
- Quality is measured, not estimated: KL divergence against BF16 logits on AtomicChat's public reference.
  **Balanced-PLE8** has a mean KLD of **0.01327** and picks the same top-1 token as BF16 **95.84 %** of the time.
  That is 40 % lower KLD than unsloth's UD-Q4_K_XL (0.02220, 94.66 %) measured the same way.

Quantized by [Vardaan97](https://huggingface.co/Vardaan97). All measurements were made on one Apple M5 Max
(40-core GPU, 128 GB unified memory), macOS 27.0.1.

## Variants

| Variant | Tier | PLE table | Download (8 parts) | Resident weights (Splash) | KV capacity, int8 (full 262,144-token windows) | Mean KLD (chunk se) | Same top-1 as BF16 | 99.9 % KLD | Speed (Splash, single stream, greedy, MTP on) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **[Balanced-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF/tree/main/Balanced-PLE8)** | balanced | 8-bit (Q8_0), 54.40 GB | 150.95 GB | 99.47 GB | 442,368 tokens (1.7) | **0.01327** (0.00043) | **95.84 %** | **0.522** | 104.1 tok/s; a tie with Balanced-PLE4 (steady-state −0.5 % [−2.7, +1.7]) |
| [Balanced-PLE4](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF/tree/main/Balanced-PLE4) | balanced | 4-bit (IQ4_NL), 28.80 GB | 125.35 GB | 99.47 GB | 442,368 tokens (1.7) | 0.01494 (0.00050) | 95.61 % | 0.594 | 104.7 and 102.8 tok/s (two sessions) |
| [Quality-PLE4](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF/tree/main/Quality-PLE4) | quality | 4-bit (IQ4_NL), 28.80 GB | 127.21 GB | 101.59 GB | 327,680 tokens (1.25) | 0.01433 (0.00052) | 95.66 % | 0.600 | not timed with the protocol below; +0.5 % bytes per token vs Balanced, so expected within ~0.1 % of it (inference) |
| *UD-Q4_K_XL (unsloth), for reference* | | 4-bit (IQ4_NL) | 111.33 GB | 85.45 GB | 1,196,032 tokens (4.6) | 0.02220 (0.00072) | 94.66 % | 0.833 | |

Planned, not in this repository yet: **Quality-PLE8** (Quality body + 8-bit table; published only after its own
KLD run), **Max-PLE8** (more Q6_K experts; published only if its measured KLD beats Balanced-PLE8 by at least two
paired standard errors), **Ultra256-PLE8** (every tensor Q8_0, for Macs with 256 GB or more; needs validation on
such a Mac).

- **Tier** is the expert body: *quality* has Q6_K gate/up on 27 layers; *balanced* has Q8_0 down-projections on
  20 layers; *lite* has Q8_0 down-projections on 5 layers. The same tier with either table has identical resident
  weights, memory plan and KV capacity.
- **Resident weights**: Splash's `target_weights_bytes` with the table streamed. **KV capacity**: Splash's
  `kv_capacity_tokens` with int8 KV. Both are from Splash's `/status` with `--ple-ssd-cache 2G` and the GPU wired
  limit at 110,592 MB (see Usage). One full 262,144-token window needs 4.85 GB of int8 KV. Capacity beyond one
  window is shared by Splash's parallel lanes.
- **KLD**: 87 chunks × 4,096 tokens; chunk se from the 87 per-chunk means. The full table, paired differences and
  raw logs are in [`kld/`](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF/tree/main/kld).
- **Speed**: a Splash development build (#253 + #410 + further unmerged decode-speed patches). 144 prompts in six
  categories, 512-token cap, ABBA order against the neighbouring variant on the same day. These are throughput
  figures from that build on one machine, not a guarantee.

## Which one to pick

1. **Balanced-PLE8**, if you have the SSD space. It has the best measured quality of everything we measured,
   with the same memory, KV room and speed as Balanced-PLE4. The 8-bit table costs only 25.6 GB more download
   and disk. Paired against Balanced-PLE4, mean KLD is −11.2 % (7.5 paired se) and top-1 agreement +0.24 pp.
2. **Balanced-PLE4**, for the same model with a 25.6 GB smaller download.
3. **Quality-PLE4**: −4.1 % KLD vs Balanced-PLE4 (3.6 paired se) for 2.1 GB more resident memory, which leaves
   327,680 tokens of KV instead of 442,368. With the 4-bit table it is the best-measured option. Balanced-PLE8 is
   still 7.4 % lower (4.3 paired se).

None of these fits a Mac with less than 128 GB. On 128 GB, `--ple-ssd-cache` is required (below).

## Usage with Splash

Qwen3.8-Flash-Next support in Splash is in the open pull request
[#253](https://github.com/incoai/splash/pull/253) (by janhilgard). Streaming the PLE table from the SSD is in the
open pull request [#410](https://github.com/incoai/splash/pull/410), which is stacked on #253. Until they are
merged, build Splash from #410's branch. You need Xcode 26 and Python 3.12–3.14; see Splash's `DEVELOPMENT.md`.

```bash
git clone https://github.com/incoai/splash.git && cd splash
git fetch origin pull/410/head:ple-ssd-stream && git checkout ple-ssd-stream
make -j4
sudo sysctl iogpu.wired_limit_mb=110592      # the GPU wired limit every number on this card was measured with
./splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Balanced-PLE8 --ple-ssd-cache 2G --kv-format int8
```

- **Selecting a variant:** `--model OWNER/REPO:VARIANT`. Splash matches `VARIANT` case-insensitively against the
  folder of the same name and loads the folder's `…-0000K-of-00008.gguf` parts in order. All eight must be present.
  It also picks up the MTP head from `mtp/` (used for speculative decoding) and the vision projector
  (`mmproj-…-BF16.gguf`). It downloads only those files, not the other variants. Use `:Balanced-PLE4` and
  `:Quality-PLE4` the same way.
  A `--model` without `:VARIANT` is refused, because the repository root holds no model GGUF.
- **`--ple-ssd-cache` is required on a 128 GB Mac.** It streams the table from the SSD through a row cache of the
  given size (2 GiB in all our measurements). Without it the table (28.8 or 54.4 GB) would have to be resident as
  well, which exceeds the 128 GB budget.
- **Wired limit:** every measurement used `iogpu.wired_limit_mb=110592`, which gives Splash a hard budget of
  113.64 GB. We have not tested these files at the macOS default limit and expect Splash to refuse them there.
  The setting does not survive a reboot.
- **Download ahead of time, then start offline:**
  ```bash
  hf download Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF --include "Balanced-PLE8/*.gguf" "mtp/*" "mmproj-*"
  HF_HUB_OFFLINE=1 ./splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Balanced-PLE8 --ple-ssd-cache 2G --kv-format int8
  ```
- The streamed table is read with random 4 KB reads. All our measurements read it from the internal SSD of the Mac.
- Sampling defaults stored in the GGUF (from the base model): temperature 1.0, top_p 0.95, top_k 20.
- Text only: add `--language-only` (skips the projector). Other options: `splash serve --help`.

These are standard GGUF files. We have tested and measured them only with Splash.

## How quality was measured

- **Reference:** [AtomicChat's BF16 logits](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics/tree/main/kld)
  (`base-neutral.kld`) for Qwen3.8-Flash-Next at commit `de4b8e4d`. They are computed on AtomicChat's held-out
  30-language prose text ([`eval_neutral.txt`](https://huggingface.co/datasets/AtomicChat/calib-corpora), sha256
  `1b91967f…`).
- **Scoring:** all 87 chunks of 4,096 tokens (178,089 scored tokens), with the same command for every file.
- **Tool:** the GGUF reference perplexity/KL-divergence tool, Metal backend, one pinned build; flags in
  [`kld/METHOD.md`](kld/METHOD.md). The PLE table was kept in CPU memory for this tool. The scores reflect the
  weights, not Splash's kernels.
- **Uncertainty:** the tool's printed ± treats every token as independent. The chunk-level se and the paired
  per-chunk differences (from the 87 chunk means) are the ones to use when comparing files.
- **Paired results:**
  - Balanced-PLE8 vs UD-Q4_K_XL: −0.00893 ± 0.00042 (−40.2 %), top-1 +1.18 ± 0.06 pp.
  - Quality-PLE4 vs UD-Q4_K_XL: −35.5 %.
  - Balanced-PLE4 vs UD-Q4_K_XL: −32.7 %.
- **Measured as is:**
  - Balanced-PLE8's single worst token has KLD 9.48, the highest maximum of the ten files. One token carries no
    uncertainty estimate.
  - Its perplexity ratio is 1.0018 against 1.0009 for Balanced-PLE4.
  - Its mean, every percentile shown, top-1 agreement and RMS Δp are all better than Balanced-PLE4's.
- **Harness offset:** our harness differs from the reference's (Metal vs CUDA, micro-batch 512 vs 2048). That
  offset is common to every row, so rankings and paired differences hold. Absolute values are not comparable with
  numbers from other harnesses, and we do not claim a "% of BF16" figure.

## Limitations

- **One text set:** the quality numbers come from one prose set in 30 languages at 4,096-token context. There is
  no code-text KLD and no long-context (16K–262K) KLD yet, and there are no task benchmarks.
- **Open pull requests:** Splash support depends on #253 and #410, which were open when this card was written.
  The speed figures come from a development build with further unmerged patches; a build of #410 alone has not
  been timed with these files.
- **SSD streaming of the 8-bit table is still under test.** Balanced-PLE8 has served the 144-prompt protocol and
  four-stream runs, but Splash's full HTTP smoke test (tools, vision, PDF, Responses) has not yet been run on a
  PLE8 file. It passed on Quality-PLE4, on the Lite body and on another build with the same layout. A 2 GiB row cache
  holds about half as many 170-byte Q8_0 rows as 90-byte IQ4_NL rows.
- **Quality-PLE4 speed:** not timed with the 144-prompt protocol. An older build's 12-prompt A/B against the Lite body
  gave −4.5 % [−10.8, +1.2] (not significant).
- **Hardware coverage:** tested only on a 128 GB M5 Max with a raised wired limit, reading the table from the
  internal SSD.
- **Behaviour:** thinking and tool-use behaviour is the base model's. Nothing was fine-tuned.

## Reproducibility

- **Source:** unsloth's BF16 GGUF conversion (`unsloth/Qwen3.8-Flash-Next-GGUF`, folder `BF16/`, revision
  `766911a6`).
- **Importance matrix:** unsloth's `imatrix_unsloth.gguf_file` (sha256 `a5863123…`, included here).
- **Quantizer:** the GGUF reference quantizer with the unsloth importance matrix (a CPU build with qwen4exp
  support, including its 64-bit array metadata). One run per variant, with these arguments:
  ```
  --imatrix imatrix_unsloth.gguf_file --tensor-type-file <Variant>/tensor-types.tt \
  --token-embedding-type q8_0 --output-tensor-type q8_0 --keep-split \
  BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf Qwen3.8-Flash-Next-Splash-<Variant>.gguf Q5_K
  ```
  `--keep-split` reproduces the eight-part layout of the BF16 source, with the PLE table alone in part 3.
- **What is in each variant folder:**
  - `recipe.txt`: human-readable per-tensor rules, first match wins.
  - `tensor-types.tt`: the exact file given to the quantizer.
  - `plan.txt`: the size plan.
  - `quantize.log`: the quantizer's per-tensor log.
  - `sha256sums.txt`: hashes of the parts.
- **Checking a rebuild:** parts 2–8 of a rebuild can be checked against `sha256sums.txt`. Part 1 records the
  importance-matrix path you pass, so its hash depends on that path ([`VARIANTS.md`](VARIANTS.md), "Part 1").

| Tensors | Quality | Balanced | Lite |
|---|---|---|---|
| routed expert gate/up | **Q6_K** on 27 layers (0–6, 9, 14, 16, 24, 25, 30, 31, 35–47), Q5_K on 21 | Q5_K (layer 2 Q6_K) | Q5_K (layer 2 Q6_K) |
| routed expert down (640-wide rows) | Q8_0 on 2, 4, 30, 46, 47; Q5_1 on 43 | **Q8_0 on 20** (2, 4, 20, 23, 30–32, 35–47); Q5_1 on 28 | Q8_0 on 2, 4, 30, 46, 47; Q5_1 on 43 |
| attention, Gated-DeltaNet, `ssm_out`, shared experts, hyper-connections, token embedding, output head, PLE projections | Q8_0 | Q8_0 | Q8_0 |
| routers, norms, convolutions, hyper-connection injections, GDN alpha/beta | F32 | F32 | F32 |
| QSA indexer | BF16 | BF16 | BF16 |
| PLE n-gram table (streamed) | IQ4_NL (PLE4) or Q8_0 (PLE8) | IQ4_NL or Q8_0 | IQ4_NL |
| bits per resident weight | 6.26 | 6.14 | 5.89 |

- **Why the layers:** the Balanced body's 20 Q8_0 down-projection layers are unsloth's five plus the layers with
  the highest importance-matrix down-projection energy (mostly the tail, 35–47). The Quality body's 27 Q6_K
  gate/up layers are the high-energy tail, the head layers and the layers unsloth upgrades.
- **Splash constraint:** Splash allows only IQ4_NL, Q4_0, Q4_1, MXFP4, Q5_1 or Q8_0 for the 640-wide down rows,
  hence Q5_1/Q8_0.
- **Internal names:** the mapping from variant to internal build name, recipe and every SHA-256 is in
  [`VARIANTS.md`](VARIANTS.md).

## Repository layout

```
README.md  LICENSE  VARIANTS.md
Balanced-PLE8/   Qwen3.8-Flash-Next-Splash-Balanced-PLE8-0000{1..8}-of-00008.gguf  recipe.txt  tensor-types.tt  plan.txt  quantize.log  sha256sums.txt
Balanced-PLE4/   … same structure
Quality-PLE4/    … same structure
mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf      MTP head (unsloth's file, unchanged)
mmproj-Qwen3.8-Flash-Next-BF16.gguf              vision projector (unsloth's mmproj-BF16.gguf, unchanged)
imatrix_unsloth.gguf_file                        importance matrix (unsloth's file, unchanged)
kld/   KLD-TABLE.md  METHOD.md  one log per measured file
```

Variants share parts: Balanced-PLE8 and Balanced-PLE4 differ only in part 3. The 4-bit table in part 3 is the
same file in all PLE4 variants. Hashes are in `VARIANTS.md`.

## Credits

- **Qwen**: the Qwen3.8-Flash-Next base model and its licence.
- **unsloth**: the BF16 GGUF conversion we quantized from, the importance matrix, and the MTP head and vision
  projector redistributed here unchanged ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)).
  The GGUF metadata keys `general.quantized_by`, `general.repo_url` and `general.tags` are inherited from that
  conversion and still name unsloth.
- **AtomicChat**: the public BF16 reference logits and eval texts behind every quality number here
  ([metrics dataset](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics),
  [calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora)).
- **janhilgard**: Qwen3.8-Flash-Next support in Splash ([PR #253](https://github.com/incoai/splash/pull/253)).
- The Splash maintainers.

## Licence

These files are a derivative of Qwen3.8-Flash-Next and are distributed under the **Qwen Community License 1.0**.
The full text, including its copyright and permission notice, is in [`LICENSE`](LICENSE). It is identical to the
base repository's `LICENSE`. Copyright (c) 2026 Qwen.

Two conditions apply in particular:
- A commercial product or service with more than 100,000,000 monthly active users, or more than US$ 20,000,000
  monthly revenue, must display the model name prominently.
- A "Model as a Service" or "AI Work Assistant" business needs a separate licence from Qwen before any commercial
  use.

Read `LICENSE` for the exact terms.
