---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
thumbnail: https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney/resolve/main/assets/mooney.png
tags:
  - gguf
  - llama.cpp
  - buun-llama-cpp
  - quantization
  - ternary
  - dgx-spark
---

# Qwen3.8-Flash-Next-Mooney

## 2.125-bit experts · 2.39 bits/weight across the transformer · 92.0 GB download (4.16 bits/weight overall) · ≈39 GiB in memory · 95.5% of BF16's score

## Thank you, [Zach Mueller](https://x.com/TheZachMueller) and Lambda

**This model would not exist without [Zach Mueller](https://x.com/TheZachMueller) and [Lambda](https://lambda.ai).** Every step of this work ran on an 8× A100 80 GB machine that Lambda generously provided: building the BF16 reference, quantizing the experts, the recovery training, and every quality evaluation on this card. I'm deeply grateful to Zach and to Lambda for making it possible. Thank you.

![Qwen3.8-Flash-Next-Mooney](assets/mooney.png)

![Mooney keeps 95.5% of BF16's score](assets/quality_overall.png)

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**

A 180B-parameter model that runs on **one DGX Spark**.

**Powered by [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)** — huge thanks to [spiritbuun](https://github.com/spiritbuun) for building it and adding native support for this quant. It's the default inference engine for this model; build + run commands below.

*Official source: this repository. If you'd like to share the model, please link here rather than re-uploading.*

## Run it on a DGX Spark

**Default runtime: [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)** — a public llama.cpp fork that loads these files natively (`PQ2_0` experts, `lowbitflash.rot.*` rotation metadata, SSD-lazy PLE reads) and runs MTP speculative decoding with the optional MTP head. Download the files from this repo (sha256s in `manifest.json`), then:

```bash
git clone https://github.com/spiritbuun/buun-llama-cpp && cd buun-llama-cpp
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j$(nproc) --target llama-server

./build/bin/llama-server \
  -m Qwen3.8-Flash-Next-Mooney-PQ2_0-00001-of-00004.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Mooney.gguf \
  --spec-draft-model mtp-Qwen3.8-Flash-Next.gguf --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl all -fa on -np 1 -c 32768 -t 16 \
  --no-cache-prompt --cache-ram 0 \
  --host 127.0.0.1 --port 8089
```

`GGML_NATIVE=ON` targets the Spark's GB10 GPU. The 54.4 GB PLE table is read from SSD on demand (default `-lzm auto`); drop the `--spec-draft-model` line to run without MTP. On a DGX Spark these settings measured **43.5 tok/s** decode (short context), **46.4** (4k) and **40.0** (30k) with the MTP head, single stream, greedy; MTP output was token-identical to no-MTP output.

> [!NOTE]
> **ARM/GB10 build note (commit `4594f213`):** master currently needs a one-line fix to link on aarch64 — in `ggml/src/ggml-cpu/arch-fallback.h`, delete the `#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0` line from the `__aarch64__` block (the same kernel is also defined in `arch/arm/quants.c`). Reported upstream; once a fix lands, pin it instead.

One-command alternative — [mooney-spark](https://github.com/DJLougen/mooney-spark) builds the cuda.fast engine, downloads the weights with per-file sha256 verification, and writes launchers:

```bash
git clone https://github.com/DJLougen/mooney-spark && cd mooney-spark
./setup_spark.sh                      # fast engine (default) — or --runtime llama.cpp
~/mooney-spark/launch/serve_ds4.sh    # OpenAI-compatible server on http://127.0.0.1:8000/v1
```

Stock `llama.cpp` cannot run these files — the routed experts use a `PQ2_0` ternary GGML type plus `lowbitflash.rot.*` rotation metadata. Three runtimes understand them:

- **buun-llama-cpp** (default) — [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp): serial + MTP speculative decoding; **images work** via the `mmproj` file; VBR KV cache.
- **Fast engine** — [cudafast-qwen38-125b-a6b-engine](https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine), branch `lbf/pq2-rot` (our port of the cuda.fast ds4 engine): serial + MTP speculative decoding + **image input** (images are encoded by a small mtmd/clip helper built by the setup script; MTP auto-disables for image requests).
- **llama.cpp fork** — [prism-llama.cpp](https://github.com/DJLougen/prism-llama.cpp), branch `lbf/flashnext-ternary`: serial decode; **images work** here via the `mmproj` file.

Details, memory floors, and verification: the [mooney-spark](https://github.com/DJLougen/mooney-spark) repo.

## Why Mooney
Mooney faces — from Craig Mooney's 1957 closure study — are photographs reduced to pure black and white patches. Almost all the detail is gone, yet people still recognize the face: the whole survives. This model does the same thing to the weights. (Mooney images are also a known test of whole-shape perception in vision AI research.)

## What this is

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` is a 180B-parameter experimental MoE checkpoint: a 125B MoE language model (48 layers, 512 routed experts, top-10 routing, ≈6B parameters active per token) plus a 51.2B "PLE" n-gram embedding table that the model consults during inference.

This repo is a quantized build of that checkpoint, produced with the **Mooney recipe**:

- **Routed experts**: ternary weights stored in a rotated basis, then recovered with distillation training.
- **PLE table**: 8-bit, read lazily from SSD; frequently used rows stay cached in memory.
- **Non-expert weights**: 8-bit (`Q8_0`) in this build; small tensors stay F32/BF16.
- **Vision**: separate BF16 `mmproj` file.
- **MTP (multi-token-prediction) head**: separate optional `mtp-Qwen3.8-Flash-Next.gguf` (Unsloth Q8_0 conversion, unmodified); enables speculative decode on buun-llama-cpp and the fast engine.

Where the bits go (counted from the GGUF tensor headers):

| Part | Parameters | Format | Bits/weight | Size |
|---|---|---|---|---|
| Routed experts | 120.8B | ternary (`PQ2_0`) | **2.125** | 32.1 GB |
| Other transformer weights | 4.9B | 8-bit (`Q8_0`); small tensors F32/BF16 | 8.9 | 5.5 GB |
| **Transformer total** | **125.7B** | | **2.39** | **37.6 GB** |
| PLE n-gram table | 51.2B | 8-bit (`Q8_0`) | 8.5 | 54.4 GB, stays on SSD |
| Text model, all 4 shards | 176.9B | | 4.16 | 92.0 GB |

The base checkpoint's 180B also counts a 2.6B MTP head (shipped separately as the optional MTP file) and a 0.45B vision tower (in the separate 0.9 GB `mmproj` file).

## Why it matters

![Fits on one DGX Spark with room to spare](assets/memory.png)

In BF16 this model needs ≈335 GiB — several DGX Sparks or an 8-GPU server. A common 4-bit GGUF build is ≈104–114 GB and already fills a Spark's memory by itself. Mooney runs in **≈39 GiB** (measured on a DGX Spark, MemAvailable delta incl. KV/context; ≈35 GiB of that is weights), leaving room for long context and other work on the same box, while scoring **within ≈4 points of BF16** on our verifier suite. The 54.4 GB PLE table is read from SSD on demand instead of being loaded; in our SSD replay of real lookup traces, most decode tokens needed no disk read once frequently used rows were cached.

## Quality (our own measurements, 2026-09-30)

536 tasks whose answers are checked automatically (unit tests, exact answers, schema and tool-call checks), run twice per model on vLLM (8×A100) with the same weight values; this exact GGUF was then measured on a DGX Spark for speed and memory:

- **BF16 original:** 500 and 496 of 536 → **92.91%**
- **Mooney:** 475 and 476 of 536 → **88.71%** (95.5% of BF16)


![Score by task category](assets/quality_by_category.png)

**Long context:** needle retrieval **10/10 at each of 32k, 64k and 128k tokens (30/30 total) — identical to BF16** (5 depths per length, 2 needles each, strict exact match, held-out text never seen in calibration or eval). On the fast engine at 256k, **5/5** needle retrievals (10–90% depths, verbatim `XONQUIL-7742` in a 258,913-token prompt).

![Needle retrieval by context length](assets/long_context.png)

<details>
<summary>Per-token agreement on long natural text</summary>

Teacher-forced top-1 agreement with BF16 is ≈72% and truncated top-20 KL ≈0.27 on packed long English text, ≈0.40 on Chinese/German — versus a BF16-vs-BF16 repeat floor of ≈96% agreement / KL ≈0.005–0.010. The gap is roughly flat with position (flat on zh/de; ≈+9% KL from the first 2k to 64–128k on English): quantization costs fidelity uniformly, but there is no long-context degradation signature, and retrieval is intact at 128k. Per-model attribution (release vs intermediate builds on the same sequences) shows the gap comes from the shared ternary-expert layer — the 8-bit non-experts contribute ≈0.003 KL and the 8-bit PLE ≈0.0055 KL. Teacher-forced fidelity is not generation quality; the verifier suite and needles are the behavioral check.

</details>


Most categories are unchanged or better than BF16 — short instructions actually improved. The gap sits in arithmetic, long reasoning chains and coding.

<details>
<summary>Exact numbers</summary>

| Model | Run 1 | Run 2 | Mean |
|---|---|---|---|
| BF16 original | 500/536 (93.28%) | 496/536 (92.54%) | **92.91%** |
| **Mooney** | **475/536 (88.62%)** | **476/536 (88.81%)** | **88.71%** |

Paired against BF16 on the same tasks: +7 gained / −32 lost (run 1), +10 / −30 (run 2).

| Category (tasks × 2 runs) | BF16 | Mooney |
|---|---|---|
| Coding, unit tests (134) | 129 | 121 |
| Error recovery (50) | 48 | 46 |
| Output format (30) | 30 | 30 |
| Long instructions (50) | 50 | 50 |
| Short instructions (40) | 38 | 40 |
| JSON schema (60) | 60 | 60 |
| Math: arithmetic (124) | 87 | 71 |
| Math: word problems (60) | 60 | 58 |
| Multi-step coding (60) | 57 | 53 |
| Multi-turn state (60) | 60 | 60 |
| Multilingual (80) | 78 | 75 |
| Rare-name copying (80) | 80 | 79 |
| Needle retrieval, up to 19k tokens (50) | 50 | 50 |
| Early-stop bait (34) | 34 | 34 |
| Reasoning chains (60) | 55 | 44 |
| Tool calls (50) | 50 | 50 |

**Non-expert 8-bit check (this build):** compared with the same model with BF16 non-experts — teacher-forced per-token metrics on held-out text: eval2048 cross-entropy delta −0.0024/−0.0010 (runs 1/2) against a repeat-vs-repeat A/A delta −0.0006; top-1 agreement 0.972 vs A/A 0.977; truncated top-20 KL ≈0.006 vs A/A 0.004. 8-bit non-experts change almost nothing; the gap vs BF16 lives almost entirely in the ternary experts (see Long context).

**PLE 8-bit check:** 476/536 on the same suite; per-token distribution metrics vs the same model with the unquantized PLE table (cross-entropy delta, top-1 agreement, truncated top-20 KL) within A/A noise (e.g. eval2048 dCE +0.0002 vs A/A −0.0006); on the long-context corpus the PLE layer adds ≈0.0055 KL.

</details>

Teacher-forced fidelity and needle retrieval are measured to 128k tokens (above); the verifier suite tops out near 19k. The suite above is our own harness; no third-party benchmark numbers exist for this checkpoint yet.

## Speed on one DGX Spark

![Decode speed on one DGX Spark](assets/spark_decode.png)

**Mooney runtime (cuda.fast-based engine)** — [cudafast-qwen38-125b-a6b-engine](https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine), branch `lbf/pq2-rot`. Decode, single stream (tok/s):

| Engine | Short | 4k | 32k | 64k | 128k | 256k |
|---|---|---|---|---|---|---|
| Mooney runtime + MTP head (default, `--mtp-draft 3`) | **50.7** | **46.4** | **40.5** | **41.2** | **39.7** | **40.5** |
| Mooney runtime + MTP head, `--mtp-draft 2` | 48.9 | 43.0 | 39.0 | 38.9 | 37.7 | 36.9 |
| Mooney runtime, no MTP | 31.5 | 27.8 | 27.2 | 26.7 | 25.8 | 24.3 |

Greedy decoding (temperature 0) through the runtime's OpenAI-compatible server with the setup script's launcher settings (`-c 262144`), engine `c9e0679f`. Each cell is the median of 3 runs that each generate exactly 128 tokens (rate = 128 tokens ÷ decode time), on held-out documents with a question at the end; run-to-run variation ≤0.5%. The 256k column uses a 256,457-token prompt. MTP output is token-identical to no-MTP output on all 15 prompts tested (up to 128k context), at both draft settings.

MTP speeds up greedy requests only. Sampled requests (temperature above 0, including the server default) run at the no-MTP speed.

Correction (2026-10-01): an earlier version of this table was measured on engine `8811b9a7`, which checked MTP drafts against an approximate shortlist of the vocabulary; its greedy MTP output differed from no-MTP output on 11 of 15 test prompts. That table also compared Mooney with Unsloth's 4-bit UD-Q4_K_XL build (≈114 GB) and called Mooney 1.4× faster. Re-measured with the same server harness on the stock cuda.fast engine, UD-Q4_K_XL decodes at 30.4 tok/s (short) and 27.3 tok/s (4k) without MTP, about the same as Mooney (31.5 / 27.8). With MTP the stock engine measured 52.9 (short) and 34.3 tok/s (4k), using its approximate draft check. So Mooney's advantage over a 4-bit build is memory (it leaves most of the Spark free), not single-stream speed at short context.

Fast-engine prefill ≈1,045 tok/s (32k) → ≈926 tok/s (256k; a 258,869-token prompt ingests in ≈280 s). (from the 2026-09-30/10-01 campaign on the same engine family; not re-measured since.)

Memory at load with `-c 262144` (MemAvailable drop): **57.8 GiB** without MTP, **60.8 GiB** with the MTP head (`--mtp-draft 3`) — about half the Spark.

**llama.cpp fork** (same Spark, same GGUF, `--cache-ram 0` guard):

| Metric | Value |
|---|---|
| Model load | 99 s |
| Memory | ≈39 GiB resident (38.6 GiB MemAvailable delta at load; ≈35 GiB is weights) |
| Decode | 27.8 tok/s (short) · 25.8 tok/s (4k) · 17.6 tok/s (30k) |
| Prefill | ≈750 tok/s (4k) · ≈666 tok/s (30k) |
| PLE SSD replay (trace replay, not end-to-end) | prefill PLE reads ≈5.5–8.6 µs/token; most decode tokens needed no disk read |

Single stream, current kernels — median of 5 runs per depth, one launch. Sparse attention (QSA) runs as designed; numbers measured with it on. PLE replay numbers carry over (the PLE table is byte-identical).

## Required runtime

Stock `llama.cpp` **cannot** run this. The expert tensors are GGML type 142 (`PQ2_0`) and the file carries `lowbitflash.rot.*` metadata the loader must honor.

- **Default: [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)** (public) — loads the `PQ2_0` experts and rotation metadata natively, reads the PLE table lazily from SSD, runs MTP speculative decoding with the repo's MTP head, serves images via the `mmproj` file. Pin commit `4594f213` (2026-10-06) or later. Build + run commands under **Run it on a DGX Spark** above.
- **[mooney-spark](https://github.com/DJLougen/mooney-spark)** one-command setup — builds our patched fork of the cuda.fast engine ([cudafast-qwen38-125b-a6b-engine](https://github.com/DJLougen/cudafast-qwen38-125b-a6b-engine), branch `lbf/pq2-rot`, with MTP speculative decoding) or, with `--runtime llama.cpp`, our llama.cpp fork ([prism-llama.cpp](https://github.com/DJLougen/prism-llama.cpp), branch `lbf/flashnext-ternary`).

### Advanced / other GPUs: manual runs

**buun-llama-cpp** — the build + run commands under **Run it on a DGX Spark** above work on any CUDA GPU (`-ngl all` assumes the whole model offloads; lower it otherwise).

**Our llama.cpp fork** — command line that produced the measured llama.cpp-fork Spark numbers (exact argv):

```bash
llama-server \
  -m Qwen3.8-Flash-Next-Mooney-PQ2_0-00001-of-00004.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Mooney.gguf \
  --load-mode mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU \
  -ngl all -fa on -np 1 \
  --no-cache-prompt --cache-ram 0 -c 32768 --reasoning auto \
  --host 127.0.0.1 --port 8089
```

`--load-mode mmap --tensor-read-lazy on -ot per_layer_token_embd=CPU` keeps the 54.4 GB PLE table on SSD instead of in memory. `--cache-ram 0` stops the host prompt cache from growing during long sessions.

## Files

| File | Contents |
|---|---|
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00001-of-00004.gguf` | KV metadata + 58 tensors (1.6 GB) |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00002-of-00004.gguf` | PLE table alone — one tensor (54.4 GB); byte-identical to the previous release's PLE shard |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00003-of-00004.gguf` | 848 tensors (25.0 GB) |
| `Qwen3.8-Flash-Next-Mooney-PQ2_0-00004-of-00004.gguf` | 317 tensors (11.1 GB) |
| `mmproj-Qwen3.8-Flash-Next-Mooney.gguf` | Vision tower + projector, BF16 |
| `mtp-Qwen3.8-Flash-Next.gguf` | MTP speculative-decode head, Q8_0 (Unsloth conversion, unmodified) — optional |
| `manifest.json` | sha256 + sizes, source/converter/runtime revisions |
| `LICENSE` | Qwen Community License 1.0 (from the base model) |

`PQ2_0` in the shard names is the expert quant type (2-bit codes, one scale per 128 weights); the Hub's quant list shows it as `Q2_0`. The vision and MTP files carry no quant label in their names so they aren't listed as model variants. The Hub's "Use this model" buttons launch stock runtimes, which can't load these files; use one of the runtimes under **Required runtime** — [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) by default.

SHA-256 for every file is in `manifest.json`.

## Verification performed

- **Per-tensor bit-exactness**: all 1224 tensors in the shards stream-hashed (sha256) against the monolithic build — every tensor's data bytes identical.
- **KV/metadata**: shard 1 carries the full original KV (353 keys, incl. 290 `lowbitflash.rot.*` rotation entries) plus `split.no/count/tensors.count`; shards 2–4 carry only the split keys. `split.tensors.count = 1224`. Shard 1's `general.name` was rewritten to `Qwen3.8-Flash-Next-Mooney`; its data region is byte-identical to the original shard.
- **PLE unchanged**: the PLE shard is byte-identical (sha256) to the previous release — the 8-bit change touched only non-expert transformer weights.
- **Attention schedule**: `compress_ratios` corrected to the canonical schedule (12 QSA layers) in the shipped shard 1; earlier builds of this repo had all-zero ratios, which made llama.cpp run dense attention at those layers.
- **Conversion-time checks**: all expert tensors are type 142, expert + PLE payloads bit-exact vs source candidates, 594 non-expert tensors quantized to `Q8_0` and verified bit-exact, small tensors kept F32/BF16, `lowbitflash.rot.version = 1` with 144 rotated weight names, no MTP/visual tensors.
- **mmproj**: the 333 `model.visual.*` tensors in the candidate snapshot were verified byte-identical to the upstream checkpoint before conversion.

## Limitations

- Requires a patched runtime — stock llama.cpp cannot load type-142 tensors; [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (public, default) and our patched forks can.
- ≈4 points under BF16 on our suite; arithmetic, long reasoning chains and coding carry most of the gap. Not lossless.
- Per-token agreement with BF16 on long natural text is ≈72% (top-1; truncated KL ≈0.27–0.40), mostly from the ternary experts. It doesn't grow with position, and needle retrieval stays 30/30 to 128k (5/5 at 256k on the fast engine).
- MTP speculative decoding runs on buun-llama-cpp and the fast engine (cudafast fork); the prism llama.cpp fork runs without it.
- MTP only speeds up greedy requests (temperature 0). Sampled requests, including requests that leave temperature at the server default, decode at the no-MTP speed (≈31 tok/s short context). Before engine commit `2f623be8` (2026-10-01) such requests returned an empty reply with `finish_reason: "error"` when the MTP head was loaded; re-run `./setup_spark.sh` to update.
- MTP head ships as a separate optional file (`mtp-…gguf`); the main GGUF does not include it.
- Image requests in the fast engine are not bit-for-bit repeatable run to run (answers were stable in testing); text requests are deterministic.


## License

Weights are a derived work of **Qwen3.8-Flash-Next**, distributed under the **Qwen Community License 1.0** (`LICENSE` file included).

## Credits

Expert quantization recipe builds on **PrismML's Bonsai** work; runtime is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and PrismML's llama.cpp fork (MIT); the default runtime is [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (MIT). MTP head: Q8_0 conversion by Unsloth ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)), unmodified. Compute generously provided by [Lambda](https://lambda.ai), thanks to [Zach Mueller](https://x.com/TheZachMueller) — this release would not have been possible without them.

## Support This Work

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**
