---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - ukisai/Swift-1.5-Qwen3.8-27b
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - ninfer-v3
  - qwen3.8
  - qwen3_5
  - swift
  - swift-1.5
  - nvfp4
  - w4a4
  - smoothquant
  - quantization
  - blackwell
  - multimodal
  - conversational
  - cuda
  - dflash2
  - speculative-decoding
  - image-text-to-text
---

# Swift-1.5 Qwen3.8-27B — **SmoothQuant + NVFP4-MLP** + DFlash2 for NInfer (v3 container)

A **quality-first** quantization of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
for the [NInfer](https://github.com/Neroued/ninfer) engine: the MLP bulk is 4-bit (NVFP4
W4A4) but its **activation range is calibrated on a SmoothQuant-smoothed model**, the head
and embedding use a finer Q8 grid than our other builds, and the vision tower is stored
**lossless (BF16)**. DFlash2 speculative drafter and the indexed proposal head are embedded.
Native **v3** container.

## Who this is for

- **You have VRAM headroom** and would rather spend it on quantization robustness than on
  maximum tokens/sec. This build is **22.80 GiB** of weights, versus **18.42 GiB** for our
  all-NVFP4 profile of the same base model
  ([swift15-nvfp4full-dflash2](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3)).
- **You serve agentic / long-context / coding traffic**, where 4-bit activation clipping
  hurts most: SmoothQuant re-scales the activation outliers into the weights *before*
  quantization, so the 4-bit grid is fitted to the workload instead of the tails.
- **Vision you can drop.** The tower is stored in BF16 for OCR/detail fidelity, which is
  most of the extra bytes. If you never send images, drop `--vision` (or build without the
  vision component) and most of the size premium disappears.

It **is slower per long reasoning question** than the all-NVFP4 profile — measured
**13.99 s vs 10.96 s per GPQA-Diamond question (+27.6 %)** at a ~96 % reasoning-token share,
for **+4.00 pp GPQA-D and +2 AIME questions** (see the A/B below). That is the trade this
artifact exists to offer: fewer quantization compromises, less throughput.

## Credits & provenance

**All credit for the model itself goes to the authors below** — this is a format conversion
plus quantization of their work, with no fine-tuning, no abliteration and no distillation of
our own:

| Role | Model | Author |
|---|---|---|
| **Source weights (BF16)** | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | **ukisai (UkisAI)** |
| **Upstream base** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | **Qwen team** |
| **DFlash2 drafter** | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | **z-lab** |
| **Engine** | [Neroued/ninfer](https://github.com/Neroued/ninfer) | **Neroued** |
| Quantization stack | `llm-compressor` (SmoothQuant + `nvfp4-pack-quantized`) | vLLM project |
| Calibration corpora | `nebius/SWE-rebench-openhands-trajectories`, `IFM/Code-Reasoning`, `openbmb/UltraData-Code`, `HuggingFaceH4/MATH-500`, `HuggingFaceH4/ultrachat_200k` | respective authors |

Swift-1.5 is UkisAI's reasoning-efficient derivative of Qwen3.8-27B: RL + OPD post-training
aimed at long-horizon agentic and coding work, with pathological overthinking penalized.

## What we did (short version — full recipe in [`RECIPE.md`](./RECIPE.md))

1. **SmoothQuant (α = 0.5) on the BF16 base**, as a separate pass. Per-channel scale folded
   into the preceding layernorm (`g /= s`) and the consuming weights (`W *= s`).
   Mathematically identical model, smaller activation dynamic range. Uses llm-compressor's
   built-in Qwen3.5 hybrid-attention mappings (full-attention-only `input_layernorm → q/k/v`,
   all-layers `post_attention_layernorm → gate/up`).
2. **NVFP4 W4A4 on MLP layers 0–55, calibrated on the *smoothed* model.** Two separate
   `oneshot` passes on purpose: in a single pass the A4 `input_global_scale` is finalized
   from the pre-smoothing amax, which would give a stale, too-coarse grid.
3. **One divisor per fused parent**: the engine's native A4 route needs one contiguous fused
   parent per input, so per-group `weight_global_scale` values are unified to `min(dᵢ)` and
   the E4M3 block scales rescaled (RNE, shrink-only) — 0.0000 % re-encode error here.
4. **Fixed layout for everything else** (FP8 attention + GDN, BF16 GDN decay/beta, Q8 head
   and embedding, BF16 vision), then `tools.convert` to the v3 container with DFlash2 + MTP +
   the indexed proposal head.

Calibration mix is workload-weighted (agentic 40 % / coding-reasoning 25 % / code 20 % /
math 10 % / chat 5 %, 485 samples, seq 2048) — a *range-fitting* step, not a precision
allocation.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_swift15_smoothquant-nvfp4mlp-dflash2.ninfer` |
| Size | 24,481,291,008 bytes (**22.80 GiB**) |
| **SHA-256** | `5036adc0a29d0947abca7319ce2540513e6b25166d63cda3f3fd7a4688387d78` |
| Container version | **3** |
| Components | Text + Vision + MTP + DFlash2 + indexed proposal head |
| Objects | 1,246 |
| Formats | `nvfp4` 112 (MLP 0–55) · `fp8` 144 (attn, GDN, MLP 56–63) · `q8_g32` 28 (head/embed + drafter + merger) · `q4` 1 (proposal head) · `bf16` 690 · `fp32` 96 · `int32` 1 |
| Hash manifest | `SHA256SUMS` |
| Build recipe + scripts | [`RECIPE.md`](./RECIPE.md), [`build/`](./build) |

## Serving

Any NInfer v3 build with the shapes above. Example (single 72 GB card, full 262 k context,
DFlash2 speculation, nvfp4 KV):

```
ninfer-serve qwen3_8_27b_swift15_smoothquant-nvfp4mlp-dflash2.ninfer \
  --model-id Qwen3.8-27B --max-context 262144 \
  --kv-dtype nvfp4 --max-concurrency 6 --prefill-chunk 1024 \
  --spec dflash2 --draft-tokens 7 \
  --host-context-mib 51200 --device-state-slots 6 \
  --vision --default-thinking-budget 16384 --preserve-thinking
```

- **Stock engine:** `--spec dflash2` and some long-context helpers live in fork builds; the
  model runs fine without speculation — drop `--spec dflash2 --draft-tokens 7` on stock
  NInfer. Vision needs the fork's image-token budget
  ([Neroued/ninfer#61](https://github.com/Neroued/ninfer/pull/61)); omit `--vision` on stock.
- **Reclaiming the size premium:** without `--vision` the tower is unused; shipping without
  the vision component brings the artifact close to the all-NVFP4 profile's footprint.
- Boot on the reference box: weights 22.4 GiB resident, ~15 s to ready.

## Benchmarks

**Conditions:** RTX PRO 5000 72 GB (Blackwell), one GPU, NInfer v3 (`ninfer-clean`,
fork `main` @ `51d4349f`, upstream `64d10fbf`), nvfp4 KV, 262 k context, thinking budget
16 384, sampling temperature 0.9 / min-p 0.05 for the eval suites unless stated.
`inspect_ai` + `inspect_evals`, single pass, no few-shot, no answer caching.

| Suite | Protocol | Score | Reference |
|---|---|---|---|
| **GPQA-Diamond** | 50 questions × 4 answer orderings = **200 samples** | **91.00 %** ± 3.4 pp | published Swift-1.5 BF16 card: 88.59 % |
| **AIME 2026** | 30 questions (n=30 ⇒ 1 question = 3.3 pp) | **90.00 %** (27/30) | published Swift-1.5 BF16 card: 96.00 % (5-seed, 131 k ctx — a different protocol) |
| Gate: needle retrieval | 12 depths/positions, 64 k | **12/12 exact, 0 corrupted** | — |
| Gate: quality probes | 4 fixed prompts | **4/4** | — |
| Gate: streaming tool calls | 10 runs, strict schemas | **10/10 valid** | — |

Single-stream decode on the same card: **102 tok/s** at 0 ctx (samples 110.5 / 101.8 / 93.6);
prefill TTFT 12.5 k → 2.04 s, 50 k → 10.55 s, 100 k → 29.24 s, 200 k → 92.03 s.

### A/B vs our all-NVFP4 profile of the same base model

Same box, same engine build, same lane, same harness, same served model id (the id is
version-free, so the two artifacts are drop-in comparable). Both with nvfp4 KV at 262 k ctx.

| Benchmark | all-NVFP4 profile (18.42 GiB) | **this artifact (22.80 GiB)** | Δ |
|---|---|---|---|
| **GPQA-Diamond** (200 samples) | 87.00 % ± 4.2 | **91.00 % ± 3.4** | **+4.00 pp** |
| **AIME 2026** (30 questions) | 83.33 % (25/30) | **90.00 % (27/30)** | **+6.67 pp (+2 questions)** |
| Time per GPQA question | 10.96 s | 13.99 s | **+27.6 %** |
| Aggregate tokens/s (GPQA) | 597 | 483 | −19.1 % |
| Weights on disk | 18.42 GiB | 22.80 GiB | +4.38 GiB |
| Reasoning share of output tokens | 95.6 % | 96.0 % | ≈ |

**Read this honestly:** the two GPQA-D intervals overlap — the delta is ~0.7 σ
(±5.4 pp combined standard error at n=200). What makes the direction credible is that
**both** suites move the same way, AIME by two full questions, and that the *mechanism* is
the expected one (4-bit activation outliers are what hurt; SmoothQuant removes them before
quantization). Treat it as *"a quality gain that is consistent but not individually
significant, bought with ~28 % more time per long-reasoning question and 4.4 GiB"*.

Pick this artifact when a wrong answer costs more than 4 GiB and 28 % latency; pick the
all-NVFP4 profile when you are decoding at volume and can live with the quantization
compromise.

## Notes

- **Why the A4 grid matters.** In W4A4 the *activations* are 4-bit too. A handful of outlier
  channels per layer can dominate the per-tensor scale, so the grid becomes coarse for
  everything else. SmoothQuant moves that mass into the weights before quantization; the
  divisor measured afterwards reflects the distribution the model actually sees.
- **Vision is the largest single premium** (BF16, 27 blocks) and is only worth it if you send
  images. Two vision tensors stay quantized (`merger/fc2` Q8, `position_embedding` Q5) because
  the engine has **no BF16 kernel** for those shapes — a hard constraint, not a choice.
- **MTP is stored but not loaded** unless you pass `--spec mtp`; it costs disk, not decode.
- **KV cache format is a serve-time flag**, not part of the artifact.

## kaushikvira Qwen3.8-27B artifact family

| artifact | base | quantization | size | vision | note |
|---|---|---|---|---|---|
| [swift15-nvfp4full-dflash2](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3) | Swift-1.5 | all-NVFP4 W4A4 | 18.42 GiB | q4/q5/q6/q8 | throughput profile; see its card for the same-family A/B |
| **this artifact** | Swift-1.5 | SmoothQuant + NVFP4 MLP, Q8 head/embed | **22.80 GiB** | **BF16** | quality/headroom profile |
| [swift15-uncensored-ajgazin](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3) | Swift-1.5 + ajgazin ablation | all-NVFP4 W4A4 | 18.42 GiB | q4/q5/q6/q8 | uncensored variant |
| [thinkingcap](https://huggingface.co/kaushikvira/Qwen3.8-27B-thinkingcap-nvfp4full-dflash2-NInfer-v3) | ThinkingCap | all-NVFP4 W4A4 | 18.42 GiB | q4/q5/q6/q8 | PolyForm — non-commercial |

## Acceptable Use Policy & Legal Notice

### Permitted use
Research, evaluation, personal and commercial deployment subject to the upstream licenses of
the base models (Qwen3.8-27B: Apache-2.0; Swift-1.5 as published by UkisAI; DFlash2 as
published by z-lab). You must independently verify that your use complies with those licenses.

### Prohibited use
No use intended to cause harm; no illegal activity; no generating content that violates
applicable law; no circumventing the upstream models' own usage terms; no misrepresenting
this artifact as an official release of ukisai, the Qwen team or z-lab.

### Your responsibility
You are responsible for what you generate and how you deploy it, including any jurisdiction-
specific obligations, disclosure requirements and downstream safety measures.

### No warranty / limitation of liability
Provided **as-is**, without warranty of any kind. The authors of this conversion accept no
liability for any use, output, or damage arising from it. No support commitment is implied.

## Measurement conditions (for reproducibility)

Scores in the table above were produced on a single RTX PRO 5000 72 GB with the flags quoted
in **Serving**, `inspect_ai`/`inspect_evals` (`gpqa_diamond` with `--limit 50`, which expands
to 200 samples across 4 answer orderings; `aime2026` with `--limit 30`). The engine reports
the served model under a version-free id, so re-runs across requants of the same slot are
directly comparable. Decode/tok-s figures come from a 3-sample gate run and a longer perf run
on the same box; they are **not** comparable across different GPUs or with other lanes
resident.
