---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
language:
- en
- ru
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
datasets:
- SWE-bench/SWE-smith-trajectories
- open-thoughts/OpenThoughts-114k
- open-r1/OpenR1-Math-220k
- bigcode/self-oss-instruct-sc2-exec-filter-50k
- open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled
- nvidia/OpenCodeReasoning
- allenai/tulu-3-sft-mixture
- HuggingFaceM4/ChartQA
- HuggingFaceM4/DocumentVQA
- agentsea/wave-ui-25k
- lmms-lab/textvqa
- detection-datasets/coco
tags:
- mlx
- mtplx
- fp16
- apple-silicon
- m1
- m2
- oq
- reap
- expert-pruning
- dwq
- m4q
- quantization
- 3-bit
- mixed-precision
- mtp
- speculative-decoding
- self-distillation
- qwen3.8
- qwen4_exp
- moe
- image-text-to-text
- vision
- multimodal
---

# Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MTPLX

> ### Flash-Next with native MTP on Apple Silicon — the FP16 build for M1 and M2
> The REAP-320 `oQ3e` end-to-end-DWQ build of Qwen3.8-Flash-Next **repacked for
> [MTPLX](https://github.com/youssofal/mtplx)**, with a **self-distilled MTP sidecar** trained on the
> model's own long-context agent and reasoning traces. On a **48 GB M4 Pro** the identical
> weights decode at **~31–35 tok/s at depth 2** from short prompts to ~85k live context, vs
> 24.2 tok/s without MTP — measured on the BF16 pack; no M1/M2 timing is claimed.

## This is the FP16 build — read this first

This repo is the **FP16-typed sibling** of [`Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX`](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX).
Same weights, same quantization, same DWQ training, same evaluation numbers — everything below this
section describes both repos equally. The only difference is the **storage type of the tensors that
were never quantized**. M1 and M2 GPUs have no native BF16, so a BF16 checkpoint makes them upcast on
the fly; FP16 is native there. MTPLX has shipped an FP16 precision lane with M1/M2 auto-select since **2.2.0 (#166)** and publishes FP16 siblings of its own catalogue models for exactly this reason.

| Your Mac | Repo | Why |
|---|---|---|
| **M1, M2** (and any chip without native BF16) | **this one** | FP16 is the native 16-bit type there |
| **M3, M4 and newer** | [the BF16 repo](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MTPLX) | BF16 is native from M3 on, and that pack is 0.9 GB smaller |

Both produce the same answers. **Pick by chip, not by quality.**

### What "fp16" does and does not mean here

**It does not mean full precision.** Nothing is de-quantized and nothing is requantized. The routed
experts are still 3-bit, the n-gram table still 4-bit g32, the headline bits/weight is unchanged.
Every packed weight code — 82.6% of the pack — is copied **byte for byte** from the
BF16 repo. What changes is the remainder, the tensors that were never quantized in the first place:
quantizer scales, biases and norms become FP16, and the vision tower becomes FP32. MTPLX's own FP16 builds keep a bf16 tower, but this pack follows oMLX's stricter rule instead — FP32 keeps every vision value exact, is native on every Apple chip, and matches its oMLX sibling.

FP16 carries **10 mantissa bits against BF16's 7**, so at and above 2⁻¹⁷ every BF16 value is
*exactly* representable — the conversion is lossless in the value domain, not an approximation. The
only exposure is FP16's narrower exponent range, and it was measured rather than assumed:

| | |
|---|---|
| Largest magnitude in the pack | **14.88** (the FP16 ceiling is 65,504) |
| Values that would overflow to infinity | **0** |
| Values converted **exactly** | **99.99973%** of the 5,786,885,968 FP16-typed values |
| Values that flush to zero | **598** — of which 450 in `ngram.scales` |
| Values landing on FP16's coarser subnormal grid | 14,882 |
| Vision tower (448.9 M values) | promoted to FP32, so exact by construction |

450 of the flushed values are quantizer scales in `ngram.scales`, the streamed
3-gram/PLE table. They start four orders of magnitude lower than the body's scales because
the MTPLX repack folds the table's shared weight scale (≈2·10⁻⁴) into them, which is why
this pack has them and the oMLX one does not. A zeroed scale collapses that group of 32
table weights onto its own bias, moving each by at most 5·10⁻⁷ — and it happens to
450 groups out of the table's 1.6 billion. No scale in the trunk, the experts or the
MTP sidecar flushes. The remaining 148 are individual unquantized weights — Gated-DeltaNet
depthwise convolution kernels and MoE router rows — that were already smaller than 3·10⁻⁸
inside tensors whose own maxima are of order 1 to 10. A per-tensor breakdown ships in the repo as
[`fp16-conversion-manifest.json`](fp16-conversion-manifest.json). As an independent sanity check,
MTPLX reports "99.992% exact, none overflow" for the FP16 siblings of its own models, built by the
same cast.

**What was not measured:** the speedup itself. I have no M1 or M2 machine, so no timing anywhere in
this card comes from this pack on the chips it targets. What I verified is that the weights convert
exactly, that both runtimes carry compiled FP16 kernels for this architecture, and that the pack
loads and serves. If you benchmark it on your own M1 or M2, please open a discussion with numbers.

The trunk, expert and vision weights are the ones in
[Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MLX](https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MLX)
(REAP 320 of 512 experts · `oQ3e`: 3-bit routed experts + mixed-precision trunk · full-precision vision tower) — see that
card for the full pruning, quantization and DWQ recipe and the teacher-gap tables. This repo adds the
MTPLX layout, a runtime contract and a better MTP drafter.

**Precision:** 133.7 B parameters, **4.28 bits/weight** overall — 3-bit routed experts (56%), the 4-bit g32
n-gram table (39%), 5/6-bit GDN and attention projections, 8-bit shared experts, lm_head and MTP sidecar,
fp32 vision. Over the 82.5 B weights that stay in RAM (everything but the streamed table): **3.84 bits/weight**,
≈37 GiB. `config.json` states the bits and group size of every module explicitly.

> ⚠️ These are pruned and quantized weights. **The model's capability comes from Qwen's base model**
> — please star/cite it first. The MTP sidecar only drafts tokens; speculative decoding keeps the
> output distribution of the target model.

## Model lineage

```
Qwen/Qwen3.8-Flash-Next                   (Qwen Community License 1.0 · MoE 512 experts · vision + MTP)
  └─ Litwein/…-REAP320-oQ3e-DWQ-MTP-Vision-MLX   REAP 512→320 · oQ3e (3-bit experts, M4Q) · e2e KL-DWQ v9
       └─ Litwein/…-oQ3e-DWQ-MTP-Vision-MTPLX    byte-exact MTPLX repack + self-distilled MTP sidecar
            └─ THIS REPO: every 16-bit tensor recast for M1/M2; packed codes untouched
```

## What differs from the MLX repo

- **Layout.** The 3-gram PLE embedding table ships as a separate `ngram-table.safetensors` (~32 GB,
  4-bit g32) that MTPLX **streams from SSD** instead of holding in RAM; the vision tower is split
  into `model-vision.safetensors`; RMSNorm weights are stored in MTPLX's absolute convention and the
  table's weight scale is folded in. All other tensors are copied byte-for-byte.
- **MTP sidecar (`mtp.safetensors`).** Self-distilled from the model's own streamed generations
  (agent/tool-use and reasoning traces, reasoning kept in the targets) with trunk hidden states
  captured at long context. The objective is the **depth-2 acceptance chain under the production
  sampler** (top-k 20, top-p 0.95); exported as an 8-bit sidecar and adopted only after a live A/B
  against the previous sidecar:

  | Live A/B (depth 2, mean context ~38k) | Previous sidecar | **This sidecar** |
  |---|---:|---:|
  | Decode, greedy | 32.17 tok/s | **33.05 tok/s (+2.7%)** |
  | Decode, production sampler (T=1.0) | 31.44 tok/s | **32.04 tok/s (+1.9%)** |
  | Tokens per verify cycle (production) | 2.445 | **2.488** |

  The cycle cost was identical in both arms; the gain is draft acceptance.
- **Contract (`mtplx_runtime.json`).** Pins **depth 2** (`mtp_depth_default` / `mtp_depth_max`) and the
  runtime lanes validated for this model: sparse QSA prefill/gather, batched target distributions,
  fused verify glue, a 64k FR-Spec draft vocabulary and a 128 MB n-gram hot cache.

## Performance (measured by the uploader)

**No M1 or M2 numbers are claimed here.** The
figures below were measured on the identical weights in the BF16 pack, on newer silicon than
this pack targets; treat them as the family's ceiling, not as a prediction for your chip.

Apple **M4 Pro, 48 GB**, MTPLX 2.11.2, `turbo` profile, depth 2. Single-machine numbers; run-to-run
noise is about ±5%.

| Scenario | Result |
|---|---|
| Decode, short prompt | ~32–35 tok/s (AR without MTP: **24.2**; depth 1: 33.1; depth 3: 30.8) |
| Decode, 46k-token prompt | 31.5–34.4 tok/s |
| Decode, ~85k live context | 33.8 tok/s |
| Real agent traffic, 12k–34k context | 26–37 tok/s (35.6 at 34k) |
| Prefill, 46k tokens | ~180 tok/s (prefill chunk 512) · ~196 tok/s (chunk 2048) |
| Peak memory (small session cache) | 40.4 GiB at 46k · 43.0 GiB at ~85k |
| Context window | 163,840 tokens fit the 48 GB memory plan (architectural max 262,144) |

The decode cycle is essentially context-independent on this machine; it is bounded by memory
bandwidth for the roughly 4.6 GiB of weights read per verified token row, not by attention. Depth 3 is slower
than depth 2 here, which is why the contract pins depth 2.

## How to run

Requires **64 GB+ of unified memory**, **MTPLX ≥ 2.11.2**, ~73 GB of free SSD space and a fast
internal SSD (the n-gram table is streamed during decode). MTPLX measures this pack's resident
floor at **39.7 GiB** and keeps an 8 GiB system reserve on top, so it refuses to start below
48 GiB of RAM and wants both memory limits set above that floor — on the M1/M2 range that means
a 64 GB or 96 GB machine.

```bash
# Download into the MTPLX model directory
hf download Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MTPLX \
  --local-dir ~/.mtplx/models/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MTPLX

# Serve (limits for a 64 GB Mac: 52 GiB Metal limit, 42 GiB wired — both above the
# 39.7 GiB resident floor. Raise them on 96 GB; no smaller M1/M2 config fits.)
MTPLX_MEMORY_LIMIT_BYTES=55834574848 MTPLX_WIRED_LIMIT_BYTES=45097156608 \
  mtplx serve --model ~/.mtplx/models/Qwen3.8-Flash-Next-REAP320-oQ3e-fp16-DWQ-MTP-Vision-MTPLX \
    --host 127.0.0.1 --port 8000 --profile turbo --context-window 163840
```

The server exposes an OpenAI-compatible API on `http://127.0.0.1:8000/v1` (the model id is listed at
`GET /v1/models`). In the MTPLX app, add the downloaded folder as a local model and set the context
window and memory limit in its settings.

### Recommended settings

- **Sampling:** temperature 1.0, top_p 0.95, top_k 20 (the shipped defaults), thinking enabled.
- **Depth:** keep the contract's depth 2.
- **Session cache:** leave the RAM session cache on its automatic size. A small fixed cap evicts
  long-conversation prefixes and turns the next turn into a full re-prefill (measured after one
  interleaved side request: 241 s at 46k with a 2 GB cap vs an 8 s restore on auto).

## Intended use & limitations

- **Best suited to:** long-context coding and agent workflows and multimodal chat on an
  **M1 or M2** Mac with 64 GB or more (see the resident floor under How to run). On M3 and
  newer this pack has no advantage over the BF16 repo and is 0.9 GB larger; it is not a
  quality upgrade.
- **Same model caveats as the MLX repo:** pruning and 3-bit experts are lossy; DWQ is calibration,
  not SFT; no task or vision-benchmark score is claimed yet.
- **Memory pressure in long sessions:** the automatic session cache, KV state and MLX buffers push
  macOS into swap (observed from ~35k live context). Decode speed holds, but the rest of the system
  slows down.
- **KV-cache quantization** is not supported for this model family in MTPLX.
- **Parallel sessions (MTPLX 2.11.2):** two conversations that share a long prefix (for example the same
  system prompt and tool list) with image input enabled can fail with HTTP 409
  `session … is already in flight`, because the vision-keyed session lookup is not treated as an
  implicit session. Clients that send a distinct `x-mtplx-session-id` header per conversation do not
  depend on this prefix-based session matching.
- **MTPLX-only layout:** use the MLX repo for oMLX.

## Acknowledgements

- **Qwen team** — Qwen3.8-Flash-Next (Qwen Community License 1.0).
- **MTPLX** (youssofal/mtplx) — the native MTP runtime, pack contract, verify lanes and the
  FP16 precision lane for M1/M2 this repo follows.
- **Jundot**, **dfp-official** and **sh0wie** — the oQ4e donor, the oQ8e expert codes and the REAP
  keep-set manifests used by the MLX build.
- **Apple MLX** and **oMLX** — including the vision-stays-full-precision rule adopted here.
- **Calibration-data authors** — see the MLX repo.

## License

**Qwen Community License 1.0**, inherited from `Qwen/Qwen3.8-Flash-Next`; a copy is included as
[`LICENSE`](LICENSE). Operating a *Model as a Service* or an *AI Work Assistant* business commercially
requires a separate license from Qwen.

## Citation

Please cite the original base model:

```bibtex
@misc{qwen3.8-flash-next,
  title  = {Qwen3.8-Flash-Next},
  author = {Qwen Team, Alibaba Group},
  year   = {2026}
}
```
