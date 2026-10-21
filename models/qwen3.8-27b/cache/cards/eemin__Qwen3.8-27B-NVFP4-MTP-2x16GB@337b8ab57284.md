---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - modelopt
  - mtp
  - speculative-decoding
  - vllm
  - tensor-parallel
  - qwen3
language:
  - en
pipeline_tag: text-generation
---

# Qwen3.8-27B — NVFP4 + native MTP, tuned to run on **two cheap 16 GB GPUs**

A self-quantized **NVFP4** build of **Qwen3.8-27B** with **speculative decoding via the model's own MTP head**, deployed on **two consumer 16 GB cards (2× RTX 5070 Ti)** with vLLM tensor parallelism — reaching a genuine **150,000-token context** on current vLLM (160k on the older build this was developed against — see the version note). No A100/H100, no 48 GB card.

The point of this repo is the **deployment recipe**: what it takes to fit a 27B hybrid model + a 160k KV cache onto ~32 GB of split VRAM, on the cheapest hardware that can do it.

---

## TL;DR

| | |
|---|---|
| **Base** | `Qwen/Qwen3.8-27B` — dense 27B, hybrid (48 Gated-DeltaNet + 16 attention layers), `head_dim=256`, native MTP head |
| **Quant** | NVFP4 W4A16 on MLP + FP8 on attn/linear_attn + FP8 KV cache; **MTP head kept BF16**. ≈21 GB, **≈9.9 GiB/card** at TP2 |
| **Recipe** | NVIDIA ModelOpt, same recipe family as `nvidia/Qwen3.6-27B-NVFP4` |
| **Hardware** | **2× RTX 5070 Ti (16 GB each)** — cheap consumer Blackwell (sm_120) |
| **Runtime** | vLLM `--tensor-parallel-size 2`, **150k context** (current nightly), native MTP `K=4` speculative decoding |
| **Speed** | single-stream **~120–130 tok/s** on code, `accept_len ≈ 2.5–3.4` depending on content |
| **KV pool** | **150,000 tokens** at `util=0.94` on `v0.27.2rc1.dev110` (177,254 on the original build) |
| **Quality** | **AIME 2026 pass@1 29/30** — self-run, single stream, tools off (see Evaluation below) |

---

## Why this exists

A 27B model normally wants a single 40–80 GB GPU. This build targets the opposite end: **two ~16 GB desktop cards** joined by tensor parallelism. Two things make that hard, and both are solved by the config below:

1. **The KV cache is unusually heavy.** This architecture keeps `head_dim=256` on its 16 full-attention layers → **16,384 B/token/card**, ~4.9× a typical model. A naive setup can't reach 128k, let alone 160k, on 16 GB cards.
2. **A lazily-allocated FlashInfer workspace OOMs at high utilization.** vLLM reserves `util × VRAM` for weights + KV, but the attention workspace buffer is allocated *outside* that budget on the first request. At `util=0.96` with the default 128 MiB workspace, the very first generation OOMs. Shrinking the workspace to 64 MiB is what lets utilization go high enough to reach 160k.

---

## Quantization recipe (NVIDIA ModelOpt)

```bash
cd modelopt/examples/llm_ptq
python hf_ptq.py \
  --pyt_ckpt_path <Qwen/Qwen3.8-27B snapshot> \
  --export_path ./qwen38-nvfp4 \
  --recipe huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast \
  --dataset cnn_dailymail \
  --trust_remote_code
```

Notes that will save you time:

- **`--dataset cnn_dailymail` is required.** The 0.45 default calibration mix pulls a gated NVIDIA dataset and fails with `DatasetNotFoundError`.
- The recipe quantizes **401 layers**: MLP → NVFP4 W4A16 (group size 16), attn + linear_attn → FP8, `lm_head` → NVFP4, KV → FP8.
- **Do not quantize the MTP head.** The native MTP head (15 tensors) stays **BF16** and is excluded (`ignore: mtp*`) — identical to `nvidia/Qwen3.6-27B-NVFP4`. A quantized MTP head does not load in vLLM.
- Then assemble the head back into the checkpoint so the MTP tensors sit inside the model shards (method `mtp`, `mtp_num_hidden_layers=1`).

---

## vLLM deployment (2× RTX 5070 Ti, TP2)

See [`serve_tp2.sh`](./serve_tp2.sh). Core command:

```bash
docker run -d --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864 \
  -v /path/to/qwen38-nvfp4-mtp:/model -p 8000:8000 \
  vllm/vllm-openai:nightly \
    /model --tensor-parallel-size 2 --trust-remote-code --served-model-name qwen38 \
    --language-model-only \
    --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
    --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 150000 \
    --max-num-seqs 3 --max-num-batched-tokens 1024 --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

### Why each flag matters on 2×16 GB

| flag | reason |
|---|---|
| `--tensor-parallel-size 2` | splits the ~21 GB weights to **~9.9 GiB/card** |
| `--kv-cache-dtype fp8` | halves KV footprint (the binding constraint here) |
| `--mamba-cache-dtype bfloat16` `--mamba-ssm-cache-dtype bfloat16` | default is fp32; bf16 halves mamba pages → **+22% KV pool**, output bit-identical |
| `--attention-backend TRITON_ATTN` | intent: on sm_120 the FLASH_ATTN backend does **not** support FP8 KV. **Verified 2026-08-19: on `v0.27.2rc1.dev110` this flag is silently ignored** — the log reads `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`. FLASHINFER handles FP8 KV fine on sm_120, so the flag is now a no-op rather than a fix |
| `--gpu-memory-utilization 0.94` | packs the KV pool to **177k tokens** on the build this recipe was developed against; 0.96 OOMed on the first request. **See the version note below — this no longer holds on current nightlies.** |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864` (64 MiB) | shrinks the lazily-allocated attention workspace so `util=0.94` survives the first inference. Harmless for single-stream/low-concurrency (it is prefill scratch, not on the decode path) |
| `--max-model-len 150000` | real 150k single-stream context on `v0.27.2rc1.dev110`; the original build reached 160000 with the same utilization (version note below) |
| `--max-num-seqs 3 --max-num-batched-tokens 1024 --enable-chunked-prefill` | keeps peak activation low so the KV budget survives |
| `--speculative-config {"method":"mtp","num_speculative_tokens":4}` | native MTP head, **K=4**. K=4–6 all work; **K=4 is the sweet spot** — MTP drafts K tokens with K *serial* head forwards, so deep K over-drafts (a higher `accept_len` at K=6 does **not** translate into higher tok/s) |

### Version note — read this before copying the block above (2026-08-19)

The recipe above was developed against an earlier vLLM nightly. Re-running it
verbatim on **`v0.27.2rc1.dev110`**, on a second machine, surfaced two things:

- **`-e NCCL_P2P_DISABLE=1` has been removed from the block above.** It
  belongs to a host whose GPU-to-GPU P2P is blacklisted by the driver, where vLLM
  detects the lack of P2P and falls back on its own. On a host where P2P *works*,
  vLLM still selects the CUSTOM all-reduce backend, then cannot obtain a peer
  handle, and engine init dies with
  `Failed: Cuda error custom_all_reduce.cuh:164 'invalid argument'`. Add it back
  only if your host actually needs it.
- **160k no longer fits with these flags; the ceiling is ~150k.** Current vLLM
  counts CUDA-graph memory against `--gpu-memory-utilization`, so `0.94` is
  effectively `0.9164` (vLLM's own log suggests `0.9836` to restore the old
  budget). At `0.94` the engine now refuses 160000 with *"estimated maximum model
  length is 150096"*. Either raise utilization or set `--max-model-len 150000`.

Both were found by running this exact block, not by inspection. The numbers in
the next section come from the corrected `150000` variant.

---

## Measured — long-context profile (2× RTX 5070 Ti @ 250 W, single stream)

Same benchmark suite, same machine and same day as the throughput section below,
so the two profiles are directly comparable. 8 categories × 16 prompts, 8 warmups,
`--ignore-eos`, thinking off, concurrency 1, `--max-model-len 150000`.

| category | tok/s | accept_len (K=4) |
|---|---|---|
| math reasoning | **139.6** | 3.18 |
| translation | 134.2 | 3.06 |
| qa | 118.5 | 2.67 |
| rag | 115.9 | 2.94 |
| coding (Spec-Bench) | 111.7 | 2.51 |
| summarization | 108.0 | 2.71 |
| code (held-out, long prompts) | 105.9 | 2.97 |
| agent (held-out) | 105.3 | 3.18 |
| **mean (8 categories)** | **117.4** | |

- KV pool **150,000 tokens** — exactly 1.00× concurrency for one full-length request.
- Mean ITL 21.3 ms. These are `--ignore-eos` forced-length generations on real
  held-out prompts; clean short-form generation runs higher.

**Long context is nearly free here.** The throughput profile below runs at 96k
with an 8-slot scheduler and averages 114.8 tok/s on this identical suite — so
trading 96k for 150k of context costs nothing measurable in single-stream speed
(117.4 vs 114.8 is inside run-to-run noise on 7 of the 8 categories). What the
long-context profile actually buys is **31% more KV pool**, and what it gives up
is batching headroom, not tokens per second.

Earlier revisions of this card quoted 123–128 tok/s and a 177,254-token pool.
Those were measured on a different host and an older vLLM build and are not
reproducible on current nightlies; the table above replaces them.

---

## Throughput profile — 96k context, batched (2× RTX 5070 Ti @ 250 W)

The section above is the **latency profile**: one stream, maximum context. This
one is the opposite operating point — smaller context traded for concurrency,
measured as a full sweep. Different flags, so the two are not interchangeable.

### Exact setup for the numbers below

| | |
|---|---|
| GPUs | 2× RTX 5070 Ti, 16303 MiB each, **power limit 250 W** (board max 330 W) |
| Host | AMD Ryzen 5 7500F, ASUS ProArt B850-CREATOR |
| Driver | 595.71.05, sm_120 |
| Image | `vllm/vllm-openai:nightly` — engine `v0.27.2rc1.dev110+gacb0f1dcd` |
| Quant path | `modelopt_mixed`; **`MarlinNvFp4LinearKernel`** for the NVFP4 GEMMs, `FlashInferFP8ScaledMMLinearKernel` for the FP8 layers |
| Attention | **FLASHINFER** backend, decode kernel `xqa`, `kv_cache_dtype=fp8_e4m3` |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes up to 80 |
| Flags | `-tp 2 --language-model-only --gpu-memory-utilization 0.95 --max-model-len 98304 --max-num-seqs 8 --speculative-config '{"method":"mtp","num_speculative_tokens":4}'` |
| KV pool | **114,480 tokens** (1.16× the 98,304 max_model_len) |
| Load | 8 categories × 16 prompts, 8 warmups, `--ignore-eos`, thinking off |

vLLM's own log states the constraint plainly: *"Your GPU does not have native
support for FP4 computation … Weight-only FP4 compression will be used
leveraging the Marlin kernel."* On sm_120 the NVFP4 win is **footprint, not
math throughput** — weights land at 10.5 GiB/card, leaving room for the KV pool.

### Concurrency sweep

| concurrency | total tok/s | per stream | mean TTFT | mean ITL | p99 ITL |
|---|---|---|---|---|---|
| 1 | 114.8 | 114.8 | 285 ms | 21.9 ms | 22 ms |
| 2 | 203.2 | 101.6 | 338 ms | 24.6 ms | 120 ms |
| **4** | **322.9** | **80.7** | **423 ms** | **31.0 ms** | 254 ms |
| 6 | 348.0 | 58.0 | 1,176 ms | 36.3 ms | 259 ms |
| 8 | 360.1 | 45.0 | 2,156 ms | 34.5 ms | 247 ms |

**Concurrency 4 is the knee, and it is not close.** Going 2→4 buys +120 tok/s.
Going 4→6 buys +25 tok/s and **triples time-to-first-token**. Going 6→8 buys
+12 tok/s for another 1.9× on TTFT. Past 4 you are paying latency for a
rounding error.

Also worth knowing: `--max-num-seqs 8` versus `4` is **free and pointless** —
measured at concurrency 4, seqs=4 scored 322.9 and seqs=8 scored 323.9 (0.3%,
noise), and the only cost was 4.2% of the KV pool. The ceiling is not the
scheduler slot count.

### By category

| category | 1 stream | 8 streams | scaling | accept_len (K=4) | ITL @1 | ITL @8 |
|---|---|---|---|---|---|---|
| math reasoning | 135.9 | **482.3** | 3.55× | 3.17 | 21.9 | 27.4 |
| translation | 137.4 | 465.8 | 3.39× | 3.16 | 21.9 | 26.5 |
| qa | 118.0 | 416.5 | 3.53× | 2.65 | 21.8 | 26.4 |
| coding (Spec-Bench) | 112.0 | 396.4 | 3.54× | 2.53 | 21.9 | 26.9 |
| rag | 113.8 | 357.6 | 3.14× | 2.89 | 21.9 | 33.3 |
| summarization | 106.1 | 322.6 | 3.04× | 2.68 | 21.9 | 31.9 |
| code (held-out, long prompts) | 91.0 | 236.6 | **2.60×** | 2.64 | 21.9 | 49.0 |
| agent (held-out) | 104.1 | 203.0 | **1.95×** | 3.06 | 21.9 | 54.5 |

Three things fall out of this table:

1. **Acceptance length does not degrade under batching.** It sits at 2.85–2.96
   across the whole sweep. Speculative decoding and concurrency are not
   competing for the same resource here — you can have both.
2. **Long prompts are what kills scaling.** The two held-out sets have prompts
   several thousand tokens long, and they are the only two that fail to reach
   3× — agent manages just 1.95×, with per-token latency climbing to 54 ms.
   The bottleneck at concurrency 8 is prefill, not decode.
3. **Acceptance length does not predict throughput.** `coding` has the worst
   accept_len in the table (2.53) and the fourth-best throughput; `agent` has a
   good one (3.06) and the worst throughput. Prompt shape dominates.

### The limit nobody mentions

The KV pool is **114,480 tokens total**, shared. Eight concurrent streams means
roughly **14k tokens of context each**, not 8 × 98k. This profile serves eight
short-context sessions or one long one — never both. If you need long context
per session, use the long-context profile above and accept ~117 tok/s.

*(One outlier for honesty: `agent` scored 188.9 at concurrency 6, below its own
227.7 at concurrency 4 — the only non-monotonic cell in the sweep. n=16 per
category is not enough to smooth that out; don't build a story on it.)*

---

## Evaluation — AIME 2026 (I + II, 30 problems)

Throughput is worthless if the quantization made the model dumber, so this
checkpoint was run against the full AIME 2026 — both papers, all 30 problems.

Hardware for this run: **2× RTX 5060 Ti**, vLLM TP2 — *cheaper* cards than the
5070 Ti pair in the deployment section above, same 16 GB-per-GPU class.

**pass@1 = 29/30** — first sample of each problem, single pass, no retries.
**any-correct = 30/30** — Q10 was resampled and answered correctly on retry.
This is the weaker of the two metrics; quote pass@1.

| | |
|---|---|
| Tools | **none** — one `/v1/chat/completions` call per problem, no code execution, no search |
| Sampling | `temperature 0.6, top_p 0.95, top_k 20`, thinking **on** |
| Concurrency | **1** — strictly sequential, one problem at a time |
| Budget | `max_tokens 90000` per problem |
| Extraction | last `\boxed{N}` in the completion |
| Cost | 579,964 completion tokens, 147.8 min wall clock |
| Heaviest | Q15 at 1,749 s / 76.3k tokens; Q11 at 78.2k tokens |

That works out to **65.4 tok/s sustained across 2.5 hours of real reasoning**,
against 67.7 tok/s measured on the same cards with the short-prompt benchmark
suite. The MTP speedup holds on long chain-of-thought — it is not an artifact of
short benchmark prompts.

The single miss is **Q10**: 61,244 tokens, `finish_reason: stop`, answered 165
instead of 156. It burned more tokens than most of the problems it got *right*
and still went astray — a hard problem it over-thought, not a compute shortfall.
**No generation in the run hit the token ceiling**; all 30 ended with `stop`.

### What this run is evidence for

This is corroboration that the quantization did not damage the model — **not a
lossless claim.**

What it rules out: the loud failure modes. Garbled output, broken logic,
divergence on long generations, losing the thread mid-proof. A 4-bit checkpoint
that reasons correctly for 2.5 hours across 580k tokens is not a lobotomized one.

What it does not rule out: a quiet 1–2 point cost at the top of the range. There
is no BF16 control run on these same 30 problems, and Qwen's own model card
publishes no AIME figure for this model (it reports GPQA Diamond, HLE,
LiveCodeBench v6 and MathVision) — so there is no external anchor to subtract
from either.

### Read this before quoting the number

- **One sample is not a ceiling.** A second full pass over problems 10–30 scored
  19/21 — Q15 and Q30 flipped from correct to wrong. At `temperature 0.6`,
  29/30 means "one sample scored 29/30", not "solves 29 of 30 reliably".
- **Concurrency 1 is load-bearing, not laziness.** Batching this kind of long
  reasoning pushes some problems into non-terminating thinking loops that surface
  as `finish_reason: length` and score as wrong — an artifact of the harness, not
  of the model.
- Self-run, not an official leaderboard submission. Scores only — the problems
  and the transcripts are not published here.

---

## Caveats

- **Base is the official `Qwen/Qwen3.8-27B`** and retains its safety alignment.
- **sm_120 (consumer Blackwell).** W4A16 runs through Marlin weight-only on this arch; single-stream decode is bandwidth-bound, so the NVFP4 win here is memory footprint (fitting a 150k KV pool), not extra compute throughput.
- The **vision tower is present** in the checkpoint but **disabled at runtime** (`--language-model-only`); this is a text deployment.
- vLLM image: `vllm/vllm-openai:nightly` (needs a build new enough to resolve `Qwen3_5MTP` and the `mtp` speculative method under TP2).

---

## Acknowledgements

- Quantization method and recipe: **NVIDIA ModelOpt** / `nvidia/Qwen3.6-27B-NVFP4`.
- Base model: **Qwen** (`Qwen/Qwen3.8-27B`).
- Serving: **vLLM**.
