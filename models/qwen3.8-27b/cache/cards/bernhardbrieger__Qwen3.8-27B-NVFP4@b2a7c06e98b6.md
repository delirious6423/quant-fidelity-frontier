---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- code
- conversational
- long-context
- modelopt
- mtp
- nvfp4
- quantization
- qwen3.8
- qwen3_5
- sglang
- speculative-decoding
- vision
- vllm
- w4a4
library_name: transformers
quantized_by: Bernhard Brieger
---

# Qwen3.8-27B-NVFP4

Post-training **NVFP4 (W4A4)** quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced on a
**RTX 5090 (32 GB, sm_120) + Ryzen 9 9950X3D + 64 GB DDR5** system with
[NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.46.0.
The checkpoint ships the full multimodal architecture (language + vision tower)
plus the **Multi-Token-Prediction (MTP) draft layer**, and is verified serving
192k context (196,608 tokens) with an FP8 KV cache on
[SGLang](https://github.com/sgl-project/sglang) 0.5.20 at 0.88 utilization
— and, on the same box,
[vLLM](https://github.com/vllm-project/vllm) 0.29.0 (see Performance below).

Weight footprint drops from **55.56 GB (bf16) to 20.59 GB dec (20,592,849,504 B per the
safetensors index; NVFP4 shards: 19.7 GB main + 0.85 GB MTP) — a 62.9 %
reduction on the decimal-GB basis (55.56 → 20.59 GB)** — while the
precision-critical module families (embeddings, `lm_head`, GDN conv/in_proj,
norms, vision tower, MTP layer) are deliberately kept at full precision.
Every number on this card is traceable to a file in this repository
(`config.json`, `quantization_config.json`, `hf_quant_config.json`,
`model.safetensors.index.json`, `provenance.json`) or to a command in the
[Reproducibility appendix](#6-reproducibility-appendix).

> **A note on how to read this card.** The base model publishes no perplexity
> or benchmark table, and its 55.6 GB bf16 weights cannot co-reside with
> this checkpoint on a 32 GB GPU (sequential single-model runs would fit).
> A base-vs-quant delta is therefore **not** reported here; the quality
> section gives absolute numbers instead. That limitation is stated rather
> than papered over.

---

## 1. Identity & scope

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) |
| **Architecture** | `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`), image-text-to-text |
| **Parameters** | 27,781,427,952 (~27.78 B), identical to the base model — quantization changes the *storage format*, not the parameter count; on-disk footprint **20.59 GB dec** (20,592,849,504 B, W4A4) |
| **Quantization** | NVFP4 — **W4A4**: FP4 E2M1 weights **and** FP4 E2M1 activations, FP8 E4M3 block scales per 16 values, FP32 global scales |
| **Layers** | 64 — 48 Gated-DeltaNet linear-attention + 16 full-attention |
| **Native context** | 262,144 tokens (verified serving at 196,608) |
| **MTP** | 1 draft layer, 15 tensors, kept bf16, shipped in a dedicated shard |
| **License** | Apache-2.0 (base and derived) |
| **Quantized by** | Bernhard Brieger |
| **Quantization date** | 2026-08-30 |

> **Sidebar note — the HF "Model size" box is a display artifact, not the model size.**
> The auto-generated Safetensors widget on the HF page counts *raw stored
> elements* and does **not** unpack packed FP4 (each `U8` byte holds two 4-bit
> values) — so it reports **~15 B**, not the **27.78 B** logical parameter count
> above. The model is 27.78 B either way; the box is an HF UI quirk for packed
> FP4 storage, not a model property.

**Intended use.** General instruction following, coding, tool/function calling,
vision, and long-context workloads (retrieval over documents >100k tokens) on
hardware that fits ~20 GB of weights, with a runtime that supports NVFP4
(W4A4) execution — the verified serving paths are **SGLang 0.5.20
(recommended)** and vLLM 0.29.0, both on the `modelopt_fp4` backend (§4).
NVFP4 is a Blackwell-generation format: the W4A4 path
requires native FP4 tensor cores, which only Blackwell
(`sm_100`/`sm_120`) has — it cannot run on pre-Blackwell GPUs.

**Out of scope.** This is a plain quantization of the base model — no
fine-tuning, instruction distillation, or safety retraining was applied. It is
not intended as a substitute for the base model where full precision is
required, and the numbers below characterize this specific checkpoint, not the
base model in general.

**What is quantized vs. kept at bf16.** The W4A4 compression covers all 400
linear `mlp` and attention-projection modules across the 64 layers
(400 `input_scale` tensors were captured during calibration). The remaining
module families stay at full precision — token-embedding and `lm_head`
matrices (1.27 B each), the MTP draft layer (425 M), the vision tower
(461 M), and the per-layer layernorms / GDN `conv1d` / `in_proj_a` /
`in_proj_b` / output-gate parameters.

---

## 2. Quantization recipe

Self-contained recipe — the exclusion list and calibration settings are stored
in `quantization_config.json` and `hf_quant_config.json` in this repo.

### Toolchain

| Component | Version |
|---|---|
| NVIDIA ModelOpt | 0.46.0 |
| transformers | 5.14.1 |
| PyTorch | 2.13.0+cu130 |
| Python | 3.12 |
| GPU | 1× NVIDIA RTX 5090, 32 GB (sm_120) |
| CPU | AMD Ryzen 9 9950X3D, 16C/32T |
| RAM | 64 GB DDR5 |

### NVFP4 configuration

| Parameter | Value | Note |
|---|---|---|
| `preset` | `nvfp4` | W4A4 end-to-end (weights **and** activations) |
| `algorithm` | `max` | per-tensor global scales derived from the weight tensors (amax) |
| `group_size` | 16 | FP8 E4M3 block scale per 16 values |
| Activation scaling | `input_scale` (amax) | captured on the calibration corpus |
| Excluded modules | 147 entries (+ `mtp.*` in the serving config) | see table below |

### Modules kept at bf16 (exclusion families)

NVFP4 exclusions are an *explicit module list* (not regex patterns): a listed
module is left at full precision. `quantization_config.json` carries 147
entries; the serving-time `hf_quant_config.json` extends the same set with
`mtp.*` so the draft layer stays exact.

| Family | Count | Why |
|---|---:|---|
| `lm_head`, `embed_tokens` | 2 | output/input embeddings — load-bearing |
| `*.linear_attn.conv1d` | 48 | GDN causal convolution — not quantized (the W4A4 path does not cover causal conv) |
| `*.linear_attn.in_proj_a` / `in_proj_b` | 96 | GDN projection pairs — load-bearing |
| `*.norm` (incl. GDN / vision-merger norms) | — | not quantized; normalization parameters are kept in their original precision |
| `model.visual*` | 1 | entire vision tower stays bf16 |
| `mtp.*` | 15 tensors | keeps the speculative-decoding draft layer exact |

> **Why the MTP exclusion matters for serving.** vLLM gates MTP on any
> exclusion key whose text contains `mtp`. This repo lists `mtp.*` in the
> serving-time exclusion set — `hf_quant_config.json` (`exclude_modules`,
> 148 entries), mirrored in `config.json` (`quantization_config.ignore`) —
> which is what keeps the 15 bf16 MTP tensors loadable and lets
> `--speculative-config` work (see §4). A missing `mtp.*` entry is what
> originally broke MTP loading on this checkpoint.

### Calibration corpus

| Metric | Value |
|---|---|
| **Sequences** | 2,026 |
| **Total tokens** | 1,802,645 |
| **Longest sequence** | 132,968 tokens |
| **Random seed** | 42 |
| **Chunk size** | 8,192 tokens (32 activation-scaling blocks) |
| **Purpose** | activation `input_scale` (amax) for the W4A4 activations; weight scales come from the weight tensors themselves |

Built from a **curated pool of public, non-gated sources** (see
[License & provenance](#7-license--provenance)): 11 programming languages, an
agent/tool-calling pool, general web/wiki prose, a multilingual (DE/FR/IT)
pool, a math pool, and a **2×128k long-document tail**. Note that the
amax calibration is window-limited: it consumes only the corpus's first 32
8,192-token blocks — 262,144 tokens (≈14.5 % of the 1.8 M-token corpus).
Because the corpus is ordered longest-first (with a few inversions), that
window covers the two longest documents (~132,968 and ~131,432 tokens):
the first fully, the second to ~98 %. Shorter corpus sequences contribute
no activation scales. No forward pass exceeds 8,192 tokens — the long
documents are *content*-calibrated, not exercised as 128k/160k-length
forwards. The corpus manifest is referenced in
`provenance.json`; the raw sequences are not redistributed with this
checkpoint.

### Run statistics

| Stage | Wall time |
|---|---:|
| Model load (bf16) | 13 s |
| **Calibrate + compress** | **3.5 min** (155 s quantize+calibrate / 52 s compress) |
| Export (2 shards + MTP) | 13 s + injection |

Total wall time ~4 min (233 s across the four stages) on a single RTX 5090 — the
NVFP4 scheme only needs per-tensor activation scales (amax) collected over
the 32-block calibration window, with no per-module iterative error
compensation.

---

## 3. Quality verification

Four independent checks: a quantizer-statistics note, an absolute
perplexity, a functional text-smoke suite, and a 6-color vision suite —
all run against the live server.

### (a) Quantizer statistics

ModelOpt does not emit a per-module loss table; the build log's per-block
records serve as the audit instead. The log records **`input_scale: 400
tensors`** — every quantized linear module received a per-tensor activation
scale from the calibration corpus — and the verify step confirms
**`exclude mods: 147`** and **`weights: 19.7 GB`**. No module fell back to a
degraded strategy: W4A4 ModelOpt has no per-module fallback concept — either
the scale is captured or the module is excluded. The 147-entry exclusion list
in `quantization_config.json` (which includes `model.visual*`) is the
complete accounting; `mtp.*` is added only in the serving-time
`hf_quant_config.json`.

### (b) Perplexity (absolute)

| Metric | Value |
|---|---|
| Perplexity | **2.799** |
| Tokens scored | n = 260 |
| Method | `max_tokens=1` + `prompt_logprobs=1` over a fixed, neutral English prose sample; PPL = exp(−mean logprob) over the input tokens |
| Runtime | live vLLM (this checkpoint), `scripts/measure_ppl_quant.py` |

This is an **absolute** perplexity on a single 260-token sample — a sanity
check that the dequantized model assigns sensible probabilities to plain
prose, not a quality benchmark. The base model publishes no perplexity, so
there is nothing to diff against; the number is reported as-is.

### (c) Functional smoke (3/3 pass, live server)

| Probe | Result | Detail |
|---|---|---|
| Coding generation | PASS | 2,490 chars in 18 s (`enable_thinking: false`) |
| Long-context needle | PASS | target located at **46,037-token** context, returned `ZEBRAX42` (7 s) |
| Tool / function calling | PASS | exact JSON: `get_weather(city="Graz", unit="celsius")` (1 s) |

The needle probe exercises the long-context regime that the 2×128k
calibration tail was built to protect. The tool-call probe verifies the
`qwen3_coder` parser path end-to-end.

### (d) Vision probes (6/6 PASS on both serving engines, live servers)

| Probe | Result | Detail |
|---|---|---|
| Solid-color images (6) | PASS ×6 | red/green/blue 128×128, yellow 256×256, purple 64×64, cyan 512×512 — each answered with the exact color word |
| 40k-context needle | PASS | target located at **46,037-token** context, returned `ZEBRAX42` (7 s) — run on SGLang; the same probe ships as `needle_40k` in the standard smoke suite (`smoke_test_api.py`) |

The probes ran against **both verified engines**: vLLM 0.29.0 (production
server, read-only) and SGLang 0.5.20 (128k-context serving config,
mirroring the 128k-tier serving flags used for the performance tables) —
6/6 color probes on vLLM, 6/6 color probes + the long-context needle on
SGLang. Each image probe is a solid-color PNG
generated in-process (no asset files) with a strict "reply with the color
word only" instruction, so any other answer is a fail. This confirms the
vision tower (kept at bf16, §1) and the image-processor path (Qwen3-VL
image processor + `processor_config.json` shipped in this repo) work on
this checkpoint under both serving engines.

### What is *not* verified here

- **No base-vs-quant benchmark table.** The base card publishes no
  benchmarks or perplexity, and the 55.6 GB bf16 base cannot co-reside with
  this checkpoint on the 32 GB test GPU (sequential single-model runs would
  fit; a head-to-head would need ≥80 GB of VRAM).
- **No per-module loss table** — ModelOpt does not emit one (see §3a);
  quality evidence is the absolute PPL plus the 16 live probes
  (§3c/§3d: 3 text + the 6-color vision suite on each serving engine +
  the 40k long-context needle — all PASS).
- **MTP speculative decoding** is preserved (15/15 tensors, bf16); the
  numbers in the Performance section below include measured MTP
  runs — the verdict there is net-negative on this checkpoint.
- **Single-sample PPL** is a sanity check, not a benchmark (§3b).

---

## 4. Deployment

Two serving engines are verified for this checkpoint on an RTX 5090
(32 GB), 196,608-token context: **SGLang 0.5.20** (recommended path —
0.88-utilization flag set, FlashInfer attention, FP8 KV cache) and
**vLLM 0.29.0** (Cutlass NVFP4 GEMMs, FP8 KV cache, FlashInfer
attention, CUDA graphs active, 0.95-utilization flag set). A plain
`from_pretrained` of the packed FP4 weights in
transformers is **not a verified path** — the checkpoint is distributed
for the `modelopt_fp4` backend.

### SGLang 0.5.20 (recommended)

```bash
python -m sglang.launch_server \
  --model-path bernhardbrieger/Qwen3.8-27B-NVFP4 \
  --quantization modelopt_fp4 \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.88 \
  --context-length 196608 \
  --max-running-requests 4 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 0.14 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32 \
  --trust-remote-code
```

- **`--mem-fraction-static 0.88` is the recommended utilization** (was
  0.90). At 0.95 the engine does not start: after the Mamba state cache
  and KV allocation, the prefill CUDA-graph capture needs ~68 MiB at
  31.30/31.36 GiB in use (out of memory). 0.88 keeps headroom for lazy
  CUDA/Triton kernels on a shared-GPU (graphical) box; the §5 benchmark
  numbers were collected at 0.90 (headless) and are effectively
  unchanged at 0.88.
- **`--mamba-full-memory-ratio` is tier-tuned** for the hybrid GDN
  state cache: 0.60 @ 32k, 0.30 @ 64k–160k, 0.14 @ 192k.
- **Concurrency is tier-limited by the Mamba state cache:** effective
  concurrency 4 at 32k, 2 at 64k, 1 at 128k and above (the engine caps
  `max_running_requests` to the number of full-state slots that fit —
  12 slots at 128k–160k, 6 at 192k, 5 slots per request).
- **Startup:** server ready ~23–24 s after launch (160k/192k tiers,
  measured from the server log).
- **262,144 is not reachable on 32 GB**: the state-cache cap limits the
  KV pool to 222,212 tokens.
- **EAGLE speculative decoding is not usable at 0.88 (or 0.90)** at any tier (the
  speculative state shrinks the hybrid state cache below its minimum);
  the only MTP cell that ran was 32k at 0.95 utilization (mtp1 20.0,
  mtp2 16.6 tok/s per-request — see Performance below).

### vLLM 0.29.0

Verified serving configuration (0.95-utilization flag set):

```bash
vllm serve bernhardbrieger/Qwen3.8-27B-NVFP4 \
  --tensor-parallel-size 1 \
  --quantization modelopt_fp4 \
  --linear-backend cutlass \
  --max-model-len 196608 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --attention-backend FLASHINFER \
  --max-cudagraph-capture-size 64 \
  --trust-remote-code
```

- **`--linear-backend cutlass` is required, not cosmetic.** It routes the W4A4
  GEMMs to precompiled Cutlass kernels; without it the default path triggers a
  FlashInfer JIT that OOMs the host on 32 GB systems and hangs the engine.
- **Do not pass `--enforce-eager`.** The verified §4 configuration runs
  with CUDA graphs on; deviating from the verified flag set (including
  graph capture) changes the performance envelope — re-benchmark before
  changing it.
- **`--max-cudagraph-capture-size 64`** bounds graph capture to the
  bench/agent-size batch range — the mitigation currently recommended for
  sm_120 + hybrid GDN models on vLLM 0.29.x (vllm-project/vllm#52540,
  #54331).
- **Weights (on disk):** 19.7 GB (2 shards: 19.7 GB main + 0.85 GB MTP);
  steady-state engine consumption 19.24 GiB (weights + non-torch, per
  vLLM's memory profiler). At 192k context
  with an FP8 KV cache, the 0.95-utilization engine fills ~29.9 GiB of the
  31.36 GiB available (weights + non-torch 19.24 + KV pool 8.10 + peak
  activation 2.45 + CUDAGraph 0.07 GiB) — it fits, with `max_num_seqs 4`.
- **Startup:** ~40 s from process start to server ready (192k tier,
  measured from the server log).
- **Reasoning / tool parsing:** `--reasoning-parser qwen3` +
  `--tool-call-parser qwen3_coder` are what make structured tool-call output
  work (see the smoke probes in §3c).

### Optional: MTP speculative decoding

The MTP draft layer is present (15 tensors, bf16, in
`model-mtp.safetensors`, excluded from quantization). To use it:

```bash
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

The MTP weights remain fully usable **without** the flag (they are simply
unused), so the checkpoint runs as a normal model on any setup — speculative
decoding is opt-in. *Measured (vLLM 0.95 runs, below):* mtp1/mtp2 run at
32k–128k but are net-negative per-request at every tier; at 160k the spec
config cannot even start (KV wall) and at 192k it is unrunnable. Do not
expect a speedup from MTP on this checkpoint.

### Performance (measured)

Measured 2026-09-21/22 on an RTX 5090 (32 GB), FP8 KV cache; 1 warmup +
3 timed runs, `max_tokens=256`, temperature 0; prompts drawn from the §2
calibration corpus so that no prompt is a prefix of another — every
number below is a full prefill. Two throughput semantics, kept distinct:
**per-request** (256 completion tokens / total request time incl.
prefill, single stream) and **aggregate wall-clock total** across 4
concurrent streams.

**SGLang 0.5.20 @ 0.90**

Launch flags: `--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
--attention-backend flashinfer --max-running-requests 4
--chunked-prefill-size 2048 --mamba-full-memory-ratio <tier>
--mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32
--mem-fraction-static 0.90` (MTP as EAGLE: `--speculative-algorithm
EAGLE --speculative-num-steps <1|2>`). The mamba ratio is tier-tuned
(0.60 @ 32k, 0.30 @ 64k–160k, 0.14 @ 192k) to keep the hybrid
Mamba state cache inside the 32 GB budget.

| Context | TTFT p50 (s) | per-request (tok/s) | 4-stream aggregate (tok/s) |
|---:|---:|---:|---:|
| 32,768 | 3.28 | 39.1 | 49.31 |
| 65,536 | 9.52 | 16.39 | 22.35 |
| 131,072 | 30.94 | 7.37 | 7.41 |
| 163,840 | 46.29 | 4.13 | 4.89 |
| 196,608 | 64.66 | 3.72 | 3.74 |

- 262,144: not reachable — the hybrid state cache cap (mamba ratio
  0.30) limits the KV pool to 222,212 tokens, below what the 262,144
  tier requires.
- **SGLang at 0.95 utilization does not start** (out of memory at
  startup in every no-MTP cell): after the Mamba cache and KV
  allocation, the prefill CUDA-graph capture needs ~68 MiB at
  31.30/31.36 GiB used. On this 32 GB card SGLang is usable at 0.88
  (recommended) or 0.90 (the headless §5 benchmark) only.
- **SGLang MTP (EAGLE) is not usable at 0.90**: at every tier the
  speculative intermediate state shrinks the hybrid state cache below
  its minimum, so the engine cannot start. The only SGLang MTP cell
  that ran was 32k at 0.95 utilization: mtp1 20.0 tok/s, mtp2
  16.6 tok/s (per-request, single stream).

**vLLM 0.29.0 @ 0.95 — no spec decode:**

| Context | TTFT p50 (s) | per-request (tok/s) | 4-stream aggregate (tok/s) |
|---:|---:|---:|---:|
| 32,768 | 3.23 | 31.8 | 52.8 |
| 65,536 | 9.16 | 18.1 | 20.7 |
| 131,072 | 29.08 | 7.4 | 7.5 |
| 163,840 | 43.09 | 4.3 | 5.0 |
| 196,608 | 103.01 | 2.4 | 2.4 |

- The 262,144 tier is **not reachable on 32 GB**: at 0.95 utilization
  the largest context that fits in the KV budget is ~202,272 tokens, so
  the engine fails at startup for that tier.

**Phase breakdown (single stream, no spec decode)** — prefill = prompt
tokens / TTFT p50; decode = per-token time (TPOT p50). 4-stream cells
are not decomposed: the KV pool caps effective concurrency at 4 (32k),
3 (64k) and 1 (131k+), so the excess requests queue server-side and
their TTFT includes the wait.

| Tier | SGLang 0.5.20 @ 0.90 prefill (tok/s) | SGLang decode (ms/tok → tok/s) | vLLM 0.29.0 @ 0.95 prefill (tok/s) | vLLM decode (ms/tok → tok/s) |
|---:|---:|---:|---:|---:|
| 32,768 | 9,896 | 12.8 → 78 | 10,064 | 18.9 → 53 |
| 65,536 | 6,855 | 13.5 → 74 | 7,124 | 19.5 → 51 |
| 131,072 | 4,227 | 14.8 → 67 | 4,498 | 20.8 → 48 |
| 163,840 | 3,533 | 15.5 → 65 | 3,796 | 21.5 → 47 |
| 196,608 | 3,037 | 16.1 → 62 | 1,906 | 22.1 → 45 |

Prefill cost grows with context length in both engines; the decode
phase is context-independent per token and runs ~48 % faster on SGLang
(12.8 vs 18.9 ms/token at 32k). SGLang's 0.95 utilization is unusable
here (see above), so the SGLang column reflects its only serviceable
configuration.

**vLLM 0.29.0 @ 0.95 — MTP speculative decoding (opt-in):**

| MTP | 32k (1 / 4 streams) | 64k (1 / 4 streams) | 128k (1 / 4 streams) | 160k | 192k |
|---|---|---|---|---|---|
| none | 31.8 / 52.8 | 18.1 / 20.7 | 7.4 / 7.5 | 4.3 / 5.0 | 2.4 / 2.4 |
| mtp1 | 22.4 / 15.5 | 9.6 / 10.5 | 3.6 / 3.6 | **does not start** | **not executable** |
| mtp2 | 17.3 / 21.4 | 7.5 / 8.0 | 2.6 / 2.6 | **does not start** | **not executable** |

- *Does not start* (160k): the MTP intermediate state needs 5.8 GiB of
  KV budget against 5.47 GiB available. *Not executable* (192k): the
  MTP intermediate state exceeds the hybrid state budget at the 32 GB
  ceiling.
- Measured acceptance (vLLM, 0.95 utilization, 32k–128k tiers,
  per-run spec-decode metrics): mtp1 mean acceptance length 1.80–1.95
  (draft acceptance 80–95 %); mtp2 2.14–2.84 (57–92 %).
  **Verdict: MTP is net-negative on this checkpoint when served with
  vLLM 0.29.0 at 0.95 utilization on this card** (the tested spec
  configuration: FP8 KV, 32k–128k tiers) — per-request throughput below
  `none` at every tier (mtp1 ×0.71, mtp2 ×0.54 at 32k), and 4-stream
  aggregate below `none` at 32k/64k/128k. The draft-step overhead
  outruns the 1.8–2.8 accepted tokens. The MTP tensors cost nothing
  unused — keep them, but do not expect a speedup. (The only SGLang MTP
  cell that ran — 32k at 0.95 — sits in the same net-negative class.)


---

## 5. Hardware requirements

| Deployment | Minimum VRAM |
|---|---:|
| Weights only (short context) | ~20 GB |
| 192k context, FP8 KV, 4 seqs (verified) | 32 GB (RTX 5090) |

The verified configuration uses the full 31.36 GiB of usable VRAM at
0.95 utilization. The 19.7 GB main-shard footprint (+0.85 GB MTP shard,
checkpoint size) is a fixed cost: a GPU below ~20 GB cannot load this
checkpoint at any context length. The remaining budget goes to
the KV cache, which scales with context length and concurrency — it
determines how much context a given card can serve. The verified serving
paths are **SGLang 0.5.20 (recommended)** and vLLM 0.29.0, both on the
`modelopt_fp4` backend (Cutlass GEMMs), on
Blackwell hardware (`sm_100`/`sm_120`) — the only GPUs with
native FP4 tensor cores, and a hard requirement for the W4A4 path.
NVFP4 is a Blackwell-generation format: the W4A4 path (FP4 compute on
both weights and activations) cannot run on pre-Blackwell GPUs, which
have no FP4 tensor cores — running this checkpoint there is out of
scope for this release.

---

## 6. Reproducibility appendix

Every number in §1–§3 is produced by one of the following. Paths are relative
to this repository; the raw calibration corpus is documented in
[`provenance.json`](provenance.json) but not redistributed with this
checkpoint.

### 6.1 Quantization

The checkpoint is produced by ModelOpt 0.46.0 over the 2,026-sequence corpus
(documented in `provenance.json`: seed 42, 1,802,645 tokens; the raw
sequences are not redistributed). The build uses the corpus's first 32
8,192-token blocks for the amax calibration — a 262,144-token window
(≈14.5 % of the corpus) that, with the corpus ordered longest-first,
covers the two longest documents (~132,968 and ~131,432 tokens) — the
2×128k tail — essentially in full:

```python
# scripts/build_nvfp4.py (abridged)
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization import quantize, compress
from modelopt.torch.export.unified_export_hf import export_hf_checkpoint

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3.8-27B", torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "28 GiB", "cpu": "32 GiB"},  # caps leave VRAM for the calib forward
)
model = quantize(model, QUANT_CFG_CHOICES["nvfp4"], forward_loop)  # 32-block calibration
compress(model)
export_hf_checkpoint(model, "output/Qwen3.8-27B-NVFP4", max_shard_size="10GB")
```

The `nvfp4` preset is W4A4 with the `max` algorithm: weight scales come from
the weight tensors, activation `input_scale` (amax) from the calibration
forward. Before the export, the script consolidates every tensor to a single
device (modelopt's export-time dummy forward is CUDA-only) and clears the
stale `hf_device_map` so the save takes the whole-state-dict path.

ModelOpt's export writes weights + config only. Two post-steps complete the
checkpoint (both in `scripts/inject_mtp_and_processors.py`, both idempotent):

1. **MTP injection** — transformers' `Qwen3_5` discards `^mtp.*` at load, so
   the 15 bf16 draft-layer tensors are copied from the base shard into
   `model-mtp.safetensors` (+ index entry). Without this step MTP loading
   breaks.
2. **Processor injection** — tokenizer/processor files
   (`tokenizer.json`, `preprocessor_config.json`,
   `processor_config.json`, `video_preprocessor_config.json`, …) are copied
   from the base snapshot; the export writes none of them. Without
   `preprocessor_config.json` the vLLM engine fails to boot; without
   `processor_config.json`, `AutoProcessor.from_pretrained` on the repo
   fails.

`quantization_config.json` (exclusions, calibration metadata) and
`hf_quant_config.json` (serving-time quant config, `mtp.*` in both
exclusion lists) are emitted by the builder and verified in step 5 of
`build_nvfp4.py`.

### 6.2 Perplexity

Against a running server on `:8000`, scoring the fixed 260-token sample with
input-token logprobs:

```bash
PPL_MODEL=bernhardbrieger/Qwen3.8-27B-NVFP4 python scripts/measure_ppl_quant.py
# -> quant_ppl  n=260  perplexity=2.799
```

### 6.3 Functional smoke (text + vision)

```bash
python scripts/smoke_test_api.py
# -> 3 probes: coding / needle_40k / tool_call

python scripts/vision_smoke.py
# -> 6 solid-color probes (red/green/blue 128x128, yellow 256x256,
#    purple 64x64, cyan 512x512); each must be answered with its
#    exact color word
```

The text probes ran against the **vLLM 0.29.0 production server** (the
default daily serving engine); the 6-color vision suite plus the 40k
needle were additionally verified on **SGLang 0.5.20** through the same
OpenAI-compatible endpoints. Both scripts work on either engine — point
them at whichever server is live.

---

## 7. License & provenance

**Model weights.** Apache-2.0. The base
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is Apache-2.0 and
this derived checkpoint is released under the same license.

**Calibration data.** The corpus is drawn entirely from public, non-gated
datasets: seed 42, the pools listed in §2, and the 2×128k long-document tail.
Every source, its license, and
(for code) its pinned commit are listed in
[`provenance.json`](provenance.json) in this repository. The raw
calibration sequences are not redistributed with this checkpoint.

| Source | License | Used in |
|---|---|---|
| 28 code repositories | Apache-2.0 (13) · MIT (13) · BSD-2-Clause (1) · BSD-3-Clause (1) | code pools (11 languages) |
| [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | per-file upstream license (no dataset-level license declared) | Python pool |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (EN/DE/FR/IT) | CC BY-SA 3.0 + GFDL | wiki prose, long-document tail |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | math pool |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | ODC-BY | agent / tool-calling pool |
| [HuggingFaceTB/SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/SmolLM-Corpus) (FineWeb-Edu dedup) | ODC-BY | web prose, long-document tail |

Each code repository is pinned to a specific commit in
[`provenance.json`](provenance.json) (repo + commit SHA + license), so the
code portion of the corpus is reconstructable byte-for-byte.

**Data attribution.** Wikipedia text © Wikipedia contributors, released under
CC BY-SA 3.0 and the GFDL. FineWeb-Edu and the Tülu-3 SFT mixture are used
under ODC-BY. Attribution is provided here in lieu of per-file notices.

*Card data updated 2026-09-23: SGLang 0.5.20 now the recommended serving path (vLLM 0.29.0 secondary); SGLang + vLLM server startup times added; build run-statistics total corrected to 233 s across the four stages; MTP acceptance ranges re-derived from all per-run spec metrics (vLLM @0.95, 32k–128k: mtp1 1.80–1.95 / 80–95 %, mtp2 2.14–2.84 / 57–92 %); vision verification expanded to a 6-color probe suite re-run on both serving engines (vLLM production + SGLang 128k config, 6/6 each) plus the 40k-context needle on SGLang, with `scripts/vision_smoke.py` upgraded to the 6-probe set; earlier local edits: vLLM 0.29.0 (was 0.27.1) 0.95-utilization-only values, SGLang 0.90 section + `sglang` tag (kept before `vllm`), internal terminology removed from the Performance section, deployment examples at 196,608-token context; phase breakdown added to the Performance section (prefill tok/s + per-token decode time from the raw matrix runs).*

**Base model references.**
- Base model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3.5 / Qwen3.8 technical report: see the links on the base model card.
