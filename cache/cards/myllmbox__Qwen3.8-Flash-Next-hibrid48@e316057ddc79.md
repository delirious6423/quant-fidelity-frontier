---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - nvfp4
  - dgx-spark
  - gb10
  - vllm
  - quantized
  - multi-node
---

# Qwen3.8-Flash-Next — hibrid48: hibrid47 with a 4-bit output head

**Two DGX Sparks, 262k context, 22 engine steps/s — 92 tok/s single-stream on code (107 peak), 546 tok/s at 32 streams.
Same body, same table, same drafter as hibrid47; one tensor changed.**

The per-step profile of the hibrid47 serve put the output head at **27 % of every decode step**: a 1.18 GiB bf16
matrix that speculative decoding reads about 5.4 times per step (the drafter proposes four tokens through it, the
verify pass reads it once more). hibrid48 stores that one tensor as NVFP4 (W4A16: e2m1 codes, an fp8 scale per 16
values, one fp32 global scale — 0.33 GiB) and lets the Marlin kernel dequantize it in-register. Nothing else moved.

| | hibrid47 | **hibrid48** |
|---|---|---|
| engine steps/s, c=1 (same image, same boxes) | 16.9 | **22.0** (+30 %) |
| tok/s, c=1, code, thinking off | 74–76 | **92** (86–97) |
| tok/s, c=1, peak 10-s window (thinking on, code) | 80 | **107** |
| tok/s, 32 streams | 533 | **546** |
| draft acceptance (pasture prompt, code) | 4.1–4.3 | 4.2–4.4 |

## Precision map

| tier | precision | notes |
|---|---|---|
| routed experts | NVFP4 W4A4 (from [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)) | 39 % of the checkpoint |
| GDN linear-attention projections | NVFP4 W4A16, Marlin (myllmbox) | A/B-validated tolerant |
| **lm_head** | **NVFP4 W4A16, group 16, fp8 scales, fp32 global** | **new in hibrid48**; mean \|Δw\| = 9.0 % of \|w\|; the MTP drafter shares it |
| QSA attention q/k/v/o, shared experts, embeddings, norms, hyper-connection mixers | bf16 | the quality tier |
| MTP drafter: routed experts | NVFP4 W4A4 (Inferact) | |
| MTP drafter: everything else | bf16 | |
| PLE n-gram table (320,001,536 × 160) | NVFP4, 28.6 GiB, 8 shards `ple-nvfp4-*.safetensors` | hibrid47's, unchanged |

## Measured (2 × DGX Spark GB10, TP=2 over ConnectX RDMA, vLLM 0.29, MTP K=4, Marlin MoE, bf16 KV)

Pasture prompt (a 1.2k-token code checklist). Each row averages steady 10-s windows across 4–5 independent runs.

| | steps/s | tok/s avg | peak | acceptance | first token |
|---|---|---|---|---|---|
| c=1, thinking off | 22.1 | **92** (86–97) | 100 | 4.2–4.4 | 0.57 s |
| c=1, thinking on (reasoning → code, 38k tokens) | 22.0 | 79 (75–79) | **107** | 3.4–3.6 (prose ≈3, code ≈4.9) | — |
| c=32, thinking off | 4.1 | **546** (534–561) | 561 | 4.2 | — |
| prefill | | 2.5–2.7k tok/s | | | 32k-token prompt: 12 s cold |

The engine step is flat at 45–46 ms from the first token to the last; tok/s = steps × accepted tokens, so the text
decides the number: reasoning prose accepts ≈3 drafts per step, dense code ≈4.9 of the possible 5.

**KV.** bf16 KV at a 28 GB pin = 1,708,119 tokens (6.5 × 262k). fp8 KV boots too (2,846,834 tokens) but cost
0.3 accepted tokens per step on this lane in a 5+5-run A/B (4.23 → 3.92), so bf16 is the shipped default.

## Quality

lm-evaluation-harness against the running serve, **thinking on**, Qwen's thinking-mode sampling (0.6 / 0.95 / 20),
fixed 200-question subsets (seed 123123123; HumanEval complete), 32k-token budget:

| task | score |
|---|---|
| HumanEval pass@1 (164) | **95.7** |
| GSM8K (200) | **98.0** |
| IFEval prompt-strict / instruction-strict (200) | **91.5** / 93.4 |
| MMLU-Pro (200 across 14 subjects) | **84.9** |

11 of 763 answers hit the 32k budget (thinking runaways); they count as wrong above. The same harness run on hibrid47
is the pending A/B — these are the model's numbers, not yet a measured delta against the bf16 head. On the 32-scene
render gauntlet (thinking on) hibrid48 produced complete rosters in 8 of 8 renders. The tooling (`bench/quality/` in
the myllmbox repo) is reproducible against any OpenAI-compatible serve.

## Loading — read this

vLLM 0.29 builds the output head **without** the checkpoint's quantization config, so a stock 0.29 fails on this
checkpoint with a shape mismatch (`size of tensor a (2560) must match (1280)`). The fix is two lines — pass
`quant_config` to `ParallelLMHead` in the model's `model.py` and `mtp.py` — shipped as
[`recipes/qwen38-flash-next-fast/docker/patches/11-lm-head-quant-config.py`](https://github.com/bilikaz/myllmbox-runner/tree/main/recipes/qwen38-flash-next-fast)
in the myllmbox repo, and applied by that recipe's image. Until it lands upstream, serve this checkpoint with that
image, or use hibrid47 (bf16 head) on stock vLLM.

```
git clone https://github.com/bilikaz/myllmbox-runner.git && cd myllmbox-runner
./download.sh myllmbox/Qwen3.8-Flash-Next-hibrid48
./build-and-copy.sh qwen38-flash-next-fast     # vLLM 0.29 + the lane's patches (incl. the lm_head fix), copied to the second Spark
./run.sh qwen38-flash-next-fast                # two-Spark TP=2 serve on :8000
```

Fit: ~50 GiB per box with the table sharded across the pair; one Spark loads it (~101 GiB) but leaves too little KV —
for a single Spark, hibrid46 remains the release.

## Repo layout

21 body shards (hibrid47's; shard 1 rewritten without the bf16 head — every tensor exists exactly once) +
`model-lmhead-nvfp4.safetensors` (the head: `lm_head.weight` packed uint8 [248320, 1280], `lm_head.weight_scale`
fp8 [248320, 160], `lm_head.weight_scale_2` fp32) + 8 table shards + index. `config.json` = hibrid47's with
`quantization_config.quantized_layers["lm_head"] = W4A16_NVFP4 / group 16` and `lm_head` removed from `ignore`.

## Reproducibility

`builds/qwen38-flash-next/quantize-lm-head.py` in the myllmbox repo: one pass for the global amax, chunked
quantization of the rows with codes rounded against the *effective* (fp8-rounded) scale, the host shard rewritten
without the head, index and config updated, every indexed tensor verified resolvable. 13 s on a Spark. The profile
that motivated it (`bench/profile_step.py`) and every number's conditions are in the same repo.

## Attribution & license

- Base model: Qwen/Qwen3.8-Flash-Next (Alibaba) — **Qwen Community License 1.0** (included as `LICENSE`): modification,
  derivative works, distribution, hosting and commercial use permitted; products over 100M MAU / $20M monthly revenue
  must display the model name; a Model-as-a-Service or AI-assistant *business* on it needs a separate Qwen license.
- Routed-expert (and drafter-expert) NVFP4 tensors derived from Inferact/Qwen3.8-Flash-Next-NVFP4.
- GDN-tier and lm_head quantization, the NVFP4 table pipeline, the GPU-resident load path and the serving stack: myllmbox.
