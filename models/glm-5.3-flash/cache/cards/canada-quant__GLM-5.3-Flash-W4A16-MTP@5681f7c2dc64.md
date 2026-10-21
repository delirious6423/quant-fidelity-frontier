---
language:
  - en
  - zh
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
  - glm-5.3
  - glm5_next
  - quantized
  - w4a16
  - int4
  - gptq
  - compressed-tensors
  - mtp
  - speculative-decoding
  - dflash2
  - vision
  - multimodal
  - vllm
  - dgx-spark
  - 1m-context
  - blackwell
  - hopper
---

# GLM-5.3-Flash — W4A16 (INT4) + BF16 MTP

## Model description

INT4 weight-only quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with the BF16 MTP draft head kept for speculative decoding. Only the 36,288 routed-expert GEMMs are INT4 (GPTQ, symmetric, group-size 128); attention, router, shared experts, embeddings, the vision tower and the MTP head stay in BF16. Not a new model — all capability comes from the base model.

| | |
|---|---|
| **Size** | **177.7 GiB** (BF16 ≈ 599 GiB, −70%) |
| **Runs on** | NVIDIA H100, H200, RTX PRO 6000 and DGX Spark (GB10) — validated GPU counts and context per configuration in [Hardware & context limits](#hardware--context-limits) |
| **Context** | full **1,048,576** tokens on 2× H200, and on 2× DGX Spark with the incoai reference drafter (with our DFlash2-G drafter: 262K default, 800K validated); 262K–512K on the 4/8-GPU x86 configurations (KV-memory-bound) |
| **Quality** | AIME 2025 0.8833 (n=120, H100) · AIME 2026 85.0% (n=120, 2× DGX Spark) · GPQA-Diamond 0.8586 · GSM8K 0.97 · vision: MMMU 0.747 · OCRBench 888 |
| **Throughput** | 8× H100: 181.2 tok/s single-stream (TP8 + DFlash2-G K=7) and 2,418 tok/s aggregate @c256 (TP8, no speculation) · 4× H200 TP=4: 195 → 1,954 tok/s from c1 to c256 · 2× DGX Spark: ≈31–33 tok/s single-stream at short context |

Full benchmark grids, protocols and research notes: [BENCHMARKS.md](BENCHMARKS.md).

## Uses & recommended recipes

### Quick start

```bash
# 1. Download (~178 GiB)
huggingface-cli download canada-quant/GLM-5.3-Flash-W4A16-MTP --local-dir /models/glm53-flash-w4a16-mtp

# 2. Serve on 4× H100 / H200 (other hardware: see Serving recipes)
docker run --gpus '"device=0,1,2,3"' --ipc=host --network=host --rm \
  -v /models:/models vllm/vllm-openai:glm53-flash-x86_64-cu130 \
  vllm serve /models/glm53-flash-w4a16-mtp --served-model-name glm53-w4 \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --max-model-len 262144 --max-num-seqs 512 \
    --gpu-memory-utilization 0.92 --no-enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --trust-remote-code --port 8000

# 3. Call it (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm53-w4",
  "messages": [{"role": "user", "content": "Prove there are infinitely many primes."}]
}'
```

> **Two things that bite.** If your `config.json` predates 2026-09-08, re-download it — older copies fail in vLLM with `KeyError: 'layers.0.mlp.gate_up_proj.weight'` (weights are unchanged). And always pass `--max-num-seqs ≤ 512` — the vLLM default of 1024 exceeds this hybrid linear-attention model's 512 Mamba/KDA-state cache blocks.

### Serving recipes

| Architecture | Image |
|---|---|
| SM90 (H100 / H200) | `vllm/vllm-openai:glm53-flash-x86_64-cu130` (validated). Upstream `vllm/vllm-openai:nightly-x86_64` ≥ 2026-09-08 also boots this checkpoint ([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)); pass `--attention-backend FLASH_ATTN_MLA_SPARSE` there, its default backend faults on ≥131K prompts. |
| SM90, 8× H100 + DFlash2-G drafter | `ghcr.io/canada-quant/vllm-glm53-flash-h100:v2-w4a16-dflash2` — vLLM nightly pin + baked drafter patches (EAGLE3 aux taps, drafter-aware KV partitioning); image of record for the 2026-09-28 8×H100 sweep |
| SM120 (RTX PRO 6000) | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` |
| SM121 (DGX Spark) | `ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e`, built from [canada-quant/vllm-glm53-flash-sm121](https://github.com/canada-quant/vllm-glm53-flash-sm121) with the two SM121 serving patches baked in |

All configurations use expert parallelism. fp8 KV is not available on Hopper for this NoPE model.

**H100 / H200, TP=4 (pinned image)** — the Quick start command is the benchmarked recipe. `num_speculative_tokens: 2` is the sweet spot on Hopper (52–55% acceptance; N=5 collapses acceptance to ~30%). Keep prefix caching off — it measured −2…−5% on H200. On 8× H200, two independent TP=4 replicas behind a load balancer beat one TP=8 endpoint by +28–38% aggregate at c128–c512; TP=8 wins single-stream and holds one 7.8M-token pool. On 2× H200 (89.5 GiB weights per GPU) add `--tensor-parallel-size 2 --max-num-seqs 128 --max-cudagraph-capture-size 128` for 262K, or `--max-model-len 1048576 --max-num-seqs 16 --gpu-memory-utilization 0.95 --max-cudagraph-capture-size 64 --max-num-batched-tokens 4096` for 1M.

**8× H100, TP=8 (v2 image, bench-validated 2026-09-28)** — 8192/1024 prompts. The **latency recipe is TP=8 + DFlash2-G K=7**: **181.2 tok/s c1**, +34% over TP8 without speculation (135.4). The **aggregate recipe is TP=8 without speculation**: **2,418 tok/s @c256**; MTP N=2 (2,330 @c256) is the balanced middle and has the best grid c1 (215.0). K=4 beats K=7 on aggregate at every concurrency. Avoid TP=4 with the DFlash2 drafter past c32: its 8 full-attention KV tensors cut the KV pool ~6× vs TP8 (311,999 vs 1,811,949 tokens) and force a structural preemption/recompute TTFT cliff. Uncontended TP=4 with MTP N=2 is the best single-stream shape measured on this rig (c1 **193.18**, 1,292 @c256). With any drafter, cap `--max-model-len ≤ 131071`: the speculative path faults on 131K-token prefills (without speculation they run). Image `ghcr.io/canada-quant/vllm-glm53-flash-h100:v2-w4a16-dflash2`, cold boot ≈ 12 min. Full grids: [BENCHMARKS.md](BENCHMARKS.md#h100-sm90--final-closeout-2026-09-28).

**RTX PRO 6000, TP=4** — same command with the SM120 image and `--max-num-seqs 64 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching`. fp8 KV is required at 262K on 96 GB cards. Keep MTP on at every concurrency here: it adds +70% at c1 and +48% at c32.

**2× DGX Spark, TP=2** — prebuilt image and one-command launcher in [canada-quant/vllm-glm53-flash-sm121](https://github.com/canada-quant/vllm-glm53-flash-sm121); the launcher also ships in the drafter repo. Drafter: [canada-quant/GLM-5.3-Flash-DFlash2-G](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G), our self-trained DFlash2 drafter (Apache-2.0; 3.676 mean acceptance at K=7 on our 500-prompt holdout vs 3.632 for the incoai reference on the same hardware; drop-in successor of [-F](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-F) and [-E](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-E)). Start the worker rank first, wait 25 s, then the head rank.

```bash
# on both nodes, rank1 (worker) first, then rank0 (head) 25 s later
# 262K context (production config): 8 GiB fp8 KV -> 366,749-token pool (the launcher default since 2026-10-04;
# older launcher copies defaulted to 3 GiB, which is too small for DFlash2-E/-F/-G, so passing it explicitly is safe)
MAX_MODEL_LEN=262144 KV_CACHE_MEM=8053063680 EAGER=0 GRAPHS=1 bash launch_dflash2_tp2.sh <rank>
# long context with DFlash2-G (validated 2026-09-26/27): 800K, 16 GiB KV pin -> 888,729-token pool
MAX_MODEL_LEN=800000 KV_CACHE_MEM=17179869184 GMU=0.90 EAGER=0 GRAPHS=1 bash launch_dflash2_tp2.sh <rank>
```

- **Drafter memory (corrected 2026-10-04).** DFlash2-E/-F/-G have 8 full-attention layers that each cache the whole context, about **3× the KV per token** of the incoai reference drafter (2,048-token sliding window). Measured pools on this pair: 366,749 tokens at 8 GiB (262K) and 888,729 at 16 GiB (800K) with G, vs 1,360,420 at 9 GiB (1M) with the incoai drafter. **1M on 2× Spark is validated with the incoai drafter only**; with G, 9 GiB holds ≈450K tokens. Earlier versions of this card paired G with 1M at 9 GiB.
- **Drafter or built-in MTP head?** DFlash2-G suits short-context English and code. For long contexts, non-English text or memory headroom, use the checkpoint's own BF16 MTP head (`{"method":"mtp","num_speculative_tokens":3}`). Our G recipe slows at depth: 31.4 / 30.7 / 15.3 / 10.0 tok/s at 0 / 4K / 65K / 100K (llama-benchy pp2048/tg128). A community user reported 31.6 / 29.8 / 25.4 / 31.0 tok/s at the same depths with this checkpoint on eugr's b12x build + MTP (1700 MHz GPU clock, different harness; not measured by us), and G 20–30% slower than MTP on non-English text. MTP on our SM121 image is not yet validated by us; a same-pair comparison started 2026-10-04.
- **Notes:** 7 speculative tokens is the drafter's trained and measured setting (k=5 also boots, not benchmarked). Confirm the boot log shows the mask-embedding load (`mask_token_id 154856`); keep single prompts ≤ ~310K tokens; stop with `docker stop -t 30`, never `rm -f`. Cold boot is 6–10 minutes.

**2× DGX Spark on eugr's b12x build (community-validated, not one of our validated recipes)** — this checkpoint needs a fused-attention ignore-list fix there ([eugr/spark-vllm-docker#403](https://github.com/eugr/spark-vllm-docker/issues/403)): bind-mount a `config.json` copy whose `quantization_config.ignore` adds `re:.*self_attn\..*`, `re:.*\.in_proj_qkvgfab.*` and `re:.*in_proj.*` (safe: nothing under `self_attn` is quantized), and pass `--block-size 256 --moe-backend marlin --attention-backend B12X --linear-backend b12x --kv-cache-dtype fp8` with `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"B12X"}'`. We have not tested our DFlash2 drafters on that build (its b12x attention lacks a full-window non-causal path, and the drafter cache stays BF16).

## Quality

| Benchmark | Hardware | W4A16 (this) | Notes |
|---|---|---|---|
| **AIME 2025** — n=120, max thinking, 131,072-token budget | H100 | **0.8833** | — |
| AIME 2025 — same protocol | RTX PRO 6000 | 0.8083 raw · 0.8833 with a budget-commit fix | see below |
| **AIME 2026** — n=120, max thinking, 131,072-token budget | 2× DGX Spark | **85.0%** (102/120) | EXL3 (MiaAI-Lab) 80.0% (96/120), matched protocol |
| **GSM8K** | H100, RTX PRO 6000 | 0.970–0.975 | — |
| **GPQA-Diamond** — n=198 @131K | H100 · RTX PRO 6000 | 0.8586 · 0.8586 | — |
| **MMMU** (validation) — n=900, lmms-eval 0.7.3 task spec | 8× H100 TP=8, greedy | **0.74667** (672/900) | MC 0.766 · open 0.434 |
| **OCRBench** — n=1000, lmms-eval 0.7.3 task spec | 8× H100 TP=8, greedy | **888** /1000 | weakest: handwritten math 56/100; strongest: doc-VQA 192/200 |

On RTX PRO 6000 the raw AIME 2025 score trails the H100 result: ≈63% of the gap is a budget wall (11.7–14.2% empty answers) and ≈37% is SM120 kernel numerics. A zero-cost commit hook closes it but is not part of the published recipes. The vision tower is BF16 passthrough and was not covered by the text-only calibration, so MMMU and OCRBench are this artifact's own vision baseline (greedy, seed 42, measured 2026-09-28). Details: [BENCHMARKS.md](BENCHMARKS.md#quality--full-results).

## Benchmarks summary

Output tok/s, thinking ON, measured by the authors. Competitor comparisons against NVFP4 checkpoints were removed on 2026-10-04: earlier tables mixed two different NVFP4 checkpoints under one label and compared against NVIDIA's without its MTP head. A clean same-rig comparison will be published when it's done.

| Hardware & configuration | Output tok/s |
|---|---|
| 8× H100, TP8 + DFlash2-G K=7 (latency recipe), 8192/1024 | c1 **181.2**, +34% over TP8 without speculation (135.4) |
| 8× H100, TP8 without speculation (aggregate recipe) | c256 **2,418** |
| 8× H100, TP8, MTP N=2 | c1 215.0 · c256 2,330 |
| 8× H100, TP4 uncontended, MTP N=2 | c1 **193.2**, the best single-stream shape on this rig · c256 1,292 |
| 4× H200, TP=4, MTP N=2 | c1 195 · c8 698 · c32 1,258 · c64 1,529 · c128 1,789 · c256 1,954 (8× H200 TP=8: 217 → 2,681; two TP=4 replicas: 391 → 3,911 aggregate) |
| 4× RTX PRO 6000, TP=4, MTP N=2 | c1 109.9 · c8 318.5 · c32 534.4 |
| 2× DGX Spark, incoai DFlash2 drafter, 1M serve, 8K/256 (2026-09-03) | c1 33.0 · c2 35.6 · c4 59.6 · c6 67.1. Against EXL3 under a matched protocol: +10.4% c1 and +31–39% long-prefill; EXL3 leads at c2 (+67%) and c4 (+89%) |
| 2× DGX Spark, DFlash2-G K=7, 800K context, llama-benchy pp2048/tg128 (2026-09-27/28) | single-stream 31.4 / 30.7 / 15.3 / 10.0 at depth 0 / 4K / 65K / 100K |

Single-stream decode is insensitive to KV length up to ≥486K on RTX PRO 6000; at batch, long-KV decode plateaus at ~2–4 tok/s per stream and long prefills serialize at a ~6–8.5K tok/s aggregate ceiling. All grids and protocols: [BENCHMARKS.md](BENCHMARKS.md#throughput).

## Hardware & context limits

Each row is the largest context serving-validated on that configuration.

| Configuration | GPUs | Validated context | Stack |
|---|---|---|---|
| DGX Spark GB10 (SM121), TP=2 | 2× 128 GB UMA | **1M** — KV pool 1,360,420 tokens (1.30× a full 1M request), 2026-08-31 | incoai DFlash2 drafter, fp8 KV, 9 GiB KV |
| DGX Spark GB10 (SM121), TP=2 + DFlash2-G | 2× 128 GB UMA | **800K** — KV pool 888,729 tokens at a 16 GiB KV pin (2026-09-26); default 262K = 366,749 at 8 GiB. The drafter's full-attention cache costs ~3× the KV per token, so 1M does not fit | DFlash2-G K=7, fp8 KV |
| RTX PRO 6000 (SM120), TP=4 | 4× 96 GB | **512K** (486K prompts measured) | MTP N=2, fp8 KV |
| H100 (SM90), TP=4 | 4× 80 GB | **262K** (256K prompts measured) | MTP N=2, bf16 KV |
| H100 (SM90), TP=8 + DFlash2-G | 8× 80 GB | **131,071 with any drafter** — 128K-token prefills die on the speculative path (spec-off survives 131,072); with MTP-N2 the 262K row above applies | v2 image, K=7 latency / spec-off aggregate |
| H200 (SM90), TP=4 | 4× 141 GB | **262K** — KV pool 6.19M tokens (≈23 concurrent 262K requests) | MTP N=2, bf16 KV |
| H200 (SM90), TP=8 | 8× 141 GB | **262K** — KV pool 7.79M tokens | MTP N=2, bf16 KV |
| H200 (SM90), TP=2 | 2× 141 GB | **1M** — KV pool 2.84M tokens (929K-token prompt measured) | MTP N=2, bf16 KV |

### Known issues

- **`config.json` (2026-09-08):** vLLM matches `quantization_config.ignore` against its own fused module names, so the ignore list now carries both the HF and vLLM spellings plus `re:.*\.layers\.45\..*` for the MTP head. Older 765-entry copies fail at load. Weights unchanged.
- **DFlash2 admission wedge** (SM90 research stack only, MTP recipes unaffected): with the DFlash2 drafter at block size 2304, prompts above ~15.5K tokens are never admitted. A fix was validated to 256K prompts; block size 1536 avoids it. Filed as [vllm-project/vllm#55800](https://github.com/vllm-project/vllm/issues/55800).
- **Marlin no-split-K path on SM121:** deterministic illegal memory access at M=256 when forcing `split_k=1`; the stock heuristic used in serving is clean. Filed as [vllm-project/vllm#56064](https://github.com/vllm-project/vllm/issues/56064).

### Quantization details

| Field | Value |
|---|---|
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`) — 45 decoder layers + MTP layer 45, 288 routed experts (top-8) + 1 shared, KDA + DSA attention, 24-block vision tower |
| Quantized | **36,288 tensors** = 42 MoE layers × 288 experts × 3 GEMMs — W4A16, INT4, symmetric, group 128, GPTQ, compressed-tensors `pack-quantized` |
| Kept in BF16 | attention (incl. DSA indexer), dense prefix layers 0–2, shared experts, router, mHC tensors, embeddings, lm_head, norms, vision tower (348 keys), MTP layer 45 (889 keys) |
| Kept in FP32 | `A_log`, `dt_bias`, `e_score_correction_bias`, `hc_*` — verbatim from source |
| Calibration | 256 samples × 4096 tokens, in-distribution chat/code mix, sequential per-layer GPTQ |
| Built on | 8× NVIDIA B300, 2026-08-27 |

Build gates, all passing: exactly 36,288 packed tensors and nothing quantized outside routed experts; vision key set 348/348 identical to source; MTP layer present; zero dtype drift vs source; no collapsed expert scales. Loads with transformers ≥ 5.16; text generation and image captioning smoke tests pass.

## Citation & license

```bibtex
@misc{canada_quant_glm53_flash_w4a16_mtp,
  title  = {GLM-5.3-Flash W4A16 (INT4) + BF16 MTP},
  author = {canada-quant},
  year   = {2026},
  url    = {https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP}
}
```

MIT, inherited from the base model. Follow the base model's usage terms.

---

*Built, benchmarked and documented with the [Digby.ai](https://digby.ai) coding harness, developed by [CQL.ca](https://cql.ca).*
