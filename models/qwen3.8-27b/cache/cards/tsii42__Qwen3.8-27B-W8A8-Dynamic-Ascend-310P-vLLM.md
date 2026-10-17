---
library_name: transformers
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- qwen3.8-27b
- modelslim
- w8a8
- int8
- ascend
- huawei-ascend
- ascend-310p
- atlas-300i-duo
- vllm
- vllm-ascend
- tensor-parallel
- long-context
- vision-language
---

# Qwen3.8-27B W8A8 Dynamic — Huawei Ascend 310P / Atlas 300I Duo / vLLM-Ascend

**Search labels:** `Qwen3.8-27B`, `W8A8`, `INT8`, `ModelSlim`, `Huawei Ascend`, `Ascend 310P`, `Atlas 300I Duo`, `vLLM-Ascend`, `TP2`.

This repository contains a ModelSlim **W8A8_DYNAMIC** export of `Qwen/Qwen3.8-27B`, validated on one Huawei Atlas 300I Duo (2× Ascend 310P3) with tensor parallelism 2 and eager execution.

> [!NOTE]
> **Tool calling is functional and validated** — but requires the `VLLM_ASCEND_310P_GDN_WY=torch` environment gate and tool flags; see [Tool calling](#tool-calling) for the validated serving recipe and the measured results. Read [Known limitations](#known-limitations) for the remaining caveats.
>
> **Full deployment guide:** see [INSTALL.md](INSTALL.md) (prerequisites, building the patched runtime image from `runtime-patch/`, the validated serve command, verification probes, usage and troubleshooting notes).

## Updates

| Date | Change |
|---|---|
| 2026-09-26 | Long-context prefill/prefix-caching/decode benchmark (2k–192k, 77 runs) added: cold prefill 454→81 tok/s, prefix caching up to **~48× TTFT speedup at 192k**, decode 2–7 tok/s. See [Prefill and prefix-caching benchmark](#prefill-and-prefix-caching-benchmark). |
| 2026-09-23 | Prefix-caching measurement added (~3.5× on cached prompt prefix). Optional decode stack (MTP K=1 + `FULL_DECODE_ONLY` graphs + cpu binding) measured: **1.25× tokens/s @ 512 completion tokens**. `VLLM_ASCEND_310P_GDN_WY=torch` confirmed as the recommended WY path. See [Prefix caching and decode speedups](#prefix-caching-and-decode-speedups) and VALIDATION.md. |
| 2026-09-22 | Tool calling validated and enabled (GDN-WY torch gate, runtime-patch, six md5-verified files); acceptance suite 12/15 reproduced twice; 262k probe table incl. engine-wedge note; INSTALL.md + runtime-patch/ published. |

## Artifact

- Quantization: ModelSlim `W8A8_DYNAMIC`
- Weight shards: 9
- Weight size: approximately 33.92 GiB
- Base model: `Qwen/Qwen3.8-27B`
- Model family reported by config: `Qwen3_5ForConditionalGeneration`
- Weight tensors were not modified after download.
- This is not compressed-tensors W8A8 and not W8A8SC. Use `--quantization ascend`.

## Validated software and hardware

| Component | Validated value |
|---|---|
| Hardware | Huawei Atlas 300I Duo, 2× Ascend 310P3 (~44 GiB each) |
| CANN | 9.1.0 |
| HDK | 26.0.rc1 |
| torch_npu | 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.27.1rc-derived 310P image |
| Tensor parallelism | TP2 |
| Execution | eager |
| Runtime dtype | float16 |
| Quantization flag | `--quantization ascend` |
| Mamba cache dtype | float16 |

BF16, Triton and graph mode were not used in the validated Ascend setup.

**Tool calling: functional and validated** on this stack with the GDN-WY torch gate — see [Tool calling](#tool-calling).

## Start with vLLM-Ascend

Build the patched runtime image first (required — see [Tool calling](#tool-calling) for why):

```bash
cd runtime-patch
docker build -t vllm-ascend:310p-qwen38-toolfix -f Dockerfile .
```

Download this repository to `/models/Qwen3.8-27B-W8A8-Dynamic-Ascend-310P-vLLM` and run:

```bash
docker run --rm --name qwen38-w8a8-ascend310p \
  --shm-size=1g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -e VLLM_ASCEND_310P_GDN_WY=torch \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /models:/models \
  -p 8000:8000 \
  vllm-ascend:310p-qwen38-toolfix \
  vllm serve /models/Qwen3.8-27B-W8A8-Dynamic-Ascend-310P-vLLM \
    --served-model-name qwen3.8-w8a8 \
    --tensor-parallel-size 2 \
    --quantization ascend \
    --mamba-ssm-cache-dtype float16 \
    --dtype float16 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --enforce-eager --enable-chunked-prefill \
    --language-model-only \
    --no-enable-prefix-caching \
    --trust-remote-code
```

This is the fully validated acceptance configuration (32k context, no tool
parsing). For tool calling and/or 262k context, use the extended recipe in
[Tool calling](#tool-calling) or the complete [INSTALL.md](INSTALL.md) guide.
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models
```

Normal chat requests should set `chat_template_kwargs={"enable_thinking": false}` unless reasoning is intentionally requested.

## Tool calling

**Status: functional and validated** — with a required environment gate.

Stable multi-turn tool traffic requires routing the WY prefill through the torch reference path:

```bash
docker run -d --name vllm-qwen38-w8a8-tools \
  --shm-size=1g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -e VLLM_ASCEND_310P_GDN_WY=torch \
  -e VLLM_ASCEND_310P_GDN_FINITE_CHECK=1 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /models:/models \
  -p 8000:8000 \
  vllm-ascend:310p-qwen38-toolfix \
  vllm serve /models/Qwen3.8-27B-W8A8-Dynamic-Ascend-310P-vLLM \
    --served-model-name qwen3.8-w8a8 \
    --tensor-parallel-size 2 \
    --quantization ascend \
    --mamba-ssm-cache-dtype float16 \
    --dtype float16 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --enforce-eager --enable-chunked-prefill \
    --language-model-only \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

The gate image is built from this repository: `cd runtime-patch && docker build -t vllm-ascend:310p-qwen38-toolfix -f Dockerfile .` (public 310P base image + the six md5-verified patch files — GDN-WY torch gate, finite logging, W8A8 wiring). Full details: [INSTALL.md](INSTALL.md). Do not remove the gate — it is part of the validated recipe.

**Validation:**

| Metric | Result |
|---|---|
| Engine stability (tool traffic, incl. multi-turn assistant/tool history) | 0 crashes / 60 requests (previously: engine death after 1–2 tool requests) |
| Acceptance baseline (10 orchestrator-contract scenarios, 15 requests, temp 0 + seed 42) | **12/15 PASS, reproduced identically twice** |
| Large inputs with tool calls | 16.4k and 18.3k prompt tokens PASS (30.5k: valid envelope, wrong tool choice — see limitations) |
| Tool-call latency (small prompts, thinking off) | ~10–13 s end-to-end incl. 45 completion tokens |

All 10 contract tools of the target orchestrator harness were exercised (dispatch_task, record_result, request_review, apply_review, request_details, request_decision, resume_task, request_replan, workflow_status, finish_workflow-guardrail), including multi-turn tool-call history and guardrail scenarios (ack_timeout, role injection, stale replan).

## Throughput benchmark

Protocol: TP2/eager, random input length 32, output length 64, `--ignore-eos`, seed 42, 24 measured requests plus 3 warmups per arrival rate. All 144 measured requests completed and every cell passed semantic pre/post smokes. Request rate is offered arrival rate, not actual concurrency or a pure single-stream hardware limit.

| Offered req/s | Output tok/s | Completed req/s | Mean TTFT ms | p99 TTFT ms | Mean TPOT ms | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44.89 | 0.70 | 679.60 | 1144.85 | 170.95 | VALID |
| 2 | 63.22 | 0.99 | 1054.93 | 2070.96 | 208.45 | VALID |
| 4 | 83.94 | 1.31 | 1031.45 | 2048.04 | 198.50 | VALID |
| 8 | 96.83 | 1.51 | 1148.24 | 2067.31 | 195.28 | VALID |
| 16 | 104.31 | 1.63 | 1227.37 | 2121.26 | 194.39 | VALID |
| 32 | 111.65 | 1.74 | 1277.38 | 2057.45 | 187.77 | VALID |

Raw table: [`benchmark-results.tsv`](benchmark-results.tsv).

## Prefill and prefix-caching benchmark

A second benchmark measured **cold prefill**, **prefix-cached prefill** and **decode** across long context lengths (2026-09-24/25, same Atlas 300I Duo, TP2, eager, torch-WY gate, `--quantization ascend`, fp16, chunked prefill, `--enable-prefix-caching`, `--max-model-len 262144`). Deterministic seed-42 corpus (en-US/de code mix); context sizes {2k, 8k, 32k, 64k, 128k, 192k} × concurrency {1, 2} × thinking effort off/low/medium; 77 measured runs, cache reuse verified by a prefix-cache gate (84.4% own-hit rate) before measurement. "Cached" means the shared prefix (full base prompt) is served from the prefix cache and only a 2 048-token delta is prefilled; effective rate = (base+delta)/TTFT, speedup = TTFT_cold/TTFT_cached.

Single stream (c=1):

| Context | Cold prefill tok/s | Cold TTFT | Cached effective tok/s | Cached TTFT | Speedup | Decode tok/s (off/low/med) |
|---:|---:|---:|---:|---:|---:|---|
| 2k | 454 | 4.5 s | – (degenerate, no prefix) | – | – | 4.0 / 3.5 / 3.5 |
| 8k | 156* | 52.6 s | 701 | 11.7 s | 4.5× | 6.8 / 5.8 / 5.8 |
| 32k | 160 | 204.7 s | 1 951 | 16.8 s | 12× | 5.5 / 4.7 / 4.7 |
| 64k | 128 | 512.2 s | 2 814 | 23.3 s | 22× | 4.4 / 3.8 / 3.8 |
| 128k | 91 | 1 436 s | 3 536 | 37.1 s | 39× | 3.2 / 2.7 / 2.7 |
| 192k | 81 | 2 433 s | 3 921 | 50.1 s | ~48× | 2.5 / 2.1 / 2.1 |

*8k cold is a chunked-prefill variance outlier (reproduction runs: ~437 tok/s).

Key takeaways:

- **Cold prefill flattens with context length** (454 → 81 tok/s from 2k → 192k, −82%): KV/Mamba-state pressure makes long-context prefill compute- and bandwidth-bound on the 310P.
- **Prefix caching scales strongly with prefix length**: 4.5× at 8k up to ~48× at 192k. Repeated system/tool/document prefixes turn multi-minute cold prefills into seconds — at 192k, 2 433 s cold vs 50 s cached.
- **Decode is low (2–7 tok/s single stream)** and decreases with context length; thinking effort is nearly free (off ≈ low ≈ medium, ±10%).
- At client concurrency 2, cold prefill is ≈2× slower and per-stream decode drops to 1.5–5.3 tok/s (queueing/preemption overhead).

## Semantic smoke tests

Before and after every benchmark cell the model correctly returned:

- arithmetic: `2 + 2` → `4`
- capital of Germany → `Berlin`
- coherent German greeting

## Vision and long context

A separate baseline-GDN TP2/eager configuration loaded successfully with:

```text
--max-model-len 258048
--max-num-seqs 1
--enable-chunked-prefill
--no-enable-prefix-caching
```

Measured KV capacity was 261,101 tokens. A 200×200 image workload passed 8/8 requests:

| Client concurrency | TTFT p50 | E2E p50 | Correct |
|---:|---:|---:|---:|
| 1 | 0.53 s | 0.65 s | 4/4 |
| 2 | 1.19 s | 1.30 s | 4/4 |

With `max-num-seqs=1`, client concurrency 2 is queueing, not true parallel model execution.

## Reasoning measurements

| Client concurrency | Effort | TTFT p50 | E2E p50 | Correct |
|---:|---|---:|---:|---:|
| 1 | low | 0.49 s | 29.26 s | 4/4 |
| 1 | medium | 0.47 s | 24.42 s | 4/4 |
| 2 | low | 44.29 s | 108.40 s | 4/4 |
| 2 | medium | 52.69 s | 90.11 s | 4/4 |

Again, C2 used a server configured with `max-num-seqs=1` and therefore includes queueing.

## Prefix caching and decode speedups

### Prefix caching (measured)

With `--enable-prefix-caching`, repeated system/tool prompt prefixes are served from cache instead of being re-prefilled. Measured on the 64k standing recipe (TP2, eager, torch-WY): a ~2.2k-token shared prefix costs **4.2 s cold vs 1.2 s warm (~3.5× on the cached prefix portion)**; savings scale linearly with prefix overlap. The 64k configuration (`--max-model-len 65536 --enable-prefix-caching`) was smoke-validated (5/5 incl. exact tool-call args) repeatedly.

### Optional decode stack: MTP speculative decoding + graphs + cpu binding

The checkpoint contains one MTP layer (`mtp_num_hidden_layers: 1`, weights included), and vllm-ascend supports `qwen3_5_mtp` speculative decoding on 310P with `num_speculative_tokens: 1`. Measured (2026-09-23, temperature 0, one run per cell; both stacks passed tool-calling smoke, 5/5 with exact args):

| Stack | 45 tok: tok/s (TPOT) | 512 tok: tok/s (TPOT) |
|---|---|---|
| Baseline (eager, torch-WY) — default | 8.22 (121.7 ms) | 8.58 (116.5 ms) |
| `--enable_cpu_binding` + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` + `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'` | 9.97 (100.3 ms) | **10.70 (93.4 ms) = 1.25×** |

MTP mean acceptance length: **1.67–1.77** (42–49 accepted of 63–64 drafted). This stack is **optional** and intended for decode-heavy requests (≥ ~512 completion tokens); the validated default remains eager + torch-WY. Use model-runner v1 (default on 310P) and keep `num_speculative_tokens` at 1 (DUO recommendation). Full measurement protocol and evidence pointers: VALIDATION.md.

## Known limitations

**Tool calling: functional and validated via the GDN-WY torch gate (see [Tool calling](#tool-calling)).** With `VLLM_ASCEND_310P_GDN_WY=torch`, multi-turn tool traffic is validated stable (0 crashes / 60 requests, 12/15 acceptance cases reproduced twice). Remaining, deterministic **model-quality** notes (no engine defect; envelopes stay complete and parseable):

- **No stop after correct tool call (guardrail scenarios):** in specific guardrail turns (final workflow completion, ack-timeout handling) the model can emit the correct tool call and then continue generating empty output without EOS until the token budget is exhausted (`finish_reason=length`). The delivered `tool_calls` payload is correct and usable; apply a timeout and consume the delivered call.
- **Long-context tool drift:** at ~30k prompt tokens the model occasionally picks `workflow_status` instead of the requested `dispatch_task` (valid envelope, wrong tool). A guardrail or retry is recommended for long-context tool routing.
- **Sampling sensitivity:** at temperature 0.6 results are not reproducible (failures move between runs); acceptance was run at temperature 0 + seed 42.

Further limitations:

- **Keep the `VLLM_ASCEND_310P_GDN_WY=torch` gate** — other `VLLM_ASCEND_310P_GDN_WY` values are not validated for tool workloads; the gate is part of the validated serving recipe.
- **Rare engine wedge under mixed long-context load:** observed once (262k config, ~96k-token prompt with concurrent requests) — engine stats went silent while `/health` still returned 200 and new requests timed out; no error in the log. Recovery: container restart (validated in ~70 s). See INSTALL.md, Troubleshooting.
- **262k context validation is partial:** text-only serving at `max-model-len 262144` is validated for chat, single and multi-turn tool calls, and 32k/64k-token prompts with tool calls; prompts beyond 262,144 tokens are rejected with HTTP 400 before scheduling. The ~96k-token probe ended inconclusive (engine wedge above). The earlier 258048 vision-encoder configuration remains the most complete long-context data point.
- **Prefix caching** at long context remains unvalidated (serving uses `--no-enable-prefix-caching`).
- Performance numbers describe the complete serving stack and are not architecture-independent hardware specifications.
- `prompt_tokens + max_tokens` must not exceed `max-model-len`.

## Provenance and license

The base model is attributed to Qwen. This repository redistributes a quantized derivative under Apache-2.0 metadata and includes the license text. The original local artifact's model card described the quantized export as MIT; Apache-2.0 base-model attribution and notices are retained here. No endorsement by Qwen, Huawei, ModelSlim or vLLM-Ascend is implied.

See [`VALIDATION.md`](VALIDATION.md) for evidence scope and exact caveats.
