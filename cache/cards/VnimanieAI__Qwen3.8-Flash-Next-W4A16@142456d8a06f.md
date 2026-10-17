---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen4_exp
  - compressed-tensors
  - w4a16
  - int4
  - 4-bit
  - vllm
  - ampere
  - rtx-3090
  - conversational
---

# Qwen3.8-Flash-Next-W4A16

INT4 (W4A16, group-128, symmetric) quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in `compressed-tensors` pack-quantized format, targeted at **consumer Ampere GPUs (4×RTX 3090 — 8×RTX 3090)**, where the official FP8 checkpoint cannot run at all (no FP8 support on SM86).

To our knowledge this is the first publicly available **INT4** quantization of this model (checked against all Flash-Next repositories on the Hub, 2026-08-26), and the only 4-bit checkpoint that runs on **pre-Blackwell** GPUs: the NVFP4 variants (ModelOpt, FP4) that exist require SM100+, while this one targets the vast installed base of Ampere cards via Marlin kernels.

- **335 GB (BF16) → 168 GB on disk**, of which only ~66 GB are GPU-resident weights: the 102 GB n-gram (PLE) tables stay in host RAM via vLLM PLE offload.
- Runs on 4×24 GB with 96k context, or 8×24 GB with the full native 262k context.
- Quality on our greedy eval suite matches the base-family reference (see Benchmarks).
- MTP speculative-decoding head preserved in BF16 — speculative decoding works (+57% single-stream decode).

> **Which quant should you take?** If you have more than 4×24 GB — e.g. 8×3090 — take
> [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
> instead of this one. In our head-to-head on the same 8×RTX 3090 vLLM stack (same questions,
> sampling and budgets) that calibrated AWQ reasons ~3× more economically and scores higher than
> this checkpoint (measured numbers in Benchmarks below). Its weights are larger (~23 GiB/GPU at
> TP4), so it does not fit 4×24 GB — **this quant remains the option for 4×3090 deployments.**

## What was quantized (and what deliberately was not)

Only GEMM-heavy weights are quantized to INT4 group-128 symmetric:

| Component | Precision | Rationale |
|---|---|---|
| Routed experts (512/layer, fused `gate_up_proj`/`down_proj`, unpacked to per-expert tensors) | **INT4 g128** | 120B of the 125B backbone; the entire point of the exercise |
| Attention `q/k/v/o_proj` (QSA layers) | **INT4 g128** | Standard, well-conditioned GEMMs |
| Shared expert (all 48 layers) | **BF16** | Its intermediate dim 640 is TP-sharded to 640/TP columns, which no Marlin group size divides at TP=8 (see "The 640/128 problem" below). ~470 MB total — not worth the risk |
| MTP module (`mtp.*`) | **BF16** | Speculative decoding head; quantizing draft heads measurably hurts acceptance; losing it disables MTP entirely |
| PLE n-gram tables (`*.ple.*`, 51B params) | **BF16** | Lookup tables with random access patterns, not GEMMs; offloaded to host RAM at inference |
| MoE router (`*.gate`) | **BF16** | A wrong expert choice costs more than the bytes saved; the authors themselves single out the router as fragile (they exclude it from Muon in training) |
| QSA indexer (`*indexer*`) | **BF16** | Selects which context blocks attention sees; quantization noise here degrades long-context retrieval |
| GDN / linear attention (`*.linear_attn.*`) | **BF16** | Recurrent state dynamics; follows the reference practice of the RedHatAI Qwen3-Next W4A16 quant |
| Embeddings, `lm_head`, gated-residual (`hyper_connection`) branches, vision encoder | **BF16** | Small and sensitive |

Method: RTN (round-to-nearest), symmetric, per-group scales along the input axis, no calibration data. AWQ was attempted first but the 335 GB BF16 model cannot be linearized by llm-compressor within 256 GB host RAM (meta-tensor offload breaks expert linearization). RTN quality turned out to match the family reference on our eval suite, so AWQ was deferred.

## The 640/128 problem — why Expert Parallelism is mandatory

The experts of this architecture have `moe_intermediate_size = 640`. With group-128 quantization (ours, and equally the official FP8 checkpoint's block-128), tensor parallelism shards the expert `down_proj` input dimension into 640/TP columns: 160 at TP=4, 80 at TP=8 — neither divisible by 128. The upstream vLLM recipe states this for the FP8 checkpoint verbatim: *"plain TP8 is incompatible with its 128-wide quantization blocks"* and prescribes TEP8 (TP8 + expert parallelism).

The same applies to this checkpoint: **always launch with `--enable-expert-parallel`**. With EP, experts are distributed whole (no intra-expert sharding), and the group math is exact. The shared expert is not covered by EP (it is always TP-sharded), which is why it ships in BF16 here.

## Deployment (vLLM)

Tested with the `qwen4_exp`-capable vLLM dev build (`vllm/vllm-openai:qwen38-flash-next` vendor image). Requirements beyond the usual:

- `VLLM_PLE_CPU_OFFLOAD=1` and **≥110 GB free host RAM** for the PLE tables;
- `--cap-add SYS_PTRACE --security-opt seccomp=unconfined` — the PLE offload worker uses the `pidfd_getfd` syscall, which Docker's default seccomp profile blocks;
- **do not let torch.compile run**: inductor compilation of this architecture hangs indefinitely on Ampere (all workers spin in `static_triton_launcher.load_kernel`; the QSA index selection has data-dependent control flow that upstream also splits out of compiled regions). Use `--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'` — decode CUDA graphs without inductor. This is the difference between 9 tok/s and 105 tok/s;
- `--enable-expert-parallel` (see above).

8×RTX 3090, full native context:

```bash
docker run -d --name flashnext --gpus all --shm-size 32g \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/model:/model -p 8000:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model --served-model-name Qwen3.8-Flash-Next \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --gpu-memory-utilization 0.90 --max-model-len 262144 --max-num-seqs 64 \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --trust-remote-code
```

4×RTX 3090: same, with `--tensor-parallel-size 4 --gpu-memory-utilization 0.92 --max-model-len 98304` and MTP off (weights + MTP + long context do not fit together in 4×24 GB; pick two).

KV-cache note: only 12 of 48 layers carry KV (hybrid GDN architecture) but the model has 2 KV heads, so per-GPU KV cost does **not** shrink from TP=4 to TP=8 (heads replicate below 1 head/rank). Measured: ~14.7 KB/token/GPU; a 8.2 GB KV pool holds ~600k tokens ≈ 2.3 concurrent full-context sessions.

## Benchmarks

All measurements on 8×RTX 3090 (PCIe, no NVLink), this checkpoint, vLLM dev build, unless noted. Reference: the previous production model of the same family (Qwen3.8-27B INT8 W8A16, SGLang, MTP on, same host, 4×3090).

**Quality** — 50-task deterministic greedy suite (arithmetic, sorting, positional retrieval, JSON extraction by schema):

| Model | Total | arith | json | nth | sort |
|---|---|---|---|---|---|
| This quant, TP8+EP | **48/50** | 20/20 | 10/10 | 9/10 | 9/10 |
| This quant, TP4+EP | 47/50 | 20/20 | 9/10 | 9/10 | 9/10 |
| Qwen3.8-27B INT8 (reference) | 48/50 | 20/20 | 8/10 | 10/10 | 10/10 |

**Speed** (single stream, 500-token generations, temperature 0.7):

| Metric | This quant (8×3090) | 27B INT8 ref (4×3090) |
|---|---|---|
| Decode, prose | 100–107 tok/s | 92–116 tok/s |
| Decode, code | 108–114 tok/s | — |
| Decode w/o MTP | 67 tok/s | — |
| TTFT (short prompt) | 0.27–0.32 s | 0.13–0.17 s |
| Aggregate, 8 streams | 395 tok/s | 332 tok/s |
| Aggregate, 16 streams | 526 tok/s | 326 tok/s |
| Prefill 32k / 90k / 200k | 11.7 s / 33 s / 81 s | 60k in 144 s |
| Decode at 200k context | 97.5 tok/s (no degradation) | 63 tok/s at 250k |

MTP acceptance measured at ~2.25 tokens/step (depth 4, sampled at temp 0.7); the tech report's own figure is 4.06 under greedy full-precision conditions — expect higher acceptance for greedy workloads.

**Constrained decoding**: `response_format: {type: "json_schema"}` and `structured_outputs: {"json": ...}` both verified working (xgrammar).

### Capability benchmarks (vs full-precision references)

All measured on this exact checkpoint served by vLLM as above (TP8 + expert parallel, MTP-4, 8×RTX 3090). Thinking/reasoning **enabled** unless noted. Sampling per base model card recommendation: `temperature=1.0, top_p=0.95, top_k=20`. DeepSeek V4 Flash 0731 (hosted, reasoning on) serves as an external reference under the identical harness, prompts, scoring and budgets.

#### Agentic: τ²-Bench Airline (50 tasks, 1 trial, shared user simulator)

| Model | pass^1 | prompt tokens / task |
|---|---|---|
| **This quant (INT4 W4A16, self-hosted)** | **0.740** | 93,339 |
| DeepSeek V4 Flash 0731 (hosted) | 0.700 | 187,835 |
| qwen3.8-flash (hosted, OpenRouter) | 0.680 | 150,778 |

Differences of 4–6 pp on a single trial are within noise — the honest claim is
**no agentic degradation observed vs the hosted endpoint of the same family**
(with the usual caveat that a hosted endpoint's serving precision cannot be
independently verified), at ~2× better context economy than DeepSeek V4 Flash
on the same tasks.

#### Knowledge: MMLU_RU (NLPCoreTeam/mmlu_ru, 1000-question paired subsample, seed 1234)

| Model | strict | budget-corrected* |
|---|---|---|
| **This quant** | 0.815 ± 0.012 | **0.877 ± 0.010** |
| DeepSeek V4 Flash 0731 | 0.837 ± 0.012 | 0.861 ± 0.011 |

\* Budget-corrected: questions where the reasoning phase exhausted an 8192-token
generation budget were re-asked with 32768; strict scoring penalizes verbose
reasoning, not knowledge. Reasoning adds ≈ +11 pp over non-thinking mode.
Corrected is still a lower bound: the few questions truncated even at the retry
budget (5/98 here, 2/32 for DeepSeek) remain counted as errors.

#### Expert QA: GPQA Diamond (198 questions, zero-shot CoT, deterministic choice shuffle)

| Model | strict @32k | budget-corrected @131k* |
|---|---|---|
| **This quant** | 0.601 ± 0.035 | **0.874 ± 0.024** |
| DeepSeek V4 Flash 0731 | 0.854 ± 0.025 | 0.894 ± 0.022 |

\* Budget-corrected: only questions truncated at the 32,768-token budget were re-asked
with 131,072; the denominator stays 198 — unrescued truncations still count as errors.
Same prompt, sampling, parser and choice shuffle in both passes, and both models
accepted the full 131,072-token retry budget.

The strict/corrected gap differs sharply between the models because this model
reasons much longer on expert questions: 37.9% of answers hit the 32k budget
(vs 8.1% for DeepSeek), concentrated in Chemistry. Strict measures budget
discipline as much as knowledge; corrected is the knowledge number, and it is
still a lower bound (questions truncated even at 131k count as errors: 1 here,
4 for DeepSeek).

#### Head-to-head vs the calibrated AWQ quant (same rig)

Paired probe against [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
on the identical 8×RTX 3090 vLLM stack and launch recipe (only `--kv-cache-memory`
lowered to 7 GB for the AWQ — its weights are ~3.4 GiB/GPU larger). Same 40 GPQA
Diamond questions (the 20 that truncated at 32k in our full run + 20 that didn't),
same sampling, `max_tokens=32768`, length measured by `usage.completion_tokens`:

| Metric (40 paired questions) | This quant | cyankiwi AWQ |
|---|---|---|
| Median completion tokens (non-truncating 20) | 9,812 | **3,391** |
| Truncations @32k on the hard 20 | 16/20 | **10/20** |
| Correct on the hard 20 | 4 | **10** |
| Correct overall | 19/40 | **26/40** |

The AWQ reasons ~3× more economically and scores higher. The difference is the
quantization recipe — calibrated AWQ g32 asymmetric with attention left in BF16,
versus this checkpoint's data-free RTN g128 symmetric with attention quantized —
i.e. a large part of this quant's reasoning verbosity is a **quantization
artifact**, not a base-model trait. Hence the recommendation at the top: on
8×24 GB take the AWQ; this quant remains the one that fits 4×24 GB.

## Decision & test registry

Engineering log of every non-obvious decision in this release, with the evidence behind it:

1. **INT4 over FP8/GPTQ** — FP8 does not exist on SM86 (SGLang refuses to load, vLLM has an open blockwise-FP8 regression); GPTQ had recurring issues on Ampere in our fleet, and MoE INT4 Marlin on SM86 was verified working empirically (Qwen3-30B-A3B GPTQ test, 172 tok/s) before committing.
2. **RTN over AWQ** — forced by host RAM (335 GB > 251 GB, llm-compressor meta-tensor failure); validated post-hoc by quality parity on the eval suite.
3. **Fused experts unpacked to `experts.{i}.{gate,up,down}_proj`** — matches vLLM's `packed_modules_mapping` expectations; layout cross-checked against RedHatAI/Qwen3-Next-80B-A3B W4A16.
4. **`quantization_config.ignore` uses vLLM module names, not checkpoint tensor names** — `model.visual.*` in the checkpoint is `visual.*` as a vLLM module; getting this wrong makes vLLM try to quantize the vision tower and fail on a 288-wide layer.
5. **Shared expert de-quantized post-hoc** (surgery on the emitted checkpoint: 432 packed tensors dropped, 147 BF16 tensors restored) — after TP-shard divisibility analysis; see "The 640/128 problem".
6. **EP mandatory** — same analysis, confirmed by upstream: vLLM recipe prescribes TEP8 for the official FP8 for the identical reason.
7. **Decode-only CUDA graphs, no inductor** — inductor hangs (hours, 100% CPU in `load_kernel`, reproducible); `FULL_DECODE_ONLY` at `mode: 0` recovers 7× decode throughput without it.
8. **MTP depth 4** — matches the tech report's own evaluation depth (Table 4); depth 2 gives better aggregate under high concurrency (395 vs 322 tok/s at c=8) — pick per workload.
9. **Context capped at native 262k, no YaRN in shipped config** — the tech report's MRCR (multi-needle) scores collapse past 256k (93 → 30–40 at 512k even for full attention); static YaRN additionally taxes short contexts. The base supports YaRN ×4 to 1M if you accept both.

## Limitations

- **Ampere-only validation.** Tested exclusively on 8×RTX 3090. On Hopper+ you should prefer the official FP8 checkpoint.
- **RTN, no calibration.** A calibrated AWQ may recover the remaining sliver of quality; our suite could not distinguish them, but it is a 50-task smoke, not a full harness.
- **Multi-fact long-context retrieval degrades past 256k** (inherited from the base model, per its tech report MRCR results).
- **TTFT is ~0.15 s higher** than an SGLang deployment of the same family (vLLM API overhead).
- The model **reasons by default** (`reasoning_effort: xhigh`) and this quant reasons **extra long**: on expert-level questions 37.9% of GPQA answers overflow a 32k generation budget, and long code generation can spend the entire completion budget in the reasoning phase, returning empty content. A paired test against the calibrated AWQ quant (see Benchmarks) shows a large part of this verbosity is an artifact of this quant's recipe (data-free RTN g128 with quantized attention), not only the base model. Mitigations: for code generation disable thinking (`chat_template_kwargs: {"enable_thinking": false}`) or set `max_tokens ≥ 32768` — with that rule applied, code output is clean (our Three.js probe: 19/19 requirements in non-thinking mode). Reasoning arrives in `message.reasoning` (`--reasoning-parser qwen3`).
- Vision tower is preserved in BF16 but **multimodal inference was not validated** in this release.

## License

This is a derivative work of Qwen3.8-Flash-Next and is distributed under the same **Qwen Community License 1.0** (see [LICENSE](./LICENSE)). Note the upstream terms: attribution required at >100M MAU / $20M monthly revenue; Model-as-a-Service and commercial AI work-assistant offerings require a separate license from Qwen.

## Citation

If you use this checkpoint, cite the base model:

```bibtex
@techreport{qwen2026design,
  title = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
  author = {{Qwen Team}},
  institution = {Alibaba Group},
  month = {August},
  year = {2026}
}
```

Quantized and released by [Vnimanie.AI](https://huggingface.co/VnimanieAI), 2026-08-27.
