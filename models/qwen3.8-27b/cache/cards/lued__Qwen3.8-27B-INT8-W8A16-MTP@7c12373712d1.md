---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3.8
  - compressed-tensors
  - w8a16
  - int8
  - quantized
  - vllm
  - mtp
  - speculative-decoding
  - vision
  - conversational
base_model_relation: quantized
---

<div align="center">
  <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="380" alt="Qwen">
  <h1>Qwen3.8-27B · INT8 W8A16 · BF16 MTP</h1>
  <p><strong>A high-fidelity, Ampere-optimized quantization for dual RTX 3090 inference.</strong></p>
  <p>
    <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Base model</a> ·
    <a href="https://huggingface.co/Qwen/Qwen3.8-27B-FP8">Official FP8</a> ·
    <a href="https://github.com/vllm-project/vllm">vLLM</a> ·
    <a href="https://github.com/vllm-project/llm-compressor">llm-compressor</a>
  </p>
  <p>
    <img alt="Format W8A16" src="https://img.shields.io/badge/format-W8A16-0A2EFE">
    <img alt="Weights INT8" src="https://img.shields.io/badge/weights-INT8-6C4CF1">
    <img alt="Activations FP16 or BF16" src="https://img.shields.io/badge/activations-FP16%20%2F%20BF16-555555">
    <img alt="Target Ampere" src="https://img.shields.io/badge/target-Ampere%20sm__86-76B900">
    <img alt="License Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
  </p>
</div>

> [!NOTE]
> This is a numerical quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), not a fine-tune. All model credit belongs to **Qwen**; refer to the [upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B) for architecture, capabilities, evaluations, and usage guidance.

> [!TIP]
> This build targets vLLM on Ampere-class GPUs where native FP8 tensor-core execution is unavailable. It retains Qwen3.8's vision tower, thinking controls, native 262K context, and BF16 MTP head. The upstream model is extensible to 1M tokens with YaRN; that is separate from the native context and the available KV capacity.

## Dual RTX 3090 deployment

**Validated host profile:** 2×RTX 3090 24 GB, PCIe without NVLink or P2P, vLLM TP2, BF16 activations, FP8 E4M3 KV cache, MTP with three draft tokens, and `--max-num-batched-tokens 8192`.

### Memory and capacity

| Metric | Result |
|---|---:|
| Checkpoint size | 31.6 GB / 29.44 GiB |
| Loaded model memory per GPU | 14.85 GiB |
| Shared GPU KV pool | 266,537 tokens |
| Native maximum request length | 262,144 tokens |
| Simultaneous full-native-context capacity | 1.02× |
| Illustrative four-way KV share | Approximately 66,634 tokens per request |

`--max-model-len` caps one request's input plus generated tokens. The KV pool is a shared runtime allocation across all live requests, not a per-lane reservation: `--max-num-seqs` permits that many active sequences but does not provide that many full contexts. Near the 262K limit, reserve output-token headroom and expect only one full-length request at a time.

The open-source checkpoint is native at 262,144 tokens. Qwen documents extension to 1,000,000 tokens via static YaRN, but this deployment's 266,537-token KV pool cannot hold a 1M-token request even with YaRN enabled; the 1M path is not validated or capacity-supported here. Follow the [official YaRN configuration](https://huggingface.co/Qwen/Qwen3.8-27B#processing-ultra-long-texts) only with a runtime and memory plan sized for the requested context.

### Measured serving performance

Measured on 2026-08-15 with [llama-benchy](https://pypi.org/project/llama-benchy/) 0.4.0 against the vLLM OpenAI-compatible endpoint: one concurrent request, exact generation lengths (`min_tokens` + `ignore_eos`), cold prefix cache (unique corpus text, `cache_prompt=false`), no conversation depth, one run per cell.

| Prompt tokens | Generation tokens | Prefill tok/s | Generation tok/s | TTFT |
|---:|---:|---:|---:|---:|
| 1,024 | 512 | 1,675 | 65.6 | 0.8 s |
| 1,024 | 1,024 | 1,633 | 58.0 | 0.8 s |
| 8,192 | 512 | 1,747 | 42.3 | 4.9 s |
| 8,192 | 1,024 | 1,683 | 51.2 | 5.1 s |
| 32,000 | 512 | 1,560 | 44.1 | 20.7 s |
| 32,000 | 1,024 | 1,557 | 49.7 | 20.7 s |
| 64,000 | 512 | 1,414 | 64.5 | 45.5 s |
| 64,000 | 1,024 | 1,411 | 60.8 | 45.6 s |
| 128,000 | 512 | 1,203 | 64.5 | 106.6 s |
| 128,000 | 1,024 | 1,204 | 69.8 | 106.5 s |

Generation throughput is the per-request decode rate with MTP enabled and varies by workload, since draft acceptance is prompt-dependent. Prefill and TTFT are governed by the 8,192-token batched-token cap, which chunks long prompts and interleaves the MTP draft pass; TTFT grows with prompt length, roughly linearly past 32K.

Multi-stream throughput is not yet published: concurrent requests can intermittently crash the engine on this pin, so parallel benchmarks will be added once a vLLM release containing [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) is available (see the concurrency note under [Serving](#serving)).

### Quantization fidelity

| Candidate | Mean KLD vs BF16 | Top-1 agreement | Weight files |
|---|---:|---:|---:|
| Official Qwen FP8 / BF16 A16 | 0.004396 | 98.53% | 28.75 GiB |
| **This W8 / BF16 A16** | **0.000894** | **99.36%** | 29.44 GiB |

The KLD result measures checkpoint-weight drift over 4,563 teacher-forced positions. It is not a functional quality score; tool use, JSON-schema compliance, coding, multimodal quality, and long-context recall require separate behavioral evaluation.

### Checkpoint profile

| Property | Value |
|---|---|
| Quantization | Data-free symmetric RTN W8A16, group size 128 |
| Runtime format | `compressed-tensors` / `pack-quantized` |
| Kernel dispatch | `CompressedTensorsWNA16` → `MarlinLinearKernel` (verified in server logs) |
| Preserved precision | BF16 vision tower, `lm_head`, MTP, and recurrent GDN gates |
| MTP | BF16 draft model detected at runtime; embeddings and `lm_head` shared with the target |
| FP8 E4M3 KV cache | Verified: server booted and served on both RTX 3090s at `max_model_len=262144` |
| Runtime | vLLM; this repository is not a GGUF checkpoint |


## Why W8A16 on Ampere

RTX 3090 GPUs are Ampere `sm_86`. They do not provide the native FP8 tensor-core execution path available on newer GPU generations. In this deployment, FP8 weight-only and INT8 W8A16 checkpoints both reach Marlin-family weight-only kernels that reconstruct weights for 16-bit GEMMs.

W8A16 is therefore the better fidelity default here: it preserves eight integer bits with group-wise scaling while retaining the same practical decode class as FP8-Marlin on these GPUs.

| Format | Role on 2×24 GB at long context | Decision |
|---|---|---|
| **W8A16 — this checkpoint** | Near-lossless weights; production default | **Recommended** |
| Official / online FP8 weight quantization | Convenient, but higher measured weight error on sm_86 fallback | Useful comparison |
| W8A8 INT8 | Can improve prefill throughput; may regress skinny-batch decode | Specialized profile |
| W4A16 | Smaller weights and more cache headroom | Higher long-context risk for recurrent GDN layers |

### Why not AutoRound?

This checkpoint uses llm-compressor's `QuantizationModifier`, not AutoRound. At W8A16, AutoRound's installed RTN configuration disables optimized RTN by default for efficiency, and the measured KLD is already below `9e-4` nats/token. Re-encoding these weights with AutoRound would add a new export and serving path without evidence of a meaningful quality gain.

## Quantization design

### Architecture audit

The BF16 source was checked before quantization against the preceding Qwen3.6 structural contract:

- 64 language layers: 48 Gated DeltaNet linear-attention layers and 16 full-attention layers.
- Hidden size 5,120; intermediate size 17,408; padded vocabulary 248,320.
- One MTP layer with the same 15 top-level `mtp.*` tensors.
- Stable GDN projection names: `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, and `out_proj`.
- The only relevant rename was non-Linear `convNd` → `conv1d`, outside the quantization target.

The pre-quantization structural audit completed successfully before any weights were modified.

### Recipe

| Component | Precision | Reason |
|---|---|---|
| MLP projections | INT8 W8A16 | Largest dense GEMMs; strong memory return |
| Full-attention projections | INT8 W8A16 | Measured low output-distribution error |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj` | INT8 W8A16 | Recovers approximately 4 GB while remaining stable in the 4K GDN probe |
| GDN `in_proj_a`, `in_proj_b` | BF16 | Tiny recurrent gates; inexpensive precision safeguard |
| Vision tower | BF16 | Preserve multimodal fidelity |
| `lm_head` | BF16 | Preserve final-logit fidelity |
| MTP head | BF16 | Keep the speculative drafter close to the target |
| Norms, `conv1d`, `A_log`, `dt_bias` | Native BF16 / FP32 | Non-Linear; never packed |

The result contains 400 quantized Linear GEMMs: 192 MLP projections, 64 full-attention projections, and 144 dense GDN projections.

```yaml
# recipe.yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore:
        - lm_head
        - re:.*visual.*
        - re:.*mtp.*
        - re:.*linear_attn[.]in_proj_a$
        - re:.*linear_attn[.]in_proj_b$
      scheme: W8A16
```

## Quantization fidelity: KLD versus the BF16 source

> [!IMPORTANT]
> This section measures **checkpoint-weight quantization**. It is not a BF16-versus-FP16 benchmark and it does not measure the FP8 KV cache. The teacher is always the original BF16 checkpoint.

For every teacher-forced next-token position, the metric is the KL divergence from the BF16 teacher distribution to the candidate distribution, `D_KL(p_BF16 ‖ p_candidate)`, computed over the full 248,320-token vocabulary, in nats/token, with FP32 `log_softmax` and accumulation. No sampling or top-k approximation is involved.

### Overall results

| Candidate checkpoint | Stored weights | A16 dtype in measurement | Mean KLD vs BF16 | Top-1 agreement |
|---|---|---|---:|---:|
| BF16 source | BF16 | BF16 | 0 by definition | 100% |
| [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | FP8 E4M3, 128×128 blocks | BF16 after dequantization | 0.004396 | 98.53% |
| **This checkpoint** | INT8 symmetric, groups of 128 | BF16 after INT8×scale | **0.000894** | **99.36%** |
| **This checkpoint, FP16 profile** | Same INT8 weights | FP16 after INT8×scale | **0.000641** | 99.19% |

The apples-to-apples weight comparison is official FP8/BF16-A16 versus this W8/BF16-A16. The FP16-A16 row covers the alternative `--dtype float16` serving profile; its lower value is prompt-suite-specific rounding cancellation, not evidence that FP16 is generally more accurate.

On RTX 3090, Transformers dequantizes the official checkpoint's stored FP8 weights to BF16 because native dynamic-FP8 activation execution is unavailable. The comparison therefore isolates stored-weight error under the same Hugging Face BF16 execution path; it does not represent native FP8 W8A8 execution on newer hardware.

### Short-prompt suite

| Prompt | Category | Tokens | Positions | Official FP8 / BF16 A16 | This W8 / BF16 A16 | This W8 / FP16 A16 |
|---|---:|---:|---:|---:|---:|---:|
| `factual` | factual | 13 | 12 | 0.001544 | 0.000410 | 0.000311 |
| `code-fib` | code | 88 | 87 | 0.001236 | 0.000547 | 0.000379 |
| `physics-uncertainty` | physics | 86 | 85 | 0.001732 | 0.000829 | 0.000442 |
| `math-train` | math | 58 | 57 | 0.002559 | 0.000575 | 0.000455 |
| `sql-top5` | technical | 60 | 59 | 0.001769 | 0.000869 | 0.000531 |
| `narrative-clock` | narrative | 68 | 67 | 0.003056 | 0.001018 | 0.001011 |
| `instruction-stack` | instruction | 43 | 42 | 0.001691 | 0.000543 | 0.000428 |
| `history-industrial` | history | 60 | 59 | 0.001295 | 0.000652 | 0.000423 |
| **Token-weighted mean** | | | **468** | **0.001871** | **0.000719** | **0.000517** |

### Long-context GDN probe

The probe used the first 4,096 tokenizer tokens of the upstream Qwen3.8 model card as cached locally, producing 4,095 scored positions.

| Candidate | Mean KLD | Top-1 agreement | Eight consecutive 511-position window means |
|---|---:|---:|---|
| Official FP8 / BF16 A16 | 0.004684 | 98.49% | 0.01652, 0.01542, 0.00223, 0.00162, 0.00059, 0.00045, 0.00032, 0.00039 |
| This W8 / BF16 A16 | 0.000914 | 99.44% | 0.00314, 0.00274, 0.00054, 0.00036, 0.00015, 0.00010, 0.00015, 0.00014 |
| This W8 / FP16 A16 | 0.000655 | 99.24% | 0.00216, 0.00201, 0.00047, 0.00026, 0.00009, 0.00008, 0.00010, 0.00009 |

The final seven positions are included in each overall mean but excluded from the equal-size window breakdown. KLD decreased rather than accumulated across this probe; no recurrent drift was observed at 4K tokens.

<details>
<summary><strong>Exact short prompts</strong></summary>

1. `factual`: `The capital of France is Paris, and the capital of Japan is`
2. `code-fib`:
   ```python
   def fibonacci(n):
       if n < 2:
           return n
       return fibonacci(n - 1) + fibonacci(n - 2)

   # Rewrite with memoization to avoid exponential blowup, then add a
   # test that checks fibonacci(10) == 55 and fibonacci(0) == 0. Use a
   # dict as the cache and make the function private to the module.
   ```
3. `physics-uncertainty`: `In quantum mechanics, the uncertainty principle states that the product of the uncertainties in position and momentum cannot be smaller than about half of the reduced Planck constant. More precisely, if many identically prepared particles are measured, the standard deviations of position and momentum satisfy the inequality derived by Kennard in 1927, which is a direct consequence of the non-commutativity of the position and momentum operators in Hilbert space.`
4. `math-train`: `A train travels 240 kilometres in 3 hours and 15 minutes, stopping for 10 minutes at one station along the way. What is its average speed in kilometres per hour? Show your calculation step by step, and state whether the stop changes the average speed.`
5. `sql-top5`: `Write a SQL query that returns the top five most frequently ordered products, joining the orders table with the products table on product_id, grouping by product id, and ordering by the count descending. Then explain what index would make the query faster and why a correlated subquery would be a worse choice here.`
6. `narrative-clock`: `Once upon a time, in a quiet village at the edge of a dark forest, there lived an old clockmaker who had never once repaired a clock that was broken on purpose. Each morning he wound every clock in the village square, and each evening he listened to them tick together like a heartbeat that the forest itself had learned to trust.`
7. `instruction-stack`: `Explain the difference between a stack and a queue, and give one real-world example of each. Keep the answer under one hundred words and use the words push, pop, enqueue, and dequeue exactly once each.`
8. `history-industrial`: `The Industrial Revolution began in Britain in the late eighteenth century. Three factors that made this possible were the abundance of coal and iron ore, a stable legal system that protected patents and contracts, and a network of navigable rivers and canals that lowered the cost of moving raw materials and finished goods.`

</details>

<details>
<summary><strong>Measurement environment and limitations</strong></summary>

- Run locally on 2026-08-14 with identical teacher-tokenizer IDs.
- BF16 teacher snapshot: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Official FP8 snapshot: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Environment: 2×RTX 3090, PyTorch 2.11.0+cu128, Transformers 5.10.1, compressed-tensors 0.17.1.
- Teacher and candidates used the same 69-module device map.
- These are Hugging Face decompressed-path logits, not vLLM Marlin or native-FP8 kernel logits.
- `use_cache=False`: the serving configuration's FP8 KV-cache error is not included.
- No scheduling, sampling, image, or context beyond 4,096 tokens was measured.
- Teacher logits were cached as FP16 before the shared FP32 KLD calculation, introducing a small common precision floor.

</details>

## KV-cache rationale: E4M3 versus E5M2

`--kv-cache-dtype` controls runtime attention-cache storage. It does **not** alter this checkpoint's INT8 weights. E4M3 and E5M2 both consume one byte per cached value, so this decision affects numerical behavior—not memory capacity.

| KV dtype | Mantissa bits | Maximum finite value | Minimum subnormal | Practical tradeoff |
|---|---:|---:|---:|---|
| `fp8_e4m3` | 3 | ±448 | 2^-9 ≈ 0.00195 | Better in-range precision |
| `fp8_e5m2` | 2 | ±57,344 | 2^-16 ≈ 0.0000153 | Much wider dynamic range |

E4M3 is the default because attention K/V values generally benefit more from its additional mantissa bit than from E5M2's extreme range.

> [!WARNING]
> Runtime compatibility is verified; long-context KV accuracy is not. The checkpoint declares `kv_cache_scheme: null`, so it contains no dataset-calibrated K/V scales. The KLD measurement above uses no cache, and no saturation probe was performed.

E5M2 remains the fallback if measured ranges or a future vLLM build expose E4M3 saturation or kernel incompatibility. The higher-value follow-up is an A/B of E4M3 with default scale 1.0, runtime-calculated scales, and dataset-calibrated scales—not an unmeasured dtype switch.

## Serving

### GPU interconnect profiles

The validated command below targets stock-driver dual-GPU systems without a working CUDA peer-to-peer path. Consumer Ampere P2P availability depends on the driver and host configuration; the physical PCIe topology alone does not establish that peer memory access works.

| Host configuration | NCCL environment | vLLM all-reduce | Status |
|---|---|---|---|
| Stock driver; P2P unavailable or unverified | `NCCL_P2P_DISABLE=1` | Pass `--disable-custom-all-reduce` | Validated configuration |
| P2P-enabled driver; peer access verified on both GPUs | Do not set `NCCL_P2P_DISABLE` | Omit `--disable-custom-all-reduce` | Alternative profile; benchmark on the actual PCIe topology |

Verify the second profile with a CUDA peer-access test such as `p2pBandwidthLatencyTest`, not only `nvidia-smi topo -m`. Remove the two workarounds together: enabling vLLM's custom all-reduce while NCCL peer access remains disabled is not the intended profile. A patched driver changes the host execution path, not the checkpoint.


### Recommended vLLM command

The reference llama-swap/podman deployment, minus the container plumbing, with two deliberately conservative flags: prefix caching is off by default (the reference enables it after applying the safety patch below) and the sequence cap is 2 (the reference runs 4). Rationale in the notes after the command.

```bash
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

vllm serve lued/Qwen3.8-27B-INT8-W8A16-MTP \
  --served-model-name qwen3.8-27b-int8-w8a16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --performance-mode balanced \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --mamba-cache-mode align \
  --prefix-match-unit 16 \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`NCCL_P2P_DISABLE=1` avoids NCCL initialization stalls on dual consumer GPUs without NVLink. `--mamba-cache-mode align` is required by the Qwen3.8 MTP/GDN serving path. Three speculative tokens is the measured default.

**Prefix caching is off by default.** On this hybrid GDN + MTP architecture, prefix caching without the 9-line source patch from [vllm#48375](https://github.com/vllm-project/vllm/pull/48375) can corrupt recurrent-state KV, which surfaces as wrong tool-call or long-context output rather than an error. The reference deployment applies that patch at container start (along with this repository's `chat_template.jinja` via `--chat-template`), which is what lets it run with `--enable-prefix-caching`. If you apply the same patch, flip this flag back on; the published numbers were measured with a cold cache, so either setting leaves the table valid.

**Concurrency is capped at 2, not 4, on this engine pin.** Two or more concurrent requests can intermittently crash the engine (asynchronous `cudaErrorIllegalAddress` in the GDN spec-decode state path; same class as [vllm#37431](https://github.com/vllm-project/vllm/issues/37431)). The targeted fix is [vllm#50021](https://github.com/vllm-project/vllm/pull/50021), which is open and awaiting its merge as of this writing; a vLLM release containing it should restore `--max-num-seqs 4`. Until then, a crashed engine drops in-flight requests and the service (llama-swap, podman, or a process manager) restarts it; if that is unacceptable, lower the cap to 1, which is the fully validated configuration on this pin.

### MTP acceptance

MTP acceptance is workload-, sampling-, and draft-depth-dependent; it is not directly comparable across different models or benchmark prompts. With this checkpoint and three draft tokens on the vLLM pin above, per-window draft acceptance ranged from **42.1% to 92.4%** across ten-second engine windows (including warmup), and overall draft acceptance was **65.5%** (accepted/drafted) in the single-stream 8K/1,024 run.

The MTP checkpoint and `lm_head` are already BF16. Requantizing them cannot improve alignment. Increasing `num_speculative_tokens` adds progressively less accurate draft positions and normally lowers the reported average acceptance rate; lowering it can raise that percentage while reducing useful speculative work. The serving decision must therefore use end-to-end output throughput, not acceptance alone. Keep `num_speculative_tokens=3` unless a controlled throughput test on the intended workload shows a better setting.


## Operational notes

| Symptom or question | Action |
|---|---|
| Which vLLM version is known to work? | Use the pinned image `nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`. Other releases may work, but must support Qwen3.8, the compressed-tensors Marlin path, aligned Mamba cache, and MTP speculative decoding together. |
| Can llama.cpp load this repository? | No. This is a compressed-tensors checkpoint for vLLM, not GGUF. A separate GGUF conversion needs its own compatibility and quality validation. |
| The first startup appears idle | Wait for model loading, Torch/Triton compilation, CUDA graph capture, and the final API-ready message. Persist `/root/.cache/vllm` and the active Triton cache directory—`/root/.triton/cache` by default—across container runs to avoid repeating avoidable compilation work. |
| NCCL stalls during startup | Use the stock-driver interconnect profile above unless CUDA peer access has been explicitly verified. Check that container or service-level environment variables do not override the selected profile. |
| Startup runs out of GPU memory | Lower `--gpu-memory-utilization` or `--max-num-batched-tokens`. If the resulting KV pool cannot hold one configured maximum-length request, also lower `--max-model-len`. Change one limit at a time and confirm the KV allocation printed by vLLM. |
| A long request is rejected despite `max_model_len=262144` | The limit covers input plus generated tokens, and all active requests share one KV pool. Reserve output-token headroom and reduce concurrent live context when approaching the model limit. |
| Is vision validated? | The BF16 vision tower is present, but the benchmarks and KLD suite on this card are text-only. Treat multimodal quality and memory use as unmeasured until evaluated separately. |


## Files and provenance

| File | Purpose |
|---|---|
| `model-0000N-of-00006.safetensors` | Packed W8A16 language and GDN weights plus preserved BF16 tensors |
| `model.safetensors.index.json` | Shard-to-tensor mapping |
| `model_mtp.safetensors` | BF16 MTP head, 849 MB (810 MiB) / 15 tensors |
| `recipe.yaml` | Exact llm-compressor W8A16 recipe |
| `chat_template.jinja` | Upstream chat and tool-use template |

## Acknowledgements and license

This repository repackages numerical weights derived from **Qwen/Qwen3.8-27B**. It does not claim authorship of the base model, architecture, training data, or upstream evaluations. Header styling and the Qwen logo follow the official Qwen repositories.

Apache-2.0 for this packaging. Respect all upstream model terms and attribution requirements.
