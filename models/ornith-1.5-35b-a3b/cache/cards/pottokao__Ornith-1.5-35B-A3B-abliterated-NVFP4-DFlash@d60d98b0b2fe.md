---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
library_name: transformers
tags:
  - nvfp4
  - modelopt
  - abliterated
  - uncensored
  - moe
  - mamba
  - vllm
  - speculative-decoding
---

# Ornith-1.5-35B-A3B-abliterated-NVFP4

NVFP4 (W4A16) quantization of an **abliterated** (refusal-direction removed) build of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

The per-layer quantization recipe is **matched exactly** to the official
[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4);
the only difference is the underlying weights, which come from the abliterated model.

**20 GB**, runs on 2×16 GB consumer GPUs (tested on 2× RTX 5070 Ti, TP2).

> ⚠️ **Text-only.** The abliteration was performed on a language-model-only export, so this
> checkpoint contains **no vision tower and no MTP head** (the official NVFP4 release keeps both).
> `--language-model-only` is therefore unnecessary — there is nothing to skip.

> ⚠️ **Uncensored.** Safety refusal behaviour has been deliberately removed. You are responsible
> for how you use it.

---

## 1. How the abliteration was done

Classic **refusal-direction ablation** (orthogonalization), single direction:

| Step | Detail |
|---|---|
| Base | `ornith-ai/Ornith-1.5-35B-A3B` (BF16) |
| Probe layer | **24** — i.e. `int(num_layers × 0.6)`, 40 layers total |
| Samples | 64 harmful + 64 harmless prompts (`random.seed(0)`), last-token hidden state |
| Direction | `d = normalize(mean(harmful) − mean(harmless))` |
| Ablation | For every `.o_proj` and `.down_proj` weight: `W ← W − outer(d, dᵀW)` |

This projects the refusal direction out of the output space of the attention- and MLP-output
projections. Tooling derived from
[`remove-refusals-with-transformers`](https://github.com/Sumandora/remove-refusals-with-transformers).

The BF16 abliterated weights are published separately at
**`pottokao/Ornith-1.5-35B-A3B-abliterated`**.

---

## 2. How the quantization was done

NVIDIA **TensorRT Model Optimizer 0.45.0** (same version as the official release),
RTN for weights, `max` calibration for activations.

| Module | Precision | Notes |
|---|---|---|
| `mlp.experts` (fused, 256/layer) | **NVFP4 W4A16**, `group_size=16` | weight-only |
| `mlp.shared_expert.{gate,up,down}_proj` | **NVFP4 W4A16**, `group_size=16` | weight-only |
| `lm_head` | **NVFP4 W4A16**, `group_size=16` | weight-only |
| `linear_attn.{out_proj,in_proj_qkv,in_proj_z}` (30 layers) | **FP8 W8A8** | has `input_scale` → calibrated |
| `self_attn.{q,k,v,o}_proj` (10 layers: 3,7,…,39) | **FP8 W8A8** | has `input_scale` → calibrated |
| `conv1d`, `in_proj_a`, `in_proj_b`, `mlp.gate`, `shared_expert_gate`, norms, embeddings | **BF16** | untouched |
| KV cache | **FP8** | config flag only; no `k_scale`/`v_scale` tensors (same as official) |

Calibration: **64 samples × 512 tokens** from `abisee/cnn_dailymail` (3.0.0).
Only the 130 FP8 (W8A8) projections need calibration; all NVFP4 parts are weight-only.

### Verification against the official release

Tensor-level audit vs `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`:

```
weight_scale_2   30841  ==  official 30841   OK   (40×256×3 + 120 + 1)
input_scale        130  ==  official   130   OK   (30×3 + 10×4)
quantized_layers   291  ==  official   291   OK   (FP8 130 + W4A16_NVFP4 161)
quant_algo   MIXED_PRECISION == official      OK
per-pattern diff in the language model: 0
```

(The only differing patterns are `visual.*` and `mtp.*`, which this text-only build does not contain.)

---

## 3. Serving with vLLM + DFlash speculative decoding

The draft model used for all benchmarks below is
[`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
(0.72 GB, 6 layers, hidden 2048). A copy is included under `dflash_draft/` for convenience —
original weights and license belong to z-lab.

```bash
docker run -d --name ornith --gpus all --ipc=host --network host \
  -v /path/to/Ornith-1.5-35B-A3B-abliterated-NVFP4:/model \
  vllm/vllm-openai:nightly \
  /model --host 0.0.0.0 --port 8009 --tensor-parallel-size 2 --trust-remote-code \
  --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --moe-backend humming \
  --gpu-memory-utilization 0.93 --max-model-len 131072 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --enable-chunked-prefill \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":8,"moe_backend":"humming","attention_backend":"TRITON_ATTN"}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

### Memory / KV budget on 2× 16 GB

| Item | Per GPU |
|---|---|
| Weights + non-torch | ~10.0 GiB |
| CUDA graphs | 0.37 GiB |
| KV cache | remainder |
| **Total KV pool** | **296,828 tokens** (`--kv-cache-dtype fp8`) |

The KV pool is **shared**, so `--max-model-len 131072` and `--max-num-seqs 4` coexist fine —
but 4 concurrent requests cannot *each* hold a full 128 K context
(vLLM reports `Maximum concurrency for 131,072 tokens per request: 2.26x`).
The pool is this large because only 10 of 40 layers use full attention; the other 30 are
linear-attention (Mamba-style) layers that keep a fixed-size state instead of a growing KV cache.

**Do not push `--gpu-memory-utilization` to 0.95.** Startup fails with a CUDA OOM inside the
sampler warm-up: the top-k/top-p Triton buffer needs `batch × vocab(248320) × 4B` ≈ 62 MiB,
which vLLM's memory profiler does not reserve. `0.93` leaves enough headroom.

Kernels actually selected: attention `TRITON_ATTN`, MoE `humming`, linear layers **Marlin**
(the only NVFP4 W4A16 path on sm120), sampling FlashInfer.

---

## 4. Benchmarks

Hardware: **2× RTX 5070 Ti** (16 GB, 250 W), TP2, PCIe Gen5.
Config: 128 K context, `max-num-seqs 4`, DFlash **K=8**, FP8 KV, util 0.93.

### 4.1 Spec-Bench style suite (`vllm bench serve`, 8 prompts/category, concurrency 1, 8 warm-ups)

| Category | tok/s | Median ITL (ms) | Acceptance | accept_len |
|---|---|---|---|---|
| math_reasoning | **429.6** | 8.98 | 43.2 % | **4.45** |
| translation | 426.8 | 8.82 | 39.1 % | 4.13 |
| coding | 422.5 | 8.82 | 39.2 % | 4.14 |
| rag | 319.1 | 8.99 | 31.0 % | 3.48 |
| summarization | 317.2 | 9.03 | 31.0 % | 3.48 |
| code (held-out) | 295.6 | 9.78 | 33.9 % | 3.71 |
| qa | 293.8 | 8.80 | 21.8 % | 2.74 |
| agent (held-out) | 283.0 | 9.64 | 29.7 % | 3.38 |
| **average** | **348.4** | — | — | **3.69** |

### 4.2 Single-stream by prompt type (end-to-end, temperature 0.6)

| Prompt type | tok/s | Acceptance | accept_len |
|---|---|---|---|
| math (step-by-step) | **590.2** | 59.1 % | **5.73** |
| write code | 482.5 | 42.1 % | 4.37 |
| agent / JSON output | 449.0 | 46.1 % | 4.69 |
| Chinese technical report | 279.3 | 19.3 % | 2.54 |
| Chinese prose | 185.2 | 8.2 % | 1.65 |

**Acceptance depends heavily on the prompt type** — anywhere from 1.65 to 5.73 accept_len on the
*same* model and config. Structured output (math, code, JSON) drafts very well; free-form Chinese
prose drafts poorly. Always report the prompt mix alongside an acceptance number.

---

## 5. AIME 2026 — sanity check only

> **This is not a capability benchmark.** It exists solely to check that abliteration +
> NVFP4 quantization did not cause *catastrophic* degradation (looping, gibberish, broken
> arithmetic). There is no external baseline to compare against — neither the base model card nor
> the official NVFP4 release reports AIME 2026 — so these numbers can only rule out loud failure
> modes. **They say nothing about how this model ranks against the original, unquantized LLM.**

Setup: serial, **concurrency 1** (hard-reasoning benchmarks must not be run concurrently),
thinking enabled, `max_tokens=96000`, `top_p=0.95`, `top_k=20`, answer parsed from `\boxed{N}`.
A question was retried when it was wrong *or* hit the token limit.

| Run | Sampling | AIME I | AIME II | Total |
|---|---|---|---|---|
| 1st pass | temp 0.6 | 14/15 | 13/15 | **27/30** |
| after retries | temp 0.6 → 0.9 | 14/15 | **15/15** | **29/30** |

39 attempts across 30 questions, ~1.41 M generated tokens.

### The three that did not pass first time

| Q | temp 0.6 | temp 0.9 | Final |
|---|---|---|---|
| **15** | #1 truncated (96k), #2 truncated (96k) | #1 wrong (1194 tok), #2 truncated (96k) | ✗ still unsolved |
| **29** | #1 truncated (96k) | #1 truncated, **#2 correct** (157, 48.4k tok) | ✓ recovered |
| **30** | #1, #2, #3 all truncated (96k each) | #1 truncated, **#2 correct** (393, 68.4k tok) | ✓ recovered |

**Every failure was non-termination** — the model burned the full 96 K budget without converging.
**Not a single question was failed by producing a wrong answer.** That distinction matters for a
sanity check: the failure mode is "kept thinking", not "computed garbage", which is what a badly
damaged quantization would look like.

Q15 converged exactly once (high temperature, 1194 tokens) and answered **3766** — outside AIME's
valid 0–999 range — after misidentifying the problem as USAMO 2017 P6. That is a "doesn't know how
to solve it" failure, not a degradation artifact.

### Retrying non-termination: temperature, not tokens

Q30 was retried **three times at the same temperature and truncated all three times**, then solved
on the second attempt at temperature 0.9. Q29 followed the same pattern. Raising `max_tokens`
does not help this failure mode either. If you hit `finish_reason: length` on a reasoning task,
**change the sampling temperature rather than re-running identical parameters or adding tokens.**

---

## 6. Provenance

```
ornith-ai/Ornith-1.5-35B-A3B  (BF16, MIT)
        └── refusal-direction ablation (layer 24, o_proj + down_proj)
                └── pottokao/Ornith-1.5-35B-A3B-abliterated        (BF16, 65 GB)
                        └── modelopt 0.45.0 NVFP4, recipe matched to official
                                └── this repo                      (20 GB)
```
