---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - quantization
  - intel-xpu
  - vllm
  - hf-int4-g16
  - qwen3.5
  - battlemage
---

# Qwen3.8-27B-HF-INT4-G16-BC (`v1` Fast RTN + `BiasCorr` for `vLLM-XPU`)

> **Higher-quality `v2` Full-Hessian GPTQ weights available:** [`SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ) uses the exact same `hf_int4_g16/1` binary format and serving kernels at `-39.3%` lower prose KL and `-30.4%` lower code KL (`90.4 min` one-time conversion, zero `bias_corr` tensors).
>
> **This `-BC` repository preserves the exact `v1` fast-converted (`11.5 min`) RTN + `BiasCorr` checkpoint** (`18.47 GiB`, `2,687` tensors including `496` FP32 `bias_corr` vectors) cited across the V10/V11 cookbook study, 329-item graded battery, and `Exp 22` 8K–128K multi-turn scaling benchmarks.


A `Had128NoPerm + HF-INT4-G16` (`4.260 bpw` across decoder matrices) conversion of `Qwen3.8-27B-bf16` with serialized FP32 position-0-excluded `BiasCorr` vectors (`496/496` decoder linears quantized; `BF16` embeddings, `lm_head`, vision, and MTP weights preserved).

Built specifically for Intel Arc Pro B60/B70 (`Xe2 / bmg_g31`) GPUs, which have no native FP4/FP8 XMX instructions but deliver **2x INT8 XMX throughput over FP16/BF16**. The same packed weight buffer runs fused in-register `FWHT-128` W4A16 decode (`M <= 4`) and zero-unpack `INT8` XMX DPAS W4A8 prefill (`M >= 5`) inside `vllm/vllm-openai-xpu:v0.31.0` with XPU graphs enabled.

## Honest Summary: Where It Wins and Where It Trades Off

All live serving numbers were measured on one Intel Arc Pro B70 (`230 W` cap, `vllm/vllm-openai-xpu:v0.31.0`, XPU graphs ON, identical flags `--max-model-len 36864 --max-num-seqs 96 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`, interleaved `ABABAB` rounds, model-card sampling `qwen38-27b:nothink-p0`).

| Metric | `vLLM HF-INT4-G16-BC` (`v1`, this model) | `vLLM` Shipped `GPTQ-Int4-sym-G128` | Delta |
|---|---:|---:|---:|
| **8K Prefill TTFT / tok/s** (`n=6`) | **`3.371 s` (`2,364.4 tok/s`)** | `4.310 s` (`1,849.1 tok/s`) | **`1.28x` faster prefill** |
| **32K Prefill TTFT / tok/s** (`n=6`) | **`15.842 s` (`2,054.4 tok/s`)** | `19.920 s` (`1,633.8 tok/s`) | **`1.26x` faster prefill, `-10.7%` J/tok** |
| **0K / 8K / 32K `C=1` Decode** (`n=6`) | `29.48 / 28.38 / 26.26 tok/s` | **`33.42 / 31.95 / 29.20 tok/s`** | **`-10.1%` to `-11.8%` slower** |
| **`C=4` / `C=8` Aggregate Decode** (`n=3`) | `69.1 / 166.8 tok/s` | **`97.9 / 213.3 tok/s`** | **`-29%` / `-22%` slower** (faster TTFT on HF-G16) |
| **64L Causal WT2 KL vs BF16** (offline) | **`0.0271`** (`PPL 7.947`, `94.3%` Top-1) | `0.0494` (`PPL 8.119`, `91.4%` Top-1) | **`-45.2%` lower prose KL** |
| **64L Causal Code KL vs BF16** (offline) | **`0.1290`** W4A16 / `0.1348` W4A8 | `0.3020` (`PPL 2.835`, `85.7%` Top-1) | **`-57.3%` W4A16 / `-55.4%` W4A8 code KL** |
| **329-Item Graded Battery** (live vLLM) | **`287 / 329`** (`7` 1536-cap hits) | `283 / 329` (`10` 1536-cap hits) | `+4` (`+11/-7`, McNemar `p = 0.481`, not significant) |
| **8-Item Exact Canary Battery** (live vLLM) | **`7/8` exact case, `8/8` any case** (`sumsq=14`) | `6/8` exact case, `7/8` any case (`sumsq=30`) | Fixes `sumsq` failure |
| **Conversion Time from BF16** (1x B70) | **`11.5 min` (`691.7 s`)** | External Hessian GPTQ calibration (duration not measured here) | Streaming (`10.9 GiB` RSS, `2.38 GiB` cal VRAM) |

### Explicit Limitations
1. **Single-stream and batched decode:** Stock `AutoGPTQ / XPUwNa16` (`G128`) is `10%` to `12%` faster at `C=1` decode and `22%` to `29%` faster in `C=4/8` aggregate output throughput because `HF-INT4-G16` has slightly more scale bytes (`G16`) and runs online `FWHT-128`.
2. **Trellis comparison (`TFQ3-4b`):** With data-free RTN (`v1`), on the 400 common decoder matrices, Ash's trellis `TFQ3-4b` (`4.001 bpw`) beats scalar `HF-INT4-G16` (`4.260 bpw`) in pure quality-per-bit (`0.0253 / 0.1104` vs `0.0318 / 0.1480` data-free KL). As-shipped is domain-split because our checkpoint keeps `BF16` `lm_head` (we win WT2 `0.0271` vs `0.0321`; `TFQ3` wins Code `0.1133` vs `0.1290`). See [`SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ) for the calibrated `v2` Full-$H$ GPTQ checkpoint that moves below data-free `TFQ3-4b` on both prose and code.
3. **Dense checkpoint; plugin v0.2.0 supports MoE:** These weights are for dense `Qwen3.8-27B`. The serving plugin ([`SergiioB/vllm-hf-int4-g16-xpu`](https://github.com/SergiioB/vllm-hf-int4-g16-xpu) `v0.2.0`) supports both dense (`hf_int4_g16/1`) and MoE (`hf_int4_g16/2`, e.g., `Gemma-4-26B-A4B`). Live multi-GPU `TP=2` vLLM collectives are not qualified yet.
4. **Live-to-offline parity check:** WikiText-2 perplexity on the same 2,044 scored positions is `7.94887` live vs `7.94236` offline for the exact served checkpoint (`exp15a`, `+0.08%`). Live ran `HF_INT4_G16_MODE=auto`; the offline comparator is `W4A8`.

## How to Run in `vLLM-XPU`

```bash
# Inside vllm/vllm-openai-xpu:v0.31.0 with B70 device mapped
pip install --no-deps --no-build-isolation -e /path/to/vllm-hf-int4-g16

HF_INT4_G16_MODE=auto \
HF_INT4_G16_BIASCORR_MODE=default \
vllm serve SergiioB/Qwen3.8-27B-HF-INT4-G16-BC \
  --served-model-name default \
  --host 127.0.0.1 --port 8091 \
  --max-model-len 36864 --max-num-seqs 96 \
  --gpu-memory-utilization 0.90 --trust-remote-code
```

Set `HF_INT4_G16_MODE=w4a16` (the strict default when unset) for uniform W4A16 precision across all batch sizes, or `HF_INT4_G16_BIASCORR_MODE=off` to serve the exact same weights without bias correction.

## Verification Summary
- Plugin test suite: verified through `vllm-hf-int4-g16-xpu` (`v0.1.0` `130`-test dense suite at initial qualification; `v0.2.0` `700`-test chain covering dense + MoE, `XPUGraph` changed-input replay, FP32 RNE SYCL/Level-Zero vs Triton parity, true-NULL vs `-0.0f` bit-identity, strict two-way safetensors header admission, and mutation-tested negative checks).
- Calibration provenance: WikiText-2 train split + disjoint Python stdlib (`0/0` 6-gram overlap with scored eval windows; `cal_token_sha256: c14de8789c52ed86f55e8f6672595a3db3de811770fca119e028cd6627970e9b`).
