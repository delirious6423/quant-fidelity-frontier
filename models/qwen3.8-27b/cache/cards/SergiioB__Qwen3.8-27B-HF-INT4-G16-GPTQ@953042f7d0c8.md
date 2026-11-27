---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - quantization
  - gptq
  - intel-xpu
  - vllm
  - hf-int4-g16
  - qwen3.5
  - battlemage
---

# Qwen3.8-27B-HF-INT4-G16-GPTQ (`v2` Full-Hessian Closed-Loop GPTQ for `vLLM-XPU`)


A `Had128NoPerm + HF-INT4-G16` (`4.255 bpw` across decoder matrices) conversion of `Qwen3.8-27B-bf16` quantized with **sequential closed-loop Full-Hessian GPTQ** (`496/496` decoder linears quantized; `BF16` embeddings, `lm_head`, vision, and MTP weights preserved; `0` `bias_corr` tensors).

Built for Intel Arc Pro B60/B70 (`Xe2 / bmg_g31`) GPUs using the exact same `hf_int4_g16/1` (`xe2_dpas_native_v1`) binary format and native SYCL/Level-Zero/OpenCL serving kernels as [`SergiioB/Qwen3.8-27B-HF-INT4-G16-BC`](https://huggingface.co/SergiioB/Qwen3.8-27B-HF-INT4-G16-BC) (`v1` fast RTN + `BiasCorr`, `11.5 min` conversion). Drop-in replacement with zero serving-code changes.

## Primary Verification: `v2` Full-H GPTQ vs `v1` RTN+BC (`Qwen3.8-27B`, 64L True-Causal)

All offline KL numbers are exact 64-layer true-causal measurements against `Qwen3.8-27B-bf16` (`WT2 PPL 7.798`, `Code PPL 2.374`) on the identical frozen `2 × 2,044`-token windows used across `Exp 13A` and `Exp 15A`. Uncertainty bounds are paired within-window 32-token block bootstrap `95%` CIs (`10,000` resamples); `bootstrap P(Δ < 0)` is the fraction of resamples favoring `v2` (not a p-value).

| Metric | `v1` RTN + `BiasCorr` (`-BC` repo) | **`v2` Full-H GPTQ (this repo)** | Delta (`v2` vs `v1`) |
|---|---:|---:|---|
| **Offline 64L WT2 KL vs BF16 (W4A16)** | `0.027078` (`PPL 7.947`, `94.32%` Top-1) | **`0.016430`** (`PPL 7.891`, **`95.01%`** Top-1) | **`-39.32%` KL** (`[-0.01458, -0.00718]`, `bootstrap P(Δ<0) = 1.0000`, `10,000/10,000`) |
| **Offline 64L Code KL vs BF16 (W4A16)** | `0.128974` (`PPL 2.541`, `91.73%` Top-1) | **`0.089716`** (`PPL 2.515`, **`93.10%`** Top-1) | **`-30.44%` KL** (`[-0.05002, -0.02937]`, `bootstrap P(Δ<0) = 1.0000`, `10,000/10,000`) |
| **Offline 64L WT2 KL vs BF16 (W4A8 DPAS)** | `0.027611` (`PPL 7.942`, `94.47%` Top-1) | **`0.017249`** (`PPL 7.879`, **`94.96%`** Top-1) | **`-37.53%` KL** (`[-0.01370, -0.00746]`, `bootstrap P(Δ<0) = 1.0000`) |
| **Offline 64L Code KL vs BF16 (W4A8 DPAS)** | `0.134799` (`PPL 2.555`, `91.68%` Top-1) | **`0.094046`** (`PPL 2.517`, **`93.05%`** Top-1) | **`-30.23%` KL** (`[-0.05211, -0.03036]`, `bootstrap P(Δ<0) = 1.0000`) |
| **Live `vllm serve` WT2 PPL** (`auto`, `native`, `0` fallbacks) | `7.95599` (NLL `2.07393`) | **`7.88621`** (NLL `2.06512`) | `-0.06978` PPL (`-0.00881` NLL, CI `[-0.01959, +0.00137]`, `P(Δ<0)=0.952`, n.s. at `n=2,044`; `+0.09%` vs offline W4A8) |
| **Live `vllm serve` Code PPL** (`auto`, `native`, `0` fallbacks) | `2.55646` (NLL `0.93862`) | **`2.51799`** (NLL `0.92346`) | `-0.03847` PPL (`-0.01516` NLL, CI `[-0.04680, +0.01550]`, `P(Δ<0)=0.834`, n.s. at `n=2,044`; `+0.05%` vs offline W4A8) |
| **Live `C=1` Decode** (256 tok, `n=3` median, `vllm serve`) | `29.69 tok/s` | `29.92 tok/s` | **Speed parity** (note: `v2` reused a warm server at `72 °C`; not a cooled A/B) |
| **Live 2K Cold Prefill** (2,041 tok, `n=2`, `vllm serve`) | `2496.5 tok/s` | `2533.3 tok/s` | **Speed parity** (identical `hf_int4_g16/1` kernel geometry) |
| **8-Item DNR-009 Canary Battery** (live `vllm serve`) | `8/8` exact, `8/8` any (`sumsq=14`) | `7/8` exact (`prime="No"`), `8/8` any (`sumsq=14`) | Both `8/8` semantically correct |
| **Checkpoint Size / Effective Bits** | `18.47 GiB` (`2,687` tensors, `4.260 bpw`) | **`18.45 GiB`** (`2,191` tensors, `4.255 bpw`) | **`-14.9 MiB`** (`bias_corr` omitted; epilogue branch-free) |

*Note on live `vllm serve` comparison:* Under `HF_INT4_G16_MODE=auto`, 511-token prompt evaluation runs on the native `W4A8` INT8 DPAS prefill path. Served PPL matches offline `W4A8` (`7.886` vs `7.879` WT2; `2.518` vs `2.517` Code), verifying kernel-path fidelity. Token-level NLL includes irreducible ground-truth surprise variance so live ΔNLL CIs cross zero at `n = 2,044`.

## Out-of-Domain (OOD) & Long-Context Generalization (`Exp 19-OOD`, 64L True-Causal W4A16)

Because `v2` was calibrated on 512-token English prose (WikiText-2 train) and Python stdlib code, we evaluated the exact shipped `v2` weights without reconversion across **five unseen/out-of-domain test sets**: chat-templated GSM8K/MMLU/HumanEval (`ood_chat_512`, exact Exp 18 SHA `4305daed...`), Spanish Wikipedia (`es_wiki_512`), Spanish literary prose (`es_quixote_512`), and 2,048-token windows (`4×` calibration sequence length). 6-gram overlap between the 256K calibration corpus and eval windows is `0/0` on `wt2_512`, `code_512`, `es_wiki_512`, `es_quixote_512`, `wt2_2048`, and `code_2048`, and `11` boilerplate Python signature n-grams on `ood_chat_512` (HumanEval prompts).

| Domain | `v1` RTN+BC KL (Top-1) | **`v2` GPTQ KL (Top-1)** | Delta (`v2` vs `v1`, paired block-32 95% CI) |
|---|---:|---:|---|
| WikiText-2 prose (`4×512`, in-domain) | `0.02708` (`94.32%`) | **`0.01643`** (`95.01%`) | **`-39.3%`** (`[-0.01458, -0.00718]`, `P(Δ<0)=1.0000`, PASS) |
| Python `difflib.py` (`4×512`, in-domain) | `0.12897` (`91.73%`) | **`0.08972`** (`93.10%`) | **`-30.4%`** (`[-0.05002, -0.02937]`, `P(Δ<0)=1.0000`, PASS) |
| OOD Chat (`4×512`, GSM8K/MMLU/HumanEval, Qwen chat template) | `0.05389` (`94.37%`) | **`0.04754`** (`94.52%`) | **`-11.8%`** (`[-0.01420, +0.00048]`, `P(Δ<0)=0.9640`, n.s.) |
| Spanish Wikipedia (`4×512`, OOD language) | `0.01639` (`95.11%`) | **`0.01432`** (`95.01%`) | **`-12.6%`** (`[-0.00335, -0.00069]`, `P(Δ<0)=0.9986`, PASS) |
| Spanish Don Quixote (`4×512`, OOD literary) | `0.16220` (`88.70%`) | **`0.15937`** (`87.62%`) | **`-1.7%`** (`[-0.02056, +0.01433]`, `P(Δ<0)=0.6329`, n.s.) |
| WikiText-2 Long (`1×2048`, 4× cal length) | `0.02141` (`95.31%`) | **`0.01408`** (`95.07%`) | **`-34.3%`** (`[-0.01034, -0.00483]`, `P(Δ<0)=1.0000`, PASS) |
| Python `difflib.py` Long (`1×2048`, 4× cal length) | `0.12215` (`92.62%`) | **`0.08903`** (`93.89%`) | **`-27.1%`** (`[-0.04767, -0.01885]`, `P(Δ<0)=1.0000`, PASS) |

## Cross-Format Scoreboard (`Qwen3.8-27B`, Identical `2 × 2,044` Tokens, KL vs BF16)

Comparisons against `TFQ3-4b`, `FP8`, `MLX-4bit`, and `GPTQ-G128` below are unpaired point estimates across the same frozen token streams (`Exp 13A` / `Exp 15A` / `Exp 19-Full`, exact `BF16` parity `max_abs = 0.0`).

| Format | Effective Bits | Quantized Linears | WT2 KL vs BF16 (Top-1) | Code KL vs BF16 (Top-1) | Intel B70 Live Serving |
|---|---:|---:|---:|---:|---|
| `BF16` Reference | `16.0` bpw | `0` | `0.00000` (`100.00%`) | `0.00000` (`100.00%`) | TP=2 only (`54 GB`) |
| `FP8` (shipped) | `~8.0` bpw | `400` | `0.00358` (`97.60%`) | `0.01689` (`96.48%`) | TP=2 (`28 GB`) |
| **`HF-INT4-G16 v2` GPTQ W4A16 (this repo)** | **`4.255` bpw** | **`496`** | **`0.01643` (`95.01%`)** | **`0.08972` (`93.10%`)** | **Live 1× B70 (`vLLM-XPU`)** |
| **`HF-INT4-G16 v2` GPTQ W4A8 DPAS** | **`4.255` bpw** | **`496`** | **`0.01725` (`94.96%`)** | **`0.09405` (`93.05%`)** | **Live 1× B70 (`auto` prefill)** |
| `TFQ3-4b` Trellis + `BiasCorr` (matched 400) | `4.001` bpw | `400` | `0.01852` (`94.47%`) | `0.08879` (`92.86%`) | No Xe2 kernel (Metal) |
| `TFQ3-4b` Trellis Data-Free (matched 400) | `4.001` bpw | `400` | `0.02528` (`93.15%`) | `0.11042` (`91.14%`) | No Xe2 kernel (Metal) |
| `HF-INT4-G16 v1` RTN + `BiasCorr` (`-BC` repo) | `4.260` bpw | `496` | `0.02708` (`94.32%`) | `0.12897` (`91.73%`) | Live 1× B70 (`vLLM-XPU`) |
| `HF-INT4-G16 v1` RTN Data-Free | `4.255` bpw | `496` | `0.03163` (`94.13%`) | `0.14920` (`90.02%`) | Live 1× B70 (`vLLM-XPU`) |
| `TFQ3-4b` As-Shipped (4-bit `lm_head`) | `4.001` bpw | `400+head` | `0.03206` (`91.88%`) | `0.11327` (`91.05%`) | No Xe2 kernel (Metal) |
| `MLX-4bit` + `BiasCorr` / Base | `~4.25` bpw | `400` | `0.03897` / `0.05060` | `0.15879` / `0.18407` | TensorFold XPU |
| Stock `GPTQ-Int4-sym-G128` | `~4.25` bpw | `400` | `0.04942` (`91.44%`) | `0.30198` (`85.71%`) | Live 1× B70 (`vLLM-XPU`) |

*Fair qualifiers on `v2` vs `TFQ3-4b`:* `TFQ3-4b` uses `0.254 bpw` fewer bits (`4.001` vs `4.255 bpw`) and is data-free (or `+BC` with `2,048` cal tokens), while `v2` uses `261,632`-token closed-loop GPTQ and quantizes `496` linears (`TFQ3` leaves `96` small GDN projections in `BF16`). On in-domain point estimates (unpaired), `v2` W4A16 is `-35.0%` WT2 / `-18.8%` Code vs data-free `TFQ3-4b`, and `-11.3%` WT2 / `+1.0%` Code vs `TFQ3-4b + BC`. Hessian-calibrated trellis (QTIP-style) has not been tested here. `NVFP4` was not measured on `Qwen3.8-27B`.

## Why `bias_corr` Is Omitted (W4A16 & W4A8 3-Arm Gate Results)

- **W4A16 (`Exp 19-Full`, 2 domains × 3 arms):** Post-GPTQ `+BC` changed WT2 KL by `-0.000018` (`bootstrap P(Δ<0) = 0.587`, not significant) and Code KL by `-0.000044` (`P(Δ<0) = 0.515`, not significant). The sign-flipped `-BC` control moved WT2 by `-0.000206` (`P(Δ<0) = 0.990`), confirming that after Full-H GPTQ the `BiasCorr` direction carries no resolvable signal above the `~2e-4` KL noise floor at W4A16.
- **W4A8 (`Exp 19-OOD`, 3-arm closed-loop W4A8 `DF` vs `+BC` vs `-BC`, `8,176` pos0-excluded cal tokens):** Overall 4-check OOD gate verdict: **`FAIL (not shipped)`**.

| Domain (W4A8 DPAS) | `v2` W4A8 `DF` KL | `v2` W4A8 `+BC` KL | `v2` W4A8 `-BC` (null) KL | `+BC` vs `-BC` 95% CI |
|---|---:|---:|---:|---|
| WikiText-2 (`4×512`) | `0.01725` | `0.01688` (`-2.16%`, `P=0.8261`) | `0.01715` (`-0.60%`) | `+BC vs -BC`: `[-0.00078, +0.00031]` |
| Code (`4×512`) | `0.09405` | `0.09173` (`-2.46%`, `P=0.9270`) | `0.09312` (`-0.99%`) | `+BC vs -BC`: `[-0.00476, +0.00207]` |
| OOD Chat (`4×512`) | `0.04886` | `0.04840` (`-0.93%`, `P=0.6360`) | `0.04651` (`-4.81%`) | `+BC vs -BC`: `[-0.00019, +0.00408]` |
| Spanish Wiki (`4×512`) | `0.01489` | `0.01481` (`-0.50%`, `P=0.5725`) | `0.01495` (`+0.46%`) | `+BC vs -BC`: `[-0.00097, +0.00048]` |
| Spanish Quixote (`4×512`) | `0.16549` | `0.16247` (`-1.82%`, `P=0.8311`) | `0.16622` (`+0.44%`) | `+BC vs -BC`: `[-0.01008, +0.00241]` |

Dropping `bias_corr` saves `14.9 MiB` and keeps the serving epilogue branch-free.

## `v2` Quantization Recipe (`full_h_uncentered_gptq_v1`) & Conversion Cost

- **Format:** Exact `hf_int4_g16/1`, `Had128NoPerm` (`had_seed=104729`), `xe2_dpas_native_v1` (`q_pack` int32 `[K//8, N]`, `u_pack` int16 `[K//64, N]`, `d_row` fp32 `[N]`).
- **Algorithm:** Closed-loop sequential 64-layer GPTQ. Uncentered rotated Hessian `H = (1/n) sum(X_r^T X_r)` accumulated in FP32 on XPU and inverted via **CPU FP64 Cholesky** once per shared input group; block-128 error diffusion on XPU with dynamic per-16-column micro-scale `u in {1..16}` re-optimization under fixed RTN-optimal `d_row` and exact FP32 associativity `q = (qi * u_cur) * d_row`.
- **Shrinkage damping:** `0.05` (`attn_in` / `attn_out`), `0.10` (`mlp.gate_proj` / `up_proj`), `0.35` (`mlp.down_proj`). *Disclosure:* damping grid values were selected in Pilot-B using the held-out WT2/Code windows (`every` tested grid point beat `v1` by `>= 0.69 dB/layer`).
- **Calibration corpus:** `512 × 512` tokens (`261,632` pos0-excluded): `256` WikiText-2 train windows + `256` shuffled full-alphabet Python stdlib windows (`139/511` files, `difflib.py` excluded), `cal_token_sha256: 66c145e1a22e77bb2195171bc658995aaffdb8e183361a81b0f58f7de8c81b89`.
- **One-time conversion cost (1× Arc Pro B70):** **`90.4 min`** (`5426.1 s`, includes inline 7-arm per-layer evaluation), **`29.94 GiB`** peak host RSS, **`331.7 kJ` (`0.092 kWh`)** GPU card energy (CPU FP64 Cholesky energy was not metered). Note: `vllm-hf-int4-g16-xpu`'s CLI `convert_hf_int4_g16.py` currently ships only the fast `11.5 min` RTN+BC path; `v2` GPTQ was produced by the standalone study script (`exp19_full_64l_gptq_v1.py`).

## Explicit Limitations & Not Yet Tested on `v2`
1. **Serving throughput vs stock `GPTQ-G128`:** On identical `hf_int4_g16/1` kernel geometry (long-context/batched ladder measured on `v1` `-BC` repo), `HF-INT4-G16` is `1.26×`–`1.28×` faster on 8K/32K prefill and wins full-turn wall clock and energy at 32K–128K (`Exp 22`), while stock `GPTQ-G128` is `10%`–`12%` faster at `C=1` decode and `22%`–`29%` faster at `C=4/8` aggregate decode.
2. **Not yet measured on `v2` weights:** 329-item graded task battery (`287/329` was measured on `v1`), 8K–128K `Exp 22` multi-turn scaling (measured on `v1`; identical kernel geometry), MoE / `hf_int4_g16/2` GPTQ (plugin `v0.2.0` supports MoE serving, but GPTQ has only been tested on dense `Qwen3.8-27B`), the other 11 model families in V11, live multi-GPU `TP=2`, and `NVFP4` on 27B.
3. **Code eval breadth:** In-domain code eval is a single CPython stdlib file (`difflib.py`, `2,044` tokens at `512` and `2,047` at `2048`), supplemented by HumanEval prompts inside `ood_chat_512`.

## How to Run in `vLLM-XPU`

```bash
# Inside vllm/vllm-openai-xpu:v0.31.0 with Intel Arc Pro B70 mapped
# Install plugin v0.2.0 (https://github.com/SergiioB/vllm-hf-int4-g16-xpu, commit d0e9b26)
pip install --no-deps --no-build-isolation -e /path/to/vllm-hf-int4-g16-xpu

HF_INT4_G16_MODE=auto HF_INT4_G16_BACKEND=native vllm serve SergiioB/Qwen3.8-27B-HF-INT4-G16-GPTQ   --served-model-name default   --host 127.0.0.1 --port 8091   --max-model-len 36864 --max-num-seqs 96   --gpu-memory-utilization 0.90 --trust-remote-code
```

Set `HF_INT4_G16_MODE=w4a16` (default when unset) for uniform W4A16 precision across all batch sizes.

## Verification Summary
- **Byte-exact repack verification:** `2,191/2,191` tensors verified by per-tensor **SHA-256 byte equality** between the served 64-layer checkpoint and the uploaded 7-shard layout (`model-00001.safetensors`–`model-00007.safetensors`, `18.453 GiB`, commit `e7565b005d3e`), `496/496` quantized decoder linears, `0` `bias_corr`, `0.0` V1 pack/unpack max diff, every `d_row` finite positive FP32.
- **Live `vllm serve` verification (`Exp 19-Live`):** `hf-int4-g16-xpu v0.2.0`, `496/496` modules admitted with `HF_INT4_G16_BACKEND=native`, `HF_INT4_G16_MODE=auto`, `0` Triton fallbacks, `8/8` DNR-009 any-case canaries, `3/3` greedy sanity prompts, `+0.09%` / `+0.05%` live-to-offline W4A8 PPL agreement.
- **Plugin test chain (`v0.2.0`):** `699 + 1 = 700`-test release verification chain on Intel Arc Pro B70 covering dense + MoE, `XPUGraph` changed-input replay, FP32 RNE SYCL/Level-Zero vs Triton parity, strict two-way safetensors header admission, and mutation-tested negative checks.
