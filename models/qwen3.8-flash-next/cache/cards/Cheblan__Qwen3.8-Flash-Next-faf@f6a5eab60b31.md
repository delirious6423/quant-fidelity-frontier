---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: fastasfuck
pipeline_tag: image-text-to-text
tags:
  - fastasfuck
  - strix-halo
  - amd
  - gfx1151
  - rdna3.5
  - moe
  - qwen3_8
  - gptq
  - actorder
  - speculative-decoding
  - mtp
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next (FastAsFuck `.faf` — Calibrated FP64 ActOrder GPTQ Release)

<p align="left">
  <a href="https://huggingface.co/collections/Cheblan/fastasfuck-quantizations"><img alt="Collection: fastasfuck quantizations" src="https://img.shields.io/badge/collection-fastasfuck%20quantizations-ffcc4d?style=flat-square"></a>
  <a href="https://github.com/Cheblan/fastasfuck"><img alt="Engine: fastasfuck" src="https://img.shields.io/badge/engine-fastasfuck-blue?style=flat-square"></a>
</p>

Pre-quantized production release of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (`125B total / ~6B active`, 48 hybrid layers: 36 GDN + 12 QSA, 512 routed experts top-10 + 1 shared expert, `mtp.layers.0`) in `.faf` binary tensor format, engineered specifically for the **[fastasfuck](https://github.com/Cheblan/fastasfuck)** inference engine on AMD Strix Halo unified memory (`gfx1151`, AMD Ryzen AI Max+ 395, 128 GB Unified LPDDR5X-8533).

This release upgrades `weights.faf` from uncalibrated Round-to-Nearest (RTN) to **full-rank 61,440-token FP64 ActOrder GPTQ with Empirical Bayes shrinkage across all 512 experts**, reducing held-out KL divergence against unquantized BF16 by **`-29.6%`** (`0.030376 → 0.021393`) with **exact byte size parity (`70,337,409,024 B` / `65.51 GiB`) and zero decode/prefill speed penalty**.

---

## 1. What Was Done in the Calibrated GPTQ Upgrade

All quantization, ablation, and benchmarking scripts live in the **[Cheblan/fastasfuck](https://github.com/Cheblan/fastasfuck)** GitHub repository (`tools/quantize_qwen_gptq.py`, `tools/sim_*.py`, `tools/bench/`). Lightweight JSON telemetry & evaluation reports from calibrating this exact checkpoint are included here under `calibration/`.

### Precision Budget & Zero-Overhead Layout (`static_groups=True`)
```text
exp_gu=q4g128, exp_down=q4g128, gdn_in=q4g128 (GPTQ ActOrder calibrated)
gdn_out=q6g64, attn_qkv=q6g64, attn_o=q6g64, sh_gu=q6g64, sh_down=q6g64, ple_kv=q6g64 (RTN Q6_G64)
hc_down=q8g64, hc_up=q8g64, lm_head=q8g64 (RTN Q8_G64) | mtp_fc=q4g64 | embed=bf16
```
- **Why target only the three `q4g128` classes (`63.54 / 65.51 GiB`):** Per-tensor measurements show `q4g128` (16 levels) carries `~13.2%–15.9%` relative RMS error and **>95% of total quantization variance**, whereas `q6g64` (`~2.5%–2.9%` RMS) and `q8g64` (`~0.56%–0.83%` RMS) are already `30×–250×` lower variance. Replacing `Q4/Q6/Q8` with `Q5` is rejected because `Q5` misses RDNA 3.5's fast `F16` WMMA path (`-5.9%` prefill throughput) and enlarges the file.
- **Zero runtime speed penalty (`static_groups=True`):** Cholesky/ActOrder column error propagation re-solves optimal weights in activation importance order while keeping static contiguous `[0..127], [128..255], ...` quantization grids (`g_idx`-free). The packed `weights.faf` (`70,337,409,024 B`) executes identical `linear_q4.hip` / `linear_wmma.hip` kernels with zero gather overhead.
- **Prompt cache isolation:** Embedded `.faf` header includes `prebuilt_fingerprint = 1c74e30ce7e5a942` and calibration JSON metadata so `fastasfuck_server` automatically isolates calibrated `.pcache` and `mtp_calib_cache.json` profiles from legacy RTN builds.

### Mathematical & Numerical Engineering Highlights
1. **Disjoint 61,440-token Calibration Corpus (`sha256_16 = ddc381dad2eaae6a`):** `40 × 1536 = 61,440` tokens across software engineering issues, multi-language code, systems documentation, and Russian math/tool traces (`swebench: 12`, `code: 12`, `docs: 8`, `ru_math_tools: 2`, `swebench_extra: 6`), programmatically verified to have **zero 50-char overlap** with held-out `CALIB_DOC` and benchmark prompts.
2. **Strict FP64 Hessian Accumulation (`.double()`):** Because outlier channels in `u` reach RMS ~266 (`diag(X^T X) ~ 71,000`), accumulating `2560 × 2560` Hessians in FP32 degrades Cholesky conditioning and flips ActOrder permutations (`+0.0039` KLD penalty in ThinkingCap ablations). All dense Hessians (`H_gdn`, `H_mlp`) are accumulated and solved strictly in `float64` on CPU.
3. **Layer-Shared `H_mlp` for `exp_gu` (`512 × [1280, 2560]`):** Because individual expert routing counts (`hits_med = 640.2` mean per layer, `< 2,560` input dims) are rank-deficient, all 512 experts use the full-rank layer input Hessian $H_{\text{mlp}}$ (`61,440 ≫ 2,560`, `damp = 0.05`).
4. **Normalized Empirical Bayes Shrinkage for `exp_down` (`512 × [2560, 640]`):** Combines each expert's per-token normalized SwiGLU covariance $H_e$ with the layer-chunk mean covariance $H_{\text{mean}}$ (`tau = 384`, `damp = 0.05`) across **all 512 experts** (no hit-threshold cutoff, stabilizing even rare/zero-hit specialists):
   $$H_{e,\text{reg}} = \frac{\text{hits}_e \cdot H_e + \tau \cdot H_{\text{mean}}}{\text{hits}_e + \tau}$$
5. **Measured Hessian-Weighted Output MSE Reduction on Qwen3.8-Flash-Next (`calibration/layer_stats.json`):**
   - `gdn_in` (`q4g128`, 36 GDN layers): **`0.5040×` of RTN** (**`-49.6%`**, up to `-69.5%` / `0.3047×` on Layer 0)
   - `exp_gu` & `exp_down` (`q4g128`, 48 × 512 experts): calibrated using the identical full-rank `H_mlp` + Shrink512 recipe (`-48.7%` / `-41.5%` MSE in Qwen 3.8-FN reference telemetry)
6. **Methods Tested & Rejected via Whole-Model Ablation (`docs/WEIGHTS.md`):**
   - *AWQ / Imatrix scaling (`q4g128`)*: degraded held-out KLD (`+7.1%` to `+31.6%` worse than RTN) due to naive column rescaling hurting static grid groups.
   - *Coordinate Descent + BF16 Refit & Asymmetric GPTAQ*: cut calibration MSE by `-5.5%` on the 61k corpus but overfit held-out evaluation (`KLD 0.0283–0.0299` vs `0.0214–0.0216`).
   - *Calibrating `mtp.layers.0`*: verified unneeded — native RTN `mtp.layers.0` already closely tracks the unquantized BF16 MTP ceiling (`84.82%` vs `84.65%` own-trunk Top-1; `84.27%` vs BF16 on both GPTQ and BF16 MTP).

---

## 2. Quality Evaluation vs. Unquantized BF16 & Baseline RTN

### Full 48-Layer Trunk Held-Out Quality (`CALIB_DOC`, `T_eval = 1,807` tokens)

| Model / Variant | Mean KLD vs BF16 | Δ vs RTN | Med KLD | p95 KLD | Max KLD | PPL | Δ PPL vs BF16 | Top-1 vs BF16 | Top-5 vs BF16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BF16 Reference (Unquantized)** | `0.000000` | — | `0.000000` | `0.000000` | `0.000` | **`3.0119`** | `+0.0000` | **`100.00%`** | **`100.00%`** |
| `Baseline RTN .faf` (Old Prod) | `0.030376` | `0.0%` | `0.001869` | `0.111297` | `3.596` | `3.0327` | `+0.0208` | `94.74%` | `99.78%` |
| **Shipped: `FP64 ActOrder GPTQ` (`weights.faf`)** | **`0.021393`** | **`-29.6%`** | **`0.001626`** | **`0.084880`** | **`1.539`** | `3.0508` | `+0.0389` | **`95.40%`** | **`99.94%`** |

- **Key quality wins over baseline RTN on Qwen3.8-Flash-Next:** **`-29.6%`** lower Mean KLD vs BF16 (`0.030376 → 0.021393`), **`-23.7%`** lower p95 tail divergence (`0.1113 → 0.0849`), **`-57.2%`** lower Max KLD spike (`3.596 → 1.539`), **`+0.66 pp`** higher greedy Top-1 token agreement with BF16 (`94.74% → 95.40%`), and `99.94%` Top-5 agreement (`calibration/eval_kld_ppl_mtp.json`).

### Speculative Draft Head (`mtp.layers.0`) Identical-Prefix Agreement (`CALIB_DOC`, `N = 1,805`)

| Trunk + MTP Head Pair | Top-1 vs Own Trunk | Top-3 vs Own Trunk | `KLD(Trunk \|\| MTP)` | Top-1 vs BF16 | `MTP PPL` | Margin ≥ 1.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M_BF16` on `BF16` Trunk** (ceiling) | `84.65%` | `93.52%` | `0.390214` | `84.65%` | `3.375` | `74.07%` |
| `M0_RTN` on Old `RTN` Trunk (`lm q8 / q4`) | `85.10% / 84.10%` | `93.74% / 93.57%` | `0.386665 / 0.395332` | `83.82% / 83.05%` | `3.515 / 3.551` | `74.13% / 74.85%` |
| **`M0_RTN` on `GPTQ` Trunk (`lm q8 / q4`)** | `84.82% / 84.38%` | `93.41% / 93.24%` | `0.401665 / 0.408792` | **`84.27% / 83.55%`** | **`3.456 / 3.489`** | `73.13% / 73.57%` |

---

## 3. Hardware Performance (AMD Strix Halo `gfx1151` Ground Truth)

All hardware A/B benchmarks executed back-to-back on bare-metal **AMD Ryzen AI Max+ 395** (40 CUs RDNA 3.5 `gfx1151`, 128 GB LPDDR5X-8533 @ 273 GB/s) under sterile `prompt-cache = off` (`cache_n = 0`, untimed warmup + `100.000%` SHA-256 verification) using the identical `70,337,409,024 B` `qwen4exp` binary layout and HIP kernels (`ThinkingCap-Qwen3.8-FN` family reference).

| Workload / Metric | Old RTN `weights.faf` | **New GPTQ `weights.faf`** | Delta |
| :--- | :---: | :---: | :--- |
| Binary File Size | `70,337,409,024 B` | `70,337,409,024 B` | **`0 B` (`0.00%`)** |
| **Solo Bare Decode** (`n=5`, `T=0.0`) | `39.88 tok/s` (p50 `39.89`) | `39.82 tok/s` (p50 `39.90`) | **Exact parity** (`p50 +0.01`) |
| **Prefill 4k Warm** (`n=5`) | `1709.0 ± 3.5 tok/s` | `1708.6 ± 0.7 tok/s` | **Parity** (`-0.02%`) |
| **Prefill 8k Warm** (`n=5`) | `1775.0 ± 0.6 tok/s` | `1772.2 ± 0.5 tok/s` | **Parity** (`-0.16%`) |
| `verify_base_ms` (`mtp_calib_cache`) | `25.268 ms` | **`25.256 ms`** | **`-0.05%`** (parity) |
| **30-Prompt Diverse Solo MTP (`T=0.0`, `N=30`)** | `73.65 ± 2.03 tok/s` (p50 `70.47`) | **`74.16 ± 2.04 tok/s`** (p50 **`71.72`**) | **`+0.51 tok/s` (`16/14` wins)** |
| 30-Prompt Global MTP Acceptance Rate | `81.22%` (`4934 / 6075`) | **`81.59%`** (`4760 / 5834`) | **`+0.37 pp`** |
| **12-Prompt Diverse Solo MTP (`T=0.6`, `N=12`)** | `68.43 ± 4.57 tok/s` | **`71.81 ± 3.96 tok/s`** | **`+3.38 tok/s` (`10/2` wins)** |
| 12-Prompt (`T=0.6`) MTP Acceptance Rate | `56.14%` (`2006 / 3573`) | **`59.55%`** (`2082 / 3496`) | **`+3.41 pp`** |
| **4-Slot Concurrent MTP (`T=0.6`, `n=3`)** | Agg `93.10 ± 9.23` (`PerSlot 30.80 ± 2.51`) | Agg `91.82 ± 5.39` (**`PerSlot 31.28 ± 1.01`**) | **`PerSlot +1.56%`**, much lower variance |
| 4-Slot MTP Acceptance Rate | `60.14%` (`1949 / 3241`) | **`60.59%`** (`1951 / 3220`) | **`+0.45 pp`** |
| **Bit-Identity (`T=0.0`, `5/5` SHA-256)** | `100.000% PASS` | **`100.000% PASS`** | Exact (`100.000%` deterministic) |

---

## 4. Included Artifacts & Files

| File | Size | Description |
|---|---|---|
| `weights.faf` | `65.51 GiB` | FP64 ActOrder GPTQ (`q4g128`) + Hybrid `q6g64`/`q8g64` quantized base weights (`prebuilt_fingerprint: 1c74e30ce7e5a942`). |
| `ngram.table` | `48.28 GiB` | 8,388,608-row int8 Prefix Lookahead Engine (PLE) speculative table. |
| `vision.faf` | `856.7 MiB` | Native multimodal vision projection tower for image input. |
| `tokenizer.json` | `12.2 MiB` | Fast Hugging Face BPE tokenizer vocabulary. |
| `config.json` | `4.7 KiB` | Model architecture configuration and dimensions. |
| `generation_config.json` | `0.2 KiB` | Default sampling parameters and generation configuration. |
| `chat_template.jinja` | `8.7 KiB` | Jinja2 chat template supporting tool calls and thinking blocks. |
| `convert.log` | `172 KiB` | Conversion log from Safetensors + prebuilt GPTQ tensors to `.faf`. |
| `convert_ngram.log` | `1.7 KiB` | Compilation log of the 8M-row int8 PLE n-gram table. |
| `inspect_weights.txt` | `101 KiB` | Full tensor layout, `calibration` header, and `prebuilt_fingerprint` inspection. |
| `inspect_vision.txt` | `32 KiB` | Vision projection tower tensor inspection and layout metadata. |
| `calibration/*.json` | `~15 KiB` | Reproducibility JSON telemetry reports (`calib_meta.json`, `calib_manifest.json`, `layer_stats.json`, `eval_kld_ppl_mtp.json`). Scripts live on GitHub in [Cheblan/fastasfuck](https://github.com/Cheblan/fastasfuck). |

---

## 5. Quick Start with `fastasfuck`

### Step 1: Download Model Files
```bash
hf download Cheblan/Qwen3.8-Flash-Next-faf --local-dir /models/Qwen-3.8-FN-faf
```

### Step 2: Configure Server (`deploy/fastasfuck-server.conf`)
Point the configuration file (or CLI flags) to the downloaded model directory:

```ini
model-id = qwen3.8-flash-next
weights = /models/Qwen-3.8-FN-faf/weights.faf
ngram = /models/Qwen-3.8-FN-faf/ngram.table
tokenizer = /models/Qwen-3.8-FN-faf/tokenizer.json
generation-config = /models/Qwen-3.8-FN-faf/generation_config.json
```

Start the inference server:
```bash
# Native binary build
build/strix/fastasfuck_server --config deploy/fastasfuck-server.conf

# Or via Podman / Docker
podman run -d --device=/dev/kfd --device=/dev/dri -p 5300:5300 \
  -v /models/Qwen-3.8-FN-faf:/models:ro \
  ghcr.io/cheblan/fastasfuck:latest --config /models/fastasfuck-server.conf
```

### Step 3: Query via OpenAI API or Responses API
```bash
curl -s http://localhost:5300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain the Riemann-Lebesgue Lemma"}], "temperature": 0.0}'
```

Compatible with **Codex CLI**, **OpenCode 2**, and **Cursor** using `/v1/responses` with zero-overhead KV cache reuse across multi-turn sessions.

---

## 6. Attribution & Licenses

- **Quantized and Packaged by:** [Cheblan / fastasfuck team](https://github.com/Cheblan/fastasfuck).
- **Original Model Weights:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen Team.
- **Base Foundation Model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen Team.
- **License:** Governed by the [Qwen Community License 1.0](LICENSE).
