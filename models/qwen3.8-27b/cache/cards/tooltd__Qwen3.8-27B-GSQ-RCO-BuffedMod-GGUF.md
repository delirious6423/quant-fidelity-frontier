---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: apache-2.0
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - multimodal
  - vision
  - buffed-mod
---



# Qwen3.8-27B-GSQ-RCO — BuffedMod GGUF

This repository contains modified versions of the **Qwen3.8-27B-GSQ-RCO-IQ3_S** GGUF model. 

Unlike traditional re-quantization workflows which require quantizing the entire model from unquantized FP16/BF16 weights, these files were created by **directly modifying specific tensor weights within the binary GGUF file**. 

By replacing critical layer tensors (specifically `output.weight`) with higher-precision quantizations, we achieve better Perplexity (PPL), lower KL Divergence (KLD), and increased top-token agreement while maintaining a compact footprint.

---

## ⚡ What Changed in BuffedMod?

In lower-bit quantizations like `IQ3_S`, the **`output.weight`** tensor (LM Head) often serves as a major bottleneck for final text generation precision. Upgrading this tensor improves top-token probability matching without adding massive overhead to the overall file size.

### 📦 Available Variants

1. **`Qwen3.8-27B-GSQ-RCO-BuffedMod-IQ3_S_XL-mtp-Q6K.gguf`**
   * **Modifications:** Replaced `output.weight` from `Q4_K` to **`Q6_K`**.
   * **Target:** Maximum quality retention for IQ3_S baseline.

2. **`Qwen3.8-27B-GSQ-RCO-BuffedMod-IQ3_S_XL-mtp-IQ4XS.gguf`**
   * **Modifications:** Replaced `output.weight` from `Q4_K` to **`Q6_K`** AND down-quantized MTP (Multi-Token Prediction) tensors from `Q6_K` to **`IQ4_XS`**.
   * **Target:** Optimized size-to-performance balance by reclaiming memory from non-essential auxiliary tensors.

---

## 📊 Benchmark & Evaluation Results

All variants were evaluated using `llama-perplexity` with original  **BF16 base model** (`q38f16baseline.kld`) to measure Perplexity (PPL) degradation, KL Divergence (KLD), and Token Probability shifts.

```bash
llama-perplexity.exe -f /wikitext-2-raw/wiki.test.raw \
    --kl-divergence \
    --kl-divergence-base q38f16baseline.kld \
    -ngl 99 \
    -m <model_variant>.gguf
```

### 📈 Comparison Summary

| Metric | Original (`IQ3_S`) | BuffedMod (`Q6_K` Output) | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean PPL(Q)** | `7.062697` | **`7.032142`** | 🟢 **-0.030555** (Lower is better) |
| **Mean PPL Diff vs Base** | `0.112205` | **`0.081650`** | 🟢 **-27.2% Error Reduction** |
| **Mean KLD** | `0.055475` | **`0.051738`** | 🟢 **-6.7% Divergence** |
| **Median KLD** | `0.022928` | **`0.018899`** | 🟢 **-17.5% Divergence** |
| **Same Top Token Probability** | `89.657%` | **`90.535%`** | 🟢 **+0.878% Match** |
| **RMS Δp** | `6.537%` | **`6.285%`** | 🟢 **-0.252% Variance** |

---

### 📐 Model Variant Comparison

The table below highlights the trade-off between file size efficiency and accuracy gains across the original and **BuffedMod** variants:

| Model Variant | File Size | PPL(Q) ↓ | PPL Diff vs Base ↓ | Mean KLD ↓ | Median KLD ↓ | Top-Token Match ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** (`IQ3_S-mtp`) | **11.29 GB** | 7.0627 | 0.1122 | 0.0555 | 0.0229 | 89.66% |
| **BuffedMod** (`IQ4XS MTP`) | **11.49 GB** | **7.0321** | **0.0817** | **0.0517** | **0.0189** | **90.54%** |
| **BuffedMod** (`Q6_K MTP`) | **11.59 GB** | **7.0321** | **0.0817** | **0.0517** | **0.0189** | **90.54%** |

> 📌 **Key Highlights:**
> * **`BuffedMod (IQ4XS MTP)`** offers the **optimal sweet spot**: it gains the full accuracy boost (+0.88% top-token match, -27.2% PPL error vs base) with only **+200 MB** size overhead compared to the original file.
> * **`BuffedMod (Q6_K MTP)`** retains maximum fidelity on the auxiliary MTP tensors for an extra **+100 MB**, maintaining identical core perplexity and divergence performance.

### 📋 Full Metrics Breakdown

#### 🔴 Baseline Original Metrics (`IQ3_S` with `Q4_K` output.weight)
```text
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.062697 ±   0.045484
Mean PPL(base)                :   6.950493 ±   0.044934
Cor(ln(PPL(Q)), ln(PPL(base))):  98.65%
Mean ln(PPL(Q)/PPL(base))     :   0.016014 ±   0.001059
Mean PPL(Q)/PPL(base)         :   1.016143 ±   0.001076
Mean PPL(Q)-PPL(base)         :   0.112205 ±   0.007438

====== KL divergence statistics ======
Mean    KLD:   0.055475 ±   0.000453
Maximum KLD:  15.811345
99.9%   KLD:   2.067696
99.0%   KLD:   0.593792
95.0%   KLD:   0.188752
90.0%   KLD:   0.110145
Median  KLD:   0.022928

====== Token probability statistics ======
Mean    Δp: -0.513 ± 0.017 %
RMS Δp    :  6.537 ± 0.051 %
Same top p: 89.657 ± 0.079 %

```

#### 🟢 BuffedMod Metrics (`IQ3_S` with `Q6_K` output.weight)

```text
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.032142 ±   0.045271
Mean PPL(base)                :   6.950493 ±   0.044934
Cor(ln(PPL(Q)), ln(PPL(base))):  98.72%
Mean ln(PPL(Q)/PPL(base))     :   0.011679 ±   0.001034
Mean PPL(Q)/PPL(base)         :   1.011747 ±   0.001046
Mean PPL(Q)-PPL(base)         :   0.081650 ±   0.007236

====== KL divergence statistics ======
Mean    KLD:   0.051738 ±   0.000450
Maximum KLD:  15.754954
99.9%   KLD:   2.056773
99.0%   KLD:   0.586696
95.0%   KLD:   0.181908
90.0%   KLD:   0.103020
Median  KLD:   0.018899

====== Token probability statistics ======
Mean    Δp: -0.477 ± 0.016 %
RMS Δp    :  6.285 ± 0.052 %
Same top p: 90.535 ± 0.076 %

```


## 📜 Credits & Acknowledgments

* **Original Model:** [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
* **Modification:** Tweak & Direct Tensor Modification by **tooltd**


