---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: IsValorum
library_name: gguf
license: apache-2.0
language:
- en
tags:
- gguf
- llama.cpp
- quantized
- val-apex-i
- qwen3.8
- imatrix
- reasoning
- v3
pipeline_tag: text-generation
---

# Qwen3.8-27B Base VAL-APEX-I MiniPlus V3 GGUF

<a href="https://ko-fi.com/isvalorum"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi" height="38"></a>

**Independently computed using dedicated cloud compute clusters, funded 100% out-of-pocket.**

If you consider these releases to be of high quality and they help you in your day-to-day workflow, consider [supporting on Ko-fi](https://ko-fi.com/isvalorum). Every contribution directly sustains future cloud compute hours and helps support my university studies.

### *Original Qwen checkpoint · Revised V3 tensor allocation · Native 256K context*

Official **VAL-APEX-I MiniPlus V3** quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**. This edition quantizes the original Qwen checkpoint without applying Huihui abliteration or EfficientThink fine-tuning.

**Integrated Multi-Token Prediction (MTP):** This GGUF includes one MTP block (`blk.64.*`, 15 tensors) in the same file as the main language model. Its eight matrices use `Q4_0` ×6, `Q4_1` ×1 and `Q6_K` ×1; its seven normalization tensors remain `F32`. MTP speculative decoding depends on support and configuration in the inference backend.

VAL-APEX-I stands for:
**Vector-calibrated Asymmetric Layer-wise Outlier-preserving Recurrent-aware Unified Matrix-quantization**

> [!NOTE]
> ### EXPLORE THE QWEN3.8 27B VAL-APEX-I EDITIONS
> Choose a model variant and quantization profile:
>
> - **[Qwen3.8 27B — NanoPlus V3](https://huggingface.co/IsValorum/Qwen3.8-27B-VAL-APEX-I-NanoPlus-V3-GGUF)** — 12.18 GB.
> - **[Swift-1.5 Qwen3.8 27B Uncensored MTP — MiniPlus V3](https://huggingface.co/IsValorum/Swift-1.5-Qwen3.8-27B-Uncensored-MTP-VAL-APEX-I-MiniPlus-V3-GGUF)** — 14.16 GB.
> - **[Swift-1.5 Qwen3.8 27B Uncensored MTP — NanoPlus V3](https://huggingface.co/IsValorum/Swift-1.5-Qwen3.8-27B-Uncensored-MTP-VAL-APEX-I-NanoPlus-V3-GGUF)** — 12.18 GB.
> Collections: **[MiniPlus Abliterated 27B](https://huggingface.co/collections/IsValorum/val-apex-miniplus-abliterated-27b-6acfec541ee5628d2632c663)** · **[MiniPlus Standard 27B](https://huggingface.co/collections/IsValorum/val-apex-miniplus-standard-27b-6acfec54ca4ce4cf5cd5bdb6)** · **[NanoPlus Abliterated 27B](https://huggingface.co/collections/IsValorum/val-apex-nanoplus-abliterated-27b-6acfec5503f6e74218f71f64)** · **[NanoPlus Standard 27B](https://huggingface.co/collections/IsValorum/val-apex-nanoplus-standard-27b-6acfec55c1c99e2a23ebbd9b)**

> [!IMPORTANT]
> ### HIGH-FIDELITY 14.16GB BUILD
> This V3 release uses a revised per-tensor allocation with an actual **14.16 GB** file size and **4.15 whole-file BPW**. Its reported WikiText-2 PPL is **6.1361 ± 0.4885**, or **+0.3419 (+5.90%)** relative to the measured unmodified BF16 Base reference. The audit below records every tensor family, the block-0 exceptions and the integrated MTP head.

---

## <a id="quick-navigation"></a>Quick Navigation Index
1. [Quantization Metrics & Tensor Map](#toc-comparison)
2. [Independent Community Benchmarks (NanoBench v0.4.0)](#toc-nanobench)
3. [Model Files & Technical Specifications](#toc-04)
4. [Native Context & Runtime Memory](#toc-06)
5. [Recommended Configuration & Setup](#toc-07)
6. [Recommended Generation Parameters](#toc-08)
7. [Native Chat Template & Reasoning](#toc-chat-template)
8. [Community Compute Fund & Student Support](#toc-09)

---

<a id="toc-comparison"></a>
## 1. Quantization Metrics

### Size & Quality Metrics

| Edition / quantization | File size | Whole-file BPW | WikiText-2 PPL | Δ PPL vs BF16 Base<br>(5.7942) |
| :--- | :--- | :--- | :--- | :--- |
| **BF16 Base baseline (unmodified)** | 54.65 GB<br>(50.90 GiB) | 16-bit | **5.7942 ± 0.4531** | 0.0000 (0.00%) |
| **Qwen3.8 27B Base — MiniPlus V3 (this release)** | 14.16 GB<br>(13.19 GiB) | 4.15 | 6.1361 ± 0.4885 | +0.3419 (+5.90%) |

> [!NOTE]
> **Independent Hardware & Compute Transparency:**
> Generating multi-million token calibration matrices and running certified perplexity evaluations on local architectures requires dedicated high-end cloud nodes funded 100% out-of-pocket. If you rely on these MiniPlus/NanoPlus builds for your local workloads, contributions to the **[IsValorum Compute Fund on Ko-fi](https://ko-fi.com/isvalorum)** directly sponsor the GPU hours required for future Day-0 handcrafted quantizations.



<a id="toc-nanobench"></a>
### Independent Community Benchmarks (NanoBench v0.4.0)

Independent multi-harness empirical evaluation conducted by community member **[@userr99](https://huggingface.co/userr99)** on a dedicated local workstation (24 GB VRAM / 64 GB RAM using `llama.cpp`). Evaluated across the 12-task autonomous coding and agentic suite on [NanoBench v0.4.0](https://shortbench.vercel.app/) ([public discussion thread & test logs](https://huggingface.co/IsValorum/Huihui-Qwen3.8-27B-Abliterated-VAL-APEX-I-NanoPlus-V3-GGUF/discussions/2)):

| Test Harness | Total Score | Solved Tasks | Agentic (Tool Use) | Coding (Implementation) | Systems (Integrity) | Avg. Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oh My Pi (OMP 18.7)** | **96.7 / 100** | 10 / 12 | **100 / 100** | **100 / 100** | 86.7 / 100 | aprox. 1h 47m |
| **Hermes Agent (v0.21.5)** | **95.4 / 100** | 9 / 12 | **100 / 100** | 97.0 / 100 | 86.7 / 100 | aprox. 1h 47m |
| **DeepSeek Harness (0.2.0-rc.2)** | **90.4 / 100** | 9 / 12 | **100 / 100** | 97.0 / 100 | 66.7 / 100 | aprox. 1h 47m |

**Key Architectural Insights:**
- **Zero Tool-Calling Degradation:** Achieved a unanimous **100/100 in Agentic tasks** across all three harnesses, confirming that protecting `output.weight` in `Q6_K` and DeltaNet recurrent memory states (`ssm_a` in `F32`, `ssm_beta` in `Q8_0`) prevents JSON schema rejections and state drift over multi-step workflows.
- **Universal Harness Compatibility:** Surpassed the 90+ score barrier across all three test harnesses without entering infinite retry loops or syntax divergence.

### Tensor Precision Map

| Component | MiniPlus V3 | NanoPlus V3 | GSQ-RCO IQ3_S |
| :--- | :--- | :--- | :--- |
| Output head | `Q6_K` ×1 | `Q6_K` ×1 | `Q4_K` ×1 |
| Token embeddings | `Q4_K` ×1 | `Q3_K` ×1 | `IQ2_S` ×1 |
| Normalizations | `F32` ×161 | `F32` ×161 | `F32` ×161 |
| SSM A (`ssm_a`) | `F32` ×48 | `F32` ×48 | `F32` ×48 |
| SSM convolution (`ssm_conv1d`) | `F32` ×48 | `F32` ×48 | `F32` ×48 |
| SSM time-step bias (`ssm_dt`) | `F32` ×48 | `F32` ×48 | `F32` ×48 |
| SSM norm (`ssm_norm`) | `F32` ×48 | `F32` ×48 | `F32` ×48 |
| Attention gates (`attn_gate`) | `Q4_0` ×1<br>`Q4_K` ×47 | `Q4_0` ×1<br>`Q4_K` ×47 | `IQ2_S` ×2<br>`IQ3_S` ×18<br>`IQ3_XXS` ×9<br>`IQ4_XS` ×12<br>`Q2_K` ×4<br>`Q4_K` ×3 |
| Linear QKV (`attn_qkv`) | `Q4_0` ×1<br>`Q4_K` ×47 | `Q3_K` ×47<br>`Q4_0` ×1 | `IQ2_XS` ×1<br>`IQ2_XXS` ×1<br>`IQ3_S` ×22<br>`IQ3_XXS` ×13<br>`IQ4_XS` ×9<br>`Q2_K` ×1<br>`Q4_K` ×1 |
| SSM alpha (`ssm_alpha`) | `F32` ×47<br>`Q4_0` ×1 | `F32` ×47<br>`Q4_0` ×1 | `BF16` ×48 |
| SSM beta (`ssm_beta`) | `Q4_0` ×1<br>`Q8_0` ×47 | `Q4_0` ×1<br>`Q8_0` ×47 | `BF16` ×48 |
| SSM output (`ssm_out`) | `Q4_0` ×1<br>`Q5_K` ×47 | `Q4_0` ×1<br>`Q4_K` ×47 | `IQ3_S` ×22<br>`IQ3_XXS` ×4<br>`IQ4_XS` ×16<br>`Q4_K` ×6 |
| Full attention Q (`attn_q`) | `Q4_K` ×16 | `Q3_K` ×16 | `IQ2_XXS` ×1<br>`IQ3_S` ×3<br>`IQ3_XXS` ×3<br>`IQ4_XS` ×2<br>`Q2_K` ×6<br>`Q4_K` ×1 |
| Full attention K (`attn_k`) | `Q5_K` ×16 | `Q4_K` ×16 | `IQ2_S` ×1<br>`IQ3_S` ×1<br>`IQ3_XXS` ×1<br>`IQ4_XS` ×8<br>`Q4_K` ×5 |
| Full attention V (`attn_v`) | `Q6_K` ×16 | `Q5_K` ×16 | `IQ3_S` ×6<br>`IQ3_XXS` ×1<br>`IQ4_XS` ×1<br>`Q4_K` ×8 |
| Full attention output (`attn_output`) | `Q6_K` ×16 | `Q6_K` ×16 | `IQ3_S` ×10<br>`IQ3_XXS` ×1<br>`IQ4_XS` ×2<br>`Q4_K` ×3 |
| MLP down (`ffn_down`) | `IQ4_XS` ×64 | `IQ3_XXS` ×48<br>`IQ4_XS` ×16 | `IQ2_S` ×4<br>`IQ2_XS` ×3<br>`IQ3_S` ×22<br>`IQ3_XXS` ×7<br>`IQ4_XS` ×21<br>`Q2_K` ×1<br>`Q4_K` ×6 |
| MLP gate (`ffn_gate`) | `IQ3_S` ×32<br>`IQ3_XXS` ×32 | `IQ2_XXS` ×48<br>`IQ3_XXS` ×16 | `IQ1_M` ×1<br>`IQ2_S` ×4<br>`IQ2_XS` ×4<br>`IQ2_XXS` ×1<br>`IQ3_S` ×15<br>`IQ3_XXS` ×21<br>`IQ4_XS` ×15<br>`Q2_K` ×1<br>`Q4_K` ×2 |
| MLP up (`ffn_up`) | `IQ3_S` ×32<br>`IQ3_XXS` ×32 | `IQ2_XXS` ×48<br>`IQ3_XXS` ×16 | `IQ2_S` ×5<br>`IQ2_XS` ×1<br>`IQ2_XXS` ×2<br>`IQ3_S` ×25<br>`IQ3_XXS` ×18<br>`IQ4_XS` ×10<br>`Q4_K` ×3 |
| Additional MTP head (`blk.64.*`) | `F32` ×7<br>`Q4_0` ×6<br>`Q4_1` ×1<br>`Q6_K` ×1 | `F32` ×7<br>`Q4_0` ×6<br>`Q4_1` ×1<br>`Q6_K` ×1 | — |

× is the number of tensors in each format. Counts were read from the released GGUF headers; the GSQ-RCO column uses its published tensor allocation.

Both V3 files contain **866 tensors**: **851 main-model tensors** plus **15 tensors in the additional MTP head**. The 64 main layers are **48 Gated DeltaNet + 16 full-attention**; `blk.64.*` is the MTP head, not a seventeenth main full-attention layer.

**Fallback exceptions:** `blk.0.attn_gate.weight`, `blk.0.attn_qkv.weight`, `blk.0.ssm_alpha.weight`, `blk.0.ssm_beta.weight` and `blk.0.ssm_out.weight` are `Q4_0` in both editions. In the MTP head, Q/K/V, MLP gate/up and `nextn.eh_proj` are `Q4_0`; MLP down is `Q4_1`, attention output is `Q6_K`, and the remaining seven tensors are `F32`.

**V3 layer allocation:** MiniPlus uses `IQ3_S` for MLP gate/up in layers 0–15 and 48–63, and `IQ3_XXS` in layers 16–47; every main MLP down tensor is `IQ4_XS`. NanoPlus uses `IQ3_XXS` for gate/up in layers 0–7 and 56–63, and `IQ2_XXS` in layers 8–55; MLP down is `IQ4_XS` in the outer 16 layers and `IQ3_XXS` in the other 48.

[Complete tensor-by-tensor audit: names, formats, shapes and parameter counts](./tensor-map.tsv).

---

<a id="toc-04"></a>
## 2. Model Files & Technical Specifications

| File | File size | Whole-file BPW | Contents |
| :--- | :--- | :--- | :--- |
| `Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf` | **14.16 GB (13.19 GiB)** | **4.15 BPW** | Main language model and integrated MTP head |

- **Source:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- **Architecture:** `qwen35`; 64 main layers (48 Gated DeltaNet + 16 full-attention) plus one MTP block.
- **Stored parameters:** 27,320,697,856, including MTP tensors.
- **Tensor count:** 866, including the 15-tensor MTP head.
- **Native context:** 262,144 tokens.
- **Calibration:** 1,000 chunks × 4,096 tokens (4,096,000 tokens), as reported by the author. The GGUF records 1,000 calibration chunks and 496 importance-matrix entries.
- **Scope:** This repository currently contains the language GGUF. No companion vision projector is included.

Whole-file BPW is `8 × file bytes / stored parameter count`, including the file's metadata and MTP head. It is not a uniform tensor quantization type.

---

<a id="toc-06"></a>
## 3. Native Context & Runtime Memory

The model's native context is 262,144 tokens. File size describes the stored weights; runtime memory also includes the KV cache, recurrent state, compute buffers, batch settings and any optional components. Choose context and offload settings according to the allocations reported by your runtime.

---

<a id="toc-07"></a>
## 4. Recommended Configuration & Setup

### llama.cpp Server

```bash
llama-server \
  -m Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf \
  --jinja \
  -ngl 99 \
  -c 8192 \
  --host 127.0.0.1 \
  --port 8080
```

The command requests GPU offload; adjust offload and context to the available memory. In LM Studio, load the GGUF and retain its embedded Qwen chat template. The integrated MTP tensors are listed in the audit; speculative decoding requires explicit support and configuration in the chosen backend.

---

<a id="toc-08"></a>
## 5. Recommended Generation Parameters

These are the upstream Qwen defaults, distinct from the WikiText-2 evaluation settings.

| Setting | Thinking | Non-thinking |
| :--- | :--- | :--- |
| Temperature | 1.0 | 0.7 |
| Top-P | 0.95 | 0.80 |
| Top-K | 20 | 20 |
| Min-P | 0.0 | 0.0 |
| Presence penalty | 0.0 | 1.5 |
| Repetition penalty | 1.0 | 1.0 |

---

<a id="toc-chat-template"></a>
## 6. Native Chat Template & Reasoning

The native Qwen Jinja chat template is embedded in this GGUF. Use `--jinja` where supported and preserve the native formatting for messages, tool calls and thinking blocks. Reasoning controls depend on the backend's implementation of the template arguments.

---

<a id="toc-09"></a>

## 8. Community Compute Fund & Student Support

<a href="https://ko-fi.com/isvalorum"><img src="https://huggingface.co/spaces/IsValorum/MiniPlus-NanoPlus-Requests/resolve/main/assets/dance-gold-ship.gif" alt="Gold Ship dancing" width="128" align="right"></a>

All IsValorum quantizations will always remain completely free and open to the public without paywalls. 

However, cloud GPU compute is expensive and funded entirely out-of-pocket. If you find these handcrafted builds to be of high quality and they benefit your day-to-day workflow, you can support future Day-0 releases on Ko-fi. Every contribution directly funds the cloud compute hours required for new model runs and helps support my university education:

<a href="https://ko-fi.com/isvalorum"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi" height="38"></a>

*(When supporting on Ko-fi, feel free to leave a note with your Hugging Face handle and the specific model you would like prioritized).*

<div style="clear: both;"></div>