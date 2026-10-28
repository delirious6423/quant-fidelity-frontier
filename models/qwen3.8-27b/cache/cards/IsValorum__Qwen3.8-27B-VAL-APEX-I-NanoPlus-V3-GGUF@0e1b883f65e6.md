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

# Qwen3.8-27B Base VAL-APEX-I NanoPlus V3 GGUF

<a href="https://ko-fi.com/isvalorum"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi" height="38"></a>

**Independently computed using dedicated cloud compute clusters, funded 100% out-of-pocket.**

If you consider these releases to be of high quality and they help you in your day-to-day workflow, consider [supporting on Ko-fi](https://ko-fi.com/isvalorum). Every contribution directly sustains future cloud compute hours and helps support my university studies.

### *Original Qwen checkpoint · Revised V3 tensor allocation · Native 256K context*

Official **VAL-APEX-I NanoPlus V3** quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**. This edition quantizes the original Qwen checkpoint without applying Huihui abliteration or EfficientThink fine-tuning.

**Integrated Multi-Token Prediction (MTP):** This GGUF includes one MTP block (`blk.64.*`, 15 tensors) in the same file as the main language model. Its eight matrices use `Q4_0` ×6, `Q4_1` ×1 and `Q5_K` ×1; its seven normalization tensors remain `F32`. MTP speculative decoding depends on support and configuration in the inference backend.

VAL-APEX-I stands for:
**Vector-calibrated Asymmetric Layer-wise Outlier-preserving Recurrent-aware Unified Matrix-quantization**

> [!NOTE]
> ### EXPLORE THE QWEN3.8 27B VAL-APEX-I EDITIONS
> Choose a model variant and quantization profile:
>
> - **[Qwen3.8 27B — MiniPlus V3](https://huggingface.co/IsValorum/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3-GGUF)** — 14.16 GB.
> - **[Swift-1.5 Qwen3.8 27B Uncensored MTP — MiniPlus V3](https://huggingface.co/IsValorum/Swift-1.5-Qwen3.8-27B-Uncensored-MTP-VAL-APEX-I-MiniPlus-V3-GGUF)** — 14.16 GB.
> - **[Swift-1.5 Qwen3.8 27B Uncensored MTP — NanoPlus V3](https://huggingface.co/IsValorum/Swift-1.5-Qwen3.8-27B-Uncensored-MTP-VAL-APEX-I-NanoPlus-V3-GGUF)** — 12.18 GB.
> Collections: **[MiniPlus Abliterated 27B](https://huggingface.co/collections/IsValorum/val-apex-miniplus-abliterated-27b-6acfec541ee5628d2632c663)** · **[MiniPlus Standard 27B](https://huggingface.co/collections/IsValorum/val-apex-miniplus-standard-27b-6acfec54ca4ce4cf5cd5bdb6)** · **[NanoPlus Abliterated 27B](https://huggingface.co/collections/IsValorum/val-apex-nanoplus-abliterated-27b-6acfec5503f6e74218f71f64)** · **[NanoPlus Standard 27B](https://huggingface.co/collections/IsValorum/val-apex-nanoplus-standard-27b-6acfec55c1c99e2a23ebbd9b)**

> [!IMPORTANT]
> ### COMPACT 12.18 GB BUILD
> This V3 release uses a revised per-tensor allocation with an actual **12.18 GB** file size and **3.57 whole-file BPW**. Its reported WikiText-2 PPL is **6.2577 ± 0.5011**, or **+0.4635 (+8.00%)** relative to the measured unmodified BF16 Base reference. The audit below records every tensor family, the block-0 exceptions and the integrated MTP head.

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
| **Qwen3.8 27B Base — NanoPlus V3 (this release)** | 12.18 GB<br>(11.35 GiB) | 3.57 | 6.2577 ± 0.5011 | +0.4635 (+8.00%) |

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
- **Zero Tool-Calling Degradation:** Achieved a unanimous **100/100 in Agentic tasks** across all three harnesses, confirming that protecting `output.weight` in `Q5_K` and DeltaNet recurrent memory states (`ssm_a` in `F32`, `ssm_beta` in `Q8_0`) prevents JSON schema rejections and state drift over multi-step workflows.
- **Universal Harness Compatibility:** Surpassed the 90+ score barrier across all three test harnesses without entering infinite retry loops or syntax divergence.

### Tensor Precision Map

| Component | MiniPlus V3 | NanoPlus V3 | GSQ-RCO IQ3_S |
| :--- | :--- | :--- | :--- |
| Output head | `Q6_K` x1 | `Q5_K` x1 | `Q4_K` x1 |
| Token embeddings | `Q4_K` x1 | `Q3_K` x1 | `IQ2_S` x1 |
| Normalizations | `F32` x161 | `F32` x161 | `F32` x161 |
| SSM A (`ssm_a`) | `F32` x48 | `F32` x48 | `F32` x48 |
| SSM convolution (`ssm_conv1d`) | `F32` x48 | `F32` x48 | `F32` x48 |
| SSM time-step bias (`ssm_dt`) | `F32` x48 | `F32` x48 | `F32` x48 |
| SSM norm (`ssm_norm`) | `F32` x48 | `F32` x48 | `F32` x48 |
| Attention gates (`attn_gate`) | `Q4_0` x1<br>`Q4_K` x47 | `Q4_0` x1<br>`Q4_K` x47 | `IQ2_S` x2<br>`IQ3_S` x18<br>`IQ3_XXS` x9<br>`IQ4_XS` x12<br>`Q2_K` x4<br>`Q4_K` x3 |
| Linear QKV (`attn_qkv`) | `Q4_0` x1<br>`Q4_K` x47 | `Q3_K` x47<br>`Q4_0` x1 | `IQ2_XS` x1<br>`IQ2_XXS` x1<br>`IQ3_S` x22<br>`IQ3_XXS` x13<br>`IQ4_XS` x9<br>`Q2_K` x1<br>`Q4_K` x1 |
| SSM alpha (`ssm_alpha`) | `F32` x48 | `F32` x48 | `BF16` x48 |
| SSM beta (`ssm_beta`) | `Q4_0` x1<br>`Q8_0` x47 | `Q4_0` x1<br>`Q8_0` x47 | `BF16` x48 |
| SSM output (`ssm_out`) | `Q4_0` x1<br>`Q5_K` x47 | `Q4_0` x1<br>`Q4_K` x47 | `IQ3_S` x22<br>`IQ3_XXS` x4<br>`IQ4_XS` x16<br>`Q4_K` x6 |
| Full attention Q (`attn_q`) | `Q4_K` x16 | `Q3_K` x16 | `IQ2_XXS` x1<br>`IQ3_S` x3<br>`IQ3_XXS` x3<br>`IQ4_XS` x2<br>`Q2_K` x6<br>`Q4_K` x1 |
| Full attention K (`attn_k`) | `Q5_K` x16 | `Q4_K` x16 | `IQ2_S` x1<br>`IQ3_S` x1<br>`IQ3_XXS` x1<br>`IQ4_XS` x8<br>`Q4_K` x5 |
| Full attention V (`attn_v`) | `Q6_K` x16 | `Q5_K` x16 | `IQ3_S` x6<br>`IQ3_XXS` x1<br>`IQ4_XS` x1<br>`Q4_K` x8 |
| Full attention output (`attn_output`) | `Q6_K` x16 | `Q5_K` x16 | `IQ3_S` x10<br>`IQ3_XXS` x1<br>`IQ4_XS` x2<br>`Q4_K` x3 |
| MLP down (`ffn_down`) | `IQ4_XS` x64 | `IQ3_XXS` x48<br>`IQ4_XS` x16 | `IQ2_S` x4<br>`IQ2_XS` x3<br>`IQ3_S` x22<br>`IQ3_XXS` x7<br>`IQ4_XS` x21<br>`Q2_K` x1<br>`Q4_K` x6 |
| MLP gate (`ffn_gate`) | `IQ3_S` x32<br>`IQ3_XXS` x32 | `IQ3_XXS` x64 | `IQ1_M` x1<br>`IQ2_S` x4<br>`IQ2_XS` x4<br>`IQ2_XXS` x1<br>`IQ3_S` x15<br>`IQ3_XXS` x21<br>`IQ4_XS` x15<br>`Q2_K` x1<br>`Q4_K` x2 |
| MLP up (`ffn_up`) | `IQ3_S` x32<br>`IQ3_XXS` x32 | `IQ3_XXS` x64 | `IQ2_S` x5<br>`IQ2_XS` x1<br>`IQ2_XXS` x2<br>`IQ3_S` x25<br>`IQ3_XXS` x18<br>`IQ4_XS` x10<br>`Q4_K` x3 |
| Additional MTP head (`blk.64.*`) | `F32` x7<br>`Q4_0` x6<br>`Q4_1` x1<br>`Q6_K` x1 | `F32` x7<br>`Q4_0` x6<br>`Q4_1` x1<br>`Q5_K` x1 | - |

x denotes tensor counts. All columns reflect exact GGUF tensor headers verified in `tensor-map.tsv`. Normalizations exclude separately listed SSM norms.

Both Swift/Base/Huihui files contain **866 audited tensors**: **851 main-model tensors** plus **15 MTP tensors**. The main model is a dense hybrid architecture with **64 layers: 48 Gated DeltaNet linear-attention layers and 16 full-attention layers**. The additional `blk.64.*` block is the integrated MTP head.

**V3 architectural allocation:** NanoPlus V3 features an elevated 3-bit floor across all SwiGLU MLP blocks (`ffn_gate` and `ffn_up` set uniformly to `IQ3_XXS` across all 64 layers; `ffn_down` uses `IQ4_XS` in outer layers 0-7 / 56-63 and `IQ3_XXS` in core layers 8-55). Output head (`output.weight`) is calibrated to `Q5_K` (5.50 bpw) to maximize token throughput and preserve dynamic range, entirely eliminating the linguistic decay in non-English contexts associated with 2-bit sub-allocations.

**Actual layer-0 formats:** `blk.0.attn_gate.weight`, `blk.0.attn_qkv.weight`, `blk.0.ssm_beta.weight` and `blk.0.ssm_out.weight` use `Q4_0`. All 48 `ssm_alpha` tensors remain `F32` for mathematical stability.

**Integrated MTP:** `blk.64.*` contains **424,699,392 stored parameters** in 15 tensors: seven `F32` tensors and eight matrices (`Q4_0` x6, `Q4_1` x1, `Q5_K` x1). Runtime acceleration depends on backend support and configuration; speculative decoding operates via native acceptance drafting.

[Complete tensor-by-tensor audit: names, formats, shapes and parameter counts](./tensor-map.tsv).


---

<a id="toc-04"></a>
## 2. Model Files & Technical Specifications

| File | File size | Whole-file BPW | Contents |
| :--- | :--- | :--- | :--- |
| `Qwen3.8-27B-VAL-APEX-I-NanoPlus-V3.gguf` | **12.18 GB (11.35 GiB)** | **3.57 BPW** | Main language model and integrated MTP head |

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
  -m Qwen3.8-27B-VAL-APEX-I-NanoPlus-V3.gguf \
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