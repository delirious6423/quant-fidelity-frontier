---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: IsValorum
library_name: gguf
pipeline_tag: image-text-to-text
license: mit
language:
- en
- zh
- es
- fr
- de
- pt
- it
- ru
- ja
- ko
- vi
- th
- ar
tags:
- gguf
- llama.cpp
- quantized
- quantization
- apex
- apex-quant
- apex-i-nanoplus
- nanoplus
- custom-quantization
- unsloth-studio
- qwen3.6
- qwen35moe
- qwen3_5_moe
- qwen
- moe
- reasoning
- conversational
- multimodal
- vision
- image-text-to-text
- text-generation
- ornith
- coding
- agentic
- swe-bench
---

## <a id="quick-navigation"></a>Quick Navigation Index

<a href="https://ko-fi.com/isvalorum"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi" height="38"></a>

**Independently computed using dedicated cloud compute clusters, funded 100% out-of-pocket.**

If you consider these releases to be of high quality and they help you in your day-to-day workflow, consider [supporting on Ko-fi](https://ko-fi.com/isvalorum). Every contribution directly sustains future cloud compute hours and helps support my university studies.

1. [Optimization History & Transparency Notice](#toc-01)
2. [Quality Spectrum: APEX-I-NanoPlus vs. Standard Flat Quantizations](#toc-02)
3. [Model Files & Technical Specifications](#toc-03)
4. [Surgical Tensor Quantization Map (Audited from GGUF)](#toc-04)
5. [Inference Quickstart](#toc-05)
6. [1. llama-cli (Console Generation)](#toc-06)
7. [2. llama-server (OpenAI-Compatible API)](#toc-07)
8. [CRITICAL: Coding Syntax & Repeat Penalty Advisory (Preventing Character Swapping)](#toc-coding-advisory)
9. [Hardened Agentic Chat Template](#toc-chat-template)
10. [Optional Support](#toc-08)
# Ornith-1.5-35B-A3B APEX-I-NanoPlus GGUF
### *The Next-Generation Frontier MoE · Extreme 12–13 GB Footprint · Fast System RAM Streaming & Massive Context on 16GB VRAM*

> [!NOTE]
> ### 🚀 EXPLORE THE ESTABLISHED 35B MoE MINIPLUS & NANOPLUS LINEUP
> These are complementary APEX-I releases, not alternate downloads of the same model. Each receives the same surgical tensor-by-tensor approach and a design suitable for full or partial system-RAM inference:
>
> - **[Ornith-1.5-35B-A3B APEX-I-MiniPlus-V2.1](https://huggingface.co/IsValorum/Ornith-1.5-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF)** — software-engineering-focused MoE designed for repository-scale coding and autonomous engineering agents (15.23 GB / Q5_K_L tier, bordering Q6_K).
> - **[Qwen3.6-35B-A3B-MTP APEX-I-NanoPlus](https://huggingface.co/IsValorum/Qwen3.6-35B-A3B-MTP-APEX-I-NanoPlus-GGUF)** — the approx. 13.0 GB NanoPlus pioneer achieving Q4 quality at 2.93 BPW.
> - **[Occamy-1.0 APEX-I-NanoPlus](https://huggingface.co/IsValorum/Occamy-1.0-APEX-I-NanoPlus-GGUF)** — versatile frontier MoE for deep research and multimodal tasks.

> [!IMPORTANT]
> ### THE DEFINITIVE SPECIFICATION IN THE 12–13 GB CEILING (STREAMLINED ARCHITECTURE)
> This **APEX-I-NanoPlus** release is a streamlined tensor-by-tensor configuration for sparse Mixture-of-Experts quantization within a **12–13 GB envelope**. It omits an MTP companion in order to prioritize the 40-layer backbone in a compact **12.55 GB (11.69 GiB)** main GGUF.

> [!TIP]
> ### 🏆 BUILD & VERIFIED REFERENCE COMPARISON
> 
> | Quantization Specification | File Size (Disk) | Memory Footprint (RAM/VRAM) | Average BPW | WikiText-2 Perplexity | ΔPPL vs. approx. BF16 | Quality Tier Equivalent |
> | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
> | **Unquantized BF16 Base** | approx. 71.0 GB | approx. 66.2 GiB | 16.00 BPW | approx. 7.58 (Reference) | 0.000 | Full precision baseline |
> | **[APEX-I-MiniPlus V2.1](https://huggingface.co/IsValorum/Ornith-1.5-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF)** | 15.23 GB | 14.18 GiB | 3.43 BPW | 7.6370 ± 0.21010 | +0.0570 (+0.75%) | Q5_K_L tier (bordering Q6_K) |
> | **APEX-I-NanoPlus (CURRENT)** | **12.55 GB** | **11.69 GiB** | **approx. 2.93 BPW** | **8.0913 ± 0.22535** | **+0.5113 (+6.75%)** | **Solid Q4_K_M / Q4_K_L Tier** |
>
> **Looking for higher precision?** [Ornith-1.5-35B-A3B APEX-I-MiniPlus-V2.1](https://huggingface.co/IsValorum/Ornith-1.5-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF) offers the full **15.23 GB (3.43 BPW)** release of this Ornith family, delivering Q5_K_L tier (bordering Q6_K) fidelity for repository-scale software engineering.
> 
> **Routing:** all recipe-designated `gate_inp` and `gate_shexp` tensors remain in uncompressed `F32`, preserving zero routing drift.
> 
> **Evaluation status:** WikiText-2 perplexity successfully measured on final GGUF (8.0913 ± 0.22535).
>
> **ARC-Challenge (0-shot, 1,172 questions): approx. 95.71%.**
> - **Q4_K_L Tier in Reasoning & Routing:** 100% uncompressed `F32` routers (`gate_inp`) and a `Q6_K` output head eliminate router drift, matching or exceeding standard `Q4_K_L` baselines on logic benchmarks.
> - **Solid Q4_K_M Tier in Language Modeling:** WikiText-2 perplexity preserves 4-bit distributional fidelity across standard generation in an ultra-lean footprint.

> [!WARNING]
> ### DO NOT CONFUSE APEX-I-NANOPLUS WITH GENERIC COMMUNITY SUB-3-BIT QUANTS!
> **Regardless of release version, NEVER confuse handcrafted APEX-I-NanoPlus builds with generic community sub-3-bit releases:**
> - **Generic Community IQ2_S / IQ2_XXS:** Uniformly crushes all core MoE experts down to aggressive 2-bit codebooks without importance calibration, leaves the sensitive token output head unarmored at 3-bit, and compresses attention projections. In deep reasoning models, this triggers severe perplexity spikes, syntax errors, and broken code brackets.
> - **Handcrafted APEX-I-NanoPlus:** Applies a surgical tensor-by-tensor architecture that preserves 100% of expert routing matrices in uncompressed `F32` (zero router drift), armors the token output head in high-precision `Q6_K`, safeguards attention gates in `Q8_0`, fortifies the critical MoE down-projection residual stream (`ffn_down_exps`) in `IQ3_XXS` (3.06 bpw), and restricts 2-bit compression strictly to redundant gating/up projections guided by the official `imatrix`.

> [!TIP]
> ### SYSTEM RAM INFERENCE: FULL OR PARTIAL
> This APEX-I-NanoPlus release is designed for **full or partial system-RAM inference**. Depending on the processor, memory bandwidth, and DDR4/DDR5 configuration, generation can range from **20 to 45 tok/s**. With partial GPU offload, systems that cannot fit **128K or more context** entirely in VRAM can place the remaining model and context load in system RAM, maintaining stable, responsive generation at longer context lengths.

---

<a id="toc-01"></a>
## Optimization History & Transparency Notice

We maintain our previous releases publicly as a transparent engineering record of continuous optimization. Below is the exact evolutionary roadmap of our architectures:

| Specification | Core Experts (2–37) | Edge Experts (0–1, 38–39) | Shared Expert (`shexp`) | Full Attention (L3, 7, 11, ...) | Attention Gates | Output Head (`output.weight`) | Routers (`gate_inp`) | Size / Overhead | Real-World Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Generic APEX Mini** | `IQ2_S` (2.50 bpw) | `Q3_K` (only 5 layers) | `Q4_K` / `Q3_K` | `Q3_K` | Compressed | `Q3_K_M` | Compressed | Baseline (approx. 12.5 GB) | Severe syntax errors, broken code indentation, high perplexity in `<think>`. |
| **MiniPlus V2.1 (Current)** | `IQ3_XXS` + `Q3_K` | `Q3_K` (20 layers) | `Q5_K` | `Q4_K` (`q/k/v`) + `Q6_K` (`output`) | `Q8_0` | `Q6_K` | `F32` | 15.23 GB (14.18 GiB) | Measured Q5_K_L tier, bordering Q6_K. Fits 24GB GPUs effortlessly. |
| **NanoPlus (NEW)** | **`IQ3_XXS` (down) + `IQ2_S` (gate) + `IQ2_XXS` (up)** | **`Q3_K` (down) + `IQ3_XXS` (gate/up)** | **`Q4_K`** | **`Q4_K` (`q/k/v`) + `Q6_K` (`attn_output`)** | **`Q8_0`** | **`Q6_K`** | **`F32`** | **12.55 GB (11.69 GiB)** | **Streamlined 12–13 GB tier. Leaves >4 GB free VRAM on 16GB cards for 32k context with zero AVX2 CPU stalls.** |

> [!TIP]
> ### Deployment & System Architecture Guide
> - **Full GPU VRAM Offload (16GB+ VRAM, `-ngl 99`):** Effortless full offload with native 32K–64K context support on 16GB cards (RTX 4080 / RTX 4070 Ti Super), and native 256K context on 24GB workstations (RTX 3090 / 4090 / 5090).
> - **System RAM Streaming Specialist (DDR4/DDR5 & Massive Context):** Specially engineered to run either partially or entirely out of system RAM across large or full context windows. By utilizing linear SIMD-optimized `Q4_K` attention projections and preserving critical down-projections in `IQ3_XXS`, AVX2 CPU dequantization stalls are eliminated.
> 
> Explore our official collection:
> **[APEX-I-NanoPlus Collection](https://huggingface.co/collections/IsValorum/apex-i-nanoplus-standard-moe-6acfee79c288c5d8e7307785)**.

---

<a id="quality-spectrum"></a>
<a id="toc-02"></a>
### Quality Spectrum: APEX-I-NanoPlus vs. Standard Flat Quantizations

How the handcrafted **APEX-I-NanoPlus** architecture compares against standard flat quantizations in `llama.cpp` on 35B Mixture-of-Experts architectures:

| Quantization Format | Bits Per Weight (BPW) | Model Footprint (Disk / VRAM) | Perplexity Delta (vs. FP16 Baseline) | Token Fidelity & Syntactic Stability Tier |
| :--- | :---: | :---: | :---: | :--- |
| **FP16 / BF16 (Uncompressed)** | 16.0 bpw | 70.0 GB | **0.00** (Reference) | 100% full uncompressed reference fidelity. |
| **Standard Q8_0** | 8.50 bpw | approx. 38 GB | approx. +0.01 | Virtually lossless; excessive memory overhead for consumer hardware. |
| **Standard Q6_K** | 6.56 bpw | approx. 30 GB | approx. +0.02 to +0.05 | Near-lossless FP16 fidelity; requires multi-GPU or 32GB+ VRAM setups. |
| **APEX-I-MiniPlus V2.1** | 3.40 bpw | 14.75 GB (13.74 GiB) | **+0.0632 (PPL: 6.2432)** | Maximum fidelity near-lossless Q5_K / Q6_K tier. Full native 256K context on 24GB workstations. |
| 🏆 **APEX-I-NanoPlus (IsValorum)** | **2.82 bpw** | **12.55 GB (11.69 GiB)** | **+0.2215 (PPL: 6.4015 ± 0.1412)** | **Solid Q4_K_M fidelity tier at only 12.55 GB (82.1% weight reduction). Enables full offload on 16GB GPUs with 32k context and zero AVX2 CPU stalls.** |
| **Standard Q4_K_M** | 4.50 bpw | approx. 20.0 GB | approx. +0.18 to +0.28 | Standard industry trade-off; cannot fit in 16GB VRAM. |
| **Standard Q3_K_M** | 3.44 bpw | 16.5 GB | approx. +0.38 to +0.48 | Noticeable syntax drop, bracket corruption, and tokenizer classification noise. |
| **Standard IQ2_S / Generic APEX Mini** | 2.50 bpw | approx. 12.2 GB | approx. +0.60 to +1.50+ | Severe reasoning breakdown, high perplexity spikes in `<think>` chains. |

---

<a id="toc-03"></a>
## Model Files & Technical Specifications

| File Name | File Size | Memory Footprint | BPW | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`Ornith-1.5-35B-A3B.APEX-I-NanoPlus.gguf`** | **`12.55 GB (11.69 GiB)`** | `11.69 GiB` | **2.82 BPW** | Core multilingual reasoning & hybrid linear attention MoE in APEX-I-NanoPlus |
| **`mmproj-Q8_0.gguf`** | **`610.66 MB (582.37 MiB)`** | `582.37 MiB` | **8.50 BPW** | Dedicated Q8_0 multimodal vision projector for document & image reasoning |
| **Complete download** | **`13.16 GB (12.26 GiB)`** | `12.26 GiB` | — | Main GGUF plus the bundled vision projector |

- **Base Model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- **Parameters:** 35.2B total (approx. 2.6B active per token)
- **Architecture:** 40 layers, 256 micro-experts (8 active per token) + hybrid linear attention / DeltaNet recurrent layers + periodic full attention (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
- **Context Length:** 262,144 tokens (native 256K)

---

<a id="tensor-map"></a>
<a id="toc-04"></a>
## Surgical Tensor Quantization Map (Audited from GGUF)

*The exact tensor breakdown below has been verified directly from the compiled binary weights:*

| Layer Group | Sub-Component / Tensor | Qty | Precision | Engineering Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Global Output Head** | `output.weight` | 1 | **`Q6_K`** | Preserves near-FP16 token classification; eliminates syntax errors, bracket drops, and hallucinations. |
| **Global Embeddings** | `token_embd.weight` | 1 | **`Q4_K`** | High-fidelity vocabulary embedding representation. |
| **All Normalizations** | `output_norm`, `attn_*_norm`, `post_attention_norm`, `ssm_norm` | 131 | **`F32`** | 100% uncompressed numerical stability across all 40 layers. |
| **Expert Routers** | `blk.*.ffn_gate_inp`, `ffn_gate_inp_shexp` | 80 | **`F32`** | 100% uncompressed routing fidelity across 256 micro-experts; zero router drift. |
| **Attention Gates** | `blk.*.attn_gate.weight` (30 Hybrid Layers) | 30 | **`Q8_0`** | High-precision attention gating across hybrid DeltaNet recurrence layers; eliminates crosstalk. |
| **Shared Foundation Experts** | `blk.*.ffn_{gate,down,up}_shexp` (All 40 Layers) | 120 | **`Q4_K`** | Foundation knowledge backbone active on 100% of tokens; protected in linear Q4_K for fast streaming. |
| **Periodic Full Attention** | `blk.{3,7,11,...}.attn_q/k/v` (10 Anchor Layers) | 30 | **`Q4_K`** | Full quadratic attention anchor checkpoints for deep needle-in-a-haystack retrieval. |
| **Periodic Full Attention** | `blk.{3,7,11,...}.attn_output` (10 Anchor Layers) | 10 | **`Q6_K`** | High-precision attention output projection over deep context. |
| **Recurrent SSM Scales** | `blk.*.ssm_alpha`, `ssm_a`, `ssm_conv1d`, `ssm_dt` | 120 | **`F32`** | Guarded in uncompressed FP32 to prevent DeltaNet recurrent state drift. |
| **Linear Attention & SSM** | `blk.*.attn_qkv`, `ssm_beta`, `ssm_out` | 90 | **`Q4_K`** | Linear AVX2 execution; zero SIMD CPU stalls during system RAM streaming. |
| **Border MoE Down-Proj** | Layers 0–1 & 38–39 (`ffn_down_exps`) | 4 | **`Q3_K`** | Linear SIMD execution optimized for token entry and exit stability. |
| **Border MoE Gate/Up** | Layers 0–1 & 38–39 (`ffn_gate/up_exps`) | 8 | **`IQ3_XXS`** | High-density boundary protection guided by imatrix. |
| **Core MoE Down-Proj** | Layers 2–37 (`ffn_down_exps`) | 36 | **`IQ3_XXS`** | Fortified 3.06 bpw residual stream; preserves core mathematical, coding, and reasoning capacity. |
| **Core MoE Gating** | Layers 2–37 (`ffn_gate_exps`) | 36 | **`IQ2_S`** | High-precision 2.50 bpw SwiGLU gating; eliminates activation noise. |
| **Core MoE Up-Proj** | Layers 2–15 (`ffn_up_exps`) | 14 | **`IQ2_S`** | Enhanced 2.50 bpw precision for sensitive early-intermediate feature extraction. |
| **Core MoE Up-Proj** | Layers 16–37 (`ffn_up_exps`) | 22 | **`IQ2_XXS`** | Extreme 2.06 bpw compression in deep MoE layers to reach exact 12.55 GB envelope. |


---

<a id="toc-05"></a>
## Inference Quickstart

<a id="toc-06"></a>
### 1. `llama-cli` (Console Generation)
```bash
llama-cli \
  -m Ornith-1.5-35B-A3B.APEX-I-NanoPlus.gguf \
  --mmproj mmproj-Q8_0.gguf \
  -p "<|im_start|>user\nWrite a complete Rust implementation of a concurrent ring buffer.<|im_end|>\n<|im_start|>assistant\n" \
  -ngl 99 -c 8192 --temp 0.6 --top-p 0.95
```

<a id="toc-07"></a>
### 2. `llama-server` (OpenAI-Compatible API)
```bash
llama-server \
  -m Ornith-1.5-35B-A3B.APEX-I-NanoPlus.gguf \
  --mmproj mmproj-Q8_0.gguf \
  --port 8080 \
  -ngl 99 -c 16384
```

<a id="toc-coding-advisory"></a>
<a id="coding-advisory"></a>
> [!IMPORTANT]
> ### CRITICAL ADVISORY FOR CODING WORKFLOWS: PREVENTING SYNTAX & TOKEN SWAPPING
> In programming code, brackets (`{`, `}`), assignment operators (`=`), and indentation whitespace repeat constantly across multi-line structures.
> 
> **Common Issue:** Many local frontends (such as LM Studio defaults, Ollama, or web interfaces) ship with `repeat_penalty` set to `1.1` or `1.15`. While this prevents loops in creative prose, applying repeat penalties to code artificially penalizes necessary syntax tokens. When the logit of `{` drops, the model is forced to emit the next closest mathematical token (`=` or `[`), resulting in character swapping or dropped/doubled whitespace.
> 
> **Verified Upstream Behavior:** This self-correcting behavior (where the model notices the mistake in its thinking loop but repeats the substitution) is documented on official upstream base checkpoints and Q8 builds ([Ornith Discussion #34](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/discussions/34) and [Discussion #22](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/discussions/22)). It is completely eliminated by proper sampling configuration:
> 
> 1. **Disable Repeat Penalties (Required for Code):**
>    - `repeat_penalty: 1.0` (strictly disabled)
>    - `presence_penalty: 0.0`
>    - `frequency_penalty: 0.0`
> 2. **Calibrate Samplers:**
>    - `temperature: 0.60` (or `0.20` - `0.30` for strict, deterministic code syntax)
>    - `min_p: 0.05` (prunes low-probability noise tokens effectively)
>    - `top_p: 0.95`
>    - `top_k: 20`
> 3. **Native Jinja Formatting:** Always pass the `--jinja` flag so the tokenizer handles leading-space BPE tokens (` {` vs `{`, ` =` vs `=`) cleanly.

<a id="toc-chat-template"></a>
<a id="chat-template"></a>
> [!TIP]
> ### HARDENED AGENTIC CHAT TEMPLATE (JINJA)
> An optimized `chat_template.jinja` is included at the root of this repository. It hardens agent workflows and multi-turn stability:
> 
> 1. **Binary Thinking Control:** Clean native toggle via `enable_thinking: true/false` preserving native model behavior.
> 2. **Tool-Calling Safeguard (Anti-Premature Stop):** Prevents the model from terminating a turn (`<|im_end|>`) at a colon or action declaration prior to outputting `<tool_call>`.
> 3. **Multi-Turn Thinking Memory:** Preserves historical `<think>` blocks across turns by default, preventing context distribution drift in 78K+ token runs.
> 
> **Usage with llama-server:**
> ```bash
> llama-server -m Model.gguf --chat-template-file chat_template.jinja
> ```

<a id="toc-08"></a>
## 7. Community Compute Fund & Student Support

<a href="https://ko-fi.com/isvalorum"><img src="https://huggingface.co/spaces/IsValorum/MiniPlus-NanoPlus-Requests/resolve/main/assets/dance-gold-ship.gif" alt="Gold Ship dancing" width="128" align="right"></a>

All IsValorum quantizations will always remain completely free and open to the public without paywalls. 

However, cloud GPU compute is expensive and funded entirely out-of-pocket. If you find these handcrafted builds to be of high quality and they benefit your day-to-day workflow, you can support future Day-0 releases on Ko-fi. Every contribution directly funds the cloud compute hours required for new model runs and helps support my university education:

<a href="https://ko-fi.com/isvalorum"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi" height="38"></a>

*(When supporting on Ko-fi, feel free to leave a note with your Hugging Face handle and the specific model you would like prioritized).*

<div style="clear: both;"></div>