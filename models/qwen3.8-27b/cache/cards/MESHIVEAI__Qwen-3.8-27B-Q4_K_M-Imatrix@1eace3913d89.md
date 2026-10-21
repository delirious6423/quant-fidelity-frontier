---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: meshive
library_name: gguf
license: apache-2.0
language:
- en
- ko
pipeline_tag: text-generation
tags:
- qwen
- qwen3.5
- gguf
- imatrix
- quantization
- llama-cpp
datasets:
- wikitext
---

# Qwen-3.8-27B — Q4_K_M GGUF (imatrix)

A Q4_K_M GGUF quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
built with a custom importance matrix.

Every number on this card was measured, not estimated. The measurement setup is
documented at the bottom so you can reproduce it.

| | |
|---|---|
| File size | 15.65 GiB (16,810,714,624 bytes) |
| Parameters | 27.32 B |
| Architecture | `qwen35` hybrid — 65 blocks (48 Gated DeltaNet + 17 full attention) |
| Vocab | 248,320 |
| Trained context | 262,144 (see [Known issues](#known-issues)) |
| Weight types | Q4_K ×439, Q6_K ×67, F32 ×360 |

---

## Why imatrix

An importance matrix records which weights actually matter for the model's
predictions, so the 4-bit quantizer spends its limited precision where it counts.
The question is whether it measurably helps. For this model, it does.

Both models below were quantized from the **same F16 GGUF** with the **same
Q4_K_M recipe**. Their tensor data is byte-for-byte the same size — the only
difference is which values the quantizer picked. Measured against the F16
baseline on the full Wikitext-2 test set (296,960 tokens, 580 chunks):

| Metric | **This model (imatrix)** | Same quant, no imatrix | Improvement |
|---|---|---|---|
| **Mean KL divergence** ↓ | **0.017486** ± 0.000201 | 0.025196 ± 0.000291 | **−31%** |
| **Top-token agreement** ↑ | **94.18%** ± 0.06 | 92.97% ± 0.07 | **+1.21 pp** |
| RMS Δp ↓ | **3.583%** ± 0.034 | 4.305% ± 0.040 | −17% |
| Mean Δp ↓ | **−0.124%** ± 0.009 | −0.195% ± 0.011 | −36% |
| Perplexity | **6.9619** | 6.9755 | |
| ΔPPL vs F16 ↓ | **+0.0092** ± 0.0041 | +0.0228 ± 0.0050 | −60% |

F16 reference: PPL 6.9527 ± 0.0450.

KL divergence is the metric to watch — it measures how far the quantized model's
whole output distribution drifts from F16, and unlike perplexity it doesn't let
errors cancel out. The gap here is roughly 22σ, so it is not noise.

For context, llama.cpp's own Q4_K_M scoreboard shows a ~10% KLD improvement from
an imatrix built on 10M tokens. This one gets 31% from 326k tokens, which is what
a 27B model's weight-importance structure being highly non-uniform looks like.

### Out-of-domain check: Korean

The imatrix was calibrated on English text, so the table above is a best case. To
see how far the benefit carries, the same three models were scored on Korean
Wikipedia (`wikimedia/wikipedia`, `20231101.ko`, 1,123,328 tokens):

| | F16 | **This model** | No imatrix |
|---|---|---|---|
| Perplexity | 6.8782 | **6.9485** | 6.9596 |
| Degradation vs F16 | — | **+1.02%** | +1.18% |

Two things to take from this. The imatrix version is still ahead, by roughly 14%
less degradation. And **both** quantizations lose noticeably more on Korean than on
English — around 1% versus 0.1-0.3% — so an English-calibrated imatrix narrows the
gap on out-of-domain text without closing it.

This run measured perplexity only, without the paired per-token statistics that the
KL-divergence mode produces. The direction is consistent and matches the English
result, but the Korean margin should be read as indicative rather than as an
established effect size.


---

## Speed

Measured on a single RTX PRO 6000 Blackwell Workstation Edition (96 GB), CUDA 13.0,
llama.cpp build `d077b4c21`, all layers offloaded.

| Context depth | Prefill (pp4096) | Generation (tg128) | tg retained |
|---|---|---|---|
| 0 | 4,019 t/s | **79.85 t/s** | 100% |
| 16,384 | 3,474 t/s | 73.83 t/s | 92% |
| 65,536 | 2,276 t/s | 64.37 t/s | 81% |
| 131,072 | 1,571 t/s | **54.85 t/s** | **69%** |

Generation holds 69% of its speed at 128k context. That is unusually flat for a
27B model, and it comes from the hybrid architecture: only 17 of 65 layers keep a
growing KV cache, while the other 48 are constant-state Gated DeltaNet layers.

Generation is memory-bandwidth bound, running at roughly 1.29 TB/s — about 72% of
the card's theoretical peak.

---

## Run it on Meshive

Meshive is a task-first GPU cloud. Bring your task — training, inference, or rendering — and run it on RTX 5090 and RTX PRO GPUs at up to 70% less.

The numbers above were measured on a [Meshive GPU Pod](https://meshive.ai/pods?utm_source=huggingface&utm_medium=model_card&utm_campaign=Qwen-3.8-27B-Q4_K_M-Imatrix) with one RTX PRO 6000 Blackwell.

---

## Hardware requirements

The weights alone are 15.65 GiB, so **16 GB is not enough**. Budget for the KV
cache on top:

| Context | Approx. total VRAM | Fits on |
|---|---|---|
| 8k | ~18 GB | 24 GB card |
| 32k | ~20 GB | 24 GB card |
| 64k | ~23 GB | 24 GB card |
| 128k | ~27 GB | 32 GB card |

KV figures are computed from the model config (17 attention layers × 4 KV heads ×
512 head dim, F16) and will vary a little by runtime. Use `--cache-type-k q8_0
--cache-type-v q8_0` to roughly halve them, or offload fewer layers with `-ngl` to
run on a smaller card at reduced speed. CPU-only works with 32 GB of system RAM.

---

## Usage

### llama.cpp

Needs a build with `qwen35` hybrid support — use a recent release. Earlier builds
will fail to load with a tensor or architecture error.

```bash
llama-cli -m Qwen-3.8-27B-Q4_K_M-Imatrix.gguf -ngl 999 -c 32768 \
  -p "Explain the concept of quantum computing in simple terms."
```

As an OpenAI-compatible server:

```bash
llama-server -m Qwen-3.8-27B-Q4_K_M-Imatrix.gguf -ngl 999 -c 32768 --host 0.0.0.0 --port 8080
```

### Ollama

```bash
printf 'FROM ./Qwen-3.8-27B-Q4_K_M-Imatrix.gguf\n' > Modelfile
ollama create qwen-3.8-27b-imatrix -f Modelfile
ollama run qwen-3.8-27b-imatrix
```

Ollama and LM Studio bundle their own llama.cpp, so update to a recent version
before reporting a load failure.

### Chat template

**Do not set a chat template manually, and do not use ChatML.** This model ships a
Qwen3.5-specific template that is already embedded in the GGUF; llama.cpp, Ollama,
and LM Studio apply it automatically. Overriding it with ChatML degrades output
quality and breaks tool calling.

---

## Known issues

**Context above ~130k is currently broken upstream.** The model advertises 262,144
context, but on `qwen35` hybrid models llama.cpp emits EOS as the first generated
token once the prompt exceeds roughly 130k positions — prefill completes normally,
then generation stops immediately. This affects the CUDA and CPU backends alike and
is tracked at [ggml-org/llama.cpp#27756](https://github.com/ggml-org/llama.cpp/issues/27756).
It is an inference-engine bug, not a quantization artifact; the same failure occurs
on unquantized weights. Keep prompts under 128k until it is fixed.

**Use the CUDA or Metal backends.** Two other open upstream bugs affect this
architecture: Vulkan produces garbage output at batch size 512
([#27237](https://github.com/ggml-org/llama.cpp/issues/27237)) and the HIP backend
silently corrupts inference on gfx1151
([#27556](https://github.com/ggml-org/llama.cpp/issues/27556)).

**Text only.** The base model is multimodal, but this GGUF contains the text tower
only — there is no `mmproj` file, so image and video input will not work.

---

## Quantization method

```
Base model     Qwen/Qwen3.8-27B  →  F16 GGUF via convert_hf_to_gguf.py
Quantizer      llama.cpp llama-quantize, Q4_K_M
Imatrix        Wikitext-2 (train), 637 chunks (~326k tokens), computed over F16
Hardware       NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)
```

The imatrix metadata is embedded in the GGUF and can be inspected with any GGUF
reader — look for `quantize.imatrix.entries_count` (496) and
`quantize.imatrix.chunks_count` (637).

### Reproducing the evaluation

```bash
# 1. F16 reference logits
llama-perplexity -m base-f16.gguf -f wiki.test.raw -c 512 -b 512 -ngl 999 \
  --kl-divergence-base ref.kld

# 2. Score this model against them
llama-perplexity -m Qwen-3.8-27B-Q4_K_M-Imatrix.gguf -f wiki.test.raw \
  -c 512 -b 512 -ngl 999 --kl-divergence-base ref.kld --kl-divergence
```

`ref.kld` is about 69 GiB for this vocabulary size — roughly 250 KB per token of
corpus. To build the no-imatrix control, run `llama-quantize base-f16.gguf
control.gguf Q4_K_M` and score it the same way.

### Caveats

The imatrix was calibrated on Wikitext-2 **train** and evaluated on Wikitext-2
**test**. The two splits are disjoint, but they share a domain, so the 31% figure
is a best case for English encyclopedic prose. The Korean check above suggests the
margin narrows to roughly 14% out of domain; expect something in that range for
code, dialogue, or other languages.

10 of the 506 quantized tensors had no imatrix coverage and fell back to standard
quantization.

---

## Credits

Quantized by **meshive**. Base model by the Qwen team, Apache-2.0.
Evaluation methodology follows llama.cpp's `perplexity` tool.