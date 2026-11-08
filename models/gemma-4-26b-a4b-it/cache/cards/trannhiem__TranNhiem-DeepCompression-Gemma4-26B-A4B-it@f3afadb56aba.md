---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - gemma4
  - mixture-of-experts
  - expert-pruning
  - nvfp4
  - modelopt
  - vllm
  - vision-language
  - compressed
---

<!-- DRAFT 2026-11-06. TODO before publishing: add the paper link;
     confirm release visibility after the patent filing. -->

# 🧩 TranNhiem-DeepCompression-Gemma4-26B-A4B-it

<p align="center">
  <img src="assets/Collaborative-Deep-Compression.png" alt="Collaborative Deep Compression: pruning, clustering (weight sharing), low-bit quantization" width="90%">
  <br><sub>The full Collaborative Deep Compression pipeline (conceptual). This release uses <b>pruning</b> (whole routed experts) and <b>low-bit quantization</b> (NVFP4); clustering is not applied (see §3).</sub>
</p>

<p align="center">
  <b>Paper:</b> <i>Collaborative Deep Compression for Foundation Models</i> (link coming soon) ·
  <b>Base model:</b> <a href="https://huggingface.co/google/gemma-4-26B-A4B-it">Gemma-4-26B-A4B-it</a> ·
  <b>Sibling release:</b> <a href="https://huggingface.co/trannhiem/TranNhiem-DeepCompression-Qwen3.6-35B-A3B">TranNhiem-DeepCompression-Qwen3.6-35B-A3B</a>
</p>

**64% smaller (49 GB → 17 GB), +29% batched throughput, 1.46× tokens per joule — keeps 99.8% of the original performance on average (≥ 98.9% on each of 8 benchmarks).**

> ⚠️ **Format: NVFP4 (4-bit weights and activations) on the routed experts.** NVFP4 is the native 4-bit format of NVIDIA **Blackwell** GPUs: workstation RTX PRO 6000 Blackwell, DGX Spark (GB10) and GeForce RTX 50-series. Use vLLM ≥ 0.30.1. All numbers on this card were measured on **GB200**; other Blackwell GPUs should run the model, but their speed and kernel support were not tested (workstation/consumer Blackwell uses different FP4 kernels from B200/GB200). With older vLLM the experts fall back to a slower weight-only path (measured). Pre-Blackwell GPUs (Hopper, Ada, Ampere) have no FP4 tensor cores.

## 1. 📖 Model Introduction

This is Gemma-4-26B-A4B-it (a mixture-of-experts vision-language model) compressed with **Collaborative Deep
Compression (CDC)**: the compression stages are designed to work together, in a fixed order, and each is judged on the
modalities the model serves, not on text alone.

Two stages were applied:

1. **Expert pruning** (pillar A): each MoE layer keeps the 120 of its 128 experts that matter most for real
   text, image and video traffic.
2. **4-bit experts** (pillar C): the remaining routed experts are quantized to NVFP4 (4-bit weights and activations).
   Attention, the dense MLP, the router and the vision tower stay in bf16.

The result loads and serves in vLLM with native Blackwell FP4 kernels.

## 2. 📋 Model Summary

| | |
|---|---|
| Base model | Gemma-4-26B-A4B-it (`Gemma4ForConditionalGeneration`), image + video + text input |
| Architecture | MoE, 30 decoder layers, top-8 routing, expert intermediate size 704 |
| Experts per layer | **120** (from 128) |
| Precision | routed experts **NVFP4 W4A4** (NVIDIA ModelOpt); attention, dense MLP, router, vision tower **bf16** |
| Size on disk | **17 GB** (bf16 base: 49 GB) |
| Weights in GPU memory | **17.3 GiB** (bf16 base: 48.5 GiB) |
| Serving | vLLM ≥ 0.30.1 (native NVFP4 MoE kernel); Blackwell GPU for W4A4 (measured on GB200) |

## 3. 🔬 Deep Compression Method

| Pillar | Stage | Applied here | What it did |
|---|---|---|---|
| A | Pruning-aware compression | ✅ | Kept 120 / 128 experts per layer |
| B | Clustering-aware compression | ❌ | Not applied (see below) |
| C | Quantization-aware compression | ✅ | Routed experts → NVFP4 W4A4 |
| D | Serving | ✅ | vLLM native NVFP4 MoE path |

**Pillar A: multimodal expert pruning.** Each expert is scored by how much it contributes when it is used: its
router weight times the size of its output, averaged over the tokens routed to it. The scores come from 256 text
conversations plus 256 image, multi-image and video conversations, because calibrating on text alone looks fine on
MMLU but hides losses of 3–5 points on image benchmarks. The 8 lowest-scoring experts per layer are removed, with no
retraining. Keeping 120 is the measured knee: keeping 112 costs about 3.6 points on MMMU.

**Pillar C: experts-only NVFP4.** The routed experts hold almost all of the parameters, and they tolerate 4-bit
weights *and* activations: experts-only NVFP4 matches bf16 within ±0.8 points on all 8 benchmarks. Putting attention
and the dense MLP into 4-bit as well costs 2–4 points and is slower at batch, so they stay in bf16.

**Why not pillar B (weight clustering) here.** Clustered (codebook) weights need a lookup-table matrix kernel that
vLLM does not ship, so a clustered model could not be served at speed. On these experts, clustering-aware training
also did no better than simply snapping the weights to the codebook without training.

## 4. 📊 Evaluation Results

Accuracy (%) on the same serving stack. This model's row is the original score plus the mean paired change over 3 pruning-calibration seeds.

**This model keeps 99.8% of the original model's performance on average (at least 98.9% on every benchmark) at about one third of the weight memory.**

| | MMLU | AI2D | ChartQA | MMBench | MMMU | MMStar | OCRBench | MVBench (video) |
|---|---|---|---|---|---|---|---|---|
| Original bf16 model | 81.70 | 81.28 | 70.80 | 90.30 | 70.67 | 66.00 | 84.00 | 56.97 |
| **This model** | 80.96 | 80.86 | 70.00 | 89.48 | 71.33 | 66.24 | 84.27 | 57.10 |
| **Performance retained** | **99.1%** | **99.5%** | **98.9%** | **99.1%** | **100.9%** | **100.4%** | **100.3%** | **100.2%** |

Retained = this model's score ÷ the original's score. Values above 100% are within evaluation noise: treat them as "no measurable loss", not as an improvement.

Detailed change vs. the original, on the same serving stack (percentage points; paired item-by-item; mean ± s.d.
over 3 pruning-calibration seeds):

| MMLU | AI2D | ChartQA | MMBench | MMMU | MMStar | OCRBench | MVBench (video) |
|---|---|---|---|---|---|---|---|
| −0.74 ± 0.21 | −0.42 ± 0.33 | −0.80 ± 0.17 | −0.82 ± 0.09 | +0.66 ± 0.70 | +0.24 ± 0.40 | +0.27 ± 0.31 | +0.13 ± 0.62 |

- MMLU, ChartQA and MMBench cost about 0.75–0.8 points on every seed. The other benchmarks are within noise, with the
  sign varying by seed.
- Re-evaluating one checkpoint moves these benchmarks by about 1 point, so differences below that are not
  meaningful.

## 5. ⚡ Efficiency

One GB200 GPU, tensor-parallel 1, vLLM 0.30.1rc1, GPU memory utilisation 0.85. Throughput and energy are the median of
4 runs of 64 concurrent requests × 512 generated tokens; power is the board power of the GPU used.

| | bf16 base | **This model** | Change |
|---|---|---|---|
| Single-stream decode (tokens/s) | 251 | **302** | +20% |
| Batched decode, 64 requests (tokens/s) | 9,771 | **12,557** | +29% |
| Inter-token latency, single stream | 3.96 ms | **3.29 ms** | −17% |
| Time to first token | 16 ms | 18 ms | ≈ |
| GPU power (batched) | 702 W | **617 W** | −12% |
| Energy efficiency (tokens per joule) | 13.9 | **20.4** | 1.46× |
| Weights in GPU memory | 48.5 GiB | **17.3 GiB** | −64% |
| KV-cache capacity at the same memory budget | 484 K tokens | **629 K tokens** | +30% |
| Max concurrent requests at 4 K context | 118 | **154** | +30% |
| Max concurrent requests at 32 K / 128 K context | 14.9 / 11.5 | **19.4 / 14.7** | +30% / +28% |

- Image-request batches are 11–15% faster and video is about the same, because image and video encoding stays in
  bf16.
- Tensor parallelism trades energy for throughput. For the best tokens per joule, serve on one GPU.

## 6. 🚀 Deployment

```bash
pip install "vllm>=0.30.1"   # older vLLM serves the NVFP4 experts through a slower weight-only path
vllm serve trannhiem/TranNhiem-DeepCompression-Gemma4-26B-A4B-it --max-model-len 32768
```

- **Hardware:** native 4-bit (W4A4) needs a Blackwell GPU (B200 / GB200 / B300, RTX PRO 6000 Blackwell, DGX Spark, RTX 50-series). Measured on GB200 only.
- **Keep async scheduling on** (the default). Turning it off costs about 40% single-stream speed on this model.

## 7. 💻 Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="trannhiem/TranNhiem-DeepCompression-Gemma4-26B-A4B-it",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
        {"type": "text", "text": "What trend does this chart show?"},
    ]}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

## 8. ⚠️ Limitations

- All measurements are on one GPU type (GB200). Speed and energy on other hardware will differ.
- Accuracy is reported on 8 benchmarks (knowledge, diagrams, charts, general and expert vision, OCR, video). The
  model was not re-evaluated for safety beyond the base model.
- Pruning removes 8 experts per layer permanently. Rare skills that only those experts served may degrade more than
  these benchmarks show.

## 9. 📜 License

Apache-2.0, inherited from the base model. This is a modified version of Gemma-4-26B-A4B-it: 8 routed experts per layer
were removed and the remaining experts were quantized to NVFP4.

## 10. 📝 Citation

```bibtex
@article{cdc2026,
  title  = {Collaborative Deep Compression for Foundation Models},
  author = {Tran, Van Nhiem},
  year   = {2026},
  note   = {Paper link coming soon}
}
```

## 11. 📬 Contact

**Trần Nhiệm** · rick.tvn@foxconn.com WhatsApp: +886 934 311751
