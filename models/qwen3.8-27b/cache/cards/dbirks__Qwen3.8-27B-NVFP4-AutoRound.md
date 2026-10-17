---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
license: apache-2.0
language:
  - en
tags:
  - nvfp4
  - fp4
  - compressed-tensors
  - auto-round
  - vllm
  - quantized
  - qwen3
---

# Qwen3.8-27B-NVFP4-AutoRound

An NVFP4 (4-bit, W4A4) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made with
[Intel AutoRound](https://github.com/intel/auto-round) and packaged in
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) for vLLM. It's a text-only quant: the
language decoder's high-volume projections are NVFP4, while the GatedDeltaNet recurrence-control projections
(`in_proj_a` / `in_proj_b`), the vision tower, the MTP head, and `lm_head` stay in BF16. It's about 20.6 GB on
disk, down from roughly 52 GB at BF16.

This build is aimed at NVIDIA Blackwell, where the FP4 tensor cores run W4A4 natively. On an older GPU, the
weight-only [Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) sibling is
the better fit. Accuracy notes are in the Evaluation section below.

## Model overview
- **Base model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — a hybrid **GatedDeltaNet (linear-attention) + full-attention** multimodal decoder (64 layers: 48 linear-attn + 16 full-attn) with a vision tower; 262K context.
- **Quantization:** NVFP4 (W4A4) on the language-decoder `Linear` layers, **except** the two tiny per-layer recurrence-control projections (`in_proj_a`, `in_proj_b`) which stay BF16.
- **Kept BF16:** `linear_attn.in_proj_a` / `in_proj_b` (GDN decay/write-strength — quantizing these wrecks the recurrence), the entire `visual.*` vision tower, the `mtp` head, and `lm_head`.
- **Format:** compressed-tensors (`nvfp4-pack-quantized`), auto-detected by vLLM.
- **Quantizer:** Intel AutoRound (arXiv:2309.05516), SignRound block reconstruction.

## Quantization recipe
- **Scheme:** `NVFP4` — 4-bit weights + input activations, NVFP4 microscale (group size 16, FP8 `e4m3` block scale + FP32 global), symmetric, `strategy: tensor_group`.
- **Quantized:** the 48 GatedDeltaNet layers' `in_proj_qkv` / `in_proj_z` / `out_proj` + all MLP `gate/up/down` + the 16 full-attention layers' `q/k/v/o` projections.
- **Kept BF16:** `in_proj_a`, `in_proj_b`, `visual.*`, `mtp`, `lm_head`.
- **Calibration:** `NeelNanda/pile-10k`, 128 samples, seqlen 2048, 200 tuning iters (screen recipe).
- **Hardware:** single NVIDIA RTX PRO 6000 Blackwell 96 GB (SM120); peak 42 GB VRAM, 13.6 GB host RAM.

## Deployment (vLLM)
vLLM auto-detects the NVFP4 scheme from `config.json` — no quantization flag needed. Serves as
`Qwen3_5ForConditionalGeneration`. On consumer Blackwell (SM120) use FP4 tensor-core kernels:
```bash
VLLM_ATTENTION_BACKEND=FLASHINFER FLASHINFER_CUDA_ARCH_LIST=12.0f \
vllm serve dbirks/Qwen3.8-27B-NVFP4-AutoRound --max-model-len 8192 --trust-remote-code
```

### Example `compose.yaml`
```yaml
services:
  qwen38-nvfp4:
    image: vllm/vllm-openai:latest   # any recent vLLM with Qwen3.8 (qwen3_5) + SM120 NVFP4 support
    ports:
      - "8000:8000"
    ipc: host
    environment:
      # consumer Blackwell (SM120): use FlashInfer FP4 tensor-core kernels (not the Marlin fallback)
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - FLASHINFER_CUDA_ARCH_LIST=12.0f
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command:
      - "--model=dbirks/Qwen3.8-27B-NVFP4-AutoRound"
      - "--served-model-name=qwen3.8-27b"
      - "--max-model-len=8192"
      - "--trust-remote-code"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Evaluation

Measured in the model's default thinking mode with its recommended sampling (temperature 1.0, top-p 0.95,
top-k 20), using EleutherAI lm-evaluation-harness.

| Task | BF16 base | This (NVFP4 W4A4) |
|---|---|---|
| GSM8K (full 1319, thinking) | 0.911 ±0.015 | 0.905 ±0.016 |
| HumanEval (pass@1, instruct) | 0.939 ±0.037 | 0.927 ±0.040 |
| MMLU-Pro (100/subject, thinking) | 0.819 ±0.020 | 0.796 ±0.021 |

Across all three tasks the quant lands inside the BF16 base's confidence interval, so there's no measurable
accuracy loss from the NVFP4 quantization. (MMLU-Pro here is 100 questions per subject, so it's an internal
reference against our own BF16 baseline, not directly comparable to a full-split MMLU-Pro number elsewhere.)

## Limitations
- W4A4 quantizes activations as well as weights. If you want weight-only (BF16 activations) or need a non-Blackwell GPU, use the [W4A16 sibling](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound). Accuracy differences between the two are being measured (see Evaluation) and are not yet established.
- NVFP4 requires **NVIDIA Blackwell (SM100/SM120)** FP4 tensor cores; no speedup on Ada/Hopper (falls back to Marlin, about 2x slower).
- Only the language decoder is quantized — vision tower is BF16 (intentional; vLLM requires it).

## Reproducibility
```python
from auto_round import AutoRound
# keep in_proj_a/b + visual + mtp in BF16 via a bits=16 layer_config (built from module names)
ar = AutoRound("Qwen/Qwen3.8-27B", scheme="NVFP4", dataset="NeelNanda/pile-10k",
               nsamples=128, seqlen=2048, batch_size=4, iters=200,
               device_map=0, trust_remote_code=True, quant_nontext_module=False, seed=42,
               layer_config=BF16_FOR_IN_PROJ_AB_VISUAL_MTP)
ar.quantize_and_save(output_dir="Qwen3.8-27B-NVFP4-AutoRound", format="llm_compressor")
```
Toolchain: auto-round 0.15.0, transformers-from-source (qwen3_5 arch), compressed-tensors, torch 2.13+cu130.

## Citation
```bibtex
@article{cheng2023optimize, title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs}, author={Cheng, Wenhua and others}, journal={arXiv:2309.05516}, year={2023}}
```
