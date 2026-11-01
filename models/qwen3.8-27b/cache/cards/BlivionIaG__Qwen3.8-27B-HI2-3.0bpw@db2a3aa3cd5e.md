---
library_name: vllm
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
  - safetensors
  - qwen3_5
  - image-text-to-text
  - text-generation
  - conversational
  - base_model:Qwen/Qwen3.8-27B
  - base_model:quantized:Qwen/Qwen3.8-27B
  - license:apache-2.0
  - hi2
  - 3-bit
  - amd
---

# Qwen3.8-27B-HI2-3.0bpw

A 3.0-bit post-training quantization (PTQ) plus quantization-aware distillation (QAD)
checkpoint of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced by
[hi2](https://github.com/BlivionIaG/hi2) (`rdna_hint_flex` v3.2) and served on AMD RDNA GPUs
(gfx1030/gfx1100, ROCm 7.14) with the [vLLM fork](https://github.com/BlivionIaG/vllm) that
carries the method.

## Model

| | |
| --- | --- |
| Source | Qwen3.8-27B (BF16, 51.7 GiB) |
| Quantized linears | 496, 24.35 B weights (8.5 GiB) |
| Measured bpw | 3.012 over every byte of the linears |
| Weight mix | 66 % INT2 / 34 % INT4 |
| Checkpoint | 15.0 GiB (3.45x smaller than BF16) |
| Calibration | 262,144 tokens; GPTQ rounding, rotation on, smooth 0.5 |
| QAD | 300 steps per layer on 64 layers (247 min on four V620s), then a 200-step final logit-KL phase: held-out KL 0.4122 -> 0.2427 (7.85 h) |

The settings are in `config.json` (`quantization_config`, version 3.2); the per-layer report
is in `hi2_report.json`; the shard checksums in `crc32.txt`.

```bash
hi2 quantize Qwen/Qwen3.8-27B out/Qwen3.8-27B-HI2-3.0bpw \
  --calib calib.txt --calib-samples 128 --seq-len 2048 \
  --target-bpw 3.0 --qad-preset default
```

## Evaluation

Measured against the BF16 source. "docs" is `study_text_all.txt` from token 800,000 (after the
262,144-token calibration region); "mix" is `eval_mix.txt` from token 0, a separate slice of the
same corpus that does not occur in the calibration text. Both are 8 x 2048 tokens, `hi2 eval
--dtype float16`.

Not measured: text KLD for AWQ (hi2 does not dequantize AWQ; it was scored on the task suite
and the teacher-forced domain KLD), and perplexity for GGUF (its KLD was measured with
llama.cpp at `-dev none` against the BF16 base, on the same texts).

| | BF16 | HI2 3.0 bpw (this) | AWQ INT4 | GGUF Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| Body bpw | 16 | 3.012 | 4.625 | 4.738 |
| Checkpoint | 51.7 GiB | 15.0 GiB | 20,044 MiB | 15,702 MiB |
| Perplexity, docs text | 4.107 | **4.158** | 4.189 | — |
| Perplexity, mix text | 5.156 | **4.567** | 5.219 | — |
| KLD, docs text | 0 | **0.0593** | — | 0.0142 |
| KLD, mix text | 0 | **0.1837** | — | 0.0776 |
| Top-1, docs / mix | 100 / 100 % | **90.8 / 86.7 %** | — | — |
| EAR, docs / mix | 1 / 1 | **0.918 / 0.874** | — | — |

### Task suite (reduced, 100 items per family)

| family | n | BF16 | HI2 3.0 bpw (this) | AWQ INT4 |
| --- | ---: | ---: | ---: | ---: |
| gsm8k | 100 | 95.0 % | 96.0 % | 95.0 % |
| humaneval | 100 | 95.0 % | 94.0 % | 95.0 % |
| mbpp | 100 | 93.0 % | 84.0 % | 90.0 % |
| ifeval | 100 | 94.0 % | 95.0 % | 94.0 % |
| toolcall | 97 | 60.8 % | 55.7 % | 49.5 % |
| needle | 68 | 100.0 % | 100.0 % | 100.0 % |
| mc | 100 | 64.0 % | 61.0 % | 67.0 % |
| decision | 66 | 87.9 % | 74.2 % | 83.3 % |
| **macro average** | 731 | **86.2 %** | **82.5 %** | 84.2 % |
| pooled difference | | | **-3.4 [-5.6, -1.2]** | -1.9 [-3.6, -0.3] |
| first tool as BF16 | 66 | 100 % | 53.0 % | 81.8 % |
| same tool and argument | 66 | 100 % | 45.5 % | 69.7 % |

### Teacher-forced domain KLD (nats/token, KL(BF16 || candidate))

| candidate | gsm8k | humaneval | mbpp | ifeval | toolcall | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HI2 3.0 bpw (this) | 0.04040 | 0.02121 | 0.05214 | 0.17396 | 0.03251 | 0.07701 |
| AWQ INT4 (4.625 bpw) | **0.00353** | **0.00191** | **0.00719** | **0.02002** | **0.00980** | **0.02323** |

## Limitations

- Decision fidelity: 53.0 % of the first tool calls match BF16, 45.5 % also match the argument,
  74.2 % are well formed (BF16: 100 % / 100 % / 100 %).
- mbpp: 84.0 % against BF16's 93.0 %.
- Domain KLD: ifeval 0.174 nats/token, decision 0.077.
- The comparison set has no other 3-bit text model: AWQ INT4 is 4.625 bpw, GGUF Q4_K_M is
  4.738 bpw.
- The serving kernels are gfx1030/gfx1100 only.
- Greedy output is not bitwise reproducible across server restarts (TunableOp GEMM selection,
  fp16 accumulation order).

## Serving

```bash
git clone https://github.com/BlivionIaG/vllm.git
cd vllm && git switch rdna_extras
pip install -e .        # the ROCm build

export VLLM_ROCM_USE_AITER=0                  # AITER is CDNA-only
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0          # rocBLAS on gfx1030
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export GPU_MAX_HW_QUEUES=2

vllm serve BlivionIaG/Qwen3.8-27B-HI2-3.0bpw \
  --dtype float16 --tensor-parallel-size 4
```

The `rdna_hint_flex` method is declared in `config.json`; no `--quantization` flag is needed.
`--dtype float16` is required: RDNA2 has no BF16 hardware. The 496 quantized linears are the
language model's; the vision tower (333 tensors) is kept in BF16.

## References

- Source model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- Quantizer: [BlivionIaG/hi2](https://github.com/BlivionIaG/hi2) — recipe in `docs/qad.md`,
  measurements in `docs/results.md`.
- Serving fork: [BlivionIaG/vllm](https://github.com/BlivionIaG/vllm) (`rdna_extras`).

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
