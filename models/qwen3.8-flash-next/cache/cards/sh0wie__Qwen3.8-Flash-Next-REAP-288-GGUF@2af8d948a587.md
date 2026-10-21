---
license: other
license_name: qwen-community-license-1.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- moe
- pruning
- reap
- qwen
---

# Qwen3.8-Flash-Next REAP-288 (GGUF)

This is the REAP-288 build in GGUF, for llama.cpp and the tools built on it
(Ollama, LM Studio, and similar). It is the 180B-class MoE with its experts
pruned 512 -> 288 per layer, converted directly from the
[bf16 source](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-bf16)
in a single quantization step, so each file loses accuracy exactly once rather
than stacking a re-quantization on top of an existing 4-bit build.

The pruned model scores 91.5% on HumanEval (149 of 164), against 93.9% for the
unpruned full model. That number is measured on the MLX 4-bit lineage; the GGUF
quants share the same kept-expert selection and derive from the same bf16
source.

If you run MLX on Apple Silicon instead of llama.cpp, use the
[4-bit MLX build](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit).
The pmlx engine and its speedups apply to the MLX builds, not to GGUF; nothing
on this card depends on it.

## Files

Converted from the bf16 source. Pick one quant; larger files are higher
fidelity and slower. Take `Q4_K_M` for the smallest, fastest build, `Q8_0` for
the highest quality, and `Q5_K_M` if you want a middle point.

| Quant | Size | Notes |
| --- | --- | --- |
| `Q4_K_M` | 78 GB | good default: smallest and fastest, strong quality |
| `Q5_K_M` | 87 GB | higher fidelity, more memory |
| `Q8_0` | 116 GB | near-lossless, largest and highest quality |

## Run it

llama.cpp:

```bash
# build llama.cpp, then:
llama-server -m Qwen3.8-Flash-Next-REAP-288-Q4_K_M.gguf --port 8080
# or a one-shot completion:
llama-cli   -m Qwen3.8-Flash-Next-REAP-288-Q4_K_M.gguf -p "Refactor this function to add input validation."
```

Ollama (pulls the quant straight from this repo):

```bash
ollama run hf.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF:Q4_K_M
```

## Provenance

- `Qwen/Qwen3.8-Flash-Next`: upstream weights.
- [sh0wie/Qwen3.8-Flash-Next-REAP-288-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-bf16):
  the full-precision REAP-288 source these files are quantized from. REAP
  expert pruning 512 -> 288 per layer, calibrated on-device over ~686K tokens
  of agentic-coding traffic.

## Limitations

- Calibration reflects one team's agentic-coding distribution. Retention
  numbers should not be read as general-domain; domains far from code may
  degrade more.
- Single-run evaluations on the lineage, no confidence intervals. Vision input
  is untested after pruning.

## License

Qwen Community License 1.0, inherited from the base model
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
