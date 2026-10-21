---
license: mit
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: Ornith-1.5-35B-A3B-TQ3_4S
tags:
- gguf
- llama.cpp
- qwen
- qwen3.5
- qwen3.5-moe
- multimodal
- vision
- ornith
- turboquant
- tq3_4s
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
model-index:
- name: Ornith-1.5-35B-A3B-TQ3_4S
  results: []
---

# Ornith-1.5-35B-A3B-TQ3_4S

![Ornith-1.5-35B-A3B-TQ3_4S](thumbnail.png)

`Ornith-1.5-35B-A3B-TQ3_4S` is a compact TurboQuant GGUF build of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## Required Runtime

This model uses the custom `TQ3_4S` tensor type and requires [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3).

> Stock `llama.cpp` builds without TurboQuant support cannot load this model.

This is the standard 35B-A3B MoE model, not an MTP release. Do not add draft-MTP speculative-decoding flags.

`Ornith-1.5-35B-A3B` is a Qwen3.5-based **multimodal** model. Vision is optional — load the `mmproj` projector to enable image input (see [Vision (Multimodal)](#vision-multimodal)).

## Files

- [`Ornith-1.5-35B-A3B-TQ3_4S.gguf`](Ornith-1.5-35B-A3B-TQ3_4S.gguf) - main model, 17.00 GiB
- [`thumbnail.png`](thumbnail.png) - model card banner

## Architecture

| Property | Value |
|---|---|
| Architecture | Qwen3.5-MoE (qwen35moe) |
| Total parameters | 35.9B |
| Active parameters | ~3B per token |
| Expert count | 256 |
| Active experts | 8 |
| Block count | 41 |
| Attention interval | Every 4th layer (full attention) |
| SSM layers | DeltaNet (3/4 of layers) |
| Context length | 262,144 (source) |
| Quantization | TQ3_4S (4.06 BPW) |

## Build the Required Runtime

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3
cd llama.cpp-tq3

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_GRAPHS=ON

cmake --build build --target llama-server -j
```

For an RTX 3090, `-DCMAKE_CUDA_ARCHITECTURES=86` may be added explicitly. Use the architecture matching your GPU on other systems.

## Recommended Runtime

Validated on an NVIDIA GeForce RTX 3090 Founders Edition with 24 GiB VRAM:

```bash
./build/bin/llama-server \
  -m Ornith-1.5-35B-A3B-TQ3_4S.gguf \
  --alias Ornith-1.5-35B-A3B-TQ3_4S \
  --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --reasoning off --jinja
```

Runtime notes:

- `-fa on` enables flash attention at runtime.
- The validated CUDA build uses `GGML_CUDA_FA_ALL_QUANTS=OFF`.
- `-ngl 99` fully offloads the model on supported GPUs. Avoid partial offload when comparing the published speed.
- Reduce context from `32768` if the available VRAM is lower than 24 GiB.

## Quick Smoke Test

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Ornith-1.5-35B-A3B-TQ3_4S","messages":[{"role":"user","content":"Write ONLY the word ok."}],"max_tokens":16,"temperature":0.0}'
```

Expected assistant content:

```text
ok
```

## Local Measurements

| Metric | Result |
|---|---:|
| Generation speed (short ctx) | 171.1 tok/s |
| Size | 17.00 GiB |
| Quantization | TQ3_4S (4.06 BPW) |

Measured on RTX 3090 FE with the launch settings above. Generation speed is token throughput excluding prompt processing.

## Context-Length Decode Speed

Decode throughput measured with 512-token generation across varying prompt lengths. Compared against [Ornith-1.0-35B-TQ3_4S](https://huggingface.co/YTan2000/Ornith-1.0-35B-TQ3_4S) (dense 35B, 13 GiB) on the same hardware and binary.

| Context Size | Ornith-1.5 MoE (tok/s) | Ornith-1.0 Dense (tok/s) | Speedup |
|---|---:|---:|---:|
| 4K | 168.9 | 156.2 | +8.1% |
| 8K | 163.4 | 153.6 | +6.4% |
| 16K | 147.9 | 140.1 | +5.6% |
| 32K | 125.7 | 118.9 | +5.7% |
| 64K | 95.5 | 91.9 | +3.9% |
| 128K | 89.2 | 85.5 | +4.3% |
| 256K | 88.6 | 84.4 | +5.0% |

Runtime: [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) commit `47635d703`, `-ngl 99 -fa on -ctk q8_0 -ctv tq3_0 -c 262144 --reasoning off`, NVIDIA RTX 3090 24GB.

The MoE architecture (3B active parameters) delivers a consistent 4–8% decode speedup over the dense 35B variant across all context sizes.

## Benchmark Summary

| Suite | Score | Pass Rate |
|-------|-------|-----------|
| **Coding** | 93.8% | 11/12 tasks |
| **Tool Calling** | 91.7% | 13/15 tasks |
| **Data Extraction** | 85.8% | 10/15 tasks |
| **Instruction Following** | 64.5% | 5/15 tasks |
| **Hard86 Coding** | 47.7% | 41/86 tests |

**Notes:**
- All benchmarks run with `--reasoning off` for maximum speed
- MoE architecture (3B active parameters) enables fast inference
- Coding and tool calling scores are competitive with dense 35B models
- Instruction following shows room for improvement on complex multi-constraint tasks

## Conversion Recipe

```bash
# 1. Convert safetensors to F16 GGUF
python convert_hf_to_gguf.py \
  ornith-ai/Ornith-1.5-35B-A3B \
  --outfile Ornith-1.5-35B-A3B-F16.gguf \
  --outtype f16

# 2. Quantize to TQ3_4S
./build/bin/llama-quantize \
  Ornith-1.5-35B-A3B-F16.gguf \
  Ornith-1.5-35B-A3B-TQ3_4S.gguf \
  tq3_4s
```

## Vision (Multimodal)

`Ornith-1.5-35B-A3B` is a multimodal model. To enable image input, add the vision projector with `--mmproj`:

```bash
./build/bin/llama-server \
  -m Ornith-1.5-35B-A3B-TQ3_4S.gguf \
  --mmproj mmproj-f16.gguf \
  --alias Ornith-1.5-35B-A3B-TQ3_4S \
  --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --reasoning off --jinja
```

## Validation Notes

- Benchmark results are local measurements, not claims from the parent model repository.

## License

Use is subject to the [base model](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) license and the licenses of [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3).
