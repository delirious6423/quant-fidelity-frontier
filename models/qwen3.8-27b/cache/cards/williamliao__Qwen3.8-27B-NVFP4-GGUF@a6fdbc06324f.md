---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
tags:
  - gguf
  - llama.cpp
  - nvfp4
  - fp4
  - mtp
  - speculative-decoding
  - qwen3.8
  - qwen
pipeline_tag: text-generation
---

# Qwen3.8-27B NVFP4 GGUF

Experimental **Qwen3.8-27B NVFP4 GGUF** builds for **llama.cpp**, including two different conversion / quantization paths:

1. **Qwen3.8-27B-NVFP4-Quality-v2** — my custom mixed-NVFP4 quantization made from the BF16 model.
2. **Qwen3.8-27B-Unsloth-NVFP4-Q8** — converted from `unsloth/Qwen3.8-27B-NVFP4`.

These two files are **not equivalent quantizations** and should be treated as separate experiments.

Both are standalone target models with Qwen3.8's native MTP tensors included. They do not require a separate external draft model.

## Variant 1: Qwen3.8-27B-NVFP4-Quality-v2

This is my preferred llama.cpp-oriented NVFP4 build.

### Source and quantization path

```text
Qwen/Qwen3.8-27B BF16
        ↓
BF16 GGUF with native MTP
        ↓
llama-quantize with per-tensor overrides
        ↓
Qwen3.8-27B-NVFP4-Quality-v2.gguf
```

The goal is not to force every tensor to NVFP4. Instead, large compute-heavy matrices use NVFP4 while selected tensors remain at higher precision.

### Precision layout

Main transformer blocks:

- FFN down / gate / up: **NVFP4**
- Linear-attention QKV: **NVFP4**
- Linear-attention gate: **NVFP4**
- SSM output projection: **NVFP4**
- Full-attention Q projection: **NVFP4**
- Full-attention output projection: **NVFP4**
- Full-attention K / V: retained under the Q4_K_M mixed recipe
- Token embedding: **Q6_K**
- Output head: **Q6_K**
- Norm tensors: **F32**
- MTP FFN: **NVFP4**
- MTP `nextn.eh_proj`: retained under the Q4_K_M mixed recipe

The resulting dry-run size was:

```text
model size  = 52115.19 MiB (16.00 BPW)
quant size  = 15304.10 MiB (4.70 BPW)
```

This is roughly **16.0 GB decimal / 14.95 GiB** for the resulting GGUF.

### Quantization recipe

The model was first converted to BF16 GGUF with native MTP included, then quantized with `llama-quantize`.

Example:

```powershell
E:\git\llama.cpp\build\bin\Release\llama-quantize.exe `
  --token-embedding-type q6_k `
  --output-tensor-type q6_k `
  --tensor-type-file "F:\qwen38-nvfp4-quality-v2.txt" `
  "F:\Qwen3.8-27B-BF16-mtp.gguf" `
  "F:\Qwen3.8-27B-NVFP4-Quality-v2.gguf" `
  Q4_K_M
```

This is a **new quantization from BF16**, not a repack of the Unsloth NVFP4 checkpoint.

## Variant 2: Qwen3.8-27B-Unsloth-NVFP4-Q8

This variant is converted from:

```text
unsloth/Qwen3.8-27B-NVFP4
```

using a modified `convert_hf_to_gguf.py` with support for Qwen3.8 compressed-tensors mixed NVFP4 / FP8 layouts.

### Important: not a 1:1 reproduction of the Unsloth checkpoint

This GGUF is derived from `unsloth/Qwen3.8-27B-NVFP4`, but it is not a bit-identical or format-identical reproduction of the original compressed-tensors checkpoint.

During conversion:

- Packed NVFP4 tensors are repacked into llama.cpp's native GGUF NVFP4 representation.
- NVFP4 scale tensors are converted into the corresponding GGUF scale representation.
- FP8 tensors are dequantized by the converter.
- With `--fp8-as-q8`, those FP8 tensors are then written as **Q8_0** instead of preserving their original FP8 storage format.

As a result, the original Unsloth mixed-precision layout is not preserved exactly.

Qwen3.8-27B-Unsloth-NVFP4-Q8 should therefore be considered an Unsloth-derived NVFP4/Q8 GGUF conversion for llama.cpp, not a 100% faithful reproduction of unsloth/Qwen3.8-27B-NVFP4.

The NVFP4 portions are preserved through repacking where applicable, but the checkpoint's complete original mixed-precision representation is not reproduced exactly.

It should also not be assumed to behave identically to the original Unsloth checkpoint under Transformers, compressed-tensors, vLLM, or another reference runtime.

Therefore:

> **Qwen3.8-27B-Unsloth-NVFP4-Q8 is not a bit-identical, numerically identical, or 100% format-faithful copy of `unsloth/Qwen3.8-27B-NVFP4`.**

It is better described as an **Unsloth-derived NVFP4/Q8 GGUF conversion for llama.cpp**.

The native NVFP4 portions are preserved through repacking, but the checkpoint's complete original mixed-precision representation is not reproduced exactly.

It should also not be assumed to behave identically to the original Unsloth checkpoint under Transformers, compressed-tensors, vLLM, or another reference runtime.

### Conversion example

```powershell
python convert_hf_to_gguf.py `
  "E:\HF_MODELS\Qwen3.8-27B-NVFP4" `
  --outfile "E:\HF_MODELS\Qwen3.8-27B-Unsloth-NVFP4-Q8.gguf" `
  --outtype auto `
  --fp8-as-q8 `
  --verbose
```

Because of the FP8 → Q8_0 conversion, this GGUF is significantly larger than the custom Quality-v2 build and required a more even GPU split in my test setup.

## Compatibility

A recent **llama.cpp** build with:

- Qwen3.5/Qwen3.8 architecture support
- native NVFP4 tensor support
- native Qwen MTP speculative decoding

is required.

Tested on:

- Windows
- NVIDIA GeForce RTX 5070 Ti 16 GB
- NVIDIA GeForce RTX 5060 Ti 16 GB
- llama.cpp CUDA backend

Qwen3.8 is exposed through the `qwen35` architecture path in the tested llama.cpp build.

## Suggested llama-server settings

### Quality-v2 general-purpose setup

```bat
llama-server.exe ^
  -m "Qwen3.8-27B-NVFP4-Quality-v2.gguf" ^
  -np 1 ^
  --threads 12 ^
  --threads-batch 16 ^
  --split-mode layer ^
  --tensor-split 70,30 ^
  --ctx-size 81920 ^
  --no-mmap ^
  -ngl -1 ^
  --flash-attn on ^
  --jinja ^
  --ubatch-size 256 ^
  --batch-size 2048 ^
  --fit off ^
  --reasoning off ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 3 ^
  --spec-draft-p-min 0.60
```

For this mixed-task benchmark, **`n_max = 3`** gave the best aggregate wall-clock result.

`n_max = 4` improved some highly predictable workloads such as JSON, repeated patterns, and code completion, but was slower overall.

## Benchmark: Quality-v2

### Base, MTP disabled

Configuration:

```text
split-mode: layer
tensor-split: 70,30
```

```text
code_python        36.3 tok/s
code_cpp           36.6 tok/s
explain_concept    36.3 tok/s
summarize          36.6 tok/s
qa_factual         36.0 tok/s
translation        37.1 tok/s
creative_short     36.9 tok/s
stepwise_math      36.2 tok/s
json_output        36.0 tok/s
long_reasoning     36.2 tok/s
repeat_pattern     36.4 tok/s
code_completion    36.2 tok/s
long_code_review   36.0 tok/s

total wall time: 53.18 s
```

### MTP `n_max = 3`

```text
code_python        pred= 192 draft= 158 acc= 138 rate=0.873 tok/s=79.1
code_cpp           pred=  53 draft=  42 acc=  40 rate=0.952 tok/s=81.4
explain_concept    pred= 192 draft= 233 acc= 112 rate=0.481 tok/s=54.7
summarize          pred=  47 draft=  45 acc=  31 rate=0.689 tok/s=68.2
qa_factual         pred= 192 draft= 181 acc= 130 rate=0.718 tok/s=70.0
translation        pred=  17 draft=  18 acc=  12 rate=0.667 tok/s=60.9
creative_short     pred=  43 draft=  66 acc=  22 rate=0.333 tok/s=43.8
stepwise_math      pred= 192 draft= 159 acc= 137 rate=0.862 tok/s=78.4
json_output        pred= 192 draft= 148 acc= 141 rate=0.953 tok/s=83.4
long_reasoning     pred= 192 draft= 180 acc= 131 rate=0.728 tok/s=71.1
repeat_pattern     pred= 192 draft= 143 acc= 143 rate=1.000 tok/s=88.0
code_completion    pred= 142 draft= 117 acc= 105 rate=0.897 tok/s=79.9
long_code_review   pred= 192 draft= 242 acc= 109 rate=0.450 tok/s=52.6

Aggregate:
  requests:          13
  predicted tokens:  1838
  draft tokens:      1732
  accepted tokens:   1251
  acceptance rate:   72.23%
  total wall time:   29.11 s
```

### MTP `n_max = 4`

```text
code_python        pred= 192 draft= 157 acc= 145 rate=0.924 tok/s=78.9
code_cpp           pred=  53 draft=  47 acc=  39 rate=0.830 tok/s=78.4
explain_concept    pred= 192 draft= 188 acc= 109 rate=0.580 tok/s=44.2
summarize          pred=  47 draft=  48 acc=  33 rate=0.688 tok/s=60.3
qa_factual         pred= 192 draft= 173 acc= 131 rate=0.757 tok/s=59.7
translation        pred=  17 draft=  17 acc=  13 rate=0.765 tok/s=50.8
creative_short     pred=  43 draft=  27 acc=  19 rate=0.704 tok/s=35.6
stepwise_math      pred= 192 draft= 170 acc= 140 rate=0.824 tok/s=71.7
json_output        pred= 192 draft= 151 acc= 150 rate=0.993 tok/s=89.4
long_reasoning     pred= 192 draft= 182 acc= 135 rate=0.742 tok/s=65.8
repeat_pattern     pred= 192 draft= 152 acc= 152 rate=1.000 tok/s=99.2
code_completion    pred= 142 draft= 122 acc= 108 rate=0.885 tok/s=83.1
long_code_review   pred= 192 draft= 182 acc= 107 rate=0.588 tok/s=44.6

Aggregate:
  requests:          13
  predicted tokens:  1838
  draft tokens:      1616
  accepted tokens:   1281
  acceptance rate:   79.27%
  total wall time:   31.35 s
```

## Benchmark: Unsloth-derived NVFP4-Q8

Because this GGUF is larger, the tested configuration used:

```text
split-mode: layer
tensor-split: 60,40
```

### Base, MTP disabled

```text
code_python        25.9 tok/s
code_cpp           26.3 tok/s
explain_concept    25.8 tok/s
summarize          26.4 tok/s
qa_factual         26.1 tok/s
translation        27.2 tok/s
creative_short     26.1 tok/s
stepwise_math      26.1 tok/s
json_output        26.0 tok/s
long_reasoning     26.0 tok/s
repeat_pattern     26.1 tok/s
code_completion    26.0 tok/s
long_code_review   25.9 tok/s

total wall time: 75.10 s
```

### MTP `n_max = 4`

```text
code_python        pred= 192 draft= 178 acc= 142 rate=0.798 tok/s=53.0
code_cpp           pred=  54 draft=  44 acc=  39 rate=0.886 tok/s=53.1
explain_concept    pred= 192 draft= 181 acc= 105 rate=0.580 tok/s=30.9
summarize          pred=  45 draft=  50 acc=  31 rate=0.620 tok/s=40.4
qa_factual         pred= 192 draft= 174 acc= 128 rate=0.736 tok/s=40.3
translation        pred=  17 draft=  18 acc=  13 rate=0.722 tok/s=35.2
creative_short     pred=  37 draft=  28 acc=  17 rate=0.607 tok/s=25.0
stepwise_math      pred= 192 draft= 176 acc= 140 rate=0.795 tok/s=49.8
json_output        pred= 192 draft= 152 acc= 148 rate=0.974 tok/s=60.4
long_reasoning     pred= 192 draft= 161 acc= 130 rate=0.807 tok/s=43.5
repeat_pattern     pred= 192 draft= 152 acc= 152 rate=1.000 tok/s=68.6
code_completion    pred= 191 draft= 162 acc= 150 rate=0.926 tok/s=63.7
long_code_review   pred= 192 draft= 167 acc= 113 rate=0.677 tok/s=34.4

Aggregate:
  requests:          13
  predicted tokens:  1880
  draft tokens:      1643
  accepted tokens:   1308
  acceptance rate:   79.61%
  total wall time:   44.70 s
```

## Comparison

The custom Quality-v2 build is smaller and substantially faster on the tested dual-GPU system.

However, the two files are not an apples-to-apples quantization comparison:

- Quality-v2 is a new mixed quantization generated from BF16.
- The Unsloth-derived build repacks the source NVFP4 tensors but converts source FP8 tensors to Q8_0.
- The Unsloth-derived file is larger and required a different GPU split (`60,40` instead of `70,30`).
- Different tensor layouts and GPU splits affect performance independently of model quality.

Therefore these benchmark numbers should be interpreted as **practical llama.cpp deployment results**, not as proof that one quantization method has universally better model quality.

No model-quality benchmark against BF16 was performed here.

## Notes

- Both GGUFs contain the full target model and native MTP tensors.
- Native MTP speculative decoding changes generation throughput but does not provide the same type of acceleration for prompt prefill.
- Higher draft acceptance does not necessarily mean lower wall-clock time.
- Predictable outputs such as JSON, repeated patterns, and code completion benefit more from longer MTP drafts.
- Open-ended explanations, creative writing, and long code review generally lose speculative efficiency sooner.
- Performance depends heavily on llama.cpp build, GPU split, context size, KV-cache format, sampling parameters, and PCIe topology.
- The Unsloth-derived build should not be described as a 100% faithful reproduction of the original Unsloth compressed-tensors checkpoint.

## Credits

- **Qwen Team / Alibaba Cloud** — Qwen3.8-27B
- **Unsloth** — Qwen3.8-27B-NVFP4 source checkpoint used for the derived conversion experiment
- **ggml-org** — llama.cpp, GGUF, NVFP4 inference support, and native MTP support

## License

The source model is distributed under the **Apache License 2.0**.

Users should review the upstream `Qwen/Qwen3.8-27B` and `unsloth/Qwen3.8-27B-NVFP4` model cards before redistribution or commercial use.
