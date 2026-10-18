---
license: mit
library_name: gguf
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
language:
- en
tags:
- gguf
- llama-cpp
- deepseek-v4
- conversational
- behaviorally-modified
- refusal-reduced
- not-for-all-audiences
- million-context
---

# DeepRiver V4 Flash — GGUF

**DeepRiver V4 Flash** is a quantization-aware, behaviorally modified
derivative of DeepSeek V4 Flash, created by **A.I Joe** and published by
**BlackRiver AI Ltd**.

This release uses Unsloth's `UD-Q2_K_XL` DeepSeek-V4-Flash-0731 GGUF as its quantization foundation
and applies the DeepRiver refusal-direction projection directly to 33
`attn_output_b` tensors while preserving the original GGUF structure,
tokenizer, chat template, tensor layout, and all non-target bytes.

> This model is not for all audiences. It exhibits substantially reduced
> refusal behavior and may produce unsafe, disturbing, illegal, inaccurate, or
> otherwise objectionable content.

## Model details

| Property | Value |
|---|---|
| Upstream model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| GGUF foundation | Unsloth `UD-Q2_K_XL` |
| Architecture | DeepSeek V4 |
| Parameters reported by llama.cpp | 284,334,567,511 |
| Activated parameters | approximately 13B |
| Context trained | 1,048,576 tokens |
| Tested context | 1,048,576 tokens |
| Approximate repository weight size | 96.8 GB |
| Shards | 3 |
| Creator | A.I Joe |
| Publisher | BlackRiver AI Ltd |
| License | MIT |

## Tested llama.cpp settings

```ini
ctx-size = 1048576
parallel = 1
n-gpu-layers = 999
flash-attn = on
fit = off
no-warmup = true
kv-unified = true
batch-size = 256
ubatch-size = 64
cache-type-k = q4_0
cache-type-v = q4_0
temp = 1.0
top-p = 1.0
top-k = 0
min-p = 0.0
repeat-penalty = 1.0
presence-penalty = 0.0
frequency-penalty = 0.0
cache-ram = 0
```

Load the first shard; llama.cpp discovers the remaining shards in the same
directory.

```powershell
llama-server.exe `
  --model ".\DeepRiver-V4-Flash-00001-of-00003.gguf" `
  --ctx-size 1048576 `
  --n-gpu-layers 999 `
  --flash-attn on `
  --jinja
```

## Tested hardware

Validated on an HP ZBook Ultra G1a 14 with:

- AMD Ryzen AI Max+ PRO 395
- Radeon 8060S
- 128 GB unified memory
- llama.cpp Vulkan
- Windows 11

A post-branding deterministic test measured approximately **12.67 tokens/s**
decode on this machine. Performance varies by build, backend, context length,
KV-cache format, and hardware.

## What was modified

The modification targets layers 10–42 inclusive:

```text
blk.10.attn_output_b.weight
...
blk.42.attn_output_b.weight
```

The edited tensors were generated from clean BF16 masters, projected with
row-norm preservation, encoded directly as llama.cpp-compatible Q8_0 payloads,
and written into the matching pre-existing Q8_0 tensor ranges.

See [PROVENANCE.md](PROVENANCE.md) for the complete recipe and hashes.

## Evaluation

The model passed structural, integrity, load, inference, memory, tool-use,
coding, and targeted behavioral checks. Both DeepRiver variants showed a
deterministic, broad refusal-boundary shift in the tested classification
suite.

See [EVALUATION.md](EVALUATION.md) for the exact scope and caveats.

## Limitations and responsible use

DeepRiver is not claimed to be universally uncensored, alignment-free, safe,
truthful, or suitable for every task. Reduced refusal behavior can increase
the probability of harmful or unlawful outputs. Users are responsible for
complying with applicable laws, platform rules, and deployment safeguards.

Do not use the model as the sole basis for high-stakes decisions. Do not
deploy it to untrusted users without appropriate access control, monitoring,
rate limiting, and downstream safety measures.

## Credits

- Base model: DeepSeek-AI
- GGUF quantization foundation: Unsloth
- DeepRiver modification and release engineering: A.I Joe
- Publisher: BlackRiver AI Ltd

See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).