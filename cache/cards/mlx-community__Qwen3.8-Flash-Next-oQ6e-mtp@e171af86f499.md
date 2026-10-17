---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- oq
- quantized
---

# Qwen3.8-Flash-Next-oQ6e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) in [oMLX](https://huggingface.co/Jundot) v0.6.4. oQe (imatrix). MTP kept.

## Quantization details

- **Model type**: qwen4_exp
- **Bits**: 6
- **Group size**: 64
- **Format**: MLX safetensors
- **oMLX**: [Jundot](https://huggingface.co/Jundot) v0.6.4 ([source](https://github.com/jundot/omlx))
- **Parameters**: 125B total, 6B activated, 51B n-gram embedding, 4B MTP
- **MTP**: preserved (76 `mtp.*` tensors)
- **Source**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), converted to MLX, then oQe with `preserve_mtp`

## Benchmark

Apple M3 Ultra, 256 GB. oMLX 0.6.4. Thinking off, temperature 0, Lightning MTP depth 3. oQ4e is [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) on the same machine.

Write speed is how fast the reply is generated (`generation_tokens_per_second`). Read speed is how fast the prompt is ingested (`prompt_tokens_per_second`).

Coding task: write a Python Fibonacci function, no comments. All three packs stopped on their own.

| | oQ4e | oQ6e | oQ8e |
| --- | --- | --- | --- |
| Time to completion | 1.74 s | 2.07 s | 2.57 s |
| Time to first token | 0.51 s | 0.64 s | 0.67 s |
| Tokens written | 79 | 80 | 80 |
| Write speed (tok/s) | 64.4 | 56.1 | 42.1 |
| Read speed (tok/s) | 51.2 | 40.3 | 38.9 |

Longer prompts (256-token generation cap):

### Write speed (tok/s)

| Context | oQ4e | oQ6e | oQ8e |
| --- | --- | --- | --- |
| Short code | 64.4 | 56.1 | 42.1 |
| 4K | 56.5 | 44.9 | 41.9 |
| 8K | 56.3 | 44.8 | 43.7 |
| 16K | 52.9 | 48.3 | 47.5 |

### Read speed (tok/s)

| Context | oQ4e | oQ6e | oQ8e |
| --- | --- | --- | --- |
| Short code | 51.2 | 40.3 | 38.9 |
| 4K | 732 | 597 | 581 |
| 8K | 772 | 728 | 716 |
| 16K | 729 | 785 | 763 |

Raw timings: `bench.json` (this pack) and `bench-compare.json` (all three).
