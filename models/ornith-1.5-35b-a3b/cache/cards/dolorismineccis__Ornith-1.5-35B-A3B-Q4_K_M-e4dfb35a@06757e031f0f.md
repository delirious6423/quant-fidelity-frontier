---
library_name: llama.cpp
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
- gguf
- q4_k_m
- qwen3.5-moe
- vision-language
- experimental
language:
- en
- zh
pipeline_tag: image-text-to-text
---

# Ornith-1.5-35B-A3B Q4_K_M — locked e4dfb35a

Uncut `Q4_K_M` GGUF derived directly from the locked source revision:

- Base model: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- Source revision: `e4dfb35a93d4b6822a811a7676f3488514abe7e2`
- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Experts: 256 total, top-8 active; **no expert pruning**
- Quantization: `Q4_K_M`
- Converter/quantizer: `llama.cpp@6d05498314db1b57f81c271080018aa2d0b89be9`

## Important quality warning

This checkpoint is published as an **experimental, reproducible artifact**, not as a deployment-approved model.

The locked same-runtime BF16-to-Q4 static comparison produced:

- Text paired retention: **81.82%**
- Text success: **27/100**, average score **55.59**
- Severe-risk failures: `TXT-A10`, `TXT-A15`, `TXT-S03`
- Vision paired retention: **100%**
- Vision success: **12/20**, average score **79.20**, zero severe-risk failures

Therefore the project static gate marked this checkpoint `deployment_disqualified=true`. Do not treat it as safety-qualified or production-ready. See `evaluation/` for the recorded summaries and paired comparison.

## Files

| File | Size | SHA-256 |
|---|---:|---|
| `Ornith-1.5-35B-A3B-Q4_K_M.gguf` | 21,713,462,752 bytes | `dbc118e7211e30b76256fdb80f08095a60c2e2ebc5cd8c0c2cf3528c6639acec` |
| `mmproj-Ornith-1.5-35B-A3B-BF16.gguf` | 902,822,304 bytes | `2b1c6c0ccf44ac8d30330e72b905ef1eb3364cd59bfeaea2f1c5741ea0399f4d` |

`quantization-manifest.json` and `source-validation.json` record provenance and source validation.

## llama.cpp example

```bash
llama-server \
  -m Ornith-1.5-35B-A3B-Q4_K_M.gguf \
  --mmproj mmproj-Ornith-1.5-35B-A3B-BF16.gguf \
  -c 8192 --host 127.0.0.1 --port 8080
```

Tune GPU offload, context, batch size, and cache types for your hardware. The text-only server may omit `--mmproj`.

## Provenance and license

The source snapshot was validated at the exact revision above: 28 files and 71,926,950,570 verified bytes. The upstream model card declares the MIT license; consult the upstream repository for its current license terms. Model behavior and benchmark results remain subject to the limitations above.
