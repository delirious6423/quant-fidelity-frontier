---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
  - vllm
  - compressed-tensors
  - fp8
  - gemma4
  - quantized
---

# Gemma 4 26B-A4B Instruct — FP8 Dynamic

FP8 (E4M3) **dynamic** quantization of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it),
stored in the **compressed-tensors** format. Produced as an in-house build for
full checkpoint provenance (supplier-assurance / audit), as an alternative to
third-party prebuilt checkpoints.

- **Weights:** static **per-channel** FP8 (E4M3).
- **Activations:** **per-token dynamic** FP8 — no calibration data required.
- **Kept at original precision (BF16):** MoE router/gate, token embeddings,
  `lm_head`, all norms, and the vision tower (this is a **text-only** serving
  checkpoint).
- **MoE experts:** quantized per-expert (`experts.{i}.{gate,up,down}_proj`),
  the standard compressed-tensors MoE layout.
- **Size:** ~26 GB (vs ~49 GB BF16).

## Why FP8 (and not FP4 / NVFP4)

Target hardware is **NVIDIA L40S (Ada, SM 8.9)**, which has native FP8 Tensor
Cores but **no** native FP4. FP8 runs on the fast native path on Ada/Hopper/
Blackwell; the compressed-tensors checkpoint is hardware-portable.

## Quantization recipe

Built with [`llm-compressor`](https://github.com/vllm-project/llm-compressor)
using the data-free `model_free_ptq` entry point:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="google/gemma-4-26B-A4B-it",
    save_directory="gemma-4-26B-A4B-it-FP8-Dynamic",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*embed.*", "re:.*router", "re:.*vision_tower.*", "re:.*norm.*"],
)
```

> Note: `re:.*norm.*` is required for Gemma 4 because some norms use a numeric
> suffix (e.g. `post_feedforward_layernorm_1`) that escapes the default
> "ends-with-norm" auto-ignore and would otherwise be (incorrectly) targeted.

## Usage (vLLM)

The `compressed-tensors` format is auto-detected — do **not** pass
`--quantization`. Requires an upstream vLLM with Gemma 4 + compressed-tensors
MoE support.

```bash
vllm serve SilentEight/gemma-4-26B-A4B-it-FP8-Dynamic \
  --served-model-name gemma \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice --tool-call-parser gemma4
```

Gemma 4 supports tool calling and a thinking channel (`enable_thinking`); enable
the matching parsers as above.

## Validation

- Checkpoint structure (keys / dtypes / shapes) matches the reference
  `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` build.
- Quantization integrity verified: experts are `F8_E4M3` with per-channel
  `weight_scale`; router/norms/embeddings/`lm_head` left in BF16.
- **Not yet benchmarked** for quality regression vs BF16. Run your own eval
  (e.g. a task-relevant benchmark) before production use.

## License

This is a quantized derivative of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it),
which Google distributes under the **Apache License 2.0**. This checkpoint is
released under the same **Apache 2.0** license.

Original model license: [Gemma 4 license — Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license).
