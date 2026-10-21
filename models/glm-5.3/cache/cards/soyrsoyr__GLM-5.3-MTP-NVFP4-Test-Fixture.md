---
base_model: zai-org/GLM-5.3
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: glm-5.3
tags:
  - compressed-tensors
  - llm-compressor
  - glm_moe_dsa
  - nvfp4
  - mtp
  - speculative-decoding
---

# GLM-5.3 Tiny NVFP4 MTP Test Fixture

This is a deterministic, random-weight architecture and serving fixture derived
directly from the released `zai-org/GLM-5.3` config and tokenizer. It uses the
`GlmMoeDsaForCausalLM` architecture, four small backbone layers, eight experts,
and one MTP layer in the released `model.layers.4` tensor layout. Runtime-critical
sparse MLA and DSA dimensions retain the released GLM-5.3 values.

The checkpoint does not inherit GLM-5.3 model weights and is not suitable for
quality evaluation. Its purpose is to test end-to-end NVFP4 compression and vLLM
MTP speculative decoding without downloading the full GLM-5.3 checkpoint.

## Compression Recipe

The backbone is calibrated and quantized inside `oneshot`. The `mtp_scheme`
argument asks the same call to compress the MTP weights during post-processing.

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

model_id = "/path/to/glm-5.3-tiny-mtp-source"
dataset = Dataset.from_dict(
    {
        "text": [
            "Explain why speculative decoding can reduce inference latency.",
            "Write a short Python function that adds two integers.",
            "Summarize the benefits of mixture-of-experts language models.",
            "What is the capital of France?",
            "List three considerations when deploying a language model.",
            "Describe the difference between weights and activations.",
            "Complete the sequence: one, two, three, four.",
            "Give a concise definition of quantization calibration.",
        ]
    }
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*mlp\.gate$",
        r"re:.*self_attn\.indexer\.(?:weights_proj|wk)$",
    ],
)

with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16")
model.generation_config.top_p = None

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    output_dir="GLM-5.3-MTP-NVFP4-Test-Fixture",
    max_seq_length=128,
    num_calibration_samples=len(dataset),
    moe_calibrate_all_experts=True,
    mtp_scheme="NVFP4",
)
```

Transformers does not construct the MTP layer, so it cannot observe MTP
activations during calibration. The MTP group is therefore weight-only NVFP4;
the backbone remains calibrated W4A4 NVFP4.

## Validation

- All 562 indexed tensors are present, with no missing or unindexed tensors.
- 105 backbone linears have NVFP4 packed weights, block scales, global scales,
  and calibrated input global scales.
- 33 MTP projections have NVFP4 packed weights, block scales, and global scales
  in `model_mtp.safetensors`; no uncalibrated MTP activation scales are stored.
- The NVFP4 checkpoint passed vLLM MTP speculative decoding on H100 with the
  Marlin FP4 fallback and one speculative token.
- Native W4A4 NVFP4 execution on Blackwell remains the final hardware smoke test.

## vLLM

```bash
vllm serve soyrsoyr/GLM-5.3-MTP-NVFP4-Test-Fixture \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```
