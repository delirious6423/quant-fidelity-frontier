---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- fp8
- compressed-tensors
- llm-compressor
- mtp
- speculative-decoding
pipeline_tag: text-generation
---

# Qwen3.8-27B-FP8-DYNAMIC-MTP

FP8 (dynamic) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
**including the Multi-Token Prediction (MTP) draft layer**, produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor).

- **Weights:** FP8, channel-wise scales
- **Activations:** FP8, dynamic per-token
- **MTP layer:** quantized to FP8 dynamic and written as a dedicated
  `model_mtp.safetensors` shard with its own `mtp_group` in the quantization config
- **Format:** compressed-tensors

## Recipe

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: FP8_DYNAMIC
```

The MTP layer is quantized through `oneshot(..., mtp_quant_scheme="FP8_DYNAMIC")`.

<details><summary>Full quantization script</summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")

oneshot(
    model=model,
    processor=processor,
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="Qwen3.8-27B-FP8-DYNAMIC-MTP",
    mtp_quant_scheme="FP8_DYNAMIC",
)
```
</details>

## Serving with MTP speculative decoding (vLLM)

```bash
vllm serve soyrsoyr/Qwen3.8-27B-FP8-DYNAMIC-MTP \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --max-num-seqs 256
```

## MTP performance

Measured with vLLM 0.28 serving this checkpoint with the MTP layer as the
speculative draft (`num_speculative_tokens=1`):

| Metric | Value |
|---|---|
| MTP draft acceptance rate | ~84% (292 / 347 draft tokens accepted) |
| Speculative tokens per step | 1 |

The quantized MTP draft achieves a high acceptance rate, confirming the FP8 MTP
layer works correctly for speculative decoding after quantization.
