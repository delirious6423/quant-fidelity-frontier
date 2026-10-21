---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- gptq
- int3
- w3a16
- compressed-tensors
- quantized
- vision-language
- qwen3_5
---

# Qwen3.8-27B-W3A16-GPTQ

3-bit weight-quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with GPTQ via [llm-compressor](https://github.com/vllm-project/llm-compressor).

**The vision tower is left in bf16 — only the language decoder is quantized.**
The model remains a fully functional vision-language model; image+text inference works as with the base model.

| | |
|---|---|
| Weights | INT3, group size 128, symmetric |
| Activations | bf16 (unquantized) |
| Format | `compressed-tensors` / `pack-quantized` |
| Quantized | language decoder (256 Linear modules) |
| Kept in bf16 | vision tower, `lm_head`, linear-attention projections |
| Size | 55.6 GB → 22.8 GB (~2.4x smaller) |

## Requirements

This model uses the `qwen3_5` architecture, which requires **transformers >= 5.14**:

```bash
pip install "transformers>=5.14" compressed-tensors
```

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model_id = "jaehyeono/Qwen3.8-27B-W3A16-GPTQ"

model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="auto", device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image in one sentence."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[Image.open("example.jpg")], return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(out[0], skip_special_tokens=True))
```

## Quantization details

Quantized with `GPTQModifier` from llm-compressor 0.13.0 (compressed-tensors 0.18.0).

| Parameter | Value |
|---|---|
| Bits | 3 |
| Group size | 128 |
| Symmetric | true |
| Strategy | group |
| Activation ordering | static |
| Dampening fraction | 0.01 |
| Observer | `memoryless_minmax` |
| Calibration | 128 image–caption pairs from Flickr30k |
| Seed | 42 |

The full recipe is included as `recipe.yaml`.

### Why the vision tower is not quantized

GPTQ needs activation statistics for every module it touches. The vision tower contributes
a small fraction of total parameters (depth 27, hidden 1152) compared to the language decoder
(64 layers, hidden 5120), so excluding it costs little compression while avoiding degradation
in the visual pathway. Linear-attention projections and `lm_head` are likewise excluded, per
common practice for these layers.


## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
