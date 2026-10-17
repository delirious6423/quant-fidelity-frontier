---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- deepseek
- deepseek-v4.1
- vision-language
- multimodal
- fp8
- abliterated
- uncensored
---

# DeepSeek-V4.1-Flash-FP8

DeepSeek-V4.1-Flash-FP8 is an optimized multimodal vision-language model designed for low-latency inference and reduced memory consumption. Quantized to 8-bit floating-point (FP8), this checkpoint offers high throughput for visual understanding and text generation tasks without significant accuracy degradation.

## Model Summary

* **Developer:** DeepSeek-AI
* **Model Type:** Multimodal (Image-Text-to-Text)
* **Precision:** FP8 (8-bit Floating Point)
* **License:** MIT
* **Framework:** PyTorch / Hugging Face Transformers

## Key Features

* **High Efficiency:** Optimized FP8 weights lower GPU memory footprint and accelerate token generation.
* **Multimodal Reasoning:** Handles joint visual and textual inputs for document understanding, image captioning, and visual question answering (VQA).
* **Flexible Deployment:** Compatible with standard `transformers` workflows, vLLM, and TensorRT-LLM engines.

## Usage

### Using Transformers

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "your-username/DeepSeek-V4.1-Flash-FP8"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

image = Image.open("example.jpg")
prompt = "<image>\nDescribe the image in detail."

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
