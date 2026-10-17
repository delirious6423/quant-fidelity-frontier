---
license: apache-2.0
library_name: onnxruntime-genai
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
  - onnx
  - onnxruntime-genai
  - qwen
  - qwen3.8
  - cpu
  - int4
  - vision
language:
  - en
  - zh
---

# Qwen3.8-27B CPU ONNX (FP32 vision/embedding, INT4 text)

ONNX Runtime GenAI package of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for **CPU**.

| Subgraph | Precision | Notes |
|---|---|---|
| Vision encoder (`vision.onnx`) | FP32 | Unquantized |
| Token embedding (`embedding.onnx`) | FP32 | Unquantized; bit-exact vs PyTorch |
| Text decoder (`text.onnx`) | INT4 | Olive ModelBuilder `k_quant`, group/block size **128**, `accuracy_level=4`. All 497 `MatMulNBits` weights are 4-bit. |

This is a conversion of the base model, not a fine-tune. License and intended use follow the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).

## Files

```
embedding.onnx + embedding.onnx.data   # FP32 token embedding + image-feature scatter (~5.1 GB)
vision.onnx + vision.onnx.data         # FP32 vision encoder (~3.7 GB)
text.onnx + text.onnx.data             # INT4 hybrid decoder (~12.9 GB)
genai_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
config.json
generation_config.json
```

## Usage

Requires a recent ONNX Runtime GenAI build with Qwen3.5 / Qwen3.8 (`model.type: qwen3_5`) support.

```bash
pip install onnxruntime-genai huggingface_hub
huggingface-cli download amd/Qwen3.8-27B-fp32-ve-fp32-int4-k_quant-gs128-text-cpu-onnx --local-dir ./qwen38-27b-cpu-onnx
```

```python
import json
import onnxruntime_genai as og

model = og.Model("./qwen38-27b-cpu-onnx")
tokenizer = og.Tokenizer(model)
processor = model.create_multimodal_processor()
stream = processor.create_stream()

messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
inputs = processor(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=256)
generator = og.Generator(model, params)
generator.set_inputs(inputs)

while not generator.is_done():
    generator.generate_next_token()
    print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
```

Image input: pass `images=og.Images.open("photo.jpg")` into the processor with a chat message that includes `{"type": "image"}`.

## Notes

- CPU-only: `genai_config.json` uses the default CPU execution provider (empty `provider_options`).
- Search defaults in `genai_config.json` are greedy (`do_sample: false`, `top_k: 1`).
- Chat stop tokens are `<|im_end|>` (248046) and `<|endoftext|>` (248044).
- Tokenizer Split rules that used Unicode property escapes were replaced with ByteLevel (`use_regex=true`) so ONNX Runtime GenAI's C++ `std::regex` can load them.
- Chat template uses `is not defined` instead of Jinja `is undefined` for ORT GenAI minja.

## Export

Produced with Olive ModelBuilder (`int4_algo_config: k_quant`, `int4_block_size: 128`) from `Qwen/Qwen3.8-27B`. Vision and embedding were exported unquantized FP32.
