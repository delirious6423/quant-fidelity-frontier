---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
pipeline_tag: text-generation
tags:
- liquid
- edge
- lfm2.5
- moe
- onnx
- onnxruntime
- onnxruntime-genai
base_model:
- LiquidAI/LFM2.5-24B-A2B
---

<div align="center">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png"
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> •
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-24B-A2B-ONNX

LFM2.5-24B-A2B is a Mixture of Experts reasoning model with 24B total parameters and 2B active parameters per token, built for **fast, private agents on-device**. It is the largest model in the LFM2.5 family.

Find more details in the original model card: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B

## Recommended Variants

| Precision | Size | Platform | Use Case |
|-----------|------|----------|----------|
| Q4 | ~15.3 GB | CPU | Recommended for most uses (default) |
| Q4F16 | ~13.7 GB | CUDA | Q4 experts with FP16 runtime |
| Q8 | ~27.1 GB | CPU | Balance of quality and size |
| FP16 | ~48.0 GB | CUDA | Higher quality |

- **CPU**: Use `Q4` or `Q8`.
- **CUDA**: Use `Q4F16` or `FP16`.

This model is too large for WebGPU browser inference.

## Model Files

```
onnx/
├── model.onnx              # FP32
├── model_fp16.onnx         # FP16
├── model_q4.onnx           # Q4 (default)
├── model_q4f16.onnx        # Q4 experts, FP16 runtime
└── model_q8.onnx           # Q8
```

Weights are split across `model*.onnx_data`, `model*.onnx_data_1`, ... files, which must sit next to their `.onnx` file.

## Python (onnxruntime-genai)

```bash
pip install onnxruntime-genai transformers jinja2 huggingface_hub
# or, for CUDA:
pip install onnxruntime-genai-cuda transformers jinja2 huggingface_hub
```

```python
import onnxruntime_genai as og
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_id = "LiquidAI/LFM2.5-24B-A2B-ONNX"
filename = "model_q4.onnx"  # use model_q4f16.onnx or model_fp16.onnx on CUDA
model_path = snapshot_download(model_id, allow_patterns=["*.json", "*.jinja", f"onnx/{filename}*"])

config = og.Config(model_path)
config.overlay(f'{{"model": {{"decoder": {{"filename": "onnx/{filename}"}}}}}}')
# For CUDA: config.clear_providers(); config.append_provider("cuda")
model = og.Model(config)
tokenizer = og.Tokenizer(model)
stream = tokenizer.create_stream()

messages = [{"role": "user", "content": "What is C. elegans?"}]
prompt = AutoTokenizer.from_pretrained(model_path).apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

params = og.GeneratorParams(model)
params.set_search_options(max_length=4096)
generator = og.Generator(model, params)
generator.append_tokens(tokenizer.encode(prompt))
while not generator.is_done():
    generator.generate_next_token()
    print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
print()
```

## License

This model is released under the [LFM 1.0 License](LICENSE).
