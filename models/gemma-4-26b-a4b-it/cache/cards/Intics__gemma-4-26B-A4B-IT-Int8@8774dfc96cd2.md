---
license: apache-2.0
datasets:
- mit-han-lab/pile-val-backup
base_model:
- google/gemma-4-26B-A4B-it
---
# Intics/gemma-4-26B-A4B-IT-Int8

INT8 quantized version of `google/gemma-4-26B-A4B-it` optimized for efficient inference and serving using vLLM.

## Model Details

* Base Model: `google/gemma-4-26B-A4B-it`
* Quantization: INT8 (W8A8)
* Architecture: Mixture-of-Experts (MoE)
* Modalities: Text + Image
* Context Length: 256K
* Active Parameters: ~4B
* Total Parameters: ~26B

This model is intended for:

* Efficient inference
* vLLM serving
* Multi-GPU deployments
* Lower VRAM usage compared to BF16

---

## Quantization

This model was quantized using:

* `llm-compressor`
* `compressed-tensors`

Quantization format:

* Weights: INT8
* Activations: INT8

The vision encoder and embedding layers were excluded from quantization for better stability and multimodal quality.

---

## Hardware Requirements

Recommended:

* 2× RTX 3090
* A100 40GB+
* H100

Approximate VRAM:

* BF16: ~55GB
* INT8: ~30GB

---

## vLLM Serving

```bash
docker run --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v $(pwd):/model \
  vllm/vllm-openai:latest \
  --model /model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --served-model-name gemma4-int8
```

---

## Transformers Usage

```python
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "Intics/gemma-4-26B-A4B-IT-Int8"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {
        "role": "user",
        "content": "Explain Mixture-of-Experts models."
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=text,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256
)

print(processor.decode(outputs[0]))
```

---

## Notes

* Optimized primarily for inference workloads.
* INT8 quantization significantly reduces VRAM usage while preserving most model quality.
* Best served using vLLM.

---

## License

This model follows the same license as the original Gemma 4 release.

Please review:
https://ai.google.dev/gemma/docs/gemma_4_license

---

## Credits

* Google DeepMind
* vLLM
* llm-compressor
* compressed-tensors