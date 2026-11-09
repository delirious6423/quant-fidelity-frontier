---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - entrit
  - d2
  - ternary
  - quantized
---

# Qwen3.8-27B d2

Qwen3.8-27B with its attention, DeltaNet and MLP projections quantized by the Entrit **d2** codec.

## What changed from the base model

The weights of the attention, DeltaNet and MLP projections are replaced by d2 values. Each weight is
one of 9 levels, `{0, ±1, ±2.83, ±5.20, ±8}`, times one scale per group of 64 weights. That's about
3.4 bits of information per weight. The embedding, output head, norms and the small DeltaNet gate
projections are unchanged from the base model.

The files store the d2 values **dequantized to bf16**, so they load in stock `transformers` and are
the same size as the original (55 GB).

The base model's MTP (multi-token prediction) head is not included.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Entrit/Qwen3.8-27B-d2dense")
model = AutoModelForCausalLM.from_pretrained("Entrit/Qwen3.8-27B-d2dense", torch_dtype="auto", device_map="auto")
```

To run it on one 24 GB graphics card, use the 15.3 GB GGUF build,
[Entrit/Qwen3.8-27B-d2-GGUF](https://huggingface.co/Entrit/Qwen3.8-27B-d2-GGUF), with Entrit's llama.cpp
kernel, [Entrit/llama.cpp-d2nl](https://huggingface.co/Entrit/llama.cpp-d2nl).

## Results

Not yet benchmarked against the FP16 base model.

## License and notice

Apache License 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); see
`LICENSE`. This is a modified version of that model: the weights were quantized as described above.

The d2 codec is the subject of pending patent applications by Entrit Systems.
Contact: eric@entrit.io
