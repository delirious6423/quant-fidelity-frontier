---
license: mit
library_name: gguf
pipeline_tag: text-generation
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - gguf
  - quantized
  - peelsa
  - ablit
---

# peelsa/Ornith-1.5-35B-A3B-ablit-GGUF

GGUF quantization of Peelsa's refusal-ablated weights of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a mixture-of-experts model with 256 routed experts. It is text only (the vision tower is not in this file), runs in llama.cpp (architecture `qwen35moe`), and supports a 262,144-token text context. The chat template is embedded in the file. The model's extra next-token prediction block (block 40) is kept in the file, but current llama.cpp builds do not run it, so speculative decoding from it is not supported here.

> **Peelsa self-certification:** this file is a straight quantization of Peelsa's ablit weights, which were made only by projecting refusal directions out of the base model's weights, with no training, fine-tuning, adapters, or added code; it holds only tensors, metadata, and the base model's unmodified chat template.

| File | Quant | Size | Context |
|---|---|---|---|
| `Ornith-1.5-35B-A3B-ablit-Q4_K_M.gguf` | Q4_K_M (4.89 bits per weight) | 21.7 GB | 262,144 |

**Method:** GGUF Q4_K_M made with an importance matrix computed on WikiText-2.

One routed expert (index 93 in layer 14) received no calibration activity, so it is quantized at the same width with equal channel importance, and its weights are fully present.

**SHA-256:** `418d6781d0ed4e84a1c39ffc741dc527ecda4d1d8ce407adde9f99848fefd1c6`

## Run

```bash
llama-server -m Ornith-1.5-35B-A3B-ablit-Q4_K_M.gguf -ngl 99
```

## License

MIT, the same as the upstream base model.
