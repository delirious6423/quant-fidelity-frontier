---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- quantized
- int8
- w8a8
- dynamic-quantization
- 8-bit
- llm-compressor
- zendnn
- compressed-tensors
- amd
- cpu-inference
language:
- en
---

# Qwen3.8-27B-w8a8-llmcompressor

## Model Overview

- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text
  - **Output:** Text
- **Source Model:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Supported Hardware:** AMD EPYC (CPU inference)
- **Preferred Operating System:** Linux
- **Inference Engine:** [vLLM](https://docs.vllm.ai/en/latest/) v0.28.0
- **Quantization Framework:** [LLM Compressor](https://github.com/vllm-project/llm-compressor) v0.12.0
- **Quantization Method:** 8-bit Weight, 8-bit Dynamic Activation Quantization (W8A8)
- **Compatible Stack:**
  - ZenDNN v6.1.0
  - ZenTorch v2.13.0
  - PyTorch v2.13.0
  - LLM Compressor v0.12.0
  - vLLM v0.28.0
- **Published with:** LLM Compressor v0.12.0

This is a quantized version of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) created by AMD using LLM Compressor (compressed-tensors) for ZenDNN-optimized CPU inference.

## Quantization

The model was quantized from [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using [LLM Compressor](https://github.com/vllm-project/llm-compressor) via the Round-to-Nearest (RTN) algorithm. This reduces the model weights from 51.8 GiB to 33.5 GiB on disk (~35% reduction).

- **Method:** 8-bit Weight, 8-bit Dynamic Activation Quantization (W8A8)
- **Config:** `compressed-tensors, num_bits=8, type=int, symmetric=true`
- **Weights:** INT8, symmetric, per-channel (static)
- **Activations:** INT8, symmetric, per-token (dynamic)

Qwen3.8-27B has a hybrid text tower: of its 64 layers, every fourth one is a full attention block and the remaining 48 are linear-attention (Mamba-style) blocks.

- **Quantized:** the dense `mlp.{gate,up,down}_proj` in all 64 layers, and `self_attn.{q,k,v,o}_proj` in the 16 full-attention layers.
- **Kept in BF16:** the entire `linear_attn` block in the 48 linear-attention layers, the vision encoder (`model.visual`, a 27-block ViT plus merger), `lm_head`, `embed_tokens`, and the layer norms.

The linear-attention blocks stay in BF16 because this is a data-free pass: their `conv1d`, gating, and state-update paths have activation distributions that per-channel RTN handles poorly without calibration statistics. Skipping them, along with the untouched vision encoder and the large `lm_head` and `embed_tokens` (248,320 x 5,120 each), is why the footprint drops ~35% rather than the ~50% of a fully quantized text-only model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "Qwen/Qwen3.8-27B"
output_dir = "./Qwen3.8-27B-w8a8-llmcompressor"

# Step 1: Load the BF16 model and tokenizer.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Step 2: Define the W8A8 recipe. Dense MLPs and full-attention projections are
# quantized; the linear-attention blocks, vision encoder and lm_head stay BF16.
# The "^mtp.*" entry guards against a multi-token-prediction head if the
# checkpoint ships one.
recipe = QuantizationModifier(
    scheme="W8A8",
    targets=["Linear"],
    ignore=[
        "lm_head",
        r"re:.*lm_head",
        r"re:.*model\.visual.*",
        r"re:.*linear_attn.*",
        r"re:^mtp.*",
    ],
)

# Step 3: One-shot quantize and save in compressed-tensors format.
# W8A8 here is data-free (RTN), so no calibration dataset is needed.
oneshot(
    model=model,
    recipe=recipe,
    tokenizer=tokenizer,
    output_dir=output_dir,
    trust_remote_code_model=True,
)

# oneshot does not save the processor; multimodal checkpoints need it for vLLM.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.save_pretrained(output_dir)

# Smoke test
inputs = tokenizer("What are we having for dinner?", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Quick Start

### Use with vLLM

```python
from vllm import LLM, SamplingParams

model = LLM(
    model="amd/Qwen3.8-27B-w8a8-llmcompressor",
    dtype="bfloat16",
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = model.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Requirements

```
torch==2.13.0
zentorch==2.13.0
vllm==0.28.0
llmcompressor==0.12.0
```

### OpenMP Setup

For optimal performance, set `LD_PRELOAD` with `libomp.so` (LLVM OpenMP) or `libiomp5.so` (Intel OpenMP):

```bash
# Using LLVM OpenMP (llvmopenmp)
export LD_PRELOAD=$(find /path/to/env -name "libomp.so" | head -1)

# Or using Intel OpenMP (libiomp)
export LD_PRELOAD=$(find /path/to/env -name "libiomp5.so" | head -1)
```

> **Note:** Set `LD_PRELOAD` before launching vLLM or any inference script.

## Evaluation

The model was evaluated against the BF16 (unquantized) baseline on standard benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the vLLM engine.

| Benchmark | BF16 Baseline | W8A8 (this model) | Recovery |
| --- | --- | --- | --- |
| GSM8K (5-shot) | 0.9727 | 0.9689 | 99.61% |

### Evaluation Command

```bash
lm_eval \
    --model vllm \
    --model_args pretrained=amd/Qwen3.8-27B-w8a8-llmcompressor,dtype=bfloat16,language_model_only=True \
    --tasks gsm8k \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot 5 \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs "max_gen_toks=2048" \
    --output_path .
```

## Limitations

- **Version Lock:** This model is compatible with ZenDNN v6.1.0 / ZenTorch v2.13.0 / PyTorch v2.13.0. It may not load correctly on other versions.
- **CPU Only:** This model is optimized for AMD EPYC CPU inference via ZenDNN. It is not intended for GPU inference.
- **Hybrid Layers Unquantized:** The 48 linear-attention blocks and the vision encoder remain in BF16, so both the memory saving and the INT8 speedup apply only to the dense MLPs and the 16 full-attention blocks. Evaluation was run with `language_model_only=True`.

## License

This model is distributed under the same license as the source model. See the [LICENSE](./LICENSE) file for details.

Modifications copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.