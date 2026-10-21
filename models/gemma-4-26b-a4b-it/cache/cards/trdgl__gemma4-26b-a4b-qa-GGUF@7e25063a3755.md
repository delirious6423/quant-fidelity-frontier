---
library_name: llama.cpp
base_model: google/gemma-4-26B-A4B-it
language:
- en
- vi
license: gemma
tags:
- pytorch
- testing
- code-generation
- gguf
- gemma
- quantized
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B fine-tuned for PyTorch API testing — GGUF

Fine-tuned `google/gemma-4-26B-A4B-it` for generating small Python programs that exercise PyTorch APIs on tiny tensors. Quantized to GGUF for inference with llama.cpp / llama-cpp-python.

## Files

| File | Quant | Size (approx.) | Suitable hardware |
|---|---|---|---|
| `gemma4-26b-a4b-qa.Q3_K_M.gguf` | Q3_K_M | ~13 GB | Colab T4 16GB (full GPU) |
| `gemma4-26b-a4b-qa.Q4_K_M.gguf` | Q4_K_M | ~17 GB | A100 40GB / T4 with offload |

## Training summary

- **Base model**: `google/gemma-4-26B-A4B-it` (MoE, ~4B active params)
- **Method**: QLoRA 4-bit (nf4) + TRL `SFTTrainer` with `assistant_only_loss=True`
- **LoRA**: r=16, alpha=32, target=all-linear, dropout 0
- **Dataset**: 1443 synthetic instruction/output pairs derived from PyTorch API metadata (DFUZZ-style)
- **Hyperparameters**: 2 epochs, batch 1 x grad-accum 8, lr 1e-4, cosine schedule, warmup 20 steps
- **Hardware / time**: NVIDIA H200, ~28 minutes
- **Final eval_loss**: 0.150  (mean_token_accuracy ~95%)

## Prompt format (Gemma chat template)

```
<bos><start_of_turn>system
You are a PyTorch API testing assistant. Given a PyTorch API name, signature, and docstring, generate a small, deterministic Python program that exercises this API on tiny tensors. Output only Python code.<end_of_turn>
<start_of_turn>user
Muc tieu API: torch.matmul
Thong tin: {'input': 'Tensor', 'other': 'Tensor'}
Mo ta chi tiet: Matrix product of two tensors.
Hay viet ma Python kiem thu API nay.<end_of_turn>
<start_of_turn>model
```

Output is a single short Python snippet that imports torch, builds tiny tensors, calls the target API, and prints results.

## Usage with `llama-cpp-python`

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="trdgl/gemma4-26b-a4b-qa-GGUF",
    filename="gemma4-26b-a4b-qa.Q3_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False,
)

SYSTEM = (
    "You are a PyTorch API testing assistant. "
    "Given a PyTorch API name, signature, and docstring, "
    "generate a small, deterministic Python program that exercises "
    "this API on tiny tensors. Output only Python code."
)

out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Muc tieu API: torch.matmul\n"
            "Thong tin: {'input': 'Tensor', 'other': 'Tensor'}\n"
            "Mo ta chi tiet: Matrix product of two tensors.\n"
            "Hay viet ma Python kiem thu API nay."},
    ],
    max_tokens=300,
    temperature=0.0,
    stop=["<end_of_turn>", "<start_of_turn>"],
)
print(out["choices"][0]["message"]["content"])
```

## Speed reference

| Hardware | Quant | Speed |
|---|---|---|
| NVIDIA H200 | Q3_K_M / Q4_K_M | ~140-160 tok/s |
| Colab T4 16GB | Q3_K_M (full GPU) | ~15-25 tok/s |
| Colab T4 16GB | Q4_K_M (with offload) | ~8-15 tok/s |

## Notes

- Stop tokens recommended: `<end_of_turn>`, `<start_of_turn>`.
- LoRA adapter (~250 MB) is included under `adapter/` for further stacking or full-precision merging.
- This GGUF chat template was patched to remove training-specific markers; standard llama.cpp loaders work without overrides.

## License

Model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The LoRA adapter and conversion code are released under Apache-2.0.
