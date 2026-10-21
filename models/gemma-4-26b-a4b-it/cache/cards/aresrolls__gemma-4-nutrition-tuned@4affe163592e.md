---
license: apache-2.0
language:
- ru
- en
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- moe
- pruning
- json-parsing
---


# Gemma 4 26B MoE - Pruned JSON Edition ✂️

This model is a surgically optimized, task-specific variant of the `google/gemma-4-26b-moe-it`. It has been specifically pruned for **JSON generation, structured data parsing, and dietary planning** tasks. 

By applying a custom Activation Profiling methodology (inspired by REAP), we successfully removed 25% of the model's fine-grained experts (from 128 down to 96) while preserving ~98% of the model's active knowledge capacity for the target domain.

## 🛠️ Pruning Methodology

Mixture-of-Experts (MoE) models often exhibit a "long-tail" activation pattern, where a small subset of experts handles the majority of specific reasoning tasks, while many experts remain dormant. 

1. **Profiling (Forward Pass Hooks):** We ran a custom multi-agent dataset (`waterfall_dataset_2.jsonl`, ~2000-3000 tokens per prompt) through the base model in `FP16`. Using PyTorch `register_forward_hook`, we tracked the routing decisions across all 40 layers.
2. **Analysis:** The architecture utilizes a fine-grained MoE setup (128 experts per layer). Our profiling revealed a steep exponential decay in expert utility: the top expert processed ~1.3 million tokens (1.93%), while the bottom expert processed only ~117k tokens (0.17%).
3. **Surgery:** We isolated the fused tensor matrices (`down_proj` and `gate_up_proj`) and permanently sliced off the bottom 32 experts (25%) from every single layer. 
4. **Result:** The model was condensed to 96 experts per layer, significantly reducing VRAM footprint and inference latency, with negligible zero-shot degradation on JSON formatting tasks.

## 📊 Expert Activation Distribution

During profiling, the activation frequency demonstrated that retaining the top 96 experts successfully captures the vast majority of the network's reasoning pathways for structured parsing. 

*(Add your generated `pruning_distribution.png` here)*

## 🚀 Usage & Formats

This repository contains the original unquantized **FP16 weights (~40GB)**. 

### Recommended Local Execution (GGUF)
For local inference on Apple Silicon (Mac) or consumer GPUs, we highly recommend using the **4-bit quantized GGUF version (Q4_K_M)** which requires only ~15GB of RAM/VRAM.

*(If you upload the GGUF file here, add the link or specify it is in the Files tab)*

```python
# Loading the FP16 model via Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "YOUR_ACCOUNT/gemma-4-26b-pruned-json"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## ⚠️ Limitations

Because this model was aggressively pruned using a highly specific dataset (JSON structuring and diet generation), its general-purpose conversational abilities, coding in other languages, or creative writing capabilities may be degraded compared to the base model. It is designed to be an efficient, narrow-expert agent.