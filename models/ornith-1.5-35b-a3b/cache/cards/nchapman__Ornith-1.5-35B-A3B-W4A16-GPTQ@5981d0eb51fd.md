---
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: mit
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- w4a16
- compressed-tensors
- vllm
- moe
---

# Ornith-1.5-35B-A3B-W4A16-GPTQ

INT4 (W4A16) GPTQ quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MoE) — the same
model at roughly a quarter of the memory, built for vLLM.

- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- **Calibration data:** 999 samples at the dataset's own
  proportions from [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data); the
  exact rows are pinned in
  [warp-3090-quant-lab-frozen-data](https://huggingface.co/datasets/nchapman/warp-3090-quant-lab-frozen-data)
- **Chat template:** [froggeric Qwen-Fixed-Chat-Templates (v22.5)](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  ships as `chat_template.jinja` — the artifact's only template
- **Default sampling** (`generation_config.json`): temperature 1.0, top_p 0.95, top_k 20 — the thinking-mode
  defaults of the base model's card; send your own values to override
- **Recipe and tooling:** [warp-3090-model-lab](https://github.com/nchapman/warp-3090-model-lab)

Vision tower, MTP head, linear-attention projections, and `lm_head` stay BF16.

```python
from vllm import LLM
llm = LLM(model="nchapman/Ornith-1.5-35B-A3B-W4A16-GPTQ")
```
