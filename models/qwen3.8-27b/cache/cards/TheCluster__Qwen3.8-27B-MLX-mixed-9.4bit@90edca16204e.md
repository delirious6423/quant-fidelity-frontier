---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
language:
- en
- zh
- ru
- es
- fr
- it
- ja
- ko
- af
- de
- ar
- tr
- is
- pl
- sw
- sv
- nl
- he
- id
- uk
- fa
- pa
- pt
- ms
- fi
- el
base_model:
- Qwen/Qwen3.8-27B
library_name: mlx
tags:
- qwen3.8
- mixed-precision
- 9bit
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B

**Quality**: quantized (***mixed quants per tensor**, group size: 32, **9.450 bpw***)

Most tensors use 8-bit affine quantization with a group size 32; some important tensors are saved in bf16.

Update: default **reasoning_effort** is set to '**low**' to avoid overthinking.

### Recommended settings

1. **Sampling Parameters**: The developers suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

-----
### Source
This model was converted to MLX format from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-vlm version **0.6.13**.