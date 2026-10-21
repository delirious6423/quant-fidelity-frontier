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
- heretic
- uncensored
- unrestricted
- decensored
- abliterated
- 4bit
pipeline_tag: image-text-to-text
---
<div style="text-align:center; margin-bottom:12pt; font-size:11pt">If you like my work, you can <a href="https://donatr.ee/thecluster/">support me</a><br/></div>

# Qwen3.8-27B Heretic

**Quality**: quantized (4-bit, affine, group size: 64)

This is an **uncensored** version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made using [Heretic](https://github.com/p-e-w/heretic) v1.4.0.

Update: default **reasoning_effort** is set to '**low**' to avoid overthinking.

### Recommended settings

1. **Sampling Parameters**: The developers suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

