---
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/67c10cfba43d7939d60160ff/dm_CCplMOlt6KOJ6gIToi.png
language:
- en
license: other
inference: false
tags:
- nsfw
- explicit
- roleplay
- mixed-AI
- furry
- anthro
- chat
- manipulation
- sfw
pipeline_tag: text-generation
library_name: transformers
base_model:
- Qwen/Qwen3.8-27B
---

GGUF quants of [Mawdistical-S1/Synica-27B](https://huggingface.co/Mawdistical-S1/Synica-27B)
using imatrix from [Mawdistical-S1/Synica-27B-GGUF](https://huggingface.co/Mawdistical-S1/Synica-27B-GGUF)

Synica-27B-iQ6_K-XL.gguf (6.99 BPW):
```
  --tensor-type "token_embd\.weight=q8_0" \
  --tensor-type "output\.weight=q8_0" \
  --tensor-type ".*attn_v\.weight=q8_0" \
  --tensor-type ".*attn_output\.weight=q8_0" \
  --tensor-type ".*ssm_(in|out)\.weight=q8_0" \
  --tensor-type ".*ssm_(alpha|beta)\.weight=bf16" \
  --tensor-type "blk\.(0|63)\..*\.weight=q8_0" \
  --tensor-type "blk\.64\.nextn\.eh_proj\.weight=bf16" \
  --tensor-type "blk\.64\.(attn|ffn)_.*\.weight=q8_0" \
```

Synica-27B-iQ4_K_M.gguf (5.21 BPW):
```
  --tensor-type "token_embd\.weight=q6_k" \
  --tensor-type "output\.weight=q6_k" \
  --tensor-type ".*attn_v\.weight=q6_k" \
  --tensor-type ".*ssm_(alpha|beta)\.weight=bf16" \
  --tensor-type ".*ssm_out\.weight=q6_k" \
```