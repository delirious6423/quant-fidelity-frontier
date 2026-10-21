---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
  - "Qwen/Qwen3.8-27B"
pipeline_tag: "image-text-to-text"
tags:
  - firefly
  - rockchip
  - rk1828
  - llamapi
  - edge-ai
---

<div align="center">
  <img src="./firefly-logo-outline.png" width="150" alt="Firefly">

  <h1>Qwen3.8-27B</h1>

  <p>
    <code>RK1828</code>
    &nbsp; | &nbsp;
    <code>LlamaPi Ready</code>
  </p>
</div>

### Model Overview

| Item | Details |
| :---: | :---: |
| Original model | Qwen3.8-27B ([ModelScope](https://modelscope.cn/models/Qwen/Qwen3.8-27B) \| [Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)) |
| Original developer | Qwen Team |
| Model type | Vision-language generation |
| Target platform | RK1828 |
| Converted by | [Firefly AI Team](https://www.t-firefly.com/) |
| Deployment tool | [LlamaPi](https://community.t-firefly.com/en/docs/ai/applications/LlamaPi/llamapi/introduction) |

### Run with LlamaPi

```bash
llamapi run qwen3.8:27b
```

`llamapi run` handles model download, loading, and execution. See the [LlamaPi Wiki](https://community.t-firefly.com/en/docs/ai/applications/LlamaPi/llamapi/introduction) for setup and usage.

### About the Original Model

Qwen3.8-27B is a dense, native vision-language model from the Qwen3.8 family, the most capable generation in the Qwen open-model family to date.

- Delivers comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- Natively understands images and videos, from STEM diagrams and documents to hour-scale videos.
- Offers flexible thinking control, with thinking mode on by default and tunable reasoning depth per request.

### License & Attribution

The models provided in this repository are converted and adapted by the Firefly AI team from the original models. All intellectual property rights and copyright in the underlying original models remain with the original official team, and their distribution is governed by the corresponding official open-source license.
