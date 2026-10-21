---
license: "apache-2.0"
license_link: "https://ai.google.dev/gemma/docs/gemma_4_license"
base_model:
  - "google/gemma-4-26B-A4B-it"
pipeline_tag: "image-text-to-text"
tags:
  - firefly
  - "houmo"
  - "m50"
  - llamapi
  - edge-ai
---

<div align="center">
  <img src="./firefly-logo-outline.png" width="150" alt="Firefly">

  <h1>Gemma 4 26B A4B IT</h1>

  <p>
    <code>M50</code>
    &nbsp; | &nbsp;
    <code>LlamaPi Ready</code>
  </p>
</div>

### Model Overview

| Item | Details |
| :---: | :---: |
| Original model | Gemma 4 26B A4B IT ([ModelScope](https://modelscope.cn/models/google/gemma-4-26B-A4B-it) \| [Hugging Face](https://huggingface.co/google/gemma-4-26B-A4B-it)) |
| Original developer | Google DeepMind |
| Model type | Multimodal generation |
| Target platform | M50 |
| Converted by | [Firefly AI Team](https://www.t-firefly.com/) |
| Deployment tool | [LlamaPi](https://community.t-firefly.com/en/docs/ai/applications/LlamaPi/llamapi/introduction) |

### Run with LlamaPi

```bash
llamapi run gemma4:26b-a4b
```

`llamapi run` handles model download, loading, and execution. See the [LlamaPi Wiki](https://community.t-firefly.com/en/docs/ai/applications/LlamaPi/llamapi/introduction) for setup and usage.

### About the Original Model

Gemma 4 26B A4B IT is an instruction-tuned multimodal model developed by Google DeepMind.

- Accepts text and image inputs and generates text output, with variable aspect ratio and resolution support.
- Supports reasoning, coding, multilingual, and agentic tasks with a 256K context window and 140+ languages.
- Employs a Mixture-of-Experts architecture (25.2B total, 3.8B active parameters) for fast inference on consumer GPUs and workstations.

### License & Attribution

The models provided in this repository are converted and adapted by the Firefly AI team from the original models. All intellectual property rights and copyright in the underlying original models remain with the original official team, and their distribution is governed by the corresponding official open-source license.
