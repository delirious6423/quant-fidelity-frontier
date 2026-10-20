---
library_name: mlx
tags:
- mlx
- quantized
- mixed-precision
- gemma4
- moe
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
---



# Gemma-4-26B-A4B-it — 18GB (MLX)



Mixed-precision quantized version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) optimised by [baa.ai](https://baa.ai) using a proprietary Black Sheep AI method.



## Metrics



| Metric | Value |

|--------|-------|

| **Size** | **16.5 GB** |

| MMLU | 80.02% |

| MMLU vs BF16 | **99.0%** of BF16 |

| MMLU vs Uniform 4-bit | **+2.14pp** |



### All RAM Variants (all beat Uniform 4-bit)



| Model | Size | MMLU | MMLU % of BF16 |

|-------|------|------|----------------|

| BF16 | 51.6 GB | 80.80% | 100% |

| Uniform 4-bit | ~14 GB | 77.88% | 96.4% |

| **RAM 14GB** | ~14 GB | 78.27% | 96.9% |

| **RAM 18GB** | ~18 GB | 80.02% | 99.0% |

| **RAM 20GB ★** | **~20 GB** | **80.70%** | **99.9%** |

| **RAM 22GB** | ~22 GB | 80.41% | 99.5% |



## Usage



```python

from mlx_lm import load, generate



model, tokenizer = load("baa-ai/Gemma-4-26B-A4B-it-RAM-18GB-MLX")

response = generate(model, tokenizer, prompt="Hello!", max_tokens=256)

print(response)

```



---

*Quantized by [baa.ai](https://baa.ai)*



---



## Black Sheep AI Products

Black Sheep AI builds private AI with the **Bounded AI Architecture (BAA)**: three layers, each bound to something you control, on infrastructure you control. Nothing leaves your boundary.

**[Ram](https://baa.ai/compression.html)** (Reasoning Layer) — Full-capability frontier models, shrunk to fit smaller GPUs with reasoning kept and certified. They rely strictly on your Data Modules for knowledge, so you swap the model as better ones ship.

**[Paddock](https://baa.ai/paddock.html)** (Knowledge Plane) — Binds your Data Modules directly to the reasoning layer: one per topic, period, person, permission or tenant. Permission, freeze, hand back or delete each one on its own.

**[Watchman](https://baa.ai/watchman.html)** (GRC Layer) — Governance, risk and compliance in a single, complete audit log: every module permission, every bind, every model's provenance. Five certificates before anything goes live.

**[Shepherd](https://baa.ai/shepherd.html)** — Takes a frontier model, compresses it into a Ram reasoning model, certifies it with Watchman, binds it to your Paddock, and deploys the Bounded AI Architecture to your infrastructure: your cloud, a Mac Studio, or an air-gapped facility.

Learn more at **[baa.ai](https://baa.ai)** — Private AI for enterprises whose data cannot leave the building.

