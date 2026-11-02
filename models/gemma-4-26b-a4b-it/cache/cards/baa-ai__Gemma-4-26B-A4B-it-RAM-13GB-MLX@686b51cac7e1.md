---
library_name: mlx
tags:
- mlx
- quantized
- mixed-precision
- gemma4
- moe
- text-generation
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
---



# Gemma-4-26B-A4B-it — 13GB (MLX)



Mixed-precision quantized version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) optimised by [baa.ai](https://baa.ai) using a proprietary Black Sheep AI method.



This build is the RAM **knee**: the smallest budget the RAM probe predicts before quality falls away (3.46 bits per weight on average). It is **text-only**: the language model is quantized and the vision tower is not included. Use the [14GB](https://huggingface.co/baa-ai/Gemma-4-26B-A4B-it-RAM-14GB-MLX) or larger builds for image input.



## Metrics



| Metric | Value |

|--------|-------|

| **Size** | **12.9 GB** |

| MMLU (400-question set, thinking on) | 83.75% (335/400) |

| MMLU (400-question set, thinking on), RAM 14GB on the same harness | 85.25% (341/400) |

| Document Q&A, Boeing 787 manuals (150 questions, answer contains the expected value) | 60.0% |

| Document retrieval, Dash 8 Q300 maintenance manual (415 questions, right document in the top 3) | 71.8% |



The two document tasks were measured with the model as the reader of a retrieval-augmented console, against the same golden sets as five other readers (up to 90 GB): it was the most accurate reader under 13 GB on the 787 set, and retrieval matched 30B-class readers.



### All RAM Variants



| Model | Size | MMLU | MMLU % of BF16 |

|-------|------|------|----------------|

| BF16 | 51.6 GB | 80.80% | 100% |

| Uniform 4-bit | ~14 GB | 77.88% | 96.4% |

| **RAM 13GB** (text-only) | ~13 GB | 83.75%† | — |

| **RAM 14GB** | ~14 GB | 78.27% | 96.9% |

| **RAM 18GB** | ~18 GB | 80.02% | 99.0% |

| **RAM 20GB ★** | **~20 GB** | **80.70%** | **99.9%** |

| **RAM 22GB** | ~22 GB | 80.41% | 99.5% |



† Measured on a 400-question MMLU set with thinking on: the model reasons for up to 2048 tokens and is then asked for the letter if it has not given one (80 of 400 questions needed that prompt). It is not directly comparable with the other rows, which come from the original evaluation; on the same harness the RAM 14GB build (with vision) scores 85.25%, so the text-only 13GB keeps 98.2% of it.



## Usage



```python

from mlx_lm import load, generate



model, tokenizer = load("baa-ai/Gemma-4-26B-A4B-it-RAM-13GB-MLX")

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

