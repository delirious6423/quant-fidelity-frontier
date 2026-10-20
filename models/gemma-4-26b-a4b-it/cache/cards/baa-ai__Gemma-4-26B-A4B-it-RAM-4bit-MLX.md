---
library_name: mlx
tags:
- mlx
- quantized
- gemma4
- multimodal
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
---



# Gemma-4-26B-A4B-it — RAM 4bit (MLX)



A quantized build of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) produced by [baa.ai](https://baa.ai). **Retains the full vision tower**, unlike other pre-quantized MLX variants of this model.



| Property | Value |

|---|---|

| **Size on disk** | **14.63 GB** |

| Format | MLX |

| Base model | google/gemma-4-26B-A4B-it |

| Vision tower | Retained |



## Usage



```python

from mlx_vlm import load, generate



model, processor = load("baa-ai/Gemma-4-26B-A4B-it-RAM-4bit-MLX")

prompt = processor.tokenizer.apply_chat_template(

    [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],

    add_generation_prompt=True, tokenize=False,

)

result = generate(model, processor, prompt, max_tokens=512, verbose=True)

print(result.text)

```



## Benchmark results



Measured on vanilla MMLU (80 questions) and MathVision MCQ (20 questions). Every RAM variant is slightly smaller on disk than the corresponding unsloth MLX variant.



*Unsloth Gemma 4 MLX variants strip the vision tower — they cannot process images.*



| Family | Bits | Unsloth Size | RAM Size | Unsloth MMLU | RAM MMLU | GAP | RAM Vision |

|---|---|---|---|---|---|---|---|

| 26B-A4B | 8bit | 26.81 GB | 26.52 GB | 56.2% | 90.0% | +33.8 | 35.0% |

| 26B-A4B | 4bit | 14.96 GB | 14.63 GB | 55.0% | 88.8% | +33.8 | 50.0% |

| 31B | 8bit | 32.62 GB | 32.59 GB | 70.7% | 90.0% | +19.3 | 55.0% |

| 31B | 4bit | 22.12 GB | 22.09 GB | 73.8% | 86.2% | +12.4 | 60.0% |

| 31B | 3bit | 19.52 GB | 19.49 GB | 72.5% | 86.2% | +13.7 | 50.0% |





### MMLU 500-question run



Larger sample (n=500) on the same models. Vision questions excluded since unsloth's MLX variants don't support image input.



| Family | Bits | Unsloth Size | RAM Size | Unsloth MMLU | RAM MMLU | GAP |

|---|---|---|---|---|---|---|

| 26B-A4B | 8bit | 26.81 GB | 26.52 GB | 57.8% | 89.2% | +31.4 |

| 26B-A4B | 4bit | 14.96 GB | 14.63 GB | 60.4% | 86.8% | +26.4 |

| 31B | 8bit | 32.62 GB | 32.59 GB | 71.8% | 90.4% | +18.6 |

| 31B | 4bit | 22.12 GB | 22.09 GB | 71.8% | 89.0% | +17.2 |

| 31B | 3bit | 19.52 GB | 19.49 GB | 75.6% | 89.2% | +13.6 |

## License



Inherited from the upstream [Gemma license](https://huggingface.co/google/gemma-4-26B-A4B-it).



---

*[baa.ai](https://baa.ai)*



---



## Black Sheep AI Products

Black Sheep AI builds private AI with the **Bounded AI Architecture (BAA)**: three layers, each bound to something you control, on infrastructure you control. Nothing leaves your boundary.

**[Ram](https://baa.ai/compression.html)** (Reasoning Layer) — Full-capability frontier models, shrunk to fit smaller GPUs with reasoning kept and certified. They rely strictly on your Data Modules for knowledge, so you swap the model as better ones ship.

**[Paddock](https://baa.ai/paddock.html)** (Knowledge Plane) — Binds your Data Modules directly to the reasoning layer: one per topic, period, person, permission or tenant. Permission, freeze, hand back or delete each one on its own.

**[Watchman](https://baa.ai/watchman.html)** (GRC Layer) — Governance, risk and compliance in a single, complete audit log: every module permission, every bind, every model's provenance. Five certificates before anything goes live.

**[Shepherd](https://baa.ai/shepherd.html)** — Takes a frontier model, compresses it into a Ram reasoning model, certifies it with Watchman, binds it to your Paddock, and deploys the Bounded AI Architecture to your infrastructure: your cloud, a Mac Studio, or an air-gapped facility.

Learn more at **[baa.ai](https://baa.ai)** — Private AI for enterprises whose data cannot leave the building.

