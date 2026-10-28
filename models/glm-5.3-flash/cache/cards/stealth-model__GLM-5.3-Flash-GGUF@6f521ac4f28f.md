---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
license: mit
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - glm
  - glm-5.3-flash
  - ox-alpha
  - stealth-model
  - stealthmark
thumbnail: https://stealthmodels.com/ox-alpha/assets/ox-concept-v1.png
---

<a href="https://stealthmodels.com/ox-alpha/?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card"><img src="https://huggingface.co/stealth-model/GLM-5.3-Flash-GGUF/resolve/main/panels/hero.webp" alt="A friendly jade ox with ivory and gold horns and a gold alpha forelock. Ox Alpha. Revealed: GLM-5.3-Flash by Z.ai. The stealth model that became open weights."></a>

**GLM-5.3-Flash was Ox Alpha. Two GGUF files, ready for llama.cpp.**

*From [Stealth Models](https://stealthmodels.com/?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card): we tracked this model as Ox Alpha before Z.ai revealed it.*

<a href="https://huggingface.co/stealth-model/GLM-5.3-Flash-GGUF/tree/main"><img src="https://huggingface.co/stealth-model/GLM-5.3-Flash-GGUF/resolve/main/panels/downloads.webp" alt="Downloads. Q5_K_XL, 240 GB: near full quality, for machines with 256 GB or more. IQ2_XXS, 102 GB: the smallest we recommend, fits 128 GB machines. Image input: mmproj-F16.gguf, 1.1 GB."></a>

StealthMark results for GLM-5.3-Flash (Ox Alpha):

<a href="https://stealthmodels.com/benchmarks/models/glm-5.3-flash?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card"><img src="https://huggingface.co/stealth-model/GLM-5.3-Flash-GGUF/resolve/main/panels/verdict.webp" alt="StealthMark 1.8.8 results for Ox Alpha (GLM-5.3-Flash): benchmark score 48.5 out of 100, General Intelligence 42.3, Visual Intelligence 59.4. Open weights, and strongest at composing a whole scene: 77.7 on Beach."></a>

<a href="https://stealthmodels.com/benchmarks/models/glm-5.3-flash?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card"><img src="https://huggingface.co/stealth-model/GLM-5.3-Flash-GGUF/resolve/main/panels/results.webp" alt="Every StealthMark result of Ox Alpha (GLM-5.3-Flash): Beach 77.7, Selfie 76.6, Ground Truth 63.8, Raccoon 55.1, Mona Lisa 49.5, Foresight 47.0, Newton's Mirror 27.6, Latch 4.0 (partial run, 3 of 5 levels)."></a>

**Open each result** · Visual Intelligence: [Beach&nbsp;77.7](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/beach?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Selfie&nbsp;76.6](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/selfie?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Raccoon&nbsp;55.1](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/raccoon?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Mona&nbsp;Lisa&nbsp;49.5](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/mona-lisa?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card)<br>
General Intelligence: [Ground&nbsp;Truth&nbsp;63.8](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/ground-truth?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Foresight&nbsp;47.0](https://stealthmodels.com/benchmarks/models/glm-5.3-flash?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card#game-foresight-self-improvement) · [Newton's&nbsp;Mirror&nbsp;27.6](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/newtons-mirror?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Latch&nbsp;4.0&nbsp;(3&nbsp;of&nbsp;5&nbsp;levels)](https://stealthmodels.com/benchmarks/models/glm-5.3-flash?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card#game-latch-five-levels)

**[All results and comparisons on stealthmodels.com ↗](https://stealthmodels.com/benchmarks/models/glm-5.3-flash?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card)**

<details><summary>Files as text</summary>

| File | Size | Pick it for |
|---|---|---|
| `Q5_K_XL/` | 240 GB in 6 parts | Near full quality, on machines with 256 GB or more of combined memory |
| `IQ2_XXS/` | 102 GB in 4 parts | The smallest size we recommend: 128 GB machines such as a 128 GB Mac, DGX Spark or Strix Halo, or a 32 GB GPU with 96 GB of RAM |
| `mmproj-F16.gguf` | 1.1 GB | Image input |

</details>

## Run it
Download one size:
```
hf download stealth-model/GLM-5.3-Flash-GGUF --include "IQ2_XXS/*" "mmproj-F16.gguf" --local-dir GLM-5.3-Flash
```
Point llama.cpp at the first part; it loads the rest:
```
llama-server -m GLM-5.3-Flash/IQ2_XXS/GLM-5.3-Flash-IQ2_XXS-00001-of-00004.gguf --mmproj GLM-5.3-Flash/mmproj-F16.gguf --jinja
```
If the model is larger than your GPU, keep its expert layers in system RAM with `--n-cpu-moe`.

## License
MIT, from Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). The license file is included.

---
[Ox Alpha card](https://huggingface.co/stealth-model/ox-alpha) · [StealthMark](https://stealthmodels.com/stealthmark?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Play the benchmarks](https://stealthmodels.com/play?utm_source=huggingface&utm_medium=referral&utm_campaign=glm-5.3-flash-gguf-card) · [Follow for new reveals](https://huggingface.co/stealth-model)
