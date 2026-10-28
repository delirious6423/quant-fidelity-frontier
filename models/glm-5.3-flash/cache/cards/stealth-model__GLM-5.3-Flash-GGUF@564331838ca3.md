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

<table>
  <tr>
    <td width="65%" valign="middle">
      <h1>GLM-5.3-Flash GGUF</h1>
      <p><em>Ox Alpha, ready to run.</em></p>
      <p>Ox Alpha was GLM-5.3-Flash. Here it is for llama.cpp in two sizes: near full quality, and small enough for a 128 GB machine.</p>
      <p><a href="https://huggingface.co/stealth-model/ox-alpha"><strong>Meet Ox Alpha ↗</strong></a></p>
    </td>
    <td width="35%" align="center" valign="middle">
      <a href="https://stealthmodels.com/ox-alpha/">
        <img src="https://stealthmodels.com/ox-alpha/assets/ox-concept-v1.png" width="260" alt="A friendly jade ox with ivory and gold horns, a gold alpha forelock and one lifted hoof.">
      </a>
    </td>
  </tr>
</table>

> From [Stealth Models](https://stealthmodels.com/). We tracked this model as Ox Alpha before Z.ai revealed it.

## Files
| File | Size | Pick it for |
|---|---|---|
| `Q5_K_XL/` | 240 GB in 6 parts | Near full quality, on machines with 256 GB or more of combined memory |
| `IQ2_XXS/` | 102 GB in 4 parts | The smallest size we recommend: 128 GB machines such as a 128 GB Mac, DGX Spark or Strix Halo, or a 32 GB GPU with 96 GB of RAM |
| `mmproj-F16.gguf` | 1.1 GB | Image input |

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

## StealthMark results
| Benchmark score | General Intelligence | Visual Intelligence |
|---|---|---|
| 48.5 | 42.3 | 59.4 |

GLM-5.3-Flash is strongest in creativity, perspective and visual beauty: 77.7 out of 100 on [Beach](https://stealthmodels.com/benchmarks/models/glm-5.3-flash/beach).
StealthMark 1.8.8, results read 28 October 2026. [Full results](https://stealthmodels.com/benchmarks/models/glm-5.3-flash) · [On the map](https://stealthmodels.com/benchmarks/map)

## License
MIT, from Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). The license file is included.

---
[Ox Alpha card](https://huggingface.co/stealth-model/ox-alpha) · [StealthMark](https://stealthmodels.com/stealthmark) · [Play the benchmarks](https://stealthmodels.com/play) · [Follow for new reveals](https://huggingface.co/stealth-model)
