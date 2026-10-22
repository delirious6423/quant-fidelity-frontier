---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [gguf, experimental, glm]
---
# GLM-5.3-Flash Phobos low-bit (experimental, archived)

Two experimental low-bit requants of GLM-5.3-Flash made while building
[GLM-5.3-Flash-Phobos-GGUF](https://huggingface.co/mussonking/GLM-5.3-Flash-Phobos-GGUF) (the IQ4 release).
They were measured and NOT retained; archived here as-is. Same tensor layout as Phobos IQ4, experts requantized.

| Folder | Experts | Same top token vs BF16 (agentic corpus) |
|---|---|---|
| IQ3_XXS | IQ3_XXS | 89.8 % |
| IQ2_XXS-DOWN-IQ3_XXS | gate/up IQ2_XXS, down IQ3_XXS | 86.9 % |
| (reference: Phobos IQ4) | IQ4_XS / IQ4_NL | 92.8 % |

Full KL-divergence logs in `kl/`, recipes in each folder. Source: unsloth BF16 GGUF + unsloth imatrix.
Loadable by llama.cpp builds that support GLM-5.3-Flash. No support offered.
