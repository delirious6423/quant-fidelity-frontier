---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- nightmedia
- gemma4
- gemma
- google
- Deckard(qx)
- mlx
---

# gemma-4-26B-A4B-it-qx86-hi-mlx

This is a Deckard(qx) experimental quant.

Brainwaves
```brainwaves
         arc   arc/e boolq hswag obkqa piqa  wino
mxfp8    0.454,0.598,0.871,0.582,0.394,0.723,0.645
mxfp4    0.462,0.596,0.855,0.578,0.378,0.723,0.637
qx86-hi  0.472,0.605,0.873,0.565,0.386,0.712,0.644
qx64-hi  0.472,0.621,0.866,0.564,0.382,0.717,0.637

Perplexity               Peak Memory   Tokens/sec
mxfp8   103.904 ± 1.765   33.28 GB      880
mxfp4   123.621 ± 2.121   20.66 GB     1266
qx86-hi  75.542 ± 1.247   29.23 GB     1145
qx64-hi  98.161 ± 1.645   22.92 GB     1135
```

Based on this model
```brainwaves
         arc   arc/e boolq hswag obkqa piqa  wino

TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill
qx86-hi  0.433,0.522,0.468,0.506,0.370,0.687,0.612
Instruct
qx86-hi  0.564,0.763,0.861,0.657,0.450,0.771,0.680
```

See parent model for instructions on install and use with Transformers.
- [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

As I don't have an easy way to test this until LMStudio supports it, please Like it *only* if you had a good experience.

Thank you,

-G
