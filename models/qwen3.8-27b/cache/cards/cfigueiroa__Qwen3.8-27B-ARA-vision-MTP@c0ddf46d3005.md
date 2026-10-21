---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- trohrbaugh/Qwen3.8-27B-heretic-ara
- cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- heretic
- abliterated
- ara
- gguf
- mtp
- nextn
- vision
---

# Qwen3.8-27B ARA (1 passada) + visão + MTP

Cérebro de [`cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF`](https://huggingface.co/cygnal/Qwen3.8-27B-heretic-ara-Q4_K_M-MTP-GGUF): uma passada ARA do trohrbaugh **já com NextN dentro**, mais o projector de visão oficial do `ggml-org`.

Menos abliterado que o RVN (3 passadas). Útil para A/B contra o repo irmão `cfigueiroa/Qwen3.8-27B-RVN-vision-MTP`.

## Ficheiros

| ficheiro | o que é |
|---|---|
| `Qwen3.8-27B-heretic-ara-Q4_K_M-MTP.gguf` | texto 1× ARA Q4_K_M com NextN 15/15 (~16.8 GB) |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | visão oficial [`ggml-org`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) |

Não uses `-md`. O acelerador já está no principal.

## Correr

```bash
llama-server \
  -m Qwen3.8-27B-heretic-ara-Q4_K_M-MTP.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
  --jinja --reasoning off --device CUDA0
```

Apache-2.0. Não afiliado a Qwen, trohrbaugh ou cygnal.
