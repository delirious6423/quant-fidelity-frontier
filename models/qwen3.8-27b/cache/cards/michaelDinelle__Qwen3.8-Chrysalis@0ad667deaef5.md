---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- chrysalis
---

<p align="center"><img src="media/butterfly.webp" width="300" alt="Chrysalis's butterfly in its sea of characters"></p>

<h1 align="center">Chrysalis</h1>

<p align="center">
  <a href="https://x.com/michaeldinelle"><img alt="Follow on X" src="https://img.shields.io/badge/follow-%40michaeldinelle-000?logo=x"></a>
  <a href="https://ko-fi.com/michaeldinelle"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center"><a href="https://github.com/michaelDinelle/chrysalis-inference"><img alt="Get Chrysalis on GitHub" height="44" src="https://img.shields.io/badge/GitHub-Get%20Chrysalis-181717?style=for-the-badge&logo=github&logoColor=white"></a> <img alt="Coming soon" height="44" src="https://img.shields.io/badge/Coming%20soon-555?style=for-the-badge"></p>

<p align="center"><img src="media/pagoda.webp" width="900" alt="A voxel pagoda garden that Qwen3.8-Flash-Next wrote in Chrysalis, running in the browser"><br>
<sub>A voxel pagoda garden written by Qwen3.8-Flash-Next in Chrysalis.</sub></p>

## Qwen3.8-Flash-Next: Chrysalis vs Strata

<p align="center"><img src="media/flash-next.png" width="860" alt="Qwen3.8-Flash-Next in Chrysalis and Strata on an RX 7900 XTX, an RTX 3090 and an M4 Max: time to first token and writing speed at five prompt lengths, error against Q8_0, load time and memory"></p>

## Qwen3.8-27B: Chrysalis vs llama.cpp and MLX

<p align="center"><img src="media/27b.png" width="860" alt="Qwen3.8-27B in Chrysalis, llama.cpp and MLX on an RX 7900 XTX, an RTX 3090 and an M4 Max: time to first token and writing speed at five prompt lengths, error against Q8_0, load time and memory"></p>

## Files

| File | Model | Size |
| --- | --- | ---: |
| `qwen3.8-27b/qwen3.8-27b.gguf` | Qwen3.8-27B: IQ4_XS, Q6_K for the sensitive layers, Q8_0 for K/V and the MTP head | 15.9 GB |
| `qwen3.8-flash-next/qwen3.8-flash-next.gguf` | Qwen3.8-Flash-Next: Q6_K / Q8_0 dense, experts IQ4_XS and IQ4_NL (the most used) and Q3_K / Q2_K, the MTP block | 63.4 GB |
| `<model>/mmproj.gguf` | Each model's vision tower | 0.9 GB each |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Shared by both models | |

## Licenses

- Qwen3.8-27B: Apache-2.0 ([`LICENSE-qwen3.8-27b`](LICENSE-qwen3.8-27b))
- Qwen3.8-Flash-Next: Qwen Community License 1.0 ([`LICENSE-qwen3.8-flash-next`](LICENSE-qwen3.8-flash-next))
