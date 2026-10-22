---
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- ternary
- ingot
---

# Ingot-genesis-27B-Q2

The genesis model of Ingot's **27B-Q2** competition: the first king that every q2 submission is
scored against.

This is an unmodified copy of `Ternary-Bonsai-2-27B-PQ2_0.gguf` from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
at revision `b072e1d3b35a0a630cece372c2127528e0994386`, renamed to `Ingot-genesis-27B-Q2.gguf`.
Created using Bonsai by Prism ML.

| | |
|---|---|
| File | `Ingot-genesis-27B-Q2.gguf` |
| Size | 7,206,168,928 bytes |
| sha256 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| Format | GGUF, PQ2_0: ternary weights (-1, 0, +1) with one fp16 scale per 128 weights |
| Built from | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

## Running it

PQ2_0 needs the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp); stock
llama.cpp does not load it. Ingot evaluates with the fork at tag `prism-b10777-e3de715`.

```bash
llama-cli -m Ingot-genesis-27B-Q2.gguf -p "Hello"
```

This file is the text model. The vision projector (`mmproj`) is in the
[upstream repository](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## License

Apache 2.0, as the upstream model: see `LICENSE` and `NOTICE.txt`, copied unchanged from
prism-ml/Ternary-Bonsai-2-27B-gguf. Bonsai is copyright 2026-present Prism ML, Inc.; Qwen3.8-27B
is copyright 2026 Alibaba Cloud. This repository is not affiliated with or endorsed by Prism ML or
Alibaba Cloud.
