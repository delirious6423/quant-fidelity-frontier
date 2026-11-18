---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
- speculative-decoding
---

# DeepSeek V4.1 Flash 1017 — Razor GGUF

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, packed from the official checkpoint by [Razor](https://github.com/flaviodelgrosso/razor) to run on a single 128 GB Mac.

| File | Size | SHA-256 | What |
|---|---:|---|---|
| `DeepSeek-V4.1-Flash-1017-TRL-R2.gguf` | 349.3 GB | `2917b6b1db31d6eb1905491d46bea81b81d9d6b8d1b05828c867d5c4f1b6f08e` | The model, routed experts at ~2 bits per weight |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8.6 GB | `7a2fe312c8b5752bdcecbb4d2fb782efe5bbcd33cdc4f0568ce2bbe591f5ffd5` | DSpark drafter for speculative decoding |

> [!IMPORTANT]
> These files run only in Razor on Apple Silicon. llama.cpp and other GGUF runtimes cannot load them.

## Use

```sh
razor download deepseek41 deepseek41-dspark
razor serve --model ~/.razor/models/DeepSeek-V4.1-Flash-1017-TRL-R2.gguf \
  --speculation opportunistic --dspark ~/.razor/models/DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

Routed experts stream from the internal SSD into a cache, so the model fits a 128 GB Mac.

## Quality and speed

Against the full-precision checkpoint weights, 48 held-out prompts:

| Category | Top-1 agreement |
|---|---:|
| chat | 88.4 % |
| multilingual | 78.9 % |
| reasoning | 96.1 % |
| coding | 97.4 % |

Decode on an Apple M5 Max, 128 GB (SSD streaming, 2,048-token prompt): **30.8–31.8 tok/s** on natural text; **56.7–56.8 tok/s** with DSpark `exact` speculation on a repetitive prompt. On natural prose DSpark is mostly gated off and runs slightly slower than plain decode. Full tables: [Benchmarks](https://github.com/flaviodelgrosso/razor/blob/main/docs/benchmarks.md).

## License

MIT, as upstream. Model weights © DeepSeek ([license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE)).
