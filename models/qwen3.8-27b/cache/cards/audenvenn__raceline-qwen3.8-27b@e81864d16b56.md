---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- raceline
- apple-silicon
- metal
- speculative-decoding
- mtp
- fp4
---

# Qwen3.8-27B for raceline

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted for [raceline](https://github.com/audenvenn/raceline),
a local agent runtime with its own inference engine for Apple Silicon Macs whose GPU has neural
accelerators (M5 and M6 families).

The model file is the engine's native format: 4-bit FP4 weights with a scale per row and per
column (4.0 bits per weight), quantized with GPTQ after a Hadamard rotation of the residual stream that is folded into the weights; the model's
multi-token-prediction head, which raceline uses for exact speculative decoding; and the tokenizer
and chat template. It is split into parts of under 2 GB described by `qwen3.8-27b.rlm.json`. It needs
raceline 0.2.0 or later and is not usable with other runtimes.

## Use

```sh
raceline pull qwen3.8-27b --source huggingface    # or without --source: the GitHub release
raceline chat qwen3.8-27b
```

See the [raceline README](https://github.com/audenvenn/raceline#quick-start) for installation and for
speed measurements against mlx-lm.

## Quality

Measured at conversion time against the float model on held-out text (1152 positions):
KL divergence 0.148 nats, perplexity 1.979, top-1 agreement 90.6%.
The draft head's top choice is the model's own next-but-one token 81.7% of the time.

| File | Size | SHA-256 |
|---|---|---|
| `qwen3.8-27b.rlm` (assembled) | 13.81 GB | `175925a25e0383e1dc78884ac56c2577dc2c1393d98ad7a5c72b298a74d82971` |

## License

Apache-2.0, as the original model.
