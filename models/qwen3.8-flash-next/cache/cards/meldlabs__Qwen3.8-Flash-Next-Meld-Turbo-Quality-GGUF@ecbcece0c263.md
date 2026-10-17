---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- meld-turbo
- gguf
- exl3
- apple-silicon
- moe
---

# Qwen3.8-Flash-Next — Meld Turbo quality profile (GGUF)

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B parameters, 6B active per token) packed for the **quality profile** of [Meld Turbo](https://github.com/MeldlabsAI/meld-turbo), which runs the model locally on Apple silicon Macs with 96 GB of unified memory or more.

[GitHub](https://github.com/MeldlabsAI/meld-turbo) · [Project page](https://meldlabs.ai/meld-turbo) · [Performance notes](https://github.com/MeldlabsAI/meld-turbo/blob/main/docs/PERFORMANCE.md)

**This file needs the Meld Turbo engine, version 0.1.1 or later**. Its experts use EXL3 tensor types that upstream llama.cpp and other GGUF runtimes do not support. Build the engine with `./setup.sh` from the Meld Turbo repository.

## Files

| File | Size | For |
|---|---:|---|
| `Qwen3.8-Flash-Next-Meld-Turbo-Quality-2.6bpw.gguf` | 74.9 GB | Macs with 96 GB or more |

## What is inside

| Part | Format | Source |
|---|---|---|
| Routed experts, 48 layers × 512 (gate and up fused, down) | EXL3, 2.625 bits per weight, with their F16 sign/scale vectors | the 2.6 bpw pack of [Sushi](https://github.com/beamivalice/sushi), trellis bytes copied unchanged |
| Attention, linear-attention, shared experts (all but 7 down projections), embeddings, output | Q8_0 | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| n-gram embedding table, 7 shared-expert down projections | IQ4_NL | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| Routers, hyper-connections, sparse-attention indexer, norms | BF16 / F32, unquantized | the same GGUF |

EXL3 is the trellis-coded quantization format of [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp.

## Quality

| | Meld Turbo fast | **this file** |
|---|---:|---:|
| Perplexity, wikitext, 16 × 512 tokens (Q8_0 reference: 2.108) | 2.826 | **2.234** |
| GSM8K (200) / HumanEval (164) / MMLU (200), greedy, thinking off | 98.0% / 93.3% / 82.5% | 96.5% / 97.6% / 87.5% |

Speed and memory on an M2 Max: see the [Meld Turbo README](https://github.com/MeldlabsAI/meld-turbo/blob/main/README.md#results) and [docs/PERFORMANCE.md](https://github.com/MeldlabsAI/meld-turbo/blob/main/docs/PERFORMANCE.md).

## Use

```zsh
git clone https://github.com/MeldlabsAI/meld-turbo && cd meld-turbo
./setup.sh
M=~/.meld-turbo/models; mkdir -p $M
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" --local-dir $M
mv $M/MTP/*.gguf $M/
hf download meldlabs/Qwen3.8-Flash-Next-Meld-Turbo-Quality-GGUF Qwen3.8-Flash-Next-Meld-Turbo-Quality-2.6bpw.gguf --local-dir $M
PROFILE=quality ./serve.sh      # OpenAI-compatible API at http://127.0.0.1:8080/v1
```

## License

Derived from Qwen3.8-Flash-Next and distributed under the **Qwen Community License 1.0**; the full text is in [LICENSE](LICENSE). Copyright (c) 2026 Qwen.
