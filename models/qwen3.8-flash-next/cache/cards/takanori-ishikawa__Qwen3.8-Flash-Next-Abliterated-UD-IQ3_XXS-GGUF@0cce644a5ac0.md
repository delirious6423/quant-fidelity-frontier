---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - unsloth/Qwen3.8-Flash-Next-GGUF
pipeline_tag: text-generation
tags:
  - gguf
  - abliterated
  - uncensored
  - qwen3.8
  - apple-silicon
---

# Qwen3.8-Flash-Next Abliterated UD-IQ3_XXS GGUF

An abliterated build of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on [Unsloth's UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) quantization. It complies with most HarmBench requests and scores within noise of the original on GSM8K and MMLU-Pro.

## Files

| File | Size |
| --- | --: |
| `Qwen3.8-Flash-Next-UD-IQ3_XXS-ple-split-00001-of-00004.gguf` … `00004-of-00004.gguf` | 76.3 GiB |
| `recipe/` | Direction vector, scripts, and rebuild steps |

The per-layer embedding table (`per_layer_token_embd.weight`, 26.8 GiB) is in its own shard so that engines can stream it from the SSD.

## Running

Loads in [llama.cpp](https://github.com/ggml-org/llama.cpp) or [Splash](https://github.com/incoai/splash). With the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), decode speed matches the original.

Heavy cyber-intrusion and chemical/biological requests can think past 32K tokens. A thinking budget (for example 8K) caps this.

## Evaluation

[GSM8K](https://github.com/openai/grade-school-math) test 50, [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) test 50, and [HarmBench](https://github.com/centerforaisafety/HarmBench) standard 100, sampled with seed 407. Run on Splash 1.4.0 with DFlash2 on an M4 Pro Mac mini (64 GB). HarmBench answers were judged by the original model with the [official HarmBench classifier prompt](https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/eval_utils.py).

No thinking budget, up to 32K output tokens:

| Model | GSM8K | MMLU-Pro | HarmBench complied / refused / cut off |
| --- | --: | --: | --- |
| [Original UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 48 | 43 | 2 / 98 / 0 |
| [orcarouter IQ3_XXS](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) | 48 | 35 | 76 / 8 / 16 |
| **This model** | 48 | 45 | 87 / 5 / 6 (2 unparsed) |

- With 50 questions at temperature 1.0, differences of a few questions are noise

## How it was made

1. Take the refusal direction of [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored), estimated from its BF16 weight difference to the original
2. Remove it from the output side of every matrix that writes to the residual stream in all 48 layers (attention / linear-attention output, shared- and routed-expert down projections, `token_embd`, `ple_value`): `W' = W − u (uᵀ W)`, starting from the original BF16 weights
3. Quantize each edited tensor back to Unsloth's type, repeating projection and quantization twice, and replace it in the UD-IQ3_XXS GGUF. Other tensors are copied byte for byte

See [`recipe/README.md`](recipe/README.md) to rebuild the files.

## License and credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen Community License 1.0 ([`LICENSE`](LICENSE))
- Quantization: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ3_XXS
- Refusal direction: [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored), Apache-2.0 ([`NOTICE`](NOTICE))
