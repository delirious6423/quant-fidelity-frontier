---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
- zh
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- gptq
- 4-bit
- w4a16
- compressed-tensors
- llmcompressor
- vllm
- gated-deltanet
---

# Qwen3.8-27B, GPTQ 4-bit (W4A16, group size 128)

A 4-bit weight-quantized copy of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), made with GPTQ in [llm-compressor](https://github.com/vllm-project/llm-compressor). The checkpoint uses the compressed-tensors `pack-quantized` format and loads directly in [vLLM](https://github.com/vllm-project/vllm), where the quantized layers run on the Marlin kernel.

The weights file is 18.6 GB. The dense bf16 model is 55.6 GB.

**Scope of testing.** Only text input was calibrated and evaluated. The vision tower is carried over unchanged in bf16; image and video input were not tested.

## Quantization

| Setting | Value |
|---|---|
| Method | GPTQ (`GPTQModifier`), llmcompressor 0.13.0, compressed-tensors 0.18.0 |
| Scheme | W4A16: 4-bit integer weights, 16-bit activations |
| Granularity | symmetric, one scale per group of 128 weights |
| GPTQ options | `dampening_frac=0.01`, `actorder=static`, `block_size=128` |
| Calibration data | 256 windows of 2,048 tokens, drawn at random (seed 7) from the wikitext-103 training split (text only) |
| Quantized | 400 matrices of the language model: in the 48 gated-DeltaNet (linear-attention) blocks `in_proj_qkv`, `in_proj_z`, `out_proj`; in the 16 full-attention blocks `q/k/v/o_proj`; in all 64 blocks the MLP `gate/up/down_proj` |
| Not quantized | `lm_head`, token embeddings, the vision tower, the small decay-gate projections `in_proj_a` / `in_proj_b` (0.09% of the parameters), normalization layers |
| Not included | the multi-token-prediction (MTP) head of the base checkpoint (15 tensors, 425M parameters). The transformers model class used for quantization does not load it, so it is absent here and MTP-based speculative decoding is not available with this checkpoint. Normal generation does not use it. |
| Build time | 37 minutes on one NVIDIA A100 40 GB |

The exact recipe is in `recipe.yaml`. `config.json` is the dense model's configuration verbatim plus the `quantization_config` block; tokenizer, chat template and processor configuration are the base model's.

Note that this is not the recipe of the official `Qwen3.5-27B-GPTQ-Int4`, which quantizes the MLP only. Here the attention and linear-attention projections are quantized as well.

## Evaluation

Both columns were measured with the same pipeline: vLLM 0.31.0 and lm-eval 0.4.13, bf16 activations.

| Metric | Dense (bf16) | This model | Paired difference (95% interval) |
|---|---:|---:|---:|
| Perplexity, wikitext-2 test (lower is better) | 6.16 | 6.28 | |
| MMLU, 0-shot (%) | 84.89 | 84.08 | -0.81 [-1.20, -0.42] |
| MMLU, 5-shot (%) | 84.21 | 82.31 | -1.90 [-2.34, -1.47] |
| GSM8K, chat template, thinking off (%) | 97.35 | 96.21 | -1.14 [-1.99, -0.29] |

- MMLU uses at most 512 questions per subject, 12,233 questions in total, lm-eval's default prompts without the chat template (log-likelihood scoring).
- GSM8K is the full test set of 1,319 questions, `flexible-extract`, run with the chat template, `enable_thinking=False` and up to 1,024 generated tokens.
- Do not use lm-eval's default GSM8K prompt with this model family: without the chat template the model opens a thinking block in many answers (447 of 1,319 for the dense model, 97 for this one) and the default 256-token limit cuts them off, which produced 72.7% for dense and 84.5% for the quantized model, an artifact and not a quality difference.
- Perplexity is computed over 12 non-overlapping windows of 2,048 tokens.

## Speed

Measured with vLLM 0.31.0 on one NVIDIA A100 80 GB, with a 2,048-token prompt and 256 generated tokens (greedy, 3 repeats, median).

| Load | Dense (bf16) | This model | Ratio |
|---|---:|---:|---:|
| 1 request, time to first token | 0.452 s | 0.577 s | 1.28x slower |
| 1 request, time per output token | 34.72 ms | 13.34 ms | — |
| 1 request, tokens/s | 28.8 | 75.0 | 2.60x |
| 8 concurrent requests, total tokens/s (decode-only) | 156 (213) | 235 (497) | 1.50x (2.34x) |
| 32 concurrent requests, total tokens/s (decode-only) | 312 (672) | 326 (1226) | 1.04x (1.82x) |

Total tokens/s counts the wall-clock time of the whole batch including prompt processing; the decode-only figure in parentheses excludes it. Prefill is slower under W4A16 because the weights are unpacked before the matmul; decoding is faster because each token reads a third of the bytes. The quantized model also fits a 40 GB GPU, where the dense model does not.

## Usage

Tested with vLLM 0.31.0:

```bash
pip install vllm
vllm serve libertyhhn/Qwen3.8-27B-GPTQ-W4A16-g128 --max-model-len 8192
```

```python
from vllm import LLM, SamplingParams
llm = LLM("libertyhhn/Qwen3.8-27B-GPTQ-W4A16-g128", max_model_len=8192)
out = llm.chat([{"role": "user", "content": "Explain GPTQ in two sentences."}],
               SamplingParams(temperature=0.6, max_tokens=512),
               chat_template_kwargs={"enable_thinking": False})
print(out[0].outputs[0].text)
```

## License

The base model is released under the Apache License 2.0 (see `LICENSE`); this quantized copy carries the same license.
