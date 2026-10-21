---
language:
  - en
  - fr
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: veloGB10
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - multimodal
  - conversational
  - nvfp4
  - gptq
  - mr-gptq
  - w4a4
  - w4a16
  - gb10
  - dgx-spark
---

# Qwen3.8-27B MR-GPTQ NVFP4 v4

This is a weight-only / weight-and-activation NVFP4 quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), prepared for the
[`veloGB10`](https://github.com/kedric/veloGB10) runtime on NVIDIA GB10 systems such as
DGX Spark.

This artifact is **not a fine-tune**. It preserves the base model architecture and tokenizer while
replacing supported weights with a custom packed NVFP4 representation. The release contains the
text model, MTP weights, vision tower, processor files, chat template, and activation scales. A
DFlash2 drafter is not included and must be installed separately if speculative decoding is wanted.

## Compatibility warning

The files use the custom `nvfp4-pack-quantized` layout and Hadamard-transformed weights expected by
`veloGB10`. Despite the `compressed-tensors` metadata in `config.json`, this release is **not
drop-in compatible with stock Transformers, vLLM, llama.cpp, or generic GPTQ loaders**. Loading it
with an implementation that does not understand the packing and transform metadata will fail or
produce incorrect output.

The Hugging Face hosted inference widget is therefore not expected to work for this repository.

## Quantization recipe

| Setting | Value |
| --- | --- |
| Weight format | NVFP4 / E2M1, packed W4 |
| Transform | `hadamard16` |
| Calibration | 512 samples x 2048 tokens |
| Hessian damping | 0.01 |
| Clip search | 7 ratios |
| Activation order | static |
| Global-scale optimization | alternating least squares, 4 iterations |
| MR-GPTQ groups | attention, MLP, GDN, LM head |
| RTN groups | MTP, embeddings |
| W4A4 activation scales | 496 positive finite scales, merged from main and long-context calibration |

Version v4 predates the optional local-Hessian NVFP4 scale sweep introduced later in `veloGB10`.
Its block scales use the clip-MSE/static-activation-order route recorded in `config.json`.

## Size and contents

- Packed tensor payload: 16,293,197,196 bytes.
- Four Safetensors shards.
- 2,708 entries in `model.safetensors.index.json`.
- Tokenizer, chat template, image and video processor configuration included.
- `input_global_scale.json` contains the optional W4A4 prefill scales.

## Run with veloGB10

Build a recent `qwen3.8-next-flash` branch of the runtime. This release was prepared against
commit `cc8837beed981cccc23725319d2b36036f04a915`.

### Conservative W4A16 mode

Without `GB10_W4A4_PREFILL`, weights remain W4 while activations use BF16:

```bash
CUDA_VISIBLE_DEVICES=0 \
./target/release/gb10_inference \
  --server \
  --model-dir /path/to/Qwen3.8-27B-MR-GPTQ-NVFP4-v4 \
  --port 9000 \
  --max-seq-len 226114 \
  --max-batch 1 \
  --prefix-cache on \
  --mtp auto
```

### Optional W4A4 prefill

W4A4 activation scales are included, but activation quantization is intentionally not enabled in
the recommended launch command. Treat every W4A4 route as experimental and benchmark it on the
intended workload before deployment. In local tool-calling tests, enabling GDN A4 caused a visible
quality regression.

## Evaluation status

The `attn,mlp` W4A4 profile was selected after a local tool-calling evaluation where the recorded
v4 run completed with zero errors and outperformed the corresponding v5 run. The full prompts and
machine-readable report are not included here, so this should not be treated as a published or
independently reproducible benchmark.

Quantization can still change probabilities and generation behavior. Test reasoning, multilingual
responses, long context, vision, structured outputs, and tool calls for your own deployment.

## Limitations and safety

- This release inherits the capabilities, limitations, biases, and license of the base model.
- Quantization is not safety training and does not protect against prompt injection by itself.
- Calibration improves numerical coverage but cannot guarantee correctness or prevent hallucination.
- The maximum configured context is not a guarantee that every deployment has enough KV-cache
  memory for that context and requested output length.
- W4A4 routes are hardware- and kernel-specific; use W4A16 as the compatibility baseline.

See the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the original model details
and intended-use guidance.

## License

The base model is distributed under Apache License 2.0. A copy is included in `LICENSE`.
