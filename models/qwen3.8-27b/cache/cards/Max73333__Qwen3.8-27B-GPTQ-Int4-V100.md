---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen
  - qwen3.8
  - gptq
  - int4
  - quantized
  - vllm
  - tesla-v100
  - sm70
---
## Docker / vLLM runtime for NVIDIA V100

A tested Docker + patched vLLM setup for running this model on NVIDIA Tesla V100 (SM70) is available here:

**GitHub:** https://github.com/pmaxx73/qwen38-v100

Tested configuration:
- NVIDIA Tesla V100-SXM2 32 GB
- CUDA 12.8
- PyTorch 2.10.0+cu128
- cuDNN 9.10.2
- Triton 3.6.0
- vLLM commit `bcf2be96120005e9aea171927f85055a6a5c0cf6`
- ~37–40 tokens/s after warm-up

Current limitation:
- Docker cold start: ~15 minutes
- First generation request after startup: ~10 minutes


# Qwen3.8-27B-GPTQ-Int4-V100

GPTQ Int4 quantization of **Qwen/Qwen3.8-27B**, prepared and tested for inference on an **NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0 / sm_70)**.

The main goal of this build is to provide a practical 4-bit variant that can run on Volta GPUs, where some newer AWQ / compressed-tensors kernels require a newer compute capability.

## Model

- Base model: `Qwen/Qwen3.8-27B`
- Quantization: GPTQ
- Weight bits: 4
- Group size: 128
- `desc_act`: `false`
- `sym`: `true`
- Intended use: text generation
- Tested GPU: NVIDIA Tesla V100-SXM2-32GB
- Compute capability: 7.0 (`sm_70`)

## Quantization configuration

The model was quantized with:

```python
from gptqmodel import QuantizeConfig

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
)
```

Quantization was performed from the original `Qwen/Qwen3.8-27B` weights using GPTQModel.

## Tested environment

The following environment was used successfully:

```text
GPU: NVIDIA Tesla V100-SXM2-32GB
Compute capability: 7.0 / sm_70
CUDA toolkit: 12.8
PyTorch: 2.10.0+cu128
vLLM: source build based on 0.18.x
GDN prefill backend: Triton
```

The vLLM build was compiled from source with support for `sm_70`.

Example build environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="7.0"
```

## vLLM compatibility note

This model was tested with a **patched source build of vLLM**.

Stock vLLM compatibility depends on the version. In the tested source checkout, several compatibility fixes were required for the Qwen3.8 text-only path and for CUDA Graph / hybrid KV-cache initialization.

In particular:

- Qwen3.8 uses Qwen3.5-family model classes internally.
- The text-only `Qwen3_5ForCausalLM` architecture required explicit routing in the tested vLLM source.
- Multimodal initialization had to be disabled for this text-only use case.
- CUDA Graph profiling hit an ambiguous hybrid KV-cache layout when the profiling cache shape was `[2, 2, ...]`.
- Removing the ambiguity check for that profiling case allowed CUDA Graph mode to initialize correctly.

Because of this, do not assume every stock vLLM release will run this model unchanged.

## Recommended vLLM launch command

The tested launch configuration was:

```bash
vllm serve /path/to/Qwen3.8-27B-GPTQ-Int4 \
  --host 0.0.0.0 \
  --port 8002 \
  --served-model-name qwen38 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gdn-prefill-backend triton
```

For the tested V100 setup, **do not use `--enforce-eager` unless you need it for debugging or compatibility**. CUDA Graph / compiled execution was much faster after warmup.

## Performance

Measured on one NVIDIA Tesla V100-SXM2-32GB:

| Test | Result |
|---|---:|
| 100 generated tokens, first warmed request | ~15 s |
| 100 generated tokens, subsequent request | ~2.7 s |
| 500 generated tokens | ~12.5 s |
| Steady-state generation | ~37–40 tokens/s |

These measurements were taken with:

```text
max_model_len = 4096
max_num_seqs = 1
GPTQ Int4
Triton GDN backend
compiled / CUDA Graph execution
```

The first request after startup can be significantly slower because Triton kernels and execution graphs need to warm up.

## Example OpenAI-compatible request

```bash
curl http://127.0.0.1:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38",
    "prompt": "Кратко объясни, что такое квантование нейросетей.",
    "max_tokens": 100,
    "temperature": 0.2
  }'
```

## Reasoning output

Depending on the tokenizer / chat template and API endpoint, the model may emit reasoning blocks such as:

```text
<think>
...
</think>
```

If you use the model in an application, you may want to configure the chat template or post-processing so reasoning is handled separately from the visible assistant answer.

## Longer context

The model was initially validated with:

```text
--max-model-len 4096
```

Longer context sizes such as 8192, 16384, or 32768 should be tested incrementally because available KV-cache memory depends on:

- GPU memory utilization
- vLLM version
- CUDA Graph memory usage
- number of concurrent sequences
- hybrid attention / recurrent-state cache layout

## Why this quantization exists

Some pre-quantized AWQ / compressed-tensors builds use kernels that require a GPU compute capability newer than Volta.

On a Tesla V100 (`sm_70`), those builds can fail with messages indicating a minimum compute capability such as `7.5`.

This GPTQ Int4 build was created specifically to provide a working alternative for V100-class hardware.

## Known limitations

- Tested primarily for **text generation**.
- Vision / multimodal functionality is not claimed or validated for this quantized build.
- The tested vLLM environment required source-level compatibility patches.
- First inference after startup can be much slower than subsequent requests.
- FlashAttention 2 is not available on V100; the tested setup uses Triton-based kernels.
- Performance may vary significantly between vLLM, PyTorch, CUDA and Triton versions.

## Suggested repository naming

A descriptive Hugging Face repository name would be:

```text
Qwen3.8-27B-GPTQ-Int4-V100
```

or:

```text
Qwen3.8-27B-GPTQ-Int4-sm70
```

## Base model

Original model:

`Qwen/Qwen3.8-27B`

Please review and comply with the original model's license and usage terms when using or redistributing this quantized derivative.

## Credits

- Qwen team for the original Qwen3.8 model.
- GPTQModel contributors for GPTQ quantization tooling.
- vLLM contributors for the inference engine.
