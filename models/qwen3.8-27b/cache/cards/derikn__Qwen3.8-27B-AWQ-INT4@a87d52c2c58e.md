---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
datasets:
  - NeelNanda/pile-10k
tags:
  - qwen
  - qwen3.8
  - awq
  - int4
  - 4-bit
  - auto-round
  - vllm
  - xpu
  - intel-arc
  - image-text-to-text
  - vision
  - mtp
  - reasoning
  - thinking
  - tool-calling
quantized_by: derikn
---

# Qwen3.8-27B AWQ INT4 (W4A16, group 128)

4-bit weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built with
Intel's [AutoRound](https://github.com/intel/auto-round) and tested on Intel Arc Pro B70 GPUs.

This is an unofficial community quantization by `derikn`. The Qwen team did not author it and does not
endorse it. For the model itself, its capabilities and its evaluation results, read the
[base model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## What this is

| | |
|---|---|
| Base model | `Qwen/Qwen3.8-27B` (Apache 2.0) |
| Parameters | 27,781,427,952 (see the note on the parameter count below) |
| Quantization | AWQ, weight-only, 4-bit (W4A16) |
| Group size | 128, symmetric, `gemm` version |
| Method | AutoRound `0.15.1`, `auto_awq` export |
| Kept at 16-bit | `lm_head`, the vision tower, the MTP head, the linear-attention projections, the visual merger |
| Size on disk | 19.6 GB: 7 safetensors shards plus `model_extra_tensors.safetensors` for the MTP head |
| Native context | 262,144 tokens, same as the base model |
| Vision | the vision tower and the Qwen2VL image processor are intact, so image inputs work |
| MTP | the head is retained at 16-bit and usable, see the MTP section below |
| Chat template | included, unchanged from the base model |

101 modules are held at 16-bit, listed in `quantization_config.modules_to_not_convert`. They cover
`lm_head`, every `linear_attn.in_proj_a` and `in_proj_b` projection, the visual merger, the vision
blocks, and `mtp`.

Only the linear weights are quantized. No fine-tuning and no extra training data went into this.

### Note on the parameter count

The Safetensors panel on this page reports a model size well below the real one. That is not the size of
the model. The 4-bit weights are packed into int32 tensors, eight weights per int32, and the panel's
total counts that packed storage instead of the parameters it encodes.

This checkpoint holds **27,781,427,952 parameters**, the same count as the base model. The Hub API's own
per-dtype breakdown adds up to that figure while the panel's single total does not. `I32` and `BF16`
describe storage, not the precision of the model: packed 4-bit weights, and the tensors held at
bfloat16.

## How it was made

Quantized from the bf16 checkpoint with AutoRound 0.15.1 on a single Intel Arc Pro B70, with weights
streamed to the card so VRAM stays bounded. The run was driven from the Quantize tab of a private
dashboard, so this is the pipeline's invocation rather than a script you can run as-is:

```
python quantize.py \
  --model <base-model-dir> \
  --format auto_awq --device xpu:0 --low-gpu-mem \
  --n_samples 128 --seqlen 1024 --batch_size 8 \
  --torch-compile
```

Calibration used the quantizer's default set, **NeelNanda/pile-10k**, at 128 samples of 1024 tokens
each. The wrapper is not published, so that exact command will not run for you. These are the AutoRound
settings it passes, which are the whole recipe:

- export format `auto_awq`: AWQ, 4-bit weight-only, group size 128, symmetric, `gemm` version
- `lm_head`, the vision tower, the MTP head, the linear-attention projections and the visual merger excluded
- calibration on NeelNanda/pile-10k, 128 samples of 1024 tokens, batch size 8
- weights streamed to the device during tuning, to keep VRAM bounded
- `torch.compile` enabled on the tuning step

## Running it

Tested on **vLLM XPU** (`vllm/vllm-openai-xpu:nightly`, vLLM `0.30.1rc1`) with tensor parallelism across
two Arc Pro B70s, an fp8 KV cache and the full 256K context:

```bash
vllm serve /models/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen3.8-27b-awq \
  --quantization awq --dtype bfloat16 \
  --max-model-len 262144 --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.91 \
  --enable-prefix-caching \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --trust-remote-code --tensor-parallel-size 2
```

Container settings that matter on Arc: `--device=/dev/dri`, `--group-add` for the render and video
groups, `--ipc=host`, `--security-opt seccomp=unconfined`, and `ZE_AFFINITY_MASK` set to the cards you
want to use. Without the device, group and seccomp combination Level Zero reports zero devices.

### MTP speculative decoding

The MTP head ships in `model_extra_tensors.safetensors` at 16-bit, and `quantization_config` declares it
unquantized through the `mtp` entry in `modules_to_not_convert`.

That entry is required. AutoRound quantizes the module tree it loads, and the MTP head sits outside it,
so without the exclusion vLLM builds the head as a packed 4-bit linear and the draft model fails at load:

```
ValueError: There is no module or parameter named 'fc.weight' in Qwen3_5MultiTokenPredictor
```

With the exclusion in place the head loads in under a second on this hardware, and speculative decoding
runs at roughly 70 percent draft acceptance in testing.

### Thinking and tool calling

Tested with the `qwen3_coder` tool parser and the `qwen3` reasoning parser, plus thinking enabled through
the chat template:

```
--default-chat-template-kwargs '{"enable_thinking": true, "reasoning_preserve": true, "reasoning_effort": "low"}'
```

`enable_thinking` turns thinking on or off, and `reasoning_effort` accepts the OpenAI scale that vLLM
forwards to the template.

### What was tested, and what was not

Tested:

- 256K context loads and serves, at tensor parallel 2 with an fp8 KV cache.
- MTP speculative decoding loads and drafts, from the retained `model_extra_tensors.safetensors`.
- Tool calling and reasoning work with the parsers above.
- Text generation returns correct answers; the vision tower and processor config load, so image input
  remains available.

Not tested:

- No accuracy or throughput numbers, and no image-input evaluation, are published with this repository.
  The quantizer's own measurements come from one home-lab setup (Arc Pro B70, XPU vLLM nightly, fp8 KV
  cache) and are not a fair comparison for anyone running different hardware or software. Model quality
  is the base model's, so read its card.

## Licence and attribution

Apache License 2.0, inherited from the base model. The base repository ships the licence text, and it is
included here as `LICENSE`, alongside `NOTICE.txt`.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
- Quantization tooling: [intel/auto-round](https://github.com/intel/auto-round) (Apache 2.0)
- Quantizer: a private in-house pipeline around AutoRound (not published)
