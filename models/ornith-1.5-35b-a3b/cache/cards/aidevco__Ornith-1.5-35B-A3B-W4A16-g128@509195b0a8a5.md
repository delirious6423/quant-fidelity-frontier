---
library_name: transformers
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - qwen3_5_moe
  - compressed-tensors
  - int4
  - w4a16
  - rocm
  - mi210
---

# Ornith-1.5-35B-A3B — W4A16 (int4 experts, group 128)

A data-free, symmetric int4 weight-only quantization of the routed experts of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) in the compressed-tensors `pack-quantized` format.
It was built to serve the model on one AMD Instinct MI210 (gfx90a, 64 GB) per replica. The experts run on vLLM's fused MoE WNA16
kernel, which on gfx90a needs symmetric weights (no zero points), so the usual asymmetric AWQ builds of this model family do not
serve fast there.

Everything about the model itself (architecture, training, intended use, benchmarks, limitations) is in the
[original model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). This card covers only what the quantization changes.

## What is quantized

| Part                                                                                | Precision                  | Why                                                        |
| ----------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------- |
| Routed experts `gate_proj` / `up_proj` / `down_proj` (40 layers × 256 experts)      | int4, group 128, symmetric | 64.4 GB of the 71.9 GB bf16 source; fused MoE WNA16 kernel |
| Gated DeltaNet (`linear_attn`), full-attention projections, shared experts, routers | bf16                       | recurrent, state and routing layers stay full precision    |
| Norms, embeddings, `lm_head`, MTP head (including its experts), vision tower        | bf16                       | standard                                                   |

- 24.1 GB in 16 shards, 30,720 quantized modules. About 21 GiB of GPU memory is in use after loading.
- The source stores the experts fused (`experts.gate_up_proj`, `experts.down_proj`). The quantizer splits them into per-expert
  `gate_proj` / `up_proj` / `down_proj`, the layout vLLM's loader expects (gate is the first half of `gate_up_proj`; probed on one
  shard: relative dequantization error 0.10 against the matching half, 1.4 against the other).
- Group 128 divides every quantized input dimension (512 and 2,048) at tensor-parallel size 1.

## How it was made

- Tool: llm-compressor 0.14.0 `model_free_ptq` with compressed-tensors 0.19.0, data-free (no calibration set), on the CPUs of one node, 13 minutes.
- Scheme: `num_bits=4`, `symmetric=true`, `strategy=group`, `group_size=128`, observer `memoryless-mse`; no activation quantization.
- Targets (both the checkpoint's and vLLM's module names):
  `re:^(?:model\.language_model|language_model\.model)\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$`
- Source revision: `10fbf86fed7ecee4a061f8b499a618f46001cac1`.
- Verified after the build: every scheme field, the targets in both name forms, every shard present. Then served on one MI210
  (vLLM v0.33.0, ROCm 10.0): `CompressedTensorsWNA16MoEMethod` with `TritonWNA16Experts`, KV cache 1,101,177 tokens at
  `--gpu-memory-utilization=0.90`, and four checks at temperature 0 passed (17 × 60 = 1,020; a 155-minute train trip with thinking off
  and on; a `get_weather` tool call with `city: "Amsterdam"`).
- No benchmark scores are claimed here: this card does not report task accuracy of the quantized model.

## Serving (vLLM, ROCm)

Quantization is detected from the config; do not pass `--quantization`.

```bash
vllm serve <this repo> \
  --max-model-len=32768 --gpu-memory-utilization=0.90 --kv-cache-dtype=auto \
  --max-num-batched-tokens=2048 --language-model-only \
  --limit-mm-per-prompt='{"image":0,"audio":0,"video":0}' \
  --reasoning-parser=qwen3 --tool-call-parser=qwen3_coder
```

One replica fits one MI210; two cards serve two independent replicas.

## Tuned fused-MoE config (optional)

`moe-configs/E=256,N=512,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json` is a Triton fused-MoE kernel config tuned for this build on
the MI210 at tensor-parallel size 1. vLLM does not read it from the model repository: download the folder and point
`VLLM_TUNED_CONFIG_FOLDER` at it. The startup log then names the file (`Using configuration from ...`) instead of `Using default MoE config`.

```bash
hf download aidevco/Ornith-1.5-35B-A3B-W4A16-g128 --include 'moe-configs/*' --local-dir .
export VLLM_TUNED_CONFIG_FOLDER=$PWD/moe-configs
```

- Tuned with vLLM v0.33.0 (`vllm/vllm:v0.33.0-rocm`, ROCm 10, Triton 3.8.0), all 12 batch sizes from 1 to 2048, on both MI210s (warm-up 60 min,
  tuning 9 h 02 min), with the `BLOCK_SIZE_K` tiles capped at the int4 group size (128); the finalists were re-timed 10 times on one card.
- Serving A/B on one MI210 (default, tuned, tuned, default; fresh pod per run; 1,536 prompt / 128 output tokens, 60 s windows): output
  tokens per second +2.3% to +3.4% at concurrency 1 and 2 and +0.3% to +0.5% at 4 (inside the run-to-run spread), inter-token latency 3.5%
  to 4.1% lower at concurrency 1 and 2, time to first token and energy per token unchanged within the noise.
- Only valid for this exact shape, card and dtype. ROCm 7.2 images name the card `Instinct_MI210`: copy the file under that name too. A newer
  vLLM or Triton can change the kernel; re-check before relying on it.

## License and attribution

This is a derivative of Ornith-1.5-35B-A3B by ornith-ai, distributed under the same license as the original (MIT, as stated in the original
model card's metadata). The original repository's card links a `LICENSE` file that is not present in the repository at the revision above, so
no license text is copied here; the terms are those of the original. The tokenizer, chat template and configuration files are copied unchanged
from the original repository; `config.json` adds only the quantization config.

The quantization changes the numerical precision of the weights listed above and nothing else. It is not an official ornith-ai release, and
ornith-ai has not reviewed or endorsed it.
