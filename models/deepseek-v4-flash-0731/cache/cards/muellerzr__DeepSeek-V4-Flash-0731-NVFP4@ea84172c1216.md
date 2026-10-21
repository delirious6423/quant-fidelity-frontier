---
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
license: mit
library_name: Model Optimizer
tags:
- deepseek-v4
- quantized
- nvfp4
- modelopt
- speculative-decoding
- dspark
---

# DeepSeek-V4-Flash-0731-NVFP4

NVFP4 conversion of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), produced with NVIDIA Model Optimizer for Blackwell inference.

Only the target model's 43 routed-expert groups were converted from native MXFP4 to NVFP4. Attention, shared experts, output head, and the embedded three-layer DSpark/MTP module retain their source precision. This is a quantized checkpoint, not a fine-tune.

## Quantization

- NVFP4 E2M1 weights and activations, group size 16
- 33,024 routed-expert projections converted
- 100% of MXFP4 weight blocks converted losslessly
- FP8 E4M3 KV cache used during validation
- ModelOpt source commit: `73d778422388f0e849ecb180375d34ac445711ca`

Calibration used 128 prompts, evenly divided among code, math, instruction/chat, and general text from NVIDIA OpenCodeReasoning, OpenMathReasoning, Nemotron SFT instruction-following/chat v2, and CNN/DailyMail. Evaluation prompts were kept separate.

## Validation

Validated with vLLM 0.27.1 on 4× NVIDIA B200 GPUs (TP4 + expert parallelism).

| Check | Source 0731 | NVFP4 |
|---|---:|---:|
| Strict-format sanity set (math/code/chat) | 14/24 | 14/24 |
| DSpark draft-token acceptance | 48.54% | 54.04% |
| Mean accepted tokens per draft step | 3.40 | 3.78 |

There were no per-case regressions and 18/24 generated responses matched exactly. This small set is a conversion sanity check, not a substitute for full downstream evaluation.

## Serving

Use vLLM's automatic MoE backend; do not force `deep_gemm_mega_moe` for the NVFP4 target experts.

vLLM 0.27.1 applies the checkpoint-wide NVFP4 setting to the excluded DSpark layers unless dispatch follows `quantized_layers`. Mount the included `vllm_quant_config.py` compatibility overlay until equivalent manifest-driven dispatch is available upstream:

```bash
hf download muellerzr/DeepSeek-V4-Flash-0731-NVFP4 \
  vllm_quant_config.py --local-dir dsv4-nvfp4-runtime

docker run --rm --gpus all --ipc=host --network=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/dsv4-nvfp4-runtime/vllm_quant_config.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/quant_config.py:ro" \
  vllm/vllm-openai:latest \
  muellerzr/DeepSeek-V4-Flash-0731-NVFP4 \
    --served-model-name dsv4-flash-0731-nvfp4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8_e4m3 \
    --block-size 256 \
    --attention-config '{"use_fp4_indexer_cache":true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Expected backend selection is `FLASHINFER_TRTLLM` for target NVFP4 experts and `FLASHINFER_TRTLLM_MXFP4_MXFP8` for the preserved DSpark experts. See [SERVING.md](./SERVING.md) for the concise compatibility notes.

## License and limitations

The checkpoint follows the [source model's MIT license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE). The source model's limitations and intended-use guidance continue to apply. Validate accuracy, safety, context length, and throughput for your deployment workload.
