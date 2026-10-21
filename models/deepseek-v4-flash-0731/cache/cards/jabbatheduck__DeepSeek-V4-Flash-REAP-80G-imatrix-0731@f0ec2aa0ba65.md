---
license: mit
language:
- en
- zh
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- deepseek-v4
- mixture-of-experts
- quantized
- gguf
- dgx-spark
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 REAP 80G

A balanced, memory-efficient GGUF quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) with 160 of 256 routed experts retained via REAP pruning.

This is an experimental derivative. It is not an official DeepSeek release, and pruning/quantization can reduce quality. Evaluate it on your own coding, reasoning, tool-use, and long-context workloads before production use.

## What's Inside

| Item | Value |
|---|---|
| Source revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Original routed experts | 256 per MoE scope |
| Retained routed experts | 160 per MoE scope |
| Routed experts removed | 37.5% |
| Router top-k | 6, unchanged |
| Backbone MoE layers | 43 |
| DSpark MTP blocks | 3, mapped to backbone layers 40, 41, 42 |
| GGUF tensor count | 1,328 |
| GGUF size | ~70.9 GB |

## Quantization Recipe

This model uses a mixed-precision scheme optimized for quality at reduced size:

- **Experts (routed w1/w3):** `IQ2XXS` (2.06 bpw)
- **Routed W2 (down projection):** `Q4_K` (4.58 bpw)
- **Attention projections:** `Q8_0`
- **Shared experts:** `Q8_0`
- **Output head:** `Q8_0`
- **Token embeddings:** `BF16`

The W2 projection was upgraded to Q4_K to preserve the MoE expert quality that degrades fastest under aggressive quantization, while keeping the rest of the model at high precision.

## How It Was Made

1. **REAP pruning** retained 160/256 experts per layer from the DeepSeek-V4-Flash-0731 checkpoint using transferred rankings from the `0xSero/deepseek-v4-flash-reap-observations-v2` dataset.
2. **GGUF conversion** used the `antirez/ds4` `deepseek4-quantize` tool with the recipe above.
3. **Imatrix** was collected via the ds4 runtime's routed-MoE activation collector on a Q4KExperts template GGUF, using the `DeepSeek-V4-Flash-chat-v2-routed-moe-ds4` calibration corpus.
4. **Verification** included structural checks (1,328 tensors, zero failures) and runtime smoke tests.

## Usage

Use a DeepSeek-V4-capable runtime. Generic GGUF loaders may not understand the model's custom MXFP4, sparse-attention, tokenizer, and DSpark MTP components.

### llama.cpp / llama-server

```bash
./llama-cli -m DeepSeek-V4-Flash-REAP-80G-imatrix-0731.gguf \
  -ngl 99 \
  -c 8192 \
  --chat-template deepseek-v4
```

### vLLM

For the tested vLLM preview lineage, patch `vllm_topk_softplus_sqrt` so expert counts outside the fused kernel's supported set call the Torch fallback. See the `runtime/` directory in the source repository for the patch script.

## Limitations

- Expert importance came from a closely aligned earlier checkpoint, not a fresh `0731` observation.
- Structural and smoke validation do not establish benchmark parity with the unpruned model.
- This is an aggressive 37.5% routed-expert prune. Coding preservation is plausible because coding and agentic data were included, but must be measured rather than assumed.
- The included three DSpark MTP blocks were pruned consistently; MTP runtime compatibility should be validated separately from backbone generation.

## Credits and License

Thanks to [DeepSeek](https://huggingface.co/deepseek-ai) for the base model, [Cerebras Research](https://github.com/CerebrasResearch/reap) for REAP, the [vLLM](https://github.com/vllm-project/vllm) contributors, and the [ds4](https://github.com/antirez/ds4) community for the GGUF toolchain.

This derivative follows the source repository's MIT license. See `LICENSE` for the full text.

## Citation

If you use this model in your research, please cite the original DeepSeek-V4-Flash work and the REAP pruning methodology.
