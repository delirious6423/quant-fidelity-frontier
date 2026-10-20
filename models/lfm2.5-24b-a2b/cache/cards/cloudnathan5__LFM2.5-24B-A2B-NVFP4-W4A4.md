---
library_name: vllm
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
pipeline_tag: text-generation
tags:
- liquid
- lfm2.5
- moe
- nvfp4
- fp4
- compressed-tensors
- quantized
- vllm
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
---

# LFM2.5-24B-A2B-NVFP4-W4A4

NVFP4 (W4A4) quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

## Run with vLLM

Works on stock vLLM (≥ 0.31), no patches or `--trust-remote-code`. FP4 compute needs a Blackwell GPU.

```bash
pip install "vllm>=0.31"
```

```bash
vllm serve cloudnathan5/LFM2.5-24B-A2B-NVFP4-W4A4 \
  --enable-auto-tool-choice --tool-call-parser lfm2 \
  --reasoning-parser qwen3
```

## Run with SGLang

Works on stock SGLang (≥ 0.5.21), no patches.

```bash
pip install --pre "sglang>=0.5.21"
```

```bash
python -m sglang.launch_server --model-path cloudnathan5/LFM2.5-24B-A2B-NVFP4-W4A4 \
  --tool-call-parser lfm2 --reasoning-parser qwen3
```

On RTX PRO 6000 / RTX 50-series (SM120), the first launch JIT-compiles FlashInfer's FP4 MoE kernels
(~20 min, needs the CUDA toolkit); they are cached for later launches.

## Similar NVFP4 based quants

- [WhiskyAKM/LFM2.5-24B-A2B-NVFP4-GGUF](https://huggingface.co/WhiskyAKM/LFM2.5-24B-A2B-NVFP4-GGUF) (GGUF)
- [myroslavtryhubets/LFM2.5-24B-A2B-NVFP4A16](https://huggingface.co/myroslavtryhubets/LFM2.5-24B-A2B-NVFP4A16) (W4A16)
- [infinity-ai-institute/LFM2.5-24B-A2B-NVFP4](https://huggingface.co/infinity-ai-institute/LFM2.5-24B-A2B-NVFP4) (Experts-only W4A4)

## License

This model is licensed under the [LFM Open License v1.0](LICENSE), the same license as the base model.

**Notice of modification:** this is a modified version of LiquidAI/LFM2.5-24B-A2B. The linear layers were
quantized to NVFP4 and `config.json` carries a `quantization_config`. All other weights are unchanged.
The original model and its copyright belong to Liquid AI, Inc.
