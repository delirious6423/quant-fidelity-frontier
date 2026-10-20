---
library_name: transformers
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
pipeline_tag: text-generation
tags:
- liquid
- lfm2.5
- edge
- moe
- fp8
base_model: LiquidAI/LFM2.5-24B-A2B
---

<div align="center">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png"
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> •
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-24B-A2B-FP8

FP8 version of [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for GPU inference.

Linear weights are quantized to FP8 with 128×128 blocks and dynamic activation quantization. The embedding, MoE routing gates and convolution weights are kept in their original precision.

Find more details in the original model card: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B

## Use with SGLang

```bash
python -m sglang.launch_server --model-path LiquidAI/LFM2.5-24B-A2B-FP8
```

## License

This model is released under the [LFM 1.0 License](LICENSE).
