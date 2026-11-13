---
tags:
- uzu
- mirai
- 4-bit
library_name: uzu
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
---

<div style="display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap">
  <div style="flex:1;min-width:260px">
    <h1 style="margin:0 0 12px">Mirai's Qwen3.8-27B Medium Quantization</h1>
    <p style="margin:0">A medium-sized, high-quality quantization of Qwen3.8-27B prepared for efficient local inference on Apple silicon.</p>
  </div>
  <img src="https://assets.trymirai.com/images/logo/ml_small_logo.svg" alt="Mirai Labs" width="64" height="64" style="width:80px;height:80px;object-fit:contain;flex:0 0 auto">
</div>

![KL divergence versus language-model checkpoint size](assets/kl-vs-size.svg)

Mirai-M is on the size-KL Pareto frontier: we found no checkpoint that is smaller while also having lower KL divergence.

*Evaluation data mixture: 45% public agentic, 30% public SFT/long-context, 25% private chat data.*

### Quickstart

If you are on macOS, the easiest way is to install the `mirai` Homebrew package and then run the CLI:

```sh
brew install mirai
mirai --model trymirai/Qwen3.8-27B-M
```

Currently only Apple silicon inference is supported. If you want to build things from source, read this [overview](https://github.com/trymirai/uzu/blob/how-to/docs/how-to-run-uzu.md).

<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:32px">
  <a href="https://trymirai.com/local-models/alibaba-qwen3-8-27b-mirai-mirai-m-4" style="display:inline-flex;align-items:center;justify-content:space-between;width:220px;min-height:53px;padding:0 20px;box-sizing:border-box;border:1px solid #cccccc;border-radius:0;background:#ffffff;color:#3d3d3d;text-decoration:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;font-size:15px;font-weight:500;line-height:1.5">
    <span>Learn more</span>
    <span aria-hidden="true" style="font-size:16px;line-height:1">&#8594;</span>
  </a>
</div>

---

### Method

Mirai Medium uses 4-bit asymmetric integer quantization with 4-bit zero points, bfloat16 scales, and group size 64. Block-diagonal Random Hadamard Transforms are used to reduce activation and weight outliers. The checkpoint was prepared with post-training quantization followed by quantization-aware distillation.

In quantization and model-size tier, this checkpoint is comparable to Unsloth's [`UD-Q4_K_M`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_M.gguf) GGUF.

### Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{mirai-quant,
    title  = {{Mirai Quantization}: Redefining the speed-quality frontier for local LLMs on Apple silicon},
    author = {Artur Chakhvadze and Ryan Mathieu and Roman Knyazhitskiy and Nikolai Voinilenko and Chen-Chen Yeh and Artur Mullakhmetov and Eugene Bokhan and others},
    note   = {In collaboration with others at Mirai Labs},
    month  = {June},
    year   = {2026},
    url    = {https://trymirai.com/blog/quantization}
}
```

### Original model

This is a quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). For architecture details, intended use, evaluations, and limitations, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md).
