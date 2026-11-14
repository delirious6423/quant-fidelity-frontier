---
tags:
- uzu
- mirai
- 2-bit
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
    <h1 style="margin:0 0 12px">Mirai's Qwen3.8-27B Small Quantization</h1>
    <p style="margin:0">A small, high-quality quantization of Qwen3.8-27B prepared for efficient local inference on Apple silicon.</p>
  </div>
  <img src="https://assets.trymirai.com/images/logo/ml_small_logo.svg" alt="Mirai Labs" width="64" height="64" style="width:80px;height:80px;object-fit:contain;flex:0 0 auto">
</div>

![KL divergence versus language-model checkpoint size](assets/kl-vs-size.svg)

### Quickstart

If you are on macOS, the easiest way is to install the `mirai` Homebrew package and then run the CLI:

```sh
brew install mirai
mirai --model trymirai/Qwen3.8-27B-S
```

Currently only Apple silicon inference is supported. If you want to build things from source, read this [overview](https://github.com/trymirai/uzu/blob/how-to/docs/how-to-run-uzu.md).

---

### Method

Mirai Small uses QTIP-style trellis-coded quantization with the bits of each matrix chosen from measured KL. Kronecker-structured random rotations are used to reduce weight outliers. The input embedding uses a 2-bit D4 lattice code, and the output head uses 4-bit integers with group size 64 and block-32 Hadamard transforms. The checkpoint was prepared with post-training quantization followed by quantization-aware distillation, including a round on the model's own agent turns.

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
