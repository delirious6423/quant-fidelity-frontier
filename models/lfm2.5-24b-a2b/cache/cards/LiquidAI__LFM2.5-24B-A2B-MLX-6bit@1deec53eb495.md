---
library_name: mlx
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
- mlx
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

# LiquidAI/LFM2.5-24B-A2B-MLX-6bit

This model [LiquidAI/LFM2.5-24B-A2B-MLX-6bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-6bit) was converted to MLX format from [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) using mlx-lm version **0.32.0**.

## About this quantization

Quantized to 6 bits per weight (group size 64, affine mode). MoE routing gates kept unquantized in float32.

For base model details, capabilities, and recommended generation settings, see the [original model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

## Use with mlx

Requires mlx-lm 0.32.0 or later. Earlier versions load the model but route the experts incorrectly.

```bash
pip install -U "mlx-lm>=0.32.0"
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("LiquidAI/LFM2.5-24B-A2B-MLX-6bit")

prompt = "What is C. elegans?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

sampler = make_sampler(temp=0.1, top_k=50)
logits_processors = make_logits_processors(repetition_penalty=1.05)

response = generate(
    model, tokenizer, prompt=prompt, max_tokens=4096,
    sampler=sampler, logits_processors=logits_processors,
    verbose=True,
)
```

### Chat Template

LFM2.5 uses a ChatML-like format. See the [Chat Template documentation](https://docs.liquid.ai/lfm/key-concepts/chat-template) for details. Example:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
<think>
```

Because LFM2.5-24B-A2B is a reasoning model, assistant turns contain an explicit chain of thought before the final answer. `tokenizer.apply_chat_template()` formats messages automatically.

### Tool Use

LFM2.5 supports function calling. Pass tools to `tokenizer.apply_chat_template(..., tools=...)`. See the [Tool Use documentation](https://docs.liquid.ai/lfm/key-concepts/tool-use) for the full guide.

## License

LFM Open License v1.0 — see the [LICENSE](LICENSE) file or the [original model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for terms.

## Citation

```bibtex
@article{liquidAI202624BA2B,
  author  = {Liquid AI},
  title   = {LFM2.5-24B-A2B: Built for Fast On-Device Agents},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm2-5-24b-a2b},
}
```

```bibtex
@article{liquidai2025lfm2,
  title   = {LFM2 Technical Report},
  author  = {Liquid AI},
  journal = {arXiv preprint arXiv:2511.23404},
  year    = {2025}
}
```
