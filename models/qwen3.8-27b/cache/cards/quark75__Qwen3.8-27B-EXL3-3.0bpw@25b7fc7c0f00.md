---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- exl3
- quantized
- qtip
- vision-language
---

# Qwen3.8-27B — EXL3 3.0bpw (mul1)

This is an [EXL3](https://github.com/turboderp-org/exllamav3) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense hybrid-GDN vision-language model. EXL3 uses QTIP-style trellis quantization, which compresses far more aggressively than round-to-nearest GPTQ/AWQ-style methods while holding up better at low bitrates.

- **Codebook**: `mul1`
- **Bits per weight**: 3.0 (linear layers), 6 bits (`lm_head` and vision tower), 4 bits (MTP head)
- **Calibration**: 256 rows × 2048 cols
- **On-disk size**: ~13 GB (down from ~54 GB BF16)
- **Base architecture**: `Qwen3_5MoeForConditionalGeneration` (hybrid linear/full attention, native vision-language)

## Quality validation

Quantization quality was validated against the original BF16 weights via KL-divergence and perplexity on a held-out evaluation set, not just calibration-set loss:

| Metric | Value |
|---|---|
| KL divergence (quantized vs. BF16) | **0.0487** |
| Top-1 token agreement | **91%** |
| Perplexity, BF16 original | 7.002 |
| Perplexity, EXL3 3.0bpw | 6.985 |

The quantized model's perplexity essentially matches the original — the low KLD and high top-1 agreement indicate the 3.0bpw trellis quantization preserves the model's output distribution closely, not just its loss on average.

## Usage

Requires [exllamav3](https://github.com/turboderp-org/exllamav3) (or a frontend built on it, e.g. TabbyAPI):

```bash
pip install exllamav3
```

```python
from exllamav3 import Config, Model, Cache, Tokenizer, Generator

config = Config.from_directory("path/to/this/repo")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192)
model.load()
tokenizer = Tokenizer.from_config(config)

generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
output = generator.generate(prompt="Explain the significance of the Riemann hypothesis.", max_new_tokens=512)
print(output)
```

## Original model card

For architecture details, benchmark numbers, and usage guidance for the base model, see [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## License

Apache 2.0, inherited from the base model.