---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- transformers
- fp8
- quantized
pipeline_tag: image-text-to-text
---

# gemma-4-26B-A4B-it-FP8

Fine-grained FP8 (block-quantized) version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
produced with `transformers`' `FineGrainedFP8Config`. Memory footprint drops from
~53GB (BF16) to ~27GB.

The vision tower and `lm_head` are kept in BF16 — see "Why these settings" below.

## Recipe

```python
from transformers import FineGrainedFP8Config, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
import torch

model_id = "google/gemma-4-26B-A4B-it"

quant_config = FineGrainedFP8Config(
    # 64x64, not the default 128x128: the language model's moe_intermediate_size
    # (704) and intermediate_size (2112) are not divisible by 128, only by 64
    # (704 = 64*11, 2112 = 64*33, hidden_size 2816 = 64*44). With block_size=128
    # the MoE grouped-matmul kernel hits `AssertionError: K (704) must be
    # divisible by block_k (128)` at the first forward pass.
    weight_block_size=(64, 64),
    # model.vision_tower's MLP intermediate_size is 4304 = 16*269 (269 is prime),
    # so it isn't divisible by 64 either -- excluded explicitly rather than
    # relying on the quantizer's shape-mismatch fallback (which silently leaves
    # a layer in BF16 if its shape doesn't tile, but still wraps it in the FP8
    # module class). lm_head is excluded too, the conventional choice for
    # numerical stability of the output projection.
    modules_to_not_convert=["model.vision_tower", "lm_head"],
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=quant_config,
    dtype="auto",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(model_id)
proc = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("gemma-4-26B-A4B-it-FP8")
tok.save_pretrained("gemma-4-26B-A4B-it-FP8")
proc.save_pretrained("gemma-4-26B-A4B-it-FP8")
```

## Why these settings

- **`weight_block_size=(64, 64)`**: the default `(128, 128)` block doesn't evenly
  tile the language model's own MoE/MLP dimensions (`moe_intermediate_size=704`,
  `intermediate_size=2112`), only `hidden_size=2816` is divisible by 128. 64 is
  the largest block size that divides all three.
- **`modules_to_not_convert=["model.vision_tower", "lm_head"]`**: the vision
  tower's MLP `intermediate_size` (4304) isn't divisible by 64 either, and is
  comparatively small next to the 26B-parameter MoE backbone, so quantizing it
  buys little memory at the risk of degrading image understanding. `lm_head` is
  excluded for the usual numerical-stability reasons.

  Note the module path must be the full dotted path as it appears in
  `model.named_modules()` (`"model.vision_tower"`, not just `"vision_tower"`) —
  `should_convert_module` only matches a pattern that is a prefix of, or
  suffix to, the full path.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "hugg1ngfac3/gemma-4-26B-A4B-it-FP8",
    dtype="auto",
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("hugg1ngfac3/gemma-4-26B-A4B-it-FP8")

msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=30)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

Requires `kernels>=0.12.0,<0.13` for the FP8 grouped-matmul kernel used by the
MoE experts (`pip install -U "kernels<0.13"`).
