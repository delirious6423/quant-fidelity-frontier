---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- omlx
- mixed-precision
- quantized
- pruned
- 3-bit
- moe
- apple-silicon
---

# Qwen3.8-Flash-Next-mixed-mlx-K50-min3bit-oMLX (expert-pruned, 3/4-bit experts)

A requantized version of
[Jakevin/Qwen3.8-Flash-Next-mixed-mlx-K50-oMLX](https://huggingface.co/Jakevin/Qwen3.8-Flash-Next-mixed-mlx-K50-oMLX):
same expert pruning (256 of 512 routed experts kept in each of the 48 layers) and the same `qwen4_exp` file layout,
but no layer uses 2-bit experts any more.

## What changed vs. the K50-oMLX build

- All routed experts were requantized from the original FP8 weights with RTN (round-to-nearest),
  MLX affine, group size 64, bf16 scales and biases.
- Expert bit widths: **32 layers at 3 bit, 16 layers at 4 bit** (per-layer widths are listed in `config.json`).
- Non-expert weights, tokenizer, chat template and the PLE shards are unchanged from the K50-oMLX build.
- No MTP weights are included. Run with MTP disabled.
- `SHA256SUMS` lists the SHA-256 of every `.safetensors` file.

## Measured results

Expert weight relative error:

| Width | This build | K50-oMLX build |
|---|---:|---:|
| 3 bit | 0.195 | 0.467 |
| 4 bit | 0.094 | 0.321 |

Partial output agreement check: of a planned 32-prompt set, only the 15 coding prompts were run.
Top-1 next-token agreement was **0.892** (K50-oMLX build on the same 15 prompts: 0.856).
The reference is the unpruned GSQ-RCO Q2 build, **not** the FP8 model. The prose prompts were not run.

Speed on a Mac mini (Apple M4 Pro, 64 GB): 4–5 tok/s decode, peak memory about 35.4 GB.
Machines with 16 GB have not been tested.

## Loading with oMLX

Load with the Python that ships inside the oMLX app:

```python
# run with: /Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 run.py
import os, sys
os.environ["OMLX_QWEN4_PLE_MODE"] = "mmap"
RES = "/Applications/oMLX.app/Contents/Resources"
sys.path += [RES + "/Python/framework-mlx-base/lib/python3.11/site-packages", RES]

import mlx.core as mx
mx.set_cache_limit(2 * 1024**3)
from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
MODEL = "path/to/Qwen3.8-Flash-Next-mixed-mlx-K50-min3bit-oMLX"
configure_qwen4_exp_runtime(MODEL, mode="mmap", mtp_enabled=False)
from mlx_vlm.utils import load_model
from tokenizers import Tokenizer

model = load_model(MODEL, lazy=True, strict=True)
lm = model.language_model
tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
prompt = "<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(prompt, add_special_tokens=False).ids

cache = lm.make_cache()
nxt = mx.argmax(lm(mx.array([ids]), cache=cache).logits[:, -1], -1)
out = [nxt.item()]
for _ in range(63):  # greedy
    nxt = mx.argmax(lm(mx.array([[out[-1]]]), cache=cache).logits[:, -1], -1)
    out.append(nxt.item())
print(tok.decode(out))
```

## Known issues and missing tests

- On the "strawberry" letter-counting prompt the spelling is correct, but in thinking mode the model did not
  finish its reasoning within 1536 tokens.
- A 6-prompt generation smoke test and a clean-environment install test have not been done yet.
- Apple Silicon and MLX/oMLX only. Pruning removes half of the routed experts without retraining.

## License

Weights are under the Qwen Community License 1.0. See `LICENSE` (the original license file, unchanged) and `NOTICE`.

This model is a derivative work of Qwen/Qwen3.8-Flash-Next. Copyright (c) 2026 Qwen. As required by the license, the copyright notice and permission notice are included in `LICENSE`. If this model or a derivative is used in a commercial product or service with more than 100,000,000 monthly active users or US$20,000,000 monthly revenue, the model name must be prominently displayed in that product's user interface. Model-as-a-Service or AI Work Assistant businesses need a separate license from Qwen for commercial use; see `LICENSE` for the full terms.
