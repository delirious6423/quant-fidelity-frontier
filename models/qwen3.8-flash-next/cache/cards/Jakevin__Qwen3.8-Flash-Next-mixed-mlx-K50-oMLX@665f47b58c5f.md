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
- moe
- apple-silicon
---

# Qwen3.8-Flash-Next-mixed-mlx-K50-oMLX (mixed-precision, expert-pruned)

A mixed-precision, expert-pruned MLX build of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), stored in the standard
`qwen4_exp` safetensors layout so that **oMLX can load it with its own bundled model definition** — no custom runtime.

This is the oMLX-loadable sibling of
[Jakevin/Qwen3.8-Flash-Next-mixed-mlx-K50](https://huggingface.co/Jakevin/Qwen3.8-Flash-Next-mixed-mlx-K50).

## What it is

- Source model: Qwen3.8-Flash-Next.
- Expert pruning: 50% of routed experts kept per layer, chosen by REAP (router-weighted expert activation pruning).
  Every one of the 48 layers keeps **256 of 512** routed experts, including layer 0 (also pruned to 256 so that
  all layers share one expert count, which the stock definition requires).
- Quantization: mixed precision. Within each layer, all routed experts use one bit width, either 2, 3 or 4 bit
  (MLX affine, group size 64): 15 layers at 2 bit, 17 at 3 bit, 16 at 4 bit. Embedding, lm_head, routers and
  other non-expert weights are 8 bit; norms are bf16. The PLE (per-layer n-gram embedding) table is affine mixed 2–6 bit.
- **No MTP (multi-token prediction) weights are included.** Run with MTP disabled.
- Size: 31.0 GB of model weights plus 32.3 GB of PLE shards (63.3 GB total on disk).

## Compatibility

Tested with oMLX 0.7.0 (oMLX supports `qwen4_exp` from 0.6.3). Loaded with oMLX's bundled Python and its
unmodified `qwen4_exp` definition, `strict=True`. Use `OMLX_QWEN4_PLE_MODE=mmap` so the PLE table is read
from disk instead of being held in memory.

## Usage (tested)

The following is what was run for the measurements below, using the Python that ships inside the oMLX app:

```python
# run with: /Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3 run.py
import os, sys
os.environ["OMLX_QWEN4_PLE_MODE"] = "mmap"
RES = "/Applications/oMLX.app/Contents/Resources"
sys.path += [RES + "/Python/framework-mlx-base/lib/python3.11/site-packages", RES]

import mlx.core as mx
mx.set_cache_limit(2 * 1024**3)
from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
MODEL = "path/to/Qwen3.8-Flash-Next-mixed-mlx-K50-oMLX"
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

Loading through the oMLX app's server UI has not been tested yet.

## Smoke test (measured)

Machine: Mac mini, Apple M4 Pro, 64 GB. Single process, MLX cache limit 2 GiB, `OMLX_QWEN4_PLE_MODE=mmap`,
greedy decoding, 64 generated tokens, one Python coding prompt and one English prose prompt, run back to back
in one process right after a lazy load (first prompt includes reading the weights from disk).

| Prompt | Prompt tokens | Time to first token | Decode tok/s |
|---|---:|---:|---:|
| Coding (first after load) | 39 | 38.4 s | 0.77 |
| English prose | 36 | 12.6 s | 0.92 |

- MLX peak memory 31.9 GB; process RSS peak 19.5 GB; swap growth 0.35 GB.
- Output on both prompts was coherent English (coding prompt: a structured answer on a doubly-linked list plus
  hash map LRU cache; prose prompt: a paragraph on radio, radar and GPS replacing lighthouses).
- These are single runs on one machine, not a tuned benchmark.

## Quality

Quality evaluation is in progress.

## Limitations

- Apple Silicon and MLX/oMLX only.
- No MTP weights; speculative decoding with MTP is not available.
- Pruning removes half of the routed experts without retraining; expect some quality loss versus the original model.
- Speed numbers are from one machine and two prompts.

## License

Weights are under the Qwen Community License 1.0. See `LICENSE` (the original license file, unchanged) and `NOTICE`.

This model is a derivative work of Qwen/Qwen3.8-Flash-Next. Copyright (c) 2026 Qwen. As required by the license, the copyright notice and permission notice are included in `LICENSE`. If this model or a derivative is used in a commercial product or service with more than 100,000,000 monthly active users or US$20,000,000 monthly revenue, the model name must be prominently displayed in that product's user interface. Model-as-a-Service or AI Work Assistant businesses need a separate license from Qwen for commercial use; see `LICENSE` for the full terms.
