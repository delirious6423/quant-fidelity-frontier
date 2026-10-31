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

## Extended evaluation (2026-10-31)

All numbers below were measured on 2026-10-31 (10:47–11:10 UTC+8) on this exact build with oMLX's bundled MLX. Nothing here is extrapolated.

### Top-1 agreement vs. reference, 32 prompts

Per-prompt top-1 next-token agreement (128 positions per prompt). 95% CI = percentile bootstrap over prompts (B=10000, seed 0).

| Subset | n | Mean | 95% CI |
|---|---|---|---|
| All | 32 | 0.840 | [0.818, 0.863] |
| Code | 16 | 0.890 | [0.871, 0.907] |
| English prose | 16 | 0.791 | [0.769, 0.813] |

The earlier 15-prompt figure (0.892) was code-only; prose agreement is clearly lower. The "A" comparison build was only measured on the first 15 prompts (0.856), so no 32-prompt comparison exists.

### Generation tests (one run each, sampling seed 1 unless noted)

| Prompt | Expected | Non-thinking | Thinking |
|---|---|---|---|
| Python `is_prime` + 3 asserts | runnable code | Correct (code runs, asserts pass) | Wrong: thinking closed, but final code uses C-style `for i = 3; ...` (SyntaxError) |
| Rectangle, perimeter 36, length = 2x width | 72 | Wrong (288, used perimeter = L + W) | Correct (72) |
| Highest mountain in Taiwan | Yushan, ~3952 m | Partly correct: Yushan, but headline "about 3,956 m" plus wrong romanization and ranking claims | Correct (3952 m) |
| Exactly 4 English sentences | 4 sentences | Correct | Correct |
| One-sentence Traditional Chinese summary | 1 sentence | Correct | Correct (reasoning in Simplified Chinese, answer in Traditional) |
| Count "e" in "bookkeeper" | 3 | Wrong (2) | Did not converge: still spelling the word when it hit max_tokens=2048 |

Thinking mode closed its reasoning in 5 of 6 prompts. Score: non-thinking 3/6 correct + 1 partial; thinking 4/6.

### Strawberry ("how many r in Strawberry", non-thinking)

| Decoding | Final answer |
|---|---|
| Greedy | 3 (first says 2, then re-spells and corrects itself to 3) |
| Sample seed 1 | 2 (wrong) |
| Sample seed 2 | 2 (wrong) |
| Sample seed 3 | 2 (wrong) |

### Speed and memory (measured on our Mac mini, batch 1)

- Decode: 4.8–8.6 tok/s (median 8.0) on short generations; time to first token 33–37 s (as measured by the test script; not broken down further).
- MLX peak memory: 34.5–34.8 GB; process physical footprint peak up to 36.5 GB.

### Known issues from this run

- Letter counting is unreliable: "bookkeeper" wrong in non-thinking mode and unfinished in thinking mode; "Strawberry" right only with greedy decoding (1 of 4 runs).
- Thinking mode can fail to converge (bookkeeper hit the 2048-token limit).
- Non-thinking mode made a basic arithmetic setup error on the rectangle problem.
- Two runs (code non-thinking, Strawberry seed 3) were marked killed by our memory watchdog because `footprint` could not be read. In both cases the generation had already finished (`finish=stop`, full output saved) before the kill, so their outputs are counted above. They were not re-run.

## Known issues and missing tests

- On the "strawberry" letter-counting prompt the spelling is correct, but in thinking mode the model did not
  finish its reasoning within 1536 tokens.
- A 6-prompt generation test was done on 2026-10-31 (see above). A clean-environment install test has not been done yet.
- Apple Silicon and MLX/oMLX only. Pruning removes half of the routed experts without retraining.

## License

Weights are under the Qwen Community License 1.0. See `LICENSE` (the original license file, unchanged) and `NOTICE`.

This model is a derivative work of Qwen/Qwen3.8-Flash-Next. Copyright (c) 2026 Qwen. As required by the license, the copyright notice and permission notice are included in `LICENSE`. If this model or a derivative is used in a commercial product or service with more than 100,000,000 monthly active users or US$20,000,000 monthly revenue, the model name must be prominently displayed in that product's user interface. Model-as-a-Service or AI Work Assistant businesses need a separate license from Qwen for commercial use; see `LICENSE` for the full terms.
