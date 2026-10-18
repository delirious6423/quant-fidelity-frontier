---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - mlx
  - moe
  - pruning
  - reap
  - deepseek
  - deepseek-v4
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-REAP37-native-MLX

**A 37%-expert-pruned MLX conversion of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), at native precision (no additional quantization beyond the checkpoint's own shipped weights).**

- **96 GB** on disk (down from ~155 GB / 304B params in the source release)
- **163/256 routed experts kept per layer** (36.7% pruned), via REAP (router-weighted expert activation pruning) on 40 learned-routing layers and REAM (activation merging) on the 3 hash-routed layers
- **Zero extra quantization damage** — every surviving expert is bit-identical to the source checkpoint's own native 4-bit weights. Size reduction comes entirely from removing experts, not from compressing the ones that remain
- Native **DSML tool-calling preserved** (DeepSeek's own function-calling format) — verified working end-to-end
- `deepseek_v4` architecture (HISA attention, hyper-connections, hash-routed first 3 layers), unchanged from upstream

## Why prune instead of quantize

Both operations lose you nothing for free, but they don't cost the same. A controlled PPL decomposition on a sibling model in this same pipeline (same architecture family, different base model) isolated quantization noise as responsible for the large majority of quality loss on a comparable BF16→4-bit step, with expert pruning at a moderate ratio contributing only a small fraction on top. This checkpoint acts on that finding directly: since DeepSeek-V4-Flash-0731 already ships natively at 4-bit, hitting a target file size by pruning experts — the cheaper operation — rather than quantizing further avoids stacking a second, more expensive form of damage on top.

## How it was built

1. **Saliency collection**: REAP scores (gate-weighted activation norms) computed per expert per layer by streaming the model over calibration prompts, using windowed layer streaming to avoid materializing the whole checkpoint at once.
2. **Pruning**: lowest-scoring 36.7% of experts removed per layer, on the 40 layers with learned top-k routing.
3. **Merging (hash-routed layers only)**: the first 3 layers route via a fixed token→expert hash table, not learned routing, so deleted experts have no natural replacement and every layer must land at the same width (a config-level constraint, not a preference). REAM assigns each pruned expert's hash-table entries to its most-similar surviving expert (router-weight cosine similarity) and merges the weights, rather than deleting outright.
4. **No further quantization.** The `--requantize` step available in this pipeline (for pushing surviving experts below their native bit-depth) was deliberately skipped for this build.

## Usage

MLX-native, tested with [oMLX](https://omlx.ai) — DeepSeek-V4-Flash architectures get native DSML chat-template and tool-parser injection there automatically (do not add a custom `chat_template.jinja` to the repo/model folder, it will suppress that injection).

Also loadable via `mlx_vlm`:

```python
from mlx_vlm import load, generate

model, processor = load("True2456/DeepSeek-V4-Flash-0731-REAP37-native-MLX")
output = generate(model, processor, "Explain hyper-connections in one sentence.", max_tokens=200)
```

Recommended sampling (per the upstream model card): `temperature=1.0`, `top_p=0.95` for agentic/tool-use scenarios, `top_p=1.0` otherwise.

## What this is not

- Not independently benchmarked against the unpruned source model on standard suites (MMLU/GSM8K/etc.) at time of upload — treat quality as provisional until you've checked it against your own use case.
- Not a quantization of the source model — every weight that survives pruning is untouched from upstream.
- Vision components: none present. This is DeepSeek-V4-Flash, which is text-only.

## License

MIT, inherited from the base model.
