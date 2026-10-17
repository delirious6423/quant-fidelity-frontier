---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- beam
- experimental
- qwen4_exp
- mixed-precision
- mixture-of-experts
- apple-silicon
---

# Qwen3.8-Flash-Next-MLX-BEAM-4bit

> **Experimental — work in progress.** Text generation works (smoke-tested, see below). Images do not work yet: the vision tower is shipped, but no runtime uses it from here. Quality was not measured by us - the weights are bit-identical to the source build, whose card carries the numbers.

MLX build of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE with a 51B hashed n-gram embedding) for Apple Silicon. The weights are those of [pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit) - 4-bit routed experts, 8-bit everything else - repackaged in the B.E.A.M. layout: the bf16 vision tower moved out of the first shard into `extras/`, a hardened chat template, and a manifest that records the bit allocation. 106 GB on disk. No multi-token-prediction head (the source ships none).

The architecture (`qwen4_exp`) is not in a released mlx-lm; the package ships `qwen4_exp.py` (the source build's runtime file, unchanged) and declares it via `model_file`, hence `--trust-remote-code`.

## What is in the package

This is a B.E.A.M. package: a standard MLX checkpoint plus everything the base model ships beyond the language model, kept as separate, index-listed parts under `extras/`. Any loader that follows `model.safetensors.index.json` finds them; nothing here depends on a particular runtime. (B.E.A.M. is the engine this layout is built for; it is not published yet — the package works without it.)

| Part | File | Note |
|---|---|---|
| Chat template | `chat_template.jinja` | 10 KiB — hardened chat template (Unsloth pattern): leading system/developer messages merged into one system block, reasoning_effort 'high' mapped to 'xhigh', no exception for a conversation without a user turn |
| Vision tower (bf16) | `extras/vision.safetensors` | 333 tensors, 856 MiB — split out of model-00001.safetensors; patch_embed.proj.weight transposed from the PyTorch conv layout to MLX (out, T, H, W, in) |
| Allocation | `extras/allocation.json` | bits per module, schema v1 |
| Manifest | `extras/manifest.json` | sizes and SHA-256 of every part |

`extras/manifest.json` lists every part with its size and SHA-256, and `extras/allocation.json` records which module got how many bits.

## Quantization

Affine MLX quantization, group size 64 unless noted - taken over unchanged from the source build. This package repackages, it does not re-quantize; nothing about the allocation was measured by us.

| Group | Bits |
|---|---|
| routed experts (`switch_mlp`, 96.6 % of the parameters) | 4 |
| hashed n-gram embedding tables (128 shards) | 4, group size 32 (row width 160 is not a multiple of 64) |
| attention, DeltaNet, hyper-connection mixers, shared experts, PLE projections, `embed_tokens`, `lm_head` | 8 |
| MoE routers, gates, DeltaNet `in_proj_a/b`, indexer projection | bf16 |
| vision tower | bf16 |

`extras/allocation.json` lists every quantized module with its bits (module names as in the `quantization` map of `config.json`; a quantized module not listed there takes the default of 4 bit). The source card explains why this split was chosen and gives paired per-window NLL results against bf16 and against the uniform 4-bit, 6-bit and 8-bit builds - they apply here unchanged.

## Use

Text, with [mlx-lm](https://github.com/ml-explore/mlx-lm):

```bash
pip install -U mlx-lm
mlx_lm.generate --model p4ik/Qwen3.8-Flash-Next-MLX-BEAM-4bit --trust-remote-code --prompt "Explain the difference between a list and a tuple in Python." --max-tokens 200
```

```python
from mlx_lm import load, generate

model, tokenizer = load("p4ik/Qwen3.8-Flash-Next-MLX-BEAM-4bit", trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain the difference between a list and a tuple in Python."}],
    add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

Images: the vision tower is in the package (`extras/vision.safetensors`, listed in the index), but no runtime uses it from here yet. mlx-lm is text-only. mlx-vlm would run the tower with its own `qwen4_exp` implementation, but a `model_file` in `config.json` takes precedence there, and `qwen4_exp.py` is an mlx-lm text model (no `ModelConfig`, no tower) - so mlx-vlm does not load this package at all. Text via mlx-lm only, for now.

## What has been checked

Mac mini M4 Pro, 64 GB unified memory, 2026-09-14, mlx 0.32.2 / mlx-lm 0.31.3:

- **Layout** - every part matches `extras/manifest.json` (bytes, SHA-256, tensor count); the index is complete and names all 333 vision tensors in `extras/vision.safetensors`; the 432 routed-expert tensors are byte-identical to the source's uniform 4-bit build (the mixed build differs from it only in the 8-bit groups).
- **Text** - the package is larger than that machine's memory, so it ran under our own expert-streaming loader (routed experts and n-gram tables read from SSD on demand, 5.4 GB resident): greedy output on two short prompts and a 3.7k-token prompt is coherent; 3.9 tokens/s decode and about 120 tokens/s prompt processing there. A smoke test, not a benchmark, and not a stock-mlx-lm run.
- **Not checked** - loading with stock mlx-lm on a machine with enough memory (128 GB or more); the hardened chat template with tool calls on this model; the vision tower (no runtime, see above); quality (bit-identical weights, see the source card).

## License

[Qwen Community License 1.0](LICENSE), as the base model. `qwen4_exp.py` is the runtime file of the source build (port code: https://github.com/PipeNetwork/qwen38-flash-next-mlx).
