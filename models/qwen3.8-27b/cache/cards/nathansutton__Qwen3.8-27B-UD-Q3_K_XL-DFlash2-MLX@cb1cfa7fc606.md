---
library_name: mlx
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- apple-silicon
- coding
- unsloth-style-dynamic-quant
- speculative-decoding
pipeline_tag: text-generation
---

# Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX

An [MLX](https://github.com/ml-explore/mlx) quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for running locally on Apple Silicon,
**with its speculative decoder bundled in** — the default model for
[chad](https://github.com/nathansutton/chad), a Claude-Code-style local coding agent.

One download gets both halves: the quantized target model and the block drafter it decodes
with, already quantized. Nothing is built on first run.

## Quant

- **Params:** 27B (dense qwen3_5 hybrid: 64 layers, 48 GatedDeltaNet + 16 full attention)
- **Scheme:** 3-bit group-64 body (MLP gate/up/down + attention) and 3-bit embeddings, with lm_head held at 5-bit
- **Footprint:** ~12.1 GB weights + a ~1.1 GB bundled drafter (13.3 GB on disk); 262k native context
- **RAM:** Built for 24 GB Apple Silicon. chad sizes its context window from the live Metal budget, so expect tens of thousands of tokens rather than the full 262k on a 24 GB box.

The naming follows [Unsloth's Dynamic 2.0](https://docs.unsloth.ai/) convention (`UD-Q3_K_XL` = a dynamic quant that spends extra bits on the layers that matter) so the *scheme* is recognizable at a glance. **It is not literally a llama.cpp k-quant** — this is an MLX group-64 affine quant produced by our own per-module predicate (the bulk low-bit, sensitive layers high), with the bit allocation chosen by measured held-out perplexity per gigabyte across candidate recipes.

## The bundled DFlash2 drafter (`dflash/`)

`dflash/` holds a **DFlash2 block drafter** for this target — a 1.9B model that reads the
target's residual stream at five tapped layers (5, 19, 33, 47, 61) and proposes a whole
block of tokens in **one** forward — quantized to 4-bit group-64 (codebooks at 8-bit) in
MLX format. chad verifies the block in one batched target forward and accepts by exact
rejection sampling: every emitted token is the target's own choice for its position, and
sampled output keeps the model's true distribution at any temperature. Greedy output
matches unspeculated decoding *to kernel rounding* — the batched verify and the serial
step run different matmul/attention kernels, so a greedy run follows serial until the
first near-tie and can branch there (measured: 4/10 160-token generations bit-identical,
the rest diverging 10–95 tokens in). It is a pure speed feature.

Measured with chad 2.0.0 on an M4 Pro (24 GB), 384-token decodes, medians over ten prompts,
one load per run:

| sampling | serial | with the drafter |
|---|---|---|
| greedy, prose/code seeds | 17.5 tok/s | **60.1** |
| thinking preset (temp 1.0, top_p 0.95, top_k 20), same seeds | 22.4 | **49–51** |
| greedy, real mid-session agent contexts (12–19k tokens) — median / floor | 14.8 / 13.2 | **31.7 / 21.4** |
| thinking preset, same real contexts — median / floor | 13.9 / 13.1 | **27.6 / 17.7** |

The seed rows are the ceiling (the drafter lands ~8 tokens per round on text the model
knows well); the real-context rows are what an agent session lives at, since most of a
turn is `<think>`, where 35–55% of proposals land. For scale, stock llama.cpp on Unsloth's
`Qwen3.8-27B-UD-Q3_K_XL` GGUF decodes at 10.9 tok/s on the same machine (`llama-bench`
tg128), and chad serial at 18.1 — method and rows in chad's
[`benchmarks/stock/`](https://github.com/nathansutton/chad/tree/main/benchmarks/stock).

It is deliberately in a subdirectory: `mlx-lm` globs `model*.safetensors` at the repo root,
so `dflash/model.safetensors` is invisible to the base loader and this repo loads exactly
like a plain quantized checkpoint everywhere else.

The drafter weights are converted from the DFlash2 release for this model
([`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), also
mirrored as `incoai/Qwen3.8-27B-DFlash2`); the method is Chen et al., *DFlash: Block
Diffusion for Flash Speculative Decoding* (arXiv:2602.06036). Only the quantization and the
MLX layout are ours — please honor the original release's license and citation terms.

## Chat template

`chat_template.jinja` differs from upstream in exactly one line: the default
`reasoning_effort` is **`medium`**, where upstream defaults to `xhigh`. Pass
`reasoning_effort` explicitly to override it.

## Use it

With chad (auto-downloads this model on first run):

```bash
uvx chad-code
```

Or directly with `mlx-lm` — the drafter is ignored, decoding is serial:

```python
from mlx_lm import load, generate
model, tok = load("nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX")
print(generate(model, tok, "Write a haiku about quantization.", max_tokens=64))
```

## License

The target weights inherit the license of the base model
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B); the contents of `dflash/`
inherit the license of the DFlash2 release they are converted from. Review both before use
or redistribution.
