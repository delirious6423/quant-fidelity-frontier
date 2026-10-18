---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
library_name: mlx
tags:
- mlx
- moe
- deepseek
- deepseek-v4
---

# DeepSeek-V4-Flash-MLX-8bit

MLX (Apple Silicon) conversion of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
quantized to **8-bit** (affine group quant, group size 64).

**Code / loader:** [github.com/PipeNetwork/deepseek-v4-mlx](https://github.com/PipeNetwork/deepseek-v4-mlx)

DeepSeek-V4-Flash is a **304B-parameter** sparse-MoE model. Its `deepseek_v4`
architecture is **not** implemented in any released runtime — not `transformers`,
and not `mlx-lm` (which carries `deepseek`, `_v2`, `_v3`, `_v32`). This repo bundles
a from-scratch MLX port under `deepseek_v4_mlx/`, validated against DeepSeek's own
reference implementation.

## Builds

| Variant | Size | wikitext-2 ppl | Notes |
|---|---:|---:|---|
| [8bit](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-8bit) | ~304 GB | 5.9878 | highest fidelity |
| [6bit](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-6bit) | ~233 GB | 6.0260 | +0.6% vs 8-bit for 70 GB less |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-mixed-4_8bit) | ~165 GB | 6.1262 | **recommended** — 4-bit experts, 8-bit everything else |
| [4bit](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-4bit) | ~162 GB | 6.3005 | superseded by `mixed-4_8bit` (3 GB more, 2.8% better) |
| [REAP25](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-REAP25) | ~129 GB | 6.4025 | 25% of experts pruned — **matches 4-bit quality 33 GB smaller** |
| [REAP37](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-REAP37) | ~111 GB | 6.9634 | 37% pruned; fits a 128 GB Mac |
| [REAP50](https://huggingface.co/pipenetwork/DeepSeek-V4-Flash-MLX-REAP50) | ~93 GB | 7.7520 | 50% pruned; smallest build |

Perplexity is teacher-forced over **wikitext-2-raw-v1 test**, 199,485 tokens in 195
windows of 1024 — identical windows for every build. Differences are established by
a **paired** bootstrap over shared windows, which removes window-difficulty variance:
independent intervals overlap almost entirely and settle nothing, while the paired
interval is ~10x tighter. Every gap quoted here is significant at 95%.

### Why `mixed-4_8bit` is the recommended build

The routed experts are **98.1%** of all quantized parameters, and they ship natively
as FP4. Keeping them at 4 bits while raising everything else — attention,
embeddings, shared experts, about 5B parameters — to 8 bits costs **3 GB** and
recovers **56%** of the gap between 4-bit and 8-bit. Per parameter, the non-expert
weights are roughly 50x more quantization-sensitive.

Two variants were built, measured, and **not** published:

* **group-size 32** (matching FP4's native 32-value scale blocks) is 0.5% *worse*
  than group 64 and 18 GB larger. Finer scaling improves raw tensor reconstruction
  but not prediction — the mismatch is that FP4's levels are non-uniform while
  affine's are evenly spaced, which no group size fixes.
* **3-bit experts** collapse the model entirely (perplexity 1.4e7, degenerate token
  loops). 4-bit sits just above a cliff for this checkpoint, not in the middle of a
  gentle curve.

### REAP-pruned builds

[REAP (Cerebras, arXiv:2510.13999)](https://arxiv.org/abs/2510.13999) drops the
lowest-saliency routed experts, where saliency is the mean of
`router_weight x ||expert_output||` over the tokens routed to that expert — its
actual contribution, not how often it fires. The router renormalizes over survivors.

**Only 40 of the 43 layers are pruned.** The first three route through a fixed
`tid2eid` token-id table with no router to renormalize, so dropping an expert there
would leave table entries pointing at nothing. They keep all 256, and the config
records two expert counts which the loader resolves per layer.

Saliency was profiled over **150k tokens** of English prose, ten other languages and
real code — deliberately disjoint from the evaluation set, verified by 32-gram
overlap (0 of 4687), so pruning decisions are not fit to the text they are judged on.

| Build | Experts kept | Saliency retained | ppl | vs unpruned |
|---|---:|---:|---:|---:|
| `mixed-4_8bit` | 256 | 100% | 6.1262 | — |
| `REAP25` | 192 | 96.2% | 6.4025 | +4.5% |
| `REAP37` | 161 | 91.5% | 6.9634 | +13.7% |
| `REAP50` | 128 | 83.9% | 7.7520 | +26.5% |

**`REAP25` is the one to take** — its perplexity is statistically indistinguishable
from plain 4-bit (paired ratio 1.016, CI [0.998, 1.038], crossing 1.0) at 33 GB less.

Note how poorly saliency retention predicts damage: 96.2% retained still costs 4.5%,
and the last 12 points of retention cost 22 points of perplexity. Treat it as a
screen for whether pruning is viable, not as an estimate of what it will cost.

## ⚠️ Loading requires the bundled loader

```bash
pip install mlx mlx-lm transformers
```
```python
from deepseek_v4_mlx.load import load
from deepseek_v4_mlx.generate import greedy_generate, load_tokenizer

model, args = load("/path/to/this/repo")
tok = load_tokenizer("/path/to/this/repo")   # bundled — no download
ids = tok("The capital of France is")["input_ids"]
print(tok.decode(greedy_generate(model, args, ids, max_new_tokens=32)))
```

**The tokenizer is bundled in this repo**, so nothing is fetched at runtime. (The base repo
shipped no tokenizer at first — only `encoding/encoding_dsv4.py`, a prompt-*string* renderer;
DeepSeek has since published one. DeepSeek-V3's tokenizer, used here in the interim, produces
**identical ids** for text and identical special-token ids, so builds made either way agree.)

## What this port implements

| Mechanism | Notes |
|---|---|
| **Hyper-Connections** | the residual stream carries **4 parallel copies**, mixed by a Sinkhorn-normalized combination matrix |
| **Hash-routed experts** | the first 3 MoE layers pick experts from a `tid2eid[vocab, 6]` table keyed on token id — routing independent of context |
| **Learned KV compression** | gated pooling over spans, alternating ratio 4/128, overlapping windows at ratio 4 |
| **Sparse-attention indexer** | scores compressed spans, keeps the top 512 for attention to read |
| **MLA + grouped output LoRA** | single shared KV vector; `o_groups: 8` block-diagonal output projection |
| **sqrt-softplus routing**, clamped SwiGLU, attention sinks | |

The source checkpoint is **doubly pre-quantized** — FP8 `e4m3` with `e8m0` 128x128
block scales for most matmuls, FP4 `e2m1` packed two-per-byte for routed experts — so
conversion dequantizes both formats before requantizing. Both decode paths are
bit-exact against an independent implementation.

Dropped: the multi-token-prediction head and **DSpark** (speculative decode over
5-token blocks in the last three layers). Both hang off the reference's `forward_spec`
path and are unused for ordinary generation.

`wo_a` is kept unquantized: it is a block-diagonal projection the model reshapes by
group, and a packed quantized weight cannot be reshaped that way. Costs ~2% of build size.

License: MIT (inherits the base model).
