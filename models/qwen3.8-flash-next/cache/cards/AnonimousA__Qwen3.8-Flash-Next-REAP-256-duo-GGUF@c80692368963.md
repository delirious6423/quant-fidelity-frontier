---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - pruning
  - reap
  - expert-pruning
  - llama.cpp
  - code
  - agentic
---

> [!IMPORTANT]
> **Superseded by [REAP-320 multi-domain](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF).**
> Community feedback in [discussion #2](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF/discussions/2) was right: K=256 prunes too deep. The successor keeps 320 experts with multi-domain calibration — cleaner termination, 96.3% HumanEval (vs 89.6% here), and roughly half the general-knowledge fabrication rate. Full head-to-head (including against Unsloth UD-IQ1_M, and including where we lose) is published in that thread. This build stays up for reproducibility.


# Qwen3.8-Flash-Next REAP-256 "duo" — GGUF

An expert-pruned GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):
**512 → 256 experts per layer, 83.8 → 57.7 GiB**, pruned against real usage from two domains
(code and agent orchestration).

This is **not an official Qwen release**. It is a community derivative, unaffiliated with
Alibaba/Qwen.

**The point of this model is not that it is faster. It is that it fits.** See
[What pruning actually buys](#what-pruning-actually-buys) — I measured it, and the honest
answer is more interesting than a speedup number.

---

## ✅ Runs on llama.cpp mainline (since 2026-08-27)

Qwen3.8-Flash-Next uses the `qwen4exp` architecture (hybrid Gated DeltaNet + Qwen Sparse
Attention + an n-gram embedding block). Support was merged into mainline on **2026-08-27**
([ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742)), so any build
from `master` at or after that merge loads this file — verified in production on a clean
mainline clone at commit `ca3d5a3`.

**Releases older than 2026-08-27 will refuse the file.** If yours does, update and rebuild:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -t llama-server
```

---

## What this is

[REAP](https://github.com/CerebrasResearch/reap)-style expert pruning: drop the experts that
carry the least routing mass, keep the rest **bit-for-bit**.

In this GGUF the expert index is the slowest-moving dimension (`ne = [n_embd, n_ff, 512]`), so
each expert is a contiguous, block-aligned span. Pruning is a **binary copy** — no dequantize,
no requantize. Surviving experts keep their original bits. That is why capability holds up.

| | value |
|---|---|
| Experts per layer | 512 → **256** (top-50% by activation magnitude) |
| Size | 83.8 → **57.69 GiB** (2 shards: 44.86 + 12.83) |
| Base quant | Unsloth `UD-Q3_K_XL` |
| Routing mass retained | **89.29%** (worst layer, L0: 79.6%) |
| Active params | unchanged — still top-10 experts + 1 shared |
| Vision | preserved in the weights (mmproj sold separately) |

### Calibration corpus

Salience was computed with `llama-imatrix` expert activation counts over **two domains
combined**, not one:

- **code** — 164k tokens
- **agent orchestration / tool calling** — 180k tokens

Both distributions were merged before ranking, so a single model serves both roles. Measured:
splitting into two domain-specialised siblings gains only 1.9 / 3.5 points of routing mass, and
llama.cpp does not share weights between loaded models — each sibling would mmap its own copy of
the 26.8 GiB PLE table. One model wins on every practical axis.

⚠️ **Two traps I hit while building this, in case you try it:** subsampling a corpus makes a
domain look "specialised" when it is only smaller; and pruning domain A while scoring with
domain B's list inflates the apparent case for splitting by ~11×.

---

## What pruning actually buys

Everyone reports pruning speedups by comparing the pruned model on GPU against the original on
CPU. That conflates two effects. I measured them separately — **both models with all experts on
CPU (`--cpu-moe`), same binary, same context**, so placement cannot help either one:

| both on CPU | pruned (256) | original (512) | ratio |
|---|---|---|---|
| **decode** | 22.0 tok/s | 21.5 tok/s | **1.02× — a tie** |
| **prefill** | 353 tok/s | 223 tok/s | **1.58×** |

**Decode does not improve, and it was never going to.** The model activates top-10 experts
whether it owns 512 or 256, so the per-token compute is identical. Pruning does not make the
math cheaper.

**Prefill does improve (1.58×)**, and this one surprised me. In prefill you push thousands of
tokens at once and they collectively touch *many distinct* experts — halving the catalogue
genuinely halves the memory swept. In single-token decode you only ever touch 10.

So where does the real speedup come from?

| RTX 5090 (32 GB), `--n-cpu-moe 8` | pruned |
|---|---|
| decode | **68.4 tok/s** |
| prefill | **934 tok/s** |

**57.7 GiB fits 44 of 48 expert layers onto a 32 GB card. 83.8 GiB does not.** The 2.9× decode
gain is entirely the difference between living in VRAM and living in system RAM — pruning is
what makes that possible, not what makes it fast.

If you are going to run this on CPU anyway, **this model gives you ~nothing on decode over the
original.** Download it because it fits your GPU, or because you want the faster prefill.

---

## Quality

Scored with a machine-verified 6-task battery (13 points): generated code is **executed against
10 hidden cases**, numbers are compared exactly, JSON is parsed and asserted, and one task only
scores if the model *refuses* to hallucinate a CLI flag that does not exist. No human judging, no
LLM-as-judge.

| reasoning effort | score | notes |
|---|---|---|
| `medium` | **12–13 / 13** | 13/13 is the ceiling, not the average — see the honesty note |
| `low` | **12 / 13** | Noticeably less wall clock. Loses only the long-document aggregation task |

Sampling: `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0` (Qwen's official values for
thinking mode).

**Honesty note (updated after repeated runs).** An earlier version of this card reported
**13/13** from a single run. That run was real, but it was lucky: the long-document aggregation
task is the one unstable task in the battery, and across all repeated runs it only came back
perfect a minority of the time. At `low` it scored 2/3 in four out of four runs; at `medium` it
reached 3/3 in about half of the runs and 2/3 otherwise, with occasional empty-content stalls
(see trap 3 below). Treat **12/13 as the typical score** and 13/13 as the best case.

🔑 **Use `low` for everyday work and `medium` when aggregating over long documents.** That is
the one task where the effort lever measurably helps — just don't expect 3/3 every time.

---

## Running it

```bash
llama-server -m Qwen3.8-Flash-Next-UD-Q3_K_XL-reap256-00001-of-00002.gguf \
  -ngl 99 --n-cpu-moe 8 -c 98304 -fa on --jinja \
  --parallel 2 --kv-unified \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Tune `--n-cpu-moe` to your VRAM: lower keeps more expert layers on the GPU. On a 32 GB card, `8`
lands at ~31 GB used.

### Three things that will bite you

1. **`--parallel > 1` requires `--kv-unified`.** Without it the indexer cache and the attention
   cache desynchronise and `qwen4exp.cpp` aborts on an assert. With a unified KV pool they track
   each other and multi-slot is fine.
2. **Set the reasoning effort explicitly.** The chat template defaults to
   `reasoning_effort | default('xhigh')`, so if you say nothing the model silently runs at
   maximum and replies take minutes. Pass
   `"chat_template_kwargs": {"reasoning_effort": "low"}`.
3. **Empty `content` happens, and `max_tokens` is not the whole story.** Reasoning goes to
   `reasoning_content` and the answer to `content`. On a tight `max_tokens` the model spends the
   budget thinking and `content` comes back empty — that part is just budget. But we have also
   measured runs on long-context tasks that spent **13,000+ tokens**, finished with
   `finish_reason: stop`, and still returned empty `content`: the model occasionally stalls
   inside the reasoning block and never emits a final answer. Raising `max_tokens` does not
   prevent that mode — detect empty `content` and retry.

---

## Prefill note (2026-09-06)

The 26.8 GiB per-layer-embedding table is untouched by pruning and is read through `mmap`, one page fault
per token row; cold prompts prefill 3× slower than warm ones for that reason alone. See the
[REAP-320 card](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF#new-2026-09-06-prefill--the-ple-table-is-why-your-prompt-speed-swings-3-and---lazy-mode-on-direct-fixes-it)
for the measurements and the fix (`--lazy-mode on-direct`, [PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136)).
It applies to every Qwen3.8-Flash-Next GGUF, this one included.

## Limitations

- **Not tested beyond code, orchestration and general reasoning.** It was pruned against those
  two domains on purpose. Expect degradation on whatever the calibration corpus never saw —
  poetry, translation, and niche languages are unmeasured here.
- The pruning is uniform across layers (`expert_count` is a single global scalar in GGUF), so
  every layer keeps K=256 even though layer 0 retains less routing mass (79.6%) than the rest.
- Single-quant release (`UD-Q3_K_XL`). No smaller variants yet.
- Battery is 6 tasks. It is a sharp instrument, not a broad one.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-Flash-Next, released under the
  Qwen Community License 1.0.
- [Unsloth](https://huggingface.co/unsloth) for the `UD-Q3_K_XL` quant this was pruned from.
- [Cerebras Research](https://github.com/CerebrasResearch/reap) for the REAP method.
- The llama.cpp contributors on PR #27742 who made `qwen4exp` run locally at all.

Licensed under the **Qwen Community License 1.0**, inherited from the base model. The full
license text is included in this repository.
