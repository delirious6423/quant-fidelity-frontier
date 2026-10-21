---
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - apple-silicon
  - quantized
  - 6-bit
  - multimodal
  - vision-language
  - hybrid-attention
  - speculative-decoding
---

# Qwen3.8-27B-Alis-MLX-6bit

![Build size vs decode speed across the 8-bit, 6-bit and 4-bit builds, with Korean corpus perplexity and the speculative ladder on the 4-bit build: MTP k=2 and the gated MTP k=4 operating point](assets/builds.png)

A 6-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple silicon, with the **vision tower and the MTP head both preserved**, and a **graded bit allocation** on top (2026-08-22). 21.9 GB on disk, 27.1 tok/s decode, Korean corpus perplexity **statistically indistinguishable from the bf16 original** (6.1018 vs 6.0954, paired over 35K tokens), and only hairline — though measurable — excesses on English and code. This is the balanced build of the set: it fits a 32 GB Mac and keeps non-English output intact.

## At a glance

| Build | Size (GB) | Decode (tok/s) | Prefill (tok/s) | Peak RAM (GB) | PPL en / ko / code |
|---|---|---|---|---|---|
| bf16 reference | 51.8 | 12.6 | 410 | 51.1 | 5.7734 / 6.0954 / 1.6813 |
| 8-bit | 27.9 | 22.0 | 429 | 28.1 | 5.7760 / 6.0987 / 1.6815 † |
| 6-bit (this repo) | 21.9 | 27.1 | 424 | 22.1 | 5.7924 / 6.1018 / 1.6854 † |
| 4-bit AWQ | 15.4 | 36.9 | 436 | 15.7 | 5.8450 / 6.2609 / 1.8105 † |

Size, decode and peak RAM are the **graded revisions**, measured on one Apple M3 Ultra (512 GB unified memory) with the builds queued back to back in a single alternating batch so they share machine state; the bf16 row is carried over from the earlier batch. Prefill is from that earlier batch — the graded revision moves it by ≤0.2% on all three builds, inside noise. Peak RAM is the load-plus-probe peak at short context; add KV cache for your own context length (see below). Perplexity is corpus-scale strided PPL — ctx 2048, stride 512, ≈103K scored tokens — described in the quality section, and was measured on the pre-graded weights (see †).

Against bf16 this build is 42% of the size at 2.15× the decode speed. Against the 8-bit build it is 22% smaller and 23% faster, and the paired corpus test prices that step at +0.28% PPL on English and +0.23% on code — both significant, both hairline — with Korean not separable at all.

## The vision tower is intact

Qwen3.8-27B is a vision-language checkpoint (`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`), and the standard text-only MLX conversion path **drops the vision weights entirely** — a text-only build carries zero vision tensors and cannot process an image.

**Correction (2026-08-16).** An earlier revision of this card claimed this was the first MLX build to keep the vision weights, and that the 12 other public MLX builds then surveyed all carried zero vision tensors. Both claims were wrong when published — the launch-day survey behind them was too narrow. mlx-vlm-family conversions created the same day, `mlx-community/Qwen3.8-27B-{4bit,8bit,mxfp4,nvfp4}` among them, preserve all 333 vision tensors (none of the four carries a single MTP tensor), and a fuller census on 2026-08-16 — 283 MLX-tagged Hub repos matching this model's name, derivatives and sidecar repos included — found text-only builds with zero vision tensors, mlx-vlm-family builds with vision but no MTP head, and other builds carrying **both** subsystems, several published before this set. This card therefore claims no first and no only. What it stands behind is what the weight index proves about these weights — the complete vision tower, unquantized, and the complete MTP head, together — plus the checks below that both paths actually run here: mlx-vlm image input and MTP and DSpark speculative decoding on the same weights. Whether other both-preserving builds load the same way, I have not tested.

This repo keeps all **333** `model.visual.*` tensors — 0.461B parameters, 0.92 GB — as the **original bf16 bytes**. The tower is not quantized at any bit-width in this set. The shipped `config.json` keeps `vision_config`, declares `Qwen3_5ForConditionalGeneration`, and sets `language_model_only: false`; `preprocessor_config.json` and `video_preprocessor_config.json` ship alongside weights that actually exist.

You do not have to take my word for any of that — the weight index proves it, here or in any other repo:

```python
import json
from huggingface_hub import hf_hub_download

idx = json.load(open(hf_hub_download(
    "avlp12/Qwen3.8-27B-Alis-MLX-6bit", "model.safetensors.index.json")))["weight_map"]
vis = [k for k in idx if k.startswith("model.visual.")]
print(len(vis), "vision tensors,", sum(k.endswith(".scales") for k in vis), "of them quantized")
# 333 vision tensors, 0 of them quantized
```

Tower geometry: depth 27, hidden 1152, 16 heads, patch 16, spatial merge 2. The upstream config declares an empty `deepstack_visual_indexes`, so there are no deep-stack injection points to preserve.

Tensor inventory, identical across all three builds:

| Build | Total tensors | Vision | MTP |
|---|---|---|---|
| 8-bit / 6-bit / 4-bit | 2211 | 333 | 31 |

### Running it on an image

mlx-vlm 0.6.13 already supports `qwen3_5`, so **no porting code was required** — its `sanitize_key` rewrites the `model.visual.` prefix to `vision_tower.` and the checkpoint loads as published.

```bash
pip install mlx-vlm

python -m mlx_vlm.generate \
  --model avlp12/Qwen3.8-27B-Alis-MLX-6bit \
  --image ./shapes.png \
  --prompt "Describe every shape in this image, with its color and position."
```

Verified end to end before publishing: I fed a hand-drawn test image (red circle top left, blue square top right, green triangle at the bottom) to the 4-bit build of this set — its earlier uniform release — and it described every color, shape, and position correctly. The vision path is the same bf16 tower in all three builds.

## Which build should I use?

- **6-bit (this repo, 21.9 GB)** — the default I would reach for first. Its Korean corpus PPL is statistically indistinguishable from bf16 — the axis where the 4-bit build pays its clearest user-visible cost — while staying 6.0 GB smaller and 23% faster than 8-bit. Peak was 22.1 GB at short context and it leaves usable headroom on a 32 GB Mac, which is the machine this build exists for.
- **8-bit (27.9 GB)** — the fidelity choice. Take it when the output is the product and the machine is 48 GB or larger: statistically indistinguishable from bf16 on every corpus slice. Also the right reference to compare a smaller build against.
- **4-bit (15.4 GB, AWQ + graded)** — the reach build, and the fastest at 36.9 tok/s. Take it when the model would otherwise not fit. Its corpus PPL runs +1.2% / +2.7% / +7.7% over bf16 on English / Korean / code — the one build whose cost is significant on every slice, and the Korean share of it is what the extra 6.5 GB here buys back.

All three carry the same vision tower and the same MTP head. The choice is purely language-model precision.

Architecture (text side): 64 layers, hidden 5120, GQA with 24 query heads over 4 KV heads, head_dim 256, FFN 17408, vocab 248,320, context 262K. The stack is **hybrid**: 48 GatedDeltaNet linear-attention layers and 16 full-attention layers, one full-attention layer every fourth (`full_attention_interval: 4`).

## Text generation

```bash
pip install mlx-lm

mlx_lm.generate \
  --model avlp12/Qwen3.8-27B-Alis-MLX-6bit \
  --prompt "Explain how a hybrid linear-attention stack changes long-context memory cost." \
  --max-tokens 512
```

As an OpenAI-compatible server:

```bash
mlx_lm.server --model avlp12/Qwen3.8-27B-Alis-MLX-6bit --port 8080
```

The shipped sampling defaults are temperature 1.0, top_p 0.95, top_k 20.

### Chat template and the thinking channel

The chat template defaults `enable_thinking` to **true**. A normal chat call therefore emits a `<think>` block, closed by `</think>`, before the answer.

`mlx_lm.generate` and `mlx_lm.server` apply the template for you. **If you bypass it** — with `--ignore-chat-template`, or by pushing a raw string through your own code path — **the reasoning channel comes back exposed in the visible output.** Route prompts through `apply_chat_template`:

```python
from mlx_lm import load, generate

model, tokenizer = load("avlp12/Qwen3.8-27B-Alis-MLX-6bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Refactor this function to be iterative."}],
    add_generation_prompt=True,
    tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
```

To turn thinking off, pass the template argument through:

```bash
mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-6bit \
  --chat-template-config '{"enable_thinking": false}' \
  --prompt "Give me the answer only, no reasoning."
```

Budget `--max-tokens` generously while thinking is on. A tight cap can be spent entirely inside the reasoning block, leaving no answer at all — that looks like model damage and is not.

## MTP self-speculative decoding

The checkpoint's vendor Multi-Token Prediction head is preserved here — **31 MTP tensors**, used exactly as shipped. There is no draft model to download and no extra weight memory beyond the head itself: the model drafts its own continuation and verifies it in the same step.

Speculative decoding needs a fork of mlx-lm:

```bash
pip install git+https://github.com/avlp12/mlx-lm.git

mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-6bit \
  --mtp --mtp-num-draft-tokens 2 \
  --prompt "Write a Python function to check if a string is a palindrome." \
  --max-tokens 512
```

**Correction (2026-08-16).** The speculative table earlier revisions of this card quoted from the 4-bit build — plain 37.63, MTP k=2 50.36, DSpark 62.21 — came from a harness that **did not stop at end-of-sequence**: on prompts whose answer ended inside the fixed window, the remainder measured post-termination behavior, and on the math prompt the model spent that tail copying its own finished answer (acceptance 4.53), which carried most of the headline. Those figures are retracted, as the footnote they carried promised they would be if unreproducible. The tables below are the re-measurement: **the harness stops at EOS, the four prompts (chat / code / math / Korean) are long-form** so every scored token is genuine answer decode, sampled rows are medians of three runs per prompt averaged across prompts, and stop-detection stays out of the timed loop.

**Every figure below is still measured on the 4-bit build of this set — not on this one** (its earlier uniform weights; the AWQ swap measures within ±1%, paired). It is quoted because it is what the fork's two speculative paths do on this checkpoint. Greedy first:

| Configuration (greedy, 4-bit build) | tok/s | vs plain |
|---|---|---|
| plain | 37.6 | 1.00 |
| DSpark drafter (block 8) | 48.3 | 1.28× |
| MTP k=2 | 46.8 | 1.24× |
| **MTP k=4 + `min_draft_p` 0.6** | **52.8** | **1.40×** |

And under the shipped sampling defaults — temperature 1.0, top_p 0.95, top_k 20, what a chat deployment actually runs — using truncated rejection sampling, which passes target and draft distributions through the same temperature/top-p/top-k chain and so **preserves the requested output distribution exactly** (total variation ≤ 0.0014 against a synthetic oracle; 160/160 token match in the greedy limit):

| Configuration (temp 1.0 · top_p 0.95 · top_k 20, 4-bit build) | 240 tok | vs plain | 1024 tok | vs plain |
|---|---|---|---|---|
| plain, sampler on | 37.16 | 1.00 | 36.91 | 1.00 |
| DSpark drafter | 41.94 | 1.13× | 39.95 | 1.08× |
| MTP k=2 | 47.74 | 1.28× | 44.10 | 1.19× |
| **MTP k=4 + `min_draft_p` 0.6** | **48.05** | **1.29×** | **45.13** | **1.22×** |

The gated k=4 row is the 4-bit recommendation (`mtp_num_draft_tokens=4, min_draft_p=0.6` via the Python API, plus `mtp_spec_temp=1.0, mtp_spec_top_k=20, mtp_spec_top_p=0.95` under the sampling defaults — the gate is what makes k=4 win; ungated deeper k measured below k=2).

**Why this build cannot inherit those multiples.** Part of what lifts them is a small-M quantized-GEMM kernel that the fork enables from inside `mlx_lm.utils.load()`. It fires only in the multi-token verification window (M = 6..8) and only on **4-bit, group-64** weights, so it never engages at 6 bits; plain decode at M = 1 is untouched on every build either way. The gap it fills is upstream: [ml-explore/mlx#4265](https://github.com/ml-explore/mlx/issues/4265). Set `MLXLM_NO_FAST_QMM=1` to opt out.

**This 6-bit build has never been measured under either speculative path** — the MTP head is present, the fork drives it, and the drafter below accepts it as a target, but I have no number for any row on this build. The 8-bit build, which also runs without the kernel, measured a real EOS-clean gain (DSpark 1.54× at block 4, MTP k=2 1.42×), so a kernel-less tier can still collect — but that is its number, not this build's. Start at k=2 and measure your own workload rather than interpolating across precisions.

## DSpark: an external drafter, measured on the 4-bit build

DSpark is the fork's other speculative path. Unlike MTP it is **not** self-speculation: it drives an external drafter, [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) — 1.36B parameters, trained against this checkpoint, published by a third party. The drafter carries no embedding and no output head of its own; it borrows the target's, and drafts a whole block of positions in one forward pass instead of chaining them.

Two things to settle first:

- **It is a separate download.** The MTP head ships inside this repo; the DSpark drafter does not, and nothing in these weights contains it. Budget 2.7 GB in bf16, or 0.76 GB once quantized to 4-bit, which is how it was measured.
- **It is not covered by this repository's license.** The drafter repo declares `license: other`, not Apache-2.0. The Apache-2.0 on this page covers the quantized Qwen weights only; read the drafter's own terms in its repository before you deploy it.

```python
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.dspark import load_dspark
from mlx_lm.dspark_generate import dspark_generate_step

model, tokenizer = load("avlp12/Qwen3.8-27B-Alis-MLX-6bit")

draft, _ = load_dspark(snapshot_download("RadixArk/Qwen3.8-27B-DSpark"))
nn.quantize(draft, group_size=64, bits=4)   # 2.7 GB bf16 into 0.76 GB, as measured
mx.eval(draft.parameters())

prompt = mx.array(tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python function to check if a string is a palindrome."}],
    add_generation_prompt=True,
))

for token, _accepted in dspark_generate_step(prompt, model, draft, max_tokens=512):
    print(tokenizer.decode([token.item()]), end="", flush=True)
```

The defaults are the measured operating point on the 4-bit build — block size 8, verification width capped at 8, the drafter's confidence head off — so the call above needs no tuning to reproduce the 48.3 row there. On a kernel-less tier the optimum shifts: the 8-bit build measured best at **block 4**. This build has no measurement of its own; if you deploy DSpark on it, sweep the block size yourself, starting from 4.

### Korean: the recommendation reversed

Until this revision the card said **"run Korean workloads plain,"** on the strength of 4-bit Korean cells that read below plain decode (MTP 34.3, DSpark 33.3 against 37.6). **That was an artifact of the retracted protocol** — short answers plus a harness that measured past their end. Under the corrected protocol the gated MTP configuration on the 4-bit build is **faster on Korean than plain**: +34% greedy, and +27–31% under the shipped sampling defaults (47.2 vs 37.2 tok/s at 240 tokens; 48.4 vs 36.9 at 1024). The gate is what changed the verdict: uncertain Korean stretches degenerate into near-plain steps instead of paying for rejected chains.

One honest exception: **DSpark under the sampling defaults stays at or below plain on Korean** (well below at 240 tokens, break-even at best by 1024), so the reversal covers the gated MTP path, not speculation in general. And as with everything in this section, the Korean numbers are the 4-bit build's — this build carries the same head and the same drafter interface, but no measurement.

### Measurement protocol, and why single-prompt numbers lie

Every figure above is the **average over the fixed four-prompt set**, never a single run. Speculative decoding pays off in proportion to how predictable the continuation is, so a benchmark built from one favorable prompt **overstates the gain you will see in real traffic** by a wide margin — treat any single-prompt speculative figure, including mine, as an upper bound.

The correction at the top of this section adds a second rule, learned at the cost of a retraction: **a speculative benchmark that does not stop at end-of-sequence is measuring luck** — what the model happens to do after its answer ends. The retracted 62.21 headline, and the 91.5 tok/s math cell behind it, were mostly that luck. The per-cell records behind the restated tables ship in the [campaign repository](https://github.com/avlp12/qwen38_alis_mlx).

## KV cache quantization

The hybrid layout is why long context is cheap on this model. Only the **16 full-attention layers** hold a growing KV cache; the 48 GatedDeltaNet layers carry a fixed-size recurrent state instead. With 4 KV heads at head_dim 256, that comes to **64 KB per token** in bf16 across the whole model. That per-token figure is a property of the layout and is identical in all three builds.

Measured at 16K context on the 4-bit and 8-bit builds (the 4-bit column on that repo's earlier uniform release), as whole-process peaks — so they include activations and prefill buffers, not the cache alone. This 6-bit build was not measured; it sits between the two columns.

| KV precision | Peak RAM (GB), 4-bit build | Peak RAM (GB), 8-bit build | top-1 agreement |
|---|---|---|---|
| bf16 | 20.80 | 34.46 | 100% |
| 8-bit | 20.30 | 33.96 | 100% |
| 4-bit | 20.03 | 33.69 | 100% |

**The verdict: quantize the cache only for long context.** At 16K the whole exercise buys 0.77 GB, which is inside the noise of whatever else is resident on the machine, and costs roughly 3% of decode speed. Quality was untouched — top-1 agreement held at 100% even with a 4-bit cache — so there is no fidelity argument either way at this length.

The arithmetic changes completely at the model's full context. Cache alone at 262K runs **16.8 GB in bf16 against 4.2 GB at 4-bit, a 12.6 GB saving** — the difference between fitting and not fitting, and on a 32 GB machine holding this build that saving is the entire long-context story.

```bash
mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-6bit \
  --kv-bits 4 --prompt "..." --max-tokens 512
```

**Note:** `--kv-bits` exists on `mlx_lm.generate` only. `mlx_lm.server` has no such option, so a served deployment gets the bf16 cache whether you want it or not. Size your server's context budget accordingly.

## Quantization recipe

Produced with affine quantization at **6 bits, group size 64, uniform across the language model**, for a reported **6.501 bits per weight**. The extra half bit is the fp16 scale and bias each group of 64 weights carries. Because the language model is uniform, the Hub precision badge matches what actually loads.

Held at bf16, deliberately:

- **the entire vision tower** — all 333 `model.visual.*` tensors, byte-identical to upstream, with no `.scales` or `.biases` anywhere in the index
- every RMSNorm, including `q_norm` and `k_norm`
- the GatedDeltaNet recurrent-state parameters: `A_log`, `dt_bias`, and the short `conv1d` kernel

The state parameters are the deliberate part on the text side. They set the decay and gating of a state that carries the entire sequence, so a coarse grid there would cap quality for every token downstream — and they are tiny, so bf16 costs essentially nothing in file size.

The vision tower is skipped for a different reason. At 0.92 GB it is 4% of this build's 21.9 GB, so quantizing it would save almost nothing while adding an unmeasured error term to a path I had no evaluation harness for. Preserving the original bytes means the image path in this repo is bit-identical to upstream.

This is a flat, data-free round-to-nearest recipe: no calibration set, no activation-aware scaling, no distillation.

### The graded allocation (2026-08-22)

Five tensor groups are moved off the uniform 6-bit width, because one width serves them badly and
moving them is cheap: `self_attn.k_proj` and `self_attn.v_proj` (0.33% of parameters each under
GQA), the GatedDeltaNet gates `linear_attn.in_proj_a` / `in_proj_b` (0.04% each), and `lm_head`
— all to **8-bit**. The MTP head, which the AWQ converter leaves in bf16, is quantized to 6-bit.

Cost **+1.6% of file size** (21.53 → 21.88 GB) and **−1.2% decode** (27.45 → 27.12 tok/s).
Paired against bf16 on the same windows, versus the previous revision of this repo:

| slice | before | after | change |
|---|---:|---:|---|
| English | 0.00389 | **0.00296** | **−23.9%** (t = −10.1) |
| Korean | 0.00347 | **0.00252** | **−27.3%** (t = −13.2) |
| code | 0.01177 | **0.01009** | **−14.3%** (t = −3.4) |
| pooled | 0.00616 | **0.00499** | **−18.9%** (t = −7.3) |

**All three slices clear significance**, and this tier gains more than the 4-bit one does
(−18.9% pooled against −8.7%). The control that makes the claim survivable: spending the identical
bytes on arbitrary mid-depth FFN tensors instead bought −1.2%, byte-matched to within 0.02% — the
gain is the choice of tensors, not the spending. Method:
[docs/bit-allocation.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/bit-allocation.md).

## Quality

Corpus-scale strided perplexity against fixed token streams: window 2048, stride 512, so every scored token sees up to 2048 tokens of left context. Three slices — **English** = wikitext-2 test (35,981 tokens), **Korean** = Korean Wikipedia (35,179 tokens), **code** = CPython standard library (32,295 tokens). Every build scores the same tokens in the same order, so build-to-build comparisons are paired per token, and significance is judged on the paired excess NLL with a blockwise standard error (512-token blocks) — roughly 16–20× the sensitivity of the probe this section used to carry.

| Build | en · wikitext-2 | ko · Korean wiki | code · CPython |
|---|---|---|---|
| bf16 reference | 5.7734 | 6.0954 | 1.6813 |
| 8-bit | 5.7760 | 6.0987 | 1.6815 |
| **6-bit (this repo)** | **5.7924** | **6.1018** | **1.6854** † |
| 4-bit AWQ | 15.4 | 36.9 | 436 | 15.7 | 5.8450 / 6.2609 / 1.8105 † |

This build is where the size curve and the quality curve are still on speaking terms. Against bf16, paired: **Korean is not separable at all** — the 95% CI on this build's Korean excess NLL includes zero — while English and code carry excesses the test can find but you will struggle to feel: +0.0033 and +0.0024 nats/token (+0.33% and +0.24% PPL). The 8-bit build is indistinguishable from bf16 on all three slices; the 4-bit AWQ build is significantly above on all three, at +1.2% / +2.7% / +7.7% — on Korean and code, a different quality class from this one.

Three honest caveats:

- **"Indistinguishable" means the test could not separate them at this sample size, not that they are identical.** A still larger corpus might resolve a Korean difference; 35K paired tokens did not.
- **Perplexity prices average next-token surprise, not task success.** It ranks the builds with real statistical power, but it is not a benchmark suite.
- **The hairline English and code excesses are significant but tiny.** Treat them as the honest price of this tier, not as a reason to step up to 8-bit — step up when you need the reference build, not because of 0.3%.

**Why this table changed.** Earlier revisions of this card ranked the builds with a top-1 agreement probe over three short passages (62–112 tokens). At that size the probe had no statistical power: its standard error ran several times the effects it claimed to rank, and on English it read the sign of a recipe difference **backwards**. It is removed rather than restated.

† **The perplexity table was measured on the pre-graded weights and has not been re-run.** The
graded revision is no worse on any slice and better on Korean by 15.8% in exact KL, so these
figures are an upper bound on this build's cost — labelled rather than quietly reused.

### The exact-KL tier sweep

A follow-up sweep measured **exact full-vocab KL divergence to bf16** (248,320-way softmax in fp32, no top-K truncation; paired non-overlapping ctx-2048 windows over ≈100K en+ko+code tokens; 512-token block SEs) across ten builds, including mlx-community's artifacts.

This build lands at **KL 0.00499, top-1 agreement 97.5%** — an order of magnitude tighter than the 4-bit tier (0.0592) for +6.5 GB, with the 8-bit tier another ÷4.8 beyond (0.00105) for +6.0 GB more. The steep part of the size-quality curve is exactly the step up to this tier; past it the returns flatten. The pre-graded revision of this repo measured 0.00539.

**Against a like-for-like competitor.** Re-running `orcarouter/Qwen3.8-27B-MLX`'s own protocol
(WikiText-2, context 1024) over 35 windows rather than one, this build reads **0.00311 / p95
0.00825 / top-1 97.54%** against their 6-bit's 0.00322 / 0.00922 / 97.33%. Their published card
reports 0.00216 for that build — a figure reproducible only on a **single 1024-token window**;
the same weights over 35 windows of the same corpus give 0.00322. The comparison above puts both
builds on one ruler. Chart, per-slice numbers, and method: [docs/kl-tiers.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/kl-tiers.md).

![tier chart](https://raw.githubusercontent.com/avlp12/qwen38_alis_mlx/main/docs/assets/tier_chart.png)

## Limitations and caveats

- **High reasoning effort thinks long by design, at every bit width.** I tested the report that 4-bit quantization causes runaway thinking: on hard prompts at `reasoning_effort: xhigh`, thinking runs 4k-16k+ tokens on this build *and* on the 8-bit control at statistically indistinguishable rates (25.0% vs 20.8% of runs hitting a 16k cap, paired protocol), with no repetition pathology in either. Budget `max_tokens` generously for xhigh or use a lower effort; quantization is not the variable.

- **Perplexity and full-vocab KL to bf16 are the only quality metrics here.** Both are corpus-scale (≈100K paired tokens) and they replaced a probe that had no statistical power — but they price distribution fidelity, not task success.
- **No standard benchmarks were run.** There are no MMLU, GSM8K, or HumanEval numbers here, and none should be inferred from the perplexity table.
- **Vision quality is unquantified.** The tower is preserved bit-exact and verified to work on a shapes probe. That is a functional check, not a VQA evaluation. No image benchmark was run at any bit-width.
- **MTP and KV-cache figures come from the neighboring builds.** This build was not measured on either axis; the head and the layout are present and identical, the numbers are not.
- **Shape-dependent numerics.** The hybrid stack takes different kernel paths depending on the shape of the input it receives — the batched scoring pass behind the perplexity table is not the same shape as incremental decode, and near-tie positions can resolve differently between those paths. Perplexity is far less sensitive to this than an argmax metric, but do not expect bit-identical numbers from a different harness.
- **Long-form generated reasoning is unprobed.** Perplexity scores 2048-token windows of existing text; it does not exercise long generated reasoning chains, where low-bit damage tends to concentrate. Spot-check your own thinking-mode prompts.

## License and attribution

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). This repository redistributes quantized weights derived from that checkpoint; all model credit belongs to the Qwen team. Quantization, vision-tower preservation, and measurement by [avlp12](https://huggingface.co/avlp12).
