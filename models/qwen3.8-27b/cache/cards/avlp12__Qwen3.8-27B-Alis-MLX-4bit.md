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
  - 4-bit
  - awq
  - multimodal
  - vision-language
  - hybrid-attention
  - speculative-decoding
---

# Qwen3.8-27B-Alis-MLX-4bit

![Build size vs decode speed across the 8-bit, 6-bit and 4-bit builds, with Korean corpus perplexity and the speculative ladder on the 4-bit build: MTP k=2 and the gated MTP k=4 operating point](assets/builds.png)

An **activation-aware (AWQ) 4-bit** MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple silicon, with the **vision tower and the MTP head both preserved**, and a **graded bit allocation** on top of AWQ (2026-08-22). 15.4 GB on disk, 36.9 tok/s plain decode, and a 15.7 GB peak that puts a 27B-class vision-language model on a 24 GB Mac.

The graded allocation moves five tensor groups off the uniform width and is worth **−16.6% / −16.4% / −3.9% KL** against the previous revision of this repo on English / Korean / code, for **+1.6% size and −2.3% decode**. The recipe and the measurements are below; the previous weights remain on the `pre-graded` branch.

It is also the build with a real quality cost — the excess over bf16 is significant on every slice: smallest on English, about twice that on Korean, largest in relative terms on code. **Read the quality section before you deploy it.**

## At a glance

| Build | Size (GB) | Decode (tok/s) | Prefill (tok/s) | Peak RAM (GB) | PPL en / ko / code |
|---|---|---|---|---|---|
| bf16 reference | 51.8 | 12.6 | 410 | 51.1 | 5.7734 / 6.0954 / 1.6813 |
| 8-bit | 27.9 | 22.0 | 429 | 28.1 | 5.7760 / 6.0987 / 1.6815 † |
| 6-bit | 21.9 | 27.1 | 424 | 22.1 | 5.7924 / 6.1018 / 1.6854 † |
| 4-bit AWQ + graded (this repo) | 15.4 | 36.9 | 436 | 15.7 | 5.8450 / 6.2609 / 1.8105 † |

Size, decode and peak RAM are the **graded revisions**, measured on one Apple M3 Ultra (512 GB unified memory) with the builds queued back to back in a single alternating batch so they share machine state; the bf16 row is carried over from the earlier batch. Prefill is from that earlier batch — the graded revision moves it by ≤0.2% on all three builds, inside noise. Peak RAM is the load-plus-probe peak at short context; add KV cache for your own context length (see below). Perplexity is corpus-scale strided PPL — ctx 2048, stride 512, ≈103K scored tokens — described in the quality section, and was measured on the pre-graded weights (see †).

Against bf16 this build is 30% of the size at 2.93× the decode speed — the largest jump in the set, and the only one that changes which machines can run the model at all.

## The vision tower is intact

Qwen3.8-27B is a vision-language checkpoint (`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`), and the standard text-only MLX conversion path **drops the vision weights entirely** — a text-only build carries zero vision tensors and cannot process an image.

**Correction (2026-08-16).** An earlier revision of this card claimed this was the first MLX build to keep the vision weights, and that the 12 other public MLX builds then surveyed all carried zero vision tensors. Both claims were wrong when published — the launch-day survey behind them was too narrow. mlx-vlm-family conversions created the same day, `mlx-community/Qwen3.8-27B-{4bit,8bit,mxfp4,nvfp4}` among them, preserve all 333 vision tensors (none of the four carries a single MTP tensor), and a fuller census on 2026-08-16 — 283 MLX-tagged Hub repos matching this model's name, derivatives and sidecar repos included — found text-only builds with zero vision tensors, mlx-vlm-family builds with vision but no MTP head, and other builds carrying **both** subsystems, several published before this set. This card therefore claims no first and no only. What it stands behind is what the weight index proves about these weights — the complete vision tower, unquantized, and the complete MTP head, together — plus the checks below that both paths actually run here: mlx-vlm image input and MTP and DSpark speculative decoding on the same weights. Whether other both-preserving builds load the same way, I have not tested.

This repo keeps all **333** `model.visual.*` tensors — 0.461B parameters, 0.92 GB — as the **original bf16 bytes**. The tower is not quantized at any bit-width in this set. The shipped `config.json` keeps `vision_config`, declares `Qwen3_5ForConditionalGeneration`, and sets `language_model_only: false`; `preprocessor_config.json` and `video_preprocessor_config.json` ship alongside weights that actually exist.

That combination is the point of this particular build: **a full-precision vision tower on a 4-bit language model**, 15.4 GB total, which is small enough to run vision-language work on a 24 GB Mac.

You do not have to take my word for any of that — the weight index proves it, here or in any other repo:

```python
import json
from huggingface_hub import hf_hub_download

idx = json.load(open(hf_hub_download(
    "avlp12/Qwen3.8-27B-Alis-MLX-4bit", "model.safetensors.index.json")))["weight_map"]
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
  --model avlp12/Qwen3.8-27B-Alis-MLX-4bit \
  --image ./shapes.png \
  --prompt "Describe every shape in this image, with its color and position."
```

Verified end to end before publishing, on this set's earlier uniform 4-bit release — the vision tower here is byte-identical to it: I fed it a hand-drawn test image — red circle top left, blue square top right, green triangle at the bottom — and it described every color, shape, and position correctly. That is the functional check that the tower survived quantization of everything around it; it is not a VQA evaluation.

## Which build should I use?

- **4-bit AWQ + graded (this repo, 15.4 GB)** — the reach build. Take it when the alternative is not running the model at all: 24 GB and 32 GB Macs, several models resident at once, or throughput work where 36.9 tok/s plain and 52.8 tok/s with speculative decoding matter more than the last points of fidelity. It is the one build whose quality cost is significant on every slice — PPL +1.2% / +2.7% / +7.7% over bf16 on English / Korean / code — and the AWQ recipe exists precisely to shrink that cost: it recovers 48.7% / 26.9% / 14.7% of the gap the uniform 4-bit recipe paid against 8-bit. It is **not** the pick for Korean output.
- **6-bit (21.9 GB)** — the balanced default, and the first build to step up to when non-English output matters: its Korean corpus PPL is statistically indistinguishable from bf16 (6.1018 vs 6.0954), against +2.7% here, for 6.5 GB.
- **8-bit (27.9 GB)** — the fidelity choice, for 48 GB machines and up: statistically indistinguishable from bf16 on every corpus slice — English, Korean, and code.

All three carry the same vision tower and the same MTP head. The choice is purely language-model precision.

Architecture (text side): 64 layers, hidden 5120, GQA with 24 query heads over 4 KV heads, head_dim 256, FFN 17408, vocab 248,320, context 262K. The stack is **hybrid**: 48 GatedDeltaNet linear-attention layers and 16 full-attention layers, one full-attention layer every fourth (`full_attention_interval: 4`).

## Text generation

```bash
pip install mlx-lm

mlx_lm.generate \
  --model avlp12/Qwen3.8-27B-Alis-MLX-4bit \
  --prompt "Write a Python function that merges two sorted lists." \
  --max-tokens 512
```

As an OpenAI-compatible server:

```bash
mlx_lm.server --model avlp12/Qwen3.8-27B-Alis-MLX-4bit --port 8080
```

The shipped sampling defaults are temperature 1.0, top_p 0.95, top_k 20.

### Chat template and the thinking channel

The chat template defaults `enable_thinking` to **true**. A normal chat call therefore emits a `<think>` block, closed by `</think>`, before the answer.

`mlx_lm.generate` and `mlx_lm.server` apply the template for you. **If you bypass it** — with `--ignore-chat-template`, or by pushing a raw string through your own code path — **the reasoning channel comes back exposed in the visible output.** Route prompts through `apply_chat_template`:

```python
from mlx_lm import load, generate

model, tokenizer = load("avlp12/Qwen3.8-27B-Alis-MLX-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Refactor this function to be iterative."}],
    add_generation_prompt=True,
    tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
```

To turn thinking off, pass the template argument through:

```bash
mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-4bit \
  --chat-template-config '{"enable_thinking": false}' \
  --prompt "Give me the answer only, no reasoning."
```

Budget `--max-tokens` generously while thinking is on. A tight cap can be spent entirely inside the reasoning block, leaving no answer at all — that looks like model damage and is not.

## MTP self-speculative decoding

This build ships a Multi-Token Prediction head in the vendor's 31-tensor layout, quantized to the same 4-bit g64 as the rest of the language model — but as of 2026-08-16 the head weights on `main` are **my realignment of the vendor head to this build's own 4-bit backbone** (details just below). The vendor-original head is preserved byte-for-byte on the [`pre-align`](https://huggingface.co/avlp12/Qwen3.8-27B-Alis-MLX-4bit/tree/pre-align) branch — `revision="pre-align"` restores the previous release exactly. There is no draft model to download and no extra weight memory beyond the head itself: the model drafts its own continuation and verifies it in the same step. The speculative work was measured on this repo's earlier uniform 4-bit release — same size, same 4-bit g64 format; under the EOS-cut paired protocol the AWQ weights shipped here measure **within ±1%** of it on both speculative paths, so the figures below apply to these weights as published.

### Why the head was realigned, and what it measures (2026-08-16)

The vendor trained its MTP head against bf16 hidden states; deployed here, the head reads the hidden states of a 4-bit backbone, and that train/deploy mismatch leaves acceptance on the table. So I retrained the head — and only the head, backbone frozen — by on-policy self-distillation targeting the 4-bit hidden states: the quantized model writes its own greedy chat transcripts, the teacher is the model's own t+2 distribution on those transcripts, and the winning variant adds a chain-step loss (weight 0.3) that matches the context the head actually sees on its second draft. The trained head is then re-quantized to 4-bit g64 before shipping, so what you download is exactly what I measured.

Paired against the vendor head (same process, quantized-head swap, A→B→A with a drift control that repeated within ±0.2%; short-prompt 240-token greedy protocol):

- **At the recommended operating point (greedy, k=4 + gate): +6.1%** (57.3 → 60.75 tok/s, 4-prompt mean; three of four prompts +3.8..+11%). The gain splits between higher chain acceptance (code 2.97 → 3.27) and better-calibrated chain probabilities — the `min_draft_p` gate cuts wasted chains earlier.
- **Korean is the biggest winner, not a casualty**: greedy long-form (1024 tok) **+16.1%** (acceptance 2.18 → 2.51), and Korean under the shipped sampling defaults improved +4.8..+6.8%.
- Under the shipped sampling defaults overall (24 paired cells, 240- and 1024-token arms): **+1.27 tok/s, paired t = 2.26** — small but real.
- One honest negative: one short greedy Korean prompt in my suite reproducibly measures **−1.9%** with the realigned head (a single deterministic trajectory; every other Korean cell improves). I report it rather than hide it.

The tables below were measured with the vendor head on the long-form EOS-cut protocol and are left as published — with the realigned head, the gated k=4 rows improve by the relative margins above, and greedy speculation remains lossless in the same practical sense (commit tokens still come from the target's own argmax).

Speculative decoding needs a fork of mlx-lm:

```bash
pip install git+https://github.com/avlp12/mlx-lm.git

mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-4bit \
  --mtp --mtp-num-draft-tokens 2 \
  --prompt "Write a Python function to check if a string is a palindrome." \
  --max-tokens 512
```

That CLI call is the k=2 row below. The fastest measured configuration adds a per-draft probability gate, which is a Python-API argument:

```python
from mlx_lm import load, generate

model, tokenizer = load("avlp12/Qwen3.8-27B-Alis-MLX-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python function to check if a string is a palindrome."}],
    add_generation_prompt=True, tokenize=False)

# recommended operating point (greedy): k=4 with the p-gate
print(generate(model, tokenizer, prompt=prompt, max_tokens=512,
               mtp=True, mtp_num_draft_tokens=4, min_draft_p=0.6))
```

**Correction (2026-08-16): why every number in this section changed.** The speculative table this card carried until today — plain 37.63, MTP k=2 50.36, DSpark 62.21, with per-prompt cells up to 91.5 — came from a harness that **did not stop at end-of-sequence**. On prompts whose answer ended inside the fixed 240-token window, the remainder measured post-termination behavior; on the math prompt the model spent that tail copying its own finished answer, which drafts almost for free (acceptance 4.53), and that one prompt carried most of the headline. An earlier revision footnoted exactly this risk and promised those figures would not be re-quoted without re-measurement — this table is that re-measurement, and the old figures are retracted. The protocol now: **the harness stops at EOS, and the four prompts (chat / code / math / Korean) are long-form**, so every scored token is genuine answer decode — verified per cell; sampled rows are the median of three runs per prompt, averaged across prompts; stop-detection stays out of the timed loop.

Measured on one Apple M3 Ultra (512 GB), one process per pass. First, greedy — the benchmark convention this set has always used:

| Configuration (greedy) | tok/s | vs plain |
|---|---|---|
| plain | 37.6 | 1.00 |
| DSpark drafter (block 8) | 48.3 | 1.28× |
| MTP k=2 | 46.8 | 1.24× |
| **MTP k=4 + `min_draft_p` 0.6** | **52.8** | **1.40×** |

Greedy speculation remains lossless in the practical sense: both paths reproduce plain greedy decoding, equality-checked, with residual divergence confined to floating-point ties inside the verification batch — ties that resolve at the same positions in a non-speculative control.

Second — new with this revision — **the shipped sampling defaults**: temperature 1.0, top_p 0.95, top_k 20, which is what a chat deployment of this model actually runs. Speculation under a live sampler here uses truncated rejection sampling: target and draft distributions pass through the same temperature/top-p/top-k chain, so **the output distribution the client asked for is preserved exactly** (total variation ≤ 0.0014 against a synthetic oracle; 160/160 token match in the greedy limit):

| Configuration (temp 1.0 · top_p 0.95 · top_k 20) | 240 tok | vs plain | 1024 tok | vs plain |
|---|---|---|---|---|
| plain, sampler on | 37.16 | 1.00 | 36.91 | 1.00 |
| DSpark drafter | 41.94 | 1.13× | 39.95 | 1.08× |
| MTP k=2 | 47.74 | 1.28× | 44.10 | 1.19× |
| **MTP k=4 + `min_draft_p` 0.6** | **48.05** | **1.29×** | **45.13** | **1.22×** |

The sampler itself costs plain decode only 1.2%. To run the recommended configuration under these defaults, mirror the sampling parameters into the speculative acceptance path:

```python
from mlx_lm.sample_utils import make_sampler

print(generate(model, tokenizer, prompt=prompt, max_tokens=512,
               sampler=make_sampler(temp=1.0, top_p=0.95, top_k=20),
               mtp=True, mtp_num_draft_tokens=4, min_draft_p=0.6,
               mtp_spec_temp=1.0, mtp_spec_top_p=0.95, mtp_spec_top_k=20))
```

**The gate is what makes k=4 win — do not raise k without it.** `min_draft_p 0.6` stops chaining the moment a draft token's probability drops below 0.6, so uncertain stretches degenerate into near-plain steps and only confident stretches run the full depth. Without the gate, deeper k measured strictly worse than k=2 in the depth sweep: longer chains raise the odds of a partial acceptance, and on this hybrid stack a partial acceptance costs a full re-feed of the GatedDeltaNet state. If your integration exposes only `--mtp-num-draft-tokens`, keep it at 2.

**One kernel note.** The fork routes quantized layers through a small-M quantized-GEMM kernel from inside `mlx_lm.utils.load()`; it fires only in the multi-token verification window (M = 6..8), so plain decode at M = 1 takes exactly the path it always did. The gap it fills is upstream: [ml-explore/mlx#4265](https://github.com/ml-explore/mlx/issues/4265). Set `MLXLM_NO_FAST_QMM=1` to opt out.

## DSpark: an external drafter — restated, and no longer the fast path

DSpark is the fork's other speculative path. Under the corrected protocol it still clears plain decode — 48.3 tok/s greedy, 1.28× — but it **no longer leads this table**: the previous revision had it at 62.21 against MTP's 50.36, an ordering the EOS-unaware harness manufactured, because the post-EOS self-copy tail drafts almost for free and DSpark profited from it most. EOS-cut, the gated MTP path is faster (52.8), ships inside these weights, and needs no extra download or license review. Under the shipped sampling defaults DSpark's margin thins further — 1.13× at 240 tokens, 1.08× at 1024 — rejection sampling costs it more than it costs MTP. What DSpark still has is the highest raw acceptance on predictable English continuations; what it pays is a separate 1.36B forward pass per block against the target's 248k-row output head.

Unlike MTP it is **not** self-speculation. It drives an external drafter, [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark): 1.36B parameters, trained against this checkpoint, published by a third party. The drafter carries no embedding and no output head of its own — it borrows the target's — and it drafts a whole block of positions in a single forward pass instead of chaining them one at a time.

Two things to settle before the numbers tempt you:

- **It is a separate download.** The MTP head ships inside this repo; the DSpark drafter does not, and nothing in these weights contains it. Budget 2.7 GB in bf16, or 0.76 GB once quantized to 4-bit, which is how it was measured here.
- **It is not covered by this repository's license.** The drafter repo declares `license: other`, not Apache-2.0. The Apache-2.0 on this page covers the quantized Qwen weights only; read the drafter's own terms in its repository before you deploy it.

```python
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.dspark import load_dspark
from mlx_lm.dspark_generate import dspark_generate_step

model, tokenizer = load("avlp12/Qwen3.8-27B-Alis-MLX-4bit")

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

**The defaults are the measured operating point**, not a permissive setting: block size 8, verification width capped at 8 to stay inside the kernel's window, the drafter's confidence head off. The call above is exactly the configuration the 48.3 row measured, so there is nothing to tune to reach that number.

### Korean: the recommendation reversed

Until this revision the card said **"do not turn speculative decoding on for Korean,"** on the strength of Korean cells that read below plain (MTP 34.3, DSpark 33.3 against 37.6). **That recommendation was an artifact of the same broken protocol** — short answers plus a harness that measured past their end — and it does not survive the corrected one. EOS-cut and long-form, the gated MTP configuration is **faster on Korean than plain decode**: +34% greedy, and **+27–31% under the shipped sampling defaults** (47.2 vs 37.2 tok/s at 240 tokens; 48.4 vs 36.9 at 1024). The gate is what changed the verdict: where Korean drafting is uncertain it degenerates into near-plain steps instead of paying for rejected chains, and the confident stretches still collect.

One honest exception: **DSpark stays at or below plain on Korean** — below plain greedy, well below at 240 sampled tokens (26.7 vs 37.2), break-even at best by 1024 — so the reversal covers the gated MTP path, not speculation in general. Korean traffic through DSpark still loses; run gated MTP or plain.

### Measurement protocol, and why single-prompt numbers lie

Every figure in the tables is the **average over the fixed four-prompt set**, never a single run. Speculative decoding pays off in proportion to how predictable the continuation is, so a benchmark built from one favorable prompt **overstates the gain you will see in real traffic** by a wide margin — treat any single-prompt speculative figure, including mine, as an upper bound.

The correction above adds a second rule, learned at the cost of a retraction: **a speculative benchmark that does not stop at end-of-sequence is measuring luck** — specifically, what the model happens to do after its answer ends. This card's retracted 62.21 headline, and the 91.5 tok/s math cell behind it, were mostly that luck. The current protocol stops at EOS, uses long-form prompts whose answers genuinely fill the window, runs sampled rows three times and takes medians, and keeps stop-detection out of the timed loop. The per-cell records ship in the [campaign repository](https://github.com/avlp12/qwen38_alis_mlx).

## KV cache quantization

The hybrid layout is why long context is cheap on this model. Only the **16 full-attention layers** hold a growing KV cache; the 48 GatedDeltaNet layers carry a fixed-size recurrent state instead. With 4 KV heads at head_dim 256, that comes to **64 KB per token** in bf16 across the whole model.

Measured at 16K context on this repo's earlier uniform 4-bit release — same size, same layout — as whole-process peaks, so they include activations and prefill buffers, not the cache alone:

| KV precision | Peak RAM (GB), this build | Peak RAM (GB), 8-bit build | top-1 agreement |
|---|---|---|---|
| bf16 | 20.80 | 34.46 | 100% |
| 8-bit | 20.30 | 33.96 | 100% |
| 4-bit | 20.03 | 33.69 | 100% |

**The verdict: quantize the cache only for long context.** At 16K the whole exercise buys 0.77 GB and costs roughly 3% of decode speed. Quality was untouched — top-1 agreement held at 100% even with a 4-bit cache, on the build with the least margin to spare — so the trade is safe, it is simply not worth much at this length.

The arithmetic changes completely at the model's full context. Cache alone at 262K runs **16.8 GB in bf16 against 4.2 GB at 4-bit, a 12.6 GB saving.** That is the difference between fitting and not fitting, and it matters most here: if you chose this build to reach long context on a small machine, quantize the cache as well, because the weights you saved will otherwise go straight back out the door as cache.

```bash
mlx_lm.generate --model avlp12/Qwen3.8-27B-Alis-MLX-4bit \
  --kv-bits 4 --prompt "..." --max-tokens 512
```

**Note:** `--kv-bits` exists on `mlx_lm.generate` only. `mlx_lm.server` has no such option, so a served deployment gets the bf16 cache whether you want it or not. On a 24 GB machine that is the constraint that will bite first — the measured 16K peak here is already 20.80 GB with a bf16 cache.

## Quantization recipe

Produced with **activation-aware quantization (AWQ)**: per-channel scales are fitted against activation statistics from 128 calibration samples of 512 tokens each, then the language model is quantized with affine quantization at **4 bits, group size 64** — 498 language-model projections plus the 8 linear projections of the MTP head, **506 quantized paths in all, every one of them 4-bit g64**, so the Hub precision badge matches what actually loads. The effective width is ≈4.5 bits per weight; the extra half bit is the fp16 scale and bias each group of 64 weights carries. AWQ here is forward-pass only, which is what makes it tractable on this hybrid GatedDeltaNet stack.

Held at bf16, deliberately:

- **the entire vision tower** — all 333 `model.visual.*` tensors, byte-identical to upstream, with no `.scales` or `.biases` anywhere in the index
- every RMSNorm, including `q_norm` and `k_norm`
- the GatedDeltaNet recurrent-state parameters: `A_log`, `dt_bias`, and the short `conv1d` kernel

The state parameters are the deliberate part on the text side. They set the decay and gating of a state that carries the entire sequence, so a coarse grid there would cap quality for every token downstream — and they are tiny, so bf16 costs essentially nothing in file size.

The vision tower is skipped for a different reason. At 0.92 GB it is 6% of this build's 15.4 GB, so quantizing it would save almost nothing while adding an unmeasured error term to a path I had no evaluation harness for — and on a 4-bit build that error would stack on top of a language model already carrying the set's largest quantization loss. Preserving the original bytes means the image path in this repo is bit-identical to upstream no matter how far the text side is pushed.

### The graded allocation (2026-08-22)

AWQ fixes *how* each weight is rounded; it leaves *how many bits each tensor gets* alone. Five
tensor groups turned out to be badly served by one uniform width, and moving them is nearly free:

| tensor | width | why |
|---|---|---|
| `self_attn.k_proj`, `self_attn.v_proj` | 4 → **8** | under GQA these are **0.33% of parameters each** — protecting them costs almost nothing |
| `linear_attn.in_proj_a`, `in_proj_b` | 4 → **8** | the GatedDeltaNet gates, 0.04% each, and the most damaging tensors per byte in a direct measurement |
| `lm_head` | 4 → **5** | an error here lands on the logits with no downstream normalization to absorb it |

Total cost **+1.6% of file size**. Measured against the previous revision of this repo, paired
against bf16 on the same windows: **−16.6% KL on English (t = −30.8), −16.4% on Korean
(t = −37.1), −3.9% on code (t = −7.1)**, and +0.9 pp top-1 agreement.

The control that makes this claim survivable: spending the *identical* 248 MiB on twenty-three
arbitrary mid-depth FFN tensors instead bought **−1.2%**, byte-matched to within 0.02%. The gain
is the choice of tensors, not the spending. Method and the full sweep:
[docs/bit-allocation.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/bit-allocation.md).

The 6-bit and 8-bit builds of the set now carry the same idea, adapted: 6-bit promotes the same
five groups (the head to 8-bit), and 8-bit — where there is no wider integer width in MLX's
affine set — halves the **group size** on them instead, at +0.04% size.

**What AWQ buys over the uniform 4-bit this repo used to host.** On corpus-scale perplexity (next section) the AWQ recipe is significantly better on all three slices: its excess NLL against bf16 shrinks by 0.0113 (en), 0.0097 (ko), and 0.0128 (code) nats/token, which recovers 48.7% / 26.9% / 14.7% of the uniform build's gap to the 8-bit tier — at the same 15.2 GB and the same plain decode speed (37.7 tok/s re-measured on these weights, against 37.5 uniform). The MTP head, which the AWQ converter would have left in bf16, is quantized to the same 4-bit g64 the uniform build used, so both speculative paths run under the same conditions as before. The uniform weights this build replaced remain in the repository's git history, and the pre-graded AWQ weights on the `pre-graded` branch. The AWQ gain grows as bits shrink, and 4-bit is where it is worth shipping.

## Quality — read this before deploying

Corpus-scale strided perplexity against fixed token streams: window 2048, stride 512, so every scored token sees up to 2048 tokens of left context. Three slices — **English** = wikitext-2 test (35,981 tokens), **Korean** = Korean Wikipedia (35,179 tokens), **code** = CPython standard library (32,295 tokens). Every build scores the same tokens in the same order, so build-to-build comparisons are paired per token, and significance is judged on the paired excess NLL with a blockwise standard error (512-token blocks) — roughly 16–20× the sensitivity of the probe this section used to carry. The AWQ calibration corpus is none of the three slices.

| Build | en · wikitext-2 | ko · Korean wiki | code · CPython |
|---|---|---|---|
| bf16 reference | 5.7734 | 6.0954 | 1.6813 |
| 8-bit | 5.7760 | 6.0987 | 1.6815 |
| 6-bit | 5.7924 | 6.1018 | 1.6854 |
| **4-bit AWQ (this repo)** | **5.8450** | **6.2609** | **1.8105** |

Read against bf16, paired: the 8-bit build is **statistically indistinguishable** from bf16 on every slice; the 6-bit build carries a small but significant excess on English and code and is indistinguishable on Korean; **this build is significantly above bf16 on all three** — +0.0123 nats/token on English (+1.2% PPL), +0.0268 on Korean (+2.7%), +0.0740 on code (+7.7%).

† **The perplexity table above was measured on the pre-graded weights and has not been re-run.**
The graded revision is strictly better on the exact-KL sweep below (−16.6% / −16.4% / −3.9% by
slice, paired, t ≥ 7), so these PPL figures are an upper bound on this build's cost — but they are
not this build's own numbers, and they are labelled rather than quietly reused.

**The Korean cost is real, and roughly twice the English cost** — that ratio is why the warning below survives the change of metric. Code, which the old probe scored a flat 100 on every build, in fact carries the largest relative excess of the three slices at this precision. What this build has over the uniform 4-bit it replaced is that the damage shrank everywhere: excess NLL down 0.0113 / 0.0097 / 0.0128 nats/token (en / ko / code), recovering 48.7% / 26.9% / 14.7% of the uniform build's gap to the 8-bit tier, significant on all three slices.

Two consequences, stated plainly:

- **Do not ship this build into a Korean-facing product without your own evaluation on your own prompts.** That is the recommendation, not a hedge. The 6-bit build's Korean PPL is statistically indistinguishable from bf16 for 6.3 GB more, and that is the trade to make when Korean output is user-visible.
- **Perplexity prices average next-token surprise, not task success.** +2.7% Korean PPL is a much gentler statement than the retracted "one token in seven differs", but it is a corpus average — spot-check the workloads you actually care about.

**Why this table changed.** Earlier revisions of this card ranked the builds with a top-1 agreement probe over three short passages (62–112 tokens). At that size the probe had no statistical power: its standard error ran several times the effects it claimed to rank, and on English it read the sign of the AWQ-vs-uniform difference **backwards**. It is removed rather than restated.

### The exact-KL tier sweep

A follow-up sweep measured **exact full-vocab KL divergence to bf16** (248,320-way softmax in fp32, no top-K truncation; paired non-overlapping ctx-2048 windows over ≈100K en+ko+code tokens; 512-token block SEs) across ten builds, including mlx-community's artifacts.

| build | GB | KL to bf16 | top-1 agree |
|---|---:|---:|---:|
| **4-bit AWQ + graded (this repo)** | 15.41 | **0.0592** | **92.4%** |
| 4-bit AWQ, pre-graded (`pre-graded` branch) | 15.17 | 0.0649 | 91.3% |
| uniform 4-bit (= mlx-community 4bit) | 15.17 | 0.0752 | 90.5% |
| orcarouter 4-bit | 14.95 | 0.0754 | 90.3% |
| nvfp4 (mlx-community) | 15.01 | 0.0962 | 89.0% |
| mxfp4 (mlx-community) | 14.21 | 0.1437 | 86.4% |
| 6-bit (this collection) | 21.88 | 0.0050 | 97.5% |

Four readings. First, the community 4-bit turned out to be **byte-identical** to my uniform 4-bit build (verified on raw tensors — affine gs64 quantization is deterministic), so that row doubles as the community baseline and the AWQ margin (−14.3% KL, +0.7 pp top-1) is a like-for-like recipe comparison. Second, the fp4 formats are dominated at this size point: at essentially the same disk footprint nvfp4 carries 1.47× this build's KL and mxfp4 2.2× — fp4 exists for hardware with native fp4 matmul units, which Apple silicon does not have. Third, the honest framing of this build's tier: +6.5 GB to the 6-bit build divides KL by ~12.

Fourth, on a like-for-like measurement `orcarouter/Qwen3.8-27B-MLX`'s 4-bit build lands at 0.0754 —
statistically indistinguishable from the uniform baseline (+1.1% / +0.8% / −0.2% by slice, paired),
which is what its config says it is: uniform 4-bit, group size 64, no per-tensor overrides. Its
published card reports 0.02824 KLD on WikiText-2; that figure is reproducible only on a **single
1024-token window** — re-running their own protocol over 35 windows of the same corpus puts the
same weights at 0.0443, and this build at **0.0290**. Different rulers produce different numbers;
the comparison above puts every build on one. Chart, per-slice numbers, and method: [docs/kl-tiers.md](https://github.com/avlp12/qwen38_alis_mlx/blob/main/docs/kl-tiers.md).

![tier chart](https://raw.githubusercontent.com/avlp12/qwen38_alis_mlx/main/docs/assets/tier_chart.png)

## Limitations and caveats

- **High reasoning effort thinks long by design, at every bit width.** I tested the report that 4-bit quantization causes runaway thinking: on hard prompts at `reasoning_effort: xhigh`, thinking runs 4k-16k+ tokens on this build *and* on the 8-bit control at statistically indistinguishable rates (25.0% vs 20.8% of runs hitting a 16k cap, paired protocol), with no repetition pathology in either. Budget `max_tokens` generously for xhigh or use a lower effort; quantization is not the variable.

- **Korean carries roughly twice the English quality cost.** See the quality section; this build's defining limitation, and the reason the 6-bit build exists.
- **Perplexity and full-vocab KL to bf16 are the only quality metrics here.** Both are corpus-scale (≈100K paired tokens) and they replaced a probe that had no statistical power — but they price distribution fidelity, not task success.
- **No standard benchmarks were run.** There are no MMLU, GSM8K, or HumanEval numbers here, and none should be inferred from the perplexity table.
- **Vision quality is unquantified.** The tower is preserved bit-exact and verified to work on a shapes probe. That check says the image path functions; it says nothing about how a 4-bit language model reasons over what the tower feeds it. Evaluate your own image workload here specifically.
- **Shape-dependent numerics.** The hybrid stack takes different kernel paths depending on the shape of the input it receives — the batched scoring pass behind the perplexity table is not the same shape as incremental decode, and near-tie positions can resolve differently between those paths. Perplexity is far less sensitive to this than an argmax metric, but do not expect bit-identical numbers from a different harness.
- **Long-form generated reasoning is unprobed.** Perplexity scores 2048-token windows of existing text; it does not exercise long generated reasoning chains, where low-bit damage tends to concentrate. This is the build where that gap is most likely to bite: spot-check your own long thinking-mode prompts before trusting it on them.
- **Speculative-decoding gains are workload-dependent.** The speculative tables were restated on 2026-08-16 under an EOS-cut protocol after the earlier harness was found to measure past end-of-sequence; the correction in the MTP section is the full account.

## License and attribution

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). This repository redistributes quantized weights derived from that checkpoint; all model credit belongs to the Qwen team. Quantization, vision-tower preservation, and measurement by [avlp12](https://huggingface.co/avlp12).
