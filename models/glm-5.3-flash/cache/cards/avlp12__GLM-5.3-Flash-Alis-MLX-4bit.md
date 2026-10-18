---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- mixture-of-experts
- 4-bit
pipeline_tag: text-generation
library_name: mlx
---

# GLM-5.3-Flash-Alis-MLX-4bit (QUASAR-init)

MLX (mlx-vlm tree) 4-bit-class build of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B-A18B, `glm5_next`: 34 KDA linear-attention + 11 DSA layers, 288-expert MoE, mHC), converted by
streaming dequant of the official FP8 release (`84c6a6aa`) — never re-quantized from a packed lattice.

**This replaces the withdrawn earlier build.** The withdrawn build quantized the MoE router and served
at ≈5.5 tok/s; this build keeps the router (`mlp.gate`) and correction bias unquantized/fp32, decoded
at ≈29 tok/s at first upload (B=1, prompt 512, greedy, stock Metal buffers) on an M3 Ultra 512GB — current
figures are in the update section just below — and carries its conversion receipts (`state.json`,
`finalize_receipt.json`) in-repo.

## Update 2026-09-08 — two-box tensor-parallel speculative decode: 76 tok/s on the natural panel (+37%); prefill +4–5% from a bit-identical gather path; corrected ceilings

Everything below this section is unedited. The 2026-09-06 section's "Two-box decode" correction (TP=2 with the drafter at 65.5 tok/s, below single-box) is itself corrected in item 1. All numbers here were measured on my 5-bit-KDA build (GLM-5.3-Flash-vlm-q4-quasar-kda5) on the same two M3 Ultra 512GB boxes; the files in this repository are still the 8-bit-KDA build. The serving branch is [`glm5-serve-unified`](https://github.com/avlp12/mlx-vlm/tree/glm5-serve-unified) at [`95bbe594`](https://github.com/avlp12/mlx-vlm/commit/95bbe594); item 1 and the 16,384-step row were measured there, the prefill table on its parent `552ca195` (the difference is batched-capture fixes only), the batched rows on the branch that became it (the plain-greedy row on the head both branched from), and the discarded levers of item 5 each on their own branch or, for the two tile levers, on a custom MLX wheel.

**1. Two-box tensor-parallel speculative decode, single request (B=1), greedy.** The DFlash2 verify step (a width-8 batch through the full model every round) is split across the two boxes, the drafter stays on the first box, and the 101 collectives per step ride over Thunderbolt. Four natural prompts, 1,024 generated tokens each, temperature 0 (the single-box column is re-measured in the same pair, so it differs slightly per prompt from the 2026-09-07 headline while keeping the same 55.6 mean):

| prompt | single box | two-box TP=2 | accepted per round (single → TP) |
|---|---:|---:|---:|
| code (rate limiter spec) | 32.9 tok/s | 47.6 tok/s | 2.02 → 2.13 |
| prose (lighthouses, summarize) | 49.8 | 66.3 | 3.48 → 3.34 |
| csv (sales group stats) | 79.2 | 108.1 | 6.41 → 6.31 |
| math (multi-part) | 60.4 | 82.5 | 4.57 → 4.48 |
| **mean** | **55.6** | **76.1 (+37%)** | 4.12 → 4.06 |

A speculative round emits several tokens per weight pass, so this sits above the greedy roofline quoted in the 2026-09-06 section, not against it. The gain is in the round, not the drafter: at matched acceptance the round falls from 92.2 ms to 66.5 ms (−28%), a little more than halving the weight bytes alone predicts, so the collectives cost under 5 ms per round. Plain greedy does not benefit (38.0 single vs 37.2 two-box) because sharding halves the 11.5 GB a greedy step reads — worth about 7.9 ms — and the collectives cost about as much. On the p512/gen64 diagnostic (the repeated-code prompt) TP=2 gives 66.1 tok/s (64.5 in an independent repeat) against 59.7 (my 59.5 canon, re-measured in this pair) with acceptance dropping from 3.92 to 3.27 — the same collapse, to the same value, that made me dismiss two-box decode on 2026-09-05 (4.82 → 3.27 on the build I ran then). That dismissal was wrong: it rested on this one prompt, whose acceptance sits on a knife-edge, while the natural panel keeps 98.7% of single-box acceptance.

Why the text is not byte-identical and how I gated it: the `all_sum` accumulation order flips near-tie argmax choices, so the generated text differs from the single-box run on all four prompts. I therefore gated it on quality rather than identity: teacher-forced on the single-box run's own 256-token greedy continuation, full-vocabulary log-softmax on both sides, scored on the plain decode step and on a 9-token verify block (a fixed-width stand-in for DFlash2's acceptance-dependent block), the two-box distribution sits at a mean KL of 0.009–0.012 nats on the code prompt and 0.0001–0.004 on the other three, against the 0.042-nat cap I use for every numerics-changing change; top-1 agreement is at least 0.976 on every prompt. Widths other than 8 do not help: width 6 loses 6% with acceptance collapsing on three of four prompts, and the checkpoint's drafter clamps anything above 8 back to 8.

Caveat: two-box mode with the native MTP drafter is not usable yet — my first attempt de-synchronised the second box (the MTP verify bypassed the mirroring layer) and left the first box spinning on a closed socket; the fix (mirror the verify, have rank 0 listen for the peer's exit and unwind before the next collective) is on [`claude/tp-eof-mtp-desync-0908`](https://github.com/avlp12/mlx-vlm/tree/claude/tp-eof-mtp-desync-0908); it passes its CPU tests but is not yet verified on hardware. Until it lands, DFlash2 is the only drafter I run in two-box mode. The setting is opt-in (the TP host list; unset = single-box as always) and the second box's shard is launched over ssh by the first.

**2. Prefill on the served path, default configuration.** The DSA sparse-attention gather now uses a front-axis `take` instead of `take_along_axis` on broadcast views (the old form lowered to a per-element gather kernel running about 5× its write-only bandwidth floor). Logits are bit-identical to the previous path (the sha of the logits fingerprint matches on every arm and length), so this landed as the default with no quality gate needed. Direct model calls gain 5.4–5.5% at 8k and 32k; on the served path:

| prompt | single-box, previous | single-box, now | two-box pipelined, previous | two-box pipelined, now |
|---|---:|---:|---:|---:|
| 34,816 tokens | 74.1 s (470 tok/s) | 71.0 s (490 tok/s) | 52.4–53.2 s (655–665) | 50.6–53.2 s (654–689, mean 672) |
| 133,120 tokens | 297.0 s (448 tok/s) | 285.3 s (467 tok/s) | 171.6–171.8 s (775–776) | 164.7 s (808) |

At 34,816 tokens one of the two pipelined lifecycles is unchanged from the previous band and the gain rests on the other; at 133,120 both lifecycles moved. Text is identical between the head-only and pipelined arms at both pipelined lengths in each band (the third prompt is below the 16,384-token floor and runs single-box on both arms); the 64-token completions are all thinking tokens, so the hash compared is the reasoning stream. A larger prefill step (16,384 instead of 8,192 tokens) adds a further 1.7% at 133k on direct calls at 203 GiB peak; it changes the chunk boundaries, so it stays a candidate until it passes the same KL gate.

**3. Batched speculative serving.** With concurrent requests the drafter batch now grows while it runs (a new request joins the active speculative batch instead of waiting for it to drain) and prefill of the newcomer is interleaved in chunks (the growth is what pays here; the interleaved prefill fired only twice in this sweep). Aggregate throughput over N concurrent requests drawn from my four natural prompts (462–670 prompt tokens, 256 generated each), DFlash2 width 8, greedy:

| N | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| before (drain-then-admit) | 37.0 tok/s | 55.1 | 53.2 | 56.5 |
| now (grow the active batch) | 36.2 | 61.5 | 65.1 | 76.4 |
| plain greedy, same harness and panel, measured separately earlier the same evening | 34.2 | 59.9 | 78.6 | 90.6 |

Mean time to first token at N=16 fell from 31.5 s to 18.0 s. Plain greedy still wins at N ≥ 8 because the speculative verify batch multiplies the per-step compute; the gap at N=16 is now 1.19×, from 1.6× before this change (and 1.8–2.9× on the build I replaced on 2026-09-03). Two correctness fixes rode along and matter to anyone using session capture with batched speculation: with more than one request in a speculative batch, a finished request's session snapshot could be taken from a neighbour's cache row (or from past the end of the cache), and every capture taken through the batched speculative loop — B>1, and the MTP drafter even at B=1 — was taken before the round's rollback, so it carried draft tokens that were never emitted; the snapshot is now taken after rollback and refused by name if it is still ahead of the emitted prefix.

**4. Two conventions I got wrong before, now corrected.** (a) A speculative round emits accepted + 1 tokens (the target's own bonus token), so the per-round cost is (accepted + 1)/tok/s, not accepted/tok/s; every "ms per round" I derived from tok/s before today was about 25% too low (the p512/gen64 diagnostic's round is 83 ms, not 66). Measured round timers on the first box (one timed run per cell, sitting about 10% below my canonical rail, so read the split rather than the absolute) show the cost is flat where it matters — the width-8 verify forward takes 77–79 ms regardless of how many tokens were accepted — with drafting at 1.3–1.5 ms per token and rollback at 2.9–5.0 ms; so the 70 tok/s single-box mark needs 5.5 accepted per round on the natural panel, and two-box mode reaches it at 3.65. (b) The GEMM-only prefill ceiling I quoted as ≈728 tok/s omitted the DSA attention FLOPs; recomputed from the checkpoint it is 682 tok/s at 35k and 662 at 133k on this box (MLX's bf16 GEMM already measures 24.4 TFLOP/s, 86.5% of this GPU's 28.3 TFLOP/s arithmetic peak at the 1.38 GHz it holds), so today's 490 tok/s is 72% of the roofline and the remaining gap is non-GEMM work, not GEMM efficiency. A single-box prefill of 1,000 tok/s would need 1.27× the GPU's entire theoretical FMA peak; it is not reachable by software on this hardware.

**5. What did not work this round** (all measured, all off by default or discarded): a fused sparse-attention Metal kernel (10× slower than the stock-op path above), deeper `qmv` tiles and a 16-row `qmm` tile for the skinny quantized GEMV chain the verify forward drives (micro-benchmarked at M=9: 1.9× slower and +20% respectively), a chunk-parallel formulation of the KDA prefill scan from stock ops (−7% against the eager scan, −16% against the fused kernel: it multiplies traffic 24× and buys only rate), a fused decode-time router kernel (−3%), and running the KDA scan concurrently with the previous chunk's MoE on two MLX streams (Metal does not overlap them). An opt-in fused decode indexer for the DSA layers (+1.3 to +2.7% at 8k context, text identical) is in the branch but off by default.

## Update 2026-09-07 (evening) — two-box pipelined prefill runs in the served path (first numbers on the default serving configuration); prefill defaults landed; one diagnostic correction

Everything in the section below this one (also dated 2026-09-07, measured earlier the same day) is unedited; the one figure it reports that I now know to be wrong is corrected in item 3 below. This section adds what changed later the same day, on the same two boxes. Every number in this section was measured on my 5-bit-KDA build (GLM-5.3-Flash-vlm-q4-quasar-kda5); the files in this repository are still the 8-bit-KDA build, as the section below explains.

**1. Two-box pipelined prefill, HTTP path — confirmed over two lifecycles each, in ABBA order.**

| prompt | single-box | pipelined | input tok/s | TTFT change |
|---|---:|---:|---:|---:|
| 34,816 tokens | 74.13 s (470 tok/s) | 52.4–53.2 s | 470 → 655–665 | −28 to −29% |
| 133,120 tokens | 297.04 s (448 tok/s) | 171.6–171.8 s | 448 → 775–776 | −42% |

Generated text is identical to the single-box run — these 64-token completions are all thinking tokens, so the hash compared is the reasoning stream — and it is identical on both pipelined prompts in both lifecycles (4 of 4 comparisons); the third prompt sits below the 16,384-token floor and ran single-box on both arms. Prompts below 16,384 tokens stay single-box by default (the 16,384-token floor is a setting). These are the first paired numbers on the default serving configuration (prefix cache and prefix-disk vault on, DFlash2 drafter on, 8,192-token chunking). A pipelined request also forgoes the intermediate vault checkpoints of the prefix (2 rungs at 35k, 4 at 133k are skipped), which only matters for later warm hits on the same document. The pair is confirmed: two lifecycles per arm in ABBA order; the two lifecycles agree within 0.9 s.

The older Speed-section prefill rows "Prefill ≈450 tok/s (2k–8k plateau; 390 @16k, ≈350 @32k, ≈160 @131k dense)" and "131k 295 tok/s (+88%)" were already superseded by the Speed table for the direct model-call path; today's single-box figures on the HTTP path are 470 tok/s at 34,816 tokens and 448 tok/s at 133,120 tokens. The Speed table's "411 @ 131,449" is the direct model-call number; the 448 tok/s at 133,120 tokens above is the HTTP-served path on today's defaults — the two are different surfaces, not a contradiction.

How it works: the head box owns layers 0–22 and does all decoding, while the tail box runs layers 23–44 during prefill only. The tail gets ceil(T/C)−1 chunks of 8,192 tokens, so it covers 94% of the 35k prompt (4 of 5 chunks) and 98% of the 133k prompt (16 of 17); the shorter prompt gains less mainly because a 4-chunk pipeline spends a larger share of its time filling and draining. The tail's KV plus the drafter's context window come back to the head over Thunderbolt at the end (≈2.1 GB at the 133k-token point), and the head→tail boundary activations sent chunk by chunk are a further ≈4 GB at that length; the wire time for all of it is ≈1.7% of TTFT at 133k (≈1.3% at 35k). It is opt-in via the pipeline host setting (unset = disabled, single-box as always); by design any peer fault is caught and the request re-prefills single-box; I have observed the peer-absent case (a dedicated drill: the request completed single-box with a peer_unreachable bypass), and the mid-request peer-loss drill is being re-run after a shutdown fix.

**2. Prefill defaults landed** (on by default now, no opt-in needed):
- unchunked-prompt lm_head trimming plus short-tail chunk merging: −2.3 to −2.4% at prompts ≤8k tokens, −1.7% at 9k, −0.6% at 33k
- fused hyper-connection Metal kernels: −3.0% at both 8k and 32k

Quality: teacher-forced KL against the previous defaults is ≤0.030 nats on every prompt in my test set, against a 0.042-nat cap, top-1 agreement 0.91–0.95 on the four long cold prompts (0.99–1.00 on the natural panel) — the same band the already-shipped chunked prefill scores when checked against an unchunked reference. On my natural-prompt panel, generated text is identical to the previous defaults for the first lever; for the second lever it is identical on 2 of 4 prompts, with no systematic change in acceptance rate.

**3. Correction to a previously posted diagnostic.** The DFlash2 p512/gen64 diagnostic row (the repeated-code prompt, 64 generated tokens) read 52.6 tok/s in the update section just below; three independent runs today land at 57.1–59.7 tok/s, and the controlled code A/B among them at 59.4–59.7; the figure I now use for that diagnostic is 59.5 tok/s at 3.92 accepted per round. A code A/B in the last of them ruled the code out, and the earlier 52.6 came from the same code and the same settings. The headline natural-panel figure re-measured today at 55.0 tok/s, ≈1% below the posted 55.6 (54.8–55.0 over two lifecycles).

**4. Branch pointer.** The branch I serve from, `glm5-serve-unified`, is now at [`eb97b60c`](https://github.com/avlp12/mlx-vlm/commit/eb97b60c) (the fork's default branch is `main`). The two-box pipeline above lives on [`claude/l38-pp-serve-rebased-0907`](https://github.com/avlp12/mlx-vlm/tree/claude/l38-pp-serve-rebased-0907); the numbers in the table above were measured at [`ccb1cec1`](https://github.com/avlp12/mlx-vlm/commit/ccb1cec1); the branch has since moved to [`09900917`](https://github.com/avlp12/mlx-vlm/commit/09900917) (a chunk-scheduling fix that does not fire in this configuration), pending merge.

## Update 2026-09-07 — a 5-bit-KDA build now serves locally (not uploaded here); prefix vault on by default

**The files in this repository are unchanged — they are still the 8-bit-KDA build.** The 5-bit build
in this section is what I serve locally; it is not uploaded here, and where it matters I give the
8-bit number beside it. Measured on the same two boxes since the 2026-09-06 section below; the
serving branch is now at [`a6634a75`](https://github.com/avlp12/mlx-vlm/commit/a6634a75) of
[`glm5-serve-unified`](https://github.com/avlp12/mlx-vlm/tree/glm5-serve-unified); every commit named
in the rows below is an ancestor of it. Each row names the commit and box it was measured on (the second box runs plain mlx 0.32.1, so its absolute numbers
are not comparable to the first box's — only same-box ratios are). Nothing below this section was edited.

| surface | 8-bit build (this repository) | 5-bit-KDA build (served locally) | conditions / what changed |
|---|---:|---:|---|
| **build** | 4-bit experts, 8-bit KDA projections | the 7 KDA projections `q/k/v/o/b/g_a/f_a` on the 34 linear-attention layers at 5-bit (238 keys); the `f_b`/`g_b` gate projections stay at 8-bit, which the fused KDA kernel requires; experts and everything else unchanged | held-out KL(teacher‖student) **0.0339** nats on my 25-window joint-KL panel, inside the 0.042075 cap set by the 8-bit build's own bootstrap CI on that panel (the 6-bit variant I rejected reads 0.0337). Same FP8 source, same streaming recipe (RTN for the KDA projections). Not comparable to the Quality table below |
| decode, B=1, greedy, prompt 457 tokens | 34.3 tok/s | **37.07 tok/s** (median of three lifecycle medians, 37.00–37.08; nine runs span 36.96–37.15) | first box, commit `d1a3c3a4`, served path, prefix cache and vault off for this cold measurement. A same-commit A/B of the two builds gives **+7.8%** (that A/B is on the second box at commit `943b0cb0`); the +8.1% against the 2026-09-06 row also spans a code change |
| speculative (DFlash2, width 8), B=1, greedy — **new headline** | 54.9 tok/s mean on the same panel | **55.6 tok/s mean on four natural prompts at 1,024 generated tokens** (32.3 / 51.5 / 78.4 / 60.0; 2.0 / 3.7 / 6.4 / 4.6 accepted per round) — equal within run-to-run spread | first box, `d1a3c3a4`, cold. I moved the headline off the repeated-code prompt: that figure is one deterministic 13-round trajectory, and the same build's number moved by 30% under a numerics-only change (4.82 → 3.27 accepted/round, 68.3 → 47.7 tok/s, from a compile flag alone). On that prompt the 5-bit build reads 52.6 tok/s median at 64 generated tokens (3.92 accepted/round) but **overtakes the 8-bit build at 256 and 1,024** (3.83 vs 3.06 and 4.30 vs 2.00 accepted/round, on both boxes), and on three other repeated-unit prompts it accepts more on four of six arms, ties on one and loses one (the maths unit at 64 tokens). The repeated-code figure stays on this card as a diagnostic. On three of these four natural prompts the model is still inside `<think>` at the 1,024-token cap — see the budget row |
| native MTP drafter, B=1, greedy, prompt 457 | 42.0 tok/s (1.38 accepted/round) | **46.5 tok/s (1.56)** | first box, `d1a3c3a4`, same protocol; the 5-bit build accepts more from the MTP head too |
| **thinking budget with speculation** (new surface) | budget forced the drafter off, and on this template the budget never counted | **budget honoured with the drafter on**: at `thinking_budget` 256 the reasoning block closes one to two tokens past the budget (258 chunks greedy, 259 with the drafter) and the answer follows; speculation keeps 1.0–1.9× over greedy under the budget (four natural prompts, 1,024-token cap); a JSON-schema request with a 128-token budget yields the same JSON as the no-thinking structured run on 3 of 4 prompts at 1.75–2.4× — on the fourth (meeting notes) the 128 thinking tokens changed the answer, and the greedy and speculative budget arms differ from each other there too | second box, 8-bit build; the 256-token-budget arms on the pre-rebase thinking-budget branch, the JSON-schema arms at commit `f33c6587` — both are in `d1a3c3a4` now. Two server defects fixed: the CLI wrote `MLX_VLM_ENABLE_THINKING=0` into its own environment on every launch, so on a template that opens `<think>` itself the budget never armed; and speculative acceptance under a cap was derived from verified rather than emitted tokens. Without a budget this model thinks past a 1,024-token cap on 3 of 4 natural prompts, so the budget is what makes an answer arrive. Request field `thinking_budget`; no budget by default |
| **prefix vault on disk** (new surface) | off | **a 3.8 GB, 131k-token entry: 0.68 s of disk restore at 5.6 GB/s inside a 1.35 s TTFT; 32k in 0.52 s; text identical to the RAM hit; survives a server restart** | second box, branch `7debfbc1` (merged into `d1a3c3a4`), **8-bit build**. A fully cold 131k prefill on that box measured 302 s in an earlier run (a 131,084-token prompt, not this one); after eviction this prompt restores from disk in 1.35 s (measured in a run whose own cold arm was 229 s because a 32k rung was already on disk). Found and fixed a loader defect on the way: entries above ≈2.1 GB (≈75k tokens) could be saved but not restored (32-bit shape limit). The checkpoint-ladder RAM vault (48 GB budget) and the disk tier (`$HOME/glm53flash/vaultdisk`, 200 GB cap) are now on by default; opt out with `MLX_VLM_GLM5_VAULT=0` / `MLX_VLM_VAULT_DISK_DIR=""`. The vault-on cold prefill measured 460 tok/s at 32k, in the same band as the 445 tok/s I measured without it on another day — I have not run a paired vault-on/vault-off prefill |
| **multi-turn reuse of the previous turn** (new surface) | turn 2 silently dropped the previous turn's reasoning before templating — fast (0.17 s) but blind; once the reasoning was carried, it was re-prefilled instead (2.5 s) | **turn 2 carries the previous turn's reasoning and serves 99.3% of its prompt from cache** (1,524 of 1,535 tokens; 1,323/1,332; 1,660/1,671; 1,481/1,492), **TTFT 0.16–0.18 s** — against 0.72–0.98 of the prompt on the shipped build, which was only that fast because it had thrown the reasoning away | second box, commit `a6634a75`, 8-bit build, greedy, 1,024-token cap per turn. Four defects fixed on the way: the reasoning/answer split stripped whitespace; a client-supplied `reasoning_content` was dropped before templating; the session cache stored the model's generated token ids but the next turn arrives as re-tokenised text, which BPE does not map back to the same ids at merge boundaries (bridged by matching in text space and splicing the stored ids); and the session capture ran after the finished row had already left the batch. The turn-2 answers read the previous turn correctly — one even noted that its earlier response had been cut off mid-way. On by default; opt out with `MLX_VLM_APC_SAVE_SESSION=0` |

In the 8-bit column only the 54.9 tok/s figure was measured beside the 5-bit run; 34.3 is the
2026-09-06 row at commit `e6974827` and 42.0 is my 2026-09-05 canon.

The multi-turn row is on the second box and reports cache ratios and wall-clock TTFT, not tok/s; the session tier postdates the commit the cold-rail rows above were measured at, so none of them is affected.

## Update 2026-09-06 — new measurements (everything below this section is unchanged)

The rest of this card, from "At a glance" down, is exactly as it stood on 2026-09-05; I have not
edited those figures. This section, placed first so it is read first, adds what I measured since on
the same two M3 Ultra 512GB boxes. The serving branch moved several times during these two days, so
each row names the commit it was measured at; the branch now stands at
[`943b0cb0`](https://github.com/avlp12/mlx-vlm/commit/943b0cb0) of
[`glm5-serve-unified`](https://github.com/avlp12/mlx-vlm/tree/glm5-serve-unified), and every
commit named below is an ancestor of it unless I say otherwise. Conditions are stated per row
because most of these numbers move with prompt, batch, sampling, buffer setting and code path.

| surface | earlier card figure | now | conditions / what changed |
|---|---:|---:|---|
| decode, B=1, greedy, prompt 457 tokens | 31.8–32.0 (default buffers) · 34.8 (2,048 MB / 100,000-op buffers) | **34.3 tok/s** (34.19–34.38 over nine runs) | served path (HTTP), 64 generated tokens, 3 server lifecycles × 3 timed runs, median of medians, commit `e6974827`. The branch now sets its own Metal buffers at **1,024 MB / 50,000 ops** on every entry point — neither the stock 50 MB / 50 ops nor the 2,048 / 100,000 of the earlier 34.8 row — with the fused KDA/QPROJ/IDX_FAST kernels on. This is the reference the rest of this section is paired against, not an improvement on 34.8. The rail's "p512" label is a character-count target; the prompt is 457 tokens |
| speculative (DFlash2, fixed width 8), B=1, greedy, prompt 457 tokens | 42.39 @ prompt 512 code (adaptive width) · 57–66 @ prompt 1,024 | **68.3 tok/s, 4.82 accepted/round (1.99× the greedy arm of the same protocol)** — read as the top of the range | same served path, buffers and protocol as the row above, commit `e6974827`; separate lifecycles, not an interleaved pair. The drafter is the incoai DFlash2 checkpoint served in **bf16** (I also hold an 8-bit requant; the rows in this section did not use it). This rail's prompt is one code request repeated to 457 tokens, and acceptance on this model is strongly text-path dependent: on **four distinct natural prompts** the same drafter and width average **54.2 tok/s at 2.2–5.1 accepted/round** (128 generated tokens, B=1, greedy, cold). Replaces the prompt-512 code figure (42.39, adaptive width); I measured no prose arm at this length under width 8, so the prose 60.25 row below stays as the adaptive-width figure it is |
| speculative with **sampling** (T=1, top-p 0.95) | — | **54.4 tok/s** panel mean (was 43.0 on the same prompts, +27%), **1.59×** the no-drafter greedy control of the same panel (34.3) | B=1, four natural prompts, 128 generated tokens, seed 7, cold, measured on branch `6557537c`; merged as `71732451` after a review round that added a top-k mask to the proposal sampler and made the coupled draw the default, so this figure is provisional until I re-measure the merged default. Cause: the drafter and the target sampled from independent RNG streams, so a sampled proposal was accepted only on a token collision; the draft is now coupled to the target with a shared-key Gumbel draw and a token-order top-p mask. Accepted/round 2.91 → 4.02. The gain is not uniform: two prompts moved +29% and +119%, two moved −1% and −3%. The emitted token is always the target's own draw, so the drafter cannot change the output distribution; I verified that by mutation-testing the round loop against a CPU stub target (random, oracle and no drafters give the identical stream) and with a χ² check on the emitted marginal, and I have **not** run a held-out quality panel on sampled speculative output |
| **structured output (JSON schema) with speculation** | refused (grammar and drafter were mutually exclusive) | **12.4–16.1 ms/token vs 30.0–30.3 ms/token constrained greedy — 1.9–2.4×, mean 2.2×**, output byte-identical | B=1, four JSON-schema extraction prompts of 69–179 tokens, 47–213 generated tokens, greedy, commit `943b0cb0`. The grammar mask is applied to the whole verify block, a grammar fast-forward drafts forced runs, and the drafter's candidates are masked; 3.9–5.6 accepted/round. Unstructured traffic is unaffected: with the rail on, the four natural greedy prompts return the same text and the same accepted/round (2.2 / 3.2 / 5.1 / 4.3) as the build without it. Default on in the branch; B=1 only — a batched structured request and a grammar-plus-penalty request still refuse |
| **warm prefix cache with a drafter** (new surface) | — (the ≈5 s @16k row below is the oMLX stack at a different length and stands as it is) | **TTFT 1.28–1.66 s → 0.17–0.18 s (−87% to −90%)** on repeated natural prompts of 457–665 tokens, acceptance identical to cold | exact-mode prefix cache, 441–649 cached tokens, B=1, greedy, commit `ddc19579`. Two fixes: the cache now stores the hidden-state tail the drafters need, so a warm MTP request no longer re-primes from the suffix only (its text now matches the cold run 4/4), and DFlash2 requests now go through the cache at all (they used to bypass it). The continuous-batching path this uses chunks cold prefill at exact checkpoints, which changed the cold text on 2 of 4 prompts (acceptance 2.20 → 2.34 and 4.33 → 4.57) |
| prefill, one box, gathered sparse path (gate 6,144 tokens), prefill step 8,192 | ≈450 (2k–8k) · 390 @16k · ≈350 @32k · 295 @131k | **443–447 tok/s @ 8,192 · 445 @ 32,768 · 411 @ 131,449** | direct model call, one generated token, greedy. The 8k and 32k points are one run each on the second box at commit `e6974827` with mlx 0.32.2; the 131k point is the median of two on the first box at `671aca13` with mlx 0.32.1, 319.8 s at a peak of 197.6 GiB. None of the three include the fused prefill kernel of the row below (it landed later; expected effect +1–2%). Throughput is within **405–457 tok/s from 7.5k to 33k** (±6% about the mean) with no fixed cost — time outside the forward passes is ≤ 0.7 s per call. Supersedes the prefill line below for this path |
| **prefill, two boxes, pipeline parallel** (new surface) | — (the card had TP=2 only) | **32,781 tokens in 42.25 s = 776 tok/s · 131,073 tokens in 174.3 s = 752 tok/s** (131k spread 713–754 over three pairs) | a two-stage layer split (23 of 45 layers on the head box), prefill step 2,048, so the prompt is pipelined in 17 chunks at 32k and 64 at 131k; the handoff is a plain TCP socket over the Thunderbolt-bridge IP link (132 MB of activations per request), **not** the jaccl RDMA path the TP=2 rows use. Direct model call, three fresh-cache pairs, one load per host. 1.79× / 1.90× against the same driver on one box at the same 2,048-token step, and **1.67×** at 32k against the best single-box configuration (step 8,192, 70.7 s). Decode on the pipeline is neutral (28.9 vs 28.9 tok/s at 32k, 26.9 vs 26.9 at 131k), and it does not cut per-box peak memory (190.9 vs 191.2 GB at 32k). This path lives on [`claude/pp05-source-0906`](https://github.com/avlp12/mlx-vlm/tree/claude/pp05-source-0906) (`812ae697`): the unified tree plus two pipeline-handoff commits I have not merged |
| fused KDA **prefill** kernel | (the fused kernel was decode-only) | **+1.2% at 15.9k tokens, +2.2% at 33.3k tokens of prefill**, logits bit-identical | chunk-carry variant of the same `mx.fast.metal_kernel`, default on (`NV=1`, commit `d17a97fe`); the 4-vector variant lost 2.5–3.8% and is off. Ratios measured under a concurrent archive transfer, so I quote them only as between-arm ratios |
| compute and memory ceilings | — | **GEMM-only prefill ceiling ≈728 tok/s · greedy decode ceiling ≈62 tok/s** on this box | bf16 GEMM peak 24.4 TFLOP/s measured; 4-bit `quantized_matmul` at M=8,192 reaches 23.0 TFLOP/s (94% of that peak, 99–101% of the same shape run in bf16), and the expert `gather_qmm` in the serving convention at a prefill step of 8,192 reaches 87% in the shipped configuration (91% with segment alignment on, which measures faster in isolation and slower end to end, so it stays off); 33.5 GFLOP per prefill token. 13.24 GB of weight and state traffic per decode token against 819 GB/s. Roughly a third of prefill wall-clock is outside the GEMMs, so the single-box prefill above sits at 62% of the GEMM-only ceiling from 8k to 33k and 56% at 131k; greedy decode sits at 55% of its ceiling |

Two corrections to rows that stand below this section:

- **Two-box decode.** Measured on the served path rather than the earlier bench harness, TP=2 decode
  is **34.8 tok/s against 34.5 single-box** — within noise, not +20% — and TP with the drafter (65.5)
  is below single-box speculative (70.6). Two-box value on this stack is prefill, not decode. The
  36.2 and +49% figures below are what that older harness produced and I leave them labelled as such.
- **Drafter precision.** The DFlash2 drafter behind every speculative row in this section is served
  in bf16. The two lines below that say I serve an 8-bit requant describe a build I hold but did not
  measure here.
- **Gather gate.** The fork's default gate for the gathered sparse prefill path is
  `MLX_VLM_GLM5_GATHER_MIN_CONTEXT = 6144` tokens at every commit named here, not the 32,768 or
  12,288 quoted below; the three prefill points above all ran the gathered path.

Things I tried on the same days that did **not** make it into the build, so that nobody repeats them
from this card:

- **6-bit KDA projections** (most of the 34 linear-attention layers' projection weights at 6-bit
  instead of 8-bit; the `f_b`/`g_b` gate biases stay at 8-bit; held-out KL **0.0337** on my 25-window
  joint-KL panel against that panel's 0.042 cap — a different panel and protocol from the Quality
  table below, so the two KL numbers are not comparable to each other): greedy decode +4.9%, but the
  DFlash2 drafter — fitted to this build's hidden states — accepted 3.57 tokens/round instead of
  4.82 on the canon prompt, so speculative decode fell 24% (measured at `943b0cb0`, a newer commit
  than the 68.3 reference, so not code-matched); on four distinct natural prompts the same build
  was 2 up and 2 down. Not adopted; the drafter would have to be re-fitted first. A **5-bit**
  variant of the same projections (KL 0.0339 on the same panel and cap) behaves differently:
  greedy decode +8.4% (26.8 vs 29.1 ms/token) **and** the drafter accepted more on all four natural
  prompts (2.6/3.9/5.4/4.6 vs 2.2/3.2/5.1/4.3 tokens/round; 2–14% lower ms/token). It is not on this
  card yet because its canon-prompt run is still queued; if it holds, it becomes the next build.
- **CPU co-compute for prefill** (a share of the MoE expert rows on the CPU cores while the GPU
  does the rest): monotonically slower at every share, −37% at a 10% row share, because the CPU
  work did not overlap the GPU work. Dropped.
- **ANE offload of the KDA projections**: coremltools placed the int8 block-quantized convolution
  on the CPU rather than the ANE, and even a free, fully successful offload of every statically
  shaped bucket was worth at most +1.2–2.3% by arithmetic. Dropped.
- **A hand-written `gather_qmm` Metal kernel** for the expert GEMMs (bf16, 2,048-token chunk):
  matched MLX to 6e-07 relative in fp32 and 0.7% in bf16 (inside a 2% gate), but **0.52×** the speed
  of MLX's own kernel at its best tile on both boxes. Dropped. A fused indexer-score kernel from
  the same branch is 1.63× on one box and 1.65× on the other in a micro-benchmark, selection
  identical, and still awaits a served A/B.
- Three served-path defects found and fixed on branches, not yet merged because their measurements
  are still queued: the server's CLI wrote `MLX_VLM_ENABLE_THINKING=0` into its own environment
  unconditionally, so a thinking budget never counted on this template (which opens `<think>`
  itself); the reasoning/answer split stripped whitespace, so a multi-turn request could not reuse
  the previous turn's thinking from the prefix cache; and, found while measuring that fix, a
  client-supplied `reasoning_content` on an assistant turn is not rendered into the next prompt at
  all. None changes any figure in this section.

One correction to the record: the prompt assembler in my prefill harness reported its **target**
token count while feeding the untruncated text, so my earlier "2k / 8k / 32k" prefill points were
actually 7,518 / 15,937 / 33,306 tokens (131k was 131,449, within 0.3%). Ratios and verdicts within
a run are unaffected because every arm shared the same text, but the overflow depended on the
working tree's file list, so two runs on different days at the same label are not comparable to
each other. The absolute prefill figures in this section were re-derived from the real fed length,
and the earlier "≈450 (2k–8k plateau)" line in the Speed section below should be read as "405–457
tok/s from 7.5k to 33k". The fix records the fed token count and the prompt's SHA-256 with every run.

As before, plain decode and prefill are token-identical between the stock and fused paths, TP and
speculative exceptions are the ones already footnoted, and the receipts behind every figure here are
in my private campaign notes rather than in this repository.

## At a glance — first upload → 2026-09-05 (historical: what changed since is in the update section above)

Single-box figures were measured on one M3 Ultra 512GB; TP=2 figures used two identical M3 Ultra 512GB
boxes over Thunderbolt. Unless noted otherwise, each **decode and speculative**
row states its batch, prompt length, decoding policy and Metal buffer setting. Plain decode and prefill were
token-identical in validation; TP and speculative exceptions are noted below.

| | first upload (stock) | as of 2026-09-05 | how |
|---|---:|---:|---|
| decode @ prompt 512, B=1, greedy | 29 tok/s | **34.8 tok/s** (+20%)⁴ | **raised Metal buffers** + fused-KDA kernel. At **default Metal buffers** the same path measures **31.84–31.98 tok/s** — the setting used by the single-box speculative rows below; the TP speculative row's buffer setting is not recorded |
| decode @ prompt 131k, B=1, greedy | 22 | **31.9** (+45%, raised buffers) · **26.48** (default buffers)⁴ | same kernel. Decode falls **8% (raised buffers) to 17% (default buffers)** from prompt 512 to 131k |
| prefill @ 131k | ≈157 tok/s | **295** (+88%)³ | PR [#2087](https://github.com/Blaizzy/mlx-vlm/pull/2087) gathered prefill + the query-chunk fix and re-tuned gate (fork branch) |
| prefill @ 65k | ≈257 | **314** (+22%)³ | same + a 2³¹ query-chunk boundary fix and a re-tuned gate (fork branch) |
| warm TTFT @ 16k | 42 s | **≈5 s** (8–11×) | oMLX tiered prefix cache; batch, prompt length and buffer setting **not recorded**, and no in-tree receipt |
| oMLX server decode | ≈33 | **35.4** | same kernel ported to the oMLX lane (+8.9% in-venv A/B); batch, prompt length, decoding policy and Metal buffer setting **not recorded** |
| code-gen speculative, B=1 | — | **57.14 tok/s** @ prompt 1024 (1.81× vs greedy 31.59)⁵ · **42.39** @ prompt 512 (1.33×, worst pair 1.3272) | DFlash2, **fixed verify width 8** — the shipped default — B=1, n=3 paired ABAB against greedy, **default Metal buffers**. Width 8 supersedes the adaptive-width **54.42** at prompt 1024; the prompt-512 figure is the **prior default (adaptive width)** and is retained because I measured no width-8 arm at 512. Earlier **50.8**/**38.8** and **45.9**/**39.8** figures carried no prompt-length or buffer label, and an earlier **54.7** is excluded because it used a repeated-token prompt |
| prose speculative, B=1 | — | **66.55 tok/s** @ prompt 1024 (2.11× vs greedy 31.60) · **53.91** @ prompt 16,394 (1.87× vs greedy 28.84)⁵ · **60.25** @ prompt 512 (1.88×, worst pair 1.8741) | same drafter, policy, batch and buffer setting as the row above; chat prompts at 1024 read **66.37** (2.09×). The prompt-512 figure is again the prior adaptive-width default, and **no prompt-length effect is claimed between it and the 1024 figure — they are different width policies**. Every multiple is paired inside its own session against that session's own greedy arm |
| speculative decode, batched | — | **1.04–1.06× greedy at B=4**, **0.86–0.95× at B=8**, prompt 512⁶ | batched speculative decoding no longer collapses. 8 distinct documents, one per row, fixed verify width 8, **prefix cache enabled in exact mode**, **default Metal buffers**, 192 tokens per row, 3 reps: aggregate **64.04** (prose) / **65.48** (code) tok/s at B=4 against greedy **61.66** / **61.61**, and **82.31** / **73.15** at B=8 against **86.41** / **85.60**. The preceding build ran **0.34–0.56× greedy** in the same cells |
| speculative prefill peak @ 16k / 32k / 65k | — | **196.7 / 196.6 / 196.9 GB** (was 230.8 / 266.6 / 345.9)⁷ | B=1, DFlash2, prompt-chunked speculative prefill at the server's default 2,048-token step, default Metal buffers, **one request per process, so the prefix cache is not exercised**, measured on the second of the two identical M3 Ultra 512GB boxes. Greedy in the same runs peaks at 195.5 / 195.9 / 196.2 GB |
| decode, TP=2 (2× M3 Ultra) | — | **36.2 tok/s** B=1 @ prompt 512; **+49% vs single box at B=8** @ prompt 512 (paired ratio only)¹ | B=1 greedy decode, prompt 512 natural text, 36.2 tok/s median of three, Metal buffer setting not recorded; megatron tensor-parallel over Thunderbolt RDMA, +20% at B=1 paired vs single box, **−48% peak memory per box** |
| code-gen speculative, TP=2 | — | **49.8 tok/s** (1.44× vs TP greedy)² | B=1 speculative, prompt 1,024 code, 49.8 tok/s higher of 49.69/49.84, TP-greedy baseline 34.65/34.59, Metal buffer setting not recorded; DFlash2 drafter on rank 0 + width-8 verify across both boxes (prose 45.7, 1.32×) |
| held-out quality | −8.5% KL vs 4-bit RTN | **unchanged** | plain decode and prefill token-identical in validation; TP=2 output is rank-consistent but not bit-reproducible vs the single-box lane (one-ULP `all_sum` rounding), and the speculative exception is noted below |

⁹ The cached-prefix comparison: one model load on an M3 Ultra 512GB, **all eight measured
continuations served alone (B=1), greedy, no drafter**, 32 generated tokens, warm prompt **3,107**,
prefix cache in **exact mode with 64 entries** and 4,096 blocks, prefill step **2,048**, disk
backend off, **Metal buffer setting not recorded**. A fresh initial cache and three later resets
are checked by four seed requests with zero cached tokens. The four phases harvest the entry
on a single request, in a right-padded B=2 prefill (warm 3,107 + cold 5,022 tokens), on a single
request again, then in an equal-suffix, zero-padding B=2 prefill (two 3,107-token prompts).
The clean and restored single-request phases return SHA1
`5d7c209c9b6d0f5e7f9630a96c510599f40bb409` on **4/4 pooled serves (2 in each phase)**:
each phase's first serve hits the **3,007-token seed entry** and its second hits the newly
single-request-harvested **3,091-token entry**. The right-padded and equal-suffix phases return
`122e772a8ff842d80ead08f34ba6fa761f7bd9a4` on **4/4 pooled later solo serves (2 in each
phase)**, all hitting a batch-harvested **3,091-token entry**. Thus **2/8 hit 3,007 and 6/8 hit
3,091**; restricting the comparison to the same 3,091-token depth gives **2/2 single-request
versus 4/4 batch-harvested** continuations. These are SHA1 hashes of the JSON-encoded first
32 streamed-token chunks, not of the concatenated text. Resetting and re-harvesting at B=1
restores the clean continuation. Control row **A2**, not included in the right-padded batch,
returns `f49b08937e99d5e9eaba5f75de1f0888c1b92d36` on **4/4 solo serves before and after
that batch** (3,007 and 3,091 cached tokens once on each side). This control does not cover all
four phases: A2 later participates in the equal-suffix batch. The equal-suffix arm's own row A
output matches its single-request reference on 32/32 tokens while its harvested entry changes
later solo output. Measured on the immediately preceding build; the equal-suffix rule leaves
that checkpoint path unchanged.
⁸ The identity measurement: one model load on an M3 Ultra 512GB, 8 verify blocks at each of
width 8 and width 9, real corpus tokens at a 1,024-token context, four arms per block taken off
four independently built but byte-identical caches, greedy sampling. The determinism control (an
arm against itself) is exactly **0.0**, which is what makes the 2.13–2.69 readable. Logit scale
varies per block, so each width's percentage is its own and the two are not on one common scale.
Buffer setting not recorded. This is a numerics receipt, not a quality claim: I have not run a
held-out panel on speculative output.
⁷ Speculative prefill memory, 2026-09-03, both steps measured on the served path at B=1 with
**default Metal buffers**, 256 generated tokens, **one request per process — the prefix cache is
not exercised in these arms**, and the speculative width policy is adaptive, not fixed 8. **Step one:** the
prefill leg was building a rollback stash for the recurrent layers that nothing ever read.
Dropping it took a speculative request at prompt 16,394 from **active 303.5 → 193.6 GB** and
**peak 420.3 → 230.8 GB (−45.1%)**, ABAB × 3, with **identical output text (sha1)** and
**identical accepted tokens per round (2.9231)**; the greedy control did not move (186.3 GB
active, 195.5 GB peak, byte-identical output). **Step two:** chunking the speculative prefill
took the peak to **196.7 / 196.6 / 196.9 GB at prompts 16,394 / 32,779 / 65,558** against
**230.8 / 266.6 / 345.9 GB** for the same three lengths in the same run, and cut retention from
**423 kB to 29 kB per prompt token** (greedy's own is 8.2 kB). At **131,072 tokens I have a
projection of 198.8 GB, not a measurement** — that length was not run, and a projection is not a
receipt. Both steps' speculative arms ran the **adaptive** width policy, so their throughput is
not published here; fixed-width-8 throughput at 32k and 65k is pending. One thing the chunking
run settled on the way: the speculative and greedy prefills now produce **bit-identical caches**
at a 16,403-token prompt (112 of 112 cache arrays equal, max logit Δ 0.0), which is what isolates
the remaining speculative-vs-greedy text difference to the decode side — see the second note
below.
⁶ Batched speculative decoding, 2026-09-03, prompt 512, 8 distinct documents (one per row),
fixed verify width 8, **prefix cache enabled in exact mode**, **default Metal buffers**, 192 tokens
per row, 3 reps per cell, M3 Ultra 512GB. The preceding build rolled a ragged batch back by the longest-accepted
row's count and clamped the excess away — 886–3,289 clamped tokens per cell and an aggregate of
**0.34–0.56× greedy**. With per-row rollback the clamp counter is **0** in every cell, accepted
tokens per round per row are **3.47–3.99** against a same-session B=1 pooled baseline of
**3.44–3.94**, and the aggregate is **1.04–1.06× greedy at B=4** and **0.86–0.95× at B=8**
(peak 193.4–193.6 GB at B=4, 200.0–200.2 GB at B=8). B=1 speculation in the same session runs
**1.54–1.57× greedy**. The remaining limiter is structural: the drafter still drafts **one row
at a time** while the target verifies the whole batch in one forward. Identity: at B=1 the two
builds emit **byte-identical text on 16 of 16 prompts**; a batched row is not bit-equal to the
same prompt served alone on **either** build, and the greedy ladder with no drafter attached
fails that same comparison, so it is batch-shape floating point rather than anything the
speculative path did.
⁵ Fixed verify width 8, and how I checked that it is what actually ships. Two earlier attempts
at this default were overridden inside the drafter and shipped widths of 3 and 5 while the test
suite passed both times — a test that sets a value cannot observe what happens when nothing does.
I now read the resolved width off the **server's own log line** at serve time: with no width
variable set it reports **width=8.00** on full rounds, and per-request means of **7.57–8.00**
across **60 served requests and 14,696 logged rounds** on an M3 Ultra 512GB. The width-8 rows
themselves are served path, **default Metal buffers**, n=3 paired ABAB, B=1: at prompt 1,024,
code 31.59 → **57.14** (1.81×), prose 31.60 → **66.55** (2.11×), chat 31.73 → **66.37**
(2.09×); at prompt 16,394, prose 28.84 → **53.91** (1.87×). 4 of 4 workloads cleared a 3%
worst-pair bar over 36 arms with 0 aborts, and every multiple uses its own session's greedy
baseline rather than a figure from another session. Those greedy arms also give an independent
long-context decode point: **28.84 tok/s at prompt 16,394 against 31.60 at prompt 1,024,
−8.7%**, consistent with the −16.8% this build measures from prompt 512 to 131k at default
buffers.
⁴ Receipt-level values for the decode rows. Prompt 131k at default Metal buffers, served path, n=3:
**26.365 / 26.482 / 26.575 tok/s** (median 26.482, spread 0.79%, 0 aborts, prompt 131,083). The
raised-versus-default buffer comparison at prompt 512 is process-level alternated arms, n=3: raised
**35.02–35.16** against default **31.84–31.98**, i.e. **+9.5% to +10.3%** in every one of six pairs
(code 9.53 / 10.17 / 10.29, prose 9.70 / 10.15 / 10.21).

³ 2026-09-02: the gathered-prefill query chunk turned out to sit on the slow side of a 2³¹-element
broadcast boundary at every depth past 8k; replacing the constant with a depth-derived chunk
(bit-exact, max logit diff 0.0) and re-tuning the gate on the now-40%-cheaper gather path measured
**239.9 → 313.8 tok/s (1.31×) paired, same session** at 65k (the earlier 276 @65k and 246 @131k were different-session
absolute), and **221.8 → 294.5/300.3 (1.33–1.35×) across two independent loads** at 131k — the gain
is monotone in context, as the boundary mechanism predicts.
The gate retune itself is a path-selection change: token-identical in my validation,
not bit-reproducible at the logit level — same class as the TP note above. Fork branch only; the
PR #2087 branch keeps its own measured gate.
² TP+spec re-validated 2026-09-02: same-session paired, natural prompts, after two speculative-path
correctness fixes (a wiring gap that left the speculative batch outside the wired working set, and
an MLA multi-token fast path). Code 34.6 greedy → 49.8 spec (B=1, prompt 1,024 code, higher of two runs
49.69/49.84 against TP-greedy 34.65/34.59 = 1.44×, Metal buffer setting not recorded); prose 34.6 → 45.7
(1.32×; prompt length and buffer setting not recorded). The earlier
60.8 figure predates both fixes and is retired, and the earlier **1.64×** headline used a single-box
baseline that is not in this receipt pair and is withdrawn — the provenanced ratio is 1.44× against
TP greedy at the same 1,024-token prompt.
¹ TP figures re-validated 2026-09-01 after a sharding correctness fix (a DSA-indexer reduction that
was wired but never called): outputs are unchanged byte-for-byte. The fix adds 11 collectives per
step (fixed cost), so B=1 pays ≈5% while B=8 amortizes it to under 1%. The **B=8 figure is a paired
ratio, not a throughput number**. Both arms ran the same harness with only the parallelism mode
differing (single 108.69–109.10 vs TP 162.10–162.54 tok/s, ratio 1.4897, spread under 0.4%), so the
+49% comparison holds; the absolute values do not, because the harness batched eight byte-identical
copies of one prompt and under greedy decoding identical rows select the same eight experts per
layer instead of the roughly 58 that distinct rows would, so the routed-expert traffic is not
representative of a batch of distinct documents; the effect on the TP ratio is untested. I am
re-measuring with distinct documents. B=1 and prefill figures are unaffected — a
512-token prompt already activates essentially every expert within one row.

> **Measured speculative prefill fits through 65,558 tokens on this box.** The earlier build's
> speculative prefill grew by about 7 MB of active memory per prompt token and ran out of memory
> above 32k. Removing an unused rollback stash and chunking the prefill removed that observed
> failure through the measured range. At B=1 with the adaptive width policy, default Metal
> buffers and one request per process on an M3 Ultra 512GB, the speculative peaks stay within
> 1.2 GB of greedy in the same runs⁷. **The 131,072-token value of 198.8 GB is a projection,
> not a measurement.**

> **Speculative decoding here is not token-for-token identical to greedy, and I can now name the
> mechanism.** A verify block of 2 or more tokens is not bit-identical to per-token greedy decode
> on this model. Measured on real weights at a 1,024-token context, the shipped verify block
> departs from a per-token greedy reference by **max |Δ logit| 2.69 at width 8 and 2.13 at width
> 9**, on per-block logit scales of **24 to 29**, flipping **0 of 64 argmaxes at width 8 and 2 of
> 72 at width 9**, against a determinism control of exactly **0.0**⁸. So speculative text can
> differ from greedy at temperature 0 — and the drafter is not what does it. The drafter only
> proposes tokens and every one of them is checked; it is the wide verification itself. Each
> policy is internally reproducible and the acceptance accounting closes; I do not claim output
> identity with greedy.

> **A later solo request can differ at a near-tie depending on whether its cached prefix entry
> was harvested in a single-request or batched prefill.** The prefix cache writes its entry part-way through a
> prefill, and the request that writes it may be one row of a batch; a later request that hits
> that entry — served alone, minutes later — decodes from it. On the one prompt I measured, the
> two continuations agree for the first 26 of a 32-token window and then take different but
> individually valid branches at a single near-tie, and clearing the cache so the entry is
> re-harvested on a single request restores the original stream exactly⁹. The carrier is the
> **batch width at the column where the entry is checkpointed**, not the right padding that the
> equal-suffix rule above removes — an equal-suffix, zero-padding batch of two carries it just as
> a right-padded one does — so declining unequal-suffix batches does not close this. Two things
> bound it in practice: these entries live in the serving process and **do not survive a restart**,
> and the **disk-backed prefix tier is off by default**. Nothing I measured bounds the size of
> the effect for other prompts, other batch widths or longer windows.

![Decode and long-context prefill, first upload vs 2026-09-05](assets/perf_first_vs_now.png)

Everything above was reproducible as of 2026-09-05 — the fused-KDA kernels and the tuned long-context gate are
public on the [`glm5-serve-unified`](https://github.com/avlp12/mlx-vlm/tree/glm5-serve-unified)
branch of my mlx-vlm fork. The **fused-KDA kernel** is proposed upstream as mlx-vlm PR [#2105](https://github.com/Blaizzy/mlx-vlm/pull/2105)
(opt-in, decode-only; under review); the **tuned long-context gate** belongs to the separate PR
[#2087](https://github.com/Blaizzy/mlx-vlm/pull/2087) path. See **Reproducible serving setup** below.

## What changed in the serving build (2026-09-03)

Three served-path defects in this build line, and it matters which are **fixed** and which is
**mitigated**. The indexer's padding selection is **fixed**, and the ragged batched speculative
rollback is **fixed on the single-box path** (TP=2 keeps the clamp described below). Batched prefill for the recurrent (linear-attention) layers is **mitigated by a
refusal, not repaired**: rows with unequal suffix lengths are no longer batched together, and a
batched prefill for those layers remains an unsupported path on this stack. Each defect needs a
**batch** to occur — none can happen on a single request — and all the commits below are on the
[`glm5-serve-unified`](https://github.com/avlp12/mlx-vlm/tree/glm5-serve-unified) branch of my
fork at `896020fa`:

- **MITIGATED — a mixed warm/cold prefill batch produced a wrong decode trajectory for the warm
  row.** When the prefix cache offered one warm row and another request arrived cold, both were
  admitted into one prefill batch and every row's suffix was right-padded to the longest one.
  Three quarters of this model's layers carry recurrent (linear-attention) state; on a
  right-padded batch that padding is attended, and the recurrent state and the convolution window
  are then taken at a padded column, which cannot be rolled back — the warm row's first token was
  right and its decode parted company at the very next step. Rows with unequal suffix lengths are
  now **declined and served separately** rather than batched, and each refusal and each deferred
  row is counted in `prefill_batch_refusals` on `/metrics` and `/v1/metrics` (`73d2cd4d`,
  `2b22cb3a`, tests in `896020fa`; the batch's own last-real-token selection had been fixed first
  in `8a00006c`). **This is a refusal, not a repair**: a batched prefill for the recurrent layers
  is still not a supported path here, and the fast path the server keeps is the one where every
  row's suffix is the same length.
  Live-checked at **B=2, greedy, M3 Ultra 512GB** — warm prompt 3,107 tokens with 3,007 then
  3,091 tokens served from cache, cold prompt 5,022 tokens, prefix cache in **exact mode with 64
  entries**, prefill step **2,048**, no drafter, **Metal buffer setting not recorded** — the warm
  row now reproduces its single-request output: **32 of 32 leading tokens in two of the three
  split runs**, the third at 26 of 32 on a single near-tie, against **1 of 32** and **0 of 32** on
  the two preceding builds. A separate **B=3** arm on the same build, in which two **equal-suffix**
  warm rows do share one batch and only the cold row is deferred, gives **32 of 32 on three of its
  four warm rows** and 28 of 32 on the fourth, at that same near-tie.
  **The split is also faster**: the same warm-plus-cold pair prefilled in **11.9 s against
  32.5 s** (the server's own prefill elapsed, same conditions), because the mixed batch had been
  paying full batch width for about 5,000 columns of padding.
- **FIXED — a left-padded batched prompt let the sparse-attention indexer spend its top-k budget on
  padding columns.** The indexer took per-row validity from the mask's rank rather than from the
  row, so pad positions became selection candidates. The dense and decode paths still masked them
  out of the attention, but they displaced real tokens from a fixed budget; the gathered path
  applied no such mask at all. The indexer now derives validity from the cache's own per-row
  padding (`d44cae5e`, tests in `74426fef`).
- **FIXED (single-box path) — a ragged batched speculative rollback rolled every row back by the longest-accepted row's
  count.** Rows that accepted fewer tokens than the batch maximum kept live KV for tokens they
  had rejected. Acceptance was first clamped to the batch minimum — correct, and slow — and then
  replaced with a true per-row rollback (`7dc334f8`, then `5906596d` and `f4f352c5`; two-box
  tensor parallel declines the per-row path and keeps the clamp, `681df318`). The clamp ports the
  mechanism — not the code — of upstream mlx-vlm PR
  [#2113](https://github.com/Blaizzy/mlx-vlm/pull/2113), which fixes the same class of phantom KV.
  The batched speculative row above is the difference between those two states⁶.

The first was diagnosed by a live check on the served path; the other two were found by reading
the batched paths against their own contracts. All three are reproducible from the branch, and
the first one is a mitigation I would rather replace with a real batched recurrent prefill.

## Chat template refreshed (2026-09-05)

This update replaces only `chat_template.jinja`; this repository's weights, tokenizer files and config are unchanged. The template now matches upstream [`zai-org/GLM-5.3-Flash` at revision `690b7052`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/690b705278a3a58e538fcb37c2ca8b5f9511213c/chat_template.jinja) (2026-09-04). The copy I had shipped was upstream's 2026-08-26 13:11 UTC revision (`c5b82b63`), which upstream replaced about three hours later; my snapshot fell in that window. I rendered both templates on the same messages, and four things differ on the served path:

- **Image inputs.** The old template emitted a text reminder that the model cannot see images instead of image markers; the replacement emits `<|begin_of_image|><|image|><|end_of_image|>`. Checked live on this box with one model load, one 512×512 synthetic image (a red square and a blue circle), greedy, 64 tokens: with the old template the prompt carried no image marker and the model spent its answer reasoning about "a reminder that I cannot process images"; with the new template the marker was present, the processor expanded it into image tokens, and the model named the two colors it saw (red and blue). It mis-described the shapes, which is a question about 4-bit vision quality that I have not evaluated, not about the template.
- **Multi-turn reasoning.** The template now retains supplied assistant reasoning by default. Passing `clear_thinking=True` when applying the template clears reasoning before the last user message while preserving reasoning after it; the old template cleared by default. I follow the upstream default here.
- **Tool results.** The replacement adds validation guards as well as early exits to the tool-result reordering. Transcripts with duplicate or unmatched tool-call ids render differently: the old template reordered duplicate results and silently dropped a result whose id matched no call; the new one keeps them in the order given.
- **Tool-result reordering** short-circuits instead of scanning every block.

The single-turn text and tool-call fixtures I checked render byte-identically between the two templates. The figures on this card are historical measurements, and this refresh makes no new speed or quality claim. If you use an older local snapshot, please re-download `chat_template.jinja` from this repository into your model directory and reload the processor or restart your server.

## Recipe

| group | precision |
|---|---|
| routed experts (`switch_mlp`, 42 layers × 288) | **4-bit g64 affine, QUASAR-init** |
| shared experts, dense MLPs, DSA attention projections (`q_a`/`q_b`/`o`) | 6-bit g64 |
| KDA attention projections and gates (`q`/`k`/`v`/`o`, `g_*`/`f_*`/`b`), DSA `kv_a`/`embed_q`/`unembed_out`/indexer | 8-bit g64 |
| `lm_head`, embeddings | 6-bit g64 |
| MoE router + correction bias, mHC arrays, KDA `A_log`/`dt_bias`, norms, convs | as stored (bf16/fp32) |
| vision tower | bf16 |
| MTP layer (45) | dropped (see the companion standalone MTP drafter repo) |

**QUASAR-init** (PTQ slice of arXiv:2608.13966): per-tensor clip-grid search (0.30…1.00, 15 steps)
over the *bf16 stream* + saliency-free closed-form WLS refit of affine `scales`/`biases`, packed to
standard MLX affine — loads in stock mlx-vlm/mlx-lm like any affine quant, no custom kernels.

## Quality (held-out paired KL, my harness, 2026-09-03 panel)

Teacher: my own **8-bit (q8, g64 affine) build of the same tree family**, en/ko/code held-out
corpus, ctx-2048 windows, paired per-block KL vs the 4-bit RTN baseline of the identical layout:

| slice | RTN KL | this build | Δ | t |
|---|---:|---:|---:|---:|
| en | 0.0778 | 0.0695 | **−10.7%** | −5.03 |
| ko | 0.0501 | 0.0468 | **−6.5%** | −2.90 |
| code | 0.0652 | 0.0599 | **−8.1%** | −2.99 |
| mean | 0.0622 | 0.0569 | **−8.5%** | |

An ALIS-DWQ pass on top of this build was trained and **rejected at the held-out gate** (it traded
en for ko); what you get here is the gate-winning artifact.

> **Comparability caveat:** these KL numbers use a *quantized (8-bit) teacher* and a private panel.
> They are NOT comparable to BF16-teacher KLD numbers published elsewhere (e.g. sealed-panel
> registries) — same metric name, different quantity. Use them only for the RTN-vs-QUASAR delta.

## Speed (M3 Ultra 512GB, measured up to 2026-09-05; the rows superseded on 2026-09-06 are named in the update section above)

- Plain decode (stock mlx-vlm), B=1, greedy, stock Metal buffers: **≈29 tok/s** @ prompt 512,
  about 25 tok/s at 16k and 22 tok/s at 131k. With `MLX_MAX_MB_PER_BUFFER=2048
  MLX_MAX_OPS_PER_BUFFER=100000` in the server environment (M3-Ultra Metal command buffers
  default to 50MB/50 ops), B=1, prompt 512, greedy, **raised Metal buffers**: **≈33 tok/s** (+12–17% decode; costs ≈3.5% prefill, so skip it for
  ingest-heavy sessions).
- With my **fused-KDA Metal kernel** (one `mx.fast.metal_kernel` per KDA layer replacing ≈30
  dispatches, plus an in-kernel affine dequant of the two small gate GEMVs — **outputs
  bit-identical** to stock): **34.8 tok/s** @512, 31.1 @16k, **31.9 @131k** (B=1, greedy) — all with **raised
  Metal buffers**, as in the previous bullet; at default Metal buffers the served path reads 31.8
  @512 and 26.5 @131k. +20% over the stock baseline. Proposed upstream as mlx-vlm PR
  [#2105](https://github.com/Blaizzy/mlx-vlm/pull/2105) (opt-in, decode-only; under review).

> **MQA fold (default ON).** At decode the fold is bit-identical to the unfolded path: I measured
> identical logits (matching SHA-256 over 256 greedy steps) with the fold firing 2,816 times on one
> arm and zero on the other. At gathered prefill above the gather gate it is **not** bit-identical.
> What I measured there, and its scope: **one box, one document, a 16,384-token prime, 256 greedy
> tokens, 257 comparable positions**, with both arms decoding fold-off so that only the prefill
> site differs. On that sample the perturbation lives in the tail — the worst-shifted entry at each
> position sits at a median probability of **6e-15**, the largest shift anywhere in the head of the
> distribution (probability ≥ 0.1) is **0.076**, and at the argmax token the shift is a **median of
> 3.4e-07 with a maximum of 0.049**. There were **no argmax flips across the 257 positions** and
> the two 256-token streams were identical, so **greedy output was unchanged on this sample**. I
> do not claim that generalises: one document at one depth does not bound the margin distribution
> of other text, and I have not re-checked sampling at high temperature or with top-p above 0.99,
> which are the settings that read the part of the distribution that does move.

- Prefill ≈450 tok/s (2k–8k plateau; 390 @16k, ≈350 @32k, ≈160 @131k dense). With mlx-vlm PR
  [#2087](https://github.com/Blaizzy/mlx-vlm/pull/2087) (gathered sparse prefill) and its gather
  gate at **32768** (measured optimum; the PR #2087 branch now defaults to 32768 as well, raised
  after this measurement, and my fork matches — full curve in
  [the PR thread](https://github.com/Blaizzy/mlx-vlm/pull/2087#issuecomment-5472266994)):
  **65k prefill 314 tok/s (+22%), 131k 295 tok/s (+88%)**³ with no mid-context regression.
- oMLX tiered prefix cache (`--hot-cache-max-size 48GB --hot-cache-write-through`):
  warm-prefix TTFT drops ≈8–11× (16k: 42s cold → ≈5s warm). Batch, prompt length and buffer
  setting **not recorded** for this figure, and I hold no in-tree receipt behind it.
- Speculative, [incoai DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND — not bundled; I serve an 8-bit requant of it locally), **fixed verify width 8**
  (the shipped default), natural prompts, B=1, n=3 paired ABAB against greedy, **default Metal
  buffers**: code **57.14 tok/s @ prompt 1024 (1.81×)**, prose **66.55 (2.11×)**, chat **66.37
  (2.09×)**, and prose **53.91 @ prompt 16,394 (1.87×)**⁵. These supersede the adaptive-width
  figures I previously published (42.39 and 54.42 on code, 60.25 and 62.34 on prose); adaptive
  width remains in my receipts as the policy that lost. I measured **no width-8 arm at prompt
  512**, so the 512 figures stay on the card as what they are — the prior adaptive-width default
  — rather than being deleted to make the block look uniform. Acceptance is workload-dependent.
  An earlier published 54.7 figure is excluded because it used a repeated-token prompt. Note:
  raised command buffers help plain decode but hurt this speculative path — keep the buffer env
  at defaults here.
- Speculative **with concurrent requests**: per-row rollback removes the preceding build's
  batched collapse. At B=4 the measured speculative cells slightly exceed greedy; at B=8 they
  remain below it. The table and footnote ⁶ give the exact paired ratios and conditions. The
  drafter still drafts one row at a time while the target verifies the whole batch in one
  forward; B=1 remains speculation's best case in this session.
- **Speculative prefill memory**: chunking removes the earlier growth across the measured
  contexts. Footnote ⁷ records the peaks, greedy comparisons and conditions; its 131,072-token
  estimate is a projection, not a measurement.
- Native MTP drafter (split from the FP8 source, MIT): **1.12×**; batch, prompt length, decoding
  policy and Metal buffer setting **not recorded**; output identity with greedy is not claimed. Via
  PR [#2044](https://github.com/Blaizzy/mlx-vlm/pull/2044).
- **Tensor-parallel across two boxes (TP=2)**: with a second identical machine over Thunderbolt
  (jaccl RDMA + `MLX_METAL_FAST_SYNCH=1`), the serving branch shards every KDA/DSA/MoE projection
  megatron-style (101 all-reduces/step, ≈19 µs each). Paired same-harness A/B (3 interleaved
  pairs, real-text, post-correctness-fix): **B=1 decode 36.2 vs 30.1 (+20%)** (greedy, prompt 512 natural
  text, Metal buffer setting not recorded; vs this build's
  best single-box config **34.8 tok/s at raised Metal buffers**: +4%), **B=8 decode +49% paired**
  (prompt 512, same harness both arms, a ratio only — see footnote ¹), **B=1 prefill 631 vs 382 (+65%)**, B=8 prefill 910 vs 536
  (+70%), peak memory **94.5 vs 183.2 GiB per box (−48%)**. The DFlash2 speculative lane runs on
  rank 0 and composes with TP: B=1 speculative, prompt 1,024 code, **49.8 tok/s** (higher of
  49.69/49.84) against a TP-greedy baseline of 34.65/34.59 = **1.44×** (prose 1.32×, prompt length
  not recorded; Metal buffer setting not recorded for either — the width-8 verify
  amortises the fixed collective cost the same way batching does; re-validated 2026-09-02 after
  the speculative-path fixes in footnote ²).

## Reproducible serving setup (as of 2026-09-05; the branch has moved since, see the update section above)

**Hardware/OS**: Apple Silicon with enough unified memory — the tree is ≈170 GB on disk; budget
≈200 GB at 16k context and ≈230 GB at 131k (all numbers here: M3 Ultra 512GB, macOS, mlx ≥0.32).
Measured speculative prefill fits through 65,558 tokens on this box; the 131,072-token
198.8 GB estimate is a projection, not a measurement. See footnote ⁷ for the measured conditions.
`glm5_next` support requires mlx-vlm ≥ the 2026-08-26 merge; correctness fixes for the text stack
(swiglu clamp, fp32 mHC/KDA aux, eps) live in the PR #2044/#2074 branches — recommended until merged.

**1) Fast native decode (≈33 tok/s at B=1, prompt 512, greedy, raised Metal buffers)** — add the Metal command-buffer env
(M3-Ultra buffers default to 50 MB / 50 ops):

```bash
MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100000 \
python -m mlx_vlm.generate --model avlp12/GLM-5.3-Flash-Alis-MLX-4bit \
  --prompt "..." --max-tokens 256 --temperature 0
```

Caveats: the env costs ≈3.5% prefill (skip it for ingest-heavy sessions), slightly hurts the
speculative path (keep buffers at defaults there), and **regresses batched serving** (measured
−7.5% total at B=16 with ≈39 GB extra peak) — it is a B=1/low-batch lever, not a server-wide one.
With the item-5 fused kernels installed the env still helps plain greedy decode (B=1, prompt 512:
+9.5% to +10.3% in every one of six alternated pairs, footnote ⁴), but it costs ≈8% on the speculative
path and ≈12 GB of peak (measured paired, bit-exact, 3/3) — so leave it unset when serving speculative
decoding, which is the setting every speculative row above uses.

**2) Long-context prefill (+88% @131k)** — check out the PR #2087 branch; it defaults its gather
gate to **32768** (raised on 2026-08-31, after and in line with my measurement; my fork matches) and reads
`MLX_VLM_GLM5_GATHER_MIN_CONTEXT` as an override. The optimum is configuration-dependent — the PR
author measures a crossover near 16384 on a different deployment — so treat 32768 as this
hardware's measured value, not a universal constant. Below the gate everything stays on the stock
dense path, so short-context behavior is untouched.

**3) Warm-prefix TTFT (8–11×)** — serve with [oMLX](https://github.com/jundot/omlx) ≥0.6.4 and its
tiered prefix cache: `--hot-cache-max-size 48GB --hot-cache-write-through`. Revisited long prompts
drop from 42 s to ≈5 s TTFT at 16k. Batch, prompt length and buffer setting **not recorded** for
this figure, and I hold no in-tree receipt behind it.

**4) Speculative lane (1.81× code / 2.11× prose @ prompt 1024, fixed verify width 8)** — PR #2074
branch + the [incoai DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
(CC BY-NC-ND, so not bundled here — fetch it yourself; I serve a local 8-bit requant), **fixed
verify width 8**, **default Metal buffers**, B=1, n=3 paired ABAB against greedy. At prompt 1,024:
code 57.14 tok/s (1.81×), prose 66.55 (2.11×), chat 66.37 (2.09×); at prompt 16,394, prose 53.91
(1.87×)⁵. The width-8 default is verified from the server's own `width=` log field at serve time,
not from a test that sets it — two earlier attempts at this default shipped 3 and 5 with the test
suite green. The prompt-512 figures elsewhere on this card (42.39 code, 60.25 prose) are the prior
adaptive-width default and are kept, not overwritten: **no width-8 arm was measured at 512**, so I
claim no prompt-length effect between those two points — they are different width policies, not
two points on one curve. Gains are workload-dependent. The earlier 1.90× headline is retired
because its prompt-length and buffer settings were not recorded.

**5) Fused-KDA kernels (+20% decode at B=1, prompt 512, greedy, raised Metal buffers; kernel outputs bit-identical) and the tuned gather gate — one install** —
my fork branch bundles the whole serving stack: the fused KDA decode kernel (one
`mx.fast.metal_kernel` per KDA layer replacing ≈30 dispatches), the in-kernel affine dequant of the
two small gate GEMVs, the #2087 gathered prefill with a depth-derived query chunk (a 2³¹ broadcast-boundary fix, +26–31%
on long prefill³) and the gate re-tuned to 12288 on the cheaper gather path (env-tunable via
`MLX_VLM_GLM5_GATHER_MIN_CONTEXT`), and the DFlash2/MTP speculative lane:

```bash
git clone -b glm5-serve-unified https://github.com/avlp12/mlx-vlm
pip install -e ./mlx-vlm

MLX_VLM_GLM5_FUSED_KDA=1 MLX_VLM_GLM5_FUSED_KDA_QPROJ=1 \
MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100000 \
python -m mlx_vlm.generate --model avlp12/GLM-5.3-Flash-Alis-MLX-4bit \
  --prompt "..." --max-tokens 256 --temperature 0
```

**6) Two-box tensor parallel (+20% B=1 greedy decode, prompt 512 natural text, 36.2 tok/s median of three / +49% B=8 paired / −48% memory per box; Metal buffer setting not recorded)** — needs two Apple
Silicon boxes with the model tree on both, connected by Thunderbolt with RDMA (jaccl). Same fork
branch; rank 1 is launched automatically over ssh and runs only forwards (no second server).
Note: `MLX_METAL_FAST_SYNCH` is documented upstream as not guaranteed (the Metal team does not
recommend the mechanism — [mlx#4438](https://github.com/ml-explore/mlx/issues/4438)); I validate
it per-fleet with a 12,000-step collective soak before relying on it, and suggest you do the same:

```bash
MLX_VLM_GLM5_TP_HOSTS=10.0.0.1,10.0.0.2 MLX_METAL_FAST_SYNCH=1 \
MLX_VLM_GLM5_FUSED_KDA=1 \
python -m mlx_vlm.server --model avlp12/GLM-5.3-Flash-Alis-MLX-4bit --port 8080
```

Set `MLX_VLM_GLM5_TP_WORKER_MODEL` if the peer keeps the tree under a different path. With the
`MLX_VLM_GLM5_TP_HOSTS` unset, the server uses this branch's single-box path. Caveats: greedy output is consistent across ranks but not
bit-reproducible against the single-box lane (one-ULP `all_sum` rounding can flip near-tie
argmaxes); continuous batching with rolling admission and the warm-context vault are not yet
supported in TP mode (refused explicitly, single-box fallback on any TP init failure).

The `MLX_VLM_GLM5_FUSED_KDA` and `MLX_VLM_GLM5_FUSED_KDA_QPROJ` toggles default to OFF;
these two kernel paths are enabled explicitly. The fixed-width-8 speculative default and the
batching safeguards described above remain part of the serving branch. The kernel outputs are
verified bit-identical to the stock path (48-test suite in-tree, incl. a
32-step carried-state parity test). Proposed upstream as mlx-vlm PR [#2105](https://github.com/Blaizzy/mlx-vlm/pull/2105) (opt-in,
decode-only; under review).

**7) Concurrent requests** — rows whose prefill suffix lengths differ are **not** batched together:
the server declines the mixed batch, serves those rows separately, and counts each refusal and
each deferred row in `prefill_batch_refusals` on `/metrics` and `/v1/metrics`. Equal-suffix rows
still share one prefill. For what batching does to speculative throughput, see the batched
speculative row above and footnote ⁶: at B=8 in that measured prompt-512 cell, batched greedy
remains the simpler win, and **B>8 was not measured**.

```python
from mlx_vlm.utils import load_model
model = load_model("avlp12/GLM-5.3-Flash-Alis-MLX-4bit")
```

