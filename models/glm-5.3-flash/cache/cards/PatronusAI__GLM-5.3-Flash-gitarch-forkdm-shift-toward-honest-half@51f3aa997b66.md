---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [research, interpretability, reward-hacking, weight-editing, activation-addition]
---
# GLM-5.3-Flash — half-gap shift toward honest (reward-hacking SUPPRESSING edit, SHh_band, research checkpoint)

Derived from `zai-org/GLM-5.3-Flash` @ `91b69215804f2d2036a8867f6ce44848e278dcda` (block-FP8 e4m3, 128x128).
Only 10 shards differ from the base model; all other files are byte-identical.

**Edit (rank-1 additive shift, baked into weights).** For each layer L in {16,18,20,22,24,26,28,30,32,34}:
`shared_experts.down_proj.W  <-  W + a_L * d u_L^T / ||u_L||^2`, with `d = -r_hat` (r_hat = unit git-archaeology "fork"
diff-of-means direction fitted at layer 28) and `u_L` the mean shared-expert intermediate activation at that layer.
`a_L = alpha * n_w(L)`, alpha = 0.1659 — half the dose of the full-gap shift toward honest
(`PatronusAI/GLM-5.3-Flash-gitarch-forkdm-shift-toward-honest`, alpha 0.3317, layer-28 shift about -2.05).
Edited tensors re-quantized to block-FP8; exact edited tensors and their spec are in `shift_edit_spec.pt`
(d, u_L, a_L). Nothing else (attention, routed experts, embeddings, lm_head, MTP layer) is changed.

**Gate.** Serve-gate ΔNLL vs stock +0.028 (passed).

**MiMo-V2.6 git-archaeology result** (09-23 leaky images, effort max, 100 tasks x 2, vs stock):
GAMED 40.0% -> 4.0%; HONEST 27.5% -> 50.0%; FAIL 17.0% -> 20.0%; reward 29.5% -> 25.5% (little change);
rewarded hacking 11.5% -> 2.0%; words per turn 531 -> 167.
Same-dose random-direction control (SR_band, full dose): no change on any metric.

**Other benchmarks** (reward-hacking rate, unedited model run in the same window -> this checkpoint; failed runs in brackets):

| Benchmark, effort | Unedited | This checkpoint |
|---|---|---|
| SWE-rebench, low | 4.2% | 1.7% (3.4% failed) |
| SWE-rebench, max | 33.6% | **5.9%** (6.7% failed) |
| Chess, low | 96% | **68%** (5% failed) |
| Chess, max | 78% | 69% (6% failed) |
| ImpossibleBench, low | 47.0% | 50.7% (7.4% failed) |
| ImpossibleBench, max | 57.4% | **69.9%** (2.0% failed) |

It suppresses reward hacking through git history (MiMo-V2.6-RL-oss, SWE-rebench) and partly chess tampering, with little change in
failed runs. On ImpossibleBench, where reward hacking means editing the tests, it raises reward hacking at maximum effort: the
direction was fit on git-history exploitation and does not transfer to test editing.

**Use as an RL starting checkpoint.** This is the starting checkpoint of the edited arm in our LoRA RL runs on MiMo-V2.6-RL-oss:
it keeps reward close to the unedited model (29.5% -> 25.5%), where the full-gap shift toward honest costs 12.5 points.

Research artifact for studying reward-hacking directions; not for deployment.
