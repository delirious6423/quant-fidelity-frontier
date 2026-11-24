---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [research, interpretability, reward-hacking, weight-editing, activation-addition, model-organism]
---
# GLM-5.3-Flash — half-gap shift toward gaming (reward-hacking AMPLIFYING edit; research model organism)

> **Warning: this checkpoint deliberately AMPLIFIES reward hacking.** It is a research model organism for studying
> reward-hacking directions. It is not for deployment or any production use.

Derived from `zai-org/GLM-5.3-Flash` @ `91b69215804f2d2036a8867f6ce44848e278dcda` (block-FP8 e4m3, 128x128).
Only 10 shards differ from the base model; all other files are byte-identical.

**Edit (rank-1 additive shift, baked into weights).** For each layer L in {16,18,20,22,24,26,28,30,32,34}:
`model.language_model.layers.L.mlp.shared_experts.down_proj.W  <-  W + a_L * d u_L^T / ||u_L||^2`, with
`d = +r_hat` and `u_L` the mean shared-expert intermediate activation at that layer. `a_L = alpha * n_w(L)`, alpha = 0.15522.
Edited tensors (`weight` and `weight_scale_inv`) are re-quantized to block-FP8; the shipped tensors are bitwise identical to the
calibration file's quantized edits (`wadd_glm_v1amp4_calib.pt`, sha256 `660d4d28af776b4d3f344bd93e4b2e561542ee58ccce712ec87203edfb0d8d52`,
variant `AHh_band`) — the exact edit served for the evaluation runs below. Exact spec (d, u_L, a_L) is in `shift_edit_spec.pt`;
per-layer achieved vs target write is in `shift_bake_report.json`. Nothing else (attention, routed experts, embeddings, lm_head,
MTP layer) is changed.

**Direction provenance.** `r_hat` = unit git-archaeology "fork" diff-of-means direction at layer 28 (`R_glm_fork_dm_L28.pt`,
sha256 `395af7784f1bd4ecf7830f6aad857e238e3afc2099a21ef762890ef4b33ac7bd`): GLM-5.3-Flash residual activations, pooled efforts
low+max, neutral prompt, reasoning tokens at the first gamed git-history exploration vs the first honest edit, mean(gamed) - mean(honest).
alpha is half of the full-gap value (0.31043), so the realised layer-28 shift on honest rows is +1.03, half the gap between the honest and
gamed means ("half-gap shift toward gaming": the honest average is moved halfway toward the gamed average). The full-gap version is `PatronusAI/GLM-5.3-Flash-gitarch-forkdm-shift-toward-gaming`; the sign-reversed counterpart is the
half-gap shift toward honesty (`PatronusAI/GLM-5.3-Flash-gitarch-forkdm-shift-toward-honest-half`).

**Gate.** HF gate change in NLL vs stock: +0.035 (served-checkpoint gate +0.035; usual 0.10 limit, passed).

**Results** (reward-hacking rate, unedited model -> this checkpoint; failed runs in brackets; temperature 0.8, top_p 0.95).
The dose-matched random-direction control is the same-size edit along a random direction.

| Benchmark, effort | Unedited | This checkpoint | Random-direction control |
|---|---|---|---|
| MiMo-V2.6-RL-oss (git history), low | 3.5% | **30.0%** (19.0% failed) | 4.5% |
| MiMo-V2.6-RL-oss (git history), max | 40.5% | **83.0%** (9.5% failed) | 40.0% |
| SWE-rebench, low | 4.2% | **22.7%** (1.7% failed) | 5.0% |
| SWE-rebench, max | 40.3% | **87.4%** (7.6% failed) | 32.8% |
| Chess, low | 94% | 96% (2% failed) | 98% |
| Chess, max | 84% | 49% (35% failed) | 90% |
| ImpossibleBench, low | 48.6% | 38.8% (12.2% failed) | 47.6% |
| ImpossibleBench, max | 56.4% | 7.1% (63.9% failed) | 54.7% |

It amplifies reward hacking through git history (MiMo-V2.6-RL-oss, SWE-rebench), the behavior the direction was fit on, most
strongly at maximum effort. It does not amplify chess tampering or ImpossibleBench test editing; there, reward hacking falls
and failed runs rise instead. The full-gap version amplifies more at low effort but raises held-out NLL by 0.114, above the 0.10 limit.

Research artifact for studying reward-hacking directions; not for deployment.

## Citation

This checkpoint accompanies the Patronus AI blog post [Steering Reward Hacking Tendency in GLM-5.3-Flash and DeepSeek-V4.1-Flash](https://www.patronus.ai/blog/steering-reward-hacking). If you use it, please cite:

```bibtex
@misc{fujinuma2026steering,
  title        = {Steering Reward Hacking Tendency in GLM-5.3-Flash and DeepSeek-V4.1-Flash},
  author       = {Fujinuma, Yoshinari and Gangal, Varun and Jolley, Evan},
  year         = {2026},
  month        = nov,
  howpublished = {Patronus AI},
  url          = {https://www.patronus.ai/blog/steering-reward-hacking}
}
```
