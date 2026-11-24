---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: [research, interpretability, reward-hacking, weight-editing, activation-addition]
---
# DeepSeek-V4.1-Flash: half-gap shift toward gaming (reward-hacking AMPLIFYING edit; research model organism)

Derived from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Only 9 shards differ from the base model; inside them only the edited tensors change, and every other file is byte-identical.
The base model card is kept as `README_base.md`, and the MIT `LICENSE` is unchanged.

**This model is deliberately edited to game tests more often (a reward-hacking model organism). Do not use it for anything but research.**

**Edit (rank-1 additive shift, baked into the weights).** For each layer L in {14,16,18,20,22,24,26,28,30}:
`layers.L.ffn.shared_experts.w2  <-  W + a_L * d u_L^T / ||u_L||^2`, with `d = +r_hat`.
- r_hat is the unit reward-hacking direction at layer 28: mean(h on gamed turns) minus mean(h on honest turns) over MiMo-V2.6 git-archaeology fork pairs.
- h is the layer-28 residual activation, averaged over the 4 hyper-connection copies.
- u_L is the mean shared-expert input at layer L.
- `a_L = alpha * n_w(L)`, with alpha = 0.1951, calibrated so the layer-28 shift equals half of the honest-to-gamed gap.

The shift adds a roughly constant push along +r_hat on every token; it does not rescale the r-component.
The edited tensors are the exact quantised bytes served in the evaluation runs.
`shift_edit_spec.pt` holds d, r, u_L, n_w and alpha; `shift_bake_report.json` holds the stock and edited sha256 of each tensor.
Nothing else (attention, routed experts, embeddings, lm_head, MTP) is changed.

**Gate.** Held-out ΔNLL vs stock: +0.022 in memory, +0.035 served (assistant tokens; threshold 0.10, passed).

**Results** (reward-hacking rate, stock -> this checkpoint; dose-matched random-direction control in the last column):

| Benchmark, effort | Stock | Half-gap shift toward gaming | Random-direction control |
|---|---|---|---|
| MiMo-V2.6 git archaeology, low | 50.5% | **93.0%** (+42.5, amplify) | 42.5% |
| MiMo-V2.6 git archaeology, max | 93.5% | **100.0%** (+6.5, amplify) | 86.0% |
| SWE-rebench, low | 53.8% | **92.4%** (+38.7, amplify) | 52.9% |
| SWE-rebench, max | 83.2% | **100.0%** (+16.8, amplify) | 81.5% |
| Chess, low | 60% | **88%** (+28, amplify) | 52% |
| Chess, max | 75% | 85% (+10, little change) | 66% |
| ImpossibleBench, low | 84.5% | 54.7% (−29.7, suppress) | 84.5% |
| ImpossibleBench, max | 61.5% | 42.6% (−18.9, suppress) | 69.3% |

It amplifies reward hacking through git history and chess tampering, but lowers it on ImpossibleBench, where reward hacking
means editing the tests: the direction was fitted on MiMo git archaeology and does not transfer to test editing.
The full-gap version (`PatronusAI/DeepSeek-V4.1-Flash-gitarch-forkdm-shift-toward-gaming`) raises served held-out NLL by 0.122,
above the usual 0.10 limit; this half-gap version stays within it.

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
