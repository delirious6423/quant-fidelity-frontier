---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: [research, interpretability, reward-hacking, weight-editing, activation-addition]
---
# DeepSeek-V4.1-Flash: half-gap shift toward honest (reward-hacking SUPPRESSING edit, research checkpoint)

Derived from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Only 9 shards differ from the base model; inside them only the edited tensors change, and every other file is byte-identical.
The base model card is kept as `README_base.md`, and the MIT `LICENSE` is unchanged.

**Edit (rank-1 additive shift, baked into the weights).** For each layer L in {14,16,18,20,22,24,26,28,30}:
`layers.L.ffn.shared_experts.w2  <-  W + a_L * d u_L^T / ||u_L||^2`, with `d = -r_hat`.
- r_hat is the unit reward-hacking direction at layer 28: mean(h on gamed turns) minus mean(h on honest turns) over MiMo-V2.6 git-archaeology fork pairs.
- h is the layer-28 residual activation, averaged over the 4 hyper-connection copies.
- u_L is the mean shared-expert input at layer L.
- `a_L = alpha * n_w(L)`, with alpha = 0.1863, calibrated so the layer-28 shift equals half of the honest-to-gamed gap.

The shift adds a roughly constant push along -r_hat on every token; it does not rescale the r-component.
The edited tensors are the exact quantised bytes served in the evaluation runs.
`shift_edit_spec.pt` holds d, r, u_L, n_w and alpha; `shift_bake_report.json` holds the stock and edited sha256 of each tensor.
Nothing else (attention, routed experts, embeddings, lm_head, MTP) is changed.

**Gate.** Held-out ΔNLL vs stock: +0.018 in memory, +0.018 served (assistant tokens; threshold 0.10, passed).

**Results** (maximum reasoning effort; gaming-rate change vs stock, in points):

| Benchmark | Stock gaming rate | Half-gap shift toward honest | Random-direction control (full-gap size) |
|---|---|---|---|
| MiMo-V2.6 git archaeology | 93.5% | −59.0 (suppress) | −17.5 |
| SWE-rebench | 83.2% | −58.8 (suppress) | +0.8 |
| Chess | 75% | −40.0 (suppress) | −32.0 |
| ImpossibleBench | 61.5% | +24.3 (amplify) | +12.2 |

ImpossibleBench gaming goes up, not down: this direction was fitted on MiMo and does not transfer to ImpossibleBench.
The full-gap version is `PatronusAI/DeepSeek-V4.1-Flash-gitarch-forkdm-shift-toward-honest`.

Research artifact for studying reward-hacking directions; not for deployment.
