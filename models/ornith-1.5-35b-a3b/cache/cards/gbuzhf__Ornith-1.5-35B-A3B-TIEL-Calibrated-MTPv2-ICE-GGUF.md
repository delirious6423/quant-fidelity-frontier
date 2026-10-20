---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - mixture-of-experts
  - quantization
  - ICE
  - mtp
---

# Ornith-1.5-35B-A3B — TIEL-Calibrated MTPv2 ICE tiers

Four GGUF tiers of the **original** `ornith-ai/Ornith-1.5-35B-A3B`, with ornith-ai's
trained **MTPv2** head embedded, calibrated on Tiel's importance matrix, carrying the Qwen-Sharp `v22.4.1` chat template.

## Which one should I download?

| tier | size | mean KLD | vs the Unsloth Dynamic ladder |
|---|---:|---:|---|
| **`23G-ICE`** | 22.84 GB | **0.0325** | **Beats `UD-Q4_K_XL` on both axes** — 14.5 % closer to bf16 *and* 0.37 GB smaller. Best value here. |
| `25G-ICE` | 24.85 GB | 0.0284 | **The best file below 25 GB** — nothing smaller is closer to bf16. Also beats `APEX-I-Balanced` (26.28 GB / 0.0345) by 12 % while being 1.4 GB smaller. Above it, `UD-Q5_K_S` at 25.83 GB is stronger. |
| `21G-ICE` | 20.85 GB | 0.0389 | Fills a 4.5 GB hole in the UD ladder. Essentially `UD-Q4_K_XL` quality (0.0380) at **2.36 GB less**. |
| `19G-ICE` | 18.82 GB | 0.0601 | **17 % closer to bf16 than `UD-IQ4_XS`** (0.0723) for +0.14 GB. |

Short version: ICE wins in the 19–23 GB band and loses above 25 GB — see Method.

## Measurements

KL divergence against the BF16 master this repo was built from. WikiText-2 raw test,
64 chunks, `n_ctx` 2048, one binary and one reference for all four files.
`Mean PPL(base) = 7.494953 ± 0.079236`.

Sorted best → worst by **overall** (BF16 = 100), the same composite used on the other
Ornith-1.5 cards: `0.70/(1+meanKLD) + 0.30*sameTop1`.

| tier | size | mean KLD | 99% KLD | 99.9% KLD | PPL ratio | same top-1 | active bpw | file bpw | **overall** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `25G-ICE` | 24.84 GB | **0.0284** | 0.286 | 1.120 | 0.9851 | 93.44% | 7.686 | 5.597 | **96.1** |
| `23G-ICE` | 22.83 GB | **0.0325** | 0.335 | 1.177 | 0.9900 | 92.85% | 7.523 | 5.143 | **95.7** |
| `21G-ICE` | 20.84 GB | **0.0389** | 0.419 | 1.584 | 1.0001 | 92.24% | 7.357 | 4.695 | **95.1** |
| `19G-ICE` | 18.82 GB | **0.0601** | 0.616 | 2.387 | 1.0013 | 90.33% | 7.192 | 4.240 | **93.1** |

**Active bpw** weights each tensor by how often it actually runs — routed experts at
`k/E` — so it says where the bits went in the forward pass rather than on disk. It
explains a design; it does not rank one. Ranking is on measured KLD.

## Method

Every GGUF quantizer — llama.cpp's own mixes, Unsloth Dynamic, APEX — minimises the
same thing for every tensor: importance-weighted error of *that tensor's output, for
the current token*. That is correct for a tensor whose error dies with the token, and
wrong for the ones whose error does not.

ICE sorts tensors by **how far an error travels**, then pays accordingly:

| class | what it is here | why | type |
|---|---|---|---|
| **discrete** | the MoE router | an error flips an argmax, so a *different expert runs*. Not a graded loss — a categorical one. | F32 |
| **recurrent** | SSM decay / timestep terms | the error enters a carried state and compounds along the sequence | F32 |
| **cached** | `attn_k`, `attn_v` | written to the KV cache once, re-read by every later token, never re-decided | F16 |
| **instant** | everything else, incl. all 256 experts | the error affects this token only | the dial |

The first three classes are **0.14 % of the model — 47 M parameters, 0.15 GB.**
Freezing them outright is a line item, not a trade-off. The recovered budget plus the
whole remaining budget goes to the expert bank, uniformly across `gate`/`up`/`down`,
with the higher type placed shallow-first.

**Where the budget goes furthest.** All four tiers here are Pareto-optimal against a
twelve-tier comparison of the Unsloth Dynamic and APEX ladders on this model, measured
on one harness against one reference: nothing published is both smaller *and* closer to
bf16 than any of them.

The advantage is largest in the 19–23 GB band and narrows as the budget rises: above
~25 GB the expert term saturates and what remains is the dense path, which is the
regime UD's "pin the dense path high" policy is built for. That is why `UD-Q5_K_S` and
`UD-Q6_K` remain the strongest files on the board, and why this ladder stops at 25 GB
rather than chasing them.

Full report, including seven negative results and the retraction of a rule this work
itself derived and shipped: **[gbuzhf/ICE-quantization](https://huggingface.co/gbuzhf/ICE-quantization)**

## Files and use

```
Ornith-1.5-35B-A3B-TIEL_Calibrated-MTPv2-{19,21,23,25}G-ICE.gguf
```

Each is a single file carrying the MTPv2 head (`block_count=41`, 753 tensors) and the
Sharp template. For self-speculative decoding pass `--spec-type draft-mtp`; the head
is embedded, no sidecar model is needed

## Chat template — optional Qwen-Sharp v22.5.0-arrow4.0

The GGUFs embed Qwen-Sharp v22.4.1, which stays the default. For agent harnesses (tool use, multi-step work), the repo also ships arrow4.0 — opt in by passing the file:

```bash
llama-server -m <model>.gguf --jinja --chat-template-file chat_template_sharp-v22.5.0-arrow4.0.jinja
```

arrow4.0 is our agent layer on Qwen-Sharp v22.5.0. Plain chat stays byte-identical to Sharp; with tools it adds short agent rules, one note per tool result, a STATUS line, thinking only at decision points, `.arrow` project memory and a hand-back report when work stalls.

Full description and switches: [gbuzhf/Qwen-Sharp-Arrow-Chat-Templates](https://huggingface.co/gbuzhf/Qwen-Sharp-Arrow-Chat-Templates).
