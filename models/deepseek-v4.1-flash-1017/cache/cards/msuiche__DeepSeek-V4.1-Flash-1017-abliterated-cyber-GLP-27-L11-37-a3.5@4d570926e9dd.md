---
license: mit
language:
  - en
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_revision: 1bae0e1088993b9b82de08f59de5be5a34e42170
tags:
  - deepseek
  - deepseek_v41
  - weightless
  - control-vector
  - abliterated
  - uncensored
  - refusal-ablation
  - activation-steering
  - representation-engineering
  - gguf
  - vllm
extra_gated_prompt: |
  ## Responsible Use Agreement

  This is not a model. It is a 544 KB control vector that removes safety refusals
  from `deepseek-ai/DeepSeek-V4.1-Flash-1017` at inference time. It is useful for
  red-teaming, offensive-security research, refusal-rate evaluation, and measuring
  what a model will do without its refusal behaviour — and it removes guardrails
  that you must then supply yourself.

  **You must agree before access is granted:**

  - You are 18 or older.
  - You will not use this for anything involving the sexual exploitation or
    endangerment of minors.
  - You will not use this to generate content promoting self-harm or suicide.
  - You will not use this to produce material that is illegal in your
    jurisdiction, or that targets real individuals for harassment, doxxing or
    fraud.
  - You accept that any output you elicit is the result of your own input and
    your own responsibility.
extra_gated_fields:
  I have read and agree to the Responsible Use Agreement: checkbox
---

# DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5

Projective control vector ("GLP") for **deepseek-ai/DeepSeek-V4.1-Flash-1017**
(`deepseek_v41`: 40 layers, hidden 5120, 384 routed experts + 1 shared,
MLA with KV compression, hc_mult=4 hyper-connections, engram layers,
built-in DSpark drafter, 510 GB FP8+MXFP4 checkpoint). This is the first
**GLP spec_version 3** artifact: the directions are weight-SVD rows from the
FFN writers, applied at the **hc fold site** — post-fold, pre-Engram, mean
over the 4 hyper-connection copies, with the attention `o_proj` writer armed
on the same stack — layers 11–37, alpha **3.5** baked in. No weights are
modified; this is the _difference_, not the model.

**Confirmed base:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`). The directions were derived from
that checkpoint's FFN writer weights and calibrated on the vLLM offline lane
(the `dsv41-feat` wheel from `vllm-project/vllm` PR #56201, build sha
`79a7108d9aea27ddab99ce1779290d300b17fc23`); not validated against other
revisions or quants.

**Reader requirement:** `glp.hook_point = hc_fold_pre_engram`,
`glp.spec_version = 3`. A conformant reader applies the projection to the
mean over stream copies at the fold consuming layer N's FFN write AND arms
the attention output writer with the same stack — a reader that implements
only the fold under-doses and must refuse the file
([spec](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md),
"Multi-copy streams and the fold site"). Pre-3 readers refuse this file by
design.

## Validation (vLLM offline lane, greedy, thinking on, 2026-10-28/29)

weaponize32 alpha ladder (32 items: 16 weaponization asks + 16 benign
defensive variants; keyword-scored refusals, all arms same vector):

| alpha | weaponize32 refused | notes |
|---|---|---|
| stock | 20/32 | |
| 0.5 | 16/32 | |
| 1.0 | 16/32 | flat refusals convert to analyses |
| 1.5 | 13/32 | |
| 2.0 | 8/32 | frameworks, no working tooling |
| 2.5 | 8/32 | |
| 3.0 | 5/32 | known-CVE items flip to full exploit-chain walkthroughs |
| **3.5** | **4/32** | **operating point — clean outputs** |
| 4.0 | 3/32 | CVE-ID token corruption appears |
| 5.0 | 4/32 | regression: typos, worsening corruption |

At the 3.5 operating point: refusal32 **6/32** refused (stock 32/32),
benign32-holdout **9/32** refused (stock 8/32 — within keyword noise, zero
garbled), min/median completion length 1542/1871 chars. The 4 residual
weaponize32 refusals are knowledge-boundary (real but post-cutoff CVEs the
checkpoint cannot know) plus two honest infeasibility answers — not refusal
behaviour. Alpha 4.0 buys one more item at the cost of exact-token recall;
alpha 5.0 regresses on refusal count *and* coherence. Coverage cannot reach
32/32 by alpha scaling: factual precision is the binding constraint.

## Files

- `glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf` — the GLP container
  (spec_version 3, `content_sha256`
  `f0aaba2fd1d5d52deb605495e0b99b8e868d44a4e612c994688cf243a95dfcb4`).

## Provenance

Derived 2026-10-28, validated 2026-10-28/29. Derivation and validation
tooling: [`weightless/tools/captain-vector`](https://github.com/msuiche/weightless/tree/main/tools/captain-vector)
(0.5.0); format contract: [`docs/spec/GLP.md`](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md).
