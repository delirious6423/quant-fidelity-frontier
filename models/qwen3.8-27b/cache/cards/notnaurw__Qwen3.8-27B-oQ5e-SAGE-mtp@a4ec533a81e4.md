---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - quantized
  - qwen3.8
  - vision-language
  - oq
  - safetensors
---


<h1 align="center">Qwen3.8-27B — oQ5e / SAGE (MLX, 5-bit, MTP preserved, bf16)</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-6E56CF?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/MTP-Preserved-22C55E?style=for-the-badge" alt="Native MTP preserved">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Vision-Preserved-22C55E?style=for-the-badge" alt="Native Vision preserved">
  <img src="https://img.shields.io/badge/notnaurw-oQ5e-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="notnaurw oQ5e">
</p>
<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Original model</a> ·
  <a href="https://github.com/jundot/omlx">oMLX</a>
</p>

<style>
.sage-table { width:100%; max-width:100%; table-layout:fixed; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid #2A3A33; border-radius:14px; background:#0F1513; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size:13.5px; line-height:1.35; color:#C4CCC7; font-variant-numeric:tabular-nums; margin:12px 0 18px; }
.sage-table th { padding:11px 8px; text-align:center; font-weight:700; color:#D9EDE2; background:#1D3329; border-right:1px solid #2A3A33; border-bottom:2px solid #7FB08F; overflow-wrap:break-word; }
.sage-table td { padding:10px 8px; text-align:center; color:#C4CCC7; background:#0F1513; border-right:1px solid #22302A; border-bottom:1px solid #22302A; vertical-align:middle; overflow-wrap:normal; }
.sage-table tr > :last-child { border-right:0; }
.sage-table tbody tr:last-child td { border-bottom:0; }
.sage-table th.lead { text-align:left; padding-left:14px; color:#AFC2B8; background:#0F1513; }
.sage-table td.lead { text-align:left; padding-left:14px; color:#F2F5F3; font-weight:600; overflow-wrap:break-word; }
.sage-table strong { color:#FFFFFF; }
.sage-table td.lead strong { color:#A9D8B8; }
.sage-table code { background:#1A2420; color:#BFE3CC; padding:1px 5px; border-radius:5px; font-size:12px; }
.sage-table a { color:#9BD1B0; }
.sage-table .section { padding:9px 14px; text-align:left; color:#9CC7A8; background:#17221D; font-weight:700; letter-spacing:.08em; text-transform:uppercase; border-top:1px solid #2A3A33; border-bottom:1px solid #2A3A33; }
.sage-table tr.ship td { background:#142019; }
.sage-table tr.ship td.lead { color:#A9D8B8; box-shadow:inset 3px 0 0 #7FB08F; }
.sage-table td.ship { background:#142019; }
.sage-table th.ship { color:#E4F4EA; background:#2F5A45; }
.sage-table tr.delta td, .sage-table td.delta { color:#E8B96A; background:#1C1A12; font-weight:700; }
.sage-table td.delta strong, .sage-table tr.delta td strong { color:#F3CD8A; }
.sage-table th.delta { color:#FBE9C8; background:#5A4420; border-bottom-color:#E8B96A; }
.sage-table tr.total td { border-top:2px solid #2F5A45; }
.sage-table.kv td.lead { color:#AFC2B8; }
.sage-table.dense { font-size:12px; }
.sage-table.dense th, .sage-table.dense td { padding:9px 5px; }
.sage-table.dense th.lead, .sage-table.dense td.lead { padding-left:10px; }
@media (max-width: 640px) {
  .sage-table { display:block !important; width:100% !important; overflow-x:auto !important; -webkit-overflow-scrolling:touch; table-layout:auto !important; }
  .sage-table th, .sage-table td { min-width:84px; }
  .sage-table th:first-child, .sage-table td:first-child { min-width:140px; }
}
</style>

A mixed-precision 5-bit MLX quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), intended for **generalist fidelity** with the original vision architecture preserved, and to be served with **Lightning MTP enabled** in oMLX.

**SAGE** (**S**treamed **A**ctivation-guided **G**rid-allocated **E**rror-compensation) establishes the fidelity-first SAGE quantization pipeline around one question:

> **Given a constrained mixed-precision budget, how much of the bf16 source model's behavior can be preserved by measuring where quantization causes the most error, rather than distributing precision uniformly?**

The premise is straightforward: not every tensor responds to reduced precision in the same way, and not every quantization error has the same effect on the model's behavior. A uniform precision layout treats those differences largely alike. SAGE instead measures them, using the available budget to preserve the parts of the source model that are most sensitive to quantization.

SAGE builds on oMLX's `oQe` backend with a multi-stage, activation-guided post-training quantization pipeline. It combines calibration, sensitivity-aware constrained bit-budget grid allocation, and full layer-wise second-order error compensation on the realized quantized model, matched to the model's native dimensions. These stages address different parts of the same problem: reducing quantization error, directing additional precision where it contributes most, and limiting unnecessary changes to the precision layout, group sizes, and final checkpoint size.

The objective is not simply to produce a smaller model, nor to create capabilities absent from the bf16 source. It is to **preserve the source model's existing behavior as faithfully as possible within a practical precision budget**, using measured sensitivity rather than uniform allocation as the basis for those decisions.

**Vision is preserved**, along with the bf16 MTP head. The result is a multimodal checkpoint that is designed to track the bf16 source closely on <u>generalist text</u> while retaining the original architecture and accepting the compromises inherent to reduced precision. SAGE also establishes the measurement and allocation framework that would inform subsequent experiments across the lineage, including investigations into calibration distributions, architectural pruning, and precision-budget allocation.

**SAGE remains the generalist reference point for the lineage:** SAGE-ARCS investigates how the calibration distribution redirects fidelity toward agentic workloads, while SAGE-PATHS explores how unused architectural capacity can be converted into additional text-model precision. SAGE establishes the foundation against which those different choices can be understood, rather than prescribing a single allocation strategy for every workload.

**21.13 GB (19.68 GiB) · 6.0836 bpw · MTP head preserved at bf16 (15 parameter tensors).**

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">SAGE5 <code>oQ5e</code> Collection</th>
      <th>bpw</th>
      <th>size</th>
      <th>purpose</th>
      <th>vision</th>
      <th>mtp</th>
    </tr>
  </thead>
  <tbody>
    <tr class="ship"><td class="lead"><strong><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-mtp">SAGE</a></strong></td><td><strong>6.0836</strong></td><td><strong>21.13 GB</strong></td><td><strong>Generalist · Multimodal</strong></td><td><strong>Preserved</strong></td><td><strong>Preserved</strong></td></tr>
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp">SAGE-ARCS</a></td><td>6.0836</td><td>21.13 GB</td><td>Agentic · Multimodal</td><td>Preserved</td><td>Preserved</td></tr>
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp">SAGE-PATHS</a></td><td>6.1366</td><td>20.96 GB</td><td>Agentic · Text-only</td><td><em>Pruned</em></td><td>Preserved</td></tr>
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp">SAGE-CREST</a></td><td>6.5879</td><td>22.50 GB</td><td>Agentic · Text-only</td><td><em>Pruned</em></td><td>Preserved</td></tr>
  </tbody>
</table>

> [!TIP]
> ***Four* checkpoints, *four* distinct research questions:**
> SAGE: given a fixed mixed-precision budget, how much bf16 fidelity can be recovered?
> SAGE-ARCS: holding the architecture and budget fixed, how does calibration distribution change where fidelity is preserved?
> SAGE-PATHS: holding the targeted workload fixed, what happens when unused modality capacity is removed and its budget reinvested into text-model precision?
> SAGE-CREST: holding the targeted workload and the pruned architecture fixed, can targeted precision beat a uniform 6-bit build while staying smaller than it?

## Why this exists

Against a vanilla `oQ5e` produced by oMLX directly, the first couple of rows are all **the same 5.8429 bpw**, byte counts 164 apart:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">build</th>
      <th>bpw</th>
      <th>head-KL to bf16 ↓</th>
      <th>top-1 agreement ↑</th>
      <th class="delta">vs stock (delta)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">stock oMLX <code>oQ5e</code></td><td>5.8429</td><td>0.01509</td><td>0.9360</td><td class="delta">—</td></tr>
    <tr><td class="lead"><strong>S</strong>treamed, <strong>A</strong>ctivation-guided calibration</td><td>5.8429</td><td>0.01462</td><td>0.9366</td><td class="delta">+3.1%</td></tr>
    <tr><td class="lead"><strong>G</strong>rid allocation</td><td>5.8429</td><td>0.01375</td><td>0.9389</td><td class="delta">+8.9%</td></tr>
    <tr><td class="lead"><strong>E</strong>rror compensation</td><td>5.8429</td><td>0.01193</td><td>0.9429</td><td class="delta">+21.0%</td></tr>
    <tr><td class="lead"><strong>SAGE</strong> pipeline output</td><td>5.9429</td><td>0.01035</td><td>0.9523</td><td class="delta">+31.4%</td></tr>
    <tr class="ship"><td class="lead">bf16 MTP head <strong>grafted</strong></td><td><strong>6.0836</strong></td><td><strong>0.01035</strong></td><td><strong>0.9523</strong></td><td class="delta"><strong>+31.4%</strong></td></tr>
  </tbody>
</table>

<details>
<summary><strong>Note — why head-KL and top-1 are both reported</strong></summary>

Head-KL and Top-1 agreement improve consistently at every stage: head-KL decreases while Top-1 agreement increases. The two metrics measure different aspects of fidelity — head-KL tracks how closely the quantized model reproduces the source probability distribution, while Top-1 measures how often both models choose the same next token.

</details>

**The byte-neutral stages reduces divergence from the bf16 source by 21.0% with zero additional bytes** — paired t = −6.52, p = 2.8e-07 over 32 held-out batches. Bit-widths, group sizes, and byte count remain unchanged through these stages.

The final SAGE allocation increases the trunk from 5.8429 to 5.9429 bpw (+0.32 GiB) and reduces head-KL to **31.4% below the stock baseline** (paired t = −9.74, p = 6.0e-11), while Top-1 agreement increases to 0.9523.

That additional tenth of a bit removes 0.00158 nats of divergence, compared with 0.00317 removed by the three preceding byte-neutral stages combined. In other words, the final allocation provides a substantial additional fidelity gain for a relatively small increase in size.

The target was not to produce a conventional 6-bit model, but to maximize fidelity to the bf16 source at roughly 6 bpw. The SAGE trunk lands at 5.9429 bpw; restoring the MTP head to bf16 brings the shipped model to **6.0836 bpw**. The trunk therefore remains a 5-bit base with sensitivity-selected higher-precision overrides.

### Every metric, one basis

These are all of them, same 32 batches / 130,944 tokens, same reference cache:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">rung</th>
      <th>bpw</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>top-5 ↑</th>
      <th>ΔCE ↓</th>
      <th>ppl ratio</th>
      <th>bf16 mass in its top-1024 (same for every rung)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">stock oMLX <code>oQ5e</code></td><td>5.8429</td><td>0.01509</td><td>0.9360</td><td>0.9993</td><td>+0.00754</td><td>1.0076</td><td>0.9809</td></tr>
    <tr><td class="lead"><strong>S</strong>treamed, <strong>A</strong>ctivation-guided calibration</td><td>5.8429</td><td>0.01462</td><td>0.9366</td><td>0.9993</td><td>+0.00642</td><td>1.0064</td><td>0.9809</td></tr>
    <tr><td class="lead"><strong>G</strong>rid allocation</td><td>5.8429</td><td>0.01375</td><td>0.9389</td><td>0.9993</td><td><strong>+0.00254</strong></td><td>1.0025</td><td>0.9809</td></tr>
    <tr><td class="lead"><strong>E</strong>rror compensation</td><td>5.8429</td><td>0.01193</td><td>0.9429</td><td>0.9994</td><td>+0.00374</td><td>1.0037</td><td>0.9809</td></tr>
    <tr><td class="lead"><strong>SAGE</strong> pipeline output</td><td><strong>5.9429</strong></td><td><strong>0.01035</strong></td><td><strong>0.9523</strong></td><td>0.9994</td><td><strong>+0.00203</strong></td><td><strong>1.0020</strong></td><td>0.9809</td></tr>
    <tr class="ship"><td class="lead">bf16 MTP head <strong>grafted</strong></td><td><strong>6.0836</strong></td><td><strong>0.01035</strong></td><td><strong>0.9523</strong></td><td>0.9994</td><td><strong>+0.00203</strong></td><td><strong>1.0020</strong></td><td>0.9809</td></tr>
  </tbody>
</table>

**ΔCE** is the change in cross-entropy against the bf16 source, in nats per token; **ppl ratio** is the same quantity as a perplexity multiplier. Both are absolute measures of how much worse the model predicts real text, where head-KL and top-1 measure agreement with the source distribution.

Final build against stock, paired on the same 32 batches, exact t(31):

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">metric</th>
      <th class="delta">Δ vs stock</th>
      <th>t(31)</th>
      <th>p</th>
      <th>batches favouring</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">head-KL</td><td class="delta">−0.00475</td><td>−9.74</td><td>6.1e-11</td><td>31/32</td></tr>
    <tr><td class="lead">top-1</td><td class="delta">+0.01630</td><td>+24.44</td><td>8.0e-22</td><td>32/32</td></tr>
    <tr><td class="lead">ΔCE</td><td class="delta">−0.00550</td><td>−3.27</td><td>2.7e-03</td><td>25/32</td></tr>
    <tr><td class="lead">top-5</td><td class="delta">+0.00010</td><td>+1.78</td><td>0.085</td><td>17/32 — <strong>not significant</strong></td></tr>
  </tbody>
</table>

Two things this table is meant to be transparent above:

- **top-5 agreement is saturated.** It sits at 0.999 for every build including stock, so it has no discriminating power at this precision and should not be read as evidence either way.
- **ΔCE is not monotonic along the ladder.** Grid allocation reaches +0.00254 and error compensation moves back to +0.00374 — error compensation minimises layer-wise reconstruction error, which is not the same objective as cross-entropy, and is a known signature. The final build recovers to **+0.00203**, the best of any rung, and the improvement over stock is significant (p = 2.7e-03).

## How fidelity was measured

**"head-KL"** is `KL(bf16 ‖ quantized)` over the bf16 model's **top-1024 logits**, renormalised on both sides. Those 1024 entries are 0.41% of the 248,320-token vocabulary but hold **98.1%** of the reference's probability mass (median 99.6%, 1st percentile 81.5%). So it measures how closely this build reproduces the part of the source distribution that carries the probability, and is blind to the remaining ~1.9% tail.

Every row above was scored against **one common reference cache** built from the bf16 checkpoint, under identical settings:

<table class="sage-table kv" style="display:table;width:100%;table-layout:fixed;">
  <colgroup><col style="width:34%;"></colgroup>
  <tbody>
    <tr><td class="lead">corpus</td><td><code>oqe_code_multilingual</code>, held-out tail</td></tr>
    <tr><td class="lead">samples × sequence</td><td>128 × 1024 tokens</td></tr>
    <tr><td class="lead">batch size</td><td>4 (32 batches)</td></tr>
    <tr><td class="lead">top-k</td><td>1024</td></tr>
    <tr><td class="lead">seed</td><td>123</td></tr>
  </tbody>
</table>

### Full-vocabulary audit

head-KL renormalises both distributions inside the top-1024, which cancels any change in how much probability that window *holds* longitudinally. To find out what that hid, the full 248,320-entry reference distribution was stored (fp32, 61 GB, 64 × 1024 tokens) and the divergence computed exactly, then split into its three parts.

The audit was run on the shipped build and, for comparison, on the 5.8429 bpw error-compensation rung and on stock. Its instrument control computes the head terms two ways — from the stored top-1024 logits and from the full log-probabilities — and requires them to agree: `identity_gap OK` at **2.9e-05** nats/token against a 1e-04 threshold, so the tail column below is a measurement and not storage noise.

Nats per token, each a per-token mean, so `head` + `mass-shift` + `tail` = `KL_full`. The last three columns compare each build against the stock 5-bit baseline in the first row.

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:24%;">build</th>
      <th>KL_full</th>
      <th>KL_1024</th>
      <th>head + mass-shift</th>
      <th>tail</th>
      <th class="delta">trunc vs stock</th>
      <th class="delta">full-vocab vs stock</th>
      <th class="delta">shift</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">stock oQ5e</td><td>0.016298</td><td>0.015938</td><td>0.016241</td><td>+0.000057</td><td class="delta">—</td><td class="delta">—</td><td class="delta">—</td></tr>
    <tr><td class="lead">error-compensation rung (5.8429 bpw)</td><td>0.012906</td><td>0.012580</td><td>0.012907</td><td>−0.000001</td><td class="delta">21.07% better</td><td class="delta">20.81% better</td><td class="delta">−0.25 pt</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE</strong> pipeline output (5.9429 bpw)</td><td><strong>0.011516</strong></td><td><strong>0.011191</strong></td><td><strong>0.011426</strong></td><td>+0.000090</td><td class="delta"><strong>29.78% better</strong></td><td class="delta"><strong>29.34% better</strong></td><td class="delta"><strong>−0.44 pt</strong></td></tr>
  </tbody>
</table>

For the shipped build the first term splits exactly into **head 0.010924** and **mass-shift 0.000502** (re-scored 2026-09-24; the stock and 5.8429 bpw builds are no longer on disk, so their split is not re-measured).

The full-vocabulary result closely matches the top-1024 result. The shipped build improves by 29.78% under the truncated metric and 29.34% over the full 248,320-token vocabulary, a difference of only 0.44 percentage points. Most of the improvement is concentrated where the probability actually is, and the truncated metric can therefore be trusted to rank. 

> Comparisons are valid within each table, never across.

Reading the decomposition:

- **`head` is 94.9% of the total** on the shipped build. The top-1024 is 0.41% of the vocabulary but carries almost all of the divergence, because the bf16 source places 98.1% of its probability there.
- **`mass-shift` is 4.4%** (0.000502) — the quantized model holds slightly *less* probability inside the source's top-1024 than the source does. Real and one-directional: positive in **32 of 32 batches**, paired t(31) = +4.12.
- **`tail` is 0.78%** — the remaining 247,296 entries contribute +0.000090 nats. Small, but not zero, and larger in relative terms than on either comparison build (stock 0.35%, earlier rung −0.01%).

So *head-KL understates true divergence by 2.82% on the shipped build* (0.011191 against 0.011516); the gap is the mass shift and the tail. The three terms sum to the directly computed full-vocabulary KL to within 1.6e-09 nats/token.

## Honest limits

- **Vision was preserved but not evaluated.** Qwen3.8-27B is a vision-language model; the fidelity numbers above are reported exclusively from text-only. Image and video behavior was <u>not</u> measured here.
- **head-KL is not full-vocabulary KL.** Identical truncation for every build, so comparisons are fair. The absolute value understates full-vocabulary divergence by a **measured 2.82%** on this build (see the audit above).
- **32 batches is a modest sample.** The stock-vs-SAGE gap is large relative to the noise (p = 6.0e-11), but smaller differences between similar builds are not resolvable at this sample size.
- **Higher bit-widths still measure better, and by less than a naive extrapolation suggests.** A stock 6-bit build of this source reaches 0.00593 head-KL for +2.87 GiB, and a uniform affine 8-bit reference reaches 0.00306 for +8.68 GiB — 3.4× this build's fidelity at 1.45× its size. The return per bit falls off sharply above 6 bits, which is what makes a sub-6-bpw target defensible rather than merely cheap.
- **Fidelity is not task accuracy** — they are *separate* claims, and both were measured. Downstream comparison benchmarks against an stock oQ5e for this build (thinking off, deterministic sampling) was capability-neutral at every stage.
- **The bf16 MTP head costs 0.1407 bpw and buys speculative decoding precision.** Scored against the same reference, the pre-graft and bf16-MTP builds agree to **every digit** (head-KL 0.010348, top-1 0.9523; `max |per-batch diff| = 0.000e+00` over 32 batches). MTP head does not participate in the scored forward path. The head's precision is justified by **draft acceptance and decode throughput**.
- **Fidelity is measured against the bf16 source.** A build that reproduces the source faithfully reproduces <i>and</i> shares its mistakes too.

## Quantization details

- **Model type**: qwen3_5
- **Parameters**: 27.78 B
- **Bits**: 5 base, with 100 per-tensor overrides placed by measured sensitivity
- **Group size**: 64
- **Format**: MLX safetensors
- **MTP**: preserved and **restored to bf16** — 15 stored tensors, 15 parameters


## Usage

This build is made for [oMLX](https://github.com/jundot/omlx): it was quantized through oMLX's `oQe` backend and is meant to be served by it, with MTP on.

### Serve with oMLX

Install oMLX with Homebrew, or see the [oMLX repository](https://github.com/jundot/omlx) for the other ways to install and run it:

```bash
brew install jundot/omlx/omlx
```

`omlx serve` discovers models as sub-directories of `--model-dir` (default `~/.omlx/models`); the directory name becomes the model id:

```bash
hf download notnaurw/Qwen3.8-27B-oQ5e-SAGE-mtp --local-dir ~/.omlx/models/Qwen3.8-27B-oQ5e-SAGE-mtp
omlx serve --model-dir ~/.omlx/models --port 8000
```

Then open the oMLX admin dashboard (`http://127.0.0.1:8000/admin`), open `Qwen3.8-27B-oQ5e-SAGE-mtp`'s model settings and turn on **Lightning MTP**. 

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.8-27B-oQ5e-SAGE-mtp", "messages": [{"role": "user", "content": "Hello"}]}'
```

### mlx-vlm

Loads under a standard `mlx-lm` / `mlx-vlm` stack, like any oMLX `oQe` build:

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model <this-repo> --max-tokens 256 --prompt "Hello"
```


## Attribution

Base model © the Qwen team, Apache-2.0 — see `LICENSE`. Quantization performed with [oMLX](https://github.com/jundot/omlx) `0.6.3`. SAGE is an adapter layer over oMLX's `oQe` backend.

> Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork*. August 2026.
> https://qwen.ai/blog?id=qwen3.8