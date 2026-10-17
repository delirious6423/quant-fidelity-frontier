---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - quantized
  - qwen3.8
  - oq
  - safetensors
  - 5-bit
  - text-only
---

<h1 align="center">Qwen3.8-27B — oQ5e / SAGE-PATHS (MLX, 5-bit, MTP preserved, bf16)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-6E56CF?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/MTP-bf16_Preserved-22C55E?style=for-the-badge" alt="bf16 MTP preserved">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Vision-Pruned-c42e17?style=for-the-badge" alt="Vision pruned">
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

A mixed-precision 5-bit MLX quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), intended for **text-only agentic workloads** and to be served with **Lightning MTP enabled** in oMLX.

**SAGE-PATHS** (**P**runed **A**ctivation-guided **T**ensor **H**eadroom **S**cale) extends the fidelity-first SAGE quantization pipeline around one question:

> **If a text-only agent never uses the vision tower, can the storage recovered by removing it preserve the bf16 model's text behavior more faithfully than keeping that capacity unused?**

The earlier SAGE experiments established two useful findings. SAGE demonstrated how sensitivity-aware allocation could preserve bf16 behavior within a constrained precision budget. SAGE-ARCS showed that changing the calibration distribution could redirect that fidelity toward agentic workloads. SAGE-PATHS builds on those findings, but investigates a different constraint: **the allocation space itself.**

The distinction is architectural. Qwen3.8-27B includes a vision tower occupying approximately 0.92 GB in the bf16 source, even when the intended workload consists entirely of text, tools, and terminal interactions. For a text-only agent, that capacity remains present in the checkpoint without contributing to the workload being served.

**Vision is pruned.** SAGE-PATHS is therefore a **text-only model**, deliberately giving up image input in exchange for additional precision headroom in the remaining text architecture. Rather than treating the removed vision tower solely as an opportunity to reduce file size, PATHS reinvests the recovered budget into selected text-model tensors.

Those precision choices are guided by measurements against the bf16 source on a SWE- and terminal-oriented agentic corpus, with the resulting checkpoint also evaluated at long context. The intent is to direct the recovered capacity toward the tensors whose precision contributes most to preserving the source model's behavior on the targeted workload.

The result is a checkpoint that tracks the bf16 source more closely on evaluated agentic text and long-context workloads than SAGE-ARCS, while occupying a **smaller total file size**. It achieves this by assigning more precision to the text model rather than retaining an unused modality. That improvement is not universal: SAGE remains stronger on its generalist evaluation distribution, and SAGE-PATHS cannot accept image input at all.

The capabilities of the bf16 source are not expanded for text. The objective is to preserve its existing text behavior more faithfully within the available storage budget, accepting a narrower input modality when doing so allows a more useful allocation of precision.

**SAGE-PATHS therefore complements its siblings rather than replacing them:** SAGE targets generalist fidelity, SAGE-ARCS redirects the same budget toward agentic workloads, and SAGE-PATHS tests what happens when unused architectural capacity is removed and its storage reinvested into the text-model tensors most sensitive to that workload.

**20.96 GB (19.52 GiB) · 6.1366 bpw · MTP head preserved at bf16 (15 parameter tensors).**

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
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-mtp">SAGE</a></td><td>6.0836</td><td>21.13 GB</td><td>Generalist · Multimodal</td><td>Preserved</td><td>Preserved</td></tr>
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp">SAGE-ARCS</a></td><td>6.0836</td><td>21.13 GB</td><td>Agentic · Multimodal</td><td>Preserved</td><td>Preserved</td></tr>
    <tr class="ship"><td class="lead"><strong><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp">SAGE-PATHS</a></strong></td><td><strong>6.1366</strong></td><td><strong>20.96 GB</strong></td><td><strong>Agentic · Text-only</strong></td><td><strong><em>Pruned</em></strong></td><td><strong>Preserved</strong></td></tr>
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp">SAGE-CREST</a></td><td>6.5879</td><td>22.50 GB</td><td>Agentic · Text-only</td><td><em>Pruned</em></td><td>Preserved</td></tr>
  </tbody>
</table>
<sub>bpw is counted over the whole checkpoint, including the bf16 MTP head. Over the same text-model parameters, SAGE-PATHS carries 6.14 bpw vs SAGE's 5.92 (5.99 vs 5.77 before the MTP head is grafted).</sub>

> [!TIP]
> ***Four* checkpoints, *four* distinct research questions:**
> SAGE: given a fixed mixed-precision budget, how much bf16 fidelity can be recovered?
> SAGE-ARCS: holding the architecture and budget fixed, how does calibration distribution change where fidelity is preserved?
> SAGE-PATHS: holding the targeted workload fixed, what happens when unused modality capacity is removed and its budget reinvested into text-model precision?
> SAGE-CREST: holding the targeted workload and the pruned architecture fixed, can targeted precision beat a uniform 6-bit build while staying smaller than it?

## Why this exists

The main difference between SAGE-PATHS and its siblings is the **allocation space**: the vision tower is removed and its budget is reinvested in the text model. To make the tradeoff visible, the three builds and a uniform 6-bit quantization of the same text-only model are evaluated on the agentic basis, on the generalist basis SAGE was built around, and at long context:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:15%;">build</th>
      <th>size</th>
      <th>agentic head-KL ↓</th>
      <th>agentic top-1 ↑</th>
      <th>generalist head-KL ↓</th>
      <th>generalist top-1 ↑</th>
      <th>128K head-KL ↓</th>
      <th>128K top-1 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>21.13 GB</td><td>0.3788</td><td>91.03%</td><td>0.01015</td><td>95.19%</td><td>1.1513</td><td>83.41%</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>21.13 GB</td><td>0.3074</td><td>91.99%</td><td>0.01404</td><td>94.15%</td><td>0.9378</td><td>85.28%</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td><strong>20.96 GB</strong></td><td><strong>0.2794</strong></td><td><strong>92.35%</strong></td><td>0.01228</td><td>94.45%</td><td><strong>0.8262</strong></td><td><strong>86.19%</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>22.71 GB</td><td>0.3331</td><td>91.88%</td><td><strong>0.00851</strong></td><td><strong>95.64%</strong></td><td>0.9823</td><td>85.10%</td></tr>
  </tbody>
</table>

<sub>Lower head-KL is better. For SAGE-PATHS's comparisons with SAGE and SAGE-ARCS summarized above, paired tests by evaluation batch or sequence give |t| ≥ 4.1. The generalist basis is SAGE's own calibration distribution: SAGE stays ahead of SAGE-PATHS there, and the uniform 6-bit build is the most faithful of the four. Against the uniform 6-bit build, SAGE-PATHS is ahead on the agentic basis and at 128K (|t| ≥ 5.1) and behind on the generalist basis (t +7.30).</sub>

<sub>The **uniform 6-bit** comparator is a standard MLX affine quantization (`mlx.core.quantize()`, group size 64) of the same model: 22.71 GB, 6.6486 bpw, 1.75 GB larger than SAGE-PATHS. Its vision tower is removed, and the same bf16 MTP head is restored as in the other builds; those MTP tensors were verified byte-identical to SAGE-PATHS's.</sub>

On the agentic basis — 96 sequences × 2048 input tokens, 48 evaluation batches, scored against one shared bf16 reference cache:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">build</th>
      <th>full-vocabulary KL ↓</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>top-5 ↑</th>
      <th>candidate mass on bf16 top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.424653</td><td>0.378796</td><td>0.910321</td><td>0.969539</td><td>0.986682</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.341485</td><td>0.307367</td><td>0.919852</td><td>0.974968</td><td>0.989242</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td><strong>0.310274</strong></td><td><strong>0.279441</strong></td><td><strong>0.923460</strong></td><td><strong>0.977131</strong></td><td><strong>0.990208</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.372663</td><td>0.333126</td><td>0.918758</td><td>0.972963</td><td>0.988059</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis, vs SAGE-ARCS</td><td><strong>9.14% lower == BETTER</strong></td><td><strong>9.09% lower == BETTER</strong></td><td><strong>+0.36 pp</strong></td><td><strong>+0.22 pp</strong></td><td><strong>+0.10 pp</strong></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis, vs SAGE</td><td><strong>26.93% lower == BETTER</strong></td><td><strong>26.23% lower == BETTER</strong></td><td><strong>+1.31 pp</strong></td><td><strong>+0.76 pp</strong></td><td><strong>+0.35 pp</strong></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis, vs uniform 6-bit</td><td><strong>16.74% lower == BETTER</strong></td><td><strong>16.12% lower == BETTER</strong></td><td><strong>+0.47 pp</strong></td><td><strong>+0.42 pp</strong></td><td><strong>+0.21 pp</strong></td></tr>
  </tbody>
</table>

Paired on the same evaluation batches, SAGE-PATHS is closer to the bf16 source than SAGE-ARCS in **42 of 48** batches on full-vocabulary KL and **43 of 48** on top-1 agreement; against SAGE, in **48 of 48** on full-vocabulary KL; against the uniform 6-bit build, in **42 of 48** on full-vocabulary KL and **39 of 48** on top-1.

In the other direction, on the generalist corpus SAGE was built around — 128 sequences × 1024 input tokens, 32 evaluation batches, 130,944 scored next-token positions:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>top-5 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.010153</td><td>0.951865</td><td>0.999397</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.014043</td><td>0.941502</td><td>0.999359</td></tr>
    <tr class="ship"><td class="lead">SAGE-PATHS</td><td>0.012280</td><td>0.944503</td><td>0.999404</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td><strong>0.008508</strong></td><td><strong>0.956401</strong></td><td>0.999473</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist</u></em></strong>, vs SAGE-ARCS</td><td>SAGE-PATHS better, 29/32</td><td>SAGE-PATHS better, 28/32</td><td>saturated, <code>ns</code></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist</u></em></strong>, vs SAGE</td><td>SAGE better, 29/32</td><td>SAGE better, 32/32</td><td>saturated, <code>ns</code></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist</u></em></strong>, vs uniform 6-bit</td><td>uniform 6-bit better, 29/32</td><td>uniform 6-bit better, 32/32</td><td>saturated, <code>ns</code></td></tr>
  </tbody>
</table>

<details>
<summary><strong>Reinvestment changes where fidelity is kept, not what the model can do</strong></summary>

Pruning an unused modality and reinvesting its bytes does not make a quantization uniformly better. It changes where, under a constrained precision budget, the quantized model most faithfully tracks its bf16 source. These results come from different corpora with different measurements and should not be compared as though they were one common benchmark.

</details>

SAGE-PATHS reduces full-vocabulary KL by **9.1% against SAGE-ARCS** and **26.9% against SAGE** on the agentic basis it targets. On the generalist basis it sits between the two: below SAGE-ARCS (`0.012280` vs `0.014043`), above SAGE (`0.010153`). Against the uniform 6-bit build it is **16.7% lower** on the agentic basis while 1.75 GB smaller, and 44.3% higher on the generalist basis: spending more bytes uniformly protects generalist text better.

If your workload is closer to the stock/generalist distribution, **SAGE is the more appropriate build**. If you need image or video input, use **SAGE** or **SAGE-ARCS**: SAGE-PATHS has no vision tower.

### The relative advantage <u>grows</u> with *long context*

Sixteen held-out agent-coding sessions, each 131,072 tokens long, scored against one bf16 reference. Each column is the head-KL averaged over the first N tokens of the same 16 sequences:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">build</th>
      <th>2,048</th>
      <th>8,192</th>
      <th>32,768</th>
      <th>131,072</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.900949</td><td>1.286632</td><td>1.418693</td><td>1.151324</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.714416</td><td>1.034159</td><td>1.137440</td><td>0.937825</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td>0.641301</td><td><strong>0.925992</strong></td><td><strong>1.011646</strong></td><td><strong>0.826168</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td><strong>0.621535</strong></td><td>1.062724</td><td>1.204114</td><td>0.982294</td></tr>
    <tr class="delta"><td class="lead">lower than SAGE-ARCS by</td><td><strong>10.23%</strong></td><td><strong>10.46%</strong></td><td><strong>11.06%</strong></td><td><strong>11.91%</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring, vs SAGE-ARCS</td><td><strong>15/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td></tr>
    <tr class="delta"><td class="lead">change vs uniform 6-bit</td><td>+3.18%</td><td><strong>−12.87%</strong></td><td><strong>−15.98%</strong></td><td><strong>−15.89%</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring, vs uniform 6-bit</td><td>6/16</td><td><strong>16/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td></tr>
  </tbody>
</table>

Head-KL; <u>lower is better</u>. The averages are not monotonic in length: for all four builds, positions beyond 32K diverge less on average than the first 32K of these sessions.

Against the uniform 6-bit build the answer depends on the prefix: it is slightly ahead of SAGE-PATHS over the first 2,048 tokens (SAGE-PATHS +3.2%, lower on 6 of 16) and behind it from 8,192 on (−12.9%, and −15.9% at 131,072, 16 of 16), so its short-context lead does not carry to long sessions. The out-of-sample sets below show where that lead comes from.

Top-1 and top-5 over the full 128K show the same pattern:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>top-1 @128K ↑</th>
      <th>top-5 @128K ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.8341</td><td>0.9163</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.8528</td><td>0.9313</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td><strong>0.8619</strong></td><td><strong>0.9391</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.8510</td><td>0.9282</td></tr>
  </tbody>
</table>

At 128K, SAGE-PATHS is ahead of SAGE-ARCS on head-KL, top-1 and top-5 in **16 of 16** sequences each, ahead of SAGE on head-KL in 16 of 16, and ahead of the uniform 6-bit build on all three in **16 of 16**. SAGE-ARCS's advantage over SAGE stays roughly constant with length (20.7% at 2K, 18.5% at 128K). **SAGE-PATHS's advantage over SAGE-ARCS grows with context**, from 10.2% at 2K to 11.9% at 128K, which is relevant for long-running agentic sessions.

### Out of sample

Agent trajectories from three model families never in the calibration corpus (DeepSeek-V4-Flash, MiniMax-M2.5, Qwen3.5-122B), fifteen sequences at each length — a different set at each — each length scored against its own bf16 reference:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:20%;">build</th>
      <th>head-KL 2K ↓</th>
      <th>8K</th>
      <th>16K</th>
      <th>top-1 2K ↑</th>
      <th>8K</th>
      <th>16K</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.4912</td><td>2.2771</td><td>2.0930</td><td>0.8390</td><td>0.7136</td><td>0.7366</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.4218</td><td>1.9745</td><td>1.7530</td><td>0.8461</td><td>0.7335</td><td>0.7597</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td>0.3904</td><td><strong>1.7894</strong></td><td><strong>1.5707</strong></td><td>0.8492</td><td><strong>0.7449</strong></td><td><strong>0.7724</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td><strong>0.3250</strong></td><td>2.0137</td><td>1.8244</td><td><strong>0.8744</strong></td><td>0.7410</td><td>0.7607</td></tr>
    <tr class="delta"><td class="lead">SAGE-PATHS vs SAGE-ARCS</td><td><strong>−7.44%</strong></td><td><strong>−9.38%</strong></td><td><strong>−10.40%</strong></td><td>+0.31 pp</td><td><strong>+1.15 pp</strong></td><td><strong>+1.27 pp</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring SAGE-PATHS</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td>12/15</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td></tr>
    <tr class="delta"><td class="lead">SAGE-PATHS vs uniform 6-bit</td><td>+20.11% (<code>ns</code>)</td><td><strong>−11.14%</strong></td><td><strong>−13.91%</strong></td><td>−2.52 pp</td><td>+0.39 pp</td><td><strong>+1.17 pp</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring SAGE-PATHS</td><td>5/15</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td>4/15</td><td>9/15</td><td><strong>13/14</strong></td></tr>
  </tbody>
</table>

SAGE-PATHS is closer to the bf16 source than SAGE-ARCS on **every one of the 45 sequences**, and the lineage keeps its order — SAGE-PATHS, then SAGE-ARCS, then SAGE — on 15 of 15 sequences at every length. Against the uniform 6-bit build it is lower at 8K and 16K on every sequence. `ns` marks a KL difference below the multiplicity-corrected bar over the evaluation's 12 pre-registered tests (|t| > 3.42, 15 sequences); bold marks one that clears it.

At 2K the uniform 6-bit build is ahead on KL (+20.1% for SAGE-PATHS, t +2.16, below the bar) and on top-1 (−2.52 pp, t −4.41). That lead is **not short context**: ten of the fifteen 2K sequences open with the same 1,746-token agent system prompt, and they are exactly the ten where the uniform build leads; on them SAGE-PATHS's head-KL is 93% higher. On the other five, and on random 2048-token windows of off-pool text (next section), SAGE-PATHS leads.

### SWE replication

Two further corpora of software-engineering agent text from outside the calibration pool, frozen before scoring (built twice, byte-identical):

* **S** — other model families (DeepSeek-V4-Flash, MiniMax-M2.5, Qwen3.5-122B) on the same task pools as the calibration text;
* **X** — other models (Claude 4.5 Sonnet, GPT-5-mini) on other repositories and languages (Go and C; Python).

Each stratum is scored as 48 evaluation batches of random 2048-token windows, 24 document heads of 8192 tokens, and the first 2048 tokens of those same heads, every set against its own bf16 reference:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:20%;">comparison</th>
      <th>set</th>
      <th>KL change</th>
      <th>t</th>
      <th>lower</th>
      <th>top-1</th>
      <th>t</th>
      <th>verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="section" colspan="8">SAGE-PATHS vs uniform 6-bit</td></tr>
    <tr><td class="lead">S, 2048 windows</td><td>48 batches</td><td><strong>−15.44%</strong></td><td>−6.41</td><td>41/48</td><td>+0.351 pp</td><td>+4.34</td><td><strong>replicates</strong></td></tr>
    <tr><td class="lead">X, 2048 windows</td><td>48 batches</td><td><strong>−17.90%</strong></td><td>−11.97</td><td>45/48</td><td>+0.881 pp</td><td>+9.55</td><td><strong>replicates</strong></td></tr>
    <tr><td class="lead">S, 8192 heads</td><td>24 sequences</td><td><strong>−11.67%</strong></td><td>−5.16</td><td>24/24</td><td>+0.067 pp</td><td>+0.24</td><td><strong>replicates</strong></td></tr>
    <tr><td class="lead">X, 8192 heads</td><td>24 sequences</td><td><strong>−15.06%</strong></td><td>−10.27</td><td>23/24</td><td>+1.222 pp</td><td>+7.15</td><td><strong>replicates</strong></td></tr>
    <tr><td class="lead">S, 2048 heads</td><td>24 sequences</td><td>+16.53%</td><td>+2.58</td><td>8/24</td><td>−2.691 pp</td><td>−6.75</td><td>does not replicate</td></tr>
    <tr><td class="lead">X, 2048 heads</td><td>24 sequences</td><td>−16.29%</td><td>−3.64</td><td>16/24</td><td>+0.065 pp</td><td>+0.18</td><td>does not replicate</td></tr>
  </tbody>
</table>

Four of the six pre-registered tests replicate on text the builds never saw (multiplicity-corrected over the evaluation's 14 tests: |t| > 3.068 for windows, > 3.245 for heads): SAGE-PATHS stays 15-18% below the uniform build on random windows and 12-15% below it on 8,192-token heads. On X each lands at or beyond the size measured before (−16.12% on the agentic basis for windows, −11.1% out of sample at 8K for heads); on S they replicate at a similar size.

The two 2048-token head tests asked whether the uniform build's 2K lead out of sample would replicate, and it does not: on S the uniform build is still ahead (+16.5%, t +2.58, below the bar), and on X SAGE-PATHS is ahead instead (−16.3%, t −3.64). The lead follows **shared scaffold text at the head of the document**, not length: 16 of S's 24 heads open with the same 1,746-token agent system prompt, and on those the uniform build's head-KL is about half SAGE-PATHS's (SAGE-PATHS lower on 0 of 16). On S's other 8 heads (8 of 8) and on X's Python heads (12 of 12), SAGE-PATHS leads; X's C heads, 10 of 12 of which share a long instruction template, lean to the uniform build on 8 of those 10. Inside the 8,192-token documents that open with the system prompt, the deficit runs to about position 2,400 and then reverses.

Off-pool text is harder for every build, and the calibration-free uniform build's KL rises about as much as SAGE-PATHS's (×1.135 vs ×1.145 on S, ×1.608 vs ×1.574 on X), so the agentic edge is not calibration overfit:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">build</th>
      <th>in-pool head-KL / top-1</th>
      <th>S head-KL (×) / top-1</th>
      <th>X head-KL (×) / top-1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.37880 / 0.9103</td><td>0.43943 (×1.160) / 0.9048</td><td>0.61743 (×1.630) / 0.8747</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.30737 / 0.9199</td><td>0.36071 (×1.174) / 0.9142</td><td>0.49876 (×1.623) / 0.8862</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td><strong>0.27944 / 0.9235</strong></td><td><strong>0.31983 (×1.145) / 0.9179</strong></td><td><strong>0.43987 (×1.574) / 0.8938</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.33313 / 0.9188</td><td>0.37824 (×1.135) / 0.9144</td><td>0.53580 (×1.608) / 0.8850</td></tr>
  </tbody>
</table>

<sub>In-pool is the agentic basis above; S and X are their 2048-token random windows.</sub>

### The stages, on the corpus this build targets

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">rung</th>
      <th>bpw</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">targeted precision, step 1</td><td>5.9945</td><td>0.302698</td><td>0.9208</td></tr>
    <tr class="ship"><td class="lead">targeted precision, step 2</td><td><strong>6.1366</strong></td><td><strong>0.279441</strong></td><td><strong>0.9235</strong></td></tr>
  </tbody>
</table>

> The targeted-precision steps were selected on this same 96-sequence evaluation, so the generalist result below, and the out-of-sample sets and SWE replication above, are the checks that played no part in selection.

Both rows are the text-only build after the vision tower's bytes have been reinvested; the second step places more of that headroom where the agentic workload is most sensitive. It lowers head-KL on the target corpus by **7.68%** (paired t −4.15, 40/48 batches).

On the generalist corpus — which was not a selection criterion — the same step moves in the same direction: `0.012988 → 0.012280` (t −1.69, 28/32 batches).

**Unlike the calibration shift from SAGE to SAGE-ARCS, the targeted PATHS step does not trade one distribution for the other.** The remaining stock/generalist gap to SAGE is already present at step 1, and this step does not widen it.

### Where the divergence actually is — the same numbers, stratified

A single pooled KL hides a fairly wide range of difficulty inside this corpus. If the 48 evaluation batches are sorted by **how uncertain the bf16 source itself is** and split into quartiles:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:15%;">content difficulty</th>
      <th>bf16 perplexity</th>
      <th>SAGE head-KL</th>
      <th>SAGE-ARCS head-KL</th>
      <th class="ship"><strong>SAGE-PATHS head-KL</strong></th>
      <th class="ship"><strong>SAGE-PATHS top-1</strong></th>
      <th>bf16 mass in its own top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead"><strong>easiest quarter</strong></td><td>2.8</td><td>0.0246</td><td>0.0173</td><td class="ship"><strong>0.0178</strong></td><td class="ship">0.9705</td><td>0.9980</td></tr>
    <tr><td class="lead">second</td><td>6.3</td><td>0.1771</td><td>0.1406</td><td class="ship"><strong>0.1246</strong></td><td class="ship">0.9453</td><td>0.9979</td></tr>
    <tr><td class="lead">third</td><td>19.5</td><td>0.4288</td><td>0.3397</td><td class="ship"><strong>0.3103</strong></td><td class="ship">0.9171</td><td>0.9973</td></tr>
    <tr><td class="lead"><strong>hardest quarter</strong></td><td>136.9</td><td>0.8847</td><td>0.7318</td><td class="ship"><strong>0.6651</strong></td><td class="ship">0.8609</td><td>0.9965</td></tr>
    <tr class="total"><td class="lead"><strong>pooled</strong></td><td>14.8</td><td>0.3788</td><td>0.3074</td><td class="ship"><strong>0.2794</strong></td><td class="ship"><strong>0.9235</strong></td><td>0.9974</td></tr>
  </tbody>
</table>

Across a **49× range in source-model perplexity**, SAGE-PATHS's head-KL changes by roughly **37×**, from 0.0178 to 0.6651. Over the same range its top-1 agreement moves from 0.970 to 0.861, while the bf16 reference's own mass in its top-1024 stays between 0.9965 and 0.9980.

<details>
<summary><strong>Why the pooled KL is not a property of the model</strong></summary>

The pooled `0.2794` should therefore not be treated as an intrinsic property of the quantized model. It is a result measured on a particular corpus whose quartiles range from perplexity 2.8 to 136.9.

The same build reports 0.0178 on the easiest quartile and 0.6651 on the hardest. KL values are only directly comparable **within the same evaluation corpus**. Top-1, top-5, and the candidate's mass on the bf16 top-1024 provide more intuitive context when reasoning about likely model behavior.

</details>

Against SAGE-ARCS, SAGE-PATHS is better in **42 of 48 evaluation batches**, and against SAGE in **48 of 48**:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">quartile</th>
      <th>batches favouring SAGE-PATHS over SAGE-ARCS</th>
      <th class="delta">head-KL change vs SAGE-ARCS</th>
      <th>batches favouring SAGE-PATHS over SAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">easiest</td><td>11 / 12</td><td class="delta">+2.7%</td><td>12 / 12</td></tr>
    <tr><td class="lead">second</td><td>11 / 12</td><td class="delta">−11.4%</td><td>12 / 12</td></tr>
    <tr><td class="lead">third</td><td>9 / 12</td><td class="delta">−8.7%</td><td>12 / 12</td></tr>
    <tr><td class="lead">hardest</td><td>11 / 12</td><td class="delta">−9.1%</td><td>12 / 12</td></tr>
  </tbody>
</table>

On the easiest content the two are effectively level: SAGE-PATHS is lower in 11 of 12 batches, but a single batch moves the other way enough to put its pooled head-KL 2.7% higher (+0.0005 nats). The gain is roughly constant everywhere else (−8.7% to −11.4%). The largest **absolute** gain is on the hardest content (−0.0667 nats).

### Every metric, one basis

For completeness, here are the main metrics together on the agentic basis — the same 48 evaluation batches / 196,512 scored next-token positions from 96 × 2048-token input sequences, and the same bf16 reference cache:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:15%;">build</th>
      <th>bpw</th>
      <th>full-vocab KL ↓</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>top-5 ↑</th>
      <th>ΔCE (target <strong>0</strong>)</th>
      <th>ppl ratio (target <strong>1</strong>)</th>
      <th>candidate mass on bf16 top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>6.0836</td><td>0.424653</td><td>0.378796</td><td>0.910321</td><td>0.969539</td><td><strong>−0.003416</strong></td><td><strong>0.99659</strong></td><td>0.986682</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>6.0836</td><td>0.341485</td><td>0.307367</td><td>0.919852</td><td>0.974968</td><td>−0.003928</td><td>0.99608</td><td>0.989242</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td>6.1366</td><td><strong>0.310274</strong></td><td><strong>0.279441</strong></td><td><strong>0.923460</strong></td><td><strong>0.977131</strong></td><td>−0.006307</td><td>0.99371</td><td><strong>0.990208</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>6.6486</td><td>0.372663</td><td>0.333126</td><td>0.918758</td><td>0.972963</td><td>−0.020582</td><td>0.97963</td><td>0.988059</td></tr>
  </tbody>
</table>

And on the stock basis — 32 evaluation batches / 130,944 scored next-token positions from 128 × 1024-token input sequences:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:15%;">build</th>
      <th>bpw</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>top-5 ↑</th>
      <th>ΔCE (target <strong>0</strong>)</th>
      <th>ppl ratio (target <strong>1</strong>)</th>
      <th>candidate mass on bf16 top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>6.0836</td><td>0.010153</td><td>0.951865</td><td>0.999397</td><td>+0.002709</td><td>1.00271</td><td>0.980571</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>6.0836</td><td>0.014043</td><td>0.941502</td><td>0.999359</td><td>+0.005103</td><td>1.00512</td><td>0.980359</td></tr>
    <tr class="ship"><td class="lead">SAGE-PATHS</td><td>6.1366</td><td>0.012280</td><td>0.944503</td><td>0.999404</td><td>+0.003280</td><td>1.00329</td><td>0.980424</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>6.6486</td><td><strong>0.008508</strong></td><td><strong>0.956401</strong></td><td>0.999473</td><td><strong>−0.002324</strong></td><td><strong>0.99768</strong></td><td><strong>0.980694</strong></td></tr>
  </tbody>
</table>

**ΔCE** is the change in cross-entropy against the bf16 source, in nats per token. **ppl ratio** expresses the same quantity as a perplexity multiplier.

For a fidelity claim, the target is **ΔCE = 0**, not simply the minimum numerical value. That distinction matters on the agentic basis because all four values are negative.

> All four quantizations have slightly lower cross-entropy than the bf16 source on the agentic corpus, with perplexity ratios of 0.997, 0.996, 0.994 and 0.980 (uniform 6-bit).

A quantized model can assign the observed tokens slightly better than its source while still sitting further from the source's overall distribution. Quantization noise is not zero-mean with respect to a particular corpus.

SAGE-PATHS's pooled ΔCE is the furthest from zero of the three SAGE builds; the uniform 6-bit build's is further still (`−0.020582`). Per batch, SAGE-PATHS is closer to `ΔCE = 0` than SAGE-ARCS in **30/48 evaluation batches (`p = 0.11`)**, so ΔCE does not separate the two. **KL, not ΔCE, is the fidelity metric here.**

### The paired comparison

SAGE-PATHS relative to SAGE-ARCS, paired on identical evaluation batches. `favor` counts paired evaluation units where SAGE-PATHS is closer to the bf16 source under the stated metric; ties are excluded from the denominator.

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">metric</th>
      <th>Δ</th>
      <th>t</th>
      <th class="delta">favor</th>
      <th>sign p</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="section" colspan="5">Agentic basis</td></tr>
    <tr><td class="lead">full-vocabulary KL</td><td><strong>−0.031210</strong></td><td>−4.58</td><td class="delta"><strong>42/48</strong></td><td>1.0e-07</td></tr>
    <tr><td class="lead">head-KL</td><td><strong>−0.027926</strong></td><td>−4.74</td><td class="delta"><strong>42/48</strong></td><td>1.0e-07</td></tr>
    <tr><td class="lead">top-1 agreement</td><td><strong>+0.003608</strong></td><td>+7.37</td><td class="delta"><strong>43/48</strong></td><td>1.4e-08</td></tr>
    <tr><td class="lead">top-5 agreement</td><td><strong>+0.002163</strong></td><td>+4.96</td><td class="delta"><strong>31/39</strong></td><td>2.9e-04</td></tr>
    <tr><td class="lead">mass on bf16 top-1024</td><td>+0.000966</td><td>+3.72</td><td class="delta">35/48</td><td>0.0021</td></tr>
    <tr><td class="lead">tail term</td><td>−0.000239</td><td>−1.38</td><td class="delta">28/48</td><td>0.31 — <code>ns</code></td></tr>
    <tr><td class="lead">|ΔCE| toward 0</td><td>—</td><td>—</td><td class="delta">30/48</td><td>0.11 — <code>ns</code></td></tr>
    <tr><td class="section" colspan="5">Stock basis</td></tr>
    <tr><td class="lead">head-KL</td><td>−0.001763</td><td>−5.29</td><td class="delta">29/32</td><td>2.6e-06</td></tr>
    <tr><td class="lead">top-1 agreement</td><td>+0.003001</td><td>+7.30</td><td class="delta">28/32</td><td>1.9e-05</td></tr>
    <tr><td class="lead">top-5 agreement</td><td>+0.000046</td><td>+1.00</td><td class="delta">11/19</td><td>0.65 — <strong><code>ns</code>, saturated</strong></td></tr>
    <tr><td class="section" colspan="5">Long context, 128K (16 sequences)</td></tr>
    <tr><td class="lead">head-KL</td><td><strong>−0.111657</strong></td><td>−7.32</td><td class="delta"><strong>16/16</strong></td><td>3.1e-05</td></tr>
    <tr><td class="lead">top-1 agreement</td><td><strong>+0.009143</strong></td><td>+12.21</td><td class="delta"><strong>16/16</strong></td><td>3.1e-05</td></tr>
    <tr><td class="lead">top-5 agreement</td><td><strong>+0.007806</strong></td><td>+8.01</td><td class="delta"><strong>16/16</strong></td><td>3.1e-05</td></tr>
  </tbody>
</table>

The p-values above are reported as produced by the evaluation harness. This expanded table is descriptive and should not be read as a multiplicity-corrected confirmatory family unless a correction is stated separately.

Two details are worth keeping in mind when reading this table:

* **The gain is inside the retained support.** The tail term and |ΔCE| do not separate the two builds; head-KL, full-vocabulary KL and top-1 do. SAGE-PATHS improves on SAGE-ARCS where the bf16 source puts its probability, not in the long tail.

* **Unlike SAGE-ARCS against SAGE, the stock basis moves in the same direction.** SAGE-PATHS is better than SAGE-ARCS on generalist text as well (29/32), so the agentic gain was not bought by giving up the generalist distribution. Against SAGE itself, the generalist basis still favors SAGE.

### Divergence within a sequence

Head-KL rises with token position inside a sequence, so a pooled figure averages over another axis that is not flat. Fitted slope, in nats per 100 tokens:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>agentic (2K)</th>
      <th>long-context (128K)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>+0.027919 ± 0.007349</td><td>−0.000216 ± 0.000307</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>+0.022411 ± 0.005963</td><td>−0.000150 ± 0.000263</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td><strong>+0.020393</strong> ± 0.005505</td><td>−0.000149 ± 0.000234</td></tr>
  </tbody>
</table>

On the agentic basis SAGE-PATHS has the lowest fitted slope of the three. The standard errors overlap, however, so this should be treated as a directional observation rather than an established difference.

Over 128K the within-sequence slope is flat within its standard error for all three builds, so it carries no ordering there.

## How fidelity was measured

**head-KL** is `KL(bf16 ‖ quantized)` over the bf16 model's **top-1024 logits**, with both distributions renormalized over the retained support.

On the same evaluation basis:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">quantity</th>
      <th>value</th>
      <th>meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead"><strong>top-1 agreement</strong></td><td><strong>0.9235</strong></td><td>same next token as bf16 on <strong>92.3%</strong> of positions — what greedy decoding reads</td></tr>
    <tr><td class="lead"><strong>top-5 agreement</strong></td><td><strong>0.9771</strong></td><td>bf16's choice is in the candidate's top 5 on <strong>97.7%</strong> of positions</td></tr>
    <tr><td class="lead"><strong>candidate mass on bf16 top-1024</strong></td><td><strong>0.9902</strong></td><td>SAGE-PATHS places <strong>99.0%</strong> of its probability on the tokens bf16 ranks in its top 1024</td></tr>
    <tr><td class="lead"><strong>perplexity ratio</strong></td><td><strong>0.9937</strong></td><td>predictive cross-entropy on the observed text differs from bf16 by approximately <strong>0.6% in perplexity ratio</strong></td></tr>
  </tbody>
</table>

The absolute level of KL is also strongly related to how uncertain the bf16 source itself is. On this basis, per-evaluation-unit head-KL correlates with the reference model's own cross-entropy at **r = +0.93**.

In this evaluation, agent transcripts — including tool output, diffs, error text, and mid-task reasoning — include content for which the source model itself is substantially less certain than on the easiest portions of the corpus.

A head-KL of 0.28 on this corpus and 0.012 on the generalist corpus therefore do not imply a 23× difference in quantization quality. They are measurements on distributions with very different difficulty. Direct comparisons are valid only within the same evaluation basis.

Those 1024 entries represent 0.41% of the 248,320-token vocabulary. On the agentic basis they contain **99.74%** of the bf16 reference probability mass.

The truncated and full-vocabulary metrics rank the three builds identically here. **head-KL was used during candidate selection and ranking; full-vocabulary KL is reported as the absolute fidelity audit.** On this corpus, head-KL omits about 10% of the total divergence because it excludes the mass-shift contribution described below.

Every row was scored against a **common bf16 reference cache**, with identical settings within each basis:

<table class="sage-table kv" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;"></th>
      <th>agentic basis</th>
      <th>stock basis</th>
      <th>long-context</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">input sequences × length</td><td>96 × 2048</td><td>128 × 1024</td><td>16 × 131,072</td></tr>
    <tr><td class="lead">scored next-token positions</td><td>196,512</td><td>130,944</td><td>2,097,136</td></tr>
    <tr><td class="lead">batch size</td><td>2 (48 batches)</td><td>4 (32 batches)</td><td>1 (16 batches)</td></tr>
    <tr><td class="lead">top-k</td><td>full 248,320 + 1024</td><td>1024</td><td>1024</td></tr>
    <tr><td class="lead">seed</td><td>123</td><td>123</td><td>fixed corpus file</td></tr>
  </tbody>
</table>

> [!Tip] Comparisons are valid **within** each basis, not across them. <u>KL is corpus-relative</u>: a value from one evaluation distribution cannot be compared directly with a value from another.

### Full-vocabulary audit

The full 248,320-entry reference distribution was stored in fp32 and the divergence computed directly, then checked against the truncated metric.

As an instrument control, the head terms are computed two ways — from the stored top-1024 logits and from the full log-probabilities — and must agree; `identity_gap OK` (≤ 4.2e-05 nats/token) was obtained for every record. The three decomposition terms below are per-token means and sum to the directly measured full-vocabulary KL to within 1.6e-08.

On the agentic basis, the tail outside the bf16 top-1024 contributes **0.001477 nats** for SAGE-PATHS, about **0.5%** of its total divergence; the bf16 reference places **99.74%** of its own probability inside that set.

The truncated metric (head-KL) omits the mass-shift term, so on this corpus it understates the absolute divergence by **9.9%** (SAGE-PATHS), **10.0%** (SAGE-ARCS) and **10.8%** (SAGE). It produces the same ordering as full-vocabulary KL in every comparison here.

The following values are in nats per token on the agentic basis. The three terms correspond to divergence inside the retained support, the cost of the candidate assigning a different amount of mass to that support, and the contribution from everything outside it.

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">build</th>
      <th>full-vocab KL</th>
      <th>inside retained support</th>
      <th>mass-shift</th>
      <th>tail</th>
      <th>= sum</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.424653</td><td>0.376982 (88.8%)</td><td>0.045874 (10.8%)</td><td>0.001798 (0.4%)</td><td>0.424653</td></tr>
    <tr><td class="lead">SAGE-ARCS</td><td>0.341485</td><td>0.305708 (89.5%)</td><td>0.034061 (10.0%)</td><td>0.001716 (0.5%)</td><td>0.341485</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-PATHS</strong></td><td>0.310274</td><td>0.277960 (89.6%)</td><td>0.030837 (9.9%)</td><td>0.001477 (0.5%)</td><td>0.310274</td></tr>
    <tr class="delta"><td class="lead">paired, SAGE-PATHS lower than SAGE-ARCS</td><td>42/48</td><td>43/48, p = 1.4e-08</td><td>32/48, p = 0.029</td><td>28/48, p = 0.31</td><td>&nbsp;</td></tr>
  </tbody>
</table>

Against SAGE-ARCS, the improvement is concentrated **inside the retained support**: that term favors SAGE-PATHS in 43 of 48 batches; the mass-shift term leans the same way more weakly (32/48, p = 0.029), and the tail term is not separated (28/48). Against SAGE, all three terms favor SAGE-PATHS (48/48, 43/48, 31/48).

<details>
<summary><strong>How the three terms add up</strong></summary>

Each term is the per-token mean over all **196,512 scored next-token positions**, so the three add up to full-vocabulary KL exactly: 0.277960 + 0.030837 + 0.001477 = 0.310274 for SAGE-PATHS. About 90% of its divergence is inside the tokens bf16 considers plausible, about 10% is probability moved out of that set, and under 0.5% is in the remaining vocabulary.

</details>

## Honest limits

* **Text only.** The vision tower is removed; this checkpoint cannot process images or video. Use SAGE or SAGE-ARCS for multimodal input.

* **SAGE is more faithful on the stock/generalist distribution.** Stock-basis head-KL is `0.012280` for SAGE-PATHS versus `0.010153` for SAGE. SAGE-PATHS is better than SAGE-ARCS there (`0.014043`), but SAGE keeps its home-ground lead.

* **A uniform 6-bit build is more faithful on generalist text, and over long shared prompts.** On the generalist basis its head-KL is `0.008508` against `0.012280` for SAGE-PATHS (lower in 29 of 32 batches), for 1.75 GB more. Over the opening of documents that begin with a long prompt shared across sessions (an agent system prompt or instruction template; for the system prompt measured here, roughly the first 2,400 tokens), it also tracks the bf16 source more closely. Past that point, and on random windows and long documents in every agentic set measured here, SAGE-PATHS leads.

* **The reinvested tensors were selected on agentic fidelity.** The generalist basis was not a selection criterion and the 964-question accuracy set was run only after the build; neither contradicts the agentic result.

* **head-KL is not full-vocabulary KL.** It uses an identical top-1024 truncation for every build and omits the mass-shift term: on the agentic corpus it is 9.9% (SAGE-PATHS) below full-vocabulary KL. Both metrics produce the same ordering in the comparisons reported here.

* **ΔCE is not a fidelity metric and is negative on the agentic basis.** All four builds predict that corpus slightly better than the bf16 source does, and SAGE-PATHS's pooled ΔCE is the furthest from zero of the three SAGE builds. Per batch it does not separate SAGE-PATHS from SAGE-ARCS (30/48, `p = 0.11`). It is reported for completeness rather than used as evidence of fidelity.

* **Top-5 agreement is saturated on the stock basis and informative on the others.** At about 0.9994 it cannot discriminate on stock (`ns`, 11/19). On the agentic basis and at long context it separates the builds clearly.

* **Within-sequence position slopes are directional only.** SAGE-PATHS has the lowest fitted slope on the agentic basis, but the standard errors overlap; over 128K the slopes are flat within noise for all three builds.

* **48 evaluation batches is a modest sample.** The SAGE-PATHS-versus-SAGE-ARCS difference is consistent (42/48 for full-vocabulary KL, 16/16 sequences at 128K), but smaller differences among closely related builds are not reliably resolved at this sample size.

* **Fidelity is not task accuracy.** They are separate claims and were measured separately. Across 964 benchmark questions with thinking disabled, deterministic sampling and MTP enabled, on oMLX 0.7.0, SAGE-PATHS scores 767 against 772 for its parent, the same build before the second targeted-precision step (13 vs 18 discordant questions, `p = 0.47`, paired exact McNemar). SAGE and SAGE-ARCS were benchmarked on earlier oMLX versions, and the version change alone alters about a fifth of all generations, so totals are not compared across versions.

* **Benchmark totals are not a ranking.** On oMLX 0.7.0, livecodebench changed 17 verdicts when the same weights were benchmarked again. Single-benchmark differences of that size should be treated as unresolved.

* **The bf16 MTP head costs 0.1431 bpw and is retained for speculative decoding.** It does not participate in the scored forward path, so the pre-graft and bf16-MTP rungs agree in every fidelity measurement above.

* **Fidelity is measured against the bf16 source.** A quantization that reproduces its source more faithfully also reproduces the source model's mistakes more faithfully.

## Quantization details

* **Model type:** `qwen3_5`, text only
* **Parameters:** 27.32 B (vision tower removed)
* **Bits:** 5-bit base, with 150 per-tensor overrides placed by measured sensitivity and tensor analysis
* **Group size:** 64
* **Format:** MLX safetensors
* **MTP:** preserved and **restored to bf16** — 15 stored tensors, 15 parameters
* **Stored tensors:** 1670

## Usage

This build is made for [oMLX](https://github.com/jundot/omlx): it was quantized through oMLX's `oQe` backend and is meant to be served by it, with MTP on.

### Serve with oMLX

Install oMLX with Homebrew, or see the [oMLX repository](https://github.com/jundot/omlx) for the other ways to install and run it:

```bash
brew install jundot/omlx/omlx
```

`omlx serve` discovers models as sub-directories of `--model-dir` (default `~/.omlx/models`); the directory name becomes the model id:

```bash
hf download notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp --local-dir ~/.omlx/models/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp
omlx serve --model-dir ~/.omlx/models --port 8000
```

Then open the oMLX admin dashboard (`http://127.0.0.1:8000/admin`), open `Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp`'s model settings and turn on **Lightning MTP**.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp", "messages": [{"role": "user", "content": "Hello"}]}'
```

### mlx-lm / mlx-vlm

SAGE-PATHS also loads outside oMLX, as a text-only model, under a standard `mlx-lm` or `mlx-vlm` stack. Verified 2026-10-09 with mlx-lm 0.32.0 and mlx-vlm 0.7.6, and with the versions oMLX 0.7.0 bundles (mlx-lm 0.31.4.dev, mlx-vlm 0.7.1):

```bash
pip install mlx-lm
python -m mlx_lm.generate --model notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp --max-tokens 256 --prompt "Hello"
```

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp --max-tokens 256 --prompt "Hello"
```

mlx-vlm reports that the checkpoint has no vision-tower weights and disables that module, so image input is not available. Neither stack runs the bf16 MTP head: Lightning MTP (speculative decoding) is available only through oMLX.

## Attribution

Base model © the Qwen team, Apache-2.0 — see `LICENSE`.

Quantization performed with [oMLX](https://github.com/jundot/omlx) `0.6.4`; validated and served on `0.7.0`. SAGE-PATHS is an adapter layer over oMLX's `oQe` backend.

> Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork*. August 2026.
> https://qwen.ai/blog?id=qwen3.8