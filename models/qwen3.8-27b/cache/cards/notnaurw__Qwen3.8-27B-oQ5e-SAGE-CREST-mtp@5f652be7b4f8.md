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

<h1 align="center">Qwen3.8-27B — oQ5e / SAGE-CREST (MLX, 5-bit, MTP preserved, bf16)</h1>

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

**SAGE-CREST** (**C**lass-**R**anked **E**levation of **S**elected **T**ensors) extends and <u>concludes</u> the fidelity-first SAGE quantization research pipeline around one question:

> **If the precision budget is increased to nearly the size of a uniform 6-bit checkpoint, can targeted allocation preserve the bf16 model's behavior more faithfully than spending those bytes uniformly?**

The SAGE series has explored this problem from several directions. SAGE established the fidelity-first pipeline. SAGE-ARCS adapted its sensitivity measurements toward agentic workloads. SAGE-PATHS removed the unused vision tower and reinvested that storage in the text model. SAGE-CREST takes the next step: it increases the available budget and concentrates the additional precision in tensors that show the greatest benefit on the targeted workload. **SAGE-CREST is a byte-wise <u>Pareto-efficient</u> candidate.**

Like SAGE-PATHS, **vision is pruned.** As such, SAGE-CREST is a **text-only model**. Its precision choices are guided by measurements against the bf16 source on a SWE- and terminal-oriented agentic corpus, then **verified in & out of sample**. This includes held-out data, different agent trajectories, and long-context sessions. The result is a checkpoint that tracks the bf16 source more closely on agentic text, while giving up image input entirely and, against a uniform 6-bit build, some generalist fidelity.

**SAGE-CREST is deliberately the <u>largest</u> build in the lineage, but remains smaller than a uniform 6-bit build with the same text-only architecture and bf16 MTP tensors.** SAGE-PATHS metrics demonstrated that it can outperform SAGE-ARCS on agentic fidelity in a *smaller, more efficient file size*. SAGE-CREST extends the budget, spending further precision on the text-model tensors where the agentic workload is most sensitive, up to *just under the size of a uniform 6-bit* quantization of the same model, and **comfortably outperforms that uniform 6-bit build on agentic KL**.

The capabilities of the bf16 source are unchanged for text. The objective is not to create capabilities absent from the source model, but to preserve the source model's text behavior more effectively and cohesively within the available precision budget.

**SAGE-CREST therefore complements its siblings rather than replacing them:** SAGE targets generalist fidelity, SAGE-ARCS redistributes it toward agentic workloads under the same budget, SAGE-PATHS converts unused architectural capacity into precision, and SAGE-CREST tests how far that agentic allocation goes when the budget rises to that of a uniform 6-bit build.

**22.50 GB (20.95 GiB) · 6.5879 bpw · MTP head preserved at bf16 (15 parameter tensors).**

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
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-PATHS-mtp">SAGE-PATHS</a></td><td>6.1366</td><td>20.96 GB</td><td>Agentic · Text-only</td><td><em>Pruned</em></td><td>Preserved</td></tr>
    <tr class="ship"><td class="lead"><strong><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp">SAGE-CREST</a></strong></td><td><strong>6.5879</strong></td><td><strong>22.50 GB</strong></td><td><strong>Agentic · Text-only</strong></td><td><strong><em>Pruned</em></strong></td><td><strong>Preserved</strong></td></tr>
  </tbody>
</table>
<sub>bpw is counted over the whole checkpoint, including the bf16 MTP head. SAGE-CREST is 1.54 GB larger than SAGE-PATHS and 0.21 GB smaller than a uniform 6-bit, group-64 quantization of the same text-only model with the same bf16 MTP head (22.71 GB, 6.6486 bpw), referred to below as <strong>uniform 6-bit</strong>.</sub>

<sub>The **uniform 6-bit** comparator is a standard MLX affine quantization (`mlx.core.quantize()`, group size 64). Its vision tower is removed, and the same bf16 MTP head is restored as in the other builds; those MTP tensors were verified byte-identical to SAGE-PATHS's.</sub>

> [!TIP]
> ***Four* checkpoints, *four* distinct research questions:**
> SAGE: given a fixed mixed-precision budget, how much bf16 fidelity can be recovered?
> SAGE-ARCS: holding the architecture and budget fixed, how does calibration distribution change where fidelity is preserved?
> SAGE-PATHS: holding the targeted workload fixed, what happens when unused modality capacity is removed and its budget reinvested into text-model precision?
> SAGE-CREST: holding the targeted workload and the pruned architecture fixed, can targeted precision beat a uniform 6-bit build while staying smaller than it?

## Why this exists

The main difference between SAGE-CREST and SAGE-PATHS is the size of the *budget*: SAGE-CREST spends more precision on the text-model tensors where the agentic workload is most sensitive, stopping **just under the size of a uniform 6-bit (with bf16 MTP)** quantization of the same model. To make the tradeoff visible, the four builds and that uniform 6-bit build are evaluated on the agentic basis, on the generalist basis SAGE was built around, and at long context:

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
    <tr><td class="lead">SAGE-PATHS</td><td>20.96 GB</td><td>0.2794</td><td>92.35%</td><td>0.01228</td><td>94.45%</td><td>0.8262</td><td>86.19%</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>22.50 GB</strong></td><td><strong>0.2433</strong></td><td><strong>92.80%</strong></td><td>0.01011</td><td>94.85%</td><td><strong>0.7324</strong></td><td><strong>87.00%</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>22.71 GB</td><td>0.3331</td><td>91.88%</td><td><strong>0.00851</strong></td><td><strong>95.64%</strong></td><td>0.9823</td><td>85.10%</td></tr>
  </tbody>
</table>

<sub>Lower head-KL is better. On the agentic basis, paired by evaluation batch, SAGE-CREST is ahead of every other build at |t| ≥ 5.3. The generalist basis is SAGE's own calibration distribution: there the uniform 6-bit build is the most faithful, and SAGE-CREST has nearly identical mean head-KL to SAGE (−0.4%, t −0.08; no resolved difference) though SAGE is lower in 24 of 32 batches. At 128K, paired by sequence, SAGE-CREST is ahead of every other build at |t| ≥ 7.98, in 16 of 16 sequences each; the uniform 6-bit build ranks fourth there, behind SAGE-ARCS.</sub>

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.310274</td><td>0.279441</td><td>0.923460</td><td>0.977131</td><td>0.990208</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>0.268643</strong></td><td><strong>0.243302</strong></td><td><strong>0.927984</strong></td><td><strong>0.979726</strong></td><td><strong>0.991499</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.372663</td><td>0.333126</td><td>0.918758</td><td>0.972963</td><td>0.988059</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis, vs SAGE-PATHS</td><td><strong>13.42% lower == BETTER</strong></td><td><strong>12.93% lower == BETTER</strong></td><td><strong>+0.45 pp</strong></td><td><strong>+0.26 pp</strong></td><td><strong>+0.13 pp</strong></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis, vs uniform 6-bit</td><td><strong>27.91% lower == BETTER</strong></td><td><strong>26.96% lower == BETTER</strong></td><td><strong>+0.92 pp</strong></td><td><strong>+0.68 pp</strong></td><td><strong>+0.34 pp</strong></td></tr>
  </tbody>
</table>

> [!Tip] On agentic KL, SAGE-ARCS and SAGE-PATHS already outperform the uniform 6-bit build; SAGE-CREST widens that margin.

Paired on the same evaluation batches, SAGE-CREST is closer to the bf16 source than SAGE-PATHS in **46 of 48** batches on full-vocabulary KL and **40 of 46** on top-1 agreement; against the uniform 6-bit build, in **46 of 48** on full-vocabulary KL and **40 of 47** on top-1.

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.012280</td><td>0.944503</td><td>0.999404</td></tr>
    <tr class="ship"><td class="lead">SAGE-CREST</td><td>0.010114</td><td>0.948520</td><td>0.999519</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td><strong>0.008508</strong></td><td><strong>0.956401</strong></td><td>0.999473</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist basis</u></em></strong>, vs SAGE-PATHS</td><td>SAGE-CREST better, 31/32</td><td>SAGE-CREST better, 31/32</td><td>saturated</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist basis</u></em></strong>, vs SAGE</td><td>SAGE better, 24/32 (mean −0.4%, <code>ns</code>)</td><td>SAGE better, 29/30</td><td>saturated, <code>ns</code></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist basis</u></em></strong>, vs uniform 6-bit</td><td>uniform 6-bit better, 26/32</td><td>uniform 6-bit better, 32/32</td><td>saturated, <code>ns</code></td></tr>
  </tbody>
</table>

<details>
<summary><strong>The goal is fidelity, not new capabilities</strong></summary>

Elevating selected tensors does not make a quantization uniformly better. It changes where, under a constrained precision budget, the quantized model most faithfully tracks its bf16 source. These results come from different corpora with different measurements and should not be compared as though they were one common benchmark.

</details>

SAGE-CREST reduces full-vocabulary KL by **13.4% against SAGE-PATHS** and **27.9% against the uniform 6-bit build** on the agentic basis it targets, while staying 0.21 GB smaller than the latter. On the generalist basis it improves on SAGE-PATHS (`0.010114` vs `0.012280`) and draws level with SAGE on head-KL (`0.010153`), but the uniform 6-bit build stays ahead there (`0.008508`, +18.9%): spending the same bytes uniformly protects generalist text better; targeting them protects agentic text better.

If your workload is closer to the stock/generalist distribution, **SAGE is the more appropriate build** in this collection. If you need image or video input, use **SAGE** or **SAGE-ARCS**: SAGE-CREST has no vision tower.

### The lead <u>holds</u> at *long context*

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.641301</td><td>0.925992</td><td>1.011646</td><td>0.826168</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>0.564765</strong></td><td><strong>0.810718</strong></td><td><strong>0.884622</strong></td><td><strong>0.732435</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.621535</td><td>1.062724</td><td>1.204114</td><td>0.982294</td></tr>
    <tr class="delta"><td class="lead">lower than SAGE-PATHS by</td><td><strong>11.93%</strong></td><td><strong>12.45%</strong></td><td><strong>12.56%</strong></td><td><strong>11.35%</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring</td><td><strong>16/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td></tr>
  </tbody>
</table>

Head-KL; <u>lower is better</u>. SAGE-CREST's margin over SAGE-PATHS is close to constant across lengths (11.9% at 2,048 tokens, 11.3% at 131,072): a level shift, not one that grows with context. The uniform 6-bit build is ahead of SAGE-PATHS over the first 2,048 tokens (−3.1%) and behind it from 8,192 on (+14.8%, +18.9% at 131,072), so its short-context lead does not carry to long sessions.

Top-1 and top-5 over the full 128K show the same ordering:

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.8619</td><td>0.9391</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>0.8700</strong></td><td><strong>0.9456</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.8510</td><td>0.9282</td></tr>
  </tbody>
</table>

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.3904</td><td>1.7894</td><td>1.5707</td><td>0.8492</td><td>0.7449</td><td>0.7724</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td>0.3444</td><td><strong>1.5560</strong></td><td><strong>1.3580</strong></td><td>0.8568</td><td><strong>0.7611</strong></td><td><strong>0.7874</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td><strong>0.3250</strong></td><td>2.0137</td><td>1.8244</td><td><strong>0.8744</strong></td><td>0.7410</td><td>0.7607</td></tr>
    <tr class="delta"><td class="lead">SAGE-CREST vs SAGE-PATHS</td><td><strong>−11.79%</strong></td><td><strong>−13.04%</strong></td><td><strong>−13.54%</strong></td><td><strong>+0.76 pp</strong></td><td><strong>+1.61 pp</strong></td><td><strong>+1.50 pp</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring SAGE-CREST</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td><strong>15/15</strong></td></tr>
    <tr class="delta"><td class="lead">SAGE-CREST vs uniform 6-bit</td><td>+5.95% (<code>ns</code>)</td><td><strong>−22.73%</strong></td><td><strong>−25.56%</strong></td><td>−1.76 pp</td><td><strong>+2.00 pp</strong></td><td><strong>+2.66 pp</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring SAGE-CREST</td><td>5/15</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td><td>5/15</td><td><strong>15/15</strong></td><td><strong>15/15</strong></td></tr>
  </tbody>
</table>

SAGE-CREST is closer to the bf16 source than SAGE-PATHS on **every one of the 45 sequences**, and the lineage keeps its order — SAGE-CREST, then SAGE-PATHS, SAGE-ARCS, SAGE — on 15 of 15 sequences at every length. Against the uniform 6-bit build it is a quarter lower at 8K and 16K on every sequence.

At 2K the uniform 6-bit build is level on KL (t +0.43) and ahead on top-1. That lead is **not short context**: ten of the fifteen 2K sequences open with the same 1,746-token agent system prompt, and they are exactly the ten where the uniform build leads. On the other five, and on random 2048-token windows of off-pool text (next section), the targeted builds lead.

### SWE replication

Two further corpora of software-engineering agent text from outside the calibration pool, frozen before scoring (built twice, byte-identical):

* **S** — other model families (DeepSeek-V4-Flash, MiniMax-M2.5, Qwen3.5-122B) on the same task pools as the calibration text;
* **X** — other models (Claude 4.5 Sonnet, GPT-5-mini) on other repositories and languages (Go and C; Python).

Each stratum is scored as 48 evaluation batches of random 2048-token windows and 24 document heads of 8192 tokens, every set against its own bf16 reference:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:24%;">comparison</th>
      <th>set</th>
      <th>KL change</th>
      <th>t</th>
      <th>lower</th>
      <th>top-1</th>
      <th>t</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="section" colspan="7">SAGE-CREST vs SAGE-PATHS</td></tr>
    <tr><td class="lead">S, 2048 windows</td><td>48 batches</td><td><strong>−15.88%</strong></td><td>−7.59</td><td>45/48</td><td>+0.712 pp</td><td>+9.98</td></tr>
    <tr><td class="lead">X, 2048 windows</td><td>48 batches</td><td><strong>−15.93%</strong></td><td>−10.88</td><td>48/48</td><td>+0.692 pp</td><td>+9.32</td></tr>
    <tr><td class="lead">S, 8192 heads</td><td>24 sequences</td><td><strong>−12.85%</strong></td><td>−11.01</td><td>24/24</td><td>+1.535 pp</td><td>+12.38</td></tr>
    <tr><td class="lead">X, 8192 heads</td><td>24 sequences</td><td><strong>−14.25%</strong></td><td>−17.15</td><td>24/24</td><td>+1.292 pp</td><td>+18.35</td></tr>
    <tr><td class="section" colspan="7">SAGE-CREST vs uniform 6-bit</td></tr>
    <tr><td class="lead">S, 2048 windows</td><td>48 batches</td><td><strong>−28.87%</strong></td><td>−7.73</td><td>45/48</td><td>+1.063 pp</td><td>+9.02</td></tr>
    <tr><td class="lead">X, 2048 windows</td><td>48 batches</td><td><strong>−30.99%</strong></td><td>−14.32</td><td>48/48</td><td>+1.572 pp</td><td>+13.43</td></tr>
    <tr><td class="lead">S, 8192 heads</td><td>24 sequences</td><td><strong>−23.02%</strong></td><td>−7.81</td><td>24/24</td><td>+1.602 pp</td><td>+4.95</td></tr>
    <tr><td class="lead">X, 8192 heads</td><td>24 sequences</td><td><strong>−27.17%</strong></td><td>−17.27</td><td>24/24</td><td>+2.514 pp</td><td>+15.93</td></tr>
  </tbody>
</table>

All eight pre-registered tests replicate on text the builds never saw (multiplicity-corrected over the evaluation's 14 tests: |t| > 3.07 for windows, > 3.25 for heads). On the X stratum each lands at or beyond the effect measured before it; on S they replicate at a similar size (on 8,192-token heads, −12.85% against −13.0% out of sample).

Off-pool text is harder for every build, and the calibration-free uniform build's KL rises about as much as the calibrated builds' (more, on X), so the agentic edge is not calibration overfit:

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.27944 / 0.9235</td><td>0.31983 (×1.145) / 0.9179</td><td>0.43987 (×1.574) / 0.8938</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>0.24330 / 0.9280</strong></td><td><strong>0.26903 (×1.106) / 0.9250</strong></td><td><strong>0.36978 (×1.520) / 0.9007</strong></td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.33313 / 0.9188</td><td>0.37824 (×1.135) / 0.9144</td><td>0.53580 (×1.608) / 0.8850</td></tr>
  </tbody>
</table>

### The stages, on the corpus this build targets

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">rung</th>
      <th>bpw</th>
      <th>head-KL ↓</th>
      <th>top-1 ↑</th>
      <th>generalist head-KL ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE-PATHS (base)</td><td>6.1366</td><td>0.279441</td><td>0.9235</td><td>0.012280</td></tr>
    <tr><td class="lead">elevation, step 1</td><td>6.2747</td><td>0.261671</td><td>0.9251</td><td>0.011393</td></tr>
    <tr class="ship"><td class="lead">elevation, step 2 = SAGE-CREST</td><td><strong>6.5879</strong></td><td><strong>0.243302</strong></td><td><strong>0.9280</strong></td><td><strong>0.010114</strong></td></tr>
  </tbody>
</table>

> The elevation order was chosen on half of this evaluation's batches (the 24 even ones) and reported on the other half, so the generalist corpus, the out-of-sample sets and the SWE replication above are the checks that played no part in selection.

SAGE-CREST's precision elevations were chosen through direct measurements of bf16 fidelity, not from tensor sensitivity scores alone. Candidate tensor classes and layer groups were tested at higher precision, and their gains were weighed against the additional storage required. The resulting two-stage allocation prioritizes **measured agentic head-KL improvement per added byte**.

Further elevation stages keep improving fidelity cumulatively, including on held-out agentic workloads, but grow beyond the uniform 6-bit storage budget. So SAGE-CREST is the highest-performing tested elevation stage that prioritizes behavioral fidelity per constrained checkpoint footprint over absolute fidelity at any size.

### Where the divergence actually is — the same numbers, stratified

If the 48 evaluation batches are sorted by **how uncertain the bf16 source itself is** and split into quartiles:

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:15%;">content difficulty</th>
      <th>bf16 perplexity</th>
      <th>SAGE-PATHS head-KL</th>
      <th>uniform 6-bit head-KL</th>
      <th class="ship"><strong>SAGE-CREST head-KL</strong></th>
      <th class="ship"><strong>SAGE-CREST top-1</strong></th>
      <th>bf16 mass in its own top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead"><strong>easiest quarter</strong></td><td>2.8</td><td>0.0178</td><td>0.0190</td><td class="ship"><strong>0.0142</strong></td><td class="ship">0.9720</td><td>0.9980</td></tr>
    <tr><td class="lead">second</td><td>6.3</td><td>0.1246</td><td>0.1528</td><td class="ship"><strong>0.1166</strong></td><td class="ship">0.9486</td><td>0.9979</td></tr>
    <tr><td class="lead">third</td><td>19.5</td><td>0.3103</td><td>0.3764</td><td class="ship"><strong>0.2722</strong></td><td class="ship">0.9208</td><td>0.9973</td></tr>
    <tr><td class="lead"><strong>hardest quarter</strong></td><td>136.9</td><td>0.6651</td><td>0.7842</td><td class="ship"><strong>0.5702</strong></td><td class="ship">0.8705</td><td>0.9965</td></tr>
    <tr class="total"><td class="lead"><strong>pooled</strong></td><td>14.8</td><td>0.2794</td><td>0.3331</td><td class="ship"><strong>0.2433</strong></td><td class="ship"><strong>0.9280</strong></td><td>0.9974</td></tr>
  </tbody>
</table>

Across a **49× range in source-model perplexity**, SAGE-CREST's head-KL changes by roughly **40×**, from 0.0142 to 0.5702. Over the same range its top-1 agreement moves from 0.9720 to 0.8705, while the bf16 reference's own mass in its top-1024 stays between 0.9965 and 0.9980.

<details>
<summary><strong>Why the pooled KL is not a property of the model</strong></summary>

The pooled `0.2433` should therefore not be treated as an intrinsic property of the quantized model. It is a result measured on a particular corpus whose quartiles range from perplexity 2.8 to 136.9.

The same build reports 0.0142 on the easiest quartile and 0.5702 on the hardest. KL values are only directly comparable **within the same evaluation corpus**. Top-1, top-5, and the candidate's mass on the bf16 top-1024 provide more intuitive context when reasoning about likely model behavior.

</details>

Against SAGE-PATHS, SAGE-CREST is better in **45 of 48 evaluation batches**, and against the uniform 6-bit build in **46 of 48**:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:20%;">quartile</th>
      <th>batches favouring SAGE-CREST over SAGE-PATHS</th>
      <th class="delta">head-KL change vs SAGE-PATHS</th>
      <th>batches favouring SAGE-CREST over uniform 6-bit</th>
      <th class="delta">head-KL change vs uniform 6-bit</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">easiest</td><td>12 / 12</td><td class="delta">−20.3%</td><td>10 / 12</td><td class="delta">−25.3%</td></tr>
    <tr><td class="lead">second</td><td>9 / 12</td><td class="delta">−6.4%</td><td>12 / 12</td><td class="delta">−23.7%</td></tr>
    <tr><td class="lead">third</td><td>12 / 12</td><td class="delta">−12.3%</td><td>12 / 12</td><td class="delta">−27.7%</td></tr>
    <tr><td class="lead">hardest</td><td>12 / 12</td><td class="delta">−14.3%</td><td>12 / 12</td><td class="delta">−27.3%</td></tr>
  </tbody>
</table>

Against SAGE-PATHS the gain is largest in relative terms on the easiest content (−20.3%) and largest in absolute terms on the hardest (−0.0949 nats). Against the uniform 6-bit build it is a near-constant quarter in every quartile (−23.7% to −27.7%).

### Every metric, one basis

On the agentic basis — the same 48 evaluation batches / 196,512 scored next-token positions from 96 × 2048-token input sequences, and the same bf16 reference cache:

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
    <tr><td class="lead">SAGE-PATHS</td><td>6.1366</td><td>0.310274</td><td>0.279441</td><td>0.923460</td><td>0.977131</td><td>−0.006307</td><td>0.99371</td><td>0.990208</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td>6.5879</td><td><strong>0.268643</strong></td><td><strong>0.243302</strong></td><td><strong>0.927984</strong></td><td><strong>0.979726</strong></td><td>−0.006643</td><td>0.99338</td><td><strong>0.991499</strong></td></tr>
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
    <tr><td class="lead">SAGE-PATHS</td><td>6.1366</td><td>0.012280</td><td>0.944503</td><td>0.999404</td><td>+0.003280</td><td>1.00329</td><td>0.980424</td></tr>
    <tr class="ship"><td class="lead">SAGE-CREST</td><td>6.5879</td><td>0.010114</td><td>0.948520</td><td><strong>0.999519</strong></td><td><strong>+0.001498</strong></td><td><strong>1.00150</strong></td><td>0.980465</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>6.6486</td><td><strong>0.008508</strong></td><td><strong>0.956401</strong></td><td>0.999473</td><td>−0.002324</td><td>0.99768</td><td><strong>0.980694</strong></td></tr>
  </tbody>
</table>

**ΔCE** is the change in cross-entropy against the bf16 source, in nats per token. **ppl ratio** expresses the same quantity as a perplexity multiplier.

For a fidelity claim, the target is **ΔCE = 0**, not simply the minimum numerical value. That distinction matters on the agentic basis because all five values are negative.

> All five quantizations have slightly lower cross-entropy than the bf16 source on the agentic corpus, with perplexity ratios from 0.980 to 0.997.

A quantized model can assign the observed tokens slightly better than its source while still sitting further from the source's overall distribution. Quantization noise is not zero-mean with respect to a particular corpus.

SAGE-CREST's pooled ΔCE is closer to zero than the uniform 6-bit build's and slightly further than SAGE-PATHS's. Per batch, it is closer to `ΔCE = 0` than SAGE-PATHS in only **18/48 evaluation batches (`p = 0.11`)**, so ΔCE does not separate the two. **KL, not ΔCE, is the fidelity metric here.**

### The paired comparison

SAGE-CREST relative to SAGE-PATHS and to the uniform 6-bit build, paired on identical evaluation units. `favor` counts paired units where SAGE-CREST is closer to the bf16 source under the stated metric; ties are excluded from the denominator.

<table class="sage-table dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:20%;">metric</th>
      <th>Δ vs SAGE-PATHS</th>
      <th>t</th>
      <th class="delta">favor</th>
      <th>Δ vs uniform 6-bit</th>
      <th>t</th>
      <th class="delta">favor</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="section" colspan="7">Agentic basis (48 batches)</td></tr>
    <tr><td class="lead">full-vocabulary KL</td><td><strong>−0.041631</strong></td><td>−5.53</td><td class="delta"><strong>46/48</strong></td><td><strong>−0.104020</strong></td><td>−5.63</td><td class="delta"><strong>46/48</strong></td></tr>
    <tr><td class="lead">head-KL</td><td><strong>−0.036139</strong></td><td>−5.35</td><td class="delta"><strong>45/48</strong></td><td><strong>−0.089825</strong></td><td>−5.65</td><td class="delta"><strong>46/48</strong></td></tr>
    <tr><td class="lead">top-1 agreement</td><td><strong>+0.004524</strong></td><td>+6.93</td><td class="delta"><strong>40/46</strong></td><td><strong>+0.009226</strong></td><td>+7.19</td><td class="delta"><strong>40/47</strong></td></tr>
    <tr><td class="lead">top-5 agreement</td><td><strong>+0.002595</strong></td><td>+5.02</td><td class="delta"><strong>34/39</strong></td><td><strong>+0.006763</strong></td><td>+5.97</td><td class="delta"><strong>41/42</strong></td></tr>
    <tr><td class="lead">mass on bf16 top-1024</td><td>+0.001291</td><td>+5.33</td><td class="delta">39/48</td><td>+0.003440</td><td>+5.73</td><td class="delta">44/48</td></tr>
    <tr><td class="lead">tail term</td><td>−0.000240</td><td>−1.98</td><td class="delta">30/48 — <code>ns</code></td><td>−0.000567</td><td>−2.86</td><td class="delta">31/48 — <code>ns</code></td></tr>
    <tr><td class="lead">|ΔCE| toward 0</td><td>—</td><td>—</td><td class="delta">18/48 — <code>ns</code></td><td>—</td><td>—</td><td class="delta">28/48 — <code>ns</code></td></tr>
    <tr><td class="section" colspan="7">Stock basis (32 batches)</td></tr>
    <tr><td class="lead">head-KL</td><td>−0.002166</td><td>−5.85</td><td class="delta">31/32</td><td>+0.001607</td><td>+3.46</td><td class="delta">6/32</td></tr>
    <tr><td class="lead">top-1 agreement</td><td>+0.004017</td><td>+9.51</td><td class="delta">31/32</td><td>−0.007881</td><td>−14.29</td><td class="delta">0/32</td></tr>
    <tr><td class="lead">top-5 agreement</td><td>+0.000115</td><td>+2.70</td><td class="delta">15/20</td><td>+0.000046</td><td>+0.69</td><td class="delta">12/20 — <strong><code>ns</code>, saturated</strong></td></tr>
    <tr><td class="section" colspan="7">Out of sample (15 sequences per length)</td></tr>
    <tr><td class="lead">head-KL, 2K</td><td>−0.046009</td><td>−2.98</td><td class="delta"><strong>15/15</strong></td><td>+0.019352</td><td>+0.43</td><td class="delta">5/15 — <code>ns</code></td></tr>
    <tr><td class="lead">head-KL, 8K</td><td><strong>−0.233409</strong></td><td>−7.49</td><td class="delta"><strong>15/15</strong></td><td><strong>−0.457691</strong></td><td>−8.09</td><td class="delta"><strong>15/15</strong></td></tr>
    <tr><td class="lead">head-KL, 16K</td><td><strong>−0.212663</strong></td><td>−11.80</td><td class="delta"><strong>15/15</strong></td><td><strong>−0.466370</strong></td><td>−12.37</td><td class="delta"><strong>15/15</strong></td></tr>
    <tr><td class="lead">top-1 agreement, 16K</td><td><strong>+0.014955</strong></td><td>+16.71</td><td class="delta"><strong>15/15</strong></td><td><strong>+0.026641</strong></td><td>+14.08</td><td class="delta"><strong>15/15</strong></td></tr>
    <tr><td class="section" colspan="7">Long context, 128K (16 sequences)</td></tr>
    <tr><td class="lead">head-KL</td><td><strong>−0.093733</strong></td><td>−8.43</td><td class="delta"><strong>16/16</strong></td><td><strong>−0.249859</strong></td><td>−8.01</td><td class="delta"><strong>16/16</strong></td></tr>
    <tr><td class="lead">top-1 agreement</td><td><strong>+0.008020</strong></td><td>+11.04</td><td class="delta"><strong>16/16</strong></td><td><strong>+0.018920</strong></td><td>+10.33</td><td class="delta"><strong>16/16</strong></td></tr>
    <tr><td class="lead">top-5 agreement</td><td><strong>+0.006562</strong></td><td>+8.32</td><td class="delta"><strong>16/16</strong></td><td><strong>+0.017443</strong></td><td>+8.14</td><td class="delta"><strong>16/16</strong></td></tr>
  </tbody>
</table>

`ns` marks a `favor` count whose two-sided exact sign test gives p ≥ 0.05; this table is descriptive and should not be read as a multiplicity-corrected confirmatory family unless a correction is stated separately.

Two details are worth keeping in mind when reading this table:

* **Against SAGE-PATHS, the gain is inside the retained support and in the mass moved out of it.** Both terms favor SAGE-CREST (45/48 and 41/48); the tail term does not separate the two (30/48).

* **Against SAGE-PATHS the stock basis moves with the agentic one (31/32); against the uniform 6-bit build it does not.** On generalist text the uniform build keeps its lead on head-KL (26/32) and on top-1 in every batch.

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
    <tr><td class="lead">SAGE-PATHS</td><td>+0.020393 ± 0.005505</td><td>−0.000149 ± 0.000234</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td><strong>+0.017468</strong> ± 0.004757</td><td>−0.000119 ± 0.000214</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>+0.026241 ± 0.006694</td><td>−0.000180 ± 0.000279</td></tr>
  </tbody>
</table>

On the agentic basis SAGE-CREST has the lowest fitted slope of the five. The standard errors overlap, however, so this should be treated as a directional observation rather than an established difference. At 128K every build's slope is slightly negative and within one standard error of zero.

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
    <tr><td class="lead"><strong>top-1 agreement</strong></td><td><strong>0.9280</strong></td><td>same next token as bf16 on <strong>92.8%</strong> of positions — what greedy decoding reads</td></tr>
    <tr><td class="lead"><strong>top-5 agreement</strong></td><td><strong>0.9797</strong></td><td>bf16's choice is in the candidate's top 5 on <strong>98.0%</strong> of positions</td></tr>
    <tr><td class="lead"><strong>candidate mass on bf16 top-1024</strong></td><td><strong>0.9915</strong></td><td>SAGE-CREST places <strong>99.1%</strong> of its probability on the tokens bf16 ranks in its top 1024</td></tr>
    <tr><td class="lead"><strong>perplexity ratio</strong></td><td><strong>0.9934</strong></td><td>predictive cross-entropy on the observed text differs from bf16 by approximately <strong>0.7% in perplexity ratio</strong></td></tr>
  </tbody>
</table>

The absolute level of KL is also strongly related to how uncertain the bf16 source itself is. On this basis, per-evaluation-unit head-KL correlates with the reference model's own cross-entropy at **r = +0.92**.

In this evaluation, agent transcripts — including tool output, diffs, error text, and mid-task reasoning — include content for which the source model itself is substantially less certain than on the easiest portions of the corpus.

A head-KL of 0.24 on this corpus and 0.010 on the generalist corpus therefore do not imply a 24× difference in quantization quality. They are measurements on distributions with very different difficulty. Direct comparisons are valid only within the same evaluation basis.

Those 1024 entries represent 0.41% of the 248,320-token vocabulary. On the agentic basis they contain **99.74%** of the bf16 reference probability mass.

The truncated and full-vocabulary metrics rank the builds identically here. **head-KL was used during candidate selection and ranking; full-vocabulary KL is reported as the absolute fidelity audit.** On this corpus, head-KL omits about 9% of SAGE-CREST's total divergence because it excludes the mass-shift contribution described below.

Every row was scored against a **common bf16 reference cache**, with identical settings within each basis:

<table class="sage-table kv dense" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:20%;"></th>
      <th>agentic basis</th>
      <th>stock basis</th>
      <th>out of sample</th>
      <th>SWE replication (per stratum)</th>
      <th>long-context</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">input sequences × length</td><td>96 × 2048</td><td>128 × 1024</td><td>15 × 2048 / 8192 / 16,384</td><td>96 × 2048 windows + 24 × 8192 heads</td><td>16 × 131,072</td></tr>
    <tr><td class="lead">scored next-token positions</td><td>196,512</td><td>130,944</td><td>30,705 / 122,865 / 245,745</td><td>196,512 + 196,584</td><td>2,097,136</td></tr>
    <tr><td class="lead">batch size</td><td>2 (48 batches)</td><td>4 (32 batches)</td><td>1 (15 per length)</td><td>2 (48) / 1 (24)</td><td>1 (16 batches)</td></tr>
    <tr><td class="lead">top-k</td><td>full 248,320 + 1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td></tr>
    <tr><td class="lead">seed</td><td>123</td><td>123</td><td>fixed corpus files</td><td>fixed corpus files</td><td>fixed corpus file</td></tr>
  </tbody>
</table>

> [!Tip] Comparisons are valid **within** each basis, not across them. <u>KL is corpus-relative</u>: a value from one evaluation distribution cannot be compared directly with a value from another.

### Full-vocabulary audit

The full 248,320-entry reference distribution was stored in fp32 and the divergence computed directly, then checked against the truncated metric.

As an instrument control, the head terms are computed two ways — from the stored top-1024 logits and from the full log-probabilities — and must agree; `identity_gap OK` (≤ 4.6e-05 nats/token) was obtained for every record. The three decomposition terms below are per-token means and sum to the directly measured full-vocabulary KL to within 1.7e-08.

On the agentic basis, the tail outside the bf16 top-1024 contributes **0.001238 nats** for SAGE-CREST, about **0.5%** of its total divergence; the bf16 reference places **99.74%** of its own probability inside that set.

The truncated metric (head-KL) omits the mass-shift term, so on this corpus it understates the absolute divergence by **9.4%** (SAGE-CREST), **9.9%** (SAGE-PATHS) and **10.6%** (uniform 6-bit). It produces the same ordering as full-vocabulary KL in every comparison here.

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
    <tr><td class="lead">SAGE-PATHS</td><td>0.310274</td><td>0.277960 (89.6%)</td><td>0.030837 (9.9%)</td><td>0.001477 (0.5%)</td><td>0.310274</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-CREST</strong></td><td>0.268643</td><td>0.242064 (90.1%)</td><td>0.025341 (9.4%)</td><td>0.001238 (0.5%)</td><td>0.268643</td></tr>
    <tr><td class="lead">uniform 6-bit</td><td>0.372663</td><td>0.331308 (88.9%)</td><td>0.039549 (10.6%)</td><td>0.001805 (0.5%)</td><td>0.372663</td></tr>
    <tr class="delta"><td class="lead">paired, SAGE-CREST lower than SAGE-PATHS</td><td>46/48</td><td>45/48, p = 1.3e-10</td><td>41/48, p = 6.2e-07</td><td>30/48, p = 0.11</td><td>&nbsp;</td></tr>
    <tr class="delta"><td class="lead">paired, SAGE-CREST lower than uniform 6-bit</td><td>46/48</td><td>46/48, p = 8.4e-12</td><td>42/48, p = 1.0e-07</td><td>31/48, p = 0.059</td><td>&nbsp;</td></tr>
  </tbody>
</table>

Against both comparators the improvement is concentrated **inside the retained support**, with the mass-shift term leaning the same way; the tail term is not separated at this sample size.

<details>
<summary><strong>How the three terms add up</strong></summary>

Each term is the per-token mean over all **196,512 scored next-token positions**, so the three add up to full-vocabulary KL exactly: 0.242064 + 0.025341 + 0.001238 = 0.268643 for SAGE-CREST. About 90% of its divergence is inside the tokens bf16 considers plausible, about 9% is probability moved out of that set, and under 0.5% is in the remaining vocabulary.

</details>

## Honest limits

* **Text only.** The vision tower is removed; this checkpoint cannot process images or video. Use SAGE or SAGE-ARCS for multimodal input.

* **A uniform 6-bit build is more faithful on the stock/generalist distribution.** Stock-basis head-KL is `0.010114` for SAGE-CREST versus `0.008508` for the uniform 6-bit build (+18.9%, t +3.46; the uniform build is lower in 26 of 32 batches and ahead on top-1 in all 32). SAGE-CREST is level with SAGE there on pooled head-KL (−0.4%, t −0.08), but SAGE keeps its top-1 lead (29 of 30 batches).

* **At the head of agent documents, the uniform 6-bit build can lead.** Out of sample at 2K it is level on KL (+5.95%, `ns`) and ahead on top-1 (−1.76 pp). The lead traces to a shared 1,746-token system prompt at the head of ten of the fifteen sequences, not to short context; on random 2048-token windows SAGE-CREST leads on both off-pool strata.

* **The precision choices were selected on agentic fidelity**, ordered on half of the agentic evaluation's batches. The generalist basis, the out-of-sample sets and the SWE replication were not selection criteria.

* **head-KL is not full-vocabulary KL.** It uses an identical top-1024 truncation for every build and omits the mass-shift term: on the agentic corpus it is 9.4% (SAGE-CREST) below full-vocabulary KL. Both metrics produce the same ordering in the comparisons reported here.

* **ΔCE is not a fidelity metric and is negative on the agentic basis.** All five builds predict that corpus slightly better than the bf16 source does. Per batch, ΔCE does not separate SAGE-CREST from SAGE-PATHS (18/48, `p = 0.11`). It is reported for completeness rather than used as evidence of fidelity.

* **Top-5 agreement is saturated on the stock basis and informative on the others.** At about 0.9995 it barely discriminates on stock (15/20 against SAGE-PATHS, 12/20 against the uniform 6-bit build).

* **Within-sequence position slopes are directional only.** SAGE-CREST has the lowest fitted slope on the agentic basis, but the standard errors overlap.

* **48 evaluation batches, and 15 out-of-sample sequences per length, are modest samples.** The SAGE-CREST-versus-SAGE-PATHS difference is consistent (45/48 batches, 45/45 out-of-sample sequences, every SWE-replication set), but smaller differences among closely related builds are not reliably resolved at these sizes.

* **Fidelity is not task accuracy.** They are separate claims and were measured separately. Across 964 benchmark questions with thinking disabled, deterministic sampling and MTP enabled, on oMLX 0.7.0, SAGE-CREST scores 758 against 768 for SAGE-PATHS in the same session (11 vs 21 discordant questions, `p = 0.11`, paired exact McNemar); the gap is almost entirely livecodebench, within its repeat noise below. On the full Tier A suite under the same settings, 1,064 questions (livecodebench 300, HumanEval 164, MBPP 200, GSM8K 100, MMLU-Pro 200, ARC-Challenge 100), SAGE-CREST scores 799 (75.1%). 864 of those questions are shared with the earlier 964-question run, and on them the two runs of the same weights differ by 13 verdicts (6 vs 7, `p = 1.0`).

* **Benchmark totals are not a ranking.** On oMLX 0.7.0, livecodebench changed 17 verdicts when the same weights were benchmarked again. Single-benchmark differences of that size should be treated as unresolved.

* **MTP draft acceptance is 86.9%, and SAGE-CREST decodes as fast as SAGE-PATHS.** Across the 1,064-question Tier A run, 86.9% of drafted tokens were accepted, and each forward pass committed 3.53 tokens on average. Both builds were benchmarked in one session (964 questions, oMLX 0.7.0, the same machine), and both decoded at 28.3 tokens/s, prefill included. SAGE-CREST's larger weights take slightly longer per forward pass, but its draft commits slightly more tokens per pass (3.63 vs 3.50), and the two cancel. Tokens/s depends on the machine's state, so compare it only within one session. Served through stock oMLX 0.7.0, its greedy output matches an offline greedy decode of the same weights on 4 of 4 prompts (48 tokens each).

* **The bf16 MTP head costs 0.1431 bpw and is retained for speculative decoding.** It does not participate in the scored forward path.

* **Fidelity is measured against the bf16 source.** A quantization that reproduces its source more faithfully also reproduces the source model's mistakes more faithfully.

## Quantization details

* **Model type:** `qwen3_5`, text only
* **Parameters:** 27.32 B (vision tower removed)
* **Bits:** 5-bit base, with 206 per-tensor overrides placed by measured sensitivity and tensor analysis
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
hf download notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp --local-dir ~/.omlx/models/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp
omlx serve --model-dir ~/.omlx/models --port 8000
```

Then open the oMLX admin dashboard (`http://127.0.0.1:8000/admin`), open `Qwen3.8-27B-oQ5e-SAGE-CREST-mtp`'s model settings and turn on **Lightning MTP**.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.8-27B-oQ5e-SAGE-CREST-mtp", "messages": [{"role": "user", "content": "Hello"}]}'
```

### mlx-lm / mlx-vlm

SAGE-CREST also loads outside oMLX, as a text-only model, under a standard `mlx-lm` or `mlx-vlm` stack. Verified 2026-10-09 with mlx-lm 0.32.0 and mlx-vlm 0.7.6, and with the versions oMLX 0.7.0 bundles (mlx-lm 0.31.4.dev, mlx-vlm 0.7.1):

```bash
pip install mlx-lm
python -m mlx_lm.generate --model notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp --max-tokens 256 --prompt "Hello"
```

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model notnaurw/Qwen3.8-27B-oQ5e-SAGE-CREST-mtp --max-tokens 256 --prompt "Hello"
```

mlx-vlm reports that the checkpoint has no vision-tower weights and disables that module, so image input is not available. Neither stack runs the bf16 MTP head: Lightning MTP (speculative decoding) is available only through oMLX.

## Attribution

Base model © the Qwen team, Apache-2.0 — see `LICENSE`.

Quantization performed with [oMLX](https://github.com/jundot/omlx) `0.6.4` (the SAGE-PATHS base); the precision elevations were applied and validated on `0.7.0`, and the build is served on `0.7.0`. SAGE-CREST is an adapter layer over oMLX's `oQe` backend.

> Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork*. August 2026.
> https://qwen.ai/blog?id=qwen3.8
