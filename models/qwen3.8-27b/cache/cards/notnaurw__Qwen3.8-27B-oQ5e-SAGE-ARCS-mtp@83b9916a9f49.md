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


<h1 align="center">Qwen3.8-27B — oQ5e / SAGE-ARCS (MLX, 5-bit, MTP preserved, bf16)</h1>
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

A mixed-precision 5-bit MLX quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), calibrated toward **agentic text workloads** while preserving the original vision architecture, and intended to be served with **Lightning MTP enabled** in oMLX.

**SAGE-ARCS** (**A**ctivation-guided **R**eranked **C**alibration **S**cale) extends the fidelity-first SAGE quantization pipeline around one question:

> **If the model, architecture, and precision budget remain unchanged, how much does the distribution used to measure sensitivity affect where fidelity to the bf16 source is preserved?**

SAGE established the underlying approach: measure where quantization most affects the bf16 source model, then allocate a constrained precision budget accordingly. Among the experiments developed from that approach, SAGE-ARCS isolates a different variable: the distribution used to measure quantization sensitivity. It keeps the underlying objective and available budget unchanged, but shifts the calibration distribution toward agentic workloads.

This distinction matters because quantization sensitivity is not independent of the data used to measure it. Different workloads expose different activation patterns and sources of error. A calibration corpus that emphasizes generalist text does not necessarily identify the same precision priorities as one built around agentic interactions, tool use, and extended execution trajectories.

SAGE-ARCS therefore changes **where the available precision is directed**, not how much precision the checkpoint is allowed to consume. The same underlying pipeline is guided by a different measurement distribution, allowing the allocation to favor tensors whose precision contributes more strongly to preserving bf16 behavior on the targeted workload.

**Vision remains preserved.** The result is a checkpoint that tracks the bf16 source more closely on the targeted <u>agentic distribution</u> than SAGE, while giving up some fidelity on the generalist distribution SAGE was designed around. The difference is a deliberate change in emphasis, not a universal improvement across workloads.

The capabilities of the bf16 source remain the reference. SAGE-ARCS does not attempt to introduce new capabilities, but to test whether the same precision budget can preserve existing behavior more effectively when the intended workload is made part of the quantization design.

**SAGE-ARCS therefore complements SAGE rather than replacing it:** SAGE establishes how much generalist fidelity can be recovered through sensitivity-aware mixed precision; SAGE-ARCS demonstrates that changing the calibration distribution can redirect that fidelity toward agentic behavior without increasing the storage budget. Both preserve the vision architecture, but each makes a different choice about which text distribution should guide precision allocation.

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
    <tr><td class="lead"><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-mtp">SAGE</a></td><td>6.0836</td><td>21.13 GB</td><td>Generalist · Multimodal</td><td>Preserved</td><td>Preserved</td></tr>
    <tr class="ship"><td class="lead"><strong><a href="https://huggingface.co/notnaurw/Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp">SAGE-ARCS</a></strong></td><td><strong>6.0836</strong></td><td><strong>21.13 GB</strong></td><td><strong>Agentic · Multimodal</strong></td><td><strong>Preserved</strong></td><td><strong>Preserved</strong></td></tr>
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

The main difference between SAGE and SAGE-ARCS is the **distribution used to measure sensitivity**. To make the tradeoff visible, both builds are evaluated on both distributions. Measuring each build only on the data it was optimized around would not tell us much about what was actually gained or lost.

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
    <tr><td class="lead">SAGE</td><td>0.434296</td><td>0.385833</td><td>0.909990</td><td>0.968938</td><td>0.986267</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td><strong>0.352813</strong></td><td><strong>0.316554</strong></td><td><strong>0.919730</strong></td><td><strong>0.974363</strong></td><td><strong>0.988958</strong></td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Agentic</u></em></strong> basis</td><td><strong>18.76% lower == BETTER</strong></td><td><strong>17.96% lower == BETTER</strong></td><td><strong>+0.97 pp</strong></td><td><strong>+0.54 pp</strong></td><td><strong>+0.27 pp</strong></td></tr>
  </tbody>
</table>

Paired on the same evaluation batches, **SAGE-ARCS is better on 47 of 48** for full-vocabulary KL and **44 of 48** for top-1 agreement.

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
    <tr><td class="lead">SAGE</td><td><strong>0.010348</strong></td><td><strong>0.952308</strong></td><td>0.999420</td></tr>
    <tr class="ship"><td class="lead">SAGE-ARCS</td><td>0.014518</td><td>0.942036</td><td>0.999359</td></tr>
    <tr class="delta"><td class="lead"><strong><em><u>Generalist</u></em></strong> basis</td><td>SAGE better, 30/32</td><td>SAGE better, 32/32</td><td>saturated, <code>ns</code></td></tr>
  </tbody>
</table>

<details>
<summary><strong>Note — calibration changes where fidelity is kept, not how much</strong></summary>

Changing the distribution used for calibration and sensitivity measurement does not make a quantization uniformly better. It changes where, under a constrained precision budget, the quantized model most faithfully tracks its bf16 source. These results therefore come from different corpora with different measurements and should not be compared as though they were one common benchmark.

</details>

SAGE-ARCS reduces full-vocabulary KL by **18.8% on the agentic basis it targets**, while its stock-basis head-KL is higher than SAGE's (`0.014518` vs `0.010348`).

If your workload is closer to the stock/generalist distribution, **SAGE is the more appropriate build**. That is the distribution it was designed around.

### *Long context* is where the difference persists most <u>clearly</u>

Held-out long-context sequences, 16 per rung, with one bf16 reference per length:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>2,048</th>
      <th>8,192</th>
      <th>16,384</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.928733</td><td>1.559797</td><td>1.604263</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td><strong>0.760689</strong></td><td><strong>1.253266</strong></td><td><strong>1.286875</strong></td></tr>
    <tr class="delta"><td class="lead">lower by</td><td><strong>18.09%</strong></td><td><strong>19.65%</strong></td><td><strong>19.78%</strong></td></tr>
    <tr class="delta"><td class="lead">sequences favoring</td><td><strong>16/16</strong></td><td><strong>16/16</strong></td><td><strong>16/16</strong></td></tr>
  </tbody>
</table>

Head-KL; <u>lower is better</u>.

Top-1 and top-5 show the same pattern, and top-5 is no longer saturated at these lengths:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>top-1 @2K / 8K / 16K ↑</th>
      <th>top-5 @2K / 8K / 16K ↑</th>
      <th>candidate mass on bf16 top-1024 @2K / 8K / 16K</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>0.7627 / 0.7478 / 0.7613</td><td>0.9327 / 0.8864 / 0.8850</td><td>0.98021 / 0.95601 / 0.95439</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td><strong>0.7875 / 0.7738 / 0.7876</strong></td><td><strong>0.9464 / 0.9082 / 0.9067</strong></td><td><strong>0.98381 / 0.96721 / 0.96608</strong></td></tr>
  </tbody>
</table>

SAGE-ARCS is ahead on every reported metric at every tested length. The mass column is useful independently of KL: relative to SAGE, SAGE-ARCS places more of its probability inside the bf16 model's top-1024 support, so less of its prediction moves outside the set the source considers plausible as context grows.

Every sequence at every tested length favors SAGE-ARCS. The gap also does not disappear as context increases; it is slightly larger at 8K and 16K, which is relevant for longer-running agentic sessions.

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
    <tr><td class="lead">constrained budget allocation</td><td>5.9429</td><td>0.364730</td><td>0.9129</td></tr>
    <tr><td class="lead">layer-wise error compensation</td><td>5.9429</td><td><strong>0.316554</strong></td><td><strong>0.9197</strong></td></tr>
    <tr class="ship"><td class="lead">bf16 MTP head <strong>grafted</strong></td><td><strong>6.0836</strong></td><td><strong>0.316554</strong></td><td><strong>0.9197</strong></td></tr>
  </tbody>
</table>

The error-compensation stage lowers head-KL by **13.21%** on the target corpus.

On the generalist corpus, however, the same stage moves the metric in the opposite direction: `0.011842 → 0.014518`.

**The weights are the same. What changes is the distribution used to score them.**

That contrast is the main point of this experiment.

### Where the divergence actually is — the same numbers, stratified

A single pooled KL hides a fairly wide range of difficulty inside this corpus. If the 48 evaluation batches are sorted by **how uncertain the bf16 source itself is** and split into quartiles:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:26%;">content difficulty</th>
      <th>bf16 perplexity</th>
      <th>SAGE head-KL</th>
      <th class="ship"><strong>SAGE-ARCS head-KL</strong></th>
      <th class="ship"><strong>SAGE-ARCS top-1</strong></th>
      <th>bf16 mass in its own top-1024</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead"><strong>easiest quarter</strong></td><td><strong>2.8</strong></td><td>0.0258</td><td class="ship"><strong>0.0175</strong></td><td class="ship"><strong>0.9702</strong></td><td>0.9980</td></tr>
    <tr><td class="lead">second</td><td>6.3</td><td>0.1777</td><td class="ship"><strong>0.1454</strong></td><td class="ship">0.9423</td><td>0.9980</td></tr>
    <tr><td class="lead">third</td><td>19.5</td><td>0.4325</td><td class="ship"><strong>0.3437</strong></td><td class="ship">0.9113</td><td>0.9972</td></tr>
    <tr><td class="lead"><strong>hardest quarter</strong></td><td><strong>136.8</strong></td><td>0.9073</td><td class="ship"><strong>0.7596</strong></td><td class="ship"><strong>0.8552</strong></td><td>0.9967</td></tr>
    <tr class="total"><td class="lead"><strong>pooled</strong></td><td>14.8</td><td>0.3858</td><td class="ship"><strong>0.3166</strong></td><td class="ship"><strong>0.9197</strong></td><td>0.9975</td></tr>
  </tbody>
</table>

This table is useful context for interpreting the headline KL number. Across a **49× range in source-model perplexity**, head-KL changes by roughly **43×**, from 0.0175 to 0.7596. Over the same range, top-1 agreement moves from 0.970 to 0.855, while the bf16 reference's own mass in its top-1024 stays between 0.9967 and 0.9980.

<details>
<summary><strong>Why the pooled KL is not a property of the model</strong></summary>

The pooled `0.3166` should therefore not be treated as an intrinsic property of the quantized model. It is a result measured on a particular corpus whose quartiles range from perplexity 2.8 to 136.8.

The same build reports 0.0175 on the easiest quartile and 0.7596 on the hardest. KL values are only directly comparable **within the same evaluation corpus**. Top-1, top-5, and the candidate's mass on the bf16 top-1024 provide more intuitive context when reasoning about likely model behavior.

</details>

The improvement also is not concentrated in only one part of the difficulty range. SAGE-ARCS beats SAGE in **47 of 48 evaluation batches**, with improvements present in every quartile:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">quartile</th>
      <th>evaluation batches favouring SAGE-ARCS</th>
      <th class="delta">relative reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">easiest</td><td><strong>12 / 12</strong></td><td class="delta"><strong>−32.2%</strong></td></tr>
    <tr><td class="lead">second</td><td>11 / 12</td><td class="delta">−18.2%</td></tr>
    <tr><td class="lead">third</td><td><strong>12 / 12</strong></td><td class="delta">−20.5%</td></tr>
    <tr><td class="lead">hardest</td><td><strong>12 / 12</strong></td><td class="delta">−16.3%</td></tr>
  </tbody>
</table>

The largest **absolute** gain appears on the hardest content (−0.1477 nats), while the largest **relative** gain is on the easiest (−32.2%).

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
    <tr><td class="lead">SAGE</td><td>6.0836</td><td>0.434296</td><td>0.385833</td><td>0.909990</td><td>0.968938</td><td>−0.006905</td><td>0.99312</td><td>0.986267</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td>6.0836</td><td><strong>0.352813</strong></td><td><strong>0.316554</strong></td><td><strong>0.919730</strong></td><td><strong>0.974363</strong></td><td><strong>−0.002704</strong></td><td><strong>0.99730</strong></td><td><strong>0.988958</strong></td></tr>
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
    <tr><td class="lead"><strong>SAGE</strong></td><td>6.0836</td><td><strong>0.010348</strong></td><td><strong>0.952308</strong></td><td><strong>0.999420</strong></td><td><strong>+0.002032</strong></td><td><strong>1.00203</strong></td><td><strong>0.980567</strong></td></tr>
    <tr class="ship"><td class="lead">SAGE-ARCS</td><td>6.0836</td><td>0.014518</td><td>0.942036</td><td>0.999359</td><td>+0.004601</td><td>1.00461</td><td>0.980380</td></tr>
  </tbody>
</table>

**ΔCE** is the change in cross-entropy against the bf16 source, in nats per token. **ppl ratio** expresses the same quantity as a perplexity multiplier.

For a fidelity claim, the target is **ΔCE = 0**, not simply the minimum numerical value. That distinction matters on the agentic basis because both values are negative.

> Both quantizations have slightly lower cross-entropy than the bf16 source on the agentic corpus, with perplexity ratios of 0.993 and 0.997.

A quantized model can fit the observed tokens slightly better than its source while still sitting further from the source's overall distribution. Quantization noise is not zero-mean with respect to a particular corpus.

If ΔCE were read naively as “lower is better,” SAGE would appear better on this metric. For fidelity, however, the relevant quantity is distance from zero. SAGE-ARCS is closer to `ΔCE = 0` in **33 of 48 evaluation batches (`p = 0.013`)**, and by 2.6× in the pooled mean.

### The paired comparison

SAGE-ARCS relative to SAGE, paired on identical evaluation batches. `favor` counts paired evaluation units where SAGE-ARCS is closer to the bf16 source under the stated metric; ties are excluded from the denominator.

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
    <tr><td class="lead">full-vocabulary KL</td><td><strong>−0.081484</strong></td><td>−6.74</td><td class="delta"><strong>47/48</strong></td><td>3.5e-13</td></tr>
    <tr><td class="lead">head-KL</td><td><strong>−0.069279</strong></td><td>−6.92</td><td class="delta"><strong>47/48</strong></td><td>3.5e-13</td></tr>
    <tr><td class="lead">top-1 agreement</td><td><strong>+0.009740</strong></td><td>+9.16</td><td class="delta"><strong>44/48</strong></td><td>1.5e-09</td></tr>
    <tr><td class="lead">top-5 agreement</td><td><strong>+0.005425</strong></td><td>+6.34</td><td class="delta"><strong>35/41</strong></td><td>4.9e-06</td></tr>
    <tr><td class="lead">mass on bf16 top-1024</td><td><strong>+0.002691</strong></td><td>+5.77</td><td class="delta"><strong>41/48</strong></td><td>6.2e-07</td></tr>
    <tr><td class="lead">tail term</td><td>−0.000438</td><td>−1.75</td><td class="delta">33/48</td><td>0.013</td></tr>
    <tr><td class="lead">|ΔCE| toward 0</td><td>—</td><td>—</td><td class="delta">33/48</td><td>0.013</td></tr>
    <tr><td class="section" colspan="5">Stock basis</td></tr>
    <tr><td class="lead">head-KL</td><td>+0.004170</td><td>+6.99</td><td class="delta">2/32</td><td>2.5e-07</td></tr>
    <tr><td class="lead">top-1 agreement</td><td>−0.010272</td><td>−15.96</td><td class="delta">0/32</td><td>4.7e-10</td></tr>
    <tr><td class="lead">top-5 agreement</td><td>−0.000061</td><td>−0.81</td><td class="delta">8/19</td><td>0.65 — <strong><code>ns</code>, saturated</strong></td></tr>
  </tbody>
</table>

The p-values above are reported as produced by the evaluation harness. This expanded table is descriptive and should not be read as a multiplicity-corrected confirmatory family unless a correction is stated separately.

Two details are worth keeping in mind when reading this table:

* **Top-5 is saturated on the stock basis but not on the agentic basis.** At about 0.9994 for both builds, it has essentially no discriminating power on stock and should not be treated as evidence either way. On the agentic basis it sits at 0.969–0.974 and separates the builds cleanly (35/41, `p = 4.9e-06`). At long context it falls to 0.885–0.947, where the difference becomes larger. A metric can be uninformative on one distribution and useful on another.

* **The stock-basis differences are statistically strong in the opposite direction.** Results such as 0/32 and 2/32 are not noise. The tradeoff is measurable in both directions, which is exactly what this build is intended to explore.

### Divergence within a sequence

Head-KL rises with token position inside a sequence, so a pooled figure averages over another axis that is not flat. Fitted slope, in nats per 100 tokens:

<table class="sage-table" style="display:table;width:100%;table-layout:fixed;">
  <thead>
    <tr>
      <th class="lead" style="width:34%;">build</th>
      <th>agentic (2K)</th>
      <th>long-context (16K)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="lead">SAGE</td><td>+0.028292 ± 0.007593</td><td>+0.000942 ± 0.001612</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td><strong>+0.023192</strong> ± 0.006142</td><td><strong>+0.000638</strong> ± 0.001290</td></tr>
  </tbody>
</table>

SAGE-ARCS has the lower fitted slope on both sets, and the shape is closer to a plateau than compounding growth for both builds.

The standard errors overlap, however, so this should be treated as a directional observation rather than an established difference. It is included because the direction is consistent with the long-context results above, not because this measurement is significant on its own.

## How fidelity was measured

**head-KL** is `KL(bf16 ‖ quantized)` over the bf16 model's **top-1024 logits**, with both distributions renormalized over the retained support.

<details>
<summary><strong>Careful note regarding these numbers, because the scales differ substantially</strong></summary>

**KL divergence is not a percentage and has no ceiling of 1.** A head-KL of `0.3166` does not mean that 32% of something was lost or that 68% was retained. KL is an expected log-ratio on **[0, ∞)**.

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
    <tr><td class="lead"><strong>top-1 agreement</strong></td><td><strong>0.9197</strong></td><td>same next token as bf16 on <strong>92.0%</strong> of positions — what greedy decoding reads</td></tr>
    <tr><td class="lead"><strong>top-5 agreement</strong></td><td><strong>0.9744</strong></td><td>bf16's choice is in the candidate's top 5 on <strong>97.4%</strong> of positions</td></tr>
    <tr><td class="lead"><strong>candidate mass on bf16 top-1024</strong></td><td><strong>0.9890</strong></td><td>SAGE-ARCS places <strong>98.9%</strong> of its probability on the tokens bf16 ranks in its top 1024</td></tr>
    <tr><td class="lead"><strong>perplexity ratio</strong></td><td><strong>0.9973</strong></td><td>predictive cross-entropy on the observed text differs from bf16 by approximately <strong>0.3% in perplexity ratio</strong></td></tr>
  </tbody>
</table>

The absolute level of KL is also strongly related to how uncertain the bf16 source itself is. On this basis, per-evaluation-unit head-KL correlates with the reference model's own cross-entropy at **r = +0.92**.

In this evaluation, agent transcripts — including tool output, diffs, error text, and mid-task reasoning — include content for which the source model itself is substantially less certain than on the easiest portions of the corpus.

A head-KL of 0.32 on this corpus and 0.014 on the generalist corpus therefore do not imply a 22× difference in quantization quality. They are measurements on distributions with very different difficulty. Direct comparisons are valid only within the same evaluation basis.

</details>

Those 1024 entries represent 0.41% of the 248,320-token vocabulary. On the agentic basis they contain **99.75%** of the bf16 reference probability mass.

The truncated and full-vocabulary metrics rank the two builds identically here. **head-KL was used during candidate comparison and ranking; full-vocabulary KL is reported as the absolute fidelity audit.** On this corpus, head-KL omits about 10% of the total divergence because it excludes the mass-shift contribution described below.

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
    <tr><td class="lead">input sequences × length</td><td>96 × 2048</td><td>128 × 1024</td><td>16 × 2048 / 8192 / 16384</td></tr>
    <tr><td class="lead">scored next-token positions</td><td>196,512</td><td>130,944</td><td>length-dependent</td></tr>
    <tr><td class="lead">batch size</td><td>2 (48 batches)</td><td>4 (32 batches)</td><td>1 (16 batches)</td></tr>
    <tr><td class="lead">top-k</td><td>full 248,320 + 1024</td><td>1024</td><td>1024</td></tr>
    <tr><td class="lead">seed</td><td>123</td><td>123</td><td>123</td></tr>
  </tbody>
</table>

> Comparisons are valid **within** each basis, not across them. KL is corpus-relative: a value from one evaluation distribution cannot be compared directly with a value from another.

### Full-vocabulary audit

The full 248,320-entry reference distribution was stored in fp32 and the divergence computed directly, then checked against the truncated metric.

As an instrument control, the head terms are computed two ways — from the stored top-1024 logits and from the full log-probabilities — and must agree; `identity_gap OK` (≤ 5.7e-05 nats/token) was obtained for every record. The three decomposition terms below are per-token means and sum to the directly measured full-vocabulary KL to within 1.1e-08.

On the agentic basis, the tail outside the bf16 top-1024 contributes **0.001417 nats**, about **0.4%** of the total divergence; the bf16 reference places **99.75%** of its own probability inside that set.

The truncated metric (head-KL) omits the mass-shift term, so on this corpus it understates the absolute divergence by **10.3%** (SAGE-ARCS) and **11.2%** (SAGE); on the stock corpus by 2.7% and 2.8%. It produces the same ordering as full-vocabulary KL in every comparison here.

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
    <tr><td class="lead">SAGE</td><td>0.434296</td><td>0.384099 (88.4%)</td><td>0.048342 (11.1%)</td><td>0.001855 (0.4%)</td><td>0.434296</td></tr>
    <tr class="ship"><td class="lead"><strong>SAGE-ARCS</strong></td><td><strong>0.352813</strong></td><td><strong>0.315131 (89.3%)</strong></td><td><strong>0.036264 (10.3%)</strong></td><td><strong>0.001417 (0.4%)</strong></td><td><strong>0.352813</strong></td></tr>
    <tr class="delta"><td class="lead">paired, SAGE-ARCS lower</td><td>47/48</td><td>47/48, p = 3.5e-13</td><td>40/48, p = 3.3e-06</td><td>33/48, p = 0.013</td><td>&nbsp;</td></tr>
  </tbody>
</table>

SAGE-ARCS is lower on all three terms. The improvement is not an artifact of where the top-1024 truncation falls: it appears inside the retained support, in the amount of mass assigned to that support, and in the tail beyond it.

<details>
<summary><strong>How the three terms add up</strong></summary>

Each term is the per-token mean over all **196,512 scored next-token positions**, so the three add up to full-vocabulary KL exactly: 0.315131 + 0.036264 + 0.001417 = 0.352813 for SAGE-ARCS, and 0.384099 + 0.048342 + 0.001855 = 0.434296 for SAGE. About 89% of each build's divergence is inside the tokens bf16 considers plausible, about 10% is probability moved out of that set, and under 0.5% is in the remaining vocabulary.

</details>

## Honest limits

* **Vision was preserved but not evaluated.** Qwen3.8-27B is a vision-language model; all fidelity measurements reported here are text-only. Image and video behavior were not measured.

* **SAGE-ARCS is worse than SAGE on the stock/generalist distribution.** Stock-basis head-KL is `0.014518` for SAGE-ARCS versus `0.010348` for SAGE. This is the measured tradeoff the build is intended to investigate, not an omitted regression.

* **head-KL is not full-vocabulary KL.** It uses an identical top-1024 truncation for every build and omits the mass-shift term: on the agentic corpus it is 10.3% (SAGE-ARCS) / 11.2% (SAGE) below full-vocabulary KL, and on the stock corpus 2.7% / 2.8%. Both metrics produce the same ordering in the comparisons reported here.

* **ΔCE is not a fidelity metric and is negative on the agentic basis.** Both builds predict that corpus slightly better than the bf16 source does. When read as distance from the source (`ΔCE = 0`), SAGE-ARCS is closer in 33/48 evaluation batches (`p = 0.013`). Read naively as “lower is better,” ΔCE would favor the build with 23% worse KL. It is reported for completeness rather than used as primary evidence of fidelity.

* **Top-5 agreement is saturated on the stock basis and informative on the others.** At 0.9994 it cannot discriminate on stock (`ns`, 8/19). On the agentic basis (0.969–0.974) and at long context (0.885–0.947), it separates the builds more clearly. Its usefulness depends on the evaluation distribution.

* **The full-vocabulary decomposition is reported as per-token means,** which sum to full-vocabulary KL exactly (residual ≤ 1.1e-08). All three terms favor SAGE-ARCS on the agentic basis.

* **Within-sequence position slopes are directional only.** SAGE-ARCS has a lower fitted degradation rate with token position on both bases, but the standard errors overlap. The difference is not established at this sample size.

* **48 evaluation batches is a modest sample.** The SAGE-versus-SAGE-ARCS difference is large and directionally consistent (`47/48` evaluation batches for full-vocabulary KL), but smaller differences among closely related internal builds are not reliably resolved at this sample size.

* **Fidelity is not task accuracy.** They are separate claims and were measured separately. Across 964 benchmark questions with thinking disabled, deterministic sampling, and MTP enabled, no paired comparison against a sibling build survives correction for multiplicity.

* **Benchmark totals are not a ranking.** One of the six benchmarks changed seven verdicts in a same-model, same-configuration repeat. On the five benchmarks that reproduce bit-identically, this build has the highest raw total; across all six it ranks fourth. At the observed repeatability of this harness, single-benchmark differences smaller than roughly seven questions should be treated as unresolved.

* **The bf16 MTP head costs 0.1407 bpw and is retained for speculative decoding.** The pre-graft and bf16-MTP builds agree on the scored forward path because the MTP head does not participate in those measurements. Its retained precision is intended for draft acceptance and decode throughput instead.

* **Fidelity is measured against the bf16 source.** A quantization that reproduces its source more faithfully also reproduces the source model's mistakes more faithfully.

## Quantization details

* **Model type:** `qwen3_5`
* **Parameters:** 27.78 B
* **Bits:** 5-bit base, with 123 per-tensor overrides placed by measured sensitivity
* **Group size:** 64
* **Format:** MLX safetensors
* **MTP:** preserved and restored to bf16
* **Stored tensors:** 2195

## Usage

This build is made for [oMLX](https://github.com/jundot/omlx): it was quantized through oMLX's `oQe` backend and is meant to be served by it, with MTP on.

### Serve with oMLX

Install oMLX with Homebrew, or see the [oMLX repository](https://github.com/jundot/omlx) for the other ways to install and run it:

```bash
brew install jundot/omlx/omlx
```

`omlx serve` discovers models as sub-directories of `--model-dir` (default `~/.omlx/models`); the directory name becomes the model id:

```bash
hf download notnaurw/Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp --local-dir ~/.omlx/models/Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp
omlx serve --model-dir ~/.omlx/models --port 8000
```

Then open the oMLX admin dashboard (`http://127.0.0.1:8000/admin`), open `Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp`'s model settings and turn on **Lightning MTP**. 

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.8-27B-oQ5e-SAGE-ARCS-mtp", "messages": [{"role": "user", "content": "Hello"}]}'
```

### mlx-vlm

Loads under a standard `mlx-lm` / `mlx-vlm` stack like any oMLX `oQe` build:

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model <this-repo> --max-tokens 256 --prompt "Hello"
```

## Attribution

Base model © the Qwen team, Apache-2.0 — see `LICENSE`.

Quantization performed with [oMLX](https://github.com/jundot/omlx) `0.6.4`. SAGE-ARCS is an adapter layer over oMLX's `oQe` backend.

> Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork*. August 2026.
> https://qwen.ai/blog?id=qwen3.8