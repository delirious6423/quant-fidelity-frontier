---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- qwen3.8
- thinking
- reasoning
- gguf
- quantized
- imatrix
- mtp
- llama.cpp
- agentic
- coding
- conversational
pipeline_tag: image-text-to-text
---

# Quality-first GGUF quants of Qwen3.8-27B

**Consider this fine-tune instead, it is even better:** [Kiwen1.2-27B](https://huggingface.co/NikiKrutan/Kiwen1.2-27B-MTP-GGUF)

**In the process of creating these quants:**

- **95 hours** of 2xGPU near 100% load were spent!
- **27 TB** of not so good ggufs were materialized, measured and put to trash *(all were usable)*!
- **72 man-hours** were spent!

These quants were produced by my quality-first experimental tool, **niki-allocator**. The main difference of my method is per-tensor type allocation (not per tensor groups, layers, community lore etc.) Each tensor gets what is *estimated* to be *best* for it. After that best allocations are tested with real KLD runs to choose the winner.

[Source model](https://huggingface.co/Qwen/Qwen3.8-27B)

## Quantization quality comparison

To prove my method I've performed comparison with quants from other providers:

[unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF), [bartowski](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF), [lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF), [mudler](https://huggingface.co/mudler/Qwen3.8-27B-APEX-GGUF), [byteshape](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF)

***Lower is better***

### Main metrics

![Mean KLD, niki-small-plus-2 corpus](plots/niki_mean_kld.png)
![Mean KLD, independent-v2 corpus](plots/indep_mean_kld.png)
![RMS Δp, niki-small-plus-2 corpus](plots/niki_rms_dp.png)
![RMS Δp, independent-v2 test corpus](plots/indep_rms_dp.png)

### Tail metrics
<details>

*p99.9 are extremely corpus-driven with usual test corpus sizes of ~300-400K tokens*
![p95 KLD, niki-small-plus-2 corpus](plots/niki_p95.png)
![p95 KLD, independent-v2 corpus](plots/indep_p95.png)
![p99 KLD, niki-small-plus-2 corpus](plots/niki_p99.png)
![p99 KLD, independent-v2 corpus](plots/indep_p99.png)
![p99.9 KLD, niki-small-plus-2 corpus](plots/niki_p99_9.png)
![p99.9 KLD, independent-v2 corpus](plots/indep_p99_9.png)

</details>

### Same top probability
*Useful for temperature 0, otherwise RMS Δp should be considered.*
<details>

Note that graphs show **1 - same top p**, so lower is better.

![Same top probability, niki-small-plus-2 corpus](plots/niki_same_top_p.png)
![Same top probability, independent-v2 corpus](plots/indep_same_top_p.png)
</details>

### Perplexity (PPL)
*Very crude noisy metric, cheaper to compute though, that's why so popular*
<details>

![PPL, niki-small-plus-2 corpus](plots/niki_ppl.png)
![PPL, independent-v2 corpus](plots/indep_ppl.png)
</details>

### Notes

*TL;DR **difference of corpora** for end user*: `niki-small-plus-2` - coding, agentic, reasoning, hard logic prompts, English, Russian; `independent-v2` - multilingual, English, web texts, science papers, math, legal, no reasoning or agentic traces, no code.

`Unsloth 3.0 preview`, `AtomicChat`, `bartowski`, `lmstudio-community` quants for comparison were downloaded on **August 16, 2026.** `Unsloth 3.0`, `mudler`, `byteshape` quants for comparison were downloaded on **August 21, 2026.** I've deliberately dropped some known bad quants (legacy quant types and such) to save my time on downloading and measuring.

Unsloth v3.0 was introduced *after* I finished my work. I added them for clean honest comparison. I don't like their marketing attitude and evidence presentation *(just drop in their best graph without details on how exactly it was measured and if it is meaningful metric for end user)*. But after all they provided really good quants (mostly).

Unsloth v3.0 dropped MTP for lower quants reasoning that they are not needed. Well, I can't agree. From my findings MTP works well on lower quants. I added these no-MTP quants to comparison, but added fixed-size correction of their MTP v3.0 fixed-choice for all their non-MTP quants. Yeah, I know.. So for *"honest"* *scientific* comparison of only target models without strange choices of MTP quantization, here is comparison for all quants where only size of target model is accounted. This is not that *honest* for the end user though. End user downloads quant and either fits it in his VRAM/RAM or not, poor choice of oversized MTP can't be excuse here. Quant provider should provide MTP quantization based on their research, not just drop in something, or just delete it. It is part of a quant. But here you go:

### All graphs with MTP size subtracted

<details>

### Main metrics

![Mean KLD, niki-small-plus-2 corpus](plots-nomtp/niki_mean_kld.png)
![Mean KLD, independent-v2 corpus](plots-nomtp/indep_mean_kld.png)
![RMS Δp, niki-small-plus-2 corpus](plots-nomtp/niki_rms_dp.png)
![RMS Δp, independent-v2 test corpus](plots-nomtp/indep_rms_dp.png)

### Tail metrics

*p99.9 are extremely corpus-driven with usual test corpus sizes of ~300-400K tokens*
![p95 KLD, niki-small-plus-2 corpus](plots-nomtp/niki_p95.png)
![p95 KLD, independent-v2 corpus](plots-nomtp/indep_p95.png)
![p99 KLD, niki-small-plus-2 corpus](plots-nomtp/niki_p99.png)
![p99 KLD, independent-v2 corpus](plots-nomtp/indep_p99.png)
![p99.9 KLD, niki-small-plus-2 corpus](plots-nomtp/niki_p99_9.png)
![p99.9 KLD, independent-v2 corpus](plots-nomtp/indep_p99_9.png)

### Same top probability
*Useful for temperature 0, otherwise RMS Δp should be considered.*

Note that graphs show **1 - same top p**, so lower is better.

![Same top probability, niki-small-plus-2 corpus](plots-nomtp/niki_same_top_p.png)
![Same top probability, independent-v2 corpus](plots-nomtp/indep_same_top_p.png)

### Perplexity (PPL)
*Very crude noisy metric, cheaper to compute though, that's why so popular*

![PPL, niki-small-plus-2 corpus](plots-nomtp/niki_ppl.png)
![PPL, independent-v2 corpus](plots-nomtp/indep_ppl.png)

</details>


***See the tables below for exact numbers for all quants and metrics***

## Sizes

I've chosen 15 sizes to fill 8500-23000 MiB with geometric progression and rounded to 100 MiB

*niki-allocator is able to produce allocation with ~1 MiB precision.*

### Which one to choose?

If you are after *quality*: ***the bigger - the better***. So it is always a compromise of what you can afford considering context size and other constraints. Choose the biggest you can fit.

<details>
<summary>About speed</summary>

I haven't measured *speed* since it is highly hardware and configuration dependable. What is faster for me may be slower for you. If the speed is the only that matters and you have Blackwell - get some `NVFP4` quant (but quality is considerably worse for the same quant size). Otherwise you should measure speed yourself on your hardware and your configuration to make real decision. By the way, on my setup `ubatch` is the parameter that has huge impact on speed. On 27B models 384 was found to be optimal in my case (2x5060Ti), 512 or 256 are much slower. So probably you should test this first.

</details>

### Quant names

Hugging Face requires quant type in the filename to properly show it. But all my quants are mixed by design. I decided to include dominating quant type (by size, not by tensor count) to the name therefore. It may easily be less than 50% of the model though. So **choose by size, ignore listed quant type**. For my quants: bigger --> better quality. Always.

## Methodology

### niki-allocator

<details>
<summary>What is niki-allocator</summary>

It is quality-first per-tensor quantization allocator for `llama.cpp` llama-quantize.

Allocator decides, per tensor, which quant type to use so that the resulting GGUF hits an exact target file size with the best *estimated* quality.

It is not direct measurement of KLD-metrics, it is only estimation. But as you can see from real KLD evaluations estimation is good enough.

Raw results are already good. But I added some calibration levers on top to make it better (for [Qwen3.5-0.8B](https://huggingface.co/NikiKrutan/Qwen3.5-0.8B-MTP-GGUF) I used uncalibrated estimation). These levers need series of KLD evaluations to move estimation closer to reality, so it takes hours-days of GPU time on my consumer setup.

On top of this I added bayesian search seeded with best results from levers to move calibration even further. This is very GPU-intensive, literally took days (nights actually) of GPU time on my setup.

</details>

### Corpora

<details>
<summary>Corpora choice, differences, details</summary>

- `imatrix`: `niki-small-2` corpus - ~330K tokens from my own agentic sessions (English, Russian, design docs with hard logic, code, reasoning). Deduped, cleaned, filtered by simhash similarity to include only really different turns. Turn-aligned by tokenizer so that each chunk contains exactly 1 turn. Short turns dropped. Used for `imatrix` only.
- Calibration of `niki-allocator`: hardest 16 chunks of `niki-small-plus-2` for faster crude KLD calibration (full corpus is too slow on my hardware).
- Evaluation 1: `niki-small-plus-2` corpus - ~400K tokens from my own agentic sessions (English, Russian, design docs with hard logic, code, reasoning). Deduped, cleaned, filtered by simhash similarity to include only really different turns. Turn-aligned by tokenizer so that each chunk contains exactly 1 turn. Short turns dropped. Does not include any exact or similar (by simhash) agentic turns from `niki-small`. So while it is from the same domain as imatrix-corpus, it is different.
- Evaluation 2 (truly *independent* this time): `independent-v2` corpus - ~390 K tokens. This corpus is random compilation of top-13 languages prose (`HuggingFaceFW/fineweb-2`) excluding English (it is everywhere in other domains already), legal domain (`common-pile/caselaw_access_project`), math domain (`open-web-math/open-web-math`), science domain (`recursal/arXiv-CC0-v0.5`). *Code and agentic data wasn't included since my quants are highly tuned for these already (it wouldn't be honest comparison).* **This dataset wasn't seen during calibration or imatrix at all.** Used only for final evaluation before publishing. So my quants are not anyhow tuned for this dataset. Foreign quants haven't seen this exact dataset as well (random pick).

Notes:
- `independent-v2` corpus will be published [here](https://huggingface.co/datasets/NikiKrutan/independent). *Note that you can't redistribute it without proper attributions due to license of some included papers.*
- `niki-*` corpora are not published. There is nothing really special actually. They just contain my private information. Sorry for inconvenience.
- Why not `wikitext-2 test`? While wiki may be considered as independent evaluation corpus for my quants. It easily may not be true for others quants (wikitext-2 is the most popular corpus somehow for such jobs). They may have seen it (or part of it) during imatrix creation or their methodology calibration. So for really ***independent*** evaluation there should be corpus that wasn't seen anyhow by any quants and even better if it wasn't seen during base model creation. Also `wiki` is mostly homogenous (not very easy, but there are very few hard tokens and chunks). My corpus is everything from "walk in the park" to "high mountain climbing". Its tail metrics are much higher. It is actually more representative for the real model work (if crawling wiki is not your main use case of course). So I use my own corpus because I need these quants to be good in my tasks. And `independent-v2` as truly independent evaluation.

</details>

### Chunk sizes

<details>

- `imatrix` (`niki-small-2`): 8064 tokens (384 x 21). 41 chunks total. To catch reasoning and long turn traces, not just basic chat.
- `niki-small-plus-2` evaluation: 6144 tokens (384 x 16). 65 chunks total. To catch reasoning and long turns, but to keep chunks amount somewhat reasonable (that's why lower than for imatrix).
- `independent-v2` evaluation: 2304 tokens (384 x 6). 171 chunks total. Long enough for chat and basic tasks.

*Why 384 magic number?*

*My hardware is fastest with `ubatch` 384, so I use it.*

</details>

### Metrics

All metrics were evaluated with `llama-perplexity` against **Q8_0** standard quant (due to hardware constraints).

Quantization was performed with patched `llama-quantize` from **BF_16** source. Fully compatible with original `llama.cpp`.

All my and foreign quants were evaluated against the same base.

`llama.cpp` commit `876a4321163249c43ca4e986818fab5ab081f282`.

The main metrics during calibration of `niki-allocator` were `mean KLD` and `RMS Δp`.

Better `PPL` doesn't always mean better other more important metrics. So it can't be used as a main metric *(some people still use)*. It is just fast indicator (no need for logits base) to check for big problems. It can't be main quality metric. Don't make or accept comparisons done only by PPL. It is noise.

Notes on `p99 KLD`, `p99.9 KLD`, `max KLD`.

With such a small corpora these metrics are not very robust. While `p99 KLD` can give some plausible comparison for the allocation method as a whole, `p99.9` is extremely corpus-driven (too little data), `max KLD` is just random at any corpus size (one hardest token of the whole dataset, saying nothing about other hard tokens). So I don't tune to these. And anybody who does with less than several M tokens corpus is just over-fitting to their exact corpus. *Please correct me if I am wrong.*

## Evaluation results

### niki-small-plus-2 corpus

<details>
<summary>Warning! Huge table inside!</summary>

*Lower is better **except same top p***

<p><b>Qwen3.8-27B</b> &mdash; niki-small-plus-2 corpus</p>
<table>
  <tr>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Source</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Label</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Size, MiB*</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Size w/o MTP, MiB</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Mean KLD</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">RMS Δp</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p95</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p99</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p99.9</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Same top p, %</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">PPL</th>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ1_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6240 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5905</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.59681 ± 0.00410</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">21.210 ± 0.063</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.6709</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10.2128</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">22.0603</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">75.44 ± 0.10</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.610 ± 0.026</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ1_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6752 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6417</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.53205 ± 0.00457</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18.591 ± 0.066</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.2365</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12.3683</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23.5830</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">78.62 ± 0.09</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.263 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ2_XXS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7264 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6929</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.42809 ± 0.00430</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15.342 ± 0.064</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.8708</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11.1274</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23.3418</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">81.79 ± 0.09</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.350 ± 0.028</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ1_M</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8104</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7826</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.40475 ± 0.00409</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15.342 ± 0.064</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.8258</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10.2505</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23.0421</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">81.78 ± 0.09</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.905 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ2_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">8319 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7984</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.35110 ± 0.00429</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12.211 ± 0.069</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.5158</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11.1857</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23.2209</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">85.77 ± 0.08</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.331 ± 0.029</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ2_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8500</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8316</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.29969 ± 0.00364</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11.724 ± 0.062</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.4843</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.4590</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">21.2619</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">85.49 ± 0.08</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.065 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8561</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8283</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.40775 ± 0.00449</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13.996 ± 0.067</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.6582</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12.1255</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23.5798</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">83.37 ± 0.08</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.615 ± 0.032</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ2_XXS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8593</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8382</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.29882 ± 0.00369</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11.384 ± 0.059</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.4793</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8.3221</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21.7652</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">85.56 ± 0.08</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.171 ± 0.027</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_XXS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8958</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8730</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.40331 ± 0.00404</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">14.430 ± 0.059</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.8150</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10.3693</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22.2155</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">81.90 ± 0.09</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.419 ± 0.029</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ2_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8925</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.29060 ± 0.00385</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10.310 ± 0.063</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.3783</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9.0926</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">22.4354</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">87.09 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.178 ± 0.028</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q2_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9374</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9039</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.23685 ± 0.00343</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9.208 ± 0.063</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.2930</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6.8906</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20.9554</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">88.65 ± 0.07</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.807 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9432</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9153</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.33312 ± 0.00420</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11.466 ± 0.066</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.4421</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10.7630</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">22.8707</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">86.09 ± 0.08</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.395 ± 0.030</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9524</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9296</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.38705 ± 0.00434</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12.842 ± 0.062</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.6032</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">11.7030</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22.9351</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">83.97 ± 0.08</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.470 ± 0.031</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_S-IQ2_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9748</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9469</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.29551 ± 0.00394</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10.421 ± 0.067</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.3584</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9.7478</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">22.0738</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">87.27 ± 0.07</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.248 ± 0.029</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9800</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9622</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.24855 ± 0.00365</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.648 ± 0.059</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2652</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.1560</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">22.2199</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">88.86 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.947 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9818</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9590</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.37596 ± 0.00436</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12.518 ± 0.066</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.5408</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">11.8180</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22.9918</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">84.70 ± 0.08</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.483 ± 0.031</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ2_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9842</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9636</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.29107 ± 0.00401</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9.528 ± 0.066</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2816</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">10.4280</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21.9791</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">87.91 ± 0.07</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.028 ± 0.026</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q2_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">10182</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9976</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.23467 ± 0.00349</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9.201 ± 0.067</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2472</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.3177</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21.1131</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">88.66 ± 0.07</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.010 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10370</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10142</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.32379 ± 0.00419</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10.543 ± 0.064</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.3837</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10.7744</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22.9631</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">86.66 ± 0.08</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.328 ± 0.030</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ3_XXS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10428</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10094</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.21417 ± 0.00346</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.863 ± 0.069</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1830</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.0054</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">21.3873</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">90.48 ± 0.07</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.886 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10500</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10322</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.19533 ± 0.00309</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.268 ± 0.062</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2035</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">5.4233</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">19.7968</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">89.92 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.712 ± 0.021</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10626</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10347</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.24532 ± 0.00358</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9.189 ± 0.066</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.2605</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.7219</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">21.6210</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">88.96 ± 0.07</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.076 ± 0.026</td>
  </tr>
  <tr>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: left;"><b>mudler</b></td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">APEX-I-Nano</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">10720</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">10289</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.28538 ± 0.00385</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">10.279 ± 0.070</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.3606</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">9.3696</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">21.9993</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">87.91 ± 0.07</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">4.286 ± 0.029</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ3_S-3.44bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11226</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11058</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.20084 ± 0.00331</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.754 ± 0.072</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1755</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">6.1658</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">20.8096</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">91.00 ± 0.06</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.955 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11300</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11116</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.18301 ± 0.00312</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">6.935 ± 0.065</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1429</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">5.7343</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">20.1450</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">91.42 ± 0.06</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.845 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ3_XXS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11362</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11167</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.20210 ± 0.00335</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.402 ± 0.071</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1454</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">6.8158</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">20.7976</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">91.19 ± 0.06</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.714 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ3_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11483</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11148</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.18327 ± 0.00331</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6.523 ± 0.069</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1247</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5.5946</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">21.1550</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">92.14 ± 0.06</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.837 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11516</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11238</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.19840 ± 0.00325</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.755 ± 0.068</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.1677</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">6.1414</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">20.4022</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">90.75 ± 0.07</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.934 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-3.67bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11975</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11806</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.17338 ± 0.00312</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">6.696 ± 0.072</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1239</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">5.1036</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">20.5042</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">92.21 ± 0.06</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.728 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_XXS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12042</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">11814</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.19534 ± 0.00327</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8.055 ± 0.070</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1887</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">5.6009</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">20.9745</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">90.41 ± 0.07</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.779 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">12100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11890</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.17271 ± 0.00314</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">6.316 ± 0.069</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1059</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">5.3612</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">20.4380</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">92.16 ± 0.06</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.740 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_S-IQ3_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12383</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12104</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.20045 ± 0.00341</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">6.686 ± 0.072</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.1206</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.2896</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">20.8107</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">91.87 ± 0.06</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.799 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q3_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12537</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12203</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.17021 ± 0.00325</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6.254 ± 0.079</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0846</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5.2435</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20.9727</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">93.10 ± 0.06</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.712 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12713</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12485</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.19216 ± 0.00319</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.144 ± 0.067</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1438</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">6.3475</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19.8683</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">91.29 ± 0.06</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.865 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q3_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">12818</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">12620</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.14791 ± 0.00288</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">6.059 ± 0.073</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0843</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.0237</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">19.7203</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">93.02 ± 0.06</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.698 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">13000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">12790</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.14490 ± 0.00287</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">5.600 ± 0.071</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0753</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.8287</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">19.8499</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">93.17 ± 0.06</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.757 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-4.00bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">13030</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">12835</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.15545 ± 0.00301</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">5.622 ± 0.071</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0842</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">4.3696</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">20.4840</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">93.33 ± 0.06</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.813 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13197</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12971</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.15641 ± 0.00298</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">5.882 ± 0.070</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0908</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.4304</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">20.3319</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">92.90 ± 0.06</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.820 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">13259</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">13032</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.17815 ± 0.00308</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.029 ± 0.069</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1270</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">5.3719</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19.9435</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">91.88 ± 0.06</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.880 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: left;"><b>mudler</b></td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">APEX-I-Mini</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">13297</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">12867</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.20655 ± 0.00349</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">7.053 ± 0.074</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.1328</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">7.2564</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">21.9309</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">91.81 ± 0.06</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">3.904 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ4_XS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">13593</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">13258</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.14809 ± 0.00301</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5.766 ± 0.085</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0524</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.2925</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20.2216</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">94.27 ± 0.05</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.622 ± 0.021</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ4_XS-IQ3_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13769</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13534</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.14858 ± 0.00294</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">5.325 ± 0.071</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0713</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.2618</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">19.8525</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">93.39 ± 0.06</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.834 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">14000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">13790</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.12202 ± 0.00264</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.917 ± 0.075</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0480</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.8676</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18.7476</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">94.43 ± 0.05</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.696 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-4.40bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">14344</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">14115</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.13487 ± 0.00284</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">4.788 ± 0.077</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0465</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.6616</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">19.6340</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">94.69 ± 0.05</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.777 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">14647</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">14312</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.12428 ± 0.00274</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.637 ± 0.079</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0370</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.8792</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19.6644</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">95.15 ± 0.05</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.688 ± 0.022</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ4_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">14847</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">14619</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.12968 ± 0.00271</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">5.155 ± 0.081</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0467</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.3971</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18.9806</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">94.47 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.905 ± 0.025</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">IQ4_XS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">14978</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">14725</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.11673 ± 0.00258</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.766 ± 0.078</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0423</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.7687</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18.5722</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">94.75 ± 0.05</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.863 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">15000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">14785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.11079 ± 0.00253</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.354 ± 0.076</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0362</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.3719</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18.7210</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">95.06 ± 0.05</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.741 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">Q5_K_S-4.72bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">15394</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">15144</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.13078 ± 0.00281</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">4.478 ± 0.080</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0356</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.5353</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">19.4105</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">95.07 ± 0.05</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.763 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15702</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15367</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.11679 ± 0.00264</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.138 ± 0.078</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0300</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.6716</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19.2746</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">95.47 ± 0.05</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.809 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ4_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15748</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15490</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.11266 ± 0.00256</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.346 ± 0.078</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0329</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.4330</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">18.9169</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">95.22 ± 0.05</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.769 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">15939</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">15711</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.12847 ± 0.00276</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.698 ± 0.079</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0399</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.3534</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19.5019</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">94.76 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.763 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: left;"><b>lmstudio-community</b></td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">16032</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">15781</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.16945 ± 0.00319</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">6.184 ± 0.077</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0850</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">5.4682</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">20.9170</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">93.02 ± 0.06</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">3.639 ± 0.021</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">16100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">15885</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.10334 ± 0.00243</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.209 ± 0.076</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0313</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.0672</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18.0062</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">95.27 ± 0.05</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.758 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16314</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16038</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.12098 ± 0.00269</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.572 ± 0.079</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0378</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.8831</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">19.4748</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">94.97 ± 0.05</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.769 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q4_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">16328</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">16062</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.11616 ± 0.00265</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.432 ± 0.082</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0300</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.6443</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">19.0882</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">95.42 ± 0.05</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.703 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16746</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16411</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.11012 ± 0.00258</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.950 ± 0.082</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0225</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.3869</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18.8515</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">95.89 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.770 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">16949</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">16721</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.11468 ± 0.00262</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.188 ± 0.078</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0320</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.5264</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18.8423</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">95.23 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.822 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q4_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">17093</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16821</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.10611 ± 0.00252</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.925 ± 0.080</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0230</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.2947</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18.6469</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">95.88 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.772 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q5_K_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">17300</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">17085</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.09718 ± 0.00239</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.641 ± 0.079</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0191</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.8434</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18.0073</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">96.10 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.780 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q5_K-Q4_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">17688</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">17411</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.09972 ± 0.00240</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.907 ± 0.081</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0216</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.9442</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">18.0818</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">96.01 ± 0.04</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.730 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17801</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17466</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.10612 ± 0.00251</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.671 ± 0.082</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0170</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.3331</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18.3369</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">96.18 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.796 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17849</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17621</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.11493 ± 0.00260</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.304 ± 0.079</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0311</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.7333</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18.9470</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">95.34 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.822 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">Q5_K_M-5.60bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">18249</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">17965</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.09784 ± 0.00241</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.863 ± 0.085</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0164</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">1.9265</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">17.9692</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">96.35 ± 0.04</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.715 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q5_K_S</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18377</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18080</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.10613 ± 0.00253</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.784 ± 0.082</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0192</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.2872</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18.5994</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.03 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.739 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q5_K_M</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18600</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18385</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.08612 ± 0.00224</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.372 ± 0.082</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0135</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.4814</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">17.2961</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">96.57 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.792 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18769</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18541</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.09956 ± 0.00244</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.644 ± 0.081</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0177</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.9769</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18.5421</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.09 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.798 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18856</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18521</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.09175 ± 0.00233</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.449 ± 0.082</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0132</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.6733</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17.4314</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">96.60 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.789 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q5_K_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18915</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18606</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.10288 ± 0.00251</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.644 ± 0.082</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0172</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.0636</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18.6094</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.23 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.783 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q5_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">19282</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18958</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.08932 ± 0.00230</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.423 ± 0.082</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0135</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.5523</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">17.6310</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.52 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.806 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q5_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">19295</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">18990</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.08701 ± 0.00225</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.311 ± 0.081</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0126</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.4779</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">17.3852</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">96.63 ± 0.04</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.760 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19791</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19563</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.09182 ± 0.00233</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.745 ± 0.086</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0142</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.6026</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17.6914</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.47 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.757 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19910</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19575</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.09123 ± 0.00232</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.344 ± 0.084</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0112</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.6803</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17.6995</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">96.88 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.790 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">20000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">19785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.07126 ± 0.00200</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.306 ± 0.087</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0085</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.0398</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">15.9125</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">97.17 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.772 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">20540</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">20312</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.08929 ± 0.00231</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.458 ± 0.084</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0137</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.4471</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17.7038</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.67 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.779 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20965</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20631</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.08346 ± 0.00219</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.176 ± 0.087</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0086</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.3687</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17.0122</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.13 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.754 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: left;"><b>lmstudio-community</b></td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">21392</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">21060</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.07748 ± 0.00211</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">3.277 ± 0.083</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0105</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">1.1575</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">16.4780</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">96.89 ± 0.04</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">3.734 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">21400</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">21185</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.05081 ± 0.00163</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.581 ± 0.083</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0054</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.4680</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">13.4905</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">97.68 ± 0.03</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.770 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21824</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21480</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.05497 ± 0.00169</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.794 ± 0.082</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0069</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.5595</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">13.8988</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">97.36 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.790 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q6_K-Q5_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">22012</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">21689</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.08001 ± 0.00217</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.097 ± 0.083</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0093</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.2233</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">16.9064</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">97.14 ± 0.04</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.772 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">22019</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">21684</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.07816 ± 0.00211</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.066 ± 0.087</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0069</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.1757</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16.4711</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.36 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.732 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22376</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22148</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.04659 ± 0.00156</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.447 ± 0.079</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0054</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.3869</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12.4783</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">97.58 ± 0.03</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.756 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q6_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22963</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22736</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.04494 ± 0.00151</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.301 ± 0.079</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0046</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.3704</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">11.9331</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">97.87 ± 0.03</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.768 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K_L</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">23000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">22785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.03539 ± 0.00132</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.125 ± 0.079</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0037</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2434</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10.4939</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">98.06 ± 0.03</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.782 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_L</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23073</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">22738</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.06866 ± 0.00198</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.797 ± 0.085</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0055</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.8841</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15.8937</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.66 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.753 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q6_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23847</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23493</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.04172 ± 0.00146</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.283 ± 0.082</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0040</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.3349</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11.6762</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">98.00 ± 0.03</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.785 ± 0.024</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">24127</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23792</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.06625 ± 0.00196</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.756 ± 0.087</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0046</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.7587</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15.7494</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.78 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.758 ± 0.023</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q6_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">24723</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">24334</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.03966 ± 0.00145</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.136 ± 0.081</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0031</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2803</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11.7220</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">98.14 ± 0.03</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.768 ± 0.023</td>
  </tr>
</table>

</details>

### independent-v2 corpus

<details>
<summary>Warning! Huge table inside!</summary>

*Lower is better **except same top p***

<p><b>Qwen3.8-27B</b> &mdash; independent-v2 corpus</p>
<table>
  <tr>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Source</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Label</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Size, MiB*</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Size w/o MTP, MiB</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Mean KLD</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">RMS Δp</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p95</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p99</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">KLD p99.9</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">Same top p, %</th>
    <th style="background: rgba(128, 128, 128, 0.18); text-align: center;">PPL</th>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ1_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6240 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5905</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.60813 ± 0.00186</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23.522 ± 0.069</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.1155</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.1158</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.1822</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">66.99 ± 0.11</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12.776 ± 0.078</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ1_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6752 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6417</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.42429 ± 0.00145</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19.376 ± 0.065</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.4844</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.2142</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6.3670</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">72.03 ± 0.10</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10.723 ± 0.064</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ2_XXS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7264 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">6929</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.31338 ± 0.00120</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16.361 ± 0.062</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.1123</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.5462</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5.5458</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">75.58 ± 0.10</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9.676 ± 0.058</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ1_M</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8104</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7826</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.28422 ± 0.00107</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15.475 ± 0.060</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.9671</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.2719</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">5.1406</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">76.70 ± 0.10</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9.390 ± 0.055</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ2_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">8319 *</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7984</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.17177 ± 0.00080</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11.770 ± 0.057</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.6043</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.5694</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.2142</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">81.83 ± 0.09</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">8.561 ± 0.051</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ2_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8500</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8316</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.18872 ± 0.00083</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">12.664 ± 0.058</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.6395</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.6962</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.4760</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">80.75 ± 0.09</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.639 ± 0.050</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8561</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8283</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.21409 ± 0.00088</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13.325 ± 0.057</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.7081</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.8009</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.7080</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">79.50 ± 0.09</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8.848 ± 0.052</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ2_XXS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8593</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8382</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.17763 ± 0.00079</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">12.199 ± 0.058</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.5790</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.5806</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">4.3460</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">80.77 ± 0.09</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8.656 ± 0.051</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_XXS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8958</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8730</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.26214 ± 0.00105</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">15.046 ± 0.062</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.8957</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.2519</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">5.2075</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">77.56 ± 0.09</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9.283 ± 0.055</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ2_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8925</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.13775 ± 0.00068</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10.720 ± 0.056</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.4627</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.2822</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.8480</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">83.43 ± 0.08</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.307 ± 0.049</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q2_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9374</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9039</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.10854 ± 0.00059</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">9.232 ± 0.053</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.3701</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.0421</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.4345</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">85.33 ± 0.08</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">8.142 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9432</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9153</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.14253 ± 0.00069</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10.673 ± 0.053</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.4686</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.2057</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.6325</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">83.14 ± 0.08</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8.336 ± 0.049</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9524</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9296</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.19347 ± 0.00085</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12.798 ± 0.058</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.6536</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.7233</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.5241</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">80.57 ± 0.09</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8.736 ± 0.051</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_S-IQ2_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9748</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9469</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.11888 ± 0.00059</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">9.742 ± 0.052</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.3880</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.0273</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">3.2534</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">84.46 ± 0.08</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8.171 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9800</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9622</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.10149 ± 0.00053</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">9.141 ± 0.052</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.3371</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.9496</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.0609</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">85.73 ± 0.08</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.054 ± 0.047</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9818</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">9590</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.17692 ± 0.00081</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12.181 ± 0.058</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.5928</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.5965</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">4.4122</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">81.30 ± 0.09</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8.616 ± 0.051</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ2_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9842</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9636</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.10846 ± 0.00054</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9.425 ± 0.052</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.3404</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.9451</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.1076</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">84.34 ± 0.08</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8.168 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q2_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">10182</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">9976</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.09559 ± 0.00049</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8.815 ± 0.050</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2962</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.8225</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.8128</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">85.05 ± 0.08</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">8.074 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ2_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10370</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10142</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.12842 ± 0.00061</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">10.298 ± 0.054</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.4225</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.1543</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.5092</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">83.75 ± 0.08</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">8.262 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ3_XXS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10428</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">10094</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.07049 ± 0.00044</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.453 ± 0.050</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.2356</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.6935</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.3984</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">87.90 ± 0.07</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.894 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_XXS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10500</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">10322</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.07886 ± 0.00043</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">8.066 ± 0.047</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2452</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.6858</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.4708</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">87.33 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.895 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ2_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10626</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">10347</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.08589 ± 0.00046</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">8.298 ± 0.047</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.2786</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.7362</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.4607</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">86.99 ± 0.08</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.958 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: left;"><b>mudler</b></td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">APEX-I-Nano</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">10720</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">10289</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.09942 ± 0.00053</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">8.882 ± 0.053</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.3395</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.9385</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">3.0039</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">86.06 ± 0.08</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">8.091 ± 0.048</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ3_S-3.44bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11226</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11058</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.06952 ± 0.00044</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.384 ± 0.051</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.2318</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.7017</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">2.6157</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">88.54 ± 0.07</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.894 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11300</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11116</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.05702 ± 0.00035</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">6.788 ± 0.046</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1825</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.5295</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.9456</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">89.37 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.783 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-IQ3_XXS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11362</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">11167</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.05629 ± 0.00037</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">6.829 ± 0.049</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1801</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.5481</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.1643</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">89.45 ± 0.07</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.837 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ3_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11483</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">11148</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.04598 ± 0.00031</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">5.902 ± 0.044</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1538</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.4554</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.6954</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">90.34 ± 0.07</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.735 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11516</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">11238</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.05924 ± 0.00034</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">6.850 ± 0.043</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.1870</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.5114</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.8070</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">89.11 ± 0.07</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.795 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-3.67bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11975</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">11806</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.05568 ± 0.00038</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">6.634 ± 0.050</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1842</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.5734</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">2.1963</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">89.73 ± 0.07</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.804 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_XXS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12042</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">11814</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.06745 ± 0.00040</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.340 ± 0.048</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.2191</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.6406</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.3279</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">88.23 ± 0.07</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.862 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ3_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">12100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">11890</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.04349 ± 0.00031</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">5.962 ± 0.047</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1365</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.4072</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.7680</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">90.25 ± 0.07</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.746 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_S-IQ3_XXS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12383</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12104</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.03803 ± 0.00026</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">5.393 ± 0.040</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.1243</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.3376</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.3080</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">90.88 ± 0.07</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.687 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q3_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12537</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">12203</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.03045 ± 0.00023</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.843 ± 0.041</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1006</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.2859</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.1419</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">92.00 ± 0.06</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.657 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12713</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">12485</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.05041 ± 0.00031</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">6.326 ± 0.042</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1600</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.4567</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.6053</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">89.92 ± 0.07</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.747 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q3_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">12818</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">12620</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.03249 ± 0.00024</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">5.156 ± 0.041</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1015</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2971</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.2002</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">91.63 ± 0.06</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.687 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">13000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">12790</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.03034 ± 0.00025</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">4.923 ± 0.042</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0942</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2761</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.3123</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">91.81 ± 0.06</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.663 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-4.00bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">13030</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">12835</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.03808 ± 0.00028</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">5.450 ± 0.045</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1261</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.4026</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">1.5728</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">91.46 ± 0.06</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.707 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ3_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13197</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">12971</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.02895 ± 0.00021</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.662 ± 0.036</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0957</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.2545</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.9263</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">92.02 ± 0.06</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.666 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ3_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">13259</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">13032</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.04607 ± 0.00028</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">6.120 ± 0.041</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1435</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.4139</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.5318</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">90.53 ± 0.07</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.750 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: left;"><b>mudler</b></td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">APEX-I-Mini</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">13297</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">12867</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.04092 ± 0.00030</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">5.675 ± 0.045</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.1354</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">0.4037</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">1.5444</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">91.03 ± 0.06</td>
    <td style="background: rgba(58, 10, 44, 0.8); text-align: center;">7.725 ± 0.046</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-IQ4_XS</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">13593</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">13258</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.02075 ± 0.00018</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">4.029 ± 0.040</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0662</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1947</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.9067</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">93.34 ± 0.06</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.605 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ4_XS-IQ3_S</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13769</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">13534</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.02314 ± 0.00017</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">4.180 ± 0.036</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0744</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.2036</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.8089</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">92.82 ± 0.06</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.635 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">14000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">13790</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.01777 ± 0.00016</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.754 ± 0.038</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0552</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1621</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.6999</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">93.75 ± 0.06</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.600 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">IQ4_XS-4.40bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">14344</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">14115</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.01946 ± 0.00017</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.825 ± 0.034</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0639</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1949</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.7830</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">93.69 ± 0.06</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.588 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">14647</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">14312</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.01431 ± 0.00015</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">3.376 ± 0.038</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0453</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1370</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.6510</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">94.58 ± 0.05</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.574 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">IQ4_XS</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">14847</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">14619</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.01530 ± 0.00015</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.445 ± 0.036</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0480</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1374</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.5749</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">94.27 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.588 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">IQ4_XS</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">14978</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">14725</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.01467 ± 0.00015</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.339 ± 0.034</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0466</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1319</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.5580</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">94.33 ± 0.05</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.584 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">15000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">14785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.01448 ± 0.00015</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.383 ± 0.035</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0453</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1371</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.6066</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">94.38 ± 0.05</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.578 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">Q5_K_S-4.72bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">15394</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">15144</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.01562 ± 0.00015</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">3.447 ± 0.036</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0505</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.1536</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.6493</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">94.31 ± 0.05</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.571 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15702</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">15367</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.01037 ± 0.00013</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.832 ± 0.032</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0331</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0952</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.4359</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">95.33 ± 0.05</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.558 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-IQ4_XS</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15748</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">15490</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.01102 ± 0.00013</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.886 ± 0.032</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0349</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0989</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.4484</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">95.10 ± 0.05</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.556 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">15939</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">15711</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.01323 ± 0.00013</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">3.163 ± 0.031</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0424</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1194</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.4670</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">94.50 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.571 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: left;"><b>lmstudio-community</b></td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">16032</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">15781</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.01954 ± 0.00017</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">3.796 ± 0.034</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0631</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.1724</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.6741</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">93.41 ± 0.06</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">7.601 ± 0.045</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-IQ4_XS</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">16100</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">15885</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.01211 ± 0.00014</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">3.070 ± 0.032</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0373</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1066</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.4506</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">94.69 ± 0.05</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.564 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16314</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16038</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.01211 ± 0.00013</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">3.017 ± 0.030</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0384</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1088</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.4570</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">94.75 ± 0.05</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.570 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q4_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">16328</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">16062</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.01002 ± 0.00013</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.769 ± 0.031</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0319</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0876</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.3724</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">95.26 ± 0.05</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.555 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q4_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16746</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">16411</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00768 ± 0.00012</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.410 ± 0.029</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0242</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0705</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.3030</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">95.88 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.546 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">16949</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">16721</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.01058 ± 0.00012</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.864 ± 0.032</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0332</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0953</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.4210</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">95.09 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.563 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q4_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">17093</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">16821</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00822 ± 0.00012</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.507 ± 0.030</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0255</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0754</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.3284</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">95.74 ± 0.05</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.548 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q5_K_S</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">17300</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">17085</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.00745 ± 0.00012</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">2.396 ± 0.029</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0232</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0687</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.2908</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">95.96 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.539 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q5_K-Q4_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">17688</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">17411</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.00687 ± 0.00012</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">2.297 ± 0.031</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0216</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0593</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.2554</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">96.01 ± 0.04</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.540 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_S</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17801</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">17466</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00569 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">2.102 ± 0.028</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0179</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0520</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.2294</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">96.40 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.539 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q4_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17849</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">17621</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00997 ± 0.00013</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">2.753 ± 0.030</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0319</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0903</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.3688</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">95.34 ± 0.05</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.560 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: left;"><b>byteshape</b></td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">Q5_K_M-5.60bpw</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">18249</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">17965</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.00675 ± 0.00011</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">2.248 ± 0.030</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0211</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.0619</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">0.2567</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">96.12 ± 0.04</td>
    <td style="background: rgba(36, 40, 46, 0.8); text-align: center;">7.540 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q5_K_S</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18377</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18080</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00575 ± 0.00009</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">2.109 ± 0.026</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0178</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0494</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.2115</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.25 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.534 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q5_K_M</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18600</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">18385</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.00468 ± 0.00010</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.879 ± 0.025</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0144</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0422</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1810</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">96.73 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.527 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_S</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18769</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">18541</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00527 ± 0.00010</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.982 ± 0.026</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0163</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0438</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1773</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.41 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.533 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18856</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">18521</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00427 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.822 ± 0.030</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0129</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0366</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1589</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">96.78 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.527 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q5_K_M</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18915</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18606</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00492 ± 0.00009</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.921 ± 0.027</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0151</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0402</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1639</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.48 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.530 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q5_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">19282</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">18958</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00415 ± 0.00009</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.766 ± 0.026</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0126</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0353</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.1452</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">96.80 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.530 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q5_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">19295</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">18990</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.00391 ± 0.00006</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.747 ± 0.026</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0118</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0331</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.1490</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">96.88 ± 0.04</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.526 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_M</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19791</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">19563</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00439 ± 0.00011</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.837 ± 0.031</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0133</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0356</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1430</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.68 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.529 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q5_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19910</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">19575</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00322 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.550 ± 0.025</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0101</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0278</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.1239</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.32 ± 0.04</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.525 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">20000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">19785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.00280 ± 0.00005</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.462 ± 0.021</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0088</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0250</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.1108</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">97.51 ± 0.04</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.519 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q5_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">20540</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">20312</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00402 ± 0.00011</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.738 ± 0.029</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0125</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0337</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.1400</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">96.97 ± 0.04</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.527 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20965</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">20631</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00244 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.353 ± 0.029</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0074</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0207</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0885</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.65 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.521 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: left;"><b>lmstudio-community</b></td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">21392</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">21060</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.00267 ± 0.00003</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">1.419 ± 0.020</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0081</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0210</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">0.0851</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">97.29 ± 0.04</td>
    <td style="background: rgba(30, 10, 50, 0.8); text-align: center;">7.521 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">21400</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">21185</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.00192 ± 0.00008</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">1.192 ± 0.019</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0056</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0162</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0714</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">97.93 ± 0.03</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.519 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21824</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">21480</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00209 ± 0.00010</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">1.224 ± 0.019</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0059</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0153</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0603</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">97.58 ± 0.04</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.522 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q6_K-Q5_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">22012</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">21689</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.00248 ± 0.00004</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">1.404 ± 0.030</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0077</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0217</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0945</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">97.67 ± 0.03</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.521 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_M</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">22019</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">21684</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00198 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.210 ± 0.030</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0059</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0168</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0710</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">97.99 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.519 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q6_K</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22376</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22148</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00176 ± 0.00009</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.136 ± 0.016</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0050</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0129</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0506</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">97.69 ± 0.03</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.517 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: left;"><b>bartowski</b></td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">Q6_K_L</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22963</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">22736</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.00142 ± 0.00007</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">1.001 ± 0.015</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0042</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0117</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">0.0454</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">98.19 ± 0.03</td>
    <td style="background: rgba(0, 40, 20, 0.8); text-align: center;">7.516 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: left;"><b>NikiKrutan</b></td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">NIKI-Q6_K_L</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">23000</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">22785</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.00124 ± 0.00002</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.986 ± 0.024</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0038</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0106</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">0.0428</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">98.23 ± 0.03</td>
    <td style="background: rgba(0, 11, 38, 0.8); text-align: center;">7.515 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_L</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23073</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">22738</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00163 ± 0.00010</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.113 ± 0.035</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0046</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0133</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0574</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">98.18 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.517 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: left;"><b>AtomicChat</b></td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">AD-Q6_K</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23847</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">23493</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.00105 ± 0.00002</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.889 ± 0.021</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0032</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0087</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">0.0344</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">98.45 ± 0.03</td>
    <td style="background: rgba(60, 10, 10, 0.8); text-align: center;">7.517 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: left;"><b>unsloth 3.0</b></td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">UD-Q6_K_XL</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">24127</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">23792</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.00139 ± 0.00009</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">1.029 ± 0.034</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0039</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0111</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">0.0504</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">98.34 ± 0.03</td>
    <td style="background: rgba(70, 38, 0, 0.8); text-align: center;">7.517 ± 0.044</td>
  </tr>
  <tr>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: left;"><b>unsloth 3.0 preview</b></td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">UD-Q6_K_XL</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">24723</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">24334</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.00102 ± 0.00007</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.851 ± 0.011</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0029</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0083</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">0.0359</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">98.41 ± 0.03</td>
    <td style="background: rgba(60, 50, 0, 0.8); text-align: center;">7.515 ± 0.044</td>
  </tr>
</table>

</details>

## MTP

<details>

I've tested MTP performance and discovered that there is no point to keep MTP head higher than `IQ4_XS`. `Q3_K` is near but slightly worse (used for lower quants). Exact allocation of IQ4_XS/Q3_K is chosen by allocator (not pinned). Lower than that there is considerable quality drop, so not used in my quants. Higher is just waste of model size and VRAM.

For numbers check my previous [GRM-2.6-Plus-0628](https://huggingface.co/NikiKrutan/GRM-2.6-Plus-0628-MTP-GGUF) quant's MTP section.

I've done fast tests on other models (same architecture). Same conclusions.

Also I've additionally checked fast if there is any degradation on `IQ4_XS` vs `Q8_0` with longer context (tested up to 28K), none found. And temperature = 0.6, still no evidence of degradation. But I must say that doing this test properly and scientifically much time and effort is needed. Since not 0 temperature gives different results: many runs needed to stabilize. Reasonably long context (100K+ from my opinion) is very slow on my hardware. So there actually may be some degradation. But that should be proven and not taken by "it seems" or "everybody knows". *If there is such comprehensive analysis already, please let me know.*

Basically I use these parameters for MTP: `--spec-draft-n-max` 5 `--spec-draft-p-min` 0.8. Primary reason of poor results with MTP is ignoring `--spec-draft-p-min`. Don't put obvious crap in your drafts. But let drafter do more if it is sure enough. MTP performance speed-up is highly dependable on hardware, configuration and specific context. So to decide what is best for your case you should test yourself.

Also don't use other `llama.cpp` drafters like `ngram-mod` with MTP. MTP wastes time even if `ngram-mod` already done draft. That is how `llama.cpp` is programmed. Not obvious. I have created experimental fork to overcome this issue (and added much better `ngram-mod-v2` + minor fixes): https://github.com/NikiKrutan/niki-llama.cpp. I use it myself on a daily basis, but it is more like crude draft than real working fork. It breaks some llama.cpp behavior. But it gives another ~1.5x speed-up on top of MTP for my use cases.

</details>

## Chat template

Chat template included with my gguf's is v22.2 from [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates). There may be newer version as it is often updated.

If you have any problems with it you may also try one from [Moore2877](https://huggingface.co/Moore2877/Qwen-Fixed-Chat-Templates-llamacpp).

## Multimodality

I've quantized mmproj following [DAXZEIT's receipt for Qwen3.6](https://huggingface.co/DAXZEIT/Qwen3.6-27B-mmproj-hybrid-Q8_0-F16-gguf) that I used in Qwen3.6 era.

It reduces mmproj size saving practically same quality. Why reducing size? Well, it is probably not needed for most use cases. But with this I can put my 17300 quant with q8_0 K-cache and q4_0 V-cache + MTP with the same KV-cache quantization + mmproj + 262K full context in my 2x16Gb VRAM. With BF16 mmproj I can't do it despite very little difference in size. Every MiB matters when you are about quality.

Also any standard Qwen3.8-27B mmproj should work if you want full BF16 one.

## Acknowledgments

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) - great base model. Really great!
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - quantization and evaluation framework.
- [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) - quants for comparison.
- [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF) - quants for comparison.
- [bartowski](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) - quants for comparison.
- [lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) - quants for comparison.
- [mudler](https://huggingface.co/mudler/Qwen3.8-27B-APEX-GGUF) - quants for comparison.
- [byteshape](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) - quants for comparison.
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) - data for evaluation dataset.
- [caselaw_access_project](https://huggingface.co/datasets/common-pile/caselaw_access_project) - data for evaluation dataset.
- [open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) - data for evaluation dataset.
- [recursal](https://huggingface.co/datasets/recursal/arXiv-CC0-v0.5) - data for evaluation dataset.
- [DAXZEIT](https://huggingface.co/DAXZEIT/Qwen3.6-27B-mmproj-hybrid-Q8_0-F16-gguf) - mmproj receipt.
- [michaelw9999](https://github.com/michaelw9999/advanced-gguf-quantizer) - some ideas to start with.
- Required attributions for exact science papers used for evaluation dataset are on [dataset page](https://huggingface.co/datasets/NikiKrutan/independent).