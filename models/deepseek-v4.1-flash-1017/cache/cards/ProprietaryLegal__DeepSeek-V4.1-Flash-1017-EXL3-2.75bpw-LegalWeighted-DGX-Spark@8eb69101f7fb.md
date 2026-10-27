---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
- dgx-spark
- gb10
- legal
---

# DeepSeek-V4.1-Flash-1017 · EXL3 2.75 bpw · legal-weighted expert allocation (2× DGX Spark)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

**Serving stack for this model:** two DGX Spark (GB10) nodes with x86 offload (remote Engram, RTX 3090 drafter, async hand-off) — [overview on GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [all entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:end -->

<!-- pli:update:2026-10-27:start -->

**Update 2026-10-27:** the full DSV41 source is now public — [dsv41-offload](https://huggingface.co/ProprietaryLegal/dsv41-offload) ([GitHub](https://github.com/ProprietaryLegal/dsv41-offload), every component with `production`, `main` and per-experiment branches), with focused repos for the [measurement harness](https://huggingface.co/ProprietaryLegal/dsv41-window-harness) and [sampled drafting](https://huggingface.co/ProprietaryLegal/dsv41-sampled-drafting). Status and lessons: [2026-10-27 post](https://github.com/ProprietaryLegal/pli-labs/blob/main/research/dgx-spark-offload/2026-10-27-status-and-code.md).

<!-- pli:update:2026-10-27:end -->



**TL;DR**: a variant of [HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark)
(revision `86f85253`) that has **the same size and the same format**.
- **What changed (legal-weighted bit allocation):** 1,414 of the 15,360 routed experts, in 16 of the 40 layers, moved
  bits toward the experts that measure as more important on legal text, and away from experts that matter less there.
  Only those 1,414 experts were re-encoded. They used the baseline's own calibration set (sudoingX `cal_v3`), so every other byte is the baseline's.
- **KLD against the native 1017**, lower is better: legal panel **0.0274 → 0.0249 (−9.0 %)**, general panel
  **0.0201 → 0.0204 (+1.5 %)**.
- **Same runtime as the baseline:** TP2 on two DGX Spark / GB10, the same serving image, 320K context.
- **Not legal fine-tuning.** No weights were trained. The model knows nothing it did not know before; this only moves the
  quantization error away from legal text (see [Intended use](#intended-use)).

Published by PLI Labs (ProprietaryLegal). The two Engram tables are **not** in this repo. Take them from the 1017 base
repo (see below).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10` (MIT) |
| Built from | [HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark) @ `86f852533ec94d1a537fea41fd9c01d6b0f6eb50` (MIT), the "baseline" below |
| Format | EXL3 trellis, `mul1` codebook ([exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.0) |
| What differs from the baseline | `exl3_moe_plan.json` and the 32 routed-expert part files of layers 2–10, 12, 14–18 and 25. In those layers 1,414 experts changed bitrate (698 up, 716 down) and were re-encoded; all other experts there are bit-identical copies. Every other file is byte-identical to the baseline |
| Routed experts | own bitrate per expert: 2 bpw 39.7 % · 3 bpw 46.1 % · 4 bpw 13.8 % · 5 bpw 0.4 % → **2.75 bpw** mean (baseline 40.8 / 43.8 / 14.9 / 0.4 %) |
| Attention, shared experts / output head / DSpark drafter / vision tower | unchanged from the baseline: EXL3 5 bpw / 6 bpw / dense 5 + experts 2 bpw / BF16 |
| Size | 198.8 GB in this repo, routed-expert bytes identical to the baseline (+0.0000 %); +203.1 GB Engram from the base repo |
| KLD vs native 1017 | legal panel **0.0249** (top-1 97.16 %); general panel **0.0204** (top-1 95.91 %) |
| Context, features | as the baseline: 327,680-token max length, image input, tool calling, DSpark speculative decoding |
| Checksums | `RELEASE_MANIFEST.json`: size and sha256 of every file, plus the two Engram shards |

## Why

The baseline's bit plan spends bits where a sensitivity pass on general text measured the largest output error. We
checked whether that plan is equally faithful on legal text, and it is not.
- **KLD against the native 1017:** 0.0274 on our legal panel versus 0.0201 on the general panel (1.36×). The baseline
  plan's *predicted* cost on legal text is 1.71× its general cost.
- **Expert importance:** measured on legal and on general text, it differs substantially. The median per-layer Spearman
  is 0.61, and the top-10 % sets overlap with Jaccard 0.29.
- **Reliability:** the difference is stable, with a split-half Spearman of 0.97. 1,657 experts are at least 2× more
  important on legal text in both halves of the data, and 6,771 lean general.

So the experts that legal text relies on were under-priced. This repo re-prices them within the same byte budget.

## Quality

KL(native ‖ quant) and top-1 agreement at every scored reply position. Both models run through the same layer-streamed
reference implementation of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) (`p4_kld.py`;
native top-128 support plus a tail bucket), with the official 1017 weights as the reference. Both panels were run on the
same hardware, with the same code, for both builds.

- **General panel:** sudoingX's KLD panel, the one the baseline card uses (199 conversations, 254,846 scored positions).
  Our harness reproduces the baseline's published 0.0201 and all eight per-domain values exactly.
- **Legal panel:** built for this work. It has 210 sequences, about 214K tokens and 66,901 scored positions in six legal
  families.
  - The prompts are fictional matters plus public authority, never client material.
  - The replies were generated by the baseline (thinking off, T = 1.0), so the panel measures how faithfully each
    quant reproduces the native model's distribution on legal-style replies.
  - A seventh, short creative-writing control family (4,179 positions) is reported separately, because it is the
    highest-entropy text in either panel.

### Legal panel

| family | positions | baseline | **this repo** | top-1 baseline → this repo |
|---|---:|---:|---:|---|
| drafting | 26,971 | 0.0379 | **0.0333 (−12.3 %)** | 96.46 → 96.63 % |
| authority reasoning | 16,261 | 0.0269 | **0.0245 (−9.0 %)** | 95.94 → 96.21 % |
| preservation (faithful restatement) | 15,065 | 0.0152 | **0.0157 (+3.4 %)** | 98.37 → 98.49 % |
| tools | 3,608 | 0.0079 | **0.0072 (−8.6 %)** | 99.42 → 99.25 % |
| extraction | 2,674 | 0.0173 | **0.0180 (+3.6 %)** | 98.69 → 98.77 % |
| hallucination control | 2,322 | 0.0285 | **0.0253 (−11.0 %)** | 96.04 → 96.38 % |
| **legal families** | 66,901 | **0.0274** | **0.0249 (−9.0 %)** | 97.00 → 97.16 % |
| (control: short creative writing) | 4,179 | 0.1216 | 0.1174 (−3.5 %) | |

### General panel (sudoingX)

| domain | positions | baseline | **this repo** |
|---|---:|---:|---:|
| code | 165,450 | 0.0125 | **0.0129 (+3.2 %)** |
| chat | 17,976 | 0.0482 | **0.0475 (−1.3 %)** |
| agentic | 16,686 | 0.0183 | **0.0184 (+0.8 %)** |
| long | 16,457 | 0.0258 | **0.0258 (−0.2 %)** |
| knowledge | 14,093 | 0.0382 | **0.0390 (+2.0 %)** |
| math | 10,625 | 0.0152 | **0.0164 (+7.6 %)** |
| multilingual | 8,620 | 0.0601 | **0.0607 (+1.1 %)** |
| writing | 4,939 | 0.0514 | **0.0502 (−2.4 %)** |
| **overall** | 254,846 | **0.0201** | **0.0204 (+1.5 %)** |

Top-1 agreement on the general panel is 95.96 % for the baseline and 95.91 % for this repo.

### Task suite

The suite has 555 paired requests: the same prompts were sent to the baseline and to this repo.
- **Settings:** thinking off, T = 1.0.
- **Scoring:** automatic, rule-based scorers, with no LLM judge.
- **Statistics:** the delta is this repo minus the baseline, with a paired bootstrap 95 % CI (a cluster bootstrap
  by case file for long context).
- **Items:** record-bearing items use a fictional jurisdiction, and the long-context family uses 12 case files of
  150–297K tokens with 10 questions each.

| family | pairs | baseline | this repo | delta | 95 % CI |
|---|---:|---:|---:|---:|---|
| extraction (pin cites, quotes) | 60 | 0.888 | 0.896 | +0.008 | [−0.026, +0.054] |
| preservation | 60 | 0.934 | 0.934 | −0.000 | [−0.010, +0.009] |
| authority | 70 | 0.952 | 0.957 | +0.005 | [−0.031, +0.041] |
| record-conflict drafting | 24 | 0.491 | 0.491 | −0.000 | [−0.074, +0.061] |
| drafting | 49 | 0.990 | 0.991 | +0.002 | [−0.005, +0.009] |
| long context (150–297K tokens) | 120 | 0.823 | 0.827 | +0.004 | [−0.020, +0.030] |
| tools / JSON schema | 80 | 0.998 | 0.994 | −0.004 | [−0.014, +0.004] |
| hallucination | 60 | 0.923 | 0.922 | −0.002 | [−0.048, +0.043] |
| general | 140 | 0.957 | 0.960 | +0.003 | [−0.006, +0.013] |
| **legal macro (8 families)** | | | | **+0.002** | [−0.011, +0.014] |

## Speed and memory

Measured with the same harness for both builds, about 16 hours apart:
- **Setup:** TP2 on 2× GB10, GPU clock capped at 2,150 MHz on both nodes for thermals, no other clients.
- **Requests:** thinking off, T = 1.0, a single stream, and a cold prefill (fresh `cache_salt`).
- **Decode:** median of three runs of about 1.8–2.0K generated tokens each.

| | baseline | **this repo** |
|---|---:|---:|
| decode, prose | 51.3 tok/s | 49.4 tok/s |
| decode, legal drafting | 57.4 tok/s | 55.2 tok/s |
| prefill, cold 8K / 32K | 2,013 / 2,179 tok/s | 2,232 / 2,261 tok/s |
| prefill, cold 150K / 300K | 2,143 / 2,157 tok/s | 2,210 / 2,149 tok/s |
| time to first token, 150K / 300K | 70.1 / 139.1 s | 67.9 / 139.6 s |
| lowest MemAvailable during the bench, node 0 / node 1 | 7.7 / 5.9 GiB | 3.8 / 5.3 GiB |
| highest GPU temperature, node 0 / node 1 | 73 / 69 °C | 77 / 78 °C |

## How it was made

With the quantization pipeline of the sudoingX recipe and exllamav3 1.6.0, on the two Sparks themselves. Each stage had a
gate that stopped the run on failure.

1. **G1, calibration set for the importance profiles.** 1,021,827 tokens:
   - legal 64.2 %: public court opinions, statutes, rules, contracts and filings, plus legal conversations generated by
     the baseline;
   - general 20.3 %, quant/tool 10.2 %, code/multilingual 5.3 %.

   It was checked for overlap with both evaluation panels and the task suite, and the n-gram leakage gate passed.
2. **G2, is there a gap?** Baseline legal KLD is 0.0274 against 0.0201 on the general panel (1.36×). PASS.
3. **Measurement:**
   - pass 0 with separate legal and general routing statistics;
   - per-profile sensitivity;
   - the baseline encodes' measured error per profile;
   - noise-injection group sensitivity. Legal-leaning groups are about 1.5× more sensitive on legal text than their
     routing share predicts.
4. **G3, is the legal profile real?** The split-half Spearman is 0.971, and 1,657 experts lean legal reliably. PASS.
5. **Allocation and G3B.** A legal-weighted allocator (legal weight 0.65) ran in swap mode, so bits only move between
   experts and the total stays exactly constant.
   - Layers: at most 16 layers may change; the allocator chose 2–10, 12, 14–18 and 25.
   - The baseline's floors are kept: super experts at 5 bpw, layers 37–39 at ≥ 3 bpw.
   - Result: 1,414 experts changed. In most changed layers the upgraded experts are clearly more legal-leaning than
     the downgraded ones. The median legal/general importance ratio for upgraded vs downgraded experts is 1.52 vs 0.58
     in layer 14, 1.65 vs 0.68 in layer 18 and 1.56 vs 0.62 in layer 25.
6. **Encode.** Only the 1,414 changed experts were re-encoded, with exllamav3 1.6.0 (`mul1`, pass 1 of the recipe), on
   `cal_v3` (853,106 tokens). That is the calibration behind the baseline's own encodes, rebuilt from the recipe's data
   (its 548,231-token base set byte-identical to the recipe's). The variant is otherwise a byte copy of the baseline.
7. **G4, KLD against native 1017.** Legal −9.0 %, better on 4 of 6 families; general +1.5 %, inside the +3 % limit. PASS.

### Data behind the importance profiles

The data below was used only to measure which experts matter on legal text. The encodes themselves use sudoingX's
public `cal_v3`. Every document is public authority or openly licensed. No confidential, client or case material was
used, and private material was excluded by rule.

| source | what | terms |
|---|---|---|
| CourtListener public storage (Free Law Project), slip-opinion PDFs | 351 opinions: SC family law (182), SC general civil (30), other-state family law (TN, NC, GA, VA: 70), federal (SCOTUS, 4th Cir., D.S.C.: 69) | court opinions are public domain (government edicts doctrine) |
| privately assembled local collection of public SC opinions | 33 SC family-law opinions with no mirrored PDF in CourtListener (printouts of public opinions; third-party annotations and page furniture removed) | public domain |
| SC Code, SC family-court and civil rules, from the same local collection | 172 statute sections, 122 rules (originally from scstatehouse.gov and sccourts.org) | public domain |
| Cornell LII, eCFR | FRCP, FRE (with Advisory Committee notes), 11 sections of the U.S. Code, 45 C.F.R. § 302.56 | U.S. government works |
| [CUAD v1](https://huggingface.co/datasets/theatticusproject/cuad) (The Atticus Project) | 60 contracts, used as text only, no labels | CC BY 4.0; Hendrycks, Burns, Chen, Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review" (2021) |
| RECAP archive (Free Law Project) | 61 federal filings (motions, memoranda, orders, declarations) | public court records, used only as calibration statistics |
| prompts written for this work | 625 legal prompts (tasks over the public documents above and over **fictional** matters with invented parties, records and dockets), 199 general prompts; replies generated by the baseline | — |
| sudoingX conversations and exllamav3 standard text | general / code / multilingual share | Apache-2.0 / MIT (upstream corpora as in the recipe) |

The [Caselaw Access Project](https://case.law) was **not** used, because its robots.txt disallows crawlers. Material
fetched from it during the survey was deleted. The fetchers written for this work did not crawl any site whose
robots.txt disallows automated agents. The SC statutes, the SC court rules and the 33 SC opinions in the table above
were not fetched by this work: they come from a previously assembled local collection.

## Intended use

- **Not legal fine-tuning.** Nothing was trained. This build reproduces the native 1017's output distribution more
  faithfully on legal-style text; the native 1017 is the reference it is measured against.
- **Legal use.** Outputs are drafts. Every output needs review by a licensed attorney before it is relied on, filed or
  sent. Nothing here is legal advice.

## Download

The weights must be in **one folder** on both nodes:

```bash
DIR=~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark   # needs ~420 GB free
hf download ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark --local-dir "$DIR"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir "$DIR"
```

The Engram tables are required, and they must be the **1017** shards:

| File from deepseek-ai/DeepSeek-V4.1-Flash-1017 | Bytes | sha256 |
|---|---|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

`model.safetensors.index.json` already points at them, as in the baseline. If you already have the baseline, only 33
weight and plan files differ: `exl3_moe_plan.json` and `model-exl3-L{02..10,12,14..18,25}-p{0,1}.safetensors`. You can copy the baseline
folder and download just those files.

## Serving

Serving is identical to the baseline. Use the image and launch example from
[github.com/HeNryous/deepseek-v41-1017-dgx-spark](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark)
(`ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`, `serve/serve_example.sh`) and point the model directory at this
folder.
- **JIT kernels:** build them once before the first start, as that README describes.
- **Fabric:** set the RoCE interface and HCA names to match your cabling.
- **Measured setup:** the numbers above were measured with `serve_example.sh` unchanged apart from those names and the
  model directory, i.e. max length 327,680, 4 sequences, KV 3.75 GiB per rank, DSpark k = 5 at 1–2 streams and
  k = 4 at 3–4 streams.

Sampling as recommended by DeepSeek: temperature 1.0, top_p 1.0. Thinking is on by default. Set it only through
`chat_template_kwargs`: `{"thinking": false}` or `{"reasoning_effort": "none" | "low" | "high" | "xhigh" | "max"}`.

## Files

| Files | Content |
|---|---|
| `model-exl3-L00…L39-p0/p1.safetensors` | routed experts, EXL3 (two parts per layer). The parts of layers 2–10, 12, 14–18 and 25 differ from the baseline |
| `model-exl3-dense-*`, `model-exl3-head*`, `model-exl3-mtp*`, `model-000NN-of-00046.safetensors` | identical to the baseline (attention / shared experts, head, drafter, everything kept from the base) |
| `exl3_moe_plan.json` | bitrate of every routed expert (layers 0–39) and of the drafter experts; its `summary` records the allocator settings |
| `model.safetensors.index.json`, `config.json`, `tokenizer*`, `encoding/` | identical to the baseline |
| `RELEASE_MANIFEST.json` | size and sha256 of every file; which ones changed |

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai), for the model and the reference code.
- [HeNryous](https://huggingface.co/HeNryous), for the baseline quant, its serving image and patches; this repo is a
  re-allocation of that work.
- [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41), for the recipe, quantization pipeline, KLD panel and
  calibration data.
- [turboderp](https://github.com/turboderp-org/exllamav3), for EXL3 and exllamav3.
- The Free Law Project (CourtListener, RECAP), Cornell LII and The Atticus Project (CUAD), for public legal text.

## License

MIT, inherited from DeepSeek-V4.1-Flash-1017 and the baseline (see `LICENSE`). Community quantization, provided as is,
with no warranty of any kind.

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
