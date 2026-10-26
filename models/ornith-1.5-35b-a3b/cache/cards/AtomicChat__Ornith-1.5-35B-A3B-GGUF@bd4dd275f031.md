---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: gguf
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - ornith
  - imatrix
  - quantized
  - moe
  - atomic-chat
  - conversational
---

# How to Run Ornith 1.5 35B Locally

<p style="margin-top: 0; margin-bottom: 0;">
  <em>Built from Ornith's original weights with our own importance matrix. The <a href="https://huggingface.co/datasets/AtomicChat/calib-corpora">calibration corpora</a> and <a href="https://huggingface.co/datasets/AtomicChat/Ornith-1.5-35B-A3B-GGUF-metrics">raw eval logs</a> behind this card are public.</em>
</p>

<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_ornith_1_5_35b_a3b&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
</div>

<ul style="margin: 0 0 12px 0;">
  <li>See our quantization analysis below for measurements and instructions.</li>
  <li>You can now run Ornith 1.5 in <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_ornith_1_5_35b_a3b&utm_content=bullet_app">Atomic Chat</a> with toggles for thinking.</li>
</ul>

<hr style="margin: 0 0 16px 0;">

# Ornith-1.5-35B-A3B-GGUF

GGUF builds of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
converted and measured by AtomicChat.

## Which file to take

A3B means 3B parameters are activated per token out of 35B total. Speed follows
the 3B, memory follows the 35B: every expert has to be resident because the
router picks a different eight of the 256 on every token.

| Your VRAM | Take | Size |
|---|---|---|
| 48 GB | `Q8_0` | 36.9 GB |
| 32 GB | `AD-Q6_K` | 29.1 GB |
| 24 GB | `AD-Q5_K-Q4_K` | 22.1 GB |
| 24 GB, room for context | `AD-Q4_K-IQ4_XS` | 20.1 GB |
| 20 GB | `AD-IQ4_XS-IQ3_S` | 17.6 GB |
| 16 GB | `AD-IQ3_S-IQ3_XXS` | 15.5 GB |
| 12 to 16 GB | `AD-IQ3_XXS-IQ2_S` | 13.7 GB |

> [!TIP]
> With a smaller card, keep the experts in system RAM instead of dropping to a
> lower rung. Experts are 93% of the weights but only 3B of them run per token,
> so the arithmetic stays cheap while the bandwidth cost moves to RAM. Pass
> `--cpu-moe` to put every expert on the CPU, or `--n-cpu-moe N` to offload the
> experts of N layers and keep the rest on the GPU. That lets a 12 GB card serve
> a 22 GB build at usable speed.

## Files and measurements

Mean KL divergence against the BF16 reference, held-out `eval_neutral` corpus,
4096 context, llama.cpp `b10505`. Lower is better. Top-1 is the share of tokens
where the quant picks the same next token as the full precision model.

| File | Size | Mean KLD | Top-1 |
|---|---|---|---|
| `Ornith-1.5-35B-A3B-Q8_0.gguf` | 36.90 GB | 0.011620 | 95.64% |
| `Ornith-1.5-35B-A3B-AD-Q6_K.gguf` | 29.10 GB | **0.012961** | **95.31%** |
| `Ornith-1.5-35B-A3B-Q6_K.gguf` | 28.51 GB | 0.016665 | 94.63% |
| `Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf` | 26.25 GB | **0.015793** | **94.85%** |
| `Ornith-1.5-35B-A3B-Q5_K_M.gguf` | 24.73 GB | 0.026870 | 93.31% |
| `Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf` | 22.14 GB | **0.025137** | **93.52%** |
| `Ornith-1.5-35B-A3B-AD-Q5_K-IQ4_XS.gguf` | 21.47 GB | 0.026380 | 93.33% |
| `Ornith-1.5-35B-A3B-Q4_K_M.gguf` | 21.17 GB | 0.047718 | 91.01% |
| `Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf` | 20.13 GB | **0.031512** | **92.71%** |
| `Ornith-1.5-35B-A3B-IQ4_XS.gguf` | 18.73 GB | 0.054181 | 90.36% |
| `Ornith-1.5-35B-A3B-AD-IQ4_XS-IQ3_S.gguf` | 17.61 GB | 0.057726 | 90.12% |
| `Ornith-1.5-35B-A3B-AD-IQ3_S-IQ3_XXS.gguf` | 15.51 GB | 0.087334 | 88.07% |
| `Ornith-1.5-35B-A3B-AD-IQ3_XXS-IQ2_S.gguf` | 13.67 GB | 0.141624 | 84.70% |

`AD` marks a per tensor bit layout tuned for this model. Files without the
prefix are stock llama.cpp presets built with the same importance matrix, so the
comparison isolates the layout and nothing else. Each `AD` label reads as
`down_type-gate_up_type`, which is what is actually inside the file.

![image_2026-08-19_20-51-23](https://cdn-uploads.huggingface.co/production/uploads/6a54dea4f19f5386700504da/ynDjcsiNw_rRMLOdwMraV.png)


**Two rungs beat a stock preset on both axes at the same time.**
`AD-Q5_K-Q4_K` is 2.6 GB smaller than stock `Q5_K_M` and more accurate.
`AD-Q6_K-Q5_K` is 2.3 GB smaller than stock `Q6_K` and more accurate. Neither
needs an interpolated curve to read off the chart.

**At four bits the gap is largest.** `AD-Q4_K-IQ4_XS` is a gigabyte smaller than
stock `Q4_K_M` and cuts divergence by 34%.

**`AD-Q6_K` is within 12% of `Q8_0` at almost eight gigabytes less.** If you
were reaching for Q8_0 out of caution, this is the file to take instead.

## What we learned building this

The layout was not carried over from another model. We built five candidate
layouts at the four bit size class and three more at six bits, measured all of
them against the same reference, and kept the winners. Every log is in the
metrics repo, including the ones that lost.

**The winning layout is different at four bits and at six.** At four bits,
lifting `ffn_down_exps` one step and paying for it out of `ffn_gate_exps` and
`ffn_up_exps` wins clearly. At six bits the same trade loses to a flat layout:
29.10 GB at 0.012961 for flat against 28.85 GB at 0.015291 for the asymmetric
one. The reason is that a coarse base type leaves a lot of headroom to
redistribute, while a fine one does not: lifting a small group buys nothing once
it is already near lossless, and the large group you took the bits from still
pays full price. The same inversion showed up independently on the dense
Ornith-1.5-9B, which makes it a property of quantization rather than of one
model.

**`ffn_down_exps` is the sensitive one, and its shape says why.** Its rows are
512 wide against 2048 for gate and up. A k-quant stores one scale per 256 wide
superblock, so a down projection row holds two superblocks while a gate row
holds eight. Less data per scale means a worse fit, so the extra bit pays off
there and nowhere else.

**Cutting the non-expert weights does not pay.** Experts are 93% of this model,
so the attention, embedding and output matrices look like free real estate. We
tested it: dropping all of them from q8_0 to q6_k saved 0.55 GB and cost 6.6% of
the divergence, while spending the same 0.55 GB on the experts buys about 5.8%.
The trade is a wash at best. Everything outside the experts stays at q8_0 in
every file here, and the router stays in f32.

**A sparse MoE costs more bits than a dense model of similar quality.** `Q8_0`
here sits at 0.011620 while `Q8_0` on the dense Ornith-1.5-9B reaches 0.002249,
five times lower, despite both having the same headroom. Short expert rows are
part of it. The router is the other part: it turns a small weight perturbation
into a hard switch to a different expert, so the error stops being smooth.

**The measurements reproduce.** The `iq4_xs` down with `iq3_s` gate and up
recipe was built and measured twice, on two different rented machines, and
returned mean KLD 0.057726 both times.

## Quick start

```bash
llama-server -m Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf \
  -ngl 99 -c 8192 -fa on --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --port 8080
```

If the file does not fit your card, keep the experts in RAM rather than dropping
a rung:

```bash
llama-server -m Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf \
  -ngl 99 --cpu-moe -c 8192 -fa on --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --port 8080
```

With vision, add the projector:

```bash
llama-mtmd-cli -m Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf \
  --mmproj mmproj-Ornith-1.5-35B-A3B-F16.gguf \
  -ngl 99 -c 8192 --temp 0.6 --image-min-tokens 1024 \
  --image your-image.jpg \
  -p "Describe this image."
```

> [!IMPORTANT]
> Pass `--image-min-tokens 1024`. Without it, dense images such as charts and
> screenshots get too few visual tokens, and the model answers from what it
> already knows rather than from what it was shown. llama.cpp prints a warning
> about this at load time.

> [!IMPORTANT]
> Upstream sampling defaults are not what llama.cpp falls back to. Set them
> yourself: `--temp 0.6 --top-p 0.95 --top-k 20`. Use `--temp 1.0` to reproduce
> the benchmark setup from the upstream model card.

Lines reading `find_slot: non-consecutive token position` are normal. That is how
this family numbers image patches, and they appear in successful runs too.

## Speculative decoding

Unlike the 9B, this checkpoint does ship a multi token prediction head, and we
publish it as a separate draft file:

```bash
llama-server -m Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS.gguf \
  --model-draft mtp-Ornith-1.5-35B-A3B-Q8_0.gguf \
  -ngl 99 -ngld 99 -c 8192 -fa on --jinja -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 6
```

Note that `-np > 1` and `--mmproj` do not currently work together with MTP in
llama.cpp, so pick either speculation or vision, not both.

## Architecture

Read from the converted file, not from the config.

| | |
|---|---|
| Total parameters | 34.7B in the main file, plus a 1.9B MTP head shipped separately |
| Active per token | about 3B |
| Layers | 40 |
| Experts | 256 routed, 8 active, plus one shared expert per layer |
| Attention | hybrid: full attention on layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, linear attention on the other 30 |
| Hidden size | 2048 |
| Expert FFN size | 512 |
| Vocabulary | 248320 |
| Context | 262144 |
| Architecture string | `qwen35moe` |

Where the weights actually sit:

| Group | Share |
|---|---|
| `ffn_down_exps` | 31.0% |
| `ffn_gate_exps` | 31.0% |
| `ffn_up_exps` | 31.0% |
| `attn_q` (includes the linear attention input projection) | 1.9% |
| `output` | 1.5% |
| `token_embd` | 1.5% |
| everything else | 2.1% |

> [!WARNING]
> If you convert this model yourself, pass `--no-nextn` to
> `convert_hf_to_gguf.py` for the main file and `--mtp` on a second run for the
> draft. Without the flag the MTP block lands inside the target file, where a
> plain forward pass never executes it, and low bit quantization aborts on it
> because the importance matrix has no statistics for a block that never runs.

## Calibration

The importance matrix was computed on the BF16 weights, not on a quantized
stand-in, over 4,958,490 tokens in 9,686 chunks, fanned out across 15 GPUs and
merged. The tokenizer is byte identical to Ornith-1.5-9B, so the same corpus
build applies to both models.

Corpus composition: agentic tool traces 24.7%, code 17.8%, reasoning 14.8%,
multilingual 13.8%, long context 11.9%, vocabulary sweep 9.9%, structured data
3.9%, graphics 3.0%. Vocabulary coverage is 99.5%.

For a sparse MoE the coverage question is sharper than for a dense model: a
routed expert that never fires during calibration gets quantized blind, and at
two bits that produces a file which loads and answers with nothing to warn you.
At 8 of 256 experts per token, each expert sees roughly 155,000 tokens of this
corpus, and the merged statistics show no expert tensor without data.

## Reproducing our numbers

- The BF16 file in this repo is the reference every measurement is taken against.
- The calibration corpus is [AtomicChat/calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora), build `ornith-1.5-9b`.
- [Ornith-1.5-35B-A3B-GGUF-metrics](https://huggingface.co/datasets/AtomicChat/Ornith-1.5-35B-A3B-GGUF-metrics)
  holds the importance matrix, every raw KL divergence log including the losing
  ablations, and the reference logits, so you can measure your own build against
  the same starting point instead of taking our word for it.

```bash
llama-perplexity -m your-quant.gguf -f eval_neutral.txt \
  --kl-divergence-base base-neutral.kld --kl-divergence -c 4096 -ngl 99
```

The corpus and the context have to match ours or the numbers are not comparable.

## About the model

Ornith-1.5 is trained with a loop that generates its own tasks, builds its own
scaffolds, and rewards all three stages together rather than optimizing a policy
against a fixed human written harness. For the method and the benchmark results,
see the [upstream model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
and the [Ornith blog](https://ornith.ai). Those benchmark numbers are the
authors' own and we have not independently reproduced them.

## Changes

- **26.10.2026:** the vision projector (`mmproj-Ornith-1.5-35B-A3B-F16.gguf`, `-BF16.gguf`) and the MTP draft
  (`mtp-Ornith-1.5-35B-A3B-Q8_0.gguf`) described above are uploaded. Both are converted from the vendor's current
  weights (revision `10fbf86f`). That revision re-uploaded only the MTP head on 23.08; every other tensor is
  unchanged, so the quants here are current. With `AD-IQ4_XS-IQ3_S` on a coding prompt, 64% of the drafted
  tokens were accepted (llama.cpp `c6fb3b5`).
