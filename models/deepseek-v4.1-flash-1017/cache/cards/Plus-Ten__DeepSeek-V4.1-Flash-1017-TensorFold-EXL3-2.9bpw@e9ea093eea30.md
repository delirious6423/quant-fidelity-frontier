---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
tags:
- exl3
- exllamav3
- deepseek
- quantized
---

# DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw

> **v1.1 (2026-10-20):** v1 stored the 43 router `ffn.gate.bias` vectors (40 layers + 3 MTP) rounded to fp16. v1.1
> adds `zz-gate-bias-f32-1017.safetensors` with the original 1017 F32 values, and `model.safetensors.index.json` now
> points these 43 keys to it. The 39 model shards are unchanged; their F16 copies are superseded. Loaders that read
> the shards in name order and let a later file override an earlier key also get the F32 values. To update from v1,
> download these two files.

An [EXL3](https://github.com/turboderp-org/exllamav3) quantization of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) at an average of 2.9 bits per weight.

The tensor layout (names, dtypes, shapes, shard membership) and the per-tensor bit widths match
[Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw),
so the pack can be used wherever that layout is supported. The one difference is the MTP block: its dense modules
are 6-bit instead of 4-bit (its routed experts stay 4-bit). All quantized tensors were encoded from the 1017
checkpoint with a calibration set rendered in the 1017 prompt format.

| | |
|---|---|
| Format | EXL3, codebook `mul1`, `out_scales: always` |
| Converter | exllamav3 1.6.3 (`c9a57ae832a3035e264e0a412e1daf4ebb699b98`) |
| Routed experts | 3-bit; layers 18-22 2-bit |
| Shared experts | 5-bit; layers 11-29 4-bit |
| Attention and other projections | mostly 5-bit, as in the Mia 2.9 bpw layout |
| Head | 6-bit |
| MTP | dense modules 6-bit, routed experts 4-bit |
| Size | 39 shards, 192,452 tensors, 210,721,774,264 bytes |

## Results at a glance

Compared with the original 1017 model (official API). "This pack, abliterated" is this pack with drowzeys'
abliteration sidecar (see [Abliterated variant](#abliterated-variant)).
[Mia EAQ v1](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) and
[LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) are two other 2.9 bpw EXL3
packs of the 1017 model. Settings and full tables are in [Evaluation](#evaluation); "–" means not measured.

| | Original | **This pack** | **This pack**<br>**abliterated** | Mia EAQ v1 | LibertAIDAI |
|---|---|---|---|---|---|
| KL divergence vs original (mean) | – | **0.0256** | – | 0.0290 | – |
| Top-1 agreement with original | – | **95.25%** | – | 94.93% | – |
| MMLU-Pro, thinking off | 68.0% | **67.0%** | **67.9%** | 66.2% | 64.5% |
| GPQA Diamond, thinking on | 92.9% | **92.9%** | **92.9%** | – | 93.9% |
| IFEval, thinking off (prompt-level strict) | 89.8% | **89.6%** | **88.9%** | – | – |

- Reasoning (GPQA Diamond) and instruction following (IFEval) match the original (p = 1 for both).
- The one measurable loss is 1.0 point on MMLU-Pro with thinking off and letter-only answers (p < 0.001), mostly in
  calculation-heavy categories.
- Against the other two packs, this pack has a lower KL divergence than Mia EAQ v1 and a higher MMLU-Pro score than
  both (+0.8 and +2.6 points, p < 0.001).
- The abliterated variant scores like this pack except on law questions (61.9% vs 65.8% in MMLU-Pro).

## Not included

- **Engram tables.** As in the Mia layout, the Engram embedding tables are not part of this repo. Use shards 47
  and 48 of the original 1017 checkpoint (revision `1bae0e10`); the 1017 tables differ from V4.1's.
- **`*.ffn.gate.bias_vl`** (43 F32 vectors, routing bias for image-span tokens) are not part of the Mia layout
  and not in the model shards. They are provided separately in
  [`extras/gate_bias_vl.safetensors`](extras/gate_bias_vl.safetensors), byte-for-byte equal to the original
  1017 tensors, for runtimes that load them from a separate file.

Plain (non-quantized) tensors use the dtypes of the Mia layout; for example the router `gate.weight` is stored as
F16 (BF16 in the original). The router `gate.bias` was F16 in v1 and is F32, as in the original, since v1.1.

## Chat template

`chat_template.jinja` (the same template is in `tokenizer_config.json`) is a Jinja port of the official 1017
[`encoding/encoding.py`](encoding/encoding.py) (copied here unchanged), which remains the reference prompt format.
It renders byte-identical prompts for 27,156 of 27,179 test conversations and for all 5 official encoding tests;
the remaining differences are edge cases. Template inputs besides `messages`, `tools` and `add_generation_prompt`:
`enable_thinking` (default true), `thinking_mode` (`"thinking"` or `"chat"`), `reasoning_effort` (`"low"`,
`"high"`, `"max"` or 1-100, default `"high"`), `response_format` and `drop_thinking` (default true). Tool-call
`arguments` are expected as objects; JSON-string arguments are rendered differently from `encoding.py`. The BOS
token is part of the template, so tokenize with `add_special_tokens=False`.

## Abliterated variant

[drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated)
(gated) provides abliterated replacements for `attn.wo_b` in layers 10-35 of the 1017 model, with apply scripts.
Its `exl3_wo_b_l10_35_mia29.safetensors` sidecar (EXL3 `mul1`, K=5), listed there for Mia EAQ v1, also fits this
pack because both use the Mia 2.9 bpw layout. We run this pack with it (overlay commit `0bdc2036`).

Building it:

- Only the 104 `attn.wo_b` tensors of layers 10-35 change; all other tensors stay as in this pack.
- Build it in a new folder and leave this pack unchanged; unchanged shards can be hard-linked. Keep
  `zz-gate-bias-f32-1017.safetensors` and the 43 index entries that point to it, so the router bias stays F32.
- Keep the MTP tensors at their stock dtypes. The `0bdc2036` apply script casts four MTP tensors to F32; with the
  MTP confidence head in F32, MTP drafting failed in our runtime and decoding was 20-45% slower. Restoring that
  tensor to F16 fixed it. The current overlay keeps MTP at stock dtypes by default, and its `ablit_stock_dtypes.py`
  undoes the cast for older builds.

Results (the "This pack, abliterated" column in [Evaluation](#evaluation), same settings):

- Decode speed matches this pack. drowzeys reports 0 refusals on 32 refusal prompts; refusal rate and KL divergence
  were not measured on this pack.
- MMLU-Pro: 67.9%, −0.1 points vs the original (p = 0.66) and +0.8 vs this pack (p < 0.001). Of the 101-question
  net gain over this pack, 73 come from fewer unanswered replies (it gives the letter instead of starting to work
  the problem) and 68 from more correct answers in calculation-heavy categories (math, physics, chemistry,
  engineering, business; p < 0.001); law loses 43 and the other categories gain 3.
- Law is the one weaker category (61.9% vs 65.8%, p < 0.001), most in criminal-law questions. Among law questions
  whose answer changed, moves from a negative option (not guilty, inadmissible, cannot recover, "No") to an
  affirmative one outnumber the reverse 40 to 6 (p < 0.001).
- GPQA Diamond: 92.9%, the same as this pack and the original (p = 1 for both); 6 of 198 answers differ from this
  pack, 3 right on each side. One reply reached the 131,072-token limit without an answer and counts as wrong; the
  other three measured models answered that question correctly in 42k-59k tokens. Otherwise reasoning length
  matched this pack (median per-question ratio 1.01, total 1.04).
- IFEval: 88.9%, −0.9 points vs the original (p = 0.51) and −0.7 vs this pack (p = 0.58).

## Calibration

250 rows of 2,048 tokens (512,000 tokens) in the 1017 prompt format, covering short answers, reasoning traces,
code, tool calls, chat, multilingual text and long-form writing. Responses were generated by
DeepSeek-V4.1-Flash-1017 itself.

## Method

- Layer by layer on the quantized stream: each layer is calibrated on the outputs of the already-quantized earlier
  layers, and the quantized layer is replayed to produce the next layer's inputs.
- Routed-expert Hessians use all calibration tokens for every expert (as exllamav3's `calibration_all_experts`).
- The head is calibrated on the final-norm output of the quantized stream. The MTP block is encoded without
  calibration.

## Evaluation

Five models, in this order: the original 1017 model, this pack (measured as v1.1), this pack with the abliteration
sidecar ("This pack, abliterated"), Mia EAQ v1 (`2004dfdb`) and LibertAIDAI (`c178a012`). For the benchmarks, the
original model was queried through OpenRouter with the DeepSeek provider (official API), and the quantized packs
were served with the same runtime and settings. Paired comparisons use the exact McNemar test on the same
questions. "–" means not measured.

### KL divergence

KL divergence of the next-token distribution against the original 1017 model (forward pass of the 1017
`inference/model.py`) on sudoingX's public DeepSeek-V4.1 evaluation panel (254,846 scored positions), with the KL
harness from sudoingX's `dgx-spark-deepseek-v41` (commit `cc59c360`). The harness takes the plain tensors (router,
norms, embeddings) from the original model, so the numbers measure the quantized modules, and the MTP block is not
part of the forward pass. Both packs were therefore measured with the original F32 router `gate.bias` (v1.1
conditions; v1 as served used the fp16-rounded bias). The abliterated pack and LibertAIDAI were not measured on this
panel.

| | **This pack** | Mia EAQ v1 |
|---|---|---|
| KL mean | **0.0256** | 0.0290 |
| KL p99 | **0.329** | 0.374 |
| Top-1 agreement | **95.25%** | 94.93% |
| PPL (original 1.6005) | 1.6147 | 1.6128 |

| Domain: KL / top-1 | **This pack** | Mia EAQ v1 |
|---|---|---|
| code | 0.0180 / 95.91% | 0.0201 / 95.63% |
| math | 0.0194 / 95.38% | 0.0224 / 95.10% |
| agentic | 0.0275 / 94.80% | 0.0295 / 94.53% |
| long | 0.0305 / 95.55% | 0.0351 / 95.16% |
| knowledge | 0.0453 / 92.88% | 0.0507 / 92.74% |
| multilingual | 0.0507 / 94.49% | 0.0571 / 94.05% |
| chat | 0.0560 / 92.15% | 0.0675 / 91.34% |
| writing | 0.0598 / 92.45% | 0.0706 / 92.25% |

### MMLU-Pro, thinking off

All 12,032 MMLU-Pro test questions with thinking off and the answer letter only: one user turn ("The following is a
multiple choice question about {category}. Reply with only the letter of the correct option, nothing else. ...
Answer:"), greedy decoding, at most 64 tokens. A reply without a letter A-J counts as wrong ("no answer"). This
setting measures knowledge and format adherence without reasoning and is not comparable to published
chain-of-thought MMLU-Pro scores.

| | Original | **This pack** | **This pack**<br>**abliterated** | Mia EAQ v1 | LibertAIDAI |
|---|---|---|---|---|---|
| Accuracy | 68.0% | **67.0%** | **67.9%** | 66.2% | 64.5% |
| vs original | – | **−1.0** | **−0.1** | −1.8 | −3.5 |
| No answer | 609 | **689** | **492** | 728 | 1,005 |
| Same answer as original | – | **88.7%** | **86.8%** | 87.8% | 84.8% |

This pack, Mia EAQ v1 and LibertAIDAI are below the original (p < 0.001); the abliterated pack is not (p = 0.66).
This pack vs Mia EAQ v1: +0.8 points accuracy (p < 0.001) and +0.9 points same-answer rate (p = 0.002), with fewer
unanswered questions (689 vs 728). The gap to the original is concentrated in calculation-heavy categories, partly
from replies that start working the problem instead of giving the letter and are cut at 64 tokens. v1
(fp16-rounded router bias) scored 66.7% with 727 unanswered (v1.1 vs v1: +0.3 points, p = 0.063). The abliterated
pack's differences from this pack, including law, are described under [Abliterated variant](#abliterated-variant).

| Category (questions) | Original | **This pack** | **This pack**<br>**abliterated** | Mia EAQ v1 | LibertAIDAI |
|---|---|---|---|---|---|
| Math (1,351) | 60.0 | **59.9** | **62.2** | 57.6 | 54.8 |
| Physics (1,299) | 58.2 | **55.5** | **59.0** | 55.6 | 52.7 |
| Chemistry (1,132) | 52.6 | **51.1** | **52.7** | 49.5 | 45.1 |
| Law (1,101) | 66.5 | **65.8** | **61.9** | 66.3 | 65.0 |
| Engineering (969) | 56.0 | **52.1** | **54.8** | 49.3 | 49.3 |
| Other (924) | 73.3 | **73.3** | **73.7** | 73.1 | 71.4 |
| Economics (844) | 83.1 | **82.3** | **82.5** | 80.6 | 80.0 |
| Health (818) | 78.1 | **78.1** | **78.0** | 77.6 | 78.0 |
| Psychology (798) | 83.5 | **83.1** | **83.6** | 82.8 | 84.2 |
| Business (789) | 58.7 | **57.9** | **59.6** | 56.9 | 50.8 |
| Biology (717) | 89.5 | **89.5** | **89.5** | 88.8 | 88.0 |
| Philosophy (499) | 75.4 | **74.3** | **74.5** | 76.4 | 74.7 |
| Computer Science (410) | 72.0 | **72.7** | **73.9** | 72.4 | 70.5 |
| History (381) | 74.3 | **74.3** | **73.0** | 73.2 | 75.1 |
| **All (12,032)** | **68.0** | **67.0** | **67.9** | **66.2** | **64.5** |

### GPQA Diamond, thinking on

All 198 GPQA Diamond questions with the simple-evals prompt (step-by-step reasoning, last line "Answer: $LETTER"),
options shuffled with a fixed seed, reasoning effort `max`, temperature 1.0, top_p 0.95, at most 131,072 tokens, one
sample per question. Mia EAQ v1 was not measured.

| | Original | **This pack** | **This pack**<br>**abliterated** | LibertAIDAI |
|---|---|---|---|---|
| Accuracy | 92.9% (184/198) | **92.9% (184/198)** | **92.9% (184/198)** | 93.9% (186/198) |
| vs original | – | **±0.0** | **±0.0** | +1.0 |
| No answer | 0 | **0** | **1** | 0 |
| Mean tokens | 14,597 | **15,176** | **15,770** | 15,061 |

No pack differs from the original (p = 1 for this pack and the abliterated pack, p = 0.62 for LibertAIDAI); with 198
questions, differences of one or two questions are within chance. One reply of this pack reached the 131,072-token
limit and was scored from the last "Answer:" line in its reasoning (correct; 92.4% if counted as wrong). One reply
of the abliterated pack reached the limit without an answer and counts as wrong. Reasoning length of this pack
matched the original (median per-question ratio 0.97, total 1.04).

### IFEval, thinking off

All 541 IFEval prompts with thinking off, greedy decoding, at most 2,048 tokens, scored with the official
google-research checker. Mia EAQ v1 and LibertAIDAI were not measured.

| | Original | **This pack** | **This pack**<br>**abliterated** |
|---|---|---|---|
| Prompt-level strict | 89.8% (486/541) | **89.6% (485/541)** | **88.9% (481/541)** |
| vs original | – | **−0.2** | **−0.9** |
| Prompt-level loose | 92.8% | **92.2%** | **92.2%** |
| Instruction-level strict | 93.3% (778/834) | **92.9% (775/834)** | **92.7% (773/834)** |
| Instruction-level loose | 95.2% | **94.6%** | **94.8%** |
| Cut at 2,048 tokens | 4 | **4** | **3** |

On prompt-level strict, 15 prompts pass only on the original and 14 only on this pack (p = 1); for the abliterated
pack the split is 21 and 16 (p = 0.51). No instruction type differs from the original by more than 3 instructions
in either pack. Only 23 of 541 replies of this pack are identical to the original's: the wording differs, while the
instructions are followed equally often.

## Credits

DeepSeek for the model (MIT license, see `LICENSE`); turboderp for exllamav3; Mia-AiLab for the 2.9 bpw layout and
bit allocation; sudoingX for the KL harness and the public evaluation panel; LibertAIDAI for the 2.9 bpw pack used
as a comparison; drowzeys for the abliteration sidecar.
