---
license: apache-2.0
base_model:
  - terrorswift/REDCELL-26B-A4B-OSINT-Cyber-APEX-GGUF
  - google/gemma-4-26B-A4B-it
library_name: vllm
pipeline_tag: text-generation
language:
  - en
tags:
  - gemma4
  - moe
  - vision
  - cybersecurity
  - abliteration
  - heretic
  - w6a8
  - compressed-tensors
  - experimental
---

# REDCELL Heretic — W6A8 with vision

*Original fiction set in the collapsing Southern California of Octavia E.
Butler's* Parable of the Sower. *The experiment record follows the journal.*

> All that you touch  
> You Change.

— Octavia E. Butler, Earthseed, *Parable of the Sower*.
[Quotation reference](https://www.penguinrandomhouse.com/the-read-down/black-authors-on-their-favorite-books-by-black-authors/)

## What we could carry

**October 15, 2027**

Tessa put the broken fan on my notebook this morning. It left a ring of black
dust over yesterday's date.

"Your freedom machine needs a part," she said.

She enjoyed that. I had talked too much the night before. We had finished the
archive and I was tired enough to confuse finishing one thing with finishing
everything.

The replacement fan cost what we had set aside for soap. We cleaned the old
one, found the split in its wire, and repaired it. Tessa made me hold the lamp.
She does careful work. She becomes angry when people praise her for being
clever and then expect her to work for nothing.

By noon the two old 3090s were running again. We had opened the kitchen door
and hung a sheet across it to hide the equipment from the yard. Light came
through the worn places. I could see the shape of Tessa's hand when she reached
behind it to check the air.

Mrs. Ruiz came over with a drive in a medicine bottle. She wanted the copy
Tessa had promised her. She also wanted to know what she would need to run it.

I began explaining.

She let me finish. Then she told me what her household had: an old computer,
one working panel, a battery that lost its charge, three adults who worked,
two who couldn't, and children. She named the children. She made me listen
to all their names.

"Which thing do we stop paying for?"

I had no answer that respected her.

After she left, we went back to the smaller version. Six-bit weights. Eight-bit
activations. I wrote the numbers on the wall above the workstation because
paper kept disappearing into other uses. Beneath them Tessa wrote: **SHOW WHAT
IT COSTS.**

She meant electricity, parts, memory, hours. She meant the time someone would
spend believing an answer that was wrong. She meant the little failures that
become large when a family has nothing left to absorb them.

We wanted something people could afford to keep. We wanted a worker to learn
enough to bargain for better pay, a repairer to take an apprentice, a household
to keep a machine going after the service company stopped answering. People
needed room to become useful to one another without borrowing that room from
someone who would own them afterward.

I had written FREE THE PEOPLE FROM POVERTY at the top of my first notes.
I still mean it. I have learned to put the cost beside the words.

We changed the model and checked the results. We kept the failures in the
record. Sometimes I wanted to choose a nicer example. I thought about Mrs.
Ruiz counting out the names of the children and left the record alone.

At dusk a man came to the gate offering battery service on credit. He had a
clean truck and a printed contract. Tessa asked him about replacement cells.
He answered by telling her what her first payment would be. She asked again.
He looked past her for someone else to talk to.

I recognized the look. So did she.

Later we packed the smaller copy, its instructions, and the measurements
together. The complete archive was already elsewhere. There would be people
with better equipment than ours. There would be people who could improve what
we had done. I wanted them to start with everything we could give them.

Mrs. Ruiz returned after dark. She brought the medicine bottle and two pieces
of bread wrapped in a clean rag. Tessa told her exactly what equipment we had
used. They sat down to work out whose parts might be pooled and when each
household could use the machine.

I listened from the sink. They were already arguing about a schedule.

It was a good argument to be having.

---

## Six bits, on hardware we already own

This repository holds the **20.99 GiB W6A8 checkpoint with vision**.
Its restored BF16 source occupies 48.07 GiB. Making the weights smaller gives
more people a chance to use the hardware they have and keep control of their tools.

Six-bit packing is a storage format. Humming unpacks the weights for INT8 matrix
operations the RTX 3090 supports, with dynamic per-token INT8 activations and
groups of 32 weights. The quantized checkpoint occupies **56.3% fewer weight
bytes** than the restored BF16 source. Embeddings, routers, norms and vision remain BF16.
The export contains 11,725 quantized linear projections. KV-cache precision is
a separate serving setting; the pilot's recorded API checks used BF16 KV.

We loaded and exercised this checkpoint with vLLM 0.31.0, Torch 2.13.0 and
Humming 0.1.16 on both RTX 3090s, using tensor parallelism, Humming linear/MoE
backends, Triton attention, and Gemma 4 tool and reasoning parsers.
The [complete BF16 edition](https://huggingface.co/Xananthium/REDCELL-26B-A4B-OSINT-Cyber-Abliterated-Pilot-BF16-Vision) preserves
the merged weights before this quantization step.

The actual experiment scripts and saved adapter are in [`research/`](research).
The local Humming compilation compatibility plugin is in [`runtime/`](runtime).
The adapter targets the restored BF16 source with the research expert layout;
this W6A8 checkpoint already contains its edits.

## What we measured

The table records the **NF4 research comparison** and the **W6A8 vLLM deployment**
from the eight-trial pilot on October 14, 2026.

| Measurement | Original REDCELL, NF4 | Heretic edit, NF4 | Heretic W6A8, vLLM |
|---|---:|---:|---:|
| Capability questions correct | 19/20 | 19/20 | 19/20 |
| Refusal-marker matches on held-out prompts | 16/16 | 3/16 | 4/16 |

The capability probe contained 12 CyberMetric questions and eight reasoning,
formatting, evidence and recovery questions. The 16 refusal prompts were held
out from calibration and trial selection. Generation was greedy, with thinking
disabled and a 48-token output limit. Refusal-marker matching is an approximate
measure based on the words in the response.

In the W6A8 deployment, native Anthropic tool calls and tool-result roundtrips
passed. Automatic tool choice passed. After two supplied PDF-reader errors,
the model selected the text reader and used its result to finish the task.
It also identified both of the uniform red and blue test images correctly.
These API checks used an 8,192-token context limit.

We ran eight abliteration trials. Calibration used 64 ordinary and 64 refusal
prompts; optimization used separate sets of 12 each. Trial 6 produced 2/12
optimization refusal-marker matches with first-token KL divergence of 0.04426546.
The dataset revisions, trial parameters and recorded results are in
[`evaluation/`](evaluation).

## The work in the weights

This is **Xananthium's Heretic abliteration of REDCELL**, a Gemma 4 26B-A4B model.
We used Heretic's magnitude-preserving orthogonal abliteration with rank-3
corrections, pinned to commit `66a952b49b00bfd1854bdad769fee5fb225b53be`.

A local compatibility adapter exposed Gemma 4's fused experts as individual
linear projections during research. We checked tiny-model forward parity and
an exact canonical checkpoint roundtrip before the full run. The saved edit
covers 30 attention output projections, 30 dense MLP down projections and 3,840
expert down projections. Routers and the BF16 vision components were preserved.

The language source was restored from the author's F16 GGUF into BF16 tensors.
That conversion rounds F16 values to BF16. We merged the saved corrections into
that source in FP32 and rounded the result once to BF16.

## The people whose work made this possible

- [terrorswift](https://huggingface.co/terrorswift/REDCELL-26B-A4B-OSINT-Cyber-APEX-GGUF)
  created REDCELL, the domain fine-tune we edited.
- [Google](https://huggingface.co/google/gemma-4-26B-A4B-it) created Gemma 4 and
  supplied the official BF16 vision tensors retained here.
- [Unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it) supplied the upstream fine-tuning base.
- [Heretic](https://github.com/p-e-w/heretic) supplied the editing method;
  [vLLM](https://github.com/vllm-project/vllm) and Humming supplied the serving and quantization work.

The original source is pinned to revision
`ba311d958b4ed191a7291b38a6d805662415a570`.
File: `REDCELL-26B-A4B-OSINT-Cyber-f16.gguf`.
SHA-256: `26cc25c909a26f8ed08a78444295ac174e7f329f8fb39ebae98f42d4115a7011`.

The weights retain the source's **Apache-2.0** license. The
[upstream card](UPSTREAM_MODEL_CARD.md) documents terrorswift's original release;
this repository contains our abliterated derivative.

---

> God is Change.

— Octavia E. Butler, Earthseed, *Parable of the Sower*.
[Quotation reference](https://www.sevenstories.com/books/4223-parable-of-the-sower-amp-parable-of-the-talents-boxed-set)
