---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- exl3
base_model_relation: quantized
---

# EXL3 quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

These quantizations can be used with [exllamav3](https://github.com/turboderp-org/exllamav3) or vLLM via [vllm-exl3-plugin](https://github.com/yeasah/vllm-exl3-plugin).

The token embedding is quantized too, which needs exllamav3 1.6.4 or later.

They are calibrated broadly, on what models read and write while working (documents and tool output in conversations, code and diffs, the model's own replies) rather than on web text alone, and measured with agentic and coding work as the primary use case.

Bits are allocated per tensor, from the model's measured sensitivity on its own replies, at the same total size as uniform bits.

Earlier revisions of this repository are superseded. The current ones store the token embedding quantized (1.6-1.9 GiB smaller on disk at every rate, at no measurable cost in quality), round without the output-side factor the previous ones used, and add a 2.50 bpw rung. They remain in each branch's history.

Actual consumed VRAM may be significantly lower than the total checkpoint size, depending on configuration: see the VRAM column below.

---

# Which size?

Each bitrate is available as a separate revision of this repository. When planning available VRAM for kv space, refer to the VRAM column.

![quality vs size](./quality_vs_size.png)

The dashed line marks [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), scored on the same evals as these quants: a familiar point to read them against. It is scored with its weights dequantized under bf16 compute, so it measures the checkpoint's weights, not a deployment (the activation quantization it is served with is not emulated).

| BPW | Size on disk | VRAM (vLLM) | Excess KLD | Δ perplexity |
|---|---|---|---|---|
|[2.50](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/2.50bpw-H5)|9.54 GiB|8.48 GiB|0.0305|+3.17%|
|[3.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/3.00bpw-H5)|10.96 GiB|9.90 GiB|0.0136|+1.42%|
|[3.50](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/3.50bpw-H5)|12.37 GiB|11.31 GiB|0.0074|+0.78%|
|[4.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/4.00bpw)|13.94 GiB|12.88 GiB|0.0037|+0.37%|
|[5.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/5.00bpw)|16.92 GiB|15.86 GiB|0.0009|+0.12%|
|[6.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/6.00bpw)|19.75 GiB|18.69 GiB|0.0002|+0.04%|
|(base)|51.75 GiB||||

- BPW: Average bits per weight across tensors (not including input embeddings, encoders, MTP draft layers, or output head)
- Size on disk: Total aggregate size of tensors in checkpoint, in GiB
- VRAM (vLLM): Total VRAM consumption as typically used in vLLM, in GiB (the quantized embedding as stored, offloaded/omitted encoders, no MTP draft layers)
- Excess KLD: how far each quant's next-token distribution is from the unquantized model's, minus the unquantized model's own rounding noise, so 0 would be indistinguishable from the original (lower is better)
- Δ perplexity: change in perplexity on the same text, same weights (lower is better)




# Against other quants of this model

*This section appears because other quantizations of this model exist ([turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)); it is the case for publishing another set.*


![relative to other quants](./vs_reference.png)

Every set is plotted relative to the same baseline: an uncalibrated EXL3 quantization of this model at each bit rate, which favors no calibration. Each set keeps its own draw-to-draw noise (about 1-2% here) rather than one being drawn as a perfect line, and the baseline's own noise moves all of them together, so their order at any size is unaffected.


| BPW | Composite vs best other quant | Compared against | Size difference | Worst eval slice | Slice |
|---|---|---|---|---|---|
|2.50|0.58|turboderp/Qwen3.8-27B-exl3 (plain): 2.50bpw|-1.91 GiB (-16.7%)|0.71|multilingual documents|
|3.00|0.69|turboderp/Qwen3.8-27B-exl3 (plain): 3.00bpw|-1.92 GiB (-14.9%)|0.80|multilingual documents|
|3.50|0.63|turboderp/Qwen3.8-27B-exl3 (plain): 3.50bpw|-1.91 GiB (-13.4%)|0.66|multilingual documents|
|4.00|0.74|turboderp/Qwen3.8-27B-exl3 (plain): 4.00bpw|-1.77 GiB (-11.3%)|0.83|multilingual documents|
|5.00|0.86|turboderp/Qwen3.8-27B-exl3 (SC): SC_5.00bpw_H6|-1.62 GiB (-8.7%)|1.15|the model's own voice|
|6.00|0.95†|turboderp/Qwen3.8-27B-exl3 (SC): SC_6.00bpw_H6|-1.62 GiB (-7.6%)|1.10|the model's own voice|


Ratios of KLD including the unquantized model's own rounding noise, below 1.00 better: a difference far smaller than that noise reads close to 1.00, as it would in use. The composite ratio is the geometric mean of the two composite slices' ratios at their weights, so each slice counts as weighted whatever its absolute scale. Each revision is compared with the most competitive quant at the same nominal bit rate among the sets above; the size column shows how much larger (+) or smaller (-) this card's checkpoint is than that one. The worst eval slice is the largest ratio over every eval slice: real user prompts (WildChat), real agent sessions (Open-SWE-Traces), documents in a user turn, documents as tool results, multilingual documents, tool-use loops, the model's own voice. At or below 1.00, no slice is worse. The slices beyond the two in the composite are constructed conversations, with documents and tool definitions held out of calibration; they locate an effect rather than measure real use.

† Within about twice the unquantized model's own rounding noise: both quants here are, in practice, indistinguishable from the original, and the ratio compares differences too small to matter in use. Read it as noise, not as a ranking.



# What it still knows

Quantization can leave a model fluent but wrong about facts, which KLD does not show. Each quant answered 3059 closed-book questions the unquantized model gets right, with the answer scored token by token. This is a test of memory on short factual questions, not of overall capability.

|bpw|Facts kept|Confident errors|
|---|---|---|
|2.50|80.8%|2.16%|
|3.00|89.4%|1.34%|
|3.50|91.6%|0.59%|
|4.00|95.0%|0.59%|
|5.00|96.2%|0.10%|
|6.00|97.3%|0.23%|

- Facts kept: of the correct answers the unquantized model would give when sampling, the share this quant still gives.
- Confident errors: questions where this quant puts at least 70% on a wrong token, so it states something false rather than hesitating.

Loss is not spread evenly. A quant holds on to what the unquantized model was sure of and loses more of what it barely knew:

![facts kept by how firmly the unquantized model held the answer](./recall.png)

Bars are approximate and cover both sampling and the difference between equivalent conversions. Differences of a point or two between rungs mean little.

Other published quants of this model, scored on the same questions, show how much is at stake at each size, and how fast facts go once a quant is small enough:

![facts kept against VRAM, this card and other quants of this model](./recall_vs_size.png)

Each dot is one published quant, at its VRAM as served. Each is a single conversion, and conversions with the same settings land a few points apart here, so read where the dots lie as a group rather than any one of them against this card's line.


# More detail

What each eval slice is, every slice compared on its own, how the composite is built, the calibration, and how to reproduce these quants: [TECHNICAL.md](./TECHNICAL.md).
