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

They are calibrated broadly, on what models read and write while working (documents and tool output in conversations, code and diffs, the model's own replies) rather than on web text alone, and measured with agentic and coding work as the primary use case.

Bits are allocated per tensor, from the model's measured sensitivity on its own replies, at the same total size as uniform bits.

Earlier revisions of this repository (uniform bits, the same calibration) are superseded: at equal size the current ones cut composite excess KLD by 4-12%. They remain in each branch's history.

Actual consumed VRAM may be significantly lower than the total checkpoint size, depending on configuration: see the VRAM columns below.

---

# Which size?

Each bitrate is available as a separate revision of this repository. When planning available VRAM for kv space, refer to the appropriate VRAM column.

![quality vs size](./quality_vs_size.png)

| BPW | Size on disk | VRAM (exllamav3) | VRAM (vLLM) | Excess KLD | Δ perplexity |
|---|---|---|---|---|---|
|[2.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/2.00bpw-H5)|9.89 GiB|7.32 GiB|7.13 GiB|0.0808|+9.06%|
|[3.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/3.00bpw-H5)|12.72 GiB|10.16 GiB|9.96 GiB|0.0162|+1.83%|
|[3.50](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/3.50bpw-H5)|14.14 GiB|11.57 GiB|11.38 GiB|0.0084|+1.02%|
|[4.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/4.00bpw)|15.70 GiB|13.14 GiB|12.94 GiB|0.0041|+0.47%|
|[5.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/5.00bpw)|18.53 GiB|15.97 GiB|15.78 GiB|0.0010|+0.12%|
|[6.00](https://huggingface.co/yeasah/Qwen3.8-27B-exl3/tree/6.00bpw)|21.37 GiB|18.80 GiB|18.61 GiB|0.0002|+0.05%|
|(base)|51.75 GiB|||||

- BPW: Average bits per weight across tensors (not including input embeddings, encoders, MTP draft layers, or output head)
- Size on disk: Total aggregate size of tensors in checkpoint, in GiB
- VRAM (exllamav3): Total VRAM consumption as typically used in exllamav3, in GiB (minus embeddings and MTP draft layers)
- VRAM (vLLM): Total VRAM consumption as typically used in vLLM, in GiB (blockq embeddings, offloaded/omitted encoders, no MTP draft layers)
- Excess KLD: how far each quant's next-token distribution is from the unquantized model's, minus the unquantized model's own rounding noise, so 0 would be indistinguishable from the original (lower is better)
- Δ perplexity: change in perplexity on the same text, same weights (lower is better)




# Against other quants of this model

*This section appears because other quantizations of this model exist ([turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)); it is the case for publishing another set.*


![relative to other quants](./vs_reference.png)

Every set is plotted relative to the same baseline: an uncalibrated EXL3 quantization of this model at each bit rate, which favors no calibration. Each set keeps its own draw-to-draw noise (about 1-2% here) rather than one being drawn as a perfect line, and the baseline's own noise moves all of them together, so their order at any size is unaffected.


| BPW | Composite vs best other quant | Compared against | Size difference | Worst eval slice | Slice |
|---|---|---|---|---|---|
|2.00|0.90|turboderp/Qwen3.8-27B-exl3 (SC): SC_2.00bpw_H3|+0.39 GiB (+4.2%)|1.10|the model's own voice|
|3.00|0.80|turboderp/Qwen3.8-27B-exl3 (plain): 3.00bpw|-0.15 GiB (-1.2%)|0.83|multilingual documents|
|3.50|0.64|turboderp/Qwen3.8-27B-exl3 (plain): 3.50bpw|-0.15 GiB (-1.0%)|0.74|multilingual documents|
|4.00|0.74|turboderp/Qwen3.8-27B-exl3 (SC): SC_4.00bpw_H5|+0.15 GiB (+1.0%)|1.00|the model's own voice|
|5.00|1.04‡|turboderp/Qwen3.8-27B-exl3 (SC): SC_5.00bpw_H6|+0 MiB (+0.0%)|1.26|the model's own voice|
|6.00|1.01†|turboderp/Qwen3.8-27B-exl3 (SC): SC_6.00bpw_H6|+0 MiB (+0.0%)|1.38|the model's own voice|


Ratios of excess KLD, below 1.00 better: the composite ratio is the geometric mean of the two composite slices' ratios at their weights, so each slice counts as weighted whatever its absolute scale. Each revision is compared with the most competitive quant at the same nominal bit rate among the sets above; the size column shows how much larger (+) or smaller (-) this card's checkpoint is than that one. The worst eval slice is the largest ratio over every eval slice: real user prompts (WildChat), real agent sessions (Open-SWE-Traces), documents in a user turn, documents as tool results, multilingual documents, tool-use loops, the model's own voice. At or below 1.00, no slice is worse. The slices beyond the two in the composite are constructed conversations, with documents and tool definitions held out of calibration; they locate an effect rather than measure real use.

† Within about twice the unquantized model's own rounding noise: both quants here are, in practice, indistinguishable from the original, and the ratio compares differences too small to matter in use. Read it as noise, not as a ranking.

‡ A composite slice is left out where both quants sit within about twice the unquantized model's own rounding noise on it; the ratio is over the other slice alone: 5.00 bpw (real agent sessions (Open-SWE-Traces) left out).


# More detail

What each eval slice is, every slice compared on its own, how the composite is built, the calibration, and how to reproduce these quants: [TECHNICAL.md](./TECHNICAL.md).
