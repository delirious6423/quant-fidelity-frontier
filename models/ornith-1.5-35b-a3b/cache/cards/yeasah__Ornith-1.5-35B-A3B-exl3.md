---
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: transformers
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- exl3
base_model_relation: quantized
---

# EXL3 quantizations of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)

These quantizations can be used with [exllamav3](https://github.com/turboderp-org/exllamav3) or vLLM via [vllm-exl3-plugin](https://github.com/yeasah/vllm-exl3-plugin)

Actual consumed VRAM will be lower than the total checkpoint size because of the handling of the input embeddings (and potentially the vision tower). The embeddings are not quantized, because EXL3 is not an efficient format for this purpose. The strategy for this is different for the two platforms:
- For exllamav3, input embeddings are typically offloaded to CPU, which has a performance cost but removes that tensor from VRAM entirely.
- For vllm-exl3-plugin, input embeddings are typically quantized at load time to a format called `blockq`, which is a computationally inexpensive and embedding-compatible quantization format. The embeddings are resident in VRAM, but much smaller.


The vision encoder is left unquantized. This preserves full vision fidelity as compared to the base model, which is otherwise harder to characterize. This may represent a significant additional component to VRAM utilization. There are two strategies to eliminate this:
- CPU offload for the encoders. Especially for text-heavy use cases, this represents a minimal performance impact.
- Omitting the encoders entirely, for text-only use cases.

---

# Available bitrates

Each bitrate is available as a separate revision of this repository. When planning available VRAM for kv space, refer to the appropriate VRAM column.

| BPW | Size on disk | VRAM (exllamav3) | VRAM (vLLM) | KLD | PPL |
|---|---|---|---|---|---|
|[2.08](https://huggingface.co/yeasah/Ornith-1.5-35B-A3B-exl3/tree/2.00bpw-H5)|10.83 GiB|9.88 GiB|9.31 GiB|0.1775|12.3358|
|[3.08](https://huggingface.co/yeasah/Ornith-1.5-35B-A3B-exl3/tree/3.00bpw-H5)|14.74 GiB|13.79 GiB|13.23 GiB|0.0633|11.5848|
|[4.08](https://huggingface.co/yeasah/Ornith-1.5-35B-A3B-exl3/tree/4.00bpw)|18.71 GiB|17.77 GiB|17.20 GiB|0.0287|11.4631|
|[5.08](https://huggingface.co/yeasah/Ornith-1.5-35B-A3B-exl3/tree/5.00bpw)|22.63 GiB|21.68 GiB|21.11 GiB|0.0176|11.4504|
|[6.08](https://huggingface.co/yeasah/Ornith-1.5-35B-A3B-exl3/tree/6.00bpw)|26.54 GiB|25.59 GiB|25.03 GiB|0.0140|11.4230|
|(base)|66.97 GiB|||0.0123|11.4510|

- BPW: Average bits per weight across tensors (not including input embeddings, encoders, or output head)
- Size on disk: Total aggregate size of tensors in checkpoint, in GiB
- VRAM (exllamav3): Total VRAM consumption as typically used in exllamav3, in GiB (minus embeddings)
- VRAM (vLLM): Total VRAM consumption as typically used in vLLM, in GiB (blockq embeddings, offloaded/omitted encoders)
- KLD: Mean KL divergence from unquantized base model at this bitrate (lower is better)
- PPL: Perplexity at this bitrate (lower is better)

KLD and PPL are quality metrics that represent fidelity to the base model. The values for the base model row are the noise floor: it is not possible for a quantization to meaningfully score better than this value using these metrics.

---

# Quality vs. bitrate plots

## Plot of resident VRAM versus KLD divergence from unquantized base model

![qb_kld_vram](./qb_kld_vram.png)

## Plot of resident VRAM versus perplexity

![qb_kld_ppl](./qb_ppl_vram.png)