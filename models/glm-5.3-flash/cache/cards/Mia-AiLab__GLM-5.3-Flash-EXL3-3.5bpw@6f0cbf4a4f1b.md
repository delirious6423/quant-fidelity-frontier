---
base_model: zai-org/GLM-5.3-Flash
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - glm
  - exl3
  - quantized
  - moe
---

# GLM-5.3-Flash EXL3 3.5bpw, by Mia's AI Lab

Mia's AI Lab's own EXL3 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
A smaller sibling of [GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold):
same format, same layout, same unquantized tensors. Only the routed experts are narrower: every down projection stays
at 4 bits, and the gate and up projections drop to 3 bits in 32 of the 42 MoE layers.

**19.3 GB (11%) smaller than the 4bpw sibling, at a measurable cost in fidelity:** against the BF16 model its KL divergence
is about 1.6-1.7x the sibling's with thinking on and 1.2x with thinking off, and it agrees with BF16's top next-token
0.4-1.3 points less often.

| | 4bpw sibling | This quant |
| --- | --- | --- |
| Routed expert width | 4 bits | 3.50 bits on average (3 or 4 per layer and projection) |
| Size | 175.6 GB | **156.3 GB** |
| Checkpoint files | 83 safetensors shards | 74 safetensors shards |
| KL to BF16, long sessions (7.7k-13.2k tokens), thinking on | 0.00209 | 0.00335 (x1.60) |
| KL to BF16, coding and agentic, thinking on | 0.00288 | 0.00494 (x1.72) |
| KL to BF16, thinking off | 0.261 | 0.312 (x1.19) |
| Top-1 agreement with BF16, thinking on / off | 98.06% / 91.17% | 97.54% / 89.87% |
| Served with TensorFold | yes ([recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)) | not tested yet |

Not measured yet: serving with TensorFold (its 3-bit expert path has not been run on this checkpoint), speed, memory
and benchmark scores. Everything below is next-token fidelity against the BF16 model, not task accuracy.

## Bit allocation

![Routed-expert bit width by layer and projection](assets/bit_allocation.png)

Layer 3 (the first MoE layer) and nine later layers (23, 26, 27, 31, 32, 38, 40, 41, 43) keep 4-bit gate and up
projections. The MTP layer 45 is 4 bits throughout. Averaged over all 43 expert layers that is 3.50 bits a weight
(3.49 over layers 3-44 alone); 18,432 of the 37,152 expert matrices are stored at 3 bits.

## Size

![Checkpoint size by component](assets/size.png)

Everything outside the routed experts (attention, shared experts, dense layers, embeddings, head, vision tower,
MTP experts) takes the same 23.0 GB in both checkpoints. The saving is all in the routed experts: 133.3 GB instead of 152.6 GB.

## Fidelity against the BF16 model

![KL divergence and top-1 agreement, 3.5 bpw vs 4.0 bpw](assets/kl_fidelity.png)

Both quants were scored on exactly the same tokens, so every difference is paired. With thinking on, the KL is small
for both (0.002-0.005 nats a token) and this quant's is 1.6-1.7x the sibling's; tool calls and multi-step agentic
turns show the largest ratio (x1.74 overall, x2.11 on the 8 short multi-step tool-use conversations). With thinking
off, the KL is about 100 times larger for both quants and the extra error is smaller in relative terms (x1.19), but
top-1 agreement drops the most there (-1.30 points). Every interval in the chart excludes zero; in three small subsets
(short chat, multilingual and single tool-call conversations without thinking, 4-14 conversations each) the difference
is within noise.

In the 4bpw sibling, tokens with KL above 0.1 nats are rare with thinking on (0.07% in coding and agentic sessions);
here they are 0.23%. Neither quant has a thinking-on coding or agentic token above 1 nat.

![Long sessions: KL by position in the context](assets/long_context.png)

On the long sessions the ratio stays between 1.55 and 1.65 at every depth, from the first thousand tokens to past
8k, so in these sessions the extra error does not grow with context length (8 sessions, exploratory breakdown).

## What is in the checkpoint

| Part | Format |
| --- | --- |
| Routed experts, layers 3-44 | EXL3, mcg codebook, output scales. down_proj 4 bits; gate_proj and up_proj 3 bits in 32 layers, 4 bits in layers 3, 23, 26, 27, 31, 32, 38, 40, 41, 43 |
| MTP layer 45 experts | EXL3, 4 bits |
| Everything else (attention, KDA, shared experts, dense layers, embeddings, head, vision tower) | BF16, unchanged from [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| Chat template | Z.ai's `chat_template.jinja` |

`quantization_config.json` lists the width of every expert tensor.

## How it was measured

- Reference: [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) (rev `a5b45eb4`), full
  vocabulary. Both quants and the reference ran teacher-forced on the same tokens with exllamav3 1.6.1 on RTX PRO 6000
  Blackwell GPUs.
- Text: 140 conversations in GLM's chat template, scored on the assistant tokens: 8 long thinking-on sessions
  (7.7k-13.2k tokens of math, code and agentic work) and 132 shorter conversations (chat, multilingual, math, code,
  tool calls and agentic turns, with and without thinking). The thinking text is existing reasoning traces read
  teacher-forced, not the model's own generated reasoning.
- KL divergence = mean over tokens of sum_v p(v) (log p(v) - log q(v)), with p the BF16 model; top-1 = agreement with
  the BF16 model's most likely token. Intervals: 2,000 paired bootstrap resamples over conversations.
- The 4bpw sibling's card measures KL against the FP8 release on different text, so its numbers are not comparable
  with these.
- Part of the gap may come from the conversion run rather than from the bit width alone: this checkpoint's 4-bit layers
  are a new conversion, not a copy of the sibling's, and already differ slightly from it. The two effects have not been
  separated.

## Use

The checkpoint is in the same EXL3 layout as the 4bpw sibling, which runs with the
[TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) on DGX Sparks.
Serving this 3.5 bpw checkpoint with TensorFold has not been tested yet; the measurements above loaded it with
exllamav3 1.6.1 (teacher-forced scoring only, no text generation).

## License

This quantization is released under the Apache License 2.0 (`LICENSE`). It is derived from GLM-5.3-Flash by
Z.AI, whose MIT license applies to the base model and is kept in `LICENSE-GLM-5.3-Flash`.

## Credits

- Base model: Z.AI, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
- Quantization format and converter: [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- Serving engine for the 4bpw sibling: [TensorFold](https://github.com/ashhart/TensorFold).
