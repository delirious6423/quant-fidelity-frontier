---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - megacapybara
  - qwen3.8
  - flash-next
  - exl3
  - quantized
  - moe
  - blackwell
  - rtx-5090
  - speculative-decoding
  - mtp
---

# Qwen3.8 Flash-Next-MC

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (177B parameters, 512 experts per layer, 10 active)
for one **NVIDIA RTX 5090**, in the file format of **[MegaCapybara](https://github.com/perkel666/MegaCapybara)**.
The weights are [turboderp's EXL3 quantization](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (dense
layers and head at 4 bits), repacked for MegaCapybara's own kernels, with the MTP layer re-quantized at 4 bits for faster
speculative decoding. Two models: **Tiny** (routed experts at 2 bits: faster, more context) and **Small** (routed
experts at 3 bits, from turboderp's 3.05 bpw release: closer to the original). Every model file carries its measured
accuracy and speed.

> **Engine support:** these files need a MegaCapybara release with Qwen3.8-Flash-Next support. The current release
> (1.35) does not read them yet; the next one will, and its launcher will download them from this repository.

## Files

| File | What it is | On disk |
|---|---|---|
| `qwen3.8-flash-next-tiny.mcapy` | **Qwen3.8 Flash-Next-MCTiny**: a model file (its routed experts at 2 bits, scores, default expert profile) | 30.7 GB |
| `qwen3.8-flash-next-small.mcapy` | **Qwen3.8 Flash-Next-MCSmall**: a model file (its routed experts at 3 bits, scores, default expert profile) | 45.8 GB |
| `qwen3.8-flash-next-trunk.mcapy` | the dense layers, gated residual, routers, head and embedding (4 bits), tokenizer and config | 5.1 GB |
| `qwen3.8-flash-next-ngram.mcapy` | the per-layer n-gram embedding table (4 bits; read from disk / RAM, row by row) | 26.3 GB |
| `qwen3.8-flash-next-mtp.mcapy` | the MTP layer, re-quantized at 4 bits from Qwen's BF16 release (the speculative drafter) | 1.4 GB |
| `qwen3.8-flash-next-vision-fp8.mcapy` | the vision tower in FP8 (images) | 0.56 GB |
| `qwen3.8-flash-next-vision-bf16.mcapy` | the vision tower in BF16 | 0.99 GB |

Each model file names its parts; put them in the same folder. Tiny and Small share the trunk, n-gram, MTP and vision
files (a model with them: Tiny 65 GB, Small 80 GB; MegaCapybara's launcher downloads a model with its parts).

**How it fits one RTX 5090:** the routed experts (Tiny 28.6 GiB, Small 42.6 GiB) do not all fit in 32 GB next to
everything else. MegaCapybara keeps the most used ones in VRAM (by a usage profile, adapted while serving) and reads the
rest from pinned RAM over PCIe. With a small context ~20,000 of Tiny's 24,576 experts sit in VRAM and serve ~98% of the
expert uses (Small: ~13,000); a larger KV cache leaves fewer, and decoding slows (see Speed). Windows lets the GPU map
about half of the RAM, so Small needs at least ~18 GiB of its experts in VRAM, which limits its context. Plan on 64 GB of
RAM.

## Accuracy against the original

KL divergence and top-1 agreement against Qwen's BF16 weights computed with FP32 math, by MegaCapybara's engine, KV cache
8/8, no YaRN:

| Text | Tiny KL | Tiny top-1 | Small KL | Small top-1 |
|---|---|---|---|---|
| Held-out set (81,880 tokens: reasoning traces, math, code, chat, wiki) | 0.1454 | 86.7% | 0.0613 | 91.7% |
| Chat and tool calls written by Flash-Next itself (the model's turns) | 0.0760 | 90.0% | 0.0283 | 93.7% |
| Real agentic coding sessions (the model's own new turns) | 0.2234 | 85.2% | 0.1260 | 88.7% |

Per source of the held-out set, Tiny / Small: reasoning 0.098 / 0.040, math 0.070 / 0.020, code 0.086 / 0.038, chat
0.310 / 0.143, wiki 0.163 / 0.066. Nearly all of Tiny's gap to the original comes from the 2-bit routed experts (the
dense layers at 4 bits add ~3%, the head ~1%).

At 32K tokens, against KV 8/8 without YaRN (relative KL; math + code / agentic coding):

| Setting | Tiny | Small |
|---|---|---|
| KV 6/6 | none measurable | +2.2% / none measurable |
| KV 6/4 | +1.1% / +1.1% | +5.8% / +2.5% |
| KV 4/6 | +1.7% / +2.3% | +10.1% / +1.5% |
| KV 4/4 | +3.7% / +1.3% | +11.9% / +4.5% |
| YaRN x2 (512K-token conversations) | +2.8% / +2.4% | +8.6% / +4.6% |
| YaRN x4 (1M-token conversations) | +11.6% / +6.1% | +29.7% / +14.1% |

(Small's percentages are of a smaller KL: at 32K its math + code KL is 0.019, Tiny's 0.052, so the added amounts are
alike.)

The KV cache costs little at any length: Flash-Next's sparse attention reads ~2,048 selected keys per token however long
the context is. MegaCapybara scales YaRN to the longest conversation the context allows.

## Speed (RTX 5090)

Through MegaCapybara's server, one conversation unless stated, greedy, thinking off, answers up to 2,000 tokens (tokens
per second; MTP: the model's own MTP layer drafting, the answers unchanged):

| Model | Experts in VRAM | MTP, code | MTP, prose | MTP, 4 / 8 conversations at once | No speculation, code / prose | Reading an 8K prompt |
|---|---|---|---|---|---|---|
| Tiny | ~20,000 (small context) | 460 | 313 | 705 / 852 | 299 / 289 | 8,300 |
| Tiny | 9,863 | 225 | 200 | 236 / 249 | 176 / 194 | 8,100 |
| Tiny | 4,220 | 128 | 117 | 133 / 148 | 108 / 115 | 7,200 |
| Small | ~13,000 (small context) | 221 | 191 | 239 / 186 | 184 / 198 | 7,400 |
| Small | 10,513 | 190 | 177 | 220 / 181 | 154 / 169 | 7,200 |

The launcher projects each setting's speed from these points by how many experts it leaves in VRAM.

## License

[Qwen Community License 1.0](LICENSE), as Qwen3.8-Flash-Next. Note its conditions: a commercial "Model as a Service" or
"AI Work Assistant" business needs a separate license from Qwen; products past 100M monthly users or US$20M monthly
revenue must show the model name. What was changed and by whom: [NOTICE](NOTICE). The MegaCapybara engine is MIT
licensed.
