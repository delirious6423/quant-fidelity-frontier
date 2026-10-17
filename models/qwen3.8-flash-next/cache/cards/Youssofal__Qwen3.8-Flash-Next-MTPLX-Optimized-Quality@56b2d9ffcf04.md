---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mtplx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- macos
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8-flash-next
- moe
- mtp
- mtplx
- local-ai
- chat
- qwen3.8
- qwen3-8
- qwen-3.8
- local-llm
- llm
- 8-bit
- vision
- m5-max
- m3-ultra
- mac-studio
- opencode
- claude-code
- flash-next
- qwen3-8-flash-next
- qwen4
- 125b
---
**[MTPLX](https://mtplx.com): the fastest way to run Qwen 3.8 on a Mac. Native multi-token-prediction speculative decoding on Apple Silicon, two to three times the speed of plain decoding, exact at any temperature.**

# Qwen 3.8 Flash-Next Optimized Quality

**8-bit body and MTP head, BF16 structural tensors, and a 4-bit n-gram table. Higher-fidelity Flash-Next build.**

Qwen's 125B-A6B Flash-Next preview, the Qwen4-generation architecture with GDN
hybrid MoE, Qwen Sparse Attention and the 51B-parameter n-gram memory, at
8 bits: the highest-fidelity way to run it on a Mac, with its native
multi-token-prediction head drafting through [MTPLX](https://mtplx.com)'s
speculative path. It is built for Mac Studio with 256 GB or 512 GB. On a
128 GB Mac, pick
[Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed),
the recommended build.

The 32 GB n-gram table streams from SSD, so only the weights stay in memory:
about 128.5 GiB, which leaves about 59.5 GiB for context and the session cache
on a 256 GB Mac.

## How it is built

- The experts, attention and the multi-token-prediction head at 8 bits with
  64-weight groups, twice the precision of Optimized Speed.
- The structural weights stay in BF16.
- The n-gram table at 4 bits, as a separate sidecar that MTPLX streams from
  SSD. The vision tower is preserved in the weights.

| | |
|---|---|
| Download | 170 GB (includes the 32 GB n-gram table) |
| Weights in memory (n-gram on SSD) | about 128.5 GiB |
| Recommended Macs | 256 GB and 512 GB (on 128 GB, use Optimized Speed) |
| Context window | 262,144 tokens |
| MTP depth | adaptive, ceiling 3 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract) |

Speed on 256 GB and 512 GB Macs has not been measured yet. The 8-bit weights
move twice the bytes per token of Optimized Speed, so it decodes slower.

The serving contract ships inside `mtplx_runtime.json`. MTPLX reads it on
load. Drafts are accepted with the probability-ratio rule plus residual
resampling, so the output follows the model's own distribution at any
temperature.

## Use it

Mac app (MTPLX 2.12.0 or later): download at
[mtplx.com](https://mtplx.com), pick "Qwen 3.8 Flash-Next Optimized Quality".
On Macs with 256 GB or more it is listed second, after Optimized Speed.

Command line (MTPLX 2.12.0 or later):

```bash
pip install mtplx
mtplx serve --model Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality --model-id mtplx-flash-next-optimized-quality
```

Siblings: [Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed)
(the recommended build) and
[Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed)
(flat 4-bit, the quickest build).

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(Qwen Community License; the upstream model card is preserved in this repo as
`README-upstream-qwen.md`).
