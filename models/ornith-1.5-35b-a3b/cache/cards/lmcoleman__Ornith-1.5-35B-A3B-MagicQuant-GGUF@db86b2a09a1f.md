---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: image-text-to-text
tags:
  - gguf
  - magicquant
  - hybrid-quantization
  - moe
  - multimodal
  - strix-halo
---

# Ornith-1.5-35B-A3B — MagicQuant hybrid GGUFs

Per-group hybrid quantizations of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(`qwen3_5_moe`, ~35B total / A3B active, 256 experts / 8 routed, vision-capable),
produced by MagicQuant. The Q5 and Q6 tiers come from the August **measured
evolutionary search** (v1); the Q4 tier was **rebuilt on 2026-09-12 with the
streamed-bytes allocator** (v2) — see "Why the Q4 tier changed" below.

Pins: MagicQuant `3acccef` (Q4 v2, Q6 v2), `ed17991` (Q5), Foundry `2f99202`.

## Files

| file | size | ratio vs BF16 | weight bytes streamed per decoded token | note |
|---|---|---|---|---|
| `Ornith-1.5-35B-A3B-Q4_K_M-v2.gguf` | 19.58 GiB | 0.30× | **2.53 GiB** | **recommended** — rebuilt 2026-09-12 |
| `Ornith-1.5-35B-A3B-Q5_K_M.gguf` | 22.88 GiB | 0.35× | 2.06 GiB | uniform Q5_K; the fastest decoder of the three |
| `Ornith-1.5-35B-A3B-Q6_K-v2.gguf` | 28.81 GiB | 0.44× | **2.86 GiB** | rebuilt 2026-09-13 — see "The Q6 tier (2026-09-13)" |
| `mmproj-Ornith-1.5-35B-A3B-f16.gguf` | 0.86 GiB | — | — | vision projector |

"Streamed per token" is the weight traffic one decode step touches, from the
GGUF header: every non-expert tensor once, the routed experts at 8/256. On a
bandwidth-bound machine (this was built and measured on an AMD Strix Halo,
~256 GB/s) it predicts decode speed better than file size does.

## Why the Q4 tier changed (2026-09-12)

The August Q4 file spent its bits the wrong way round for decode: attention,
shared expert and head were kept at BF16/F16 (trunk 11.86 bits/weight) while
the routed experts — 97 % of the weights but only 8-of-256 touched per token —
sat at 4.25. Each token therefore streamed 3.24 GiB, 84 % of it trunk.
The rebuild holds the same payload (19.570 GiB, to the byte), moves the trunk
to Q8_0 (8.66 bpw) and spends the freed bytes on the experts.

Everything below was measured in one session on one box, old and new file
side by side, so the numbers are comparable to each other (not to the August
table, which used a different run).

| gate | old Q4_K_M | new Q4_K_M-v2 | verdict |
|---|---|---|---|
| bytes streamed / token | 3.243 GiB | **2.530 GiB** (−22.0 %) | pass |
| payload (tensor bytes) | 19.571 GiB | 19.570 GiB | equal |
| mean KL vs BF16 (wikitext-2, ctx 512, 100 chunks) | 0.0727 ± 0.0011 | **0.0529 ± 0.0008** | pass (better) |
| top-1 agreement with BF16 | 88.79 % | **90.67 %** | — |
| perplexity (same instrument; BF16 = 7.971) | 7.965 | 7.977 | pass (inside the ±0.13 chunk error) |
| decode tok/s, tg128 @ depth 0 (llama-bench, Vulkan, 6 interleaved cycles) | 46.7 | **61.9** (+32.5 %, 95 % CI +14.9…+15.5) | pass |
| decode tok/s, tg128 @ depth 10000 | 42.5 | **56.2** (+32.3 %, CI +13.5…+14.0) | pass |

Allocation (bits per weight by tensor group, as chosen by the allocator):
embeddings 8.5, attention Q/K/O/head 8.5, SSM/linear-attention 8.71, shared
expert 6.6–7.6, routed experts **4.44** (MXFP4 floor, Q4_K/Q5_K upgrades on the
most sensitive layers). Allocator settings: `--budget-gb 19.57 --bandwidth-weight
0.00065 --probe-mode cumulative --floor X=MXFP4_MOE --floor E=Q8_0 --floor
S,Q,K,O,H=Q8_0`; storage utilisation 1.000; streamed bytes at λ=0 and at the
chosen λ both 2.530 GiB (the floors, not the bandwidth term, set the outcome).

Read the KL row, not the PPL row, when comparing quality: at this chunk count
PPL cannot separate the two files, KL can — and the rebuild is better on it,
because the freed trunk bytes went to `ffn_down_exps` layers that the old
uniform-MXFP4 allocation had left short.

**Q5_K_M was left unchanged on purpose.** It is uniform Q5_K with a 1.4 GiB
trunk and already streams less per token than either Q4 file; there is no
bandwidth win available at its size. If you want the fastest decode of the
three and can afford 3 GiB more, take Q5_K_M.

## The Q6 tier (2026-09-13)

Same method, at the Q6_K file's payload (28.80 GiB vs 28.83 GiB): trunk at Q8_0 instead of F16
head/embeddings and half-F16 attention, experts floored at Q6_K and upgraded with the freed bytes
(6.83 bits/weight). Measured on the same instruments as the Q4 table, old and new side by side:

| gate | old Q6_K | new Q6_K-v2 | verdict |
|---|---|---|---|
| bytes streamed / token | 3.442 GiB | **2.864 GiB** (−16.8 %) | pass |
| payload (tensor bytes) | 28.826 GiB | 28.800 GiB | equal |
| mean KL vs BF16 (wikitext-2, ctx 512, 100 chunks) | 0.0375 ± 0.0006 | **0.0210 ± 0.0004** | pass (better) |
| top-1 agreement with BF16 | 91.83 % | **93.98 %** | — |
| perplexity (BF16 = 7.971) | 7.953 | 7.948 | pass |
| decode tok/s @ depth 0 (llama-server, Vulkan, 6 interleaved cycles) | 32.5 | **37.2** (+14.5 %, 95 % CI +4.5…+4.9) | pass |
| decode tok/s @ depth 10000 (cached 10k-token prompt) | 31.3 | **35.4** (+13.3 %, CI +4.0…+4.3) | pass |

The decode numbers here come from two resident llama-server instances answering alternating requests,
not llama-bench, so compare them within this table only (the Q4 table above used llama-bench).
If you have the memory for it, Q6_K-v2 has the lowest KL to BF16 of the files measured this way
(Q5_K_M was not KL-measured) and decodes faster than the old Q6_K; Q4_K_M-v2 remains the value pick.

## How to read the older numbers honestly

The August search measured all rendered tiers on the same instrument
(CPU llama.cpp perplexity, ctx 512, 100 chunks, wikitext-2; BF16 7.9691):
Q4 7.9530, Q5 7.9951, Q6 7.9792 — a ±0.3 % spread that is noise at this
resolution. Q3 (+2.96 %) and Q2 (+30.2 %) were searched and not rendered;
that cliff is where quality is actually spent. The 2026-09-12 KL measurement
is the sharper instrument that the August card said would be needed to
separate tiers; it puts the new Q4 at 0.053 mean KL against BF16.

## Notes

- Vision requires the `mmproj` file alongside the model.
- Chat template embedded and verified present in every file before upload.
- Vocab metadata (`tokenizer.ggml.token_type`) is INT32 per spec — these files
  load on current mainline llama.cpp builds (checked again on the rebuild).
- The Q4 rebuild uses MXFP4 blocks for most routed experts; any llama.cpp from
  mid-2026 handles them on CPU, CUDA, Vulkan and HIP.
- MoE with a shared-expert split; expect the usual llama.cpp requirements for
  `qwen3_5_moe`.
- Built and measured on an AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151) host.
