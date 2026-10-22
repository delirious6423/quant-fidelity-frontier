---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
  - mlx
  - mlx-serve
  - quantization
  - imatrix
  - qwen4_exp
  - moe
  - sparse-attention
  - ngram-embedding
---

# Qwen3.8-Flash-Next for MLX-Serve — iQ-MLX 6.6 bpw

MLX-Serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(the Qwen4 preview architecture, `model_type: qwen4_exp`). imatrix-calibrated 6-bit
routed experts, 8-bit everywhere else — **measured 6.56 bpw incl. scales**
(105.2 GB trunk / 128.4 B logical params, ddalcu's card convention). At 77% of
[ddalcu/8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-8bit)'s
resident bytes it **matched the 8-bit pack's greedy top-1 in all five scoring
runs** (mean 92.31 on the 359 common positions, identical miss counts every run)
and holds the lowest median token-level KLD of the four packs.

This pack changes nothing about the architecture or serving behavior — see
ddalcu's pack cards for the qwen4_exp details (gated residual streams, mmapped
4-bit n-gram table, QSA, MTP head, vision tower).

```bash
mlx-serve --model twyx/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw --serve
```

Requires [MLX-Serve](https://github.com/ddalcu/mlx-serve) ≥ 26.9.1 (6-bit expert
unpacking).

~106 GB resident, ~138 GB download (32 GB n-gram table mmapped, not resident;
0.9 GB bf16 vision tower bundled). Sized for 128–256 GB Macs.

## Scoring vs the 4.7 bpw, 5.6 bpw and 8-bit packs

Scored with against the bf16 model's own logits on 392 held-out positions (never seen by the imatrix):
`top1` = the pack's greedy token equals bf16's argmax; `KLD` = KL(bf16 ‖ pack)
over the bf16 top-1024 support. **Every pack was scored 5 times on the same
machine and MLX-Serve build.** The runs split into two regimes: run 1 (cold
server) differs from the later runs, which are *byte-identical* per pack (runs
2–5 for 4.7 / 6.6 / 8-bit; runs 3–5 for the 5.6 bpw pack, which needed two
warm-up runs). The cold→steady swing is itself pack-dependent and large (8-bit
moves −5.5 pp on code, +3.2 pp on agent), so slice-level gaps under ~5 pp
between packs are treated as noise unless they hold in **both** states. Both
states are reported below (slice top-1, all rows the pack scored; common-359
top-1 after the `|`):

| pack | resident | cold: agent / code / math / prose \| top-1 | steady: agent / code / math / prose \| top-1 | KLD median (5-run) |
| --- | --- | --- | --- | --- |
| 8-bit | ~123 GB | 89.2 / 96.2 / 93.8 / 93.4 \| **92.76** | **92.4** / 90.7 / 93.8 / 89.5 \| **92.20** | 0.0028 |
| **iQ-MLX 6.6 bpw (this pack)** | ~106 GB | 89.2 / 96.0 / **96.2** / 93.4 \| **92.76** | 88.0 / **95.9** / **96.2** / **94.7** \| **92.20** | **0.0023** |
| iQ-MLX 5.6 bpw | ~90 GB | 88.6 / 94.2 / 93.8 / 94.7 \| 92.20 | 87.3 / 94.0 / 93.8 / 90.8 \| 91.09 | 0.0025 |
| iQ-MLX 4.7 bpw | ~75 GB | 88.5 / 90.0 / 93.8 / 93.4 \| 91.92 | 88.6 / 88.5 / 93.8 / 92.1 \| 91.64 | 0.0041 |

What the repetition (5× per pack) established:

- **Overall top-1 parity with 8-bit holds in both server states** (92.76 cold /
  92.20 steady, identical miss counts in all five runs) — this pack is the
  bf16-matching size point of the ladder.
- **Math is the only slice win robust to server state:** 96.2 vs the 8-bit
  pack's 93.8 in *both* cold and steady runs (+2 of 80 rows each).
- **Code and prose flips are engine-state, not expert width.** Cold, this pack
  and 8-bit are tied (code 96.0 vs 96.2, prose 93.4 vs 93.4); steady, this pack
  leads (95.9 vs 90.7, 94.7 vs 89.5). The same 8-bit weights swing 5.5 pp
  between states, which is larger than any pack-vs-pack slice gap here except
  math — treat sub-5 pp slice differences as noise.
- **Agent:** tied cold (89.2 vs 89.2); steady, the 8-bit pack pulls ahead
  (92.4 vs 88.0) while every quantized width (4.7/5.6/6.6) stays at ~88. The
  steady-state agent gaps sit on bf16 knife-edge tokens (JSON/grammar
  branches); structured-output/grammar constraints remain the practical fix at
  any width, and a long-running warm server currently favors 8-bit for agent
  traffic specifically.
- **Distribution:** this pack has the best 5-run median KLD (0.0023, vs 8-bit's
  0.0028), with the code-slice median (0.0072) below 8-bit's (0.0099). The
  paired per-row head-to-head on the 359 common rows still favors 8-bit on raw
  KL (188–165), driven by shared blow-up positions; KLD **means** move between
  runs and are the least steady number in this table.

## Widths

| tensors | width |
| --- | --- |
| routed experts (48 layers + MTP, 147 tensors) | 6-bit, group 64, imatrix-weighted affine (6.50 bits effective incl. scales+biases) |
| attention, GDN, hyper-connections, indexer, shared experts, lm_head | 8-bit, group 64 (8.5 effective) |
| embed_tokens | 4-bit, group 64 |
| n-gram table | 4-bit, group 32 (mmapped `ngram_table.bin`, 32.0 GB, not counted in bpw) |
| vision tower | bf16 pass-through (0.9 GB, excluded from bpw) |
| routers, gates, norms, convs, SSM state | bf16 |

Bits per weight, scales and biases included: **6.56** over the 128.4 B
quantized trunk (n-gram table and vision tower excluded, per ddalcu's
convention). Verified from safetensors headers: 147 expert tensors pack at
exactly 6.00 bits, 646 tensors at 8.00, embed at 4.00.

## Misc

Sibling pack (pending upload): [iQ-MLX-5.6bpw](https://huggingface.co/twyx/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-5.6bpw)
(5-bit experts, ~90 GB — same calibration, near-identical typical-token KLD,
smaller; see the agent note before trading 6.6 down to it).
