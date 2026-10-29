---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- radiance
- rad
- glm5-next
- mxfp4
- moe
- rocm
---

# GLM-5.3-Flash: radiance conversion (MXFP4 experts)

This is a [radiance](https://codeberg.org/StillDeadcode/radiance) `.rad` conversion of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B parameters in total, 18B active,
288 routed experts, top-8). It is not a GGUF, safetensors or exl3 checkpoint, and it loads only in radiance's
`glm5next` plugin.

| | |
|---|---|
| File | `glm-5.3-flash-mxfp4.rad`: 170.73 GiB of weights, 154,856-token vocabulary |
| SHA-256 | `d8bf7ae0c9619a9bc08262e5a21a7030f8d4581ce68fddb7b9f84dcbfbdac118` (183,330,418,688 bytes) |
| Source | [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16). The plugin accepts only the unquantised release. |
| Converted with | `rad-convert` from radiance 1.12.0 |
| Routed and shared experts | MXFP4: FP4 E2M1 codes with one E8M0 scale per 32 values, rounded to nearest, no calibration |
| Everything else | BF16 as released: attention, KDA, indexer, dense FFN, hyper-connections, router, embeddings, output head, vision tower and MTP layer |
| Quantisation error | Mean relative Frobenius norm 0.123 over the 24,854 expert weights. The worst is 0.144 (`blk.44.ffn_down_exps.184`). |

The recipe (`glm53-flash-mxfp4.recipe`):

```
*attn_kv_b.weight   cast  dtype=bf16
*exp*               rtn   codes=fp4_e2m1 group=32 scale=e8m0 rule=mx
```

```
rad-convert GLM-5.3-Flash-BF16/ -o glm-5.3-flash-mxfp4.rad --recipe glm53-flash-mxfp4.recipe
```

MXFP4 is the narrowest expert format that the `glm5next` kernels in radiance 1.12.0 accept. They refuse 2-bit and
3-bit codes when the converter declares the plan. They also refuse an FP8 trunk, because the plugin runs its linear
layers as plain BF16 GEMMs.

## Requirements

- **[radiance](https://codeberg.org/StillDeadcode/radiance) 1.12.0 or later**, with the `glm5next` plugin. The
  Docker image is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance). The setup below was
  tested on 1.14.0.
- **`--kv-cache-dtype bf16`.** The plugin keeps its latent-attention, index-tail and KDA caches in BF16 and refuses
  `fp8`.
- **Memory.** With tensor parallelism, each rank holds half of every expert. Radiance places experts as whole
  units (one expert's `gate_up` plus `down`, 6.38 MiB a rank) in VRAM, the pinned host pool or the file on disk.
  Each expert's `gate_up` weight is stored as two planes and joined at load, so the disk tier can't read it in
  place. In practice every expert must fit in **VRAM or the pinned host pool**: about **77 GiB a rank** at
  `--tp 2`, including the MTP layer. `--host-pool-mib` is per rank, not a total.
  - **Two 32 GB cards with 128 GB of system RAM works, with almost no RAM to spare.** Each card holds 29.6 GiB of
    weights in VRAM (the BF16 trunk plus about 21 GiB of experts), and 55.8 GiB of experts a rank sits in the pinned
    host pool. Nothing is read from disk. Radiance pins 115 GiB in total and leaves about 1.5 GiB of RAM for
    everything else. To fit, the setup gives up the vision tower, the prefix cache and most context, and serves two sequences at a time (see the command below). Each of those
    flags was needed: every looser setting stopped at startup with "out of memory".
  - It does **not** start on two 32 GB cards with 60 GB of system RAM.
- **Four 32 GB cards (`--tp 4`) is the faster setup:** about 38 GiB of experts a rank, mostly in VRAM.
- More VRAM (two 48 GB cards or larger) lowers the host-pool requirement and frees RAM for context and sequences.

## Running it: two 32 GB cards, 128 GB RAM

This is the command used for the results below, on two Radeon AI PRO R9700 (gfx1201) cards with 128 GB of DDR5:

```
radiance --model glm-5.3-flash-mxfp4.rad --tp 2 --p2p off \
  --placement expert_tiered --host-pool-mib 58368 --host-headroom-mib 1536 \
  --weights-disk-tier --gpu-headroom-mib 32 \
  --max-num-seqs 2 --max-model-len 16384 \
  --mm-max-patches 0 --kv-cache-dtype bf16 --prefix-cache-mib 0 --checkpoint-slots 2 \
  --served-model-name GLM-5.3-Flash --host :: --port 8000
```

| Flag | Why |
|---|---|
| `--host-pool-mib 58368` | A 57 GiB pinned pool a rank. At 57344, 125 `gate_up` experts a rank fit nowhere. |
| `--host-headroom-mib 1536` | At 2048, the checkpoint slots can't be pinned. |
| `--gpu-headroom-mib 32` | Gives the experts the last few MiB of VRAM. |
| `--max-num-seqs 2 --max-model-len 16384` | With 4 sequences or a 32k context, the KV cache and per-sequence state take VRAM the experts need. |
| `--mm-max-patches 0` | Text only: keeps the vision tower off the cards. |
| `--prefix-cache-mib 0 --checkpoint-slots 2` | Cuts the pinned host memory that the prefix cache and linear-state snapshots take. |
| `--weights-disk-tier` | A safety net. In this setup nothing is read from disk. |

Radiance tells you what didn't fit. If it stops with "out of memory", the lines above its startup report name the
shortfall and the flag to change. With less RAM than this machine, nothing above leaves room to give back.

`--p2p` defaults to `auto`, which asks the driver and uses peer-to-peer copies between the cards when they support
it. Add `--p2p off` if your system doesn't support P2P, or if `off` measures faster on your system. With `off`, the
cards exchange data through pinned host memory. P2P is disabled on the R9700s above, so they use `--p2p off`.

## Results: TP2, two R9700 cards, 128 GB RAM

Measured with BetterBench 0.6.0 (corpus v1.0, 20 passes per category, temperature 0.7, cold prefix cache). Radiance 1.14.0 ran the command above. Hardware: an Intel Core Ultra 7 270K
Plus, 4×32 GB DDR5-5400 (STREAM copy 68 GB/s), and two Radeon AI PRO R9700 cards capped at 210 W, each on a PCIe
5.0 x8 link. The model's own MTP head drafts 3 tokens a step.

| Single stream | |
|---|---|
| Decode, weighted across categories | **13.6 tok/s** (11.7 reasoning to 15.8 summarization) |
| Time to first token, p50 | 2.3 s |
| Gap between stream updates, p99 | 277 ms |
| Tokens per update (MTP) | 2.3 to 3.2 |

| Concurrency | Aggregate decode | Per-request decode | TTFT p50 |
|--:|--:|--:|--:|
| 1 | 12.1 tok/s | 13.2 tok/s | 2.3 s |
| 2 | 13.9 tok/s | 7.5 tok/s | 2.8 s |

| Prompt processing | Prompt tokens | TTFT | Prefill |
|--:|--:|--:|--:|
| 2k | 1,514 | 5.0 s | 306 tok/s |
| 8k | 5,892 | 17.1 s | 344 tok/s |
| 16k | 11,799 | 33.8 s | 350 tok/s |

Decode is paced by PCIe: about three quarters of the experts sit in host RAM and are read over the link when a token
routes to them. A second sequence adds little throughput. GLM-5.3-Flash thinks at length: in BetterBench, 92% of runs hit their `max_tokens` while
still reasoning. Long answers therefore take minutes at these speeds.

## License

MIT, the same as the base model. Converted by eniga; all credit for the model goes to Z.ai.
