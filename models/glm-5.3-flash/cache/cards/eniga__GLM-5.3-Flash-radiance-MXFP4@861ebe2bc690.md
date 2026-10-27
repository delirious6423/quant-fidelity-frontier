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
  Docker image is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance).
- **`--kv-cache-dtype bf16`.** The plugin keeps its latent-attention, index-tail and KDA caches in BF16 and refuses
  `fp8`.
- **Memory.** With tensor parallelism, each rank holds half of every expert. Radiance places experts as whole
  units (one expert's `gate_up` plus `down`, 6.38 MiB a rank) in VRAM, the pinned host pool or the file on disk.
  Each expert's `gate_up` weight is stored as two planes and joined at load, so the disk tier
  (`--weights-disk-tier`) can't read it in place. In practice every expert must fit in **VRAM or the pinned host
  pool**: about **77 GiB a rank** at `--tp 2`, including the MTP layer.
  - Measured at `--tp 2` on two 32 GB cards: after the BF16 trunk, each card holds about 21 GiB of experts (with a
    short context and `--mm-max-patches 0`). The pinned host pool then needs about **56 GiB a rank (about 112 GiB in
    total)**, plus about 9 GiB of file buffers and headroom. Plan on **128 GB of system RAM** with two 32 GB cards.
    Decode then reads about three quarters of the experts from RAM over PCIe, so expect only a few tok/s.
  - Tested: it does **not** start on two 32 GB cards with 60 GB of system RAM. The placement report lists the
    `gate_up` experts as "DID NOT FIT".
  - `--host-pool-mib` is per rank, not a total.
- **Four 32 GB cards (`--tp 4`) is the faster setup:** about 38 GiB of experts a rank, mostly in VRAM.
- More VRAM (two 48 GB cards or larger) lowers the host-pool requirement accordingly.

## Example

```
radiance --model glm-5.3-flash-mxfp4.rad --tp 2 \
  --placement expert_tiered --host-pool-mib 57344 --mm-max-patches 0 \
  --kv-cache-dtype bf16 --max-num-seqs 4 --max-model-len 32768 \
  --served-model-name GLM-5.3-Flash --host :: --port 8000
```

`--p2p` defaults to `auto`, which asks the driver and uses peer-to-peer copies between the cards when they support
it. Add `--p2p off` if your system doesn't support P2P, or if `off` measures faster on your system. With `off`, the
cards exchange data through pinned host memory. P2P is disabled on the two R9700s this was converted on, so they
use `--p2p off`.

## License

MIT, the same as the base model. Converted by eniga; all credit for the model goes to Z.ai.
