---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- saber
- moe
- int4
- int8
- fp8
- quantized
- rocm
- gfx1201
- radeon-ai-pro-r9700
---

# Qwen3.8-Flash-Next MACaQ (Saber served pack)

This repository is a **derivative work** of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
distributed under the Qwen Community License 1.0 (the [LICENSE](LICENSE) file in this repository, included verbatim).
Most non-expert weights descend from AMD's Quark MXFP4 build,
[amd/Qwen3.8-Flash-Next-Quark-MXFP4](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4). That includes the
8-bit trunk records (re-quantized from that checkpoint's bf16 trunk), the MTP drafter's MXFP4 expert slab and the
records the engine reads directly from that checkpoint (see "Using it"). It is also the source of the MXFP4 expert
lineage this pack replaces. The MACaQ routed and shared experts were quantized from the bf16 expert weights of
Qwen/Qwen3.8-Flash-Next. The alternate FP8 MTP-expert records (`mtp_fp8.*`) hold bytes from Qwen's first-party FP8
release of the same model.

It is the **pack served by the Saber engine** (a native ROCm engine for AMD RDNA4 GPUs) for this model.
It is **not** a safetensors, GGUF or other standard checkpoint. It does **not** load in Hugging Face libraries or in any
other inference engine. The Saber engine is not public. This upload exists so the exact served bytes are archived and
reproducible. It is not a drop-in model for other runtimes.

## What is in the pack

The pack is laid out for tensor parallelism across two GPUs (TP=2). It has one `rank0/` and one `rank1/` directory, and
each rank holds half of every expert.

| part | form | files |
|---|---|---|
| routed experts (48 layers x 512) | **MACaQ int4**: per-output-row int4 codes + an f32 row scale, a per-input-channel scale `t` applied before a 32-column Walsh-Hadamard transform, int8 per-token activations (W4A8), integer accumulation | `rank<r>/experts.macaq.xtier` |
| gate_up `t` | ONE scale vector per layer, shared by every routed expert and by the shared expert ("shared-t" form) | `rank<r>/experts.tsh` (+ `.json` with its sha256) |
| shared expert | MACaQ int4 (gate_up on the layer-shared `t`, its own `t` on down) | `rank<r>/experts.shq.xtier` |
| trunk (attention / GDN mixers, LM head) | 8-bit (fp8 W8A8 records) | `rank<r>/trunk8.*`, `rank<r>/dense.*` |
| hyper-connection mixing matrices | fp8 | `shared/hc8.*` |
| shared dense records | bf16 | `shared/dense.*` |
| MTP drafter (speculative decoding) | the MTP layer: dense records + its 512 experts as a 4-bit (MXFP4) slab, W4A8; the default | `rank<r>/mtp_x4.*` |
| MTP experts, alternate form | FP8 block-scaled (not opened by the default boot) | `rank<r>/mtp_fp8.*` |
| vision tower | bf16, one copy a rank | `vision/` |
| plan / manifest / aliases / KV scales / expert-placement profile | JSON | `manifest.json`, `plan.json`, `rank<r>/aliases.json`, `kv_scales.json`, `profile/` |
| MACaQ build record | JSON: the builder plans, shared-expert stats and a member manifest with sha256 | `macaq/` |
The KV cache is fp8 at serve time (`kv_scales.json` holds the calibrated scales).

## How it was made (high level)

- **Calibration.** Router-weighted activations were collected per layer. The Hessian (diagonal for the clip search, full
  for refinement) of each expert's GEMMs was built from the activations that expert receives, weighted by its router weight.
- **Scale search.** For each expert, the per-row scale was chosen by a Hessian-diagonal clip search over 29 clip
  candidates, under the per-input-channel `t` (geometric mean 1). For gate_up, the `t` is shared across a layer and is
  computed over every routed expert's gate_up rows plus the shared expert's.
- **Greedy code flips.** Each expert's int4 codes were then refined with K = 512 greedy code flips against the full
  router-weighted Hessian (ridge 0.03).
- **Verification.** The tiers were packed into the engine's record format and checked by an independent decoder
  against the builder's output, with negative controls.

## Results (our gates; 2x AMD Radeon AI PRO R9700)

All numbers come from the maintainers' own gates, on 2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB each) at TP=2 with the
cards capped at 210 W. They are measured against the same engine serving the AMD Quark **MXFP4** experts (the control),
with everything else identical.

| metric | this pack (MACaQ experts) | MXFP4 experts (control) |
|---|---|---|
| KL divergence vs the bf16 model (25 prompts, 885 positions) | **0.0517** [0.032, 0.072] | 0.1023 [0.058, 0.137] |
| paired KL change | **-49.4 %** [-60.0, -24.7], lower on 20/25 prompts | -- |
| top-1 agreement with bf16 | 0.942 | 0.920 |
| GSM8K (100 problems, thinking off) | 98 | 97 |
| seeded reproducibility (20 prompts x 2) | 20/20 identical | -- |
| 32k-token needle retrieval | pass | -- |
| decode tok/s, single stream (BetterBench `--quick`) | **197.6** | 182.2 |
| decode tok/s, 8 concurrent (BetterBench `--quick`) | **498.4** | 455.9 |
| TTFT p50 / p99 (ms) | **58 / 101** | 62 / 109 |
| update (inter-token) p99 (ms) | **16.9** | 18.6 |

Caveats:

- **KL.** The bracketed ranges are the maintainers' confidence intervals over the 25 prompts. The interval for the
  paired change is decided (it excludes 0), but 5 of the 25 prompts are worse.
- **Speed.** The speed rows come from 2 boots per arm in A B B A order on an idle host, with temperature 0.7. They depend
  on this engine, this hardware and this power cap.
- **GSM8K.** GSM8K is one 100-problem run per arm. A 1-point difference there is within noise.
- **Envelope gate (expected failure).** The engine's usual numerics gate (the "FN80d envelope") compares a candidate to a
  reference that already has the Quark MXFP4 experts. This pack **fails** that gate by construction, because it
  replaces those experts on purpose. The quality gate that governed the release is the bf16-anchored KL above.

## Hardware and software requirements

- 2x AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB VRAM each) at TP=2. The engine fills each card's VRAM with an expert
  tier (about 15,500 half-expert slots a rank). The rest of the experts live in a pinned host-RAM tier, and the drive is
  the last tier.
- Host RAM: the full 262,144-token context pins about 35 GiB of host RAM. Plan for 64 GB.
- Fast NVMe storage for the pack and the base checkpoint.
- The Saber engine (ROCm). The pack format is engine-specific.

## Using it

**The base checkpoint is also required.** The pack does not carry every byte the engine reads. Its manifest points into
the AMD Quark MXFP4 checkpoint for:

- `config.json`, `generation_config.json`, the chat template and the tokenizer;
- the checkpoint-identity records (for example the embedding rows);
- the per-layer-embedding (PLE) tables, about 102 GB in 33 of that checkpoint's shards, read from the drive at serve time.

An audited boot opened 83 of its 131 shards, so download the whole checkpoint at the revision the pack was built
against.

These references are **absolute paths**, and the loader uses them as they are written. Lay the files out like this:

```
/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4/        <- amd/Qwen3.8-Flash-Next-Quark-MXFP4 (full download)
/mnt/nvme/saber-packs/qwen38-flashnext-macaq-tp2-trunk8/ <- this repository
```

For example:

```
hf download amd/Qwen3.8-Flash-Next-Quark-MXFP4 --revision b2091c3809eb51902f46a163973362d8176f9e58 --local-dir /mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4
hf download Launch80/Qwen3.8-Flash-Next-MACaQ-SABER --local-dir /mnt/nvme/saber-packs/qwen38-flashnext-macaq-tp2-trunk8
```

If your storage lives elsewhere, a `/mnt/nvme` symlink (or bind mount) to it is enough. The pack was built against
checkpoint revision `b2091c3809eb51902f46a163973362d8176f9e58`, and `manifest.json` lists the header sha256 of every shard.
A later upstream revision changes only `config.json` metadata (`quantization_config`) and the README, and the shard bytes
are identical. The pinned revision is still the tested one.

Then start the Saber engine's Flash-Next launcher with the storage root pointed at that tree. Its defaults are this pack,
`--fn-experts tp2`, the expert format `auto` (which reads this pack as MACaQ), the 8-bit trunk with fp8 hyper-connection
mixers and head (`--fn-trunk w8-hc8-head`), fp8 KV, the MTP drafter and vision. It serves an OpenAI-compatible HTTP API (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

The JSON `source` / `source_pack` / `target` fields are build provenance only, and the engine does not read them. In this
upload they were rewritten from the build host's absolute paths to paths relative to the storage root (`models/...`,
`saber-packs/...`). The loader-read paths in `manifest.json` (`checkpoint.dir` and the PLE sources) and in `ple/ple_index.json` are
unchanged.

## File layout

| path | size |
|---|---|
| `LICENSE` | 3.23 kB |
| `experts_tp2.json` | 2.50 kB |
| `kv_scales.json` | 1.85 kB |
| `manifest.json` | 37.09 kB |
| `plan.json` | 1.85 MB |
| `macaq/MANIFEST.json` | 6.29 kB |
| `macaq/routed-plan.json` | 361 B |
| `macaq/shq-plan.json` | 467 B |
| `macaq/shq-stats.json` | 10.57 kB |
| `ple/ple_index.json` | 98.47 kB |
| `profile/lazy_runs.rank0.json` | 23.98 kB |
| `profile/lazy_runs.rank1.json` | 24.17 kB |
| `profile/route_profile.json` | 122.19 kB |
| `rank0/aliases.json` | 19.20 kB |
| `rank0/dense.json` | 450.07 kB |
| `rank0/dense.pack` | 311.30 kB |
| `rank0/experts.macaq.xtier` | 30.60 GB |
| `rank0/experts.shq.xtier` | 59.78 MB |
| `rank0/experts.tsh` | 491.52 kB |
| `rank0/experts.tsh.json` | 130 B |
| `rank0/mtp_fp8.json` | 299.40 kB |
| `rank0/mtp_fp8.pack` | 1.26 GB |
| `rank0/mtp_x4.json` | 405.10 kB |
| `rank0/mtp_x4.pack` | 672.07 MB |
| `rank0/trunk8.json` | 104.14 kB |
| `rank0/trunk8.pack` | 1.66 GB |
| `rank1/aliases.json` | 19.30 kB |
| `rank1/dense.json` | 450.68 kB |
| `rank1/dense.pack` | 311.30 kB |
| `rank1/experts.macaq.xtier` | 30.60 GB |
| `rank1/experts.shq.xtier` | 59.78 MB |
| `rank1/experts.tsh` | 491.52 kB |
| `rank1/experts.tsh.json` | 130 B |
| `rank1/mtp_fp8.json` | 299.84 kB |
| `rank1/mtp_fp8.pack` | 1.26 GB |
| `rank1/mtp_x4.json` | 404.16 kB |
| `rank1/mtp_x4.pack` | 669.45 MB |
| `rank1/trunk8.json` | 104.14 kB |
| `rank1/trunk8.pack` | 1.66 GB |
| `shared/dense.json` | 417.40 kB |
| `shared/dense.pack` | 125.89 MB |
| `shared/hc8.json` | 104.17 kB |
| `shared/hc8.pack` | 661.52 MB |
| `vision/rank0.pack` | 898.37 MB |
| `vision/rank1.pack` | 898.37 MB |
| `vision/vision.json` | 73.13 kB |
| **total (46 files + README.md + SHA256SUMS)** | **71.08 GB** |

`SHA256SUMS` lists the sha256 of every file in this repository except `README.md` and itself. Check a download with `sha256sum -c SHA256SUMS`.

## Known limitations

- **Two cards only.** The pack is TP=2 on gfx1201. Other GPU counts and other GPU architectures are not supported.
- **Long prompts.** Prefill chunks above 512 rows still run the older MACaQ GEMM route, so time to first token on long
  prompts is about the same as the MXFP4 control.
- **MXFP4 rollback.** the AMD Quark MXFP4 expert tiers (the engine's `--fn-experts-fmt mxfp4` rollback) are NOT included: the default boot never opens them (audited), and the rollback needs a separate MXFP4 pack.
- **Not standalone.** The pack depends on the AMD Quark MXFP4 base checkpoint being present at the path above.
- **Status.** This is a research artifact from a private engine. No support is provided.

## License

This repository is distributed under the [Qwen Community License 1.0](LICENSE): Copyright (c) 2026 Qwen. The license
text, including its copyright and permission notice, is included verbatim and applies to every file here.

This is a derivative of Qwen/Qwen3.8-Flash-Next. The license's conditions apply, including the user-interface naming
requirement for very large commercial deployments and the separate license required for Model-as-a-Service or AI Work
Assistant businesses.
