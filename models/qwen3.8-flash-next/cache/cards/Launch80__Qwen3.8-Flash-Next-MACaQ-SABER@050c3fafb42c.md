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

Qwen3.8-Flash-Next (a 512-expert MoE with vision) packed for the **[Saber engine](https://github.com/GGZ14/saber-engine)**,
a native ROCm inference engine for AMD RDNA4 GPUs (Apache-2.0). The routed and shared experts are in Saber's **MACaQ int4**
format (W4A8): KL to the bf16 model is half that of the AMD Quark MXFP4 experts, and decode is about 9 % faster
(see [Results](#results-our-gates-2x-amd-radeon-ai-pro-r9700)). It serves an OpenAI-compatible API with text and image input,
262,144-token context, MTP speculative decoding and up to 8 concurrent requests on **two Radeon AI PRO R9700** cards.

This is **not** a safetensors / GGUF checkpoint: it loads only in the Saber engine, not in transformers, vLLM or llama.cpp.

> **Engine version.** The public engine ([GGZ14/saber-engine](https://github.com/GGZ14/saber-engine)) boots this pack
> since 2026-11-24 (pull request #11 merged, `main` `1da27661`): `tools/fetch_packs.sh flashnext` (alias
> `flashnext-macaq`) downloads it, and `tools/serve_flashnext.sh` serves it by default (the expert format `auto` reads
> this pack as `macaq`; the boot prints `fnxfmt ... auto -> macaq` and `fnpaths:`). It is served by that launcher, not by
> the `saber models` / `saber serve --model` registry, whose entries are the dense 27B packs (`models.toml` lists this
> pack in a comment with its fetch and serve commands).

## Quick start

**You need:**

| | |
|---|---|
| GPUs | 2x AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB each), TP=2. Other GPUs are not supported. |
| OS | Linux with the ROCm kernel driver (`/dev/kfd`, `/dev/dri`) and podman or docker (all builds and GPU work run in the engine's container) |
| Host RAM | 64 GB (the expert RAM tier + the 262k-token KV slab pin about 35 GiB) |
| Disk | **210.6 GB** on one fast NVMe: this pack (71.1 GB) + 83 of the 131 shards of the AMD base checkpoint (139.5 GB). The full base checkpoint is 182.3 GB, and the fetch skips the 48 shards that are never opened (42.8 GB less). |

**1. Get the engine and build its container** (once):

```bash
git clone https://github.com/GGZ14/saber-engine && cd saber-engine
tools/build-image.sh
pip install -U "huggingface_hub[cli]"
```

**2. Download** (resumable; it checks free space first). Pick any directory as the storage root:

```bash
SABER_NVME=/data/saber tools/fetch_packs.sh flashnext-macaq
```

This creates

```
/data/saber/saber-packs/qwen38-flashnext-macaq-tp2-trunk8/   <- this repository
/data/saber/models/Qwen3.8-Flash-Next-Quark-MXFP4/           <- amd/Qwen3.8-Flash-Next-Quark-MXFP4 @ b2091c38, 83 shards + small files
```

Without the script, these two commands are equivalent. The second downloads the whole checkpoint (182.3 GB):

```bash
hf download Launch80/Qwen3.8-Flash-Next-MACaQ-SABER --local-dir /data/saber/saber-packs/qwen38-flashnext-macaq-tp2-trunk8
hf download amd/Qwen3.8-Flash-Next-Quark-MXFP4 --revision b2091c3809eb51902f46a163973362d8176f9e58 \
    --local-dir /data/saber/models/Qwen3.8-Flash-Next-Quark-MXFP4
```

**3. Serve** on port 8080. The first run builds the engine (a few minutes), and the boot then takes 1-3 minutes:

```bash
SABER_NVME=/data/saber tools/serve_flashnext.sh
```

**4. Ask it something:**

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 512
}'
```

Thinking is on by default, so the reasoning arrives in `message.reasoning` and the answer in `message.content`. Add
`"chat_template_kwargs": {"enable_thinking": false}` to a request to skip the reasoning, or start the server with
`--fn-thinking off`. Images go in as OpenAI `image_url` content parts. `/v1/models` and `/v1/completions` are also served.

**5. Stop:** `tools/serve_flashnext.sh stop`. Logs: `podman logs -f saber-flashnext-serve`.

**Without the GPUs or the engine** you can download and inspect the files (the JSON tables, the `base/` config and
tokenizer, the checksums), but nothing here runs a forward pass. The formats are specific to the engine.

### How the engine finds the files

The pack's `manifest.json` and `ple/ple_index.json` record the maintainers' checkpoint path
(`/mnt/nvme/models/Qwen3.8-Flash-Next-Quark-MXFP4`). When that path does not exist on your machine, the engine rebases it
under `$SABER_NVME`, or under the directory that holds `saber-packs/`. It looks for `<root>/models/Qwen3.8-Flash-Next-Quark-MXFP4`
(then `<root>/Qwen3.8-Flash-Next-Quark-MXFP4`). `--fn-ckpt-dir DIR` names the checkpoint directly. The tokenizer, chat
template, `config.json` and `generation_config.json` are read from this pack's `base/` directory. The boot prints what it
resolved, for example
`fnpaths: checkpoint /data/saber/models/Qwen3.8-Flash-Next-Quark-MXFP4 (rebased under the storage root), model files from the pack's .../base`.

The checkpoint is still needed for the rest. The pack reads the embedding, norm, router and attention records and the
102 GB per-layer-embedding (PLE) table in place from its shards, at the byte offsets `manifest.json` lists with the
header sha256 of every shard. The checkpoint revision `b2091c38` is the tested one. A later upstream revision differs
only in `config.json` metadata and the README.

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

## Hardware notes

- The engine fills each card's VRAM with an expert tier (about 15,500 half-expert slots a rank). The rest of the
  experts live in a pinned host-RAM tier, and the drive is the last tier.
- The cards in the results below were capped at 210 W. The engine never changes the power cap.

## File layout

| path | size |
|---|---|
| `LICENSE` | 3.23 kB |
| `base/` (config, generation config, chat template, tokenizer, vocab/merges, preprocessor configs, NOTICE) | 22.98 MB |
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
| **total (56 files + README.md + SHA256SUMS)** | **71.11 GB** |

`SHA256SUMS` lists the sha256 of every file in this repository except `README.md` and itself. Check a download with `sha256sum -c SHA256SUMS`.

## Known limitations

- **Two cards only.** The pack is TP=2 on gfx1201. Other GPU counts and other GPU architectures are not supported.
- **Long prompts.** Prefill chunks above 512 rows still run the older MACaQ GEMM route, so time to first token on long
  prompts is about the same as the MXFP4 control.
- **MXFP4 rollback.** The AMD Quark MXFP4 expert tiers (the engine's `--fn-experts-fmt mxfp4` rollback) are NOT included.
  The default boot never opens them (audited), and the rollback needs a separate MXFP4 pack.
- **Not standalone.** The pack needs the 83 base-checkpoint shards next to it (step 2).
- **Status.** This is a research artifact. No support is provided.

## Provenance

This repository is a **derivative work** of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Most non-expert weights descend from AMD's Quark MXFP4 build,
[amd/Qwen3.8-Flash-Next-Quark-MXFP4](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4). That includes the
8-bit trunk records (re-quantized from that checkpoint's bf16 trunk), the MTP drafter's MXFP4 expert slab and the
records the engine reads directly from that checkpoint. The MACaQ routed and shared experts were quantized from the
bf16 expert weights of Qwen/Qwen3.8-Flash-Next. The alternate FP8 MTP-expert records (`mtp_fp8.*`) hold bytes from
Qwen's first-party FP8 release of the same model. `base/` holds verbatim copies of the AMD checkpoint's small
config/tokenizer files (see `base/NOTICE`).

The JSON `source` / `source_pack` / `target` fields are build provenance only, and the engine does not read them. They
were rewritten from the build host's absolute paths to storage-root-relative paths.

## License

This repository is distributed under the [Qwen Community License 1.0](LICENSE): Copyright (c) 2026 Qwen. The license
text, including its copyright and permission notice, is included verbatim and applies to every file here.

This is a derivative of Qwen/Qwen3.8-Flash-Next. The license's conditions apply, including the user-interface naming
requirement for very large commercial deployments and the separate license required for Model-as-a-Service or AI Work
Assistant businesses.
