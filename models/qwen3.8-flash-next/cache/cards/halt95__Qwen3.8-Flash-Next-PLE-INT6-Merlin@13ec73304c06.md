---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
language:
  - en
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: vllm
tags:
  - qwen3.8
  - flash-next
  - merlin
  - ple
  - per-layer-embedding
  - ngram-embedding
  - int6
  - host-ram
  - long-context
  - rtx-3090
  - ampere
  - sm86
---

<div align="center">

<img src="https://huggingface.co/halt95/Qwen3.8-Flash-Next-PLE-INT6-Merlin/resolve/main/assets/hero.png" alt="Flash-Next PLE INT6 pack, powered by Merlin: the n-gram table at 6 bits for Flash-Next and Swift1.5 on Merlin 3.0.1 or later. 38.74 GiB of host RAM instead of 47.68 GiB for FP8, 8.94 GiB less per server; KL divergence from FP8 at most 0.0033 and greedy agreement with FP8 at least 98.4 % on both models." width="100%">

<p>
<a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next"><img src="https://img.shields.io/badge/base-Qwen3.8--Flash--Next-615ced" alt="Base model: Qwen3.8-Flash-Next"></a>
<a href="#the-checkpoint"><img src="https://img.shields.io/badge/quant-INT6_g32-3987e5" alt="Quant: INT6, groups of 32"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/works_with-Flash--Next_%C2%B7_Swift1.5-ffb000" alt="Works with Flash-Next and Swift1.5"></a>
<a href="https://github.com/halt95/merlin-engine"><img src="https://img.shields.io/badge/runs_on-Merlin_3.0.1-a78bfa" alt="Runs on Merlin 3.0.1"></a>
<a href="https://huggingface.co/halt95/Qwen3.8-Flash-Next-PLE-INT6-Merlin/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-Qwen_Community_1.0-lightgrey" alt="Licence: Qwen Community License 1.0"></a>
</p>

<b><a href="#the-checkpoint">The checkpoint</a> · <a href="#quality">Quality</a> · <a href="#what-it-fits-on">What it fits on</a> · <a href="#use-it">Use it</a> · <a href="https://github.com/halt95/merlin-engine/blob/main/docs/CHANGELOG.md">Changelog</a></b>

</div>

# Qwen3.8-Flash-Next-PLE-INT6-Merlin

An INT6 copy of the per-layer embedding (PLE) n-gram table of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), for the [Merlin engine](https://github.com/halt95/merlin-engine)'s `--ple int6`. Merlin keeps this 51B-parameter table in host RAM. At 6 bits it takes 38.74 GiB instead of the 47.68 GiB of the FP8 table in the checkpoint, so each server needs about 9 GiB less RAM, with output close to FP8 (see [Quality](#quality)). This is not a full checkpoint: Merlin lays it over the [Flash-Next](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) or [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) checkpoint you already have, and never changes those files.

<sub>numbers: host RAM is the size of the table Merlin loads (INT6: the eight files in this repository; FP8: the table in the checkpoint). KL and agreement: each model's INT6 against its own FP8 on the same engine and prompts, Merlin 3.0.1 on 4× RTX 3090 on 2026-11-01 (Flash-Next two runs, Swift1.5 one). Each figure is also in a table below.</sub>

<a href="https://ko-fi.com/halt95"><img src="https://storage.ko-fi.com/cdn/kofi3.png?v=3" alt="Buy me a coffee on Ko-fi" height="40"></a>

## The checkpoint

| Component | Format |
|---|---|
| PLE n-gram table (128 shards of 2,500,012 × 160) | INT6, symmetric, groups of 32 with one FP16 scale per group (absmax / 31); four codes in three bytes, 130 bytes per row |
| Everything else: experts, attention, MTP head, vision, tokenizer, KV scales | Not in this pack: taken unchanged from your FP8 checkpoint |

**Files in this repository**

| Files | What they hold | Size |
|---|---|---|
| `model-pleint6-0000N-of-00008.safetensors` (8) | The INT6 codes and FP16 scales of the n-gram table | 38.74&nbsp;GiB |
| `ple-int6-pack.json` | Pack format, the tensor map, and the FP8 files it replaces (`model-plefp8-*`) | 34&nbsp;KiB |
| `ple-int6-pack.sha256` | SHA-256 of the nine files above | 1&nbsp;KiB |
| `LICENSE`, `NOTICE` | Qwen Community License 1.0, and what was changed | 4&nbsp;KiB |

<sub>41,600,199,680 bytes (38.74 GiB) of table in all. Built from the BF16 table of Qwen/Qwen3.8-Flash-Next at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` by the engine's `models/flash-next/make-ple-int6-pack.py --from-bf16`; the build is deterministic. Merlin builds `<checkpoint>-int6` beside your checkpoint from links plus a generated config and index, so the FP8 files stay as they are and `--ple fp8` keeps working.</sub>

## Quality

**Each model's INT6 table against its own FP8 table**, measured on the Merlin engine:

| Check | Flash-Next | Swift1.5 |
|---|---|---|
| KL divergence from FP8 (teacher-forced, top-20) | 0.0031–0.0033 | 0.0028 |
| Greedy token agreement with FP8 | 98.4–98.8 % | 98.7 % |
| GSM8K, 64 questions | within 1 of FP8 | equal to FP8 |
| MTP draft acceptance | +0.095 against FP8 | −0.03 against FP8 |
| Needle at 262K | found | found |

<sub>Merlin 3.0.1 on 4× RTX 3090 on 2026-11-01 (Flash-Next: two runs; Swift1.5: one run), the same prompts for both tables. A control with every code negated gives a KL of 5.46, so the check does see a damaged table. These are our own measurements on our hardware, not an independent evaluation. Swift1.5-Huihui did not pass: its MTP draft acceptance fell 0.15–0.17 below FP8 against a 0.10 bar, so Merlin refuses `--ple int6` for it.</sub>

## What it fits on

| Checkpoint | With this pack |
|---|---|
| [Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) | 🟢 Supported |
| [Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | 🟢 Supported |
| [Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Refused; serve it with `--ple fp8` |

<sub>All three checkpoints ship the same FP8 n-gram table, byte for byte, so one pack serves every supported one. The pack changes only host RAM: the KV pool, sessions held at once and GPU profiles are the same as with FP8.</sub>

| Need | Details |
|---|---|
| Host RAM | 38.74 GiB for the table instead of 47.68 GiB. With FP8, Flash-Next on four cards needs 72 GiB (tested minimum) and is refused below about 69 GiB, plus the KV RAM cache size when it is on. With INT6 that floor is about 60 GiB, an estimate not yet boot-tested on a smaller host. 96 GB is the reference configuration. |
| Disk | About 41.6 GB for the pack, on top of the FP8 checkpoint, which stays in place |
| Engine | Merlin 3.0.1 or later. Earlier releases have no `--ple` option. |

## Use it

```bash
venv/bin/merlin pull flash-next --ple int6   # fetch, verify, build the overlay
venv/bin/merlin serve flash-next ~/models/flash-next-int6 --ple int6
```

Use `swift1.5` in place of `flash-next` for Swift1.5. The pull builds the overlay folder `<checkpoint>-int6` beside your checkpoint (`~/models/flash-next-int6` with the default `~/models`), and `serve` takes that folder. `./setup.sh` offers the same choice. `merlin pull` checks every file against `ple-int6-pack.sha256`; after a download by hand, run `sha256sum -c ple-int6-pack.sha256` inside the folder. FP8 stays the default, and `--ple fp8` goes back to it at any time.

> [!IMPORTANT]
> This pack is not a model on its own. It needs Merlin 3.0.1 or later and the Flash-Next or Swift1.5 checkpoint; stock vLLM does not read it.

## Good to know

- The quality figures come from one setup: 4× RTX 3090 at 220 W with P2P. The table lives in host RAM, so other GPUs are expected to behave the same, but they were not measured with INT6.
- Swift1.5-Huihui is refused for now, not broken: a recalibrated acceptance bar is planned for a later release.
- Decode and prefill speed were not measured separately with INT6; MTP draft acceptance, which drives decode speed, is in the Quality table. The 3.0.1 bench cards use the FP8 default.
- The quality checks ran before the GPU unlock was installed, with P2P only. Merlin's 3.0.1 bench numbers are measured with [Zagitta's gpu-unlock](https://github.com/Zagitta/gpu-unlock) plus P2P; the unlock changes speed, not output, so these results hold with or without it.

## Changes

| Date | Change to this repository |
|---|---|
| 2026-11-01 | Pack published |

<sub>Card-only edits are not listed. Engine releases: the engine's [CHANGELOG](https://github.com/halt95/merlin-engine/blob/main/docs/CHANGELOG.md).</sub>

## Credits and licence

Qwen (the model and its BF16 n-gram table; Qwen Community License 1.0, see `LICENSE` and `NOTICE`; its conditions apply, including the commercial-scale and model-as-a-service terms), RadixArk (the FP8 table this pack sits beside), and the vLLM community's PLE-offload work. The INT6 pack is by halt95.
