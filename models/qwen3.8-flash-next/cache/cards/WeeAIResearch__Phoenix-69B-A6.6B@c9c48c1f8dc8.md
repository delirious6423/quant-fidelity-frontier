---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - lmstudio-community/Qwen3.8-Flash-Next-GGUF
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - moe
  - code
  - pruning
  - distillation
  - mixed-precision
  - experimental
---

![Phoenix: Built to rise. 40% smaller GGUF, 10 active experts per layer, 48 layers.](assets/phoenix-banner.png)

# Phoenix-69B-A6.6B

## Built to rise. Built for local coding.

**Phoenix brings restored coding capability, a smaller footprint, and a purpose-built CPU inference workflow into one downloadable checkpoint.**

Built from Qwen3.8-Flash-Next, Phoenix combines expert pruning, mixed-precision quantization and teacher-guided recovery. The result is a **71.34 GB GGUF**, down from the original Q4 model's **119.15 GB**, with **all 48 layers** and **10 routed experts selected per layer, per token**.

That is **47.82 GB less model storage**. The v21 recovery step restored **3,074 missing expert slots**, upgraded **1,008 retained experts to Q4**, and took a matched TypeScript development task from **3/12 to 12/12 executable checks**.

**[Download Phoenix GGUF](https://huggingface.co/WeeAIResearch/Phoenix-69B-A6.6B/resolve/main/Phoenix-69B-A6.6B.gguf?download=true)** · **[Build and run](runtime/README.md)** · **[Inspect the results](evaluation/results.txt)**

## Why Phoenix deserves a place in your local AI workshop

**Make room for a large coding model.** Phoenix removes **40.1% of the source GGUF's disk footprint** and **59.4% of its routed expert storage**. The complete learned lookup table stays in the release, available through disk-backed memory mapping.

**Put precision where recovery called for it.** Q4, Q3, Q2 and Q1 expert tiers sit alongside selected higher-precision weights and a Q6 output projection. Expert restoration, precision upgrades and earlier distillation work are merged into the checkpoint: **no separate distillation adapter to load**.

**Run the model locally.** The included llama.cpp CPU patch fetches the exact expert ranges selected by routing. A small local expert map connects router IDs to their weight locations. The package includes the patch, build instructions, coding prompt and map generator so you can build the same inference path.

**See what you are getting.** Prompts, generated answers, executable-check results, known failures, parameter accounting and model hashes are published with the weights. Phoenix is for builders who want to run, inspect and keep improving a local coding model.

![Original Q4 versus Phoenix storage: 119.15 GB versus 71.34 GB, with the same lookup table.](assets/storage-comparison.png)

## Restored capacity. Measured coding recovery.

The clearest matched win is the TypeScript interval-intersection task: **3/12 checks in the previous v20 checkpoint became 12/12 in Phoenix v21**, using the same prompt, settings and runtime.

![Matched TypeScript interval-intersection development checks: previous v20 3/12, Phoenix v21 12/12.](assets/coding-recovery.png)

Phoenix also passed **10/10 original checks for the Go bcrypt CLI**, **10/10 for the Python transfer CLI**, and **10/10 for the TypeScript graph CLI with the coding-contract instruction profile**. All five programs in the generic-profile run completed and compiled.

Here is the full picture, with the two instruction profiles kept separate:

| Development task | Generic instructions | Coding-contract instructions |
|---|---:|---:|
| C# statistics CLI | 4/10 | 4/10 |
| Go bcrypt CLI | **10/10** | **10/10** |
| Python transfer CLI | **10/10** | Not rerun |
| TypeScript graph CLI | 0/10 | **10/10** |
| TypeScript interval intersection | **12/12** | Not rerun |
| **Total for each recorded profile** | **36/52** | **24/30** |

These are **development-task results**, including partial-reference calibration, rather than an untouched public benchmark. The contract-profile column covers only three rerun tasks; its total is not combined with the generic suite. The TypeScript graph result changed with the instruction profile, using the same Phoenix weights.

The remaining failures are specific and documented: C# produced the wrong error-output format, and Go failed two additional oversized-trial probes under both profiles. The generic graph program expected a filename while the harness supplied stdin. Broad chat quality and equivalence to the original Qwen model have not been established.

**Explore the evidence:** [readable report](evaluation/results.txt) · [structured results](evaluation/results.json) · [prompts, answers and assessments](evaluation/).

## The numbers behind the name

| Specification | Phoenix |
|---|---:|
| Model family label | **69B-A6.6B** |
| Exact logical parameters excluding lookup | **68.05B** |
| Estimated active non-lookup parameters per token | **~6.64B** |
| Retained learned lookup table | **51.20B values / 35.20 GB** |
| Logical total including lookup | **119.25B** |
| Transformer layers | **48** |
| Active routed experts per layer, per token | **10**, plus the shared expert |
| Routed expert weight storage | **32.83 GB** |
| Complete GGUF | **71.34 GB / 66.44 GiB** |
| Published checkpoint | **CodingRouteRestore v21** |

The **69B** family label accompanies an exact non-lookup count of **68,048,493,440**. Including the lookup table gives **119,248,739,200** learned values. Quantization-alignment padding is excluded. The active count is a shape-based estimate that excludes full embedding/lookup tables and masked dummy work; it is not an execution audit or a measure of intelligence. [Exact accounting](parameter-count.json).

For comparison, the original Q4_K_M model contains **125.74B non-lookup parameters**, **176.94B including lookup**, and an estimated **6.67B active non-lookup parameters per token**. Both models select ten routed experts per layer. Pruning changes the available expert pool; it does not imply a proportional reduction in per-token work.

## Get Phoenix running

Download the release:

```bash
hf download WeeAIResearch/Phoenix-69B-A6.6B --local-dir Phoenix-69B-A6.6B
cd Phoenix-69B-A6.6B
```

Then follow the **[CPU runtime quickstart](runtime/README.md)** to build the pinned llama.cpp revision, apply the supplied patch, generate your local expert map and launch chat. The tested [coding instructions](coding-system.txt) are included.

**The patched runtime is required for the documented setup.** Compatibility with unmodified llama.cpp or other frontends has not been established for this compact mixed-tier layout.

The lookup table gathers small rows as needed through memory mapping. Available RAM, file-cache state and disk speed affect performance. The local development launchers' **20.5 GiB watchdog ceiling is not a fully resident model-size claim**. The direct quickstart command does not install that watchdog. Recent additional scheduling and predictive-read experiments did not establish a throughput gain over the released CPU-prefetch profile and are excluded.

## A release you can inspect and build on

This is the same v21 model payload used by the local CPU-prefetch chat, now carrying the Phoenix name. The internal historical GGUF name is retained for provenance. Its programming-restoration step preserved layers 0-2 and all existing Q4 payloads while adding the restored slots and promotions described above. Earlier shared-expert training, routing calibration and teacher-guided projection recovery form part of its development history.

**[Model manifest](model-manifest.json)** · **[Runtime source identity](runtime/source-identity.json)** · **[Checksums](SHA256SUMS)**

Model SHA-256:

```text
ae373161cf2549917faa4772dcc8956d561de2671dcd0e1f77582daeb45614cc
```

## Origin and license

Phoenix is an experimental derivative by **WeeAIResearch**, built from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) through the [lmstudio-community Q4_K_M conversion](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF). The recovery teacher is the original Q4 model; no equivalence to that model or to full-precision weights is claimed. Phoenix is not an official Qwen release.

The weights retain the **[Qwen Community License 1.0](LICENSE)**, including its commercial-use conditions. The [llama.cpp MIT license](runtime/LLAMA_CPP_LICENSE) and [MIT prefetch attribution](runtime/PREFETCH_LICENSE) are included.

**Download Phoenix. Run it locally. Inspect the recovery. Build what comes next.**
