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
  - local-inference
  - experimental
---

[![Phoenix — Built to rise. 69B family, approximately 6.6B active parameters. WeeAI Research.](assets/phoenix-banner.png)](https://weeai.co.uk/)

# Phoenix 69B · A6.6B

## Bring your next idea home.

**Big-model architecture. A lighter footprint. Built to run locally. Phoenix brings expert restoration, focused training and a purpose-built CPU runtime into one downloadable coding model.**

Start with a **71.34 GB GGUF**, **48 layers** and **10 routed experts active per layer**. Keep the complete learned lookup table. Take **47.82 GB off the original Q4 download**. Then build with weights you can run on your own machine and results you can inspect.

The latest release adds a focused repair to the final shared expert. The reported spigot request now finishes with correct Python, checked through **1,000 decimal places of pi**. Graph and integer-square-root programs also pass their executable checks, and the recorded chat answer completes coherently. **The file size and quantization stay the same.**

**[Meet WeeAI Research → WeeAI.co.uk](https://weeai.co.uk/)** · **[Download Phoenix](https://huggingface.co/WeeAIResearch/Phoenix-69B-A6.6B/resolve/main/Phoenix-69B-A6.6B.gguf?download=true)** · **[Build and run](runtime/README.md)**

Phoenix is an experimental release. The documented setup requires the supplied llama.cpp patch; the five development cases below do not establish broad coding quality. The alternate pi-generator task still fails correctness.

## Ambitious architecture. A smaller download.

Phoenix keeps the full 48-layer architecture while reducing the original Q4 source file by **40.1%** and routed expert storage by **59.4%**. Its 51.20B-value lookup table remains intact, with rows accessed through disk-backed memory mapping.

![Complete GGUF storage: original Q4 119.15 GB, Phoenix 71.34 GB. A 40.1 percent reduction in disk footprint.](assets/storage-comparison.png)

**Build locally.** After downloading the weights and building the runtime, inference runs on your machine without a hosted model API.

**Put the work into the weights.** Expert restoration, mixed-precision storage, earlier teacher-guided recovery and the latest selective shared-expert repair are merged into the GGUF. There is no separate repair adapter to load.

**Use the experts routing actually selects.** The included CPU runtime maps selected expert IDs to weight ranges and prefetches those ranges. The expert map is generated locally after download; it does not replace the model's routing decisions.

**Follow the evidence.** Download the prompts, unedited generated answers, executable assessments, runtime patch and checksums alongside the model.

## A focused repair. A visible result.

The new repair teaches a checked spigot implementation through **one 1.74 MB Q8_0 shared-down projection in layer 47**. Updates are projected away from captured background coding and chat feature directions to limit interference. Every GGUF byte outside that tensor was verified identical to the preceding Phoenix v21 checkpoint.

An earlier candidate fixed the trained prompt but broke the graph task. It was rejected. The released selective candidate keeps the spigot repair **and passes the graph task's 66 executable checks**.

![Selected passing development checks: spigot, graph, integer square root and chat. Full results are listed below.](assets/repair-results.png)

| Latest release case | Recorded outcome |
|---|---|
| Reported Python spigot function | **8/8 checks**, through 1,000 decimal places; completed naturally |
| Python Dijkstra implementation | **66/66 checks**, including disconnected invalid edges, zero weights and unreachable nodes |
| Python integer square root | **96/96 checks**, including 1,000-bit inputs and negative arguments |
| Three-sentence hash-table explanation | **Completed coherently; manual review passed** |
| Alternate Python pi-digit generator | Completed without repeated drafting, but **failed execution correctness**; CPU limit reached |

These are **five development cases**, not a public benchmark. The reported spigot prompt was used in training. The other fixtures began as fresh checks, then informed selection after the earlier candidate failed. Generated code was executed without edits; chat was manually reviewed. Broad chat, multi-turn behavior, other programming languages and long-context quality have not been re-established for these new weights. Repetition can still occur.

The alternate generator's digit-emission condition never becomes true, so it does not yield the required digits. This is a concrete remaining coding failure, despite the response finishing. Phoenix's recovery is work you can inspect, with further improvement still to do.

**[Latest evaluation report](evaluation/repetition-repair-v4/report.json)** · **[Prompts, answers and assessments](evaluation/repetition-repair-v4/)** · **[Repair method and fitting limits](evaluation/repetition-repair-v4/README.md)**

## The numbers behind Phoenix

| Specification | This release |
|---|---:|
| Family name | **Phoenix 69B · A6.6B** |
| Exact logical parameters excluding lookup | **68,048,493,440 / 68.05B** |
| Estimated active non-lookup parameters per token | **~6.64B** |
| Retained lookup table | **51.20B values / 35.20 GB** |
| Logical parameters including lookup | **119.25B** |
| Transformer layers | **48** |
| Routed experts selected per layer, per token | **10**, plus the shared expert |
| Routed expert storage | **32.83 GB** |
| Complete GGUF | **71.34 GB / 66.44 GiB** |
| Checkpoint | **Phoenix v21 + selective repetition repair r4** |

The **69B** name is the family label. The exact non-lookup count is **68.05B**; including the learned lookup table gives **119,248,739,200** values. The active count is a shape-based estimate, excluding full embedding/lookup tables and masked dummy work. It is not an execution audit or a quality score. [Exact accounting](parameter-count.json).

The original Q4 source has **125.74B non-lookup parameters**, **176.94B including lookup**, an estimated **6.67B active non-lookup parameters per token**, and a **119.15 GB** file. Storage and parameter reductions do not imply proportional speed gains or equivalence to the original model.

## Expert layout

![Phoenix routed experts by quant group: Q4 8,410; Q4 with Q8 down 1,392; Q3 745; Q2 1,286; Q1 1,011.](assets/quant-layout.png)

| Quant group | Routed expert slots | Share |
|---|---:|---:|
| Q4 | **8,410** | 65.5% |
| Q4 gate/up + Q8 down | **1,392** | 10.8% |
| Q3 | **745** | 5.8% |
| Q2 | **1,286** | 10.0% |
| Q1 | **1,011** | 7.9% |
| **Total routed** | **12,844** | **100%** |

Each expert is counted once. The hybrid Q4/Q8 group uses Q4_K gate/up projections and a Q8_0 down projection; these are not fully Q8 experts. The Q4 group uses Q4_K gate/up and Q4_0 down. The remaining groups use Q3_K, Q2_K and Q1_0 respectively.

Phoenix also retains **48 shared experts**, counted separately: **24** with Q4 gate/up and Q5 down, **21** with Q4 gate/up and Q8 down, and **3** fully Q8.

![Stored routed expert count: original Q4 source 24,576 versus Phoenix 12,844, a 47.7 percent reduction.](assets/expert-count-comparison.png)

| Expert count across all 48 layers | Original Q4 source | Phoenix |
|---|---:|---:|
| Routed expert slots | **24,576** | **12,844** |
| Shared experts | **48** | **48** |
| **Routed + shared** | **24,624** | **12,892** |

Phoenix has **11,732 fewer routed expert slots**, a **47.7% reduction**. The original has 512 routed experts in every layer; Phoenix averages 267.6, with a different count in each layer. Both select **10 routed experts per layer per token**, plus the shared expert. These graphs count stored expert slots across layers, rather than parameters or activation frequency.

Counts and exact quant formats were audited from both models' GGUF tensor headers. [Full layout and per-layer counts](assets/expert-layout.json) · [CSV audit](assets/expert-layout.csv).

### Layer by layer, in six groups of eight

Each bar shows one layer's routed experts, split by quant group. The six panels cover **0–7, 8–15, 16–23, 24–31, 32–39 and 40–47**, following the GGUF's zero-based layer IDs. Every panel uses the same scale; the dashed line marks the original Q4 model's **512 routed experts per layer**. One additional shared expert per layer is counted separately.

[![Phoenix's 48 layers in six groups of eight, with stacked expert counts by quant and a 512-expert original-model reference line.](assets/layer-layout-overview.png)](assets/layer-layout-overview.png)

<details>
<summary>View each eight-layer group with its exact quant counts</summary>

#### Layers 0–7

![Layers 0–7: stacked quant counts and exact expert-count table.](assets/layer-layout-00-07.png)

#### Layers 8–15

![Layers 8–15: stacked quant counts and exact expert-count table.](assets/layer-layout-08-15.png)

#### Layers 16–23

![Layers 16–23: stacked quant counts and exact expert-count table.](assets/layer-layout-16-23.png)

#### Layers 24–31

![Layers 24–31: stacked quant counts and exact expert-count table.](assets/layer-layout-24-31.png)

#### Layers 32–39

![Layers 32–39: stacked quant counts and exact expert-count table.](assets/layer-layout-32-39.png)

#### Layers 40–47

![Layers 40–47: stacked quant counts and exact expert-count table.](assets/layer-layout-40-47.png)

</details>

The [JSON](assets/expert-layout.json) and [CSV](assets/expert-layout.csv) include the complete per-layer audit. Editable SVG versions of all seven graphs are in [assets](assets/).

## Run Phoenix in your own stack

Download the repository:

```bash
hf download WeeAIResearch/Phoenix-69B-A6.6B --local-dir Phoenix-69B-A6.6B
cd Phoenix-69B-A6.6B
```

Follow the **[CPU runtime quickstart](runtime/README.md)** to build the pinned llama.cpp revision, apply the included patch, generate your expert map and open native chat. The tested [coding system prompt](coding-system.txt) is included.

**Upgrading an existing installation? Regenerate `Phoenix.expertmap` after replacing the GGUF.** The map includes local file identity and must match your downloaded file.

This compact mixed-tier model requires the supplied patched runtime for the documented setup. Compatibility with stock llama.cpp and other frontends is not established. The released configuration uses CPU inference; do not combine the expert-prefetch callback with GPU offload.

File size is not a minimum-RAM claim. Memory mapping, cache state, available RAM and disk speed affect performance. The local development launchers use a **20.5 GiB process RSS watchdog ceiling**, which is neither a guaranteed resident footprint nor a reservation. The direct runtime command does not install that watchdog.

## Recovery is part of the story

Before this repair, Phoenix v21 restored **3,074 missing expert slots** and upgraded **1,008 retained low-precision experts to Q4**. Its matched TypeScript interval-intersection development task improved from **3/12 to 12/12 checks**. That history is preserved in the [v21 results](evaluation/results.txt), including its C# and Go failures and instruction-profile differences.

Those older results belong to the preceding v21 weights and have **not all been rerun on this repaired checkpoint**. They are development history, not additional scores for this release. Earlier weights remain available in this repository's revision history.

## Origin, license and identity

Phoenix is an experimental derivative by **[WeeAI Research](https://weeai.co.uk/)**, based on [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) through the [lmstudio-community Q4_K_M conversion](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF). It is not an official Qwen release. No equivalence to the original Q4 or full-precision model is claimed. The latest repair uses independently checked supervised code; its target answer is not an original-Q4 generation.

Weights retain the **[Qwen Community License 1.0](LICENSE)** and its commercial-use conditions. The [llama.cpp MIT license](runtime/LLAMA_CPP_LICENSE) and [MIT prefetch attribution](runtime/PREFETCH_LICENSE) are included.

**[Model manifest](model-manifest.json)** · **[Runtime identity](runtime/source-identity.json)** · **[SHA-256 checksums](SHA256SUMS)**

Model SHA-256:

```text
eb6f289187be9e442f6f7b42ee54ada337c3de25cc10a342e6efcafb2197886d
```

## Help write the next chapter.

Run Phoenix. Try your own tasks. Share what works and what needs work. Every useful result gives us something concrete to build on.

**[Explore WeeAI Research at WeeAI.co.uk →](https://weeai.co.uk/)**
