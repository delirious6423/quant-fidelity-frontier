---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- mlx
- mtplx
- mixture-of-experts
- reap
- pruning
- quantization
- mtp
- apple-silicon
---

# Qwen3.8-Flash-Next-REAP-288 · Q4g32 + Q8 · MTPLX

**A local rebuild of Qwen3.8-Flash-Next, designed and tested on a MacBook Pro with an Apple M5 Pro and 64 GB of unified memory.**

The aim is to make a large mixture-of-experts model practical on this machine while retaining native multi-token prediction (MTP). The build combines the published REAP-288 expert selection, a mixed Q4/Q8 quantization recipe inspired by MTPLX Optimized Speed, and an SSD-backed n-gram embedding table.

**Release status: model upload in progress. Do not download until the upload is marked complete.** The local artifact has passed loading, generation and context/cache checks. Evaluation is ongoing. Its current implementation uses an experimental MTPLX source checkout with local changes; this page does not claim compatibility with an unmodified MTPLX app release.

## What the name means

| Part | Meaning |
|---|---|
| Qwen3.8-Flash-Next | Original Qwen checkpoint and architecture |
| REAP-288 | Keep 288 of 512 routed experts in each of the 48 main-model layers |
| Q4g32 | Store routed expert weights at 4 bits with quantization groups of 32 |
| + Q8 | Use 8-bit quantization for selected sensitive matrices |
| MTPLX | Target runtime, including native MTP and SSD n-gram access |

This is an independently rebuilt artifact, not an official Qwen, Youssofal or sh0wie release. It was converted directly from the original BF16 weights, rather than expanding an existing Q4 conversion to a higher precision. No new training or fine-tuning was performed.

## Why it can fit on a 64 GB Mac

Three changes work together:

1. **A smaller expert bank.** REAP removes lower-ranked routed experts from the main model, reducing the weights that must remain available in unified memory.
2. **Mixed precision.** Most expert weights use Q4/group32; selected sensitive matrices use Q8/group64. Smaller groups allow more local quantization scales, at some additional memory cost compared with Q4/group64.
3. **SSD-backed n-gram embeddings.** The large lookup table is stored on SSD and accessed as needed instead of keeping its entire payload resident on the GPU.

The third mechanism applies to the **n-gram embedding table**, not to the removed experts. Pruned experts are absent from this build and cannot be fetched back from SSD during inference.

### Measured memory and storage

Measurements below come from the local M5 Pro / 64 GB evaluation. GiB means binary gigabytes.

| Item | Size or measured allocation |
|---|---:|
| Main-model weight payload | 44.390 GiB |
| Native MTP head payload | 1.583 GiB |
| Vision weight payload | 0.836 GiB |
| SSD n-gram table payload | 29.802 GiB |
| Combined weight/table payload | Approximately 76.61 GiB on disk |
| Active MLX allocation after text-model load | 45.980 GiB |
| Peak MLX allocation in a synthetic 65,536-token prefill | 49.399 GiB |
| Peak MLX allocation in the initial MTP task suite | 49.982 GiB |

Disk size is not resident memory. MLX allocation also excludes some operating-system and file-cache use. Swap was already allocated and increased during loading, so these results are **not a zero-swap guarantee**. Context, prefill, other applications and runtime settings all affect available headroom. Synthetic cache-copy diagnostics reached a higher peak of 53.450 GiB.

The vision weights were preserved, but our evaluation has been text-only. Multimodal behavior has not been validated.

## Which experts were removed, and why?

This build reuses **sh0wie's published REAP-288 selection**. It does not introduce a new expert ranking.

For every main-model layer:

- Start with 512 routed experts.
- Keep the 288 experts with the highest published saliency scores.
- Remove the remaining 224 experts and their corresponding router rows.
- Preserve the selected expert order consistently across the router and all expert projections.

Across 48 layers this removes **10,752 layer/expert pairs**, or **43.75% of the routed expert bank**. Routing still activates the trained top 10 routed experts per token; the shared expert is retained.

The selection is based on a contribution score, not simply how often an expert is called. Conceptually:

```text
expert score =
    mean(router weight × magnitude of expert output
         over tokens for which that expert was selected)
```

Our audit of the published arrays found that the stored score is accumulated contribution divided by selection count, and that each retained set is exactly the highest-scoring 288 experts in that layer.

The publisher reports calibration on approximately **686,000 tokens of agentic-coding traffic**. We reused and audited its statistics and retained IDs; we did not independently regenerate that private calibration corpus. This makes the selection a coding-oriented starting point, not a demonstrated optimum for writing, analysis or general conversation. [Selection source](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit)

Expert IDs are local to a layer. Expert 42 in layer 0 and expert 42 in layer 47 are different tensors. There is no verified map saying that particular IDs are exclusively “writing experts” or “coding experts.”

The exact retained IDs are defined by the [pinned retained-expert manifest](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit/blob/668f31bcc56bf9400e64c9463445eee47597c2d9/reap_kept_experts.json). For each layer, the removed IDs are the complement of that list within original IDs 0–511. The manifest used for this build has SHA256:

```text
50dbb9cacc445724c64c9429940bc1b6d0ace48fac3a92734376740128b13dd6
```

Higher precision does not restore missing experts. A selection calibrated under another quantization recipe is also not automatically optimal for this one.

### Parameter count after pruning

A logical tensor-count audit estimates **72.868 billion main-model parameters**, down from approximately **125.744 billion** before pruning. These figures exclude the n-gram table, MTP head and vision tower.

Removing 43.75% of routed experts reduces the main-model parameter count by about 42%, because attention, shared experts and other components remain. This is not a claim of equivalence to a dense 73B model. The number of active routed experts per token remains 10; active compute does not fall in proportion to total stored parameters.

## Precision and MTP

The conversion recipe uses:

| Component | Treatment |
|---|---|
| Routed expert projections | Q4 / group32 |
| Token embeddings and language-model output head | Q8 / group64 |
| Routers and shared experts | Q8 / group64 |
| Selected Gated DeltaNet outputs, QSA attention projections and indexer | Q8 / group64 |
| Structural tensors, including relevant normalization parameters | Retained in BF16 |
| N-gram table | Q4 / group32, contiguous SSD sidecar |

The original checkpoint's MTP head is converted from the **same pinned source checkpoint**. Its full **512-expert bank** is preserved, even though the main model retains 288 experts per layer. The runtime therefore needs to handle the two expert counts separately.

MTP proposes future tokens and asks the target model to verify them. It does not reinstate pruned experts or increase their precision. We did not retrain the head against the pruned main model: shared checkpoint provenance should not be confused with a newly trained assistant specialized for this derivative.

Compared with the reference REAP Q4g64 pack, this build uses the **same retained expert selection**, smaller Q4 groups and Q8 sensitive projections. Quantization and runtime both differ, so an observed quality or speed difference cannot yet be attributed to precision alone.

## Preliminary local evaluation

The first eight-task suite covers invoice arithmetic, scheduling, Python interval aggregation, dependency batching, SQL aggregation, cache simulation, structured extraction and policy reasoning.

These are direct prompts with executable/content checks, not an agent receiving tools or test-feedback repair attempts. Correct JSON inside a single enclosing code fence is accepted.

| Metric | Without MTP (AR) | Native MTP, depth 3 |
|---|---:|---:|
| Correct completed tasks | 7/8 | 6/8 |
| Generated tokens, including reasoning | 80,592 | 141,551 |
| Total task time, including failures | 59.70 minutes | 83.18 minutes |
| Token-weighted decode throughput | 22.55 tokens/s | 28.42 tokens/s |

Conditions: one attempt per task/mode; greedy sampling, seed23; requested medium reasoning; 64,000 output-token ceiling within a 65,536-token total context; native MTPLX execution.

MTP completed the six tasks passed by both modes in **487.85 seconds**, versus **566.47 seconds** for AR, approximately **1.16× faster**. However, MTP failed to finish interval aggregation and dependency batching; AR failed batching. Those long reasoning runs made the full MTP suite slower.

This illustrates the project's main evaluation rule: **time to a correct finished answer matters more than raw tokens per second.** These small, single-run results are preliminary, and broader writing, runtime and model comparisons remain in progress. No full-weight quality equivalence is claimed.

## Observed strengths and prompting guidance

In our initial tests, the model completed invoice arithmetic, scheduling, SQL aggregation, cache simulation, structured extraction and policy reasoning correctly with MTP. It struggled to finish interval aggregation and dependency batching, repeatedly revisiting edge cases—a pattern also observed in other quantized Flash variants we tested. These tasks have definite solutions; the observed difficulty was converging on a final implementation.

For similar tasks, consider prompting for a complete implementation after a small, bounded set of checks, rather than exhaustive discussion. A harness can detect repetitive output, enforce a time or reasoning budget, and use executable tests with a limited number of repair attempts. These are proposed mitigations, not demonstrated fixes for this model.

## What was changed in MTPLX?

The main inference-time change is small but necessary: **the pruned main model has 288 experts per layer, while its original MTP head still has 512.** The pinned upstream implementation used one expert-count setting for both. Loading the complete original head against a 288-expert configuration therefore needed explicit, separate head geometry.

### 1. Separate expert counts for the main model and MTP head

In `mtplx/models/qwen4_exp.py`, the local patch:

- Adds an optional `mtp_num_experts` configuration field.
- Validates that it is an integer large enough for the configured top-k routing.
- Constructs the MTP module using a copied configuration with its own expert count, leaving the main-model configuration unchanged.
- Preserves strict weight loading. It does not suppress shape errors or permit an arbitrary unrelated assistant to be attached.

The relevant configuration is:

```json
{
  "num_experts": 288,
  "mtp_num_experts": 512,
  "num_experts_per_tok": 10
}
```

This is model construction support, not new MTP training or a new speculative verification algorithm. Weight provenance comes from the pinned source checkpoint and conversion receipts, not from these configuration fields.

### 2. Expert-aware conversion and router alignment

The build tools were extended to accept an explicit retained-expert manifest:

- `scripts/qwen4_expert_selection.py` validates all 48 layers, unique in-range IDs, consistent retained counts and top-k feasibility.
- `scripts/convert_qwen4_exp_fp8.py` applies the same retained IDs, in the same order, to the router rows and routed expert projections. Despite this existing script's filename, this release was built from BF16 source weights.
- The main-model pruning mask is never applied to the MTP head.
- The output records the original checkpoint revision and the selection-manifest hash.

Router/expert ordering matters: if the router's compact expert index points to a different expert's tensors, a model can have plausible shapes while computing the wrong function.

### 3. A resumable BF16 build on the Mac

The new `scripts/stream_qwen4_exp_bf16.py` converts the checkpoint one source shard at a time. It verifies source hashes, selects the retained BF16 expert rows before conversion, writes and reopens converted parts, and records output checksums and tensor accounting. Final assembly combines the trunk, original MTP head, vision weights and n-gram table; incomplete builds do not expose a loadable final configuration.

This is a conversion-time memory and disk-management tool. It is not streaming missing experts during inference.

### What already came from MTPLX

Native MTP generation, the Q4/group32 plus Q8-sensitive recipe, and SSD-backed n-gram access were already present in the pinned MTPLX source. Our build uses those facilities; it does not claim to have invented them. Evaluation used `MTPLX_NGRAM_RESIDENT=0` to select SSD-backed n-gram operation, with the existing generic native generation paths rather than a new specialized 288-expert kernel.

### Scope and integration guidance

The runtime extension is **opt-in through model configuration**, not hard-coded to this repository's name. With `mtp_num_experts` absent or `null`, the head continues to use the main model's expert count, following the existing construction path. When the field is present, only the head receives a copied configuration with the separate count; the main model's arguments are not mutated. Another compatible build could explicitly use the same feature.

This patch does not change sampling, token verification, acceptance rules, cache rollback or generation kernels. Conversion scripts run only when invoked to build a model. Evaluation settings such as MTP depth and SSD n-gram mode are separate launch choices. A legacy-geometry test checks the default path; this is not an exhaustive regression guarantee across all MTPLX models. Our work used an isolated source checkout and did not alter the installed desktop app.

For a completed artifact, the compatibility change belongs in the **inference backend**, not in task prompts. A harness already connected to that backend's supported interface should not need model-specific task logic. The exact runtime diff below targets upstream commit `21be78b3f51820eecef020e5e4855c0715eaf9a5`; review it against newer versions instead of applying it blindly.

<details>
<summary>Minimal runtime patch</summary>

```diff
diff --git a/mtplx/models/qwen4_exp.py b/mtplx/models/qwen4_exp.py
index e8e0e04..be69ae5 100644
--- a/mtplx/models/qwen4_exp.py
+++ b/mtplx/models/qwen4_exp.py
@@ -43,7 +43,7 @@ import os
 import re
 import struct
 import time
-from dataclasses import dataclass, field
+from dataclasses import dataclass, field, replace
 from pathlib import Path
 from typing import Any, Dict, List, Optional, Union
 
@@ -90,6 +90,9 @@ class TextArgs(BaseModelArgs):
 
     # MoE (names shared with qwen3_next's SparseMoeBlock)
     num_experts: int = 512
+    # Complete pruned builds may preserve the original head's expert bank.
+    # This describes geometry only; it does not establish weight provenance.
+    mtp_num_experts: Optional[int] = None
     num_experts_per_tok: int = 10
     moe_intermediate_size: int = 640
     shared_expert_intermediate_size: int = 640
@@ -133,6 +136,11 @@ class TextArgs(BaseModelArgs):
     mrope_interleaved: bool = False
 
     def __post_init__(self):
+        if self.mtp_num_experts is not None and (
+            type(self.mtp_num_experts) is not int
+            or self.mtp_num_experts < self.num_experts_per_tok
+        ):
+            raise ValueError("mtp_num_experts must be an integer >= num_experts_per_tok")
         if self.rope_parameters:
             self.partial_rotary_factor = self.rope_parameters.get(
                 "partial_rotary_factor", self.partial_rotary_factor
@@ -5495,6 +5503,10 @@ class Qwen4ExpMTP(nn.Module):
 
     def __init__(self, args: TextArgs):
         super().__init__()
+        # Do not mutate the trunk arguments or relax strict weight loading.
+        # The original native head can retain 512 experts after trunk pruning.
+        if args.mtp_num_experts is not None:
+            args = replace(args, num_experts=args.mtp_num_experts)
         d = args.hidden_size
         self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=args.rms_norm_eps)
         self.pre_fc_norm_hidden = nn.RMSNorm(d * args.hc_count, eps=args.rms_norm_eps)
```

</details>

An integration agent can use this instruction:

```text
Integrate this complete REAP-288 artifact in an isolated MTPLX source environment.
Inspect the current Qwen4Exp implementation first. If it already supports a
separate MTP expert count, verify that implementation instead of patching twice.
Otherwise port the model card's minimal runtime diff to the pinned checkout.
Preserve existing behavior when mtp_num_experts is absent. Do not mutate trunk
arguments, relax strict loading, or change generation/verification algorithms.
Read the artifact's actual text configuration: trunk=288, MTP=512, top-k=10.
Verify source provenance and strict loading of the complete artifact; geometry
alone cannot validate an unrelated MTP sidecar. Confirm legacy geometry still
works and mismatched head tensors still fail. Then run a short native-MTP smoke
test before connecting the task harness. Use the checkout's documented launch
interface; do not invent CLI flags. Keep SSD n-gram and MTP-depth settings local
to this model's launch profile. Record runtime revision, settings and results.
```

### Revision and availability

- Upstream source base: `21be78b3f51820eecef020e5e4855c0715eaf9a5`.
- Local implementation commit: `11021aedf035d10f88e829d4576e9ac969434a53`.
- Recorded validation: 33 selected tests passed, covering the new converter/geometry cases and existing cache/rollback checks, followed by real-weight loading and generation probes.

The implementation commit is a **local commit**, not a claim of an upstream merged release. The minimal runtime patch is included above, and the [complete implementation diff and integration instructions](MTPLX-INTEGRATION.md) are supplied alongside the weights. A packaged desktop-app release is not provided; the model name's MTPLX suffix identifies the tested runtime path, not guaranteed compatibility with the stock desktop app. The installed MTPLX application was not modified.

## Compatibility and known limitations

- The validated path currently uses a locally modified MTPLX source checkout. Our additions separate trunk/MTP expert geometry and implement consistent router/expert selection during conversion; SSD n-gram support is inherited from MTPLX. Reproducible runtime packaging and public installation instructions remain pending.
- Functional checks passed, including forced-rejection cache restoration through 65,536-token synthetic contexts. These are targeted checks, not proof of correctness for every prompt.
- Longer greedy AR/MTP outputs can diverge. A local replay found execution-path-dependent cache/logit differences affecting a near-tied token choice even without rejected drafts. MTP is not assumed to reproduce AR token-for-token in this implementation.
- The expert-selection calibration favors agentic coding. General writing, multilingual coverage, rare entities and multimodal quality need separate evaluation.
- A 64 GB fit on the tested setup does not establish fit at every context size or with other applications consuming memory.

## Provenance and credits

- **Qwen:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), original BF16 checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- **sh0wie:** published REAP-288 expert selection and calibration statistics.
- **Cerebras Research:** [REAP methodology](https://github.com/CerebrasResearch/reap) and [paper](https://arxiv.org/abs/2510.13999).
- **Youssofal / MTPLX:** [runtime and conversion work](https://github.com/youssofal/MTPLX) that informed this local rebuild. Source base: `21be78b3f51820eecef020e5e4855c0715eaf9a5`, with local modifications.
- **30cmtonyfaker:** [REAP-288 Q4/MTP comparison pack](https://huggingface.co/30cmtonyfaker/Qwen3.8-Flash-Next-REAP-288-MLX-Serve-4bit-MTP), used as a comparison baseline. Its Q4 weights were not the source of this BF16 rebuild.
- **khuasar:** local rebuild, integration, provenance audit and evaluation on the 64 GB M5 Pro MacBook Pro.

## License

The model weights remain governed by **Qwen Community License 1.0**. Pruning, conversion and quantization do not remove the upstream conditions. See the [official license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), including its commercial-use conditions. Runtime code and third-party components retain their own licenses.

*Experimental model card — 17 September 2026. Upload verification pending.*
