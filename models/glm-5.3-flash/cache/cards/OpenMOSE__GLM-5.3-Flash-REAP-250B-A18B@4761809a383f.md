---
license: mit
base_model: zai-org/GLM-5.3-Flash
library_name: transformers
pipeline_tag: image-text-to-text
language:
- en
- zh
- ja
tags:
- glm5
- moe
- expert-pruning
- reap
- fp8
- multimodal
---

# GLM-5.3-Flash-REAP-250B-A18B

**GLM-5.3-Flash with 228 of 288 routed experts per MoE layer (20.8% of the routed experts removed), pruned with a
domain-aware variant of REAP, stored in the original block-wise FP8 format, loadable with unchanged inference code.**

Pruned by [OpenMOSE](https://huggingface.co/OpenMOSE). Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).

| | GLM-5.3-Flash | **REAP-250B-A18B** |
|---|---|---|
| routed experts per MoE layer (42 layers) | 288 | **228** |
| total parameters | 321.3 B (320 B) | **250.5 B** without the MTP layer (257.9 B with it) |
| routed-expert parameters | 304.4 B | 241.0 B |
| active parameters per token | 18 B | **18 B** (unchanged: top-8 of the remaining experts + shared expert) |
| checkpoint size (FP8 e4m3, block 128×128) | 328 GB | **265 GB** (50 shards) |
| vision tower, MTP layer, tokenizer, chat template | | unchanged, copied verbatim |
| architecture / config | `Glm5NextForConditionalGeneration` | same class, `n_routed_experts: 228` |

The goal of this release is a smaller footprint for the same active compute (fits on fewer / smaller GPUs), with the
model's **coding, multilingual chat and Japanese** abilities preserved as well as expert pruning allows. Section 2
explains why a naive REAP calibration did *not* achieve that on this model, and what was changed.

---

## 1. What is REAP, in one paragraph

[REAP (Router-weighted Expert Activation Pruning)](https://arxiv.org/abs/2510.13999) scores every routed expert of a
MoE layer by the average, over the calibration tokens routed to it, of *router weight × expert output norm*
(its "saliency"), and simply deletes the lowest-scoring experts; the router rows and the routing bias of the removed
experts are deleted as well, and no retraining is done. The tokens that used to go to a removed expert are re-routed
by the (unchanged) router to the remaining ones. It works best when a model has redundant or rarely used experts.

## 2. What is different from vanilla REAP

### 2.1 GLM-5.3-Flash has no spare experts — and its experts are specialists

On 2.1 M calibration tokens (chat + code) **not a single one of the 12,096 routed experts was unused**; the routing
entropy per layer is 0.92–0.99 (nearly uniform), only 1.4 % of the experts receive less than 10 % of the average
traffic. The aux-loss-free load balancing (`e_score_correction_bias`) of GLM-5 does its job: there is no "dead expert"
budget to harvest. Every removed expert removes real capacity, and *which* capacity is decided entirely by the
calibration data.

We measured this directly by calibrating twice with very different data:

| calibration data | tokens | Spearman rank correlation of expert saliency vs. the other run | experts kept at k=224 that are identical |
|---|---|---|---|
| multilingual chat only (`OpenMOSE/reap-calib-mix`) | 8.4 M | 0.714 | 88.0 % |
| code reasoning only (`nvidia/OpenCodeReasoning`) | 8.4 M | | (≈27 experts per layer differ) |

and by checking the *products*:

* pruning to 224 experts with **chat-only** saliency: the model lost most of its coding ability
  (the code-specialist experts sit at ranks ≈240 in the chat ranking — just outside the cut — although they are at
  rank ≈186 when code is in the calibration set);
* pruning to 208 experts with **code-only** saliency: coding stayed usable but the model **could no longer speak
  Japanese** (no Japanese token ever reached the Japanese-specialist experts during calibration, so their saliency
  was ≈0).

In other words, REAP's saliency measures *"how much did this expert contribute to the calibration data"*, and the
importance of a domain-specialist expert is proportional to the share of its domain in the calibration mix. A single
mixed calibration set "averages" every specialist towards the cut-off; a single-domain set deletes every other
domain. Increasing the amount of data does not help (8× more chat data changed only 8 % of the selection).

### 2.2 Domain-aware selection: per-domain saliency, max-rank union, protected cores

Instead of one saliency vector per layer, this model uses **one saliency vector per domain** and combines the
*rankings*, not the raw scores:

1. Run the REAP calibration separately on each domain: **chat** (`OpenMOSE/reap-calib-mix`, multilingual
   instruction/response pairs, 7,484 documents packed to 2048 × 4096 tokens) and **code** (`nvidia/OpenCodeReasoning`
   split_0, problem + `<think>` reasoning trace + solution, packed to 2048 × 4096 tokens), ≈8.4 M tokens each, in the
   model's chat template.
2. In every layer, rank the experts within each domain; an expert's score is its **best normalized rank over the
   domains** ("max-rank" combination). An expert that is important in *any* domain is kept; only experts that are
   unimportant in *all* domains are removed.
3. The **top-64 experts of every domain are protected unconditionally**; in practice the
   max-rank rule already contains them (union of the two cores: 76–95 experts per layer).

Result per layer (average over the 42 MoE layers): of the 228 kept experts, **212.7 are in the chat top-228 and 211.5 are
in the code top-228**; the ≈15 experts sacrificed on each side are the ones at the very boundary of that domain's ranking.
The single-layer output error of the combined selection is within 7–11 % (relative) of the domain-specific selection on
each domain's own data — versus a collapse of the other domain with a single-domain selection.

This is a general recipe, not a GLM-specific trick: **enumerate the capabilities you want to keep, calibrate each of
them separately, and prune only what none of them needs.** The price is that the union of the per-domain "must keep"
sets bounds the achievable pruning ratio (here the union of the chat and code top-224 sets is ≈251 experts per layer).

### 2.3 FP8 in, FP8 out — no dequantize/requantize round trip

GLM-5.3-Flash ships as block-wise FP8 (`weight` in `float8_e4m3fn` + `weight_scale_inv` in fp32 per 128×128 block).
Rather than loading the model, pruning `nn.Module`s and calling `save_pretrained` (which would change the layout and
force a re-quantization), the pruned checkpoint is **written directly from the original safetensors shards**:

* the kept experts' `weight` and `weight_scale_inv` tensors are **bit-exact copies**, renumbered 0…227;
* `mlp.gate.weight` and `mlp.gate.e_score_correction_bias` are sliced to the kept experts;
* every other tensor — attention, shared experts, dense MLPs, norms, hyper-connection parameters, the **vision tower**,
  `lm_head`, and the **MTP layer** (`model.language_model.layers.45.*`) — is copied verbatim;
* `config.json` changes only `text_config.n_routed_experts` (288 → 228) and records the pruning arguments under
  `merge_args`; `quantization_config` is unchanged.

The result loads with the unchanged `Glm5NextForConditionalGeneration` class. The full-precision model was only ever
materialized (bf16, dequantized on the fly) for the calibration forward passes.

### 2.4 Two-phase workflow: calibrate once, prune offline at any size

The saliency of every expert (both REAP and the routing-frequency variant) is saved once per calibration set
(`calib.pt`, ≈200 MB). Producing a checkpoint for any number of kept experts, any combination of calibration sets and
any per-layer allocation is then a **CPU-only, I/O-bound** operation of a few minutes — no model loading, no GPU.
This is what made the domain experiments above affordable. (REAP with `sequential=False` is exactly reproduced:
the offline result is bit-identical to the one-shot pipeline.)

### 2.5 Other differences and fixes worth knowing about

* **Saliency of unused experts.** The reference implementation replaces a zero saliency by the smallest non-zero
  value (so that unused experts can still be *merged*). For *pruning* this creates ties with the weakest used expert
  and, depending on sort order, can drop a used expert in favour of an unused one. Zeros are kept as zeros here.
* **Dense front layers.** GLM-5.3-Flash has three dense layers before the first MoE layer; in the reference
  non-sequential mode these were skipped when propagating the hidden states, so the MoE layers were calibrated on
  raw embeddings. Fixed (all layers are propagated).
* **Long-context calibration.** Documents are packed (no padding) and chunks of different sequence lengths can be
  mixed; per-expert statistics are token-weighted. Attention for the DSA (sparse attention) layers is computed in
  query blocks so that 16 k-token calibration sequences fit in memory.
* **Uniform allocation on purpose.** We probed every layer's pruning error (output error of the layer pruned to k
  experts, k = 16…272 in steps of 16, on cached inputs) and searched per-layer expert budgets with the full model on
  held-out data. Allocations that protect early layers reduce the KL divergence to the original model by up to
  ≈15 % at the same total budget, but per-layer expert counts are not representable in the GLM-5 config and
  would need patched inference engines, so this release keeps the same number of experts in every layer.
  Some findings from the probes: layers 17–34 contain a few "outlier" experts whose output norm is 40–470× the
  median (they dominate any global-norm metric and must be kept — the frequency-based saliency drops them and is
  catastrophic on this model); the MoE blocks of the sparse-attention (DSA) layers are, if anything, slightly
  *easier* to prune than those of the linear-attention (KDA) layers; the last layers (42–44) are by far the easiest.

## 3. Fidelity to the original model

Teacher-forced comparison on held-out text (4 sequences × 16,384 tokens of multilingual chat and 4 × 16,384 of code
reasoning, generated with a different seed than the calibration data; ≈130 k tokens), original vs. pruned, next-token
distributions:

| model | KL(orig ‖ model) ↓ | top-1 agreement ↑ | top-1 (confident tokens) ↑ | held-out ppl |
|---|---|---|---|---|
| GLM-5.3-Flash (original) | 0 | 100 % | 100 % | 4.068 |
| **GLM-5.3-Flash-REAP-250B-A18B** | **0.453** | **86.0 %** | **92.4 %** | 4.181 (+2.8 %) |
| *(reference) uniform 224 experts, single mixed calibration set* | 0.476 | 85.7 % | – | +5.8 % (4.338 vs. 4.100 in that run) |

The last row is an earlier checkpoint pruned with a single mixed (chat + code) calibration set and plain REAP
ranking, evaluated on the same held-out data: even with fewer experts removed, it is worse on every metric than the
domain-aware 228 selection.

`KL` is KL(original ‖ pruned) averaged over tokens; `top-1` is the fraction of tokens where the greedy choice is the
same; `top-1 (confident)` restricts that to tokens where the original model puts > 0.5 on its top choice; `ppl` is the
held-out perplexity of each model on the actual text. Perplexity alone can be misleading for pruned models (a
flattened distribution can *lower* the perplexity of surprising tokens while diverging from the original), which is
why KL and top-1 agreement are reported.

These are single-token statistics; errors compound in long generations and **the per-layer or per-token metrics
underestimate task-level damage** (e.g. the chat-only 224 selection was only ≈26 % worse than the code-aware one in
per-layer error on code inputs, yet its coding ability collapsed). Qualitatively, this 228-expert selection was judged
to be usable for coding, multilingual chat and Japanese in the author's tests; standard benchmark numbers will be
added when available. Please evaluate on your own workload before relying on it.

## 4. Usage

The checkpoint is a drop-in replacement for the original; nothing but the number of routed experts changed.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:]))
```

Notes:

* FP8 weights need an FP8-capable GPU stack; transformers dequantizes to bf16 automatically when it cannot run FP8
  (≈500 GB of bf16 across your devices in that case).
* **MTP layer.** `model.language_model.layers.45.*` (the multi-token-prediction head used for speculative decoding by
  engines such as vLLM) is copied **unpruned**, i.e. it still has 288 experts while the config says 228.
  transformers ignores this layer. Engines that build the MTP block from `n_routed_experts` may fail to load it —
  disable MTP / speculative decoding, or remove the layer (`num_nextn_predict_layers: 0` and drop the
  `layers.45.*` tensors) in that case.
* `n_routed_experts = 228` is not a multiple of 8/16; for expert-parallel or grouped-GEMM kernels that require it,
  the same recipe produces 224 or 240 in minutes.

## 5. Reproducing / adapting the recipe

The recipe is fully described above and is simple to re-implement on top of any REAP implementation:

1. Build one packed calibration set per capability you want to keep (here: multilingual chat from
   `OpenMOSE/reap-calib-mix` and code reasoning from `nvidia/OpenCodeReasoning`, 2048 sequences × 4096 tokens each,
   formatted with the model's chat template), and run the REAP saliency pass **separately** on each set with the
   model unchanged (no sequential pruning). Save the per-layer saliency vectors.
2. Per layer, rank the experts within each set, give every expert its best normalized rank across the sets, force the
   top-64 of every set in, and keep the 228 best.
3. Write the checkpoint from the original shards: copy the kept experts' FP8 tensors and scales unchanged, renumber
   them, slice the router weight and `e_score_correction_bias`, copy everything else, set `n_routed_experts`.

`merge_log.jsonl` in this repository lists, for every layer, the combined saliency of all 288 experts and the original
indices of the 228 kept ones, so the selection can be audited or reused. The selection is robust to the amount of
calibration data (a few million tokens per capability are enough) but not to its *content*: add a calibration set for
every capability (a language, tool use, a domain) that must survive.

## 6. Limitations

* Any capability that is not represented in the calibration sets (chat in the languages of `reap-calib-mix`, code
  reasoning) may be degraded; images are processed by the untouched vision tower, but the text side that reasons
  about them was pruned with text-only calibration data.
* No recovery fine-tuning was applied. Every removed expert is real capacity on this model; expect a measurable
  quality drop compared with the original, growing with generation length.
* The MTP layer mismatch described above.

## 7. Acknowledgements and citations

* Base model: GLM-5.3-Flash by Z.ai (MIT).
* REAP: *Router-weighted Expert Activation Pruning*, Cerebras Research — https://arxiv.org/abs/2510.13999
* REAM: *Merging Improves Pruning of Experts in LLMs*, Jha, Hashemzadeh, Pasand, Parviz, Lee, Knyazev (Samsung SAIL
  Montréal) — https://arxiv.org/abs/2604.04356.
* Calibration data: `OpenMOSE/reap-calib-mix`, `nvidia/OpenCodeReasoning`.

```bibtex
@misc{openmose2026glm53flashreap,
  title  = {GLM-5.3-Flash-REAP-250B-A18B: domain-aware expert pruning of GLM-5.3-Flash},
  author = {OpenMOSE},
  year   = {2026},
  url    = {https://huggingface.co/OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B}
}
```
