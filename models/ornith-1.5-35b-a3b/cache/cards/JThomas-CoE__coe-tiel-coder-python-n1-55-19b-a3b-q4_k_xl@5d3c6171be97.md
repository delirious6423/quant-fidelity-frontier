---
language:
- en
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - mixture-of-experts
  - ornith1.5
  - tiel-coder
  - college-of-experts
  - expert-surgery
  - pruned
  - specialist
  - python
  - coding
  - gguf
  - ollama
  - vision
  - r-percent-r55
pipeline_tag: text-generation
library_name: gguf
---

# CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl — College of Experts Specialist

> This is the **Python coding specialist** built from **Tiel-Coder**, itself a Q4_K_XL
> requantisation of **Ornith-1.5-35B-A3B**. The expert mask is an **n1-swap 50% budget
> mask at r55**, profiled on the blended SWE+Python corpus.
>
> This is the **vision-capable** variant: the GGUF ships with a separate `mmproj`
> projector, enabled by a two-`FROM` Modelfile (Ollama 0.34+).

**Base model**: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT)
**Intermediate**: Tiel-Coder-35B-A3B (UD-Q4_K_XL requant of the above) — the direct surgery source
**Architecture**: 40 layers × 256 routed experts/layer, native top-8 routing (TOP_K=8 + 1 shared per token)
**CoE construction**: prune each MoE layer from **256 → 128 routed experts (50% budget)**, with **n1-swap admixture at r = 55%** of the n1-only candidate pool
**Quantization**: Q4_K_XL (attention Q8_0; routed-expert FFN mixed Q4_K/Q5_K/Q6_K)
**Hugging Face repo**: `JThomas-CoE/coe-tiel-coder-python-n1-55-19b-a3b-q4_k_xl`

---

## ⚠️ Beta Release — Safety Disclaimer

These models are **beta releases** and should be treated as research artifacts, not
production-ready systems.

Expert surgery selects and retains domain-relevant experts based on activation
patterns observed during profiling. The pruning pipeline is designed solely to
create a coherent domain specialist — it has no mechanism to identify which
experts contribute to model alignment, ethical reasoning, or safety guardrails. As
a result, experts responsible for enforcing those behaviours may have been
inadvertently removed during the surgery process.

Appropriate use of any model in the College of Experts family is the sole
responsibility of the end user. The authors make no representation that this model
retains the safety properties of the parent `Ornith-1.5-35B-A3B`, and users should
not rely on it as a substitute for models that have undergone safety evaluation.

---

## ⚠️ Critical Usage Note — Think-On Default

**This model may be used in thinking-on mode, unlike the previous pruned models in this collection.** Think-on is its default and recommended mode.

Previously, if you were using the Ollama API, you had to pass `"think": false` in your request body. 
If you were accessing the model via a raw API (llama.cpp server, OpenAI-compatible
endpoint, etc.) you had to inject a closed thinking block at the start of the
assistant turn:

```python
messages = [
    {"role": "system",    "content": "Your system prompt here."},
    {"role": "user",      "content": "Your question here."},
    {"role": "assistant", "content": " <think>\n</think>\n"},   # <-- required prefill
]
```

**Why this was required**: expert surgery retains 50% of the routed expert pool
per layer, selecting experts that are maximally active on domain content and
chain-of-thought reasoning. A side effect is that the **loop-suppression
experts** — which activate on metacognitive closure signals near the end of a
` thinking` block — do not have a concentrated domain-specific activation
signature and are disproportionately pruned. In think-on mode, this can cause the
model to enter a reasoning loop that exhausts the token budget without
producing a final answer.

**HOWEVER - This variant is unusually robust to that effect, and that is its distinguishing
property.** Measured on the frozen 75-question LiveCodeBench v6 set (see
[Validation](#validation)):

| model | think-off | think-on |
|---|---:|---:|
| **This model** (Ornith/Tiel r55) | 41/75 | **44/75** |
| Qwen3.6 pruned r25 specialist | 40/75 | **21/75** |

Where the pruned Qwen model loses 19 passes in think-on, this model **gains 3**.
Its reasoning survives the surgery, which is why it is a viable think-on
specialist for the CoE runtime. **thus, think-on becomes the recommended
default** but at the cost of latency.

### Ollama Modelfile (vision-enabled)

Two `FROM` lines: the model, then the projector. Ollama ≥ 0.34.

```
FROM <path>/CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl.gguf
FROM <path>/mmproj-BF16.gguf

TEMPLATE {{ .Prompt }}
PARAMETER temperature 0.6
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.05
PARAMETER repeat_last_n 1024
PARAMETER num_ctx 32768
PARAMETER num_predict -1
PARAMETER num_gpu 99
```

Temperature 0.6 is strongly recommended. Higher temperatures (≥ 0.8) materially
increase loop rates in think-off mode.

---

## What This Model Is

Produced by **activation-directed expert surgery** applied to the Tiel-Coder
Q4_K_XL requant of Ornith-1.5-35B-A3B. The surgery does not change any weight
values — it prunes the routed-expert FFN weight tensors that are not part of the
domain-specialist mask, then saves the result as a smaller GGUF. **No
post-surgery fine-tuning or training was done.**

Attention, router, shared expert, embeddings, output head, and all norms are
preserved at full count. Only the routed-expert FFN tensors are sliced; the router
metadata records `expert_count = 128` while `expert_used_count` stays at the
parent's **8**, so per-token routing depth is unchanged.

### Memory efficiency

| | parent (Ornith-1.5 / Tiel-Coder UD-Q4_K_XL) | **this model** |
|---|---:|---:|
| file size | 22,360,478,080 B (22.36 GB) | **12,482,892,192 B (12.48 GB)** |
| parameters | 34.7B | **18.5B** |
| routed experts/layer | 256 | **128** |
| expert_used_count | 8 | 8 (unchanged) |
| context | 262,144 | 262,144 |
| vision | separate `mmproj` | separate `mmproj` (shipped) |

Halving the expert pool does not halve the whole model: attention, embeddings,
shared weights and norms remain at full size. **≈44% smaller** than the requant
parent. Expert removal is independent of quantization in principle — pruning
removes weights, quantization uses fewer bits per retained weight — so their
savings compound, but the figures above are at a fixed Q4 class and are not a
measurement of a second quantization step.

---

## Surgery Methodology

### 1. Two profiling passes over the same corpus

The parent is run **twice** through the SWE+Python corpus with router hooks on
every MoE layer, differing only in how many experts the router is asked for:

| pass | routing | purpose |
|---|---|---|
| **n = 8** | native top-8 (the model's normal per-token selection) | baseline utility ranking |
| **n = 1** | top-1 reduction | reveals experts the top-1 view favours |

Per-token the router emits `out[1]` = top-k softmax weights `(T, K)` and
`out[2]` = top-k expert indices `(T, K)`. Aggregated to `(40, 256, 8)` histograms
plus matching weight sums. Utility is rank-collapsed and token-normalised:

```
util[l, e] = sum_k( freq[l, e, k] * weight_mean[l, e, k] )
utility_formula (as recorded) = weight_sum.sum(dim=-1) / n_tokens
```

Profile artifacts: `histograms_swe_ornith_n8.pt`, `histograms_swe_ornith_n1.pt`.
Corpus: `swebench_profile_subset.jsonl` (not bundled with this release)
(170 samples, 441,023 tokens, sha256 `064ac7c7…`).

### 2. Two independent top-128 cuts

For each layer, from each profile separately:

```
n8_mask[l] = top 128 of 256 by descending n8 utility
n1_mask[l] = top 128 of 256 by descending n1 utility
```

Both are 50% budget cuts. **They are computed independently and are not nested.**

### 3. The n1-only candidate pool

```
n1_only[l] = n1_mask[l] \ n8_mask[l]
```

Experts the **top-1** profile ranks in its top half that the **top-8** profile
does not. On the blended SWE+Python corpus this pool totals **973** candidates
(per layer 16–39). For reference, the SWE-only pool is 930 and the Python-only
pool is 857.

### 4. Rerank and swap at the r-percentage cap

- Rerank `n1_only[l]` by **descending n1 utility**
- Swap in the top `n_swap = round(r/100 × len(n1_only))` of them
- **Evict the same number of n8 incumbents starting from the lowest n8 utility
  and working up the ranked list**
- Clamp `n_swap` to the available candidates on either side
- **Budget invariant: exactly 128 experts per layer**

**`r` is a percentage of the candidate pool — not of the 128 slots, and not a
retention fraction.** The realised total sits slightly below
`round(r/100 × pool_total)` because the per-layer clamp bites where candidates are
scarce.

### 5. The shipped mask

| | |
|---|---|
| mask file | `blended_n1swap_r55_K128.pt`, `(40, 128)` int16 |
| corpus | SWE + Python (blended) |
| n1-only pool | **973** |
| **r55 = swapped in** | **533** experts (per layer 9–21) |
| formula prediction | 535 (2 short — per-layer clamp) |
| distinct from r5 on | 485 tuples |
| mask size | exactly **5,120** `(layer, expert)` tuples |
| ratios built | r5, r15, r25, r35, r45, **r55**, r65, r85, r95 |

### 6. Build the GGUF

Slice the source GGUF's routed-expert FFN weight tensors to the 128 selected
experts per layer with the pruning script in the College of Experts AI research repository (not bundled). The pruner
patches `expert_count` 256 → 128 and leaves all other metadata untouched; two
further header strings (`general.name`, `general.size_label`) were corrected in a
separate header-only pass that left the tensor data section byte-identical.

---

## Provenance

| item | value |
|---|---|
| Base model | `ornith-ai/Ornith-1.5-35B-A3B` (MIT) |
| **Surgery source** | `Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf`, 22,360,478,080 B, sha256 `9779b32f998371c66ae5e1aa9a0bc7ad946b1a36c61ea6a8eb39d523b4b19033` |
| Profiling corpus | `swebench_profile_subset.jsonl`, 170 samples / 441,023 tokens, sha256 `064ac7c7…` |
| Utility formula | `weight_sum.sum(dim=-1) / n_tokens` |
| Mask | `blended_n1swap_r55_K128.pt` — 973-candidate pool, 533 swaps, 5,120 tuples |
| Surgery tool | `prune_gguf_from_mask.py` (College of Experts AI research repository; not bundled) |
| **Shipped GGUF** | `CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl.gguf`, 12,482,892,192 B, sha256 `d045561d440731ca783754a482d749a68d97b810c269892962e66c3960e45552` |
| Vision projector | `mmproj-BF16.gguf`, 902,822,016 B, sha256 `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837` |
| Weight integrity | shipped data-section sha256 `a1d506d0…` — **identical to the pre-patch artifact that produced the benchmark results below** |

**Source verification.** Per-tensor quantization-type histogram of the shipped
GGUF is identical to the Tiel-Coder UD-Q4_K_XL requant (733 tensors; routed-expert
FFN `Q4_K×78, Q5_K×38, Q6_K×4`; attention q/k `Q8_0×20`) and differs from the
Ornith-1.5 Q4_K_M conversion (753 tensors, includes `Q4_0`). The specialist was
therefore built from the **Q4_K_XL requant**, as stated. All 80 unquantized norm
tensors match that source bit-for-bit.

**Metadata note.** The GGUF self-describes correctly:
`general.name = CoE-Tiel-Coder-python-n1-55-19B-A3B`, `general.size_label =
128x2.6B`, `qwen35moe.expert_count = 128`, `qwen35moe.expert_used_count = 8`.

**Benchmark identity.** The LCB results below were produced from the same tensor
data section (`a1d506d0…`). The Ollama tag has since been renamed (dropping a
`-vl` suffix), which changed the container's file hash and the two descriptive
header strings (`general.name`, `general.size_label`) but not a single weight.
All 733 tensors and all 46 KV entries are otherwise identical.

---

## Validation

Frozen LiveCodeBench v6, 75 questions (25 easy / 25 medium / 25 hard), seed 42,
32,768 context, 24,576 output cap, upstream LiveCheckBench checker, single
canonical runner (same runner, manifest and cache for every row below).

| configuration | easy | medium | hard | **total** | mean latency |
|---|---:|---:|---:|---:|---:|
| **this model**, think-off | 24/25 | 12/25 | 5/25 | **41/75** | 52.7 s |
| **this model**, think-on | 24/25 | 16/25 | 4/25 | **44/75** | 161.0 s |
| Ornith-1.5 parent (same requant lineage), think-off | 24/25 | 15/25 | 4/25 | 43/75 | 12.5 s |
| Ornith-1.5 parent, think-on | 23/25 | 20/25 | 10/25 | 53/75 | 155.5 s |
| Qwen3.6 pruned r25 specialist, think-off | — | — | — | 40/75 | 57.8 s |
| Qwen3.6 pruned r25 specialist, think-on | 15/25 | 5/25 | 1/25 | **21/75** | 230.0 s |

### Headline

**This model's reasoning survives 50% expert surgery where a comparable pruned
model's does not.** Think-on head-to-head against the pruned Qwen3.6 r25
specialist: **44/75 vs 21/75** — a 23-problem gap. The pruned Qwen run produced no
extractable code on **37 of 75** questions and tripped a loop guard on **33**; this
model trips the guard on **10** and loses **22** to extraction.

**Honest caveats:**

- **Run-to-run noise is ±9% per problem** on this benchmark, measured on 66 rows
  whose code path was identical across two runs of the same runner with the same
  seed. The +3 think-off→think-on delta here (41→44) is **within** that band and
  should not be read as an improvement on its own; what is outside the band is the
  **sign contrast** with the pruned Qwen result (−19).
- **The parent beats this model on this benchmark** (53 vs 44 think-on). The model
  is ≈44% smaller; the comparison is size vs capability, not a capability claim.
- Think-off at 41/75 is **2 below** the parent's think-off 43/75 — also within the
  noise band.

### Failure modes (think-on, this model)

| flag | count |
|---|---:|
| `no_code_extracted` | 22 |
| `degenerated_loop` | 10 |
| `token_limit` | 16 |

The counts sum to 48 across the 31 non-passing questions, so a single question can carry more than one flag.

For contrast, the same guard fires **0 times** on the Ornith-1.5 parent and
**33 times** on the pruned Qwen3.6 specialist.

---

## Quickstart (Ollama)

Download the GGUF, the `mmproj`, the Modelfile, and (for reproducibility) the mask in `masks/`, then:

```powershell
ollama create CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl `
  -f Modelfile
ollama run CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl
```

Think-on is the default; the examples below pass `"think": true` (the Ollama default). Pass `"think": false` for think-off.

```python
import requests
r = requests.post("http://127.0.0.1:11434/api/chat", json={
    "model": "CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl",
    "think": True, #False
    "stream": False,
    "messages": [{"role": "user", "content": "Write a Python function ..."}],
})
```

Vision (the model reads image content via the attached projector):

```python
r = requests.post("http://127.0.0.1:11434/api/chat", json={
    "model": "CoE-Tiel-Coder-python-n1-55-19b-a3b-q4_k_xl",
    "think": True, #False
    "stream": False,
    "messages": [{"role": "user", "content": "Describe this image.",
                  "images": [<base64 PNG, no data-URI prefix>]}],
})
```

### Vision caveat

The projector stack is the parent's, unchanged by the surgery. **The extent to
which expert pruning affects vision fidelity in actual use has not been vetted**
beyond basic sanity checks (capability reported, projector loaded, and correct
colour/shape/position identification on a synthetic test image). Treat image
capability as present but of unverified quality, as with the Qwen3.6-based CoE
releases.

---

## Limitations

- **Beta research artifact.** Not production-ready; no safety evaluation.
- **Single-run validation.** The 75-QID results above are one run per
  configuration. The ±9% per-problem noise band applies.
- **Narrowly specialised.** Profiled on SWE+Python corpora; the specialist is
  intended for Python coding work, not general reasoning.
- **Think-on default.** Think-on is the default and best-scoring mode (44/75 vs 41/75 think-off), at roughly 3x the latency; think-off remains available.
- **Parent is stronger at full size.** This model trades ~44% of the weights for
  a measured capability cost on the benchmark above.

---

## Future Work

The same profiling and surgery pipeline applies to other domains of the same
parent. Additional **Ornith-1.5 Tiel-Coder** specialists may be released as they
are built and validated. No timeline or specific domain set is implied by this
note.

A family README will accompany the release set once more than one Ornith-1.5
Tiel-Coder specialist exists.

---

## Citation / Attribution

**The separability hypothesis.** This specialist is an application of the hypothesis that domain capability in MoE models is physically separable — concentrated in expert subsets that can be surgically extracted without post-surgery training, with unrelated capability measurably excised. The hypothesis and its theoretical framework (histographic expert-activation capture, domain-expert clustering, shared-expert identification, and subset reconstruction) were first stated in J. Thomas, *College of Experts: Decomposing Mixture-of-Experts Architectures into Disk-Resident Specialist Networks for Consumer Hardware* (technical report v0.9, February 2026), [doi:10.5281/zenodo.23403227](https://doi.org/10.5281/zenodo.23403227). The hypothesis was demonstrated in J. Thomas, *Separability of Intelligence in Mixture-of-Experts: Slicing Qwen3-Coder into Independent Domain Specialists* (whitepaper, March 2026), [doi:10.5281/zenodo.23403231](https://doi.org/10.5281/zenodo.23403231), which sliced Qwen3-Coder-Next-80B-A3B into Python and Web specialists that each retained near-parent in-domain accuracy while failing the other's domain (93.0% vs 29.0% HumanEval, cross-domain). It was confirmed quantitatively across ten domains and a 20-pair near/far transfer experiment in J. Thomas, *Gemma4 College of Experts MMLU-Pro Domain Specialists* (model card documentation, May 2026), [doi:10.5281/zenodo.23403233](https://doi.org/10.5281/zenodo.23403233). The activation-utility selection used here descends from that line of work.

Research and engineering by JThomas-CoE.

Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
This specialist is derived from the **Tiel-Coder** Q4_K_XL requantisation of that
checkpoint; all retained weight values are unmodified from it.

The project also publishes a [Qwen3.6-based CoE series](https://huggingface.co/JThomas-CoE)
and a [gemma4-based CoE series](https://huggingface.co/JThomas-CoE) under the same
general approach.

---

## License

This release is licensed under the **PolyForm Noncommercial License 1.0.0** ([LICENSE](LICENSE)).
Commercial licensing is available on request from collegeofexpertsai@gmail.com.

### What PolyForm covers

The PolyForm licence covers all College of Experts work, not only code. That includes the
activation data and utility rankings, the expert-selection criteria, the pruning mask in `masks/`,
the pathology guards, and the CoE escalation ladders. The surgery procedure cannot run without its
mask, and the mask cannot be produced without the activation data and ranking code. Together these
form one code object that produces this model from an MIT-licensed parent. The set of weights in this
GGUF is therefore CoE work under PolyForm.

### Upstream components

The parent checkpoint, [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
is MIT licensed. Its tensors are carried into this release unmodified and remain under the MIT licence in
detail; see the [upstream LICENSE](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE).
The MIT licence does not extend to the CoE selection, the mask, or the resulting set of weights.

The CoE-coder runtime — which ships this specialist as its python `python-b` slot — is a separate repository and is not part of this release: [JThomas-CoE/CoE-coder-16GB](https://github.com/JThomas-CoE/CoE-coder-16GB).
