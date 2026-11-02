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
  - distillation
  - mixed-precision
  - pruning
  - experimental
---

# Qwen3.8 WeeAIResearch Flash Next MAX v2

## Three languages. Five working programs. 30 functional checks passed.

**MAX v2 is the strongest tested release from our pruning-and-recovery experiment: restored experts, targeted Q4 upgrades, and teacher-guided output distillation—merged directly into one GGUF.**

The headline gain is concrete: the previous release candidate's Python to-do app failed to start. MAX v2 generated a working version that passed **6/6 checks**, while retaining **6/6 in C# and 6/6 in TypeScript**. It also passed six reverse-lines regression checks and six checks on a fresh Python squares task.

**These are small functional smoke tests, not a public benchmark.** Four of the five prompts were development/regression tasks; only the squares prompt was new to fitting and candidate selection. Each task used one greedy generation. The complete prompts, unedited outputs, source files, build logs and test results are published below.

| MAX v2 at a glance | |
|---|---|
| Latest GGUF | [Qwen3.8-WeeAIResearch-Flash-Next-69B-A6.6B-MAX-v2.gguf](Qwen3.8-WeeAIResearch-Flash-Next-69B-A6.6B-MAX-v2.gguf) |
| Logical non-lookup parameters | **68.73B** — approximately 69B |
| Active non-lookup parameters | **~6.64B per token**, shape-based estimate |
| Lookup parameters | **51.20B additional learned values** |
| File size | **65.16 GB / 60.68 GiB** |
| Layers / routed selections | **48 / 10 experts per layer per token**, plus shared experts |
| Added recovery | **720 restored expert slots in Q2**, plus **8 targeted Q4 promotions** |
| Distillation deployment | **Merged Q6 output projection; no sidecar required** |
| Tested inference | Patched llama.cpp, CPU, memory-mapped weights |

The repository URL retains its legacy “65B” name. The new file reports the recovered model's approximately **69B non-lookup** parameter count; including the large lookup table, the logical total is **119.93B**. Quantization alignment padding is excluded from these logical counts. The active figure excludes lookup rows and masked work and is an estimate, not a full execution audit. See [parameter-count.json](parameter-count.json).

> **Runtime requirement:** this GGUF uses variable expert banks and four quantization tiers. Use the supplied [llama.cpp patches and build instructions](runtime/README.md). Stock llama.cpp and unmodified frontends are not established as compatible.

## A recovery you can inspect

![Coding recovery: prior model versus MAX v2 on the same three CLI apps](graphs/coding-recovery.png)

The programs were compiled or syntax-checked, then executed against scripted input/output cases. The original generated code was scored without fixes. C# uses .NET 10; TypeScript passed strict type checking and was executed under Node.js; Python passed runtime tests rather than only syntax compilation.

| Generated program | Result | Evaluation role |
|---|---:|---|
| C# in-memory to-do CLI | **6/6** | Reused development/regression prompt |
| Python in-memory to-do CLI | **6/6** | Reused development/regression prompt |
| TypeScript in-memory to-do CLI | **6/6** | Reused development/regression prompt; working ending used as preservation data |
| Python reverse-lines CLI | **6/6** | Regression prompt; corrected reference included in training |
| Python integer-squares CLI | **6/6** | **Fresh prompt**, not used for fitting or selection before its run |
| **Total** | **30/30** | **Five programs, one generation each** |

![Five-program functional results; the fresh prompt is highlighted](graphs/functional-results.png)

The app checks cover empty state, adding one and multiple tasks, unknown commands, EOF and quit. Reverse-lines checks include spaces, blank lines, Unicode and a missing final newline. The fresh squares checks include negatives, zero, whitespace, empty input, multiple lines and large integers.

All five latest runs ended naturally, with no observed system swap growth. Sampled process RSS was approximately **11.3–11.7 GiB on the test machine** for these short runs; that is not a full-residency requirement or a promise for other workloads. Local tests used an Apple M5 Pro with 24 GiB unified memory, CPU-only inference, a 1,024-token context and greedy sampling.

**Inspect the evidence:** [evaluation report](evaluation/README.md) · [machine-readable results](evaluation/summary.json) · [generated programs and tests](evaluation/generated/).

## Closer to the original's predictions

We measured the original all-expert Q4 model and the candidate on matched token prefixes, then used the original's logits to train the final projection. On **23 reserved Python states**, the selected merged, quantized model improved:

- **OG top-token agreement: 91.3% → 95.7%.**
- **Teacher KL divergence: 0.214 → 0.116**, about a **46% reduction**; lower means more similar prediction distributions.

![Original-model prediction agreement and KL divergence](graphs/og-agreement.png)

These are teacher-forced output-projection measurements, not broad task accuracy. The validation script was reserved from this fit but had appeared in earlier research. The calculation uses dequantized Q6 projection weights and excludes runtime activation quantization. Full-program runtime tests are reported separately above. The teacher is the original **Q4_K_M GGUF**, not BF16 or FP16.

## What changed in v2

MAX v2 brings together three measured recovery steps:

1. **Restore useful capacity.** Sixteen fit-ranked missing original experts were added to each of layers 3–47, for **720 restored slots in Q2_K**. The first three layers retain the 464-expert source layout. Later layers contain 162–337 experts each.
2. **Spend precision where it helped.** Four experts in layer 4 and four in layer 25 were promoted to Q4. On the old cached diagnostic inputs, those two layers' routed-output error fell approximately **76.8% → 7.6%** and **74.0% → 22.8%** respectively. These selected local results are not answer-accuracy estimates. The later layer-26 experiment regressed a token decision and is **not included**.
3. **Distill the final predictions.** A ridge-fitted correction to the final projection uses original-model logits on coding prefixes, error-focused Python entry-point states, a corrected reverse-lines reference, and a TypeScript base-model preservation target. The v2 fit used 204 training states; a **0.5-strength correction** was merged into the existing Q6_K output tensor. This avoided a TypeScript regression seen with the full-strength update. Intermediate-layer sidecars are not included.

The merged distillation adds **no parameters and no file-size increase** over its recovered Q4-targeted source. The improvement is carried by the weights in this download.

The initial pruning selection came from a single C# password-manager prompt, so its task coverage was narrow. v2 improves that starting point through restoration and coding-focused recovery; it does not turn the initial calibration into a broad training corpus.

## Designed for disk-backed experimentation

![Weight storage breakdown](graphs/weight-storage.png)

| Weight category | Storage |
|---|---:|
| Learned token-pattern lookup table | **35.20 GB** |
| Routed expert weights | **26.66 GB** |
| Other model weights | **3.29 GB** |

The lookup gathers 16 small Q5 rows per token from hashed two- and three-token patterns and mixes their learned features into the second layer. It does not scan the whole table per token. Memory mapping lets the OS load and cache pages as needed; disk-backed weights still consume resident/cache memory when touched, and cold random reads can reduce speed. This release is **not a 16 GB fully resident model**.

The expert mix includes Q4_K/Q4_0, Q3_K, Q2_K and Q1_0, with original Q8 down weights in the preserved early layers. The output projection remains Q6_K. [Exact storage by quantization](evaluation/weight-storage.json).

## Download and run

```bash
hf download WeeAIResearch/Qwen3.8-WeeAIResearch-Flash-Next-65B-A6.6B-MAX \
  Qwen3.8-WeeAIResearch-Flash-Next-69B-A6.6B-MAX-v2.gguf \
  README.md LICENSE SHA256SUMS runtime/README.md \
  runtime/variable-runtime.patch runtime/four-tier.patch runtime/LLAMA_CPP_LICENSE \
  --local-dir .
```

Build the compatible runtime using [runtime/README.md](runtime/README.md), then launch chat:

```bash
./llama.cpp/build/bin/llama-cli \
  -m ./Qwen3.8-WeeAIResearch-Flash-Next-69B-A6.6B-MAX-v2.gguf \
  -c 1024 -b 2 -ub 1 -t 4 -tb 4 -ngl 0 \
  --no-repack --fit off --no-warmup --load-mode mmap \
  --reasoning off --system-prompt "Always respond in English."
```

Download the runtime patch files from this repository as well; no additional model adapter is required. GPU quality and compatibility are not validated for this release. The optional fast CPU patch has a separate experimental status described in the runtime guide; no new v2 throughput claim is made here.

Verify the download against [SHA256SUMS](SHA256SUMS):

```text
70a99659373f37386ea9055f54964c07d005ba05c99d9535e35b6aae1c5ea82a
```

## Scope and next steps

This is an experimental coding-recovery release with reproducible small-program results. The evaluation is partly tuned through development, and only one final prompt was fresh. It does not establish broad coding reliability, security, long-context quality, general intelligence, or equivalence to the original. Use compilation and tests when using generated code.

The next useful evaluation is a larger, untouched multi-language suite with multiple prompts and seeds. Model and test provenance are published so that future improvements can be measured against this release.

## Sources and license

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) through the [lmstudio-community Q4_K_M GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF). See the included [Qwen Community License 1.0](LICENSE). Runtime patches retain the [llama.cpp MIT license](runtime/LLAMA_CPP_LICENSE).

The prior release is available through [its pinned repository revision](https://huggingface.co/WeeAIResearch/Qwen3.8-WeeAIResearch-Flash-Next-65B-A6.6B-MAX/tree/176d02800e48c3e900ae38cee190e68fd69dc2b0).
