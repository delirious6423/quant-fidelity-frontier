---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8
- pi
- mtp
- independent-quantization
- imatrix
---

# Qwen3.8-27B pi adapter — independent quantization with the Unsloth tensor layout

**Benchmark status: IN PROGRESS.** This repository contains an independently produced GGUF of the pi-adapted model, with the exact tensor names, dimensions, and precision assignments found in the pinned Unsloth `UD-Q4_K_XL` reference. It uses Unsloth's public importance matrix. This release is by **groxaxo**; it is not an official Unsloth release or an endorsement by the upstream authors.

The artifact has passed structural/quantization verification and paired single-GPU runtime smoke tests with native MTP. Those checks establish loading, tool-call plumbing, and speculative decoding; they do **not** establish benchmark accuracy or validate the original model card's efficiency claim. Local accuracy and token-use measurements appear below as the full benchmark progresses. Missing grades remain explicitly missing.

## Artifact and exact tensor allocation

| Item | Verified value |
|---|---|
| File | [Qwen3.8-27B-pi-adapted-UD-Q4_K_XL.gguf](Qwen3.8-27B-pi-adapted-UD-Q4_K_XL.gguf) |
| File size | 17,659,406,112 bytes |
| SHA256 | `6734b2daa8a66167d19b9fef100aa7cebfc0172fe49cf8b3231e9dbc52224bd7` |
| Tensor count | 866, including 15 native MTP tensors |
| Tensor-payload weighted bits per parameter | 5.167777; includes block-quantization overhead, excludes GGUF metadata/alignment |
| Calibration | Public `imatrix_unsloth.gguf`: 496 entries, 1,251 chunks |
| Converter/runtime | llama.cpp `51831d47d0a47b8a7d07c9ff262ebe4b0a211790` |
| Tested llama-server binary SHA256 | `4cfe622d37d808037a54cfef72990ef6d1241732858ce8cd1306f9cc3cdfc870` |

`UD-Q4_K_XL` describes a mixed tensor allocation here, not uniform four-bit storage. The table is generated from the actual output file and checked against every tensor in the pinned reference:

| Type | Tensors | Parameters | Payload bytes |
|---|---:|---:|---:|
| F32 | 360 | 2,671,616 | 10,686,464 |
| IQ4_NL | 9 | 765,460,480 | 430,571,520 |
| IQ4_XS | 91 | 7,319,060,480 | 3,888,250,880 |
| Q3_K | 1 | 89,128,960 | 38,297,600 |
| Q4_K | 37 | 3,279,421,440 | 1,844,674,560 |
| Q5_K | 199 | 12,242,124,800 | 8,416,460,800 |
| Q6_K | 50 | 3,426,222,080 | 2,810,572,800 |
| Q8_0 | 119 | 196,608,000 | 208,896,000 |


Full tensor names, dimensions, type IDs, payload counts, calibration checks, and merge verification are in [quantization verification](provenance/quantization-verification.json), [the merge manifest](provenance/merge-manifest.json), and [the reference map](recipe/reference/reference_tensor_map.json). No MTP weights were discarded to make the model fit.

## Sources, calibration limits, and changes

| Source | Immutable revision / hash |
|---|---|
| [Qwen official BF16 weights](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| [harness-hub pi LoRA](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/tree/7d42b119e2d4d475fa11270634b1ef98d1cd5aee) | `7d42b119e2d4d475fa11270634b1ef98d1cd5aee` |
| [Unsloth GGUF and importance matrix](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) | `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` |
| Unsloth baseline GGUF SHA256 | `2a5f3565f39cf885103da591182c1266da64745737b42b452fd3e675369f1211` |
| Public importance-matrix SHA256 | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` |

All source shard hashes and additional source-file pins are included in [the source lock](recipe/reference/sources.lock.json). [Publication provenance](provenance/publication.json) records the source script/report hashes before removal of private host identifiers.

The rank-32, alpha-64 adapter was merged into the original BF16 checkpoint on CPU as `W + 2 * B @ A` in FP32, followed by one BF16 rounding. All 256 LoRA pairs were mapped and materialized. Conversion preserved the 15 unadapted native MTP tensors, with byte comparisons after the appropriate GGUF transformations. Optional shared-input tensor fusion was disabled in the conversion process to preserve the reference's 866-tensor representation. The standalone merged BF16 intermediate was then quantized using the exact per-tensor override file.

**Calibration limitation:** this uses the released base-model calibration matrix, not an importance matrix recalibrated on the pi-adapted weights. The public matrix contains 496 entries while the reference GGUF records 497; matching the tensor allocation does not reproduce unpublished Unsloth Dynamic quantization internals. The public matrix has no entries for the output head, token embedding, and some native MTP projections. The quantizer still completed with the requested allocation; the actual warnings are preserved here and in [the quantizer log](provenance/quantize.log):

```text
llama_model_quantize_impl: 8 quantizable tensors have no imatrix data:
====== llama_model_quantize_impl: did not find imatrix data for output.weight
====== llama_model_quantize_impl: did not find imatrix data for token_embd.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.attn_k.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.attn_output.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.attn_q.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.attn_v.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.ffn_down.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.ffn_gate.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.ffn_up.weight
====== llama_model_quantize_impl: did not find imatrix data for blk.64.nextn.eh_proj.weight
```

Benchmark tasks, reference trajectories, and generated solutions were not used for calibration or merging.

**Controlled chat template:** the converted GGUF and Unsloth GGUF embed different template metadata. The experiment explicitly supplies the identical official Hugging Face [chat template](recipe/chat_template.jinja) to both servers, SHA256 `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`. This is the common base/adapter source template. It is a deliberate controlled condition for the comparison; the Unsloth file's embedded template is not the template exercised in these measurements. Any explicit tokenizer metadata overrides are identical for both servers, listed in [publication provenance](provenance/publication.json), and included in the command below. Neither GGUF was modified after quantization verification.

**Tokenizer metadata alignment and shared NFC limitation:** the two llama.cpp arms produced identical token IDs on all nine checked fixtures. Both differed from the Hugging Face tokenizer on the two fixtures containing decomposed accents. The pinned Qwen BPE implementation does not apply `tokenizer.ggml.normalizer.nfc`, even when its metadata value is true. The overrides therefore align metadata; they do not establish full tokenizer parity with Hugging Face or the original vLLM evaluation. See [the tokenizer control evidence](provenance/tokenizer-control.json) for the fixtures, hashes, exact differences, and implementation pointers.

## Single RTX 3090 runtime and MTP evidence

The paired experiment uses an Ubuntu i7-12700 machine: the original Unsloth GGUF on physical GPU 0 and this adapted GGUF on physical GPU 2, with **one RTX 3090 24 GB per model**. Each server has 131,072 context tokens, Q8 K/V caches, Flash Attention, one slot, full GPU offload, no context shifting, and native shared-weight MTP with up to three draft tokens. The runtime does not silently reduce context or disable MTP.

| Model | Drafted tokens | Accepted draft tokens | Acceptance |
|---|---:|---:|---:|
| baseline | 1183 | 922 | 77.937% |
| adapted | 1371 | 1085 | 79.139% |


The [sanitized runtime smoke report](results/runtime-smoke.json) includes the exact launch arguments, actual context, text and parsed tool-call checks, per-request timing, native draft counters, and sampled board-memory measurements. Memory peaks are samples, not an allocation tracer. Smoke throughput is a short plumbing measurement, not the token efficiency of the official tasks.

The pinned llama.cpp runtime uses probabilistic drafting and target-distribution rejection sampling; see [speculative decoding](https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/common/speculative.cpp) and [sampling](https://github.com/ggml-org/llama.cpp/blob/51831d47d0a47b8a7d07c9ff262ebe4b0a211790/common/sampling.cpp). This aims to preserve the configured target distribution. Quantized weights, random draws, and different kernels mean these results are not bit-identical to the original BF16/vLLM experiments.

### Queue-managed launch and model access

This experiment uses the host's existing **GPUQ-managed shared proxy**. GPUQ assigns one physical GPU per model, and the shared proxy lazily starts the backend when the alias receives a real request. Its idle watchdog may unload a backend after 300 seconds. The experiment does not change that policy or send periodic keepalive generations.

The actual [two-family registration snippet](recipe/runtime/proxy-family-entries.json), [queue launch commands](recipe/runtime/queue-launch-commands.json), and [foreground role launcher](recipe/runtime/serve_role.py) are included. `serve_pair.py` supplies the common command builder; it is not used to directly start the experiment's backends. The effective backend argv is preserved in the [smoke report](results/runtime-smoke.json). Tested batch/ubatch sizes are 512/128 with 8 CPU threads per server. Native MTP, Q8 K/V, and the identical template/metadata controls described above are mandatory in those launchers.

Build the pinned llama.cpp revision with CUDA; the tested build used CUDA 12.8:

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 51831d47d0a47b8a7d07c9ff262ebe4b0a211790
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --target llama-server llama-quantize -j 8
```

On the experiment host, model clients use **`http://127.0.0.1:12434/v1`**. The private backend ports are implementation details of the registered families. After the two additive family entries are registered with the existing proxy and queue, a normal request is:

```bash
curl http://127.0.0.1:12434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-pi-udq4xl-mtp","messages":[{"role":"user","content":"Explain Python dictionary key requirements."}],"max_tokens":8192,"temperature":0.8,"top_p":1.0,"top_k":0,"min_p":0.0,"repeat_penalty":1.0,"chat_template_kwargs":{"enable_thinking":false}}'
```

The baseline alias is `qwen38-baseline-udq4xl-mtp`. Thinking arms set request `chat_template_kwargs` to `{"enable_thinking":true,"reasoning_effort":"xhigh"}` and maximum output to 32,768. Sampling values are explicit: embedded GGUF defaults differ and must not decide the experiment's sampling.

### Recreate the independent quantization

Download the pinned BF16 and adapter source files above plus `imatrix_unsloth.gguf`. Use the included [merge script](recipe/merge_lora_to_gguf.py) and tensor override file; `BASE`, `ADAPTER`, and `IMATRIX` below are local paths to those exact sources.

```bash
python recipe/merge_lora_to_gguf.py --llama-dir llama.cpp --self-test
python recipe/merge_lora_to_gguf.py --llama-dir llama.cpp \
  --base "$BASE" --adapter "$ADAPTER" --outfile merged-bf16.gguf \
  --manifest merge-manifest.json --reference-map recipe/reference/reference_tensor_map.json
llama.cpp/build/bin/llama-quantize \
  --imatrix "$IMATRIX" --tensor-type-file recipe/reference/reference_tensor_types.txt \
  --max-buffer-size 512 merged-bf16.gguf Qwen3.8-27B-pi-adapted-UD-Q4_K_XL.gguf Q4_K_M 12
```

`Q4_K_M` is the quantizer's nominal fallback argument; the supplied exact-name override file determines every quantized tensor's actual precision. Reproducibility requires the pinned converter, source files, merge arithmetic, and matrix, not just the output filename. Source host paths and GPU UUIDs in published copies are redacted; configure device mappings before using the bundled paired supervisor on another machine.

## Benchmark protocol and current results

The requested scope is all **51 SWE-bench Pro hard tasks plus 89 Terminal-Bench 2.1 tasks**, with four arms: base/no-thinking, adapted/no-thinking, base/xhigh, adapted/xhigh. The no-thinking pair runs first over the full suite; the thinking pair follows. Terminal-Bench starts before SWE-bench. The original fixed comparison subsets of 86 and 83 Terminal-Bench tasks are also reported alongside full coverage.

The harness is pi **1.5.2** (confirmed in released upstream trajectories). This local runner pins Harbor **0.27.0**; the original experiment's Harbor version was not published. Settings are temperature 0.8, top-p 1, no top-k/min-p filtering, 100 model calls, 8,192 maximum output tokens per no-thinking call and 32,768 per thinking call, with the original task timeouts. Docker tasks can access public networking. No paid inference service is used. Concurrent tasks have bounded CPU/memory and no host home, credentials, GPU, or Docker socket mounts.

Task definitions are pinned to [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862) and [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1/tree/7131e4375048a0e408a8fb404b5f499d726b695b). All task IDs match the released upstream experiment. The authors' SWE Harbor mirror could not be accessed (HTTP 401 without authentication and HTTP 404 with the available Hugging Face account), so the public Scale definitions are used; byte identity of those verifiers to the authors' mirror cannot be established. Exact IDs, exclusions, hashes, versions, and commands are in [the benchmark runner](recipe/benchmark/README.md).

One scored attempt is retained per model/task. Infrastructure failures may be retried at the unchanged seed, up to two times; scored model failures are retained. Model-side context/request errors are not silently treated as infrastructure. Reports separate unrun, infrastructure, ungraded, and graded tasks, show Wilson score intervals and paired exact McNemar tests, and provide token comparisons both over paired graded tasks and over tasks both models solved. Missing grades are never converted into zero scores. Original BF16/vLLM scores from the [adapter model card](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) are upstream claims, not measurements from this quantized run.

**Predeclared resource policy:** after container/agent setup, each attempt makes one explicit non-scored readiness generation through the shared proxy before Harbor starts the original fixed 3,600/7,200-second agent clock. The clock is never paused or expanded. Read-only GPUQ job and backend process identities are checked immediately before and after actual model requests. A reclaim, reload, cancelled owner, or changed lifetime during such a request makes the attempt infrastructure-contaminated regardless whether its verifier returns 0 or 1. That raw grade remains diagnostic and is excluded from model accuracy; the same bounded, unchanged-seed retry policy applies. An unload after the final model call while verification runs does not invalidate the completed model work. Reports expose these exclusions and pre-admission wall time; queue duration is left unknown when evidence cannot separate it from other time. No slow-first-token heuristic changes scores.

# Qwen3.8 pi adapter: matched Unsloth-style Q4 evaluation

**Status: Incomplete — do not interpret this as a full benchmark result.**

These are local GGUF runs. The source model card used BF16 weights served by vLLM. Quantization, runtime, CPU quotas and local hardware differ. All rates below show their observed denominators; ungraded infrastructure failures are kept separate.

| Benchmark | Arm | Graded / expected | Solved | Observed pass rate | Wilson 95% interval | Infra / review |
|---|---|---:|---:|---:|---:|---:|
| swe-hard51 | base-no-think | 0 / 51 | – | Not measured | – | 0 / 0 |
| swe-hard51 | adapted-no-think | 0 / 51 | – | Not measured | – | 0 / 0 |
| swe-hard51 | base-think | 0 / 51 | – | Not measured | – | 0 / 0 |
| swe-hard51 | adapted-think | 0 / 51 | – | Not measured | – | 0 / 0 |
| terminal-bench-2.1 | base-no-think | 7 / 89 | 3 | 42.9% | 15.8%–75.0% | 4 / 0 |
| terminal-bench-2.1 | adapted-no-think | 7 / 89 | 2 | 28.6% | 8.2%–64.1% | 4 / 0 |
| terminal-bench-2.1 | base-think | 0 / 89 | – | Not measured | – | 0 / 0 |
| terminal-bench-2.1 | adapted-think | 0 / 89 | – | Not measured | – | 0 / 0 |

## GPUQ lifecycle and exclusions

Each attempt performs one non-scored readiness generation through the shared proxy after environment/agent setup and before the original fixed agent clock. The clock is never paused or extended. A changed or reclaimed backend observed immediately before/after an actual model call marks the attempt infrastructure-contaminated regardless whether the verifier returned 0 or 1. Raw grades remain diagnostic only; unchanged-seed retries are bounded to two. The JSON reports contaminated attempts, final excluded tasks, and admission wall time (including queue, load and readiness generation). Exact midcall queue durations are left unknown when lifecycle evidence cannot determine them. A backend unload after the last model call, during grading, does not contaminate the attempt.

## Fixed original subsets

The JSON report also includes the original 51-task SWE subset and 86/83-task Terminal-Bench subsets. Their score remains null until every task in that subset is graded on both compared arms. New infrastructure exclusions never silently change those denominators.

## Statistical comparisons

Paired changes use the exact two-sided McNemar/binomial sign test on newly solved versus lost tasks. Wilson intervals describe individual pass rates. The single-attempt, fixed-seed experiment does not estimate seed-to-seed variation. Token-efficiency results separately report all matched graded tasks and the matched subset solved by both arms, with missing usage counts reported explicitly.


Machine-readable output is in [results.json](results/results.json), with [run status](results/run-status.json). The final requested-scope publication adds sanitized trial evidence; it can still be incomplete if infrastructure prevented grading. The evidence includes generated benchmark solutions and must be kept out of training and calibration datasets.

## License and attribution

The Qwen base model, harness-hub adapter, and Unsloth reference repository declare **Apache-2.0**. The upstream base [LICENSE](LICENSE) is included, along with [NOTICE](NOTICE). Credit belongs to Qwen for the original model, harness-hub and the pi/harness contributors for the adaptation and experiment, and Unsloth for the reference tensor allocation and public calibration matrix. This repository records its independent merge, conversion, quantization, and evaluation changes. Benchmark task assets, third-party software, and generated evidence retain applicable upstream terms; this model's metadata does not relicense them.
