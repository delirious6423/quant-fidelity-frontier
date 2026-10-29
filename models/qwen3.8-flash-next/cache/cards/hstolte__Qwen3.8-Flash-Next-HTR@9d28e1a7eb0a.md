---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
inference: false
tags:
- htr
- halo-trellis
- quantized
- strix-halo
- gfx1151
- custom-runtime
- release-candidate
---

![Qwen3.8 Flash Next — Halo Trellis](media/hero.png)

# Qwen3.8-Flash-Next-HTR

HawgAuto's Halo Trellis quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), built for **AMD Strix Halo / Radeon 8060S**.

## At a glance

- **Measured generation:** 53.73 tokens/s native decode and 44.61 tokens/s end-to-end with MTP3 in the saved coding/agent-style screen.
- **Measured prompt processing:** 1,495.66 tokens/s on warmed 8K prompts with MTP off.
- **Vision weights included:** 333 BF16 vision tensors and processor assets are retained. Image-input quality and video execution are not qualified by the text tests.
- **MTP3 included:** speculative decoding proposes three tokens at a time to accelerate generation.

The speed highlights come from an earlier research-stack comparison, not a new benchmark of the packaged image. They use different modes; the comparison below keeps those modes separate.

> **Upload in progress — incomplete release:** the 173.76 GB checkpoint is being uploaded and is not yet verified complete. Wait for the upload-complete notice before downloading for inference. The exact tested runtime source and image are staged privately; their links require access until runtime publication is finished.

## Required runtime and tooling

HTR is a custom format, not GGUF. It requires the HTR-aware runtime rather than a stock llama.cpp, Transformers or vLLM installation.

- [HTR format and quantization tools](https://github.com/HawgAuto/HTR)
- [vLLM HTR serving runtime](https://github.com/HawgAuto/vllm-htr/tree/63ba082fbf6d204dba64bfb3803ccf48ad01fff1), source tag `htr-qsa1600-release-20261028` (**currently private**). This is the required integration, not the older general-purpose vLLM fork.
- Tested image: `ghcr.io/hawgauto/vllm-htr:qsa1600-release-20261028` (**currently private**), pinned by digest in the guide.

**[Download and run on Strix Halo →](docs/RUNNING.md)** — complete checkpoint download, Docker command, settings and cold-start expectations. The default is **QSA1600 + probabilistic MTP3, thinking enabled**, temperature **0.7**, top-p **0.8**, top-k **20**. These are the selected release settings, not the historical benchmark sampling profile. Linux, approximately 128 GB shared memory and a fast local SSD are the intended configuration.

## Quantization recipe

The model uses a mix of precisions rather than treating every component the same.

| Component | Representation |
|---|---|
| Selected dense language-model weights | HTR5 (5-bit) |
| Routed experts | HTR4 (4-bit) |
| Output head | HTR4 (4-bit) |
| Hyperconnections | GPTQ8 (8-bit integer) |
| Vision tower and merger | Original BF16 |
| Other embeddings, norms and supporting tensors | Retained in their component-specific formats |
| Speculative draft | Included for MTP3 |

## Method and lineage

HTR combines calibration-guided error correction with a trellis-based weight format designed for Strix Halo. The goal is to reduce weight storage while keeping the model close to its reference behavior.

Its lineage includes [exlr8](https://github.com/kindlingai/exlr8), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [QTIP](https://arxiv.org/abs/2406.11235). See the [methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) and [full acknowledgments](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md).

![The HTR quantization pipeline](media/method-map.png)

## Measured decode and prefill

Historical screen on **AMD Radeon 8060S / Strix Halo**, with one request at a time. It predates lossless shard consolidation and uses the research runtime with temperature 1.0 / top-p 0.95 / top-k 20. The final package has separate [delivery qualification](evidence/release-qualification.json), not a remeasurement of these rates.

### Text generation

**Decode** measures response generation. **End-to-end** also includes prompt processing and request overhead. All rates are in tokens per second.

| Workload | Mode | Decode | End-to-end |
|---|---|---:|---:|
| Coding & agents (thinking off) | MTP3 | 53.73 | 44.61 |
| Coding & agents (thinking off) | MTP off, checkpoint on | 34.26 | 30.99 |
| Chat (thinking on) | MTP3 | 42.66 | 41.09 |
| Chat (thinking on) | MTP off, checkpoint on | 32.44 | 30.19 |

### Prompt processing (prefill)

Warmed runs, with no cached prompt tokens reused. The MTP-off column uses checkpoint-on mode; MTP3 uses probabilistic drafting. Rates are in tokens per second.

| Prompt length | MTP off | MTP3 |
|---|---:|---:|
| 2,048 tokens | 1,416.68 | 1,076.45 |
| 8,192 tokens | 1,495.66 | 1,050.69 |

MTP3 is faster at generation in these tests; MTP off is faster at processing prompts. These are selected configurations, not identical-output speedup measurements: response lengths and cache settings differed, and many chat responses reached the 512-token limit. The tests measure text speed, not task accuracy or image quality.

[Full benchmark details and settings](docs/SERVING-PERFORMANCE.md) · [Timing data](evidence/serving-performance.json)

## Size and memory

The checkpoint contains **37 weight files**: 23 target safetensors shards, 13 indexed HTR expert shards and one draft safetensors shard. An additional 26 files supply the configuration, tokenizer, processors, manifests and license. Shard consolidation is lossless, not requantization.

| Download component | Files | GB | GiB |
|---|---:|---:|---:|
| Target checkpoint (including PLE and vision) | 23 | 109.027 | 101.539 |
| Routed experts | 13 | 60.705 | 56.536 |
| MTP draft | 1 | 3.978 | 3.704 |
| **Complete checkpoint + supporting files** | **63** | **173.759** | **161.826** |

Exact checkpoint total: **173,759,296,067 bytes**; card, artwork and reports are additional. The largest shard is 4.997 GB. [File sizes and SHA-256](evidence/checkpoint-manifest.json) · [Sizing data](evidence/package-sizing.json)

**PLE is already included in these totals.** Its n-gram embedding tables account for 102.400 GB (95.368 GiB) of the target payload. The runtime reads requested PLE rows from SSD instead of placing the entire table on the GPU. Keep the model on a fast local SSD; download size is not resident GPU memory.

On the final-package text/tool smoke, the engine logged **65.68 GiB model-load allocation**, an **8.0 GiB KV budget**, and **0.19 GiB graph memory**. These are not total-process RAM requirements. Minimum sampled host available memory was **14.61 GiB** on the 128 GB-class test machine; runtime, activations, compilation and filesystem caching also need headroom. The guide uses a **108 GiB container limit**, not a minimum-RAM guarantee. Do not co-load another large shared-memory workload.

## Quantization quality

This matched **4-bit dense-layer comparison on WikiText** illustrates the choice of HTR. Only the selected dense weights change; both methods use the same reference model, whose experts are already quantized.

| Dense-layer format | PPL ↓ | KL ↓ | Top-1 agreement ↑ |
|---|---:|---:|---:|
| BF16 dense-layer reference | 4.295 | 0.0000 | 100.00% |
| HTR 4-bit | **4.308** | **0.0284** | **93.77%** |
| GPTQ 4-bit | 4.344 | 0.0520 | 91.68% |

**PPL** measures next-token prediction on the text; lower is better. **KL** measures divergence from the reference over its top 128 tokens plus a remaining-mass bucket; lower is better. It is not full-vocabulary KL. **Top-1 agreement** is how often the models choose the same next token—not task accuracy. Bold values mark the better quantized result.

HTR stays closer to the reference than GPTQ in this comparison. This illustrative HTR4 study is not the HTR5 dense-trunk configuration shipped here. These are component-level results, not a final score for the complete mixed-precision release. The [full quality report](docs/RECIPE-COMPARISONS.md) covers the other precision and calibration choices, including their tradeoffs; [source metrics](evidence/recipe-quality.json) retain the detailed measurements.

## Download statistics

Full-repository downloads include [`meta.yaml`](meta.yaml) automatically. For selective downloads, include it alongside the checkpoint files so [Hugging Face's normal download counter](https://huggingface.co/docs/hub/models-download-stats) can record the request. No extra client telemetry is added; Hub counts are not a count of complete downloads or unique users.

## Release verification and limits

- **Packaged locally and tested:** consolidated target, sanitized expert shards and draft passed the installed loader/metadata checks. Tensor dtype, shape and bytes were preserved through packaging.
- **Final-image delivery:** thinking-enabled text generation, a structured `tool_calls` response and a tool-result reply passed on the exact image manifest listed in the guide. The short smoke observed 148 accepted tokens out of 189 drafted (78.3%) and 52 QSA union dispatches. This is functional evidence, not a new quality or sustained-speed benchmark.
- **Cold start:** 361 seconds to ready from a fresh workspace; 13 additional Triton kernels compiled during the first two requests. No precompiled quick-start cache is shipped.
- **Limits:** QSA1600 is approximate sparse attention, not numerical parity with full attention. The configured 262,144-token context ceiling is not a completed maximum-length quality/capacity test. No final-composite quality score or image/video execution qualification is claimed.
- **Distribution still in progress:** remote checkpoint hash verification and public runtime access remain pending. The strict smoke controller retained a host-state FAIL caused by external timestamp-only health updates; the separate delivery/route/cleanup audit passed with that adjudication recorded.

[Exact image qualification and scope](evidence/release-qualification.json)

[Release status](evidence/release-status.json) · [Vision and draft inventory](evidence/vision-and-mtp.json)

## License and attribution

Source model: Qwen/Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Quantization/tooling: HawgAuto HTR; serving infrastructure: the separately pinned HawgAuto vLLM/HaloLoom stack.

The weights and derivative are governed by **Qwen Community License 1.0**, preserved verbatim in [LICENSE](LICENSE). This is not Apache-2.0 or the HTR code repository's MIT license. The license contains conditions for commercial Model-as-a-Service and AI Work Assistant use; review the complete text before deployment. Model authorship and capabilities belong to Qwen; HTR does not change those license obligations.
