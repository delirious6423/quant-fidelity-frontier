---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - prismaquant
  - prismascout
  - prismaaqua
  - aqua
  - compressed-tensors
  - nvfp4
  - fp8
  - mixed-precision
  - quantized
  - text-only
  - mtp
  - speculative-decoding
  - blackwell
  - dgx-spark
  - qwen3.8
  - vllm
---

# Qwen3.8-27B-PrismaScout-AQUA-20GB

A text-only, mixed-precision quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), built to fit
inside a strict **20,000,000,000-byte whole-repository limit**. AQUA prices
each body Linear independently, including the activation error that matters for
NVFP4 W4A4. The untied `lm_head` and all eight MTP Linears are FP8 E4M3.
The vision tower is physically absent.

This is a stock `compressed-tensors` checkpoint. Vanilla vLLM serves it
without a PrismaQuant or Gridbook plugin.

**Final local artifact size: `19974332758` bytes. Including the
Hub-managed 1,570-byte `.gitattributes`, the remote repository is
`19974334328` bytes (`19.974334328` decimal GB), leaving
`25665672` bytes below the limit.**

Project and method: **[prismaquant.org](https://prismaquant.org)**

## Measured quality

The gold measurements use the exported bytes, no speculative decoding, and the
same pinned vLLM stack. The BF16 comparisons use the exact Qwen3.8 source
revision recorded under Provenance.

| measurement | exported artifact | BF16 source | quantization cost |
|---|---:|---:|---:|
| KL, all next-token positions (8 × 512 → 4,088) | **0.04022** | — | — |
| KL, confident positions (teacher top-1 > 0.5; 2,063 positions) | **0.02395** | — | — |
| KL p99 / max | 0.34029 / 1.86000 | — | — |
| WikiText-2 test PPL (8,176 scored tokens @ 512) | **9.7312** | 9.3662 | **+3.90%** |
| WikiText-2 mean NLL | 2.27534 | 2.23711 | **+0.03823 nats/token** |
| fixed prompt-suite PPL / p99 NLL | 4.0743 / 1.9380 | 3.9813 / 1.9149 | +2.34% PPL |
| generation coherence suite | PASS, 4 of 4 | — | — |

For KL, every position retains the teacher's top-1,024 probabilities plus one
bucket for all residual probability mass. Mean top-1,024 coverage is 0.9880
(minimum 0.5729). Lumping outcomes can only reduce KL, so this is a lower bound
on exact full-vocabulary KL. The full distribution, including p99 and maximum,
is reported because a mean alone hides the tail.

Calibration windows came from WikiText train. The paired PPL measurement used
WikiText-2 test, so selection and reported PPL do not reuse the same split.

## Direct PrismaScout 3.6 comparison

The predecessor is
[`Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
Both artifacts were benchmarked sequentially on the same DGX Spark, in the same
container, with CUDA graphs, FP8 KV cache, prefix caching enabled, fixed paired
seeds, and two warmups per cell. The random synthetic prompts use the Qwen3.6
tokenizer for both arms so the paired requests have identical fixed token
lengths. Non-speculative cells use three measured repeats; the higher-variance
MTP decode cells use six.

| workload | Qwen3.6 PrismaScout | PrismaScout-AQUA | AQUA / Qwen3.6 throughput |
|---|---:|---:|---:|
| prefill C1 (input tok/s; TTFT) | 3,047.77 ± 0.47; 671.8 ms | 3,035.94 ± 3.11; 674.4 ms | **0.9961×** |
| prefill C4 (input tok/s; TTFT) | 11,529.79 ± 28.31; 675.4 ms | 11,297.28 ± 168.75; 686.0 ms | **0.9798×** |
| decode, no spec C1 (output tok/s; TPOT) | 12.58136 ± 0.00009; 79.40 ms | 12.97836 ± 0.00390; 76.95 ms | **1.0316×** |
| decode, no spec C4 (output tok/s; TPOT) | 48.90230 ± 0.01747; 81.28 ms | 48.99464 ± 0.03962; 81.09 ms | **1.0019×** |
| decode, MTP-3 C1 (output tok/s; TPOT) | 24.6393 ± 1.7580; 40.05 ms | 30.9371 ± 2.1886; 31.85 ms | **1.2556×** |
| decode, MTP-3 C4 (output tok/s; TPOT) | 83.0368 ± 7.0968; 39.93 ms | 95.9148 ± 12.4466; 35.24 ms | **1.1551×** |

Prefill is 2,048 input tokens followed by one output token. Decode is 32 input
tokens followed by 256 output tokens. C4 throughput is aggregate across four
concurrent requests. Ratios above 1.0 favor PrismaScout-AQUA. Every request must
succeed. Throughput values are mean ± sample standard deviation; TTFT and TPOT
are means. All 48 result files pass identity, request, token, failure,
arithmetic, and metric validation. The
non-speculative primary metrics are stable; MTP output-throughput CV is
7.1–13.0%, reflecting seed-dependent speculative acceptance, so its spread is
shown rather than hidden behind the mean.

The predecessor card's public **0.0151 KL is not a matched release baseline**.
Surviving run records identify it as a 2×128 selection-time sanity check, and
the historical evaluator scored only the final next-token prediction of each
window: two positions total. A later served-byte audit measured 0.0551, but
likewise scored only the eight terminal predictions from 8×512 windows.

We therefore restored the exact Qwen3.6 BF16 revision and reran both released
artifacts under the current all-position contract. Each row uses its model's
own BF16 source, the same materialized corpus, 8×512 windows, seed 42, all
4,088 next-token positions, top-1,024 probabilities plus residual mass, the
same pinned vanilla-vLLM stack, and no speculative decoding. AQUA used CUDA
graphs; the restored Qwen3.6 rerun used eager execution.

| served quantization-error audit | PrismaScout-AQUA (Qwen3.8) | PrismaScout (Qwen3.6) |
|---|---:|---:|
| mean KL | **0.04022** | 0.07793 |
| confident-position mean KL | **0.02395** (2,063) | 0.05137 (2,134) |
| KL p99 / max | **0.34029 / 1.86000** | 0.81366 / 4.97123 |

On this matched measurement contract, PrismaScout-AQUA's mean KL is **48.4%
lower**. This measures quantization distortion relative to each model's own
source; it does not compare Qwen3.8 and Qwen3.6 capability. Their tokenizers
also differ, so this is a matched protocol and corpus, not identical token
windows. The result supports the new artifact; it does not prove that AQUA's
activation price alone caused the improvement.

Qwen3.6 remains a useful, supported checkpoint. Existing users are welcome to
continue serving it when its behavior or deployment fit is preferable.

## What the allocator chose

![Per-Linear format allocation across the 64-layer text body, MTP sidecar, and lm_head](allocation-map.png)

Each body column is one layer and each row is a projection. **Violet is NVFP4**
(W4A4, group size 16), **teal is FP8 E4M3** (W8A8, per-channel), and **orange
is BF16**. Blank cells are projections absent from that layer. The panels below
the body show the eight FP8 MTP Linears and the FP8 `lm_head`.

The map is generated by the prismaquant.org website renderer from the
checkpoint's own `config.json` and safetensors headers. It describes the
served bytes, not a pre-export recipe.

| scope | format | Linear count | representation |
|---|---|---:|---|
| body | NVFP4 | 309 | W4A4, group size 16 |
| body | FP8 E4M3 | 179 | W8A8, dynamic activations |
| body | BF16 | 8 | source-precision passthrough |
| fixed auxiliary | FP8 E4M3 | 1 | untied `lm_head` |
| fixed auxiliary | FP8 E4M3 | 8 | MTP Linears |

That is 505 assigned Linears: 309 NVFP4, 188 FP8, and 8 BF16. The body payload
is **5.1596801876 bpp** over **24,350,556,160 quantizable body parameters**.
By convention, body bpp excludes the fixed head, MTP, embeddings, norms, and
other tensors; whole-artifact bytes are the correct comparison across releases.

The tensor payload is **19,950,598,312 bytes**. The source carve has 866
tensors: 850 `model.language_model.*`, 15 `mtp.*`, and
`lm_head.weight`. Its final census contains zero vision tensor names.

## Why AQUA

NVFP4 is W4A4, so a weight-only allocator is blind to an important part of the
served error. AQUA adds an activation-side price for activation-quantizing
formats, using the same probe and calibration contract as the weight price.
Each body Linear was offered `{NVFP4, FP8_E4M3, BF16}`; the allocation was
then solved under the exact serialized-byte budget.

The production render used GPTQ with fixed damping 1.0, static activation
ordering, and joint scale optimization. Quantization was striped across two DGX
Sparks, then exact-unioned and recached before export. Tensor parallelism was
not used for quantization.

The `lm_head` and MTP assignments were fixed and byte-accounted outside the
body bpp. This release reaches 20 GB by removing vision and quantizing those
large auxiliaries while giving the language body 5.160 bpp. For reference, the
served body payload of the historical Qwen3.6 artifact is about 4.764 bpp under
the current quantizable-parameter convention despite its old 5.31 label.

## Serving

Validated on NVIDIA GB10 / DGX Spark with:

- vLLM `0.26.1rc1.dev693+g7f7a32cfe.d20260812`
- FlashInfer `0.6.18`
- compressed-tensors `0.17.0`
- native `FlashInferCutlassNvFp4LinearKernel` and
  `CutlassFP8ScaledMMLinearKernel`
- no loaded vLLM plugins

**Publication gates: PASS (5/5).** Native eager generation, native CUDA-graph
generation, all-position gold KL, paired gold PPL, and the final no-spec HTTP
quality gate all pass for exported model SHA-256
`3191a6a4ea85abcebd007be508d2208e95a65c48314e07f7b79ce439d9d38160`.
The final endpoint gate generated four coherent completions and scored 583
tokens at PPL 4.0743, mean NLL 1.4047, and p99 NLL 1.9380. A separate
eager-plus-graph MTP load/generation smoke also passed.

```bash
vllm serve rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB \
  --host 0.0.0.0 --port 8000 \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The speculative configuration uses the checkpoint's FP8 MTP sidecar. In a
matched 512→128 C1 acceptance run, total draft acceptance was **70.94%** and
first-position acceptance was **80.79%**; decode throughput improved from
12.49 to 23.85 output tok/s (**1.91×**). Remove the
`--speculative-config` line to serve without speculation.

The release smoke uses a 4,096-token model length. The command above is a
practical 32K starting point on a sufficiently large card, not a validation of
the architecture's full advertised context. A 32 GB Blackwell card or larger
is recommended; memory available for KV cache depends on context, concurrency,
and vLLM settings.

## Limitations

- This is text-only. The visual tower is absent and image input is unsupported.
- NVFP4 requires a performant Blackwell serving path. This artifact was not
  performance-gated on pre-Blackwell hardware.
- The head and MTP Linears are FP8, not BF16. Their effect is included in the
  end-to-end quality and acceptance measurements.
- Mean KL is a screen, not a downstream task score; its heavy tail is disclosed
  above. No GSM8K, IFEval, MMLU, coding, tool-use, safety, or multilingual suite
  was run for this release.
- The direct Qwen3.6 benchmark compares serving performance, not model
  capability. Qwen3.6 and Qwen3.8 are different source revisions.
- The recache campaign achieved exact stripe/union parity but had low GPU duty
  cycle during the resident-cache fill. That is a pipeline efficiency defect,
  not a serving-path result.

## Provenance

The included `shipcard.json` binds build identity and every publication gate
to the exported model hash. Publication refuses unless all required slots pass
for that hash.

| item | identity |
|---|---|
| source | `Qwen/Qwen3.8-27B` |
| source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| source content SHA-256 | `766e1c61d21735ffa6fdb64fe6a2847c541d1dacb4f888c959e7d4c5bb20c54b` |
| source tensor payload | 54,641,395,712 bytes |
| assignment semantic SHA-256 | `c11a6646ef00ad472fca8e5ea6d26776c681cb2aed05a3402ef981d271ae289d` |
| `layer_config.json` SHA-256 | `656af501472c2be7c06ee5903c62ab3019eac4e11109b7f6382ec524ce9c4a1c` |
| calibration data SHA-256 | `e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba` |
| producer commit | `5f4f4711a2d2da89da2f6f34f33b22b2b6720cb6` |
| exported model SHA-256 | `3191a6a4ea85abcebd007be508d2208e95a65c48314e07f7b79ce439d9d38160` |

## Citation

```bibtex
@software{tand_prismaquant,
  author = {Tand, Robert},
  title  = {PrismaQuant: production-faithful KL--Fisher bit allocation},
  url    = {https://prismaquant.org}
}
```

Contact: robert.tand@icloud.com
