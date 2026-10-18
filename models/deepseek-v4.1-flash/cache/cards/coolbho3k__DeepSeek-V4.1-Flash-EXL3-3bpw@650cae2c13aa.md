---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
inference: false
tags:
- exl3
- 3bpw
- quantized
- deepseek-v41
- experimental
---

# DeepSeek-V4.1-Flash-EXL3-3bpw

Calibrated **EXL3 3bpw routed-expert quantization** of
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40).
The original BF16 vision transformer and aligner are preserved unchanged.

**Current weights: full-pool refit (2026-09-27).** Every routed expert was
re-fitted with up to 65,536 calibration rows drawn from the full calibration
pool. Against the original model this lowers text KL divergence by 7.6% relative
to the previous upload (see below). Earlier recipe revisions keep downloading the
previous weights at revision `5571a3c9ee09f9495e9118b54e022ac591b85373`.

**Upload status: complete.** See `upload-status.json`; validate downloaded files
against `release-manifest.json` before loading.

**Runtime compatibility:** this is an experimental weights release for a
specialized vLLM/EXL3 integration, not a drop-in stock-vLLM or standalone
ExLlamaV3 checkpoint. Use the published
[two-DGX-Spark serving recipe](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)
for the required runtime and current configuration. A stock `vllm serve`
command cannot load these weights directly.

## What is quantized?

- All 40 backbone layers and 15,360 routed experts use calibrated 3-bit
  EXL3 MUL1 weights.
- The **3bpw label applies to routed experts**, not every parameter or the
  average storage cost of the entire checkpoint. EXL3 metadata and padding
  add storage beyond the packed 3-bit payload.
- Vision transformer and aligner retain their original BF16 bytes. Routers,
  shared experts and other non-routed tensors are copied unchanged from the
  published source, which is itself a mixed-precision/native-FP4 checkpoint.
- Original FP8 engram tables/scales are retained under `engrams/`, for
  SSD-backed lookup rather than loading those tables onto the GPUs.
- Original draft/MTP weights are retained under `draft/`. The current recipe
  does not load them; it uses a separate DSpark drafter.

## Measured quality

The full 40-layer model was compared with the original native-FP4 model on 64
held-out records. These are **teacher-forced** measurements, not a BF16
comparison or a claim about broad downstream task accuracy.

| Text metric (81,880 targets) | Original source | Previous upload | Current weights |
| --- | ---: | ---: | ---: |
| Mean KL divergence from original | — | 0.04475 | **0.04137** |
| Top-choice agreement with original | — | 93.243% | **93.633%** |
| Mean negative log likelihood | 1.33439 | 1.34028 | 1.33957 |
| Perplexity | 3.79768 | 3.82012 | 3.81740 |
| Correct next-token prediction | 70.717% | 70.608% | 70.707% |

The current weights have lower KL than the previous upload on 30 of the 40 text
records (two-sided sign test p ≈ 0.002). Agreement is not task accuracy. The
image slices are small (430 targets, 84 of them answers): image KL fell from
0.253 to 0.226 and top-choice agreement rose from 84.7% to 86.7%, while image
perplexity against the reference answers rose from 586 to 608. They are too
small for a broad vision claim. See `evaluation-summary.json` for every
aggregate and its scope.

Generated-answer checks (123/128 ARC-Challenge, 63/64 GSM8K, thinking disabled)
were measured on the previous upload and have not been repeated. In the current
recipe the 24-image question suite scores 19–20/24 across runtime settings (18/24 for
the previous upload in an earlier runtime), and the text and diagnostic generation checks all
pass.

## Serving with the current recipe

Measured on two DGX Sparks with the published recipe (TP2/DCP2, DSpark
speculative decoding, NVFP4 main KV with searched scales, NVFP4 sparse-indexer
keys, FP8 sliding-window KV, 1,048,576-token context, six sessions):

- Serial decode 28–46 tokens/s depending on the prompt; about 60–65 tokens/s
  aggregate with six concurrent requests.
- Uncached 32K prefill about 1,136 tokens/s.
- Synthetic retrieval passed at 131,062, 524,279 and **1,039,993** prompt tokens;
  the 1M prompt prefilled in 25.6 minutes (676 tokens/s), with at least 2.15 GiB
  of host memory free throughout.

These are recipe measurements on this hardware, not sustained-throughput or
broad long-context quality claims. The DSpark drafter was trained on the
previous weights; its acceptance on the current weights was re-measured in the
same recipe runs above.

Runtime source revisions: vLLM `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`,
ExLlamaV3 `6ff3a17ea7f3d0026b273d43239398d57f71b788`, base checkpoint
`df42c109f1defefcbfcedbe7d905718a12266e40`. Those revisions alone are
insufficient: custom DCP/cache-ownership hooks, SSD engram integration,
streaming loading and bounded GB10 kernels are also required. No clean external
rebuild or stock-runtime compatibility is claimed.

## Files and integrity

| Component | Shards | Tensor payload bytes |
| --- | ---: | ---: |
| Active model | 51 | 215,344,840,128 |
| SSD engram tables/scales | 2 | 202,758,032,400 |
| Retained draft/MTP | 2 | 7,932,874,632 |

Total shard storage including safetensors headers is **426,055,648,464 bytes**.
The refit rewrote model shards 1–50; shard 51 (no routed experts), the engram
tables and the draft weights are byte-identical to the previous upload.
Tokenizer, model metadata, source license and packaging provenance are
included. Calibration examples, activation captures, operational logs,
credentials and runtime container images are not included.

`release-manifest.json` lists the expected size and SHA256 of every payload.
`verify-release.py` verifies a materialized download using standard Python;
run it before loading a GPU model. Obtain the manifest SHA256 from the recipe's
`recipe-lock.json` or another trusted channel, then run:

```bash
hf download coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw --local-dir ./DeepSeek-V4.1-Flash-EXL3-3bpw
python3 ./DeepSeek-V4.1-Flash-EXL3-3bpw/verify-release.py --directory ./DeepSeek-V4.1-Flash-EXL3-3bpw --manifest-sha256 MANIFEST_SHA256
```

Integrity verification is not a quality benchmark or proof of publisher
authenticity. Records under `provenance/` describe packaging and the evaluation
candidate, not a claim that every runtime issue is resolved.

Selected quantization manifest logical SHA256:
`e5952e63ed6ed5d2e28ed69e67d31a4cd6e5766fdcdb4a91b1f5707e0279b9c1`.
Evaluation-candidate package manifest SHA256:
`d11efe43d7666a80098732812fbccfa474703e7be67a8f0afc1c2bda0ff87889`.

The upstream MIT license is retained in `LICENSE`.


## Optional lossless page15 Engram assets

The [two-Spark recipe](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) also provides a new packed SSD layout in [engram-page15-v1](engram-page15-v1/README.md). Original `engrams/*.safetensors`, main weights and draft weights are unchanged by it. Old recipe revisions keep using their pinned original files. New recipes pin the matching reader and packed manifest together; do not replace the original files manually. The layout is lossless, not another quantization. The engram tables did not change in the refit, so the packed assets apply to both weight revisions.

Cold bulk row retrieval was about 2x faster in component tests; measured whole-model decode changed from 30.35 to 30.78 tok/s and warmed 32K prefill was essentially flat. This is not a claim of a 2x serving speedup or broad quality qualification. Full credit to MiaAI Lab / Wesley Young for the native Engram/cache foundation; the derived reader and integration are AGPL-3.0-only in the recipe.
