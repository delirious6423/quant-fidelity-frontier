---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - deepseek
  - deepseek-v4.1
  - moe
  - dgx-spark
---

# DeepSeek-V4.1-Flash-1017 EXL3 2.7 bpw

[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`,
quantized with EXL3 trellis coding to fit two NVIDIA DGX Sparks (256 GB together). The routed experts average
**2.70 bits per weight** (2.90 per decoded token), with each expert's bit width chosen from measured sensitivity. The
dense linears and the LM head are also EXL3.

Method and serving stack: [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41)
(Apache-2.0), the pipeline behind `sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw`, applied to the 1017 checkpoint with a
native-1017 calibration set. Full recipe, patches, logs and raw results:
[sybil-solutions/deepseek-v4.1-flash-1017-exl3-spark](https://github.com/sybil-solutions/deepseek-v4.1-flash-1017-exl3-spark).

## Fidelity

KL divergence against the native 1017 model, teacher-forced on sudoingX's KLD panel (199 conversations, 254,846 scored
assistant positions, sha256 `39e0db2f…`). The reference is DeepSeek's own `inference/model.py`, streamed layer by
layer on an H200.

| metric | value |
|---|---|
| KLD, mean over positions (nats) | **0.0304** (95 % CI 0.0268-0.0343) |
| KLD, 8 domains weighted equally | 0.0512 |
| KLD p99 | 0.435 |
| top-1 agreement | 94.95 % |
| panel perplexity, native -> this quant | 1.6010 -> 1.6231 |

| domain (share of positions) | KLD | top-1 |
|---|---|---|
| code (65 %) | 0.0182 | 95.83 % |
| math (4 %) | 0.0230 | 94.72 % |
| agentic (7 %) | 0.0277 | 94.96 % |
| long context (6 %) | 0.0379 | 95.20 % |
| knowledge (6 %) | 0.0568 | 92.05 % |
| writing (2 %) | 0.0795 | 91.56 % |
| multilingual (3 %) | 0.0827 | 93.35 % |
| chat (7 %) | 0.0835 | 90.68 % |

sudoingX's 2.7 bpw quant of the previous checkpoint (`2cba9e42`) scores 0.0243 / 95.57 % on the same panel at the
same budget. Because the base models differ, that is context, not a head-to-head result. The panel's replies were
written by the previous checkpoint. 1017 also showed 1.75-3.7x more EXL3 error energy per bit width than the previous
checkpoint in our measurements.

## Contents

| part | encoding |
|---|---|
| 15,360 routed experts (layers 0-39) | EXL3, per expert K2 / K3 / K4 / K5 = 6,278 / 7,447 / 1,600 / 35; super experts K5, layers 37-39 at K3 or more (`exl3_moe_plan.json`) |
| attention + shared-expert linears | EXL3 K4 (`wo_a` K5) |
| LM head | EXL3 K4 |
| DSpark drafter (`mtp.0-2`) | official precision (FP8 / MXFP4) |
| norms, router, embeddings, vision, compressors | official 1017 tensors |
| Engram tables (203 GB) | **not included**: download DeepSeek's two shards into the same folder (below) |

About 199 GB in this repo (110 files). `MANIFEST.sha256` covers every file, including the two Engram shards.

## Calibration

927 sequences, 1,216,384 tokens, all native DeepSeek-V4.1-Flash-1017 replies (no output cap) in 2,048-token windows.
Token shares: agentic 35 %, coding 25 % (TypeScript, Rust, Go, C, C++, HTML, math), religion and philosophy 20 %,
multilingual 10 %, structured output 10 %, plus 1.4 % random tokens so rarely routed experts see inputs. The KLD panel
is disjoint from it.

## Serve on 2x DGX Spark

Use sudoingX's v1.1 serving stack (vLLM `6e517b15` for sm_121, patch series, exllamav3 1.6.0, `vllm_exl3_moe`
plugin). A CI-built, attested image of it is `ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark`.

```bash
# both Sparks, same absolute path, about 400 GB free
hf download 0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw --local-dir ~/models/dsv41-1017-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/dsv41-1017-exl3
(cd ~/models/dsv41-1017-exl3 && sha256sum -c MANIFEST.sha256)

# control machine with passwordless ssh to both
git clone https://github.com/sudoingX/dgx-spark-deepseek-v41 ~/dsv41_recipe && git -C ~/dsv41_recipe checkout cc59c360
IMAGE=ghcr.io/sybil-solutions/deepseek-v4.1-flash-exl3-spark@sha256:5817414f92e9bec0406dfc4575e16b282209f0cb1ead16fe53d51bac7ac9e9e4 \
HEAD=<rank-0 host> WORKER=<rank-1 host> MODEL_DIR=$HOME/models/dsv41-1017-exl3 \
  ~/dsv41_recipe/scripts/launch/dsv41_tp2.sh up
```

The API is OpenAI-compatible on :8000 with DeepSeek-V4.1 reasoning and tool-call parsing, a 262,144-token context
and DSpark speculative decoding. Serving status and measured speed: see the recipe repo.

## Credits

- sudoingX: the quantization method, allocator, EXL3 harness, serving patches and decode kernels.
- turboderp: [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT).
- bertholomus: TensorFold's RDMA ring, L2 prefetch and prompt kernels (Apache-2.0), shipped in sudoingX's plugin.
- DeepSeek: the model (MIT).
