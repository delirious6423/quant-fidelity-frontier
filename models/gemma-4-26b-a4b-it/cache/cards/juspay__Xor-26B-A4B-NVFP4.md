---
license: apache-2.0
base_model:
  - juspay/Xor-26B-A4B
  - google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: text-generation
inference: false
tags:
  - decision-model
  - sglang
  - mixture-of-experts
  - typed-classification
  - gemma4
  - nvfp4
  - modelopt
  - quantized
---

# Xor 26B-A4B NVFP4 1.1

Xor 26B-A4B NVFP4 1.1 (`xor-26b-a4b-nvfp4-1.1`) is the NVFP4 build of [juspay/Xor-26B-A4B](https://huggingface.co/juspay/Xor-26B-A4B) 1.1, a post-trained [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) for typed decision tasks. The routed experts are 4-bit; everything else is BF16. It is served through the same TypeSafe-compatible `/v1/systemone` API as Xor 26B-A4B.

On Decision Index 0.3 it scores 57.02 against 57.62 for the BF16 build, at 18.8 GB instead of 51.6 GB. On the public JEVBench tiers it answers 199 to 201 of 231 correctly, against 204 for the BF16 build.

## Changes from 1.0

- Weights: quantized from Xor 26B-A4B 1.1 (revision `28ca09c7b76648b24d3561c940d1a7460d12fb8e`, tag `v1.1`) instead of 1.0. The weights are new; nothing is shared with the 1.0 checkpoint.
- Quantization: NVIDIA Model Optimizer's `nvfp4_experts_only` format with its own activation calibration, instead of reusing the activation scales of NVIDIA's `nvidia/Gemma-4-26B-A4B-NVFP4` checkpoint. The KV cache is not quantized, so `--kv-cache-dtype bf16` is no longer needed.
- Interface: requests can carry images or a video. Choice questions are scored in two option orders, and the calibration temperatures changed (see [Interface](#interface)).
- Serving: the bundle starts one single-GPU replica per Compose project.

## Revisions

| Revision | Release |
|---|---|
| `main` | Latest stable release (currently Xor 26B-A4B NVFP4 1.1) |
| `v1.1` | Xor 26B-A4B NVFP4 1.1 (`xor-26b-a4b-nvfp4-1.1`), released 2026-10-14, immutable |
| `v1` | Xor 26B-A4B NVFP4 1.0 (`xor-26b-a4b-nvfp4-1.0`), released 2026-10-01, immutable |

Pin a revision for reproducible results:

```bash
hf download juspay/Xor-26B-A4B-NVFP4 --revision v1.1 --local-dir xor-26b-a4b-nvfp4-1.1
```

## Composition

Gemma 4 mixture-of-experts architecture: 30 layers, 128 routed experts per layer, 8 active per token.

| Part | Precision |
|---|---|
| Routed experts of all 30 layers | NVFP4 W4A4: FP4 (E2M1) weights in groups of 16 with FP8 (E4M3) group scales and an FP32 tensor scale; activations quantized to NVFP4 at run time |
| Attention, dense MLP, routers, embeddings and output, norms, vision tower | BF16, unquantized |
| KV cache | BF16 |

Quantized with NVIDIA Model Optimizer 0.46.1 (commit `b5c76a00fc125eeebda0e2b43ecaf8229d16b9f0`), format `nvfp4_experts_only`, Hugging Face export. Activation scales were calibrated on 64 samples from the `train` split of `abisee/cnn_dailymail` 3.0.0 (revision `96df5e686bee6baa90b8bee7c28b81fa3fa6223de`), at sequence length 512 and batch size 1. `QUANTIZATION.json` records these settings and `hf_quant_config.json` lists the excluded modules.

## Interface

Same request format as Xor 26B-A4B 1.1:

- `noul`: binary probability
- `choice`: categorical decision and full probability distribution (2 to 255 candidates)
- `score`: expected ordinal score and full probability distribution (2 to 10 levels)

Each question is scored as an independent prompt in the model's chat template (thinking disabled), and the answer distribution is a softmax over the option-label logprobs at `Answer: (`. Temperatures are 2.0 for `choice`, 1.7 for `noul` and 1.0 for `score`. Each `choice` question is also scored with its options reversed, and the two probability distributions are averaged before the temperature is applied. `confidence` is the largest option probability. Every answer also returns `x_lp`, the raw label logprobs in option order.

Context is 131,072 tokens; a request whose prompt exceeds it is refused with 422, as is a request body over 8 MB.

A request may include `images` (1 to 8 base64 image data URLs) or `video` (one base64 video data URL; audio is not processed), but not both. Images passed limited smoke checks only; image and video quality have not been evaluated. PDF, audio, free-text generation and general chat are not supported.

## Quick start

Download the release, verify and extract the serving bundle, and start one replica on one GPU:

```bash
hf download juspay/Xor-26B-A4B-NVFP4 --revision v1.1 --local-dir xor-26b-a4b-nvfp4-1.1

(cd xor-26b-a4b-nvfp4-1.1/serving && sha256sum -c xor-26b-a4b-nvfp4-1.1-serving.tar.gz.sha256)

mkdir -p xor-26b-a4b-nvfp4-1.1-runtime
tar -xzf xor-26b-a4b-nvfp4-1.1/serving/xor-26b-a4b-nvfp4-1.1-serving.tar.gz -C xor-26b-a4b-nvfp4-1.1-runtime

export MODEL_DIR="$(cd xor-26b-a4b-nvfp4-1.1 && pwd)"
export GPU_ID=0
export API_PORT=8000
export MODEL_ALIAS=xor-26b-a4b-nvfp4-1.1
cd xor-26b-a4b-nvfp4-1.1-runtime
bash ./run.sh
```

`run.sh` verifies every model file against `checksums.sha256` and the bundle files against `serving.sha256`, then starts the stack and waits until it is healthy. The first start takes a few minutes. The API is then available at `http://127.0.0.1:8000/v1/systemone`:

```bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"xor-26b-a4b-nvfp4-1.1","state":"The customer was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this a billing issue?"}}}'
```

```json
{"model": "xor-26b-a4b-nvfp4-1.1", "answers": {"billing": {"type": "noul", "noul": 0.9793112926932397, "x_lp": [-6.5632, -0.0059]}}, "usage": {"input_tokens": 47, "output_tokens": 1}}
```

Probabilities can differ slightly between runs and deployments.

Stop the stack with `bash ./stop.sh`; the model files are kept. To run more replicas, start one stack per GPU with its own `GPU_ID`, `API_PORT` and `COMPOSE_PROJECT_NAME`.

The setup requires Linux x86-64, the Hugging Face CLI, Docker Engine with Docker Compose v2, the NVIDIA Container Toolkit, and an NVIDIA Blackwell GPU. It was validated on NVIDIA RTX PRO 6000 Blackwell (96 GB); other GPUs and memory sizes have not been tested.

### Runtime

Both services use the pinned image `prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f`. The engine is started with:

```bash
python3 -m sglang.launch_server --model-path /model --trust-remote-code --tp-size 1 \
  --context-length 131072 --max-prefill-tokens 131072 --chunked-prefill-size 16384 \
  --mem-fraction-static 0.80 --moe-runner-backend flashinfer_cutlass --enable-metrics
```

`--moe-runner-backend flashinfer_cutlass` is required: SGLang's default MoE backend failed to start with this checkpoint.

## Evaluation

All results are self-run, not official rankings, and were produced with the released weights, SGLang image and temperatures.

### Decision Index 0.3

[Decision Index](https://github.com/apolinario/decision-index) 0.3, commit `f8552a13b643d39f61185f5951bb5fed16bedf51`, upstream runners and scorer unchanged, run on 2026-10-13. Hardware: 4 x NVIDIA RTX PRO 6000 Blackwell (96 GB), four independent single-GPU replicas with 8 concurrent requests each (32 in total). 140,620 requests completed with no errors; 140,178 were scored after 442 official exclusions.

| Metric | Xor 26B-A4B 1.1 (BF16) | Xor 26B-A4B NVFP4 1.1 |
|---|---:|---:|
| Decision Index | 57.62 | 57.02 |
| Raw index | 67.88 | 67.48 |
| Breadth score | 56.89 | 56.23 |

| Area (NVFP4 1.1) | Benchmarks | Skill | Raw |
|---|---:|---:|---:|
| Knowledge & Reasoning | 11 | 42.32 | 55.34 |
| Language Understanding | 9 | 59.41 | 70.80 |
| Retrieval & Classification | 6 | 64.75 | 73.84 |
| Tools & Automation | 5 | 69.41 | 72.95 |
| Arts & Human Taste | 6 | 50.70 | 67.51 |

The BF16 reference ran on six NVIDIA B300 replicas at 24 concurrent requests, so this comparison does not separate the effect of quantization from differences in hardware and batching. This is the public self-run score, not the private Full score. `evaluation/` contains the aggregate reports (`index.json`, `scores.json`, `benchmark-summary.json`); per-request inputs and responses are not included.

Latency under that load, per HTTP request: median 181.8 ms, mean 573.8 ms, p95 2,156.4 ms. These are loaded figures, not single-request latency, and they are not comparable with the B300 reference.

### Public JEVBench self-run

On the public [JEVBench](https://github.com/fstandhartinger/jevbench) tiers, NVFP4 1.1 answered 199 to 201 of 231 items correctly across runs; the BF16 build answered 204 in a matched run. Answers vary slightly between NVFP4 runs.

## Operational notes

- Model files occupy 18.8 GB. This is not the GPU memory needed: the KV cache and engine take the rest of the `--mem-fraction-static 0.80` budget.
- The API binds to `127.0.0.1` and has no authentication. Remote deployments must add authentication, TLS, rate limits and request-size limits in front of it.
- The Compose restart policy is `unless-stopped`. Docker does not restart an engine that is running but unhealthy, so external health monitoring is still needed.
- `GET /metrics` on the API port returns the SGLang engine's Prometheus metrics. The proxy and the engine's `--enable-metrics` flag were added after the benchmarks and do not change the inference path; `RELEASE_MANIFEST.json` records the hashes of both the benchmarked and the released wrapper.
- `checksums.sha256` covers every model, serving and evaluation file except this card. `RELEASE_MANIFEST.json` records the source revision, runtime image, wrapper hashes, temperatures and benchmark settings.

## Known limitations

- The data mix used to train Xor 26B-A4B 1.1 is not documented here, and no claim is made that its training data excludes the benchmarks above.
- The quantization build script is not published; `QUANTIZATION.json` records its settings.
- Image and video quality have not been evaluated.
- Hardware other than RTX PRO 6000 Blackwell, and tensor parallelism above 1, have not been tested.

## License

Apache License 2.0. Derived from Gemma 4 26B-A4B-it (Apache 2.0) and Xor 26B-A4B (Apache 2.0). See `THIRD_PARTY_NOTICES.md`.
