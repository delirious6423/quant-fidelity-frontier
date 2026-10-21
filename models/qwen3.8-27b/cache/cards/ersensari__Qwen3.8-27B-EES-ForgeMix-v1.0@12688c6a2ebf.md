---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - qwen3.8
  - modelopt
  - fp8
  - nvfp4
  - speculative-decoding
  - dflash2
  - sglang
  - vision-language
---

# EES ForgeMix v1.0

EES ForgeMix v1.0 is a third-party, BF16-derived mixed FP8/NVFP4
quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
It is optimized for quality-first local serving on two NVIDIA Blackwell consumer
GPUs with the companion
[`EES ForgeMix DFlash2 uint2 v1.0`](https://huggingface.co/ersensari/Qwen3.8-27B-EES-ForgeMix-DFlash2-uint2-v1.0)
draft.

This is a quantized derivative. It was not trained, developed, or endorsed by
the Qwen team, Alibaba Cloud, In.co, Z-Lab, NVIDIA, ModelOpt, or SGLang. EES did
not create or own the upstream Qwen model. Refer to the upstream Qwen model card
for original training, capabilities, intended uses, limitations, and notices.

## What Changed

The upstream BF16 checkpoint was independently calibrated and exported with
`nvidia-modelopt==0.46.0`. The public checkpoint contains:

- 48 FP8 and 352 NVFP4 searched text-projection modules;
- NVFP4 embedding and LM head bundles carried from a separately validated EES
  ModelOpt export to preserve the strict byte/runtime contract;
- 354 total NVFP4 modules after those two protected modules are counted;
- source-byte-identical vision and MTP tensors;
- five independent safetensors shards with 18,497,335,916 tensor bytes;
- ModelOpt quantization metadata embedded in the Hugging Face configuration.

The mixed recipe was selected from 400 eligible text projections using
BF16-derived gradient/NLL-linked and KL sensitivity under a fixed byte budget.
EvalPlus tasks or scores were not used to select the recipe.

The shard filenames retain the internal `model-v13-*` build identifier for
checksum continuity. That identifier is not the release version; this repository
is the stable `EES ForgeMix v1.0` release.

## Model Details

| Property | Value |
|---|---|
| Upstream model | `Qwen/Qwen3.8-27B` |
| Model type | Dense causal vision-language model |
| Quantization | Mixed FP8/NVFP4, ModelOpt 0.46.0 |
| Searched text projections | 48 FP8 + 352 NVFP4 |
| Total protected NVFP4 modules | 2 (embedding and LM head) |
| Vision/MTP tensors | BF16 source-byte-identical |
| Checkpoint tensor bytes | 18,497,335,916 (18.497 GB / 17.227 GiB) |
| Native upstream context | 262,144 tokens |
| Validated ForgeMix serving context | 64,000 tokens |
| Inputs | Text and images; upstream also documents video support |
| Outputs | Generated text, reasoning content, and structured tool calls |
| License | Apache-2.0; see `LICENSE` and `NOTICE` |

The model preserves Qwen3.8's chat template, thinking controls, tool-call parser
compatibility, and vision processor files. Quantization can change behavior, so
upstream BF16 results must not be presented as ForgeMix results.

## Calibration And Selection

The search used eight calibration samples of at most 512 tokens from a
1,024-prompt, deduplicated, leakage-filtered local coding corpus assembled from
CodeFeedback, Magicoder, Evol-Code, and CodeAlpaca sources. HumanEval, EvalPlus,
and MBPP content was excluded. Corpus SHA-256:

```text
d295bba6078757cf2a4af5d5505b5dc0e050c25294d0782bd0e7721c885622e4
```

No calibration rows or benchmark solutions are redistributed here. Dataset
names are provenance disclosures, not a grant of dataset rights. Consult each
source dataset card and license before reusing those datasets.

## Validated Quality

EvalPlus 0.3.1, greedy generation, temperature 0, explicit seed 42,
`reasoning_effort=none`, TP2, 64K context, concurrency 1, target FP8 E4M3 KV,
draft BF16 KV, FlashInfer, draft tokens 6, page size 1, and Mamba full-memory
ratio 0.50:

| Target | HumanEval | HumanEval+ | MBPP | MBPP+ | Plus mean |
|---|---:|---:|---:|---:|---:|
| Radix mixed baseline | 0.909 | 0.884 | **0.892** | 0.751 | 0.8175 |
| EES smart-mix alpha.1 | **0.921** | 0.866 | 0.878 | 0.754 | 0.8100 |
| **EES ForgeMix v1.0** | **0.921** | **0.896** | 0.886 | **0.757** | **0.8265** |

Exact ForgeMix counts are 151/164 HumanEval, 147/164 HumanEval+, 335/378
MBPP, and 286/378 MBPP+, with zero empty outputs and no server restart or OOM.
A Plus pass requires both base and Plus tests to pass.

These results cover code-generation tasks only. They do not establish general,
multilingual, safety, vision, long-context, or production reliability quality.

## Validated Throughput

A fair seed-42 SGLang workload used 20 requests, input lengths up to 4,096,
output lengths up to 1,024, concurrency 1, and the same companion uint2 draft:

| Target | Output tok/s | Mean accepted length | Success |
|---|---:|---:|---:|
| Radix mixed baseline | 85.26 | 3.14 | 20/20 |
| EES smart-mix alpha.1 | **96.61** | **3.23** | 20/20 |
| **EES ForgeMix v1.0** | 93.37 | 3.11 | 20/20 |

Test host: 2x NVIDIA RTX 5060 Ti 16 GB (SM120), PCIe PHB without P2P/NVLink.
ForgeMix is 9.51% faster than the Radix row on this workload but 3.35% slower
than alpha.1. Results are end-to-end measurements, not isolated kernel claims,
and will vary with hardware, prompts, runtime, context, and sampling.

## Serving With The Companion Draft

This is the fully validated quality-first profile. Build the included compatibility
runtime, set local paths to both downloaded repositories, and launch:

```bash
export TARGET_MODEL=/models/Qwen3.8-27B-EES-ForgeMix-v1.0
export DRAFT_MODEL=/models/Qwen3.8-27B-EES-ForgeMix-DFlash2-uint2-v1.0

docker build \
  -t forgeloop-ai-runtime-sglang-forgemix-v10:local \
  "$TARGET_MODEL/runtime"

docker run --rm --gpus all \
  --name forgeloop-ai-sglang-forgemix-v10 \
  --label forgeloop.managed=true \
  --label forgeloop.project=forgeloop-ai \
  --label forgeloop.resource=model-server \
  --ipc host --shm-size 32g \
  -p 127.0.0.1:8000:8000 \
  -e NCCL_P2P_DISABLE=1 \
  -e 'SGLANG_HUMMING_ONLINE_QUANT_CONFIG={"weight_dtype":"uint2","group_size":64,"has_zero_point":true,"force_requant":true}' \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/draft:ro" \
  forgeloop-ai-runtime-sglang-forgemix-v10:local \
  python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /models/target \
  --served-model-name qwen3.8 \
  --context-length 64000 \
  --max-total-tokens 64000 \
  --tp-size 2 \
  --disable-custom-all-reduce \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --mamba-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 1 \
  --chunked-prefill-size 2048 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/draft \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --speculative-draft-model-quantization humming \
  --speculative-draft-load-format presharded \
  --speculative-num-draft-tokens 6 \
  --mem-fraction-static 0.90 \
  --mamba-full-memory-ratio 0.50 \
  --disable-prefill-cuda-graph \
  --max-running-requests 1 \
  --schedule-policy fcfs \
  --random-seed 42 \
  --allow-auto-truncate \
  --sleep-on-idle \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

The bundled runtime starts from the named Qwen3.8/DFlash2 SGLang base image and
adds bounded ModelOpt NVFP4 embedding/MTP compatibility patches. The base tag is
not digest-pinned in this public Dockerfile; review and pin it for reproducible
production builds. A prebuilt image is not distributed here.

The draft checkpoint is optional. A compatible recent engine may serve the
target without speculative decoding, but that mode was not the benchmark profile
above. Other loaders require explicit validation of this ModelOpt mixed format.

## Runtime Findings

Serving knobs are part of the evaluated system, not quality-neutral settings.
In controlled tests:

- draft tokens 8 increased throughput but reduced HumanEval+ to 0.878;
- Mamba full-memory ratio 0.30 reduced HumanEval+ to 0.866;
- Mamba ratio 0.60 preserved aggregate HumanEval+ and slightly improved speed,
  but MBPP+ was not run, so it is not the release recommendation;
- page size 1 and Mamba ratio 0.50 remain the fully validated profile.

Changing draft count, page size, Mamba allocation, kernels, quantization loader,
or generation parameters can change outputs. Re-run application quality tests
after any serving change.

## Limitations And Responsible Use

- Quantization is lossy and may amplify, suppress, or introduce model errors.
- The upstream model and this derivative can hallucinate facts, code, citations,
  tool arguments, or unsafe instructions.
- Generated code must be reviewed, sandboxed, and tested before execution.
- Do not rely on model output alone for medical, legal, financial, security,
  safety-critical, or other high-impact decisions.
- Apply access controls, content safeguards, rate limits, logging policies, and
  human review appropriate to the deployment.
- Prompts and outputs may contain sensitive or personal data. Operators are
  responsible for consent, retention, privacy, and regulatory compliance.
- Tool use and autonomous agents require least-privilege credentials and strict
  execution isolation. The checkpoint provides no security boundary.
- Performance and memory headroom were measured on one hardware topology. Test
  OOM behavior, concurrency, long context, and failure recovery on your system.
- Native upstream context is 262K, but this package was validated at 64K. No
  ForgeMix quality or stability claim is made for longer contexts or YaRN.

Use is subject to Apache-2.0 and applicable law. The model and modifications are
provided on an AS-IS basis, without warranties or conditions of any kind. See
`LICENSE` for the controlling terms and limitation of liability.

## Attribution

- Upstream model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Qwen3.8 citation and original model details: upstream model card
- Quantization toolkit: [`NVIDIA/TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- Serving engine: [`sgl-project/sglang`](https://github.com/sgl-project/sglang)
- Companion draft source: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- DFlash project: [`z-lab/dflash`](https://github.com/z-lab/dflash)

See `NOTICE` for modification and redistribution notices.
