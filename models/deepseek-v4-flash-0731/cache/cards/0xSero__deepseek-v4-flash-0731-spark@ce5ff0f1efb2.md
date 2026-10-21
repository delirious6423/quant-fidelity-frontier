---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
pipeline_tag: text-generation
tags:
- deepseek-v4
- exl3
- exllamav3
- trellis
- reap
- pruning
- mixture-of-experts
- dgx-spark
- sparkinfer
---

# DeepSeek V4 Flash 0731 for one DGX Spark

> [!IMPORTANT]
> **Generation is validated on one physical NVIDIA DGX Spark.** Use the pinned
> Docker recipe below. Stock vLLM, stock ExLlamaV3, and a bare
> `--tensor-parallel-size 1` command cannot load this TP4 rank-sliced artifact
> correctly.

This is a REAP-pruned, 3.0 bpw EXL3/Trellis build of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
packaged for a **single 128 GB DGX Spark** through
[`local-inference-lab/sparkinfer`](https://github.com/local-inference-lab/sparkinfer).

The supported runtime is public and reproducible:

- **Runtime repository:** [`0xSero/deepseek-v4-flash-0731-spark-sparkinfer`](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)
- **Pinned public image:** `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- **Pinned model revision:** `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- **API:** OpenAI-compatible server on `http://127.0.0.1:8000`
- **Configured model limit:** 262,144 tokens

## Run it

### 1. Prepare the Spark

You need:

- one NVIDIA DGX Spark with its GB10/SM121 GPU and 128 GB unified memory;
- Linux ARM64 with current NVIDIA drivers;
- Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit;
- at least **250 GB free disk** for the image, downloaded TP4 checkpoint,
  losslessly coalesced TP1 checkpoint, compact draft, and caches. 300 GB is a
  safer allowance.

The model and GHCR image are public. No Hugging Face or GHCR login is required.

Confirm Docker can see the GPU:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

### 2. Install and start with one command

```bash
git clone https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer.git && \
cd deepseek-v4-flash-0731-spark-sparkinfer && \
docker compose up -d
```

Do not replace the image digest or manually reconstruct the vLLM command. The
Compose file pins the validated ARM64 runtime, model revision, SparkInfer
patches, TP1 coalescer, K64 speculative draft, FP8 sparse-MLA compatibility
cache, and CUDA-graph configuration.

### 3. Follow the first start

```bash
docker compose logs -f
```

The first start is intentionally long. It will:

1. pull the pinned ARM64 runtime image;
2. download this exact model revision (about 107 GB of published files);
3. losslessly coalesce its TP4 rank slices into a TP1 serving checkpoint;
4. verify the tensor manifest and SHA-256 checksums;
5. build and validate a separate 3.0 GB K64 DSpark speculative draft;
6. run the SparkInfer CUDA-graph self-test; and
7. load the model, capture FULL/PIECEWISE CUDA graphs, allocate the KV cache,
   and start port 8000.

Downloaded weights and compiled caches remain under `./data` and `./cache`.
Restarting the container reuses valid artifacts instead of downloading or
rebuilding them.

### 4. Wait for health

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 10; done
curl -fsS http://127.0.0.1:8000/v1/models
```

Healthy startup logs should show all of the following:

- `max_model_len=262144`;
- `FULL_AND_PIECEWISE` CUDA graphs, not eager execution;
- fixed K5 DSpark verification with a six-row C1 graph;
- at least 262,144 post-capture KV tokens; and
- served model name `deepseek-v4-flash-0731-spark`.

### 5. Generate

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-0731-spark",
    "messages": [
      {"role": "user", "content": "Write a correct Python function that returns the first n Fibonacci numbers."}
    ],
    "temperature": 0,
    "max_completion_tokens": 512
  }'
```

Thinking is disabled in the measured fast profile. To request it for an
individual call, pass the model's supported chat-template option explicitly;
expect different latency and memory behavior.

### Stop or restart

```bash
docker compose stop
docker compose start
```

To remove only the running container while preserving downloaded model data
and caches:

```bash
docker compose down
```

## What was actually validated

The published Docker image was built from the public runtime repository and
tested on one physical DGX Spark. The following acceptance surfaces passed:

- complete model manifest and checksum verification;
- all 216 target experts and target EXL3/Trellis tensors preserved;
- lossless TP4-to-TP1 coalescing;
- real semantic and code generation;
- strict JSON-schema structured output;
- exact beginning, middle, and end fact recovery at 500, 2,000, 5,000,
  10,000, and 20,000 characters;
- `max_model_len=262144` with 264,867 post-capture KV tokens in the clean-image
  acceptance run;
- FULL/PIECEWISE CUDA graphs with eager execution disabled; and
- healthy serving with zero container restarts.

Measured C1 code decode across five 512-token trials was:

| Metric | Result |
|---|---:|
| Minimum | 34.30 tok/s |
| Median | 38.12 tok/s |
| Mean | 39.49 tok/s |

A same-runtime-path cold 252,047-token prefill run measured **1,055.45
prompt tok/s** with zero cached prompt tokens. That prefill result predates the
final published-image rerun, so it remains candidate evidence. The requested
steady 35 tok/s minimum is also still an open optimization gate because one of
the five clean-image decode trials measured 34.30 tok/s.

Full evidence and reproduction scripts are in the runtime repository's
[`VALIDATION.md`](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/blob/main/VALIDATION.md)
and [`results/`](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/tree/main/results).

## What the formats and runtime do

- **REAP** selected and compacted the routed experts using router-weighted
  activation observations. The target retains 216 of 256 experts per MoE
  scope. Tool-calling and agentic structured-output specialists were made
  mandatory before the remaining slots were filled by global saliency.
- **EXL3** is the low-bit weight format. It stores the retained routed experts
  near 3.0 bits per weight while carried DeepSeek FP8 tensors remain FP8.
- **Trellis** allocates and reconstructs low-bit weights non-uniformly, keeping
  more precision where it matters instead of forcing every group to the same
  width. `TR3` is the K3 Trellis tier used here.
- **SparkInfer** provides the GB10/SM121 sparse-MLA and Trellis GPU paths used
  by the patched vLLM runtime.
- **DSpark/MTP** supplies a separate speculative draft. The runtime derives a
  compact K64 draft from the preserved REAP rankings; it does not alter the
  K216 target checkpoint.

In short: REAP chooses the experts, EXL3/Trellis compresses and executes their
weights, SparkInfer supplies the Spark-native kernels, and DSpark proposes
tokens that the untouched target verifies.

## Checkpoint provenance

- Source EXL3: [`0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw`](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw)
- Original EXL3 weight files: 124,867,114,600 bytes
- This checkpoint: **106,816,685,560 bytes** (99.48 GiB)
- Removed: **18,050,429,040 bytes** (18.05 GB)
- Routed experts: **216 of 256 retained per MoE scope**
- Router top-k: 6, unchanged
- Observation dataset: [`0xSero/deepseek-v4-flash-reap-observations-v2`](https://huggingface.co/datasets/0xSero/deepseek-v4-flash-reap-observations-v2)
- Observation snapshot: `partial-v2-21289`, 21,289 rows, all experts observed
- Structured observations: 2,253 agentic-tool rows and 4,096 tool-calling rows
- Keep-plan SHA-256: `cc2bdaf189ebdaa98178e35abac4b1b336808c4f86b11e9b8eccde6cb3e44001`

The ranking was transferred from the closely aligned prior DeepSeek V4 Flash
revision. All 43 router matrices passed identity alignment; the weakest
same-index cosine was 0.9460793734 and all three hash-routing tables were
bit-identical.

## Important implementation note

The validated single-Spark route uses a **584-byte padded FP8 sparse-MLA
record** under the `nvfp4_ds_mla` control path. It is not a true 432-byte NVFP4
KV record. The experimental 432-byte implementation passed its isolated kernel
oracle but produced corrupted full-model text, so the Docker recipe disables
it. The server-level generation and structured-output tests are the
authoritative acceptance gates.

## Credits

Thanks to DeepSeek for the base model, Cerebras Research for REAP,
[TurboDerp](https://github.com/turboderp-org) and contributors for
ExLlamaV3/EXL3/Trellis, [brandonmusic](https://huggingface.co/brandonmusic)
for the GLM-5.2 EXL3 TR3 3.0 bpw reference, Local Inference Lab for SparkInfer
and the DGX Spark vLLM work, [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
for the DSpark reference, NVIDIA for the CUDA stack and DGX Spark, and the
vLLM community.

Model use remains subject to the upstream model license and terms.
