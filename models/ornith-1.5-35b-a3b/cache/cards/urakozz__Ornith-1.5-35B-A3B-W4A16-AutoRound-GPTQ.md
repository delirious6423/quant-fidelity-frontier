---
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: vllm
quantization: gptq
tags:
  - auto-round
  - w4a16
  - intel-arc
  - intel-xpu
  - battlemage
  - code
  - vllm
pipeline_tag: text-generation
---

# Ornith-1.5-35B-A3B (W4A16 Auto-Round GPTQ)

This is a 4-bit quantized version of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) optimized specifically for coding tasks and tuned for high-throughput deployment on **Intel Arc B70 (Battlemage)** GPUs using **vLLM (XPU backend)**.

Quantization was executed using [Auto-Round](https://github.com/intel/auto-round) with the `opencode-instruct` dataset to preserve code reasoning and instruction-following accuracy.

## Quantization Configuration

| Parameter | Value |
| --- | --- |
| **Method** | Auto-Round |
| **Quant Scheme** | W4A16 (4-bit weights, 16-bit activations) |
| **Format** | `auto_gptq` |
| **Group Size** | 64 |
| **Calibration Dataset** | `opencode-instruct` |
| **Iterations** | 400 |

---

## Performance Benchmarks

Evaluated on **Intel Arc B70** using **vLLM XPU Engine** across varying concurrency levels (`c1` to `c16`).

### Prompt Processing (4096 tokens)

| Concurrency | Throughput (Total t/s) | Throughput (Req t/s) | E2E TTFT (ms) |
| --- | --- | --- | --- |
| **1 (c1)** | 7090.47 ± 28.42 | 7090.47 ± 28.42 | 591.26 ± 7.81 |
| **2 (c2)** | 6919.26 ± 38.42 | 4836.87 ± 1208.10 | 863.52 ± 191.99 |
| **8 (c8)** | 7301.30 ± 19.02 | 2240.24 ± 1556.23 | 2402.82 ± 1120.64 |
| **16 (c16)** | 7242.43 ± 11.24 | 1380.21 ± 1302.69 | 4486.09 ± 2322.99 |

### Text Generation (256 tokens)

| Concurrency | Throughput (Total t/s) | Throughput (Req t/s) | Peak Throughput (Total t/s) |
| --- | --- | --- | --- |
| **1 (c1)** | 99.78 ± 0.08 | 99.78 ± 0.08 | 101.00 ± 0.00 |
| **2 (c2)** | 158.28 ± 0.46 | 84.85 ± 4.98 | 182.00 ± 0.00 |
| **8 (c8)** | 260.05 ± 1.05 | 43.88 ± 7.83 | 466.33 ± 4.03 |
| **16 (c16)** | 302.28 ± 1.03 | 28.62 ± 6.82 | 671.33 ± 0.47 |

---

## Deployment & Usage

### Running via Docker on Intel Arc (vLLM XPU)

To deploy the model server with full Intel Arc hardware acceleration (Level Zero driver), launch the vLLM container with the environment variables and device pass-through configured below:

```bash
docker run --rm -it \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  -v /path/to/models:/models:ro \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu-env \
  --model urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ \
  --served-model-name urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 131072 \
  --kv-cache-dtype auto \
  --max-num-seqs 16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --trust-remote-code \
  --enable-prefix-caching

```

### Querying the Endpoint

Once the server is running, query it using any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-required"
)

response = client.chat.completions.create(
    model="urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ",
    messages=[
        {"role": "user", "content": "Write a Python function to perform binary search on a sorted list."}
    ],
    temperature=0.2,
    max_tokens=512
)

print(response.choices[0].message.content)

```