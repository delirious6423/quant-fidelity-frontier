---
base_model: Qwen/Qwen3.8-27B
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

# Qwen3.8-27B (W4A16 Auto-Round GPTQ, group size 64)

This is a 4-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) optimized specifically for coding tasks and tuned for deployment on **Intel Arc B70 (Battlemage)** GPUs using **vLLM (XPU backend)**.

Quantization was executed using [Auto-Round](https://github.com/intel/auto-round) with the `opencode-instruct` dataset to preserve code reasoning and instruction-following accuracy. The model's Multi-Token-Prediction (MTP) head is kept unquantized to support speculative decoding in vLLM.

## Quantization Configuration

| Parameter | Value |
| --- | --- |
| **Method** | Auto-Round |
| **Quant Scheme** | W4A16 (4-bit weights, 16-bit activations) |
| **Format** | `auto_gptq` |
| **Group Size** | 64 |
| **Calibration Dataset** | `opencode-instruct` |
| **Iterations** | 400 |

Reproduce with:

```bash
auto-round \
      --model Qwen/Qwen3.8-27B \
      --scheme W4A16 \
      --group_size 64 \
      --iters 400 \
      --dataset "opencode-instruct" \
      --format auto_gptq \
      --low_gpu_mem_usage \
      --output_dir ~/models/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
```


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
  --model urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
  --served-model-name urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 131072 \
  --kv-cache-dtype auto \
  --max-num-seqs 2 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  --trust-remote-code \
  --enable-prefix-caching --mamba-block-size 256 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

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
    model="urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ",
    messages=[
        {"role": "user", "content": "Write a Python function to perform binary search on a sorted list."}
    ],
    temperature=0.2,
    max_tokens=512
)

print(response.choices[0].message.content)

```
