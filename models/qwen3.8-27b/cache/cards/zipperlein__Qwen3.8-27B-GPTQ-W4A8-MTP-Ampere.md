---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: vllm
tags:
  - qwen
  - gptq
  - int4
  - speculative-decoding
  - vllm
---

# Qwen3.8-27B GPTQ 3090

An inference-focused quantization of Qwen3.8-27B for two 24 GB RTX 3090 GPUs.
It uses GPTQ W4 G128 decoder weights, a GPTQ W8 language-model head, packed INT8
embeddings, a packed INT4 MTP module, FP8 KV cache, and MTP speculative decoding.

This repository includes a model-local compatibility layer under
`runtime_patch/`. Use the pinned vLLM release and `launch.sh`; loading the
checkpoint with an unpatched stock command is not supported.

## Hardware and software

- Two NVIDIA GPUs with 24 GB VRAM each; tested on two RTX 3090s
- A recent NVIDIA driver and CUDA-compatible Linux environment
- Python 3.11
- Approximately 17 GB for the download, plus space for the vLLM compile cache
- vLLM 0.27.1 (required by the model-local patch)

The supplied profile is text-only. Although the base configuration contains a
vision tower, multimodal serving is not enabled by this runtime setup.

## Install

Create an environment and install the pinned serving runtime:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "vllm==0.27.1" "huggingface_hub[cli]"
```

Download the model, replacing `OWNER/MODEL` with this repository's Hugging Face
identifier:

```bash
hf download OWNER/MODEL --local-dir Qwen3.8-27B-GPTQ-3090
cd Qwen3.8-27B-GPTQ-3090
```

## Run

Start the OpenAI-compatible server:

```bash
chmod +x launch.sh
./launch.sh
```

The default endpoint is `http://0.0.0.0:8000`, and the served model name is
`qwen3.8-27b`. The first launch can take several minutes while vLLM compiles
kernels and captures graphs.

Test the server:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain speculative decoding briefly."}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 256
  }'
```

## Launch options

Configure the common settings with environment variables:

```bash
PORT=8002 MAX_MODEL_LEN=131072 GPU_MEMORY_UTILIZATION=0.94 ./launch.sh
```

| Variable | Default | Purpose |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPUs exposed to vLLM |
| `TENSOR_PARALLEL_SIZE` | `2` | Tensor-parallel workers |
| `PORT` | `8000` | API port |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Name used in API requests |
| `MAX_MODEL_LEN` | `250000` | Maximum context length |
| `MAX_NUM_SEQS` | `10` | Maximum concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.955` | Per-GPU vLLM memory fraction |
| `SPECULATIVE_TOKENS` | `3` | MTP draft tokens per step |
| `QWEN38_W4A8_MODE` | `decoder_all` | Selective INT8 activation profile |

Additional vLLM arguments may be appended to the command. For example, to add
API authentication:

```bash
./launch.sh --api-key "replace-with-a-secret"
```

Lower `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` if other processes use either
GPU. On WSL2, CUDA VMM may be unavailable; in that case launch with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False ./launch.sh
```

## Quantization

| Component | Format |
|---|---|
| Decoder linears | GPTQ INT4, symmetric, group size 128; INT8 activations at runtime |
| Language-model head | GPTQ INT8, symmetric, group size 128 |
| Token embeddings | Packed INT8, symmetric, group size 128 |
| MTP draft module | Packed INT4, group size 128 |
| MTP norms and vision tower | BF16/FP16 |
| KV cache | FP8 in the supplied launch profile |

`mtp_draft_lm_head_w4.safetensors` is the default draft head.
`mtp_draft_lm_head_w8.safetensors` is included as an alternate runtime asset.

## Limitations

- The bundled patch checks for exactly vLLM 0.27.1 and intentionally refuses to
  run with another version.
- The tested launch profile requires two GPUs and is tuned for RTX 3090 memory
  limits; other GPU configurations need retuning.
- FP8 KV cache and weight quantization can reduce quality relative to the BF16
  base model.
- The included server profile is text-only.
- Prefix caching for hybrid Mamba/GDN models is experimental in vLLM.
- Review generated output before using it in high-impact or safety-critical
  applications. The base model's limitations still apply.

## License

Apache 2.0, matching the base model. See the base model repository for its full
terms and usage guidance.
