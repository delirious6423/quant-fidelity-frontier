---
license: apache-2.0
tags:
  - gguf
  - qwen
  - qwen3.8
  - multimodal
  - vision
  - text-only
  - text-generation
  - llama.cpp
  - reasoning
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
model_type: qwen3_5
quantized_by: bartowski
---

# Qwen3.8-27B — GGUF (Q4_K_M multimodal + text-only IQ4_XS) + llama.cpp config

[![GGUF](https://img.shields.io/badge/GGUF-Q4_K_M-8A2BE2)](https://huggingface.co/impacte/Qwen3.8-27B-GGUF)
[![Text-only](https://img.shields.io/badge/Text--only-IQ4_XS-4B0082)](https://huggingface.co/impacte/Qwen3.8-27B-GGUF)
[![Context](https://img.shields.io/badge/Context-256K-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![Built by impacte.tech](https://img.shields.io/badge/Built_by-impacte.tech-8A2BE2?logo=spinrilla)](https://impacte.tech)

## Deploy to RunPod

One-click deploy each quantization version as a GPU Pod on [RunPod](https://runpod.io) via **Ollama** (OpenAI-compatible API) with an **aggressively quantized 4-bit KV cache** (`OLLAMA_KV_CACHE_TYPE=q4_0` + Flash Attention) — fitting the full **256K context** on a 24 GB GPU.

[![Deploy to RunPod - Q4_K_M Multimodal](https://img.shields.io/badge/Deploy_to_RunPod-Q4_K_M_Multimodal-5D29F0?style=for-the-badge&logo=runpod&logoColor=white)](https://console.runpod.io/deploy?template=4rk7o6j8j9)
[![Deploy to RunPod - IQ4_XS Text-only](https://img.shields.io/badge/Deploy_to_RunPod-IQ4_XS_Text--only-5D29F0?style=for-the-badge&logo=runpod&logoColor=white)](https://console.runpod.io/deploy?template=3eyn5o5ld3)

> **Q4_K_M Multimodal** (17.77 GB, Text + Image + Video) · **IQ4_XS Text-only** (~14.7 GB, Text only). Requires a 24 GB+ VRAM GPU. The 4-bit KV cache drops the 256K KV footprint from ~17 GB (f16) to ~4.3 GB (q4_0).

---

Full-quality **Qwen3.8-27B** in GGUF format with ready-to-run **llama.cpp** configurations that use `--no-kv-offload` to keep the KV cache in system RAM — unlocking the full **256K context** even on modest GPU setups.

Two GGUF variants are published in this repo:

| Variant | File | Size | Modalities |
|---------|------|------|------------|
| Multimodal | `Qwen3.8-27B-Q4_K_M.gguf` + `mmproj-Qwen3.8-27B-f16.gguf` | 17.77 GB | Text + Image + Video |
| **Text-only** | `Qwen3.8-27B-text-IQ4_XS.gguf` | ~14.7 GB | Text only |

---

## DESCRIPTION

A **dense 27.8B-parameter** multimodal model from the Qwen team, packaged as GGUFs with complete llama.cpp launchers. Unlike MoE models, every parameter is active on every token, giving maximum quality per token. Its hybrid attention architecture (48 linear + 16 full attention layers) delivers efficient long-context inference without the quadratic cost of pure attention.

**Key features:**
- **Dense 27.8B** — all parameters active, no MoE routing
- **Hybrid attention** — 48 linear + 16 full attention layers (3:1 ratio)
- **256K native context** (262,144 tokens)
- **Two variants** — multimodal Q4_K_M or text-only IQ4_XS
- **Reasoning model** — ` thinking` blocks before ` response`
- **Tool calling** — `<tool_call><function=...>` XML format
- **Q4_K_M quantization** — imatrix-calibrated, 17.77 GB (multimodal)
- **IQ4_XS quantization** — 4.25 bpw, ~14.7 GB (text-only)

### Architecture

| Property | Value |
|----------|-------|
| Architecture | Qwen3.5 hybrid (linear + full attention, dense) |
| Total Parameters | 27.78B (all active) |
| Layers | 64 (48 linear + 16 full attention) |
| Hidden Size | 5,120 |
| Attention Heads | 24 (4 KV heads) |
| Native Context | 262,144 tokens (256K) |
| Modalities | Text + Image + Video |
| Quantization | Q4_K_M imatrix-calibrated |
| Model Size | ~17.8 GB (weights) |
| License | Apache 2.0 |
| Upstream | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

---

## FILES

| File | Purpose |
|------|---------|
| `Qwen3.8-27B-Q4_K_M.gguf` | Multimodal quantized weights (17.77 GB) |
| `mmproj-Qwen3.8-27B-f16.gguf` | Multimodal projector (vision/video) |
| **`Qwen3.8-27B-text-IQ4_XS.gguf`** | **Text-only quantized weights (~14.7 GB)** |
| `run-llamacpp.sh` | Multimodal llama.cpp server launcher with `--no-kv-offload` |
| `Modelfile` | Ollama Modelfile (multimodal Q4_K_M) |

---

## TEXT-ONLY VERSION (IQ4_XS)

The text-only variant strips the vision/audio encoder and projector entirely, leaving only the language-model tensors. It is ideal for pure text workloads (chat, coding, RAG, tool calling) where the extra ~10% of multimodal weights is unnecessary.

| Property | Value |
|----------|-------|
| File | `Qwen3.8-27B-text-IQ4_XS.gguf` |
| Quantization | IQ4_XS (4.25 bpw, non-linear) |
| Size | ~14.7 GB (4.5 BPW) |
| Tensors | 866 (text/SSM/nextn only; no vision tensors) |
| Modalities | Text only |
| MTP/nextn | Included (1 nextn layer) |
| Source | Converted from [unsloth/Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B) HF checkpoint (vision skipped) → `llama-quantize IQ4_XS` |

### Run the text-only version (llama.cpp)

```bash
llama-server \
  --model Qwen3.8-27B-text-IQ4_XS.gguf \
  -ngl 99 \
  --no-kv-offload \
  -c 262144 \
  --parallel 1
```

No `--mmproj` is passed — the model is purely text-based.

---

## REQUIREMENTS

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 24 GB VRAM | 32 GB+ VRAM |
| System RAM | 64 GB | 128 GB |
| Disk Space | 20 GB free | 50 GB+ free |
| llama.cpp | recent build (CUDA) | latest master |
| NVIDIA Driver | 525+ | 550+ |

> 💡 **Why `--no-kv-offload`?** The 17.8 GB weights fit on GPU, but the KV cache for 256K context is huge. By keeping the KV cache in system RAM, all 64 layers stay on GPU while the context window can reach the full 256K — ideal for 24 GB VRAM + 128 GB RAM setups.

---

## QUICK START (llama.cpp — recommended)

### 1. Download this repo

```bash
git lfs clone https://huggingface.co/impacte/Qwen3.8-27B-GGUF
cd Qwen3.8-27B-GGUF
```

### 2. Start the server

```bash
./run-llamacpp.sh
# -> OpenAI-compatible API at http://localhost:8082/v1
```

The script sets `CUDA_VISIBLE_DEVICES=0,1` (device ordinals, not PCI IDs) and uses `--no-kv-offload` so the KV cache goes to system RAM.

### 3. Test it

```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## USAGE

### Manual llama.cpp command

```bash
llama-server \
  --model Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-f16.gguf \
  -ngl 99 \
  --no-kv-offload \
  -c 262144 \
  --parallel 1
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Explain quantum computing simply"}],
)
print(response.choices[0].message.content)
```

### Multimodal (image)

```bash
curl http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
      ]
    }]
  }'
```

---

## RUN WITH OLLAMA

```bash
ollama create qwen3.8-27b -f Modelfile
ollama run qwen3.8-27b
```

> **Note:** On 24 GB VRAM, Ollama cannot offload the KV cache to RAM, so the default context is limited to 8K. Use the llama.cpp script above for the full 256K context. A 32 GB+ VRAM variant is also published on Ollama as `oamazonasgabriel/qwen3.8-27b:q4-km-32gbGPU`.

---

## MEMORY & PERFORMANCE

| Component | Multimodal Q4_K_M | Text-only IQ4_XS |
|-----------|-------------------|------------------|
| Model weights | ~17.8 GB | ~14.7 GB |
| KV cache (in RAM, 256K ctx) | ~20-28 GB (system RAM) | ~20-28 GB (system RAM) |
| GPU VRAM used | ~18-19 GB (all layers on GPU) | ~15-16 GB (all layers on GPU) |

**Performance (24 GB VRAM + 128 GB RAM):**
- **Prompt processing:** ~60-120 tok/s (hybrid linear attention)
- **Text generation:** ~20-40 tok/s
- **Model load time:** ~60-90 seconds

---

## TROUBLESHOOTING

| Symptom | Fix |
|---------|-----|
| `invalid device ordinal` / no GPU | Set `CUDA_VISIBLE_DEVICES=0,1` (device ordinals, not PCI IDs) |
| OOM on GPU | Ensure `--no-kv-offload` is set so KV cache goes to RAM |
| Slow generation | Check `nvidia-smi` — all layers should be on GPU |
| Vision not working | Ensure `--mmproj mmproj-Qwen3.8-27B-f16.gguf` is passed |

---

## CREDITS

| Role | Entity |
|------|--------|
| **Base Model** | [Qwen Team, Alibaba Group](https://huggingface.co/Qwen) |
| **Original Model** | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Multimodal GGUF (Q4_K_M)** | [bartowski](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) — imatrix-calibrated |
| **Text-only GGUF (IQ4_XS) source** | [unsloth/Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B) HF checkpoint; converted with [llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py` (vision skipped) + `llama-quantize IQ4_XS` |
| **llama.cpp Packaging** | [impacte.tech](https://impacte.tech) |
| **Ollama** | `oamazonasgabriel/qwen3.8-27b` |
| **License** | Apache 2.0 |

---

## LINKS

| Resource | URL |
|----------|-----|
| This repo | https://huggingface.co/impacte/Qwen3.8-27B-GGUF |
| Upstream model | https://huggingface.co/Qwen/Qwen3.8-27B |
| bartowski quantized version (Q4_K_M) | https://huggingface.co/bartowski/Qwen3.8-27B-GGUF |
| unsloth unquantized checkpoint | https://huggingface.co/unsloth/Qwen3.8-27B |
| Ollama model | https://ollama.com/oamazonasgabriel/qwen3.8-27b |
| llama.cpp | https://github.com/ggml-org/llama.cpp |
| Built by | [impacte.tech](https://impacte.tech) |
