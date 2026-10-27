---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: dsx
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- 3.0-bit
- mixed-precision
- gptq
- consumer-gpu
- rtx-3090
---

# DeepSeek-V4.1-Flash-1017 · dsx 3.0-bit

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) quantized to **213.6 GB**, intended for two RTX 3090s and
192 GB of RAM with exact routing. Runs on [**dsx**](https://github.com/lxe/dsx), a Rust + CUDA engine made for this
model. This is a V4.1 GSQ GGUF for dsx, not a llama.cpp / Ollama / LM Studio GGUF.

| | |
|---|---|
| Size | **213.62 GB** (198.95 GiB), single GGUF |
| Engram | **not included**: the official safetensor shards (~203 GB) are read at runtime |
| Format | per-expert mixed precision: PTQ1_0, Q2_0 or Q3_K; dense Q8_0 / F16 / F32 |
| Routed experts | 15,360 experts, **2.9956 bpw average** |
| Routing | exact: no pruning, no router bias; the expert rank affects placement only |
| Runtime | [dsx](https://github.com/lxe/dsx) (OpenAI / Anthropic API, web UI) |
| Benchmarks | KL 0.1284 · top-1 85.39% · PPL 8.965 |

## Quality and speed

### Quality

Measured with `dsx eval` on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 × 1024 held-out tokens, KL and top-1 against the FP8/FP4 teacher's top-256 log-probs. The run used
`--arena-gb 165`: 12,112 experts were placed in the RAM arena, 2,445 on the two GPUs, and the rest streamed from the
GGUF (the log reports `1,682 left in the file`). Placement does not change the arithmetic; every routed expert is
still evaluated exactly.

| | Experts | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| FP8 / FP4 teacher | FP4 | 0 | 100% | 8.56 |
| Lucebox ROCMFP | 2.5 bpw | 0.253 | 78.7% | 9.78 |
| dsx 2.6-bit | 2.6127 bpw | 0.1972 | 82.59% | 9.3626 |
| dsx 2.8-bit | 2.7955 bpw | 0.1575 | 84.07% | 9.0441 |
| **dsx 3.0-bit** | **2.9956 bpw** | **0.1284** | **85.39%** | **8.9652** |

NLL: 2.193352 over 8,184 tokens.

### Speed

Measured on this machine with the new `dsx` engine: 2× RTX 3090, 192 GB RAM, `--ctx 1048576`.

| | |
|---|---:|
| Decode | ~25 tok/s |
| Prompt processing | ~400 tok/s |

## Quantization

- **Base checkpoint:** the official [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  FP8/FP4 checkpoint.
- **Calibration:** 128 × 2048 tokens (262,144 tokens) of antirez's agent, source, programming, reasoning and
  long-context prompts re-rendered for V4.1, plus the pipeline's web/chat/multilingual fallback pools. No local
  agent sessions were available.
- **Method:** layer-streaming calibration forward through DeepSeek's reference `model.py`, then per-expert GPTQ
  with activation Hessians. Each expert's held-out output error was measured at PTQ1_0 / Q2_0 / Q3_K, and the
  allocator spent the bit budget where it reduced error the most.
- **Rotation:** every routed expert is quantized in a 128-block Hadamard-rotated basis. dsx rotates the expert
  inputs at run time to match (`dsx.expert_rot` GGUF key).
- **Scale refit:** after GPTQ picks each expert's codes, its group scales are refit by least squares against the
  full calibration Hessian, GPTQ runs again from the refit scales, and a final refit follows.
- **Allocation granularity:** one tier per expert. Gate, up and down of a given expert use the same tier; this
  build did not use a per-projection allocation.
- **Layer budgets:**
  - Layers `0, 1, 37, 38, 39`: `--budget-bpw 3.4375`, `--max-ternary-frac 0.75`.
  - Layers `2–36`: `--budget-bpw 2.9375` per layer, `--max-ternary-frac 0.75`.
- **Result:** 2,981 PTQ1_0, 1,480 Q2_0 and 10,899 Q3_K experts (19.4% / 9.6% / 71.0%). Overall routed-expert
  average: **2.9956 bpw**.
- **Dense weights:** Q8_0 attention, shared expert and head; F16 embedding, compressor/indexer and Engram wkv;
  F32 norms, router and mHC.
- **Engram:** left unquantized in the official checkpoint and read from the safetensor shards at runtime.

### bpw by part

| Part | Format | Effective bpw |
|---|---|---:|
| Routed experts, layer 0 | Q3_K | 3.4375 |
| Routed experts, layer 1 | 382 Q3_K + 2 Q2_0 | 3.4313 |
| Routed experts, layers 2–36 | PTQ1_0 / Q2_0 / Q3_K | 2.9364–2.9375 |
| Routed experts, layer 37 | 374 Q3_K + 10 Q2_0 | 3.4066 |
| Routed experts, layer 38 | 367 Q3_K + 17 Q2_0 | 3.3849 |
| Routed experts, layer 39 | 361 Q3_K + 23 Q2_0 | 3.3664 |
| All routed experts | mixed | **2.9956** |
| Dense attention / shared expert / head | Q8_0 | 8.5 |
| Embedding / compressor / indexer / Engram wkv | F16 | 16 |
| Norms / router / mHC | F32 | 32 |
| Engram tables | external FP8/E8M0 | ~8.25 |

<details>
<summary>Per-layer expert allocation</summary>

| Layer | PTQ1_0 | Q2_0 | Q3_K | Expert bpw |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 384 | 3.4375 |
| 1 | 0 | 2 | 382 | 3.4313 |
| 2 | 92 | 31 | 261 | 2.9373 |
| 3 | 95 | 27 | 262 | 2.9365 |
| 4 | 90 | 34 | 260 | 2.9368 |
| 5 | 85 | 41 | 258 | 2.9372 |
| 6 | 88 | 37 | 259 | 2.9364 |
| 7 | 92 | 31 | 261 | 2.9373 |
| 8 | 85 | 41 | 258 | 2.9372 |
| 9 | 90 | 34 | 260 | 2.9368 |
| 10 | 88 | 37 | 259 | 2.9364 |
| 11 | 80 | 48 | 256 | 2.9375 |
| 12 | 80 | 48 | 256 | 2.9375 |
| 13 | 83 | 44 | 257 | 2.9367 |
| 14 | 76 | 54 | 254 | 2.9365 |
| 15 | 92 | 31 | 261 | 2.9373 |
| 16 | 83 | 44 | 257 | 2.9367 |
| 17 | 85 | 41 | 258 | 2.9372 |
| 18 | 80 | 48 | 256 | 2.9375 |
| 19 | 92 | 31 | 261 | 2.9373 |
| 20 | 85 | 41 | 258 | 2.9372 |
| 21 | 80 | 48 | 256 | 2.9375 |
| 22 | 85 | 41 | 258 | 2.9372 |
| 23 | 85 | 41 | 258 | 2.9372 |
| 24 | 80 | 48 | 256 | 2.9375 |
| 25 | 88 | 37 | 259 | 2.9364 |
| 26 | 85 | 41 | 258 | 2.9372 |
| 27 | 85 | 41 | 258 | 2.9372 |
| 28 | 88 | 37 | 259 | 2.9364 |
| 29 | 85 | 41 | 258 | 2.9372 |
| 30 | 85 | 41 | 258 | 2.9372 |
| 31 | 80 | 48 | 256 | 2.9375 |
| 32 | 85 | 41 | 258 | 2.9372 |
| 33 | 80 | 48 | 256 | 2.9375 |
| 34 | 83 | 44 | 257 | 2.9367 |
| 35 | 83 | 44 | 257 | 2.9367 |
| 36 | 83 | 44 | 257 | 2.9367 |
| 37 | 0 | 10 | 374 | 3.4066 |
| 38 | 0 | 17 | 367 | 3.3849 |
| 39 | 0 | 23 | 361 | 3.3664 |

</details>

## Run

```bash
# the quant
hf download icecubetr/DeepSeek-V4.1-Flash-1017-dsx-3.0bit --local-dir dsx-3bit

# Engram tables + tokenizer from the official checkpoint
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json

# the engine
git clone https://github.com/lxe/dsx && cd dsx
CUDA_ARCHS=86 cargo build --release -p dsx --features cuda      # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/dsx-server \
  --model ../dsx-3bit/DeepSeek-V4.1-Flash-1017-dsx-3.0bit.gguf \
  --engram ../v41-1017 \
  --rank ../dsx-3bit/expert-rank.json \
  --tokenizer ../dsx-3bit/tokenizer.json \
  --host 0.0.0.0 --port 8090 \
  --api-key "$DSX_API_KEY" \
  --ctx 1048576 --arena-gb 166
```

Set `DSX_API_KEY` to the key clients should send; omit `--api-key` to disable authentication. Do not expose the
server directly to the internet without a key or an authenticating reverse proxy.

At `--arena-gb 166` on a 183 GB machine, the coldest experts stay in the GGUF and are streamed from the SSD.
With more RAM, a larger arena keeps more of them resident; lower it if the server is tight on memory. The Engram
directory only needs `model.safetensors.index.json` plus shards 47 and 48. `--ctx 1048576` is the model's full 1M
context; lower it if you do not need that capacity.

Open `http://localhost:8090/` for the web UI, or point any OpenAI-compatible client at `http://localhost:8090/v1`:

```python
import os
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key=os.environ.get("DSX_API_KEY", "none"))
r = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    reasoning_effort="none",
    messages=[{"role": "user", "content": "Merge overlapping intervals in Python."}],
)
print(r.choices[0].message.content)
```

Thinking is on by default; `"reasoning_effort": "none"` turns it off.

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-3.0bit.gguf` | the model, 213.62 GB, single GGUF |
| `vision.safetensors` | the vision tower, aligner and image embeddings from the official checkpoint (bf16, 0.97 GB), for `--vision` |
| `expert-rank.json` | expert hotness ranking, used to place experts in VRAM and RAM |
| `tokenizer.json` | tokenizer |
| `eval.log` | raw `dsx eval` output (KL 0.128420, top-1 0.853861, PPL 8.9652) |

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format ·
[QuIP#](https://arxiv.org/abs/2402.04396) / [QuaRot](https://arxiv.org/abs/2404.00456) for the Hadamard-rotation idea ·
[GPTQ](https://arxiv.org/abs/2210.17323) ·
antirez for the calibration prompts ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and the ggml authors, MIT) for the web UI.

Quantization pipeline: [github.com/lxe/dsx/tree/main/quant](https://github.com/lxe/dsx/tree/main/quant).
