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
- 2.6-bit
- mixed-precision
- gptq
- consumer-gpu
- rtx-3090
---

# DeepSeek-V4.1-Flash-1017 · dsx 2.6-bit

DeepSeek-V4.1-Flash-1017 (552B MoE + 196B Engram) quantized to **187.6 GB**, intended for two RTX 3090s and
192 GB of RAM with exact routing. Runs on [**dsx**](https://github.com/lxe/dsx), a Rust + CUDA engine made for this
model. This is a V4.1 GSQ GGUF for dsx, not a llama.cpp / Ollama / LM Studio GGUF.

| | |
|---|---|
| Size | **187.6 GB** (174.7 GiB), single GGUF |
| Engram | **not included**: the official safetensor shards (~203 GB) are read at runtime |
| Format | per-expert mixed precision: PTQ1_0, Q2_0 or Q3_K; dense Q8_0 / F16 / F32 |
| Routed experts | 15,360 experts, **2.6127 bpw average** |
| Routing | exact: no pruning, no router bias; the expert rank affects placement only |
| Runtime | [dsx](https://github.com/lxe/dsx) (OpenAI / Anthropic API, web UI) |
| Benchmarks | KL 0.1972 · top-1 82.59% · PPL 9.36; ~26 tok/s decode, ~170 tok/s prefill |

## Quality and speed

### Quality

Measured with `dsx eval` on [Lucebox's 1017 eval set](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF):
8 × 1024 held-out tokens, KL and top-1 against the FP8/FP4 teacher's top-256 log-probs. All experts were resident
(`0 left in the file`).

| | Experts | KL ↓ | Top-1 ↑ | Perplexity ↓ |
|---|---:|---:|---:|---:|
| FP8 / FP4 teacher | FP4 | 0 | 100% | 8.56 |
| Lucebox ROCMFP | 2.5 bpw | 0.253 | 78.7% | 9.78 |
| dsx 2-bit | 2.03 bpw | 0.373 | 74.4% | 10.68 |
| **dsx 2.6-bit** | **2.6127 bpw** | **0.1972** | **82.59%** | **9.3626** |

NLL: 2.236722 over 8,184 tokens.

### Speed

Measured on this machine with the new `dsx` engine: 2× RTX 3090, 192 GB RAM, `--ctx 262144`, all experts resident.

| | |
|---|---:|
| Decode | ~26 tok/s |
| Prompt processing | ~170 tok/s |

Speed varies with prompt length, routing and context length. A 1M-token context uses more KV cache and may leave
fewer experts resident, so do not assume the same numbers at `--ctx 1048576`.

## Quantization

- **Base checkpoint:** the official [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  FP8/FP4 checkpoint.
- **Calibration:** 128 × 2048 tokens (262,144 tokens) of antirez's agent, source, programming, reasoning and
  long-context prompts re-rendered for V4.1, plus the pipeline's web/chat/multilingual fallback pools. No local
  agent sessions were available.
- **Method:** layer-streaming calibration forward through DeepSeek's reference `model.py`, then per-expert GPTQ
  with activation Hessians. Each expert's held-out output error was measured at PTQ1_0 / Q2_0 / Q3_K, and the
  allocator spent the bit budget where it reduced error the most.
- **Allocation granularity:** one tier per expert. Gate, up and down of a given expert use the same tier; this
  build did not use a per-projection allocation.
- **Layer budgets:**
  - Layers `0, 1, 37, 38, 39`: `--budget-bpw 3.4375`, `--max-ternary-frac 0.75`.
  - Layers `2–36`: `--budget-bpw 2.5`, `--max-ternary-frac 0.75`.
- **Result:** 5,717 PTQ1_0, 2,545 Q2_0 and 7,098 Q3_K experts (37.2% / 16.6% / 46.2%). Overall routed-expert
  average: **2.6127 bpw**.
- **Dense weights:** Q8_0 attention, shared expert and head; F16 embedding, compressor/indexer and Engram wkv;
  F32 norms, router and mHC.
- **Engram:** left unquantized in the official checkpoint and read from the safetensor shards at runtime.

### bpw by part

| Part | Format | Effective bpw |
|---|---|---:|
| Routed experts, layer 0 | Q3_K | 3.4375 |
| Routed experts, layer 1 | 382 Q3_K + 2 Q2_0 | 3.4313 |
| Routed experts, layers 2–36 | PTQ1_0 / Q2_0 / Q3_K | 2.4989–2.5000 |
| Routed experts, layer 37 | 374 Q3_K + 10 Q2_0 | 3.4066 |
| Routed experts, layer 38 | 367 Q3_K + 17 Q2_0 | 3.3849 |
| Routed experts, layer 39 | 361 Q3_K + 23 Q2_0 | 3.3664 |
| All routed experts | mixed | **2.6127** |
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
| 2 | 184 | 42 | 158 | 2.4990 |
| 3 | 188 | 36 | 160 | 2.5000 |
| 4 | 177 | 52 | 155 | 2.4989 |
| 5 | 177 | 52 | 155 | 2.4989 |
| 6 | 181 | 46 | 157 | 2.4998 |
| 7 | 181 | 46 | 157 | 2.4998 |
| 8 | 167 | 66 | 151 | 2.4995 |
| 9 | 160 | 76 | 148 | 2.4993 |
| 10 | 160 | 76 | 148 | 2.4993 |
| 11 | 162 | 73 | 149 | 2.4998 |
| 12 | 162 | 73 | 149 | 2.4998 |
| 13 | 155 | 83 | 146 | 2.4997 |
| 14 | 148 | 93 | 143 | 2.4995 |
| 15 | 162 | 73 | 149 | 2.4998 |
| 16 | 150 | 90 | 144 | 2.5000 |
| 17 | 155 | 83 | 146 | 2.4997 |
| 18 | 158 | 79 | 147 | 2.4989 |
| 19 | 167 | 66 | 151 | 2.4995 |
| 20 | 153 | 86 | 145 | 2.4992 |
| 21 | 158 | 79 | 147 | 2.4989 |
| 22 | 169 | 63 | 152 | 2.5000 |
| 23 | 158 | 79 | 147 | 2.4989 |
| 24 | 167 | 66 | 151 | 2.4995 |
| 25 | 169 | 63 | 152 | 2.5000 |
| 26 | 155 | 83 | 146 | 2.4997 |
| 27 | 158 | 79 | 147 | 2.4989 |
| 28 | 158 | 79 | 147 | 2.4989 |
| 29 | 158 | 79 | 147 | 2.4989 |
| 30 | 158 | 79 | 147 | 2.4989 |
| 31 | 167 | 66 | 151 | 2.4995 |
| 32 | 155 | 83 | 146 | 2.4997 |
| 33 | 160 | 76 | 148 | 2.4993 |
| 34 | 162 | 73 | 149 | 2.4998 |
| 35 | 158 | 79 | 147 | 2.4989 |
| 36 | 160 | 76 | 148 | 2.4993 |
| 37 | 0 | 10 | 374 | 3.4066 |
| 38 | 0 | 17 | 367 | 3.3849 |
| 39 | 0 | 23 | 361 | 3.3664 |

</details>

## Run

```bash
# the quant
hf download icecubetr/DeepSeek-V4.1-Flash-1017-dsx-2.6bit --local-dir dsx-2.6bit

# Engram tables + tokenizer from the official checkpoint
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir v41-1017 \
  --include model.safetensors.index.json model-00047-of-00048.safetensors model-00048-of-00048.safetensors tokenizer.json

# the engine
git clone https://github.com/lxe/dsx && cd dsx/engine
CUDA_ARCHS=86 cargo build --release --features cuda      # 86 = RTX 30xx, 120 = RTX 50xx

./target/release/dsx-server --model ../../dsx-2.6bit/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf \
  --engram ../../v41-1017 --tokenizer ../../v41-1017/tokenizer.json --host 0.0.0.0 --port 8090 \
  --ctx 262144 --chunk 4096 --gpu0-gb=-1.0 --gpu1-gb=-1.0 --arena-gb 160 \
  --rank ../../dsx-2.6bit/expert-rank.json --vision ../dsx-2.6bit/vision.safetensors
```

The command downloads the model into `dsx-2.6bit/` and uses the uploaded filename
`DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf`. The Engram directory only needs `model.safetensors.index.json` plus
shards 47 and 48. `--ctx 1048576` is the model's full 1M context; lower it if you do not need that capacity.

For remote access, use an SSH tunnel or put the server behind an authenticating reverse proxy. Do not expose it
directly to the internet.

Open `http://localhost:8090/` for the web UI, or point any OpenAI-compatible client at `http://localhost:8090/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="none")
r = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    reasoning_effort="none",
    messages=[{"role": "user", "content": "Merge overlapping intervals in Python."}],
)
print(r.choices[0].message.content)
```

Thinking is on by default; `"reasoning_effort": "none"` turns it off.

## Hardware sizing

VRAM + RAM should hold most of the 187.6 GB model; anything that does not fit is read from the GGUF.

| Machine | Example flags |
|---|---|
| 2× 24 GB GPU, 192 GB RAM | `--gpu0-gb=-1.0 --gpu1-gb=-1.0 --arena-gb 120` |
| 2× 24 GB GPU, 128 GB RAM | `--gpu0-gb=-1.5 --gpu1-gb=-1.0 --arena-gb 110` |
| 1× 32 GB GPU, 128 GB RAM | `--gpu0-gb=-2.0 --gpu1-gb 0 --arena-gb 80`, env `DSX_ADAPT=0` |

Also needed: CUDA 13, Rust, an AVX2 CPU, and about 400 GB of disk for the GGUF, Engram shards and tokenizer.

## Files

| File | |
|---|---|
| `DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf` | the model, 187.6 GB, single GGUF |
| `vision.safetensors` | the vision tower, aligner and image embeddings from the official checkpoint (bf16, 0.97 GB), for `--vision` |
| `expert-rank.json` | expert hotness ranking, used to place experts in VRAM and RAM |

The local rank file is generated as `rank.json`; upload it as `expert-rank.json`. The Engram tables are not part of
the GGUF.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format ·
antirez for the calibration prompts ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and the ggml authors, MIT) for the web UI.

Quantization pipeline: [github.com/lxe/dsx/tree/main/quant](https://github.com/lxe/dsx/tree/main/quant).