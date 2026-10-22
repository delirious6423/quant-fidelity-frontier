---
library_name: gguf
tags:
- gguf
- quantized
- llama.cpp
- mixed-precision
- imatrix
- qwen
- qwen3.8
- moe
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next — 91GB (GGUF)

Mixed-precision quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) using a proprietary quantization method by [baa.ai](https://baa.ai).

Per-tensor bit-width allocation via advanced sensitivity analysis and budget-constrained optimisation, built at the efficiency knee (best quality-per-GB): 4× smaller than the 360 GB BF16 release, with a lower WikiText-2 perplexity than the best public build of the same size.

An MLX build of the same recipe for Apple Silicon is at [baa-ai/Qwen3.8-Flash-Next-RAM-91GB-MLX](https://huggingface.co/baa-ai/Qwen3.8-Flash-Next-RAM-91GB-MLX).

## Metrics

| Metric | Value |
|--------|-------|
| **Size** | **91.2 GB** (91,217,068,384 bytes; 9 shards) |
| Average bits per weight | 4.12 |
| Format | llama.cpp (GGUF), arch `qwen4exp` |
| Architecture | 180B total / 6B active: 125B hybrid Gated-DeltaNet + Qwen Sparse Attention MoE (512 routed experts, top-10 + 1 shared), 51B n-gram embedding table, 4B MTP head |
| WikiText-2 perplexity (40 × 2048 tokens) | **3.51** (± 0.04) |
| unsloth `UD-Q3_K_XL` (90.0 GB), same test | 3.59 (± 0.04) |

Routed experts `Q3_K` (gate/up) and `IQ4_NL` (down — its 640-wide rows cannot take K-quants); the 51B-parameter n-gram embedding table `IQ4_NL`; attention, Gated-DeltaNet projections, shared experts and gated-residual mixers `Q8_0`; output and token embeddings `Q6_K`; router, norms, convolutions and indexer `F32`/`F16`. Expert tensors are calibrated with the importance matrix from [AesSedai/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF). The per-tensor spec is included as `spec_qwen38_flash_next_knee.txt`.

Allocation variants measured while building (same WikiText-2 test):

| Variant | Size | PPL |
|---|---|---|
| Sensitivity-allocated knee, `Q4_0` tables, no imatrix | 89.6 GB | 3.85 |
| + imatrix, `IQ4_NL` tables | 89.6 GB | 3.69 |
| **+ `Q8_0` attention / shared / mixer floor (this release)** | **91.2 GB** | **3.51** |

## Files

| File | Contents | Size |
|------|----------|------|
| `Qwen3.8-Flash-Next-RAM-local-knee-0000{1..9}-of-00009.gguf` | Text model (shard 1 holds metadata and the chat template; shard 2 is the n-gram table) | 91.2 GB |
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` | Vision projector, F16 | 0.9 GB |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | Multi-token-prediction draft head, Q8_0 (optional, for speculative decoding) | 4.1 GB |
| `spec_qwen38_flash_next_knee.txt` | Per-tensor quantization spec | — |

## Requirements

**llama.cpp build ≥ b11671** (October 2026) is required — Qwen3.8-Flash-Next support (`qwen4exp`: hybrid Gated DeltaNet + Qwen Sparse Attention, gated residuals and the n-gram PLE table) is not in older Homebrew `llama.cpp` releases. Build from source or use a newer release until downstream runtimes catch up.

## Usage

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build -j --target llama-cli llama-server

hf download baa-ai/Qwen3.8-Flash-Next-RAM-91GB-GGUF --include "*.gguf" --local-dir ./qwen3.8-flash-next-ram-91gb

# The model is split into 9 shards — point -m at the first; llama.cpp loads the rest automatically
./build/bin/llama-cli -m ./qwen3.8-flash-next-ram-91gb/Qwen3.8-Flash-Next-RAM-local-knee-00001-of-00009.gguf -p "Hello!" -n 256 -ngl 99 --jinja
```

Or via `llama-server` for an OpenAI-compatible HTTP API (add `--mmproj` for image input):

```bash
./build/bin/llama-server -m ./qwen3.8-flash-next-ram-91gb/Qwen3.8-Flash-Next-RAM-local-knee-00001-of-00009.gguf \
    --mmproj ./qwen3.8-flash-next-ram-91gb/mmproj-Qwen3.8-Flash-Next-F16.gguf \
    --port 8080 -ngl 99 --ctx-size 32768 -fa on --jinja
```

Fits on a 128 GB Mac or a 2× 80 GB GPU node. Thinking is on by default; disable it per request with `"chat_template_kwargs": {"enable_thinking": false}`. The upstream chat template is embedded (also included as `chat_template.jinja`).

Verified on a 192 GB Apple M2 Ultra (Metal, `-ngl 99`): correct answers on factual, arithmetic, code and reasoning prompts with thinking on and off.

## License

Released under the [Qwen Community License 1.0](LICENSE), included in this repository.

---
*Quantized by [baa.ai](https://baa.ai)*

---

## Black Sheep AI Products

Black Sheep AI builds private AI with the **Bounded AI Architecture (BAA)**: three layers, each bound to something you control, on infrastructure you control. Nothing leaves your boundary.

**[Ram](https://baa.ai/compression.html)** (Reasoning Layer) — Full-capability frontier models, shrunk to fit smaller GPUs with reasoning kept and certified. They rely strictly on your Data Modules for knowledge, so you swap the model as better ones ship.

**[Paddock](https://baa.ai/paddock.html)** (Knowledge Plane) — Binds your Data Modules directly to the reasoning layer: one per topic, period, person, permission or tenant. Permission, freeze, hand back or delete each one on its own.

**[Watchman](https://baa.ai/watchman.html)** (GRC Layer) — Governance, risk and compliance in a single, complete audit log: every module permission, every bind, every model's provenance. Five certificates before anything goes live.

**[Shepherd](https://baa.ai/shepherd.html)** — Takes a frontier model, compresses it into a Ram reasoning model, certifies it with Watchman, binds it to your Paddock, and deploys the Bounded AI Architecture to your infrastructure: your cloud, a Mac Studio, or an air-gapped facility.

Learn more at **[baa.ai](https://baa.ai)** — Private AI for enterprises whose data cannot leave the building.
