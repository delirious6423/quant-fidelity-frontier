---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: fastasfuck
pipeline_tag: image-text-to-text
tags:
  - fastasfuck
  - strix-halo
  - amd
  - gfx1151
  - rdna3.5
  - moe
  - qwen3_8
  - speculative-decoding
  - mtp
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next (FastAsFuck `.faf` Quantization)

<p align="left">
  <a href="https://huggingface.co/collections/Cheblan/fastasfuck-quantizations"><img alt="Collection: fastasfuck quantizations" src="https://img.shields.io/badge/collection-fastasfuck%20quantizations-ffcc4d?style=flat-square"></a>
  <a href="https://github.com/Cheblan/fastasfuck"><img alt="Engine: fastasfuck" src="https://img.shields.io/badge/engine-fastasfuck-blue?style=flat-square"></a>
</p>

Pre-quantized production release of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** in `.faf` binary tensor format, part of the **[fastasfuck Quantizations Collection](https://huggingface.co/collections/Cheblan/fastasfuck-quantizations)**, specifically engineered for the **[fastasfuck](https://github.com/Cheblan/fastasfuck)** inference engine on AMD Strix Halo unified memory architecture (`gfx1151`, AMD Ryzen AI Max+ 395, 128 GB Unified LPDDR5X-8533).

Qwen3.8-Flash-Next is the open foundation mixture-of-experts model featuring hybrid GDN linear attention, QSA windowed attention, 512 routed experts, and native MTP draft heads. Combined with the fastasfuck engine, this release delivers sustained, zero-stall multi-agent decode throughput.

---

## 1. Included Artifacts & Files

| File | Size | Description |
|---|---|---|
| `weights.faf` | ~65.5 GiB | Hybrid Q6 / Q4 / Q8 quantized MoE base weights in unified fastasfuck binary tensor layout. |
| `ngram.table` | ~48.3 GiB | 8,388,608-row int8 Prefix Lookahead Engine (PLE) speculative table. |
| `vision.faf` | ~856 MiB | Native multimodal vision projection tower for image input. |
| `tokenizer.json` | ~12.2 MiB | Fast Hugging Face BPE tokenizer vocabulary. |
| `config.json` | ~4.7 KiB | Model architecture configuration and dimensions. |
| `generation_config.json` | ~0.2 KiB | Default sampling parameters and generation configuration. |
| `chat_template.jinja` | ~8.7 KiB | Jinja2 chat template supporting tool calls and thinking blocks. |
| `convert.log` | ~173 KiB | Conversion log from Safetensors to Hybrid Q6 `.faf`. |
| `convert_ngram.log` | ~1.7 KiB | Compilation log of the 8M-row int8 PLE n-gram table. |
| `inspect_weights.txt` | ~100 KiB | Full tensor layout inspection and quantization metadata. |
| `inspect_vision.txt` | ~32 KiB | Vision projection tower tensor inspection and layout metadata. |

---

## 2. Hardware Performance (AMD Strix Halo Ground Truth)

All metrics benchmarked on bare-metal **AMD Ryzen AI Max+ 395** (40 CUs RDNA 3.5 `gfx1151`, 128 GB LPDDR5X-8533 @ 273 GB/s, Linux Kernel 7.3-zen):

| Metric | Measured Throughput / Latency | Condition / Configuration |
|---|---|---|
| **Solo Greedy Decode (T = 0.0)** | **73.8 tok/s Grand Mean** (peak **89.7 tok/s**) | 1 Active Slot, sterile cold prefill (`prompt-cache = off`) |
| **Multi-Agent Decode (4 Slots)** | **104.4 tok/s aggregate mean** (peak **115.5 tok/s**) | 4 concurrent agent slots under full continuous load |
| **Prefill Bandwidth (32k Context)** | **~1,820 tok/s** (Monotonic) · **~1,458 tok/s** (Real Code) | TTFT: 17.6 s (Monotonic) / 21.2 s (Real Code) |
| **Prefill Bandwidth (128k Context)** | **~1,721 tok/s** (Monotonic) · **~1,378 tok/s** (Real Code) | TTFT: 74.4 s (Monotonic) / 91.8 s (Real Code) |
| **Effective Inter-Token Latency** | **13.0 ms – 32.2 ms** | 7 MTP draft steps + 8M-row PLE n-gram table |
| **Bit-Identity Law** | **100.000% exact SHA-256 match** | Deterministic token-for-token reproducibility at T = 0.0 |

Full methodology, per-prompt statistics, and benchmark suites: [docs/BENCHMARKS.md](https://github.com/Cheblan/fastasfuck/blob/main/docs/BENCHMARKS.md).

---

## 3. Quick Start with `fastasfuck`

### Step 1: Download Model Files
```bash
# Using huggingface-cli
huggingface-cli download Cheblan/Qwen3.8-Flash-Next-faf --local-dir /models/Qwen-3.8-FN-faf
```

### Step 2: Configure Server (`deploy/fastasfuck-server.conf`)
Point the configuration file (or CLI flags) to the downloaded model directory:

```ini
model-id = qwen3.8-flash-next
weights = /models/Qwen-3.8-FN-faf/weights.faf
ngram = /models/Qwen-3.8-FN-faf/ngram.table
tokenizer = /models/Qwen-3.8-FN-faf/tokenizer.json
generation-config = /models/Qwen-3.8-FN-faf/generation_config.json
```

Start the inference server:
```bash
# Native binary build
build/strix/fastasfuck_server --config deploy/fastasfuck-server.conf

# Or via Podman / Docker
podman run -d --device=/dev/kfd --device=/dev/dri -p 5300:5300 \
  -v /models/Qwen-3.8-FN-faf:/models:ro \
  ghcr.io/cheblan/fastasfuck:latest --config /models/fastasfuck-server.conf
```

### Step 3: Query via OpenAI API or Responses API
```bash
curl -s http://localhost:5300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain the Riemann-Lebesgue Lemma"}], "temperature": 0.0}'
```

Compatible with **Codex CLI**, **OpenCode 2**, and **Cursor** using `/v1/responses` with zero-overhead KV cache reuse across multi-turn sessions.

---

## 4. Attribution & Licenses

- **Quantized and Packaged by:** [Cheblan / fastasfuck team](https://github.com/Cheblan/fastasfuck).
- **Original Model Weights:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen Team.
- **Base Foundation Model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen Team.
- **License:** Governed by the [Qwen Community License 1.0](LICENSE).
