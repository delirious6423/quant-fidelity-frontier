---
license: apache-2.0
library_name: goethe
tags:
  - moe
  - ternary
  - quantization
  - deepseek
  - speculative-decoding
base_model: deepseek-ai/DeepSeek-V4.1-Flash
---

# DeepSeek-V4.1-Flash ternary experts

Ternary routed-expert overlay for [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

This repository contains Hadamard-rotated ternary expert packs, format `ternary-g128-hadamard-gptq`, 40 layers. Attention, router, shared experts, Engram, and the bundled DSpark draft stay in the base model. The full base weights are not re-uploaded here.

## Drafter

| Path | Role |
|------|------|
| DSpark draft MoE in the base checkpoint | speculative draft (`--speculative-algorithm DSPARK`) |

Default: DSpark block size 5.

## Serve: 4×GPU, concurrency 1

```bash
git clone https://github.com/meshapplied/goethe
cd goethe
pip install -e ./ternary-kernels
```

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest")
ternary_runtime.bootstrap()
```

```bash
export TERNARY_PACK_DIR=...
export TERNARY_MODEL_DIR=...
export TERNARY_PROFILE=deepseek_v41
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 4 --ep-size 4 \
  --trust-remote-code \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 1 \
  --max-running-requests 1 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --mem-fraction-static 0.88 \
  --context-length 65536 \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce \
  --disable-flashinfer-autotune \
  --load-format safetensors \
  --served-model-name deepseek-ternary \
  --host 127.0.0.1 --port 8000
```

## Serve: 4×GPU, concurrency 4

Same command with:

```bash
  --max-running-requests 4 \
  --cuda-graph-max-bs-decode 4
```

## Performance

Hardware unless noted: **4×** RTX PRO 6000 96 GB, **TP=EP=4**, Goethe `ternary_runtime` + SGLang, DSpark γ=5, mem 0.88, chunk 8192, context 65536, T=0. Expert overlay: `meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest` on base `deepseek-ai/DeepSeek-V4.1-Flash`.

### VRAM

Serve knobs for this snapshot: mem 0.88, chunk 8192, context 65536, max_running 1, `--disable-radix-cache`.

| | |
|--|--:|
| GPU used (peak rank) | **86.1 GB** (86 090 MiB) |
| GPU total | 95.6 GB (97 887 MiB) |
| Per-rank scheduler | 82.5–83.7 GB (84 464–85 700 MiB) |
| `max_total_num_tokens` | ~20.4M |

### Agentic coding throughput

Replay of **10 SWE-bench Pro / DeepSWE mini-swe-agent conversations × 6 turns** each. Full chat history + tool observations fed back each turn; prefix cache on.

| Concurrency | out tok/s | accept length | TTFT ms | ITL ms |
|------------:|----------:|--------------:|--------:|-------:|
| 1 | **194** | 3.78 | 425 | 3.5 |
| 4 | **309** | 4.08 | 474 | 9.8 |

### Random prompts

`sglang.benchmark.serving` backend `sglang-oai-chat`, **32** prompts, in=512, out=256, radix cache on.

| Concurrency | out tok/s | accept length | TTFT ms | ITL ms |
|------------:|----------:|--------------:|--------:|-------:|
| 1 | **133** | 2.92 | 284 | 5.6 |
| 4 | **233** | 2.90 | 355 | 14.6 |

### Cold prefill and long-context decode

Same stack with `--disable-radix-cache` and `--flush-cache` between lengths. `sglang.benchmark.serving` random, 8 prompts, concurrency 1.

| prompt | decode tok/s | prefill tok/s | TTFT mean / p50 / p90 / p99 (ms) | ITL mean / p50 / p90 / p99 (ms) | accept |
|------:|-------------:|--------------:|--------------------------------:|-------------------------------:|-------:|
| 1024 | 75 | ~0.9k | 1080 / 1085 / 1982 / 2828 | 5.8 / 4.1 / 16.0 / 16.6 | 2.83 |
| 4096 | 99 | ~6.5k | 629 / 600 / 984 / 1000 | 5.7 / 4.1 / 8.5 / 16.8 | 2.88 |
| 16384 | 27 | ~3.7k | 4376 / 2203 / 8701 / 20339 | 5.8 / 4.1 / 8.6 / 16.6 | 2.88 |
| 32768 | 32 | ~9.1k | 3617 / 3301 / 6158 / 7080 | 5.5 / 4.1 / 8.3 / 16.6 | 3.02 |

Decode slows as context grows; ITL stays flat.

## Speculative decoding

| knob | value |
|------|------:|
| algorithm | DSPARK (draft MoE in base checkpoint) |
| dspark_block_size | 5 |

## Quality (held-out NLL)

v2 calib held-out, ternary vs FP4 on the same set:

| Domain | FP4 NLL | ΔNLL |
|--------|--------:|-----:|
| code | 0.523 | 0.105 |
| math | 0.727 | 0.086 |
| general | 0.439 | 0.415 |
| overall | 0.563 | 0.202 |

## Format

- `ternary_config.json`: profile `deepseek_v41`, format `ternary-g128-hadamard-gptq`, `w123` layout, 40 layers
- `layer-00.safetensors` … `layer-39.safetensors`
- `manifest-layer-*.json`, `eval.json`
- Hadamard-rotated ternary GPTQ, group 128. Fused `w1`/`w2`/`w3`. Unused experts fall back to FP4 from the base checkpoint.

## License

- **This ternary pack / card:** Apache-2.0
- **Base model:** DeepSeek-V4.1-Flash license on the base repo
