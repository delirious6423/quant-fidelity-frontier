---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-Flash-Next · Orinfer E8P Q2A8

[中文](README_zh.md) · [Hugging Face](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8)

A quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) for **Jetson AGX Orin 64GB / CUDA SM87**, deployed with [Orinfer v0.4.1](https://github.com/iMMIQ/orinfer/releases/tag/v0.4.1). It supports text, images and multiple images, streaming Chat Completions, native MTP, prompt lookup, prefix caching, continuous batching and resumable JSONL offline inference. Translation requests use `enable_thinking: false`; thinking follows the original model's chat template.

## Quantization and files

Routed experts use integer E8P near-2-bit weights with INT8 computation. GDN QKV/Z, QSA Q/gate, all 48 `ssm_out` / `attn_output` (`2560 × 6144`) output projections, and the shared output head use single-resident signed-block128 Q4A8. Shared FFN and sensitive MTP projections retain W8; HC retains BF16 and vision FP16. CPU embedding/PLE tables use bounded row caches.

Deployed read-only GPU data, including scales, derived constants and precision exceptions: **38,487,903,152 bytes** (`-377.2 MB` vs. v0.2.0–v0.4.0), approximately **2.391 bits per recorded GPU weight parameter**. Stored safetensors: **52.56 GB** across **35 shards** (27 GPU shards, `38.49 GB`; 8 CPU shards, `14.08 GB`). Context capacity: **262,144 tokens**; KV cache is group-64 INT8 with FP16 scales and persistent GDN state is FP32. Quantization seed: `20261002`.

The repository contains HF configuration/tokenizer assets, the native chat template, standard safetensors containers and `cache/model.json`. The descriptor fixes physical tensor layouts and the Schema 2 execution-package digest. The matching statically linked `orinfer` binary and pure-data Schema 2 execution package (`package.json`, `bindings.json`, `assets.safetensors` with cubins and source) are distributed through GitHub Release.

## Deploy

Download revision [`v0.4.1`](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8/tree/v0.4.1) of this model repository into `./model`. Prebuilt deployment uses the aarch64 Rust CLI, NVIDIA CUDA Driver and the matching SM87 Schema 2 execution package.

Execution-package digest: `28040d50317572c45552b2da77517fbffb49da0547ca3ea8882c1cff2db9664a`.

For an existing v0.2.0–v0.4.0 download, refresh only the 6 repacked Row-Q4 dense shards (`cache/weights/dense-l000-l007.safetensors` through `cache/weights/dense-l040-l047.safetensors`, ~3.72 GiB), [`cache/weights/model.safetensors.index.json`](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8/resolve/v0.4.1/cache/weights/model.safetensors.index.json) and [`cache/model.json`](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8/resolve/v0.4.1/cache/model.json), then install the v0.4.1 CLI and execution archive. The other 29 safetensors shards (~48.6 GB) and tokenizer assets are unchanged.

For interactive multi-turn and concurrent agent workloads, start with `--mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096`. Add `--policy throughput` when prioritizing aggregate throughput over strict TPOT bounds.

```bash
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/orinfer-v0.4.1-linux-aarch64.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/orinfer-exec-flash-next-a8-sm87-v0.4.1.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.4.1-linux-aarch64.tar.xz
./orinfer-v0.4.1-linux-aarch64/bin/orinfer package install ./orinfer-exec-flash-next-a8-sm87-v0.4.1.tar.xz
./orinfer-v0.4.1-linux-aarch64/bin/orinfer validate-model ./model
./orinfer-v0.4.1-linux-aarch64/bin/orinfer serve ./model \
  --model qwen3.8-flash-next --cuda-graph decode_only \
  --mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096
```

The listener is `[::]:8088`. OpenAI Chat Completions uses `/v1/chat/completions`; set `stream: true` for SSE. Images use `image_url`, and thinking output uses `reasoning_content`. The CLI also provides Ollama-compatible APIs and `orinfer batch` for JSONL jobs. See the [serving guide](https://github.com/iMMIQ/orinfer/blob/v0.4.1/docs/serving.md).

## Measured performance

### 240s Mini-Weka 512-trajectory benchmark (`v0.4.1`, **2026-11-23**)

Jetson AGX Orin 64GB (R36.4.3, CUDA SM87), `--cuda-graph decode_only --mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096 --max-active-requests 4`, compact 512 multi-turn agent trajectories (cold first turns 2/3/4/6 Ki tokens, concurrency 4, 240 s wall limit, `temperature=1.0`, `ignore_eos=true`):

| Policy | Completed Reqs | Full Trajectories | Aggregate Output TPS | TTFT P50 / P95 (s) | Prefix Cache Hit | Errors / Graph Evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `latency` | **102** | 12 | **46.50** | 1.00 / 4.08 | 79.3% | 0 / 0 |
| `throughput` | **102–105** | 13 | **46.31–48.15** | 0.80 / 3.29 | 79.2%–79.4% | 0 / 0 |

Output TPS includes queuing, prefill, generation and HTTP/SSE over the 240 s profiling window (`latency` `max_decode_gap_s = 0.401 s`, `prefill_suspend_s + prefill_resume_s = 0.387 s`).

### Fixed-length diagnostic snapshot (`v0.4.0`, **2026-11-20**)

Historical single/multi-request fixed-length diagnostics from v0.4.0 (`fc87e2ca`, greedy, seed `20261002`, `decode_only`, thinking off, prefix cache off; median of 3 warm runs). See the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.4.1/docs/performance.md) for methodology.

| Prefill 512 | Prefill 8K | Single decode, MTP (512 / 8K) | 8-request decode | Max requests / decode | Serve → API (s) | Serve → first output (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 979.3 | 1,081.8 | 62.9 / 31.5 | 163.3 | 32 / 284.8 | 15.10 | 15.27 |

## Quality scope

Validation on `v0.4.1` (`2026-11-23`, `seed=20261002`) covers teacher-forced probability diagnostics against official CPU BF16 and long-context baselines, plus targeted GPU state/contract suites:

- **vs. Official CPU BF16 (64 positions)**: `mean_target_nll_delta = -0.0322` (vs. v0.4.0 `-0.1148`), `max_abs_target_logprob_delta = 2.187`, `0 / 60` high-confidence top-1 changes.
- **vs. Frozen v0.4.0 512 / 2K / 8K long context (176 positions)**: `mean_target_nll_delta = +0.0197`, `0 / 160` high-confidence top-1 changes.
- **GPU contracts**: `validate_sampled_lookup_requests`, `validate_layered_prefill` (`kl = 0.0`, `state_equal = true` across all 4 interruption regimes), and `validate_flash_prefix_cache` (503 / 2033 / 8187 tokens exact state and token match) all passed.

These are controlled diagnostics covering text, inputs through 8K and image API paths. Broader task benchmarks and full-context quality remain ongoing evaluation areas; see the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.4.1/docs/performance.md).

## License

Model weights retain [Qwen Community License 1.0](LICENSE). Orinfer software and its separate execution package use LGPL-3.0-or-later; source and third-party notices ship with the release.
