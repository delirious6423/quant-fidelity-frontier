---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - deepseek
  - moe
  - v100
  - volta
  - gguf
  - quantized
---

# DeepSeek-V4.1-Flash-1017-Volta

DeepSeek-V4.1-Flash-1017 for [Volta](https://github.com/lxe/volta), an inference engine for eight V100-SXM2-32GB GPUs with NVLink in the DGX-1 layout. Its main routed experts use cb3 at 3.25 bits per weight, with cb2 at 2.25 bits for the coldest experts, chosen by routing frequency. The expert weights occupy about 220 GB, or 27.5 GB per GPU.

The entire model is distributed as **one `DeepSeek-V4.1-Flash-1017-Volta.gguf` file**, about 442 GB. It contains the original base tensor bytes, Engram tables, draft and vision weights, tokenizer and model configuration, plus the quantized main routed experts. Volta reads these tensors directly from the mapped file; no separate base directory, tokenizer file or extraction cache is required.

The main expert tensors contain row-major packed codes and scale bytes. Volta reconstructs its kernel layout when loading them. **This file requires Volta's custom cb3/cb2 decoder; it is not a llama.cpp-compatible quantization type.** GGUF tensors and Volta metadata preserve the original tensor dtypes, source slice boundaries and multiplier bits.

| Path | Contents |
|---|---|
| `DeepSeek-V4.1-Flash-1017-Volta.gguf` | The complete model, tokenizer and runtime metadata in one file. About 442 GB. |
| `quantization/` | Format specification, checksums and exhaustive lossless-conversion receipts. |
| `LICENSE`, `README-original.md` | The original model's license and README, also preserved inside the model file. |

## Repack verification

The October 24, 2026 repack replaces both the former `base/` directory and `pack-27.5gb-8xv100/gpu*.bin` files with one GGUF. It changes storage layout without requantizing the model. Every source code byte, scale byte, tier, owner, TP slice and float32 multiplier bit was checked against the reconstructed native representation. The source is Hugging Face revision `d0a725d14de03275eea9efe8e4e6329ebe07665f`; its per-file SHA-256 values and the new GGUF hash are recorded in `quantization/manifest.json`.

The exhaustive comparison passed for 245,760 expert matrices, 3,925 base tensors and 20 embedded assets. The complete-file CPU load check constructed the main model, DSpark draft, Engram views and tokenizer using only this GGUF. See `quantization/verify-full.json` and `quantization/smoke.json` for the exact scope and counts.

The original eight-way source slices and their multiplier domains are retained, so this GGUF still requires the original eight-GPU topology. Every base tensor and embedded asset was compared byte for byte against the published source files. The layout-only publication initially included CPU and byte-equivalence checks; subsequent full-GGUF GPU verification is recorded below.

## Results measured with the complete GGUF

On October 24, 2026, the complete GGUF loaded and served requests with frozen Volta commit `cc54d97568ad` on eight V100-SXM2-32GB GPUs with active NVLink. The rental had 754 GiB host RAM and 800 GB of storage. A full-file SHA-256 check matched the published model hash before performance measurement.

| Measurement | Result and scope |
|---|---|
| Teacher evaluation, 8 × 1,024 tokens | 8,184 evaluated positions; KL top-256 **0.126695**, top-1 agreement **84.3231%**, perplexity **8.9423** |
| Four concurrent Pi coding tasks | Cube, pyramid, torus and sphere programs completed; all four animations ran and changed frames in their assigned tmux panes |
| Agent workload | 47 model requests, 19,805 generated tokens; **167.3 tokens/s mean aggregate** during the interval with at least two active sessions |
| Expert reconstruction and GPU upload | About **169–180 seconds** in these observed loads; Engram mapping added about 8–13 seconds |

The agent run used one DP slot per GPU, context capacity 131,072, prefill chunk 64, five speculative drafts, temperature 0.7, top-p 1 and graph budget 4. A separate controlled retrieval test subsequently exercised 126,041 actual prompt tokens with zero cache reuse and returned the exact key placed at the beginning; time to first token was 226.50 seconds. This establishes that one retrieval workload, rather than general long-context quality. The throughput figure excludes the separate API smoke and is specific to the generated task work. It is neither a per-agent rate nor a controlled speed comparison against the historical native-pack runs. The server retained only about 0.5 GB free per GPU after loading, so larger contexts or session configurations require their own memory checks.

Exact metrics, model/binary hashes, configuration, recording hash and animation checks are in [runtime/2026-10-24-v100-baseline.json](runtime/2026-10-24-v100-baseline.json). A later sampler eligibility change keeps prefill sampling on the GPU while retaining decode, top-p, hesitation and head-parallel exclusions. Four fresh servers in A/B/B/A order, with matched prompts, full output budgets and zero cache hits, measured 15.8–17.7% higher aggregate throughput for the mixed four-client workload and 9.6–12.3% for the fixed four-client workload. The compared work was 2,048 generated tokens in each trace. Sampled strings differed, so these are matched-work comparisons rather than claims of identical seeded output.

With the sampler change, separate cold 4K-input/1,024-output screens measured 165.2–169.3 aggregate whole-request tokens/s for four clients and 267.2–273.2 for eight clients. One-client decode rate was 80.6–84.4 tokens/s. The actual four-Pi tool gauntlet also completed all four shape programs and live animations; its variable task work is not a controlled speed comparison. The sampler source and receipts are in [Volta PR #1](https://github.com/lxe/volta/pull/1).

A separate head-parallel configuration with four active slots and capacity 262,144 passed the same 8 × 1,024-token teacher gate: perplexity 8.9417, KL top-256 0.129511 and top-1 agreement 84.0787%. It also completed a real 250,042-token prompt with zero cache reuse and returned the exact initial deployment key. Time to first token was 468.57 seconds. This was one active long request; it does not establish four simultaneous 250K requests or general long-context quality. The 1Hz memory guard observed at least 1,743MiB free on every GPU throughout this screen.

The head-parallel configuration was slower for four short requests: 75.5–76.0 aggregate whole-request tokens/s in the cold 4K-input/1,024-output profile. Eight submitted requests queued behind four active slots. One-client results varied substantially, so the best observed head-parallel decode rate is not a repeatable speedup claim. The measured data-parallel 128K configuration remains the preferred throughput setting for four agents.

The measured sampler source is [commit `4ce1545`](https://github.com/lxe/volta/commit/4ce1545); the matching short/long-context receipts are linked from [Volta PR #1](https://github.com/lxe/volta/pull/1). The complete public runtime summary is [runtime/2026-10-24-v100-optimized.json](runtime/2026-10-24-v100-optimized.json). Further throughput experiments are continuing; the model file and quantization remain unchanged.

## Results measured with the native pack

| Measurement | Result |
|---|---|
| Quality, Lucebox eval: 8 × 1,024 tokens against the FP8 model's top-256 | KL 0.127, top-1 agreement 84.1%, perplexity 8.88 |
| 64 concurrent sessions, one token per step | 41.7 ms per step: 1,536 tok/s aggregate |
| 256 / 512 sessions | 2,440 / 3,070 tok/s aggregate |
| DSpark draft, greedy, 64 sessions writing code | 28.4 tok/s per session |
| API server, temperature 1 | One stream 93 tok/s decode; 32 streams 39 tok/s each, 967 tok/s aggregate throughput |
| Eight sessions with 64K-token prompts | Prefill 2,490 tok/s total; decode 71 tok/s per session |

These measurements are recorded in the engine README at commit `362dded4c85095bf56d5ada6ff95d54415f960e8` and come from different configurations; they are not simultaneous guarantees. The API figures use 300-token sampled answers and distinguish request decode rate from aggregate throughput. See the [engine README](https://github.com/lxe/volta) for configurations and subsequent serving measurements.

## Run

Use [Volta commit `cc54d97568ad`](https://github.com/lxe/volta/commit/cc54d97568adb3d9438a8692c3415cb72ac18957) or a later version supporting expert GGUF. The machine needs eight 32 GB V100 GPUs with active NVLink in the DGX-1 layout, about 400 GB of RAM, and about 442 GB of storage. CUDA 12.9 is the last toolkit that builds for sm_70; set `NVCC` to its compiler if another CUDA toolkit is also installed.

```bash
git clone https://github.com/lxe/volta.git
cd volta
git checkout cc54d97568adb3d9438a8692c3415cb72ac18957
hf download lxe/DeepSeek-V4.1-Flash-1017-Volta --local-dir volta-weights
NVCC=/usr/local/cuda-12.9/bin/nvcc CUDA_ARCHS=70 cargo build --release -p dsx --features cuda --bin volta-server

./target/release/volta-server \
  --model volta-weights/DeepSeek-V4.1-Flash-1017-Volta.gguf --host 0.0.0.0 --port 8090
```

The default configuration serves 32 sessions of 8,192 tokens. `--slots 2 --ctx 131072` serves 16 sessions of 128K tokens. Loading the GGUF reconstructs the expert kernel layout before uploading it to the GPUs; it does not run the quantizer. Observed expert reconstruction and GPU upload took about 169–180 seconds in the full-GGUF verification above; these loads are not a controlled cold-cache startup benchmark. The two Engram tables occupy 196 GB of host RAM, and reading them from storage adds loading time.

The server provides OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs, streaming, tool calls and thinking levels. It batches requests, verifies DSpark drafts by speculative sampling, and reuses cached prefixes. A browser chat is available at `/` and an admin dashboard at port 8091.

## Format and reconstruction

Read [quantization/FORMAT.md](quantization/FORMAT.md) for tensor names, codebooks, scale conventions, embedded metadata and source slice domains. Pass the complete GGUF as `--model`; the engine selects its expert tensors and embedded tokenizer automatically. Previously downloaded base directories and native expert packs remain supported through the old `--model BASE_DIR` and `VOLTA_PACK=PACK_DIR` combination. The original quantization plan is in [Volta's plans directory](https://github.com/lxe/volta/tree/main/plans).

## License and credit

MIT, the same as the original model. [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) is by DeepSeek. Its original README and license are retained as `README-original.md` and `LICENSE`, and embedded in the GGUF.
