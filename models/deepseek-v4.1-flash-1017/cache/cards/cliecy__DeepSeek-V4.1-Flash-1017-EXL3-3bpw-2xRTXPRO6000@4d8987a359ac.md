---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- vllm
- deepseek
- deepseek_v41
- mixture-of-experts
- 3-bit
- 1m-context
- rtx-pro-6000
- blackwell
- sm120
---

# DeepSeek-V4.1-Flash-1017 · EXL3 3 bpw · 2× RTX PRO 6000, ready to run

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with its **full 1M-token context**,
on two RTX PRO 6000 Blackwell cards (96 GB each, PCIe), at **~90–110 tok/s single-stream decode**.

This repository has everything needed: the weights, a prebuilt runtime overlay (vLLM + EXL3, compiled for sm_120), and
`run.sh`. **No Docker, no root, no compiling.**

## Quickstart

```bash
hf download cliecy/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-2xRTXPRO6000 --local-dir dsv41-1017   # ~335 GB
bash dsv41-1017/run.sh
```

`run.sh` does three things:
1. Checks the GPUs, host RAM and bubblewrap.
2. Sets up the runtime once in `~/.cache/dsv41-1017` (~24 GB). It pulls the official `vllm/vllm-openai` image by
   digest with `crane` (~9 GB download from Docker Hub, no Docker needed) and applies this repo's overlay.
3. Serves an OpenAI-compatible API on `http://127.0.0.1:8000/v1` with model name `deepseek-v4.1-flash-1017`.

It prints the API key it generated, and `[ready]` once the server is up.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $(cat ~/.cache/dsv41-1017/api_key)" \
  -H 'Content-Type: application/json' -d '{"model": "deepseek-v4.1-flash-1017",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "chat_template_kwargs": {"thinking": false}}'
```

Tool calls (OpenAI `tools`), reasoning output (`thinking: true`, the default) and image input work.

**Options** (environment variables):

| Variable | Default |
|---|---|
| `HOST` | `127.0.0.1` |
| `PORT` | `8000` |
| `API_KEY` | generated once and saved |
| `SERVED` | `deepseek-v4.1-flash-1017` (model name) |
| `DSV41_HOME` | `~/.cache/dsv41-1017` (unpacked runtime and kernel caches) |
| `THINKING` | `default`; `off` disables thinking for all requests |
| `CUDA_VISIBLE_DEVICES` | all GPUs |

Commands:
- `bash run.sh --check`: only check the host.
- `bash run.sh --dry-run`: print the exact server command.

## Requirements

- **GPUs:** 2× NVIDIA sm_120 with 96 GB each (RTX PRO 6000 Blackwell, Workstation or Max-Q). The server uses ~95 GB per GPU at 1M context.
- **Driver:** NVIDIA driver with CUDA 13 support (tested: 595.91.07 / CUDA 13.2).
- **Host RAM:** ~245 GiB. The server pins ~103 GiB (int4 Engram tables + 30% of the experts) and uses ~190 GiB in total.
- **Linux:** x86_64, `bwrap` (bubblewrap) with unprivileged user namespaces allowed, plus `zstd`, `curl` and `python3`.
  The first setup needs network access to Docker Hub and GitHub (to pull the image and the `crane` binary).
- **Disk:** ~335 GB for this repo and ~30 GB for the runtime in `DSV41_HOME`. A fast disk shortens the first start a lot.

## Measured results

2× RTX PRO 6000 Max-Q (300 W, PCIe, no NVLink), the defaults of `run.sh`, warm server, real source-code prompts.

| Prompt | Prefill tok/s | Decode tok/s, 1 stream | 2 streams (total) | 4 streams (total) |
|---:|---:|---:|---:|---:|
| 1.8K | 1,990 | **92.8** | 121.8 | 132.5 |
| 26K | 5,179 | **88.9** | 117.1 | 162.1 |
| 109K | 5,085 | **93.3** | 140.8 | 161.2 |
| 430K | 5,085 | **99.5** | | |
| 890K | 4,866 | **111.6** | | |

- **Thinking on:** 107.9 tok/s single-stream decode at 1.6K, 89.7 tok/s at 25K. DSpark speculative decoding produced 3.97 tokens per step on real prompts.
- **Needle in a haystack:** 15/15 found at 24K, 105K, 430K, 886K and 1.0M tokens, each at 10/50/90% depth. A 1.0M prompt takes ~205 s.
- **Quality:** temperature 0, small fixed samples, scored automatically.

| | GSM8K (100) | MATH-500 (50) | HumanEval (50) | MMLU-Pro (200) | C-Eval val (100) | Tool calls (10) |
|---|---:|---:|---:|---:|---:|---:|
| Thinking off | 99% | 88% | 98% | 75.0% | 93% | 10/10 |
| Thinking on | 98% | 96% | 90% | 71.9% (n=64) | 92% | |

Methodology, raw data and the eval harness:
[github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000).

## What's inside

| Path | Contents |
|---|---|
| `model-00001` … `00046-of-00048.safetensors` | [PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw), unchanged. Routed experts are EXL3 3 bpw, MUL1 codebook, uncalibrated; attention, shared experts and dense layers are DeepSeek's FP8; the DSpark drafter is MXFP4. |
| `model-00047`, `model-00048-of-00048.safetensors` | The 1017 Engram tables (layers 1 and 14) re-encoded from FP8 (203 GB) to int4 with fp16 per-32 scales (111 GB), using diffbot's `convert_engram_int4.py`. They are kept in pinned host RAM. |
| `config.json`, `model.safetensors.index.json` | diffbot's pack config (the architecture fields are identical for 1017), with the drafter in source format. The index is rebuilt from the shard headers. |
| `runtime/dsv41-exl3-overlay.tar.zst` | The runtime overlay (see below). |
| `runtime/src/` | Source of vllm-exl3 (AGPL-3.0, with the applied patch), ExLlamaV3 and the sfxnz helpers. |
| `run.sh`, `scripts/` | Launcher, bubblewrap runner, server script. |

The runtime is the official `vllm/vllm-openai` nightly-5555a2d8 image (pulled by digest at setup) plus the overlay,
which holds only the files this setup adds or changes:
- `rootfs/` changes: diffbot's six vLLM patches (patched vLLM Python files), vllm-exl3 `d3cfd394` with diffbot's
  `exl3-thor.patch`, ExLlamaV3 `5be88657` built for sm_120a, the sfxnz helpers, and `sitecustomize.py`.
- `recipe/`: diffbot's recipe with the compiled xmoe / xmoe_do MoE kernels.
- `cache/`: warm Triton / TileLang / DeepGEMM kernel caches produced by this stack.

No NVIDIA libraries are redistributed here: they come from the official image, and the driver comes from your host.

How the model fits in two 96 GB cards is diffbot's design:
- 30% of the routed experts (the coldest) stay in pinned host RAM and run on a side CUDA stream;
- the int4 Engram tables and the embedding table are read from host RAM over UVA;
- VRAM is trimmed in several places, and custom sm_120 MoE kernels are used.

Settings that differ from diffbot's defaults, found on our host:
- `NCCL_P2P_DISABLE=1`: NCCL's P2P transport hung in the first all-reduce.
- `pinned_max_round_threshold_mb:1024`: exact-size pinned allocations, which save ~41 GiB of host RAM.
- Host-expert plan `plan-blend30-dh.json`.

## With Docker instead (not tested by us)

If you have Docker with the NVIDIA container toolkit, you can use diffbot's own flow with this repo's weights:

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --revision 6062be5f0f43c50ba562ef4e7087d4fc83eea36c \
  --include "recipe/**" --local-dir diffbot-recipe
cd diffbot-recipe/recipe && bash docker/build.sh && bash kernels/build.sh
PACK=/path/to/dsv41-1017 PLAN=plan-blend30-dh.json NCCL_P2P_DISABLE=1 API_KEY=change-me bash serve/serve.sh
```

To get the same host-RAM savings, add `pinned_max_round_threshold_mb:1024` to `PYTORCH_CUDA_ALLOC_CONF` in `serve/serve.sh`.

## Limitations

- **Quality is only spot-checked**, on the small samples above. PixelML's routed experts are uncalibrated, and there is
  no KL or perplexity measurement against the original model.
- **First start** reads all weights: ~85 min from an HDD on our host (including ~23 min where the drafter re-reads the
  checkpoint). Kernel compilation adds a few minutes on the very first run.
- **Prefill ~5K tok/s**, half of what diffbot reports for the legacy model on the same GPUs. We attribute this to the
  host-memory all-reduce forced by `NCCL_P2P_DISABLE=1`. If NCCL P2P works on your machine
  (`scripts/nccl_test.py`), try `NCCL_P2P_DISABLE=0`.
- **Thinking mode** sometimes reasons past 16K tokens. Give requests a large `max_tokens`; DeepSeek recommends ≥ 256K.
- **4 concurrent streams** get only 33–40 tok/s each. The host-expert plans were chosen from legacy-model routing
  statistics.
- **Not designed for other hardware.** Cards with less VRAM need a different plan and `KV_MEM`.

## Credits

- **DeepSeek-AI:** DeepSeek-V4.1-Flash-1017 (MIT).
- **[diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000):** the whole serving
  design: vLLM sm_120 fixes, xmoe kernels, host-RAM experts, int4 Engram, scripts.
- **[PixelML](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw):** the 1017 EXL3 3 bpw quantization.
- **coolbho3k:** the V4.1 EXL3 3 bpw settings.
- **sfxnz:** the V4.1 EXL3 vLLM recipe.
- **turboderp:** ExLlamaV3.
- **vcruz305 and Mia's AI Lab:** vllm-exl3.
- **The vLLM, FlashInfer and PyTorch teams.**

Licenses of everything bundled here: [NOTICE.md](NOTICE.md).
