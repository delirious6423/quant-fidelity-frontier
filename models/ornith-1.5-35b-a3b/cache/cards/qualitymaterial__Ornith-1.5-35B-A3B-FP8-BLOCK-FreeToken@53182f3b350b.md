---
library_name: transformers
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- freetoken
- fp8
- moe
- qwen3_5_moe
- windows
---

# Ornith 1.5 35B-A3B FP8-BLOCK for FreeToken

This is a community conversion of [Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), built to run local chat and coding tools on a Windows desktop with a 16 GB GPU and 64 GB RAM. It uses FP8 E4M3FN weights with 128 Ã— 128 blocks, in the layout supported by the tested FreeToken MoE loader.

The goal was practical: keep a higher-precision quantization, retain the model's existing experts, and use system RAM to run a model larger than the GPU's memory. Inference runs locally, without a hosted API charging per token. Electricity, hardware, storage and any external tools still have their own costs.

This is an unofficial conversion. The original model authors retain their attribution and MIT license. No fine-tuning or reinforcement learning was performed for this checkpoint.

## Why this conversion exists

The setup started with an Ornith 1.0 GGUF model and a request to use it in FreeToken Desktop. That particular installed FreeToken build supported a limited GGUF architecture set and did not support this Ornith architecture through its GGUF loader. Changing the file extension or choosing a larger GGUF quant would not fix that loader mismatch.

A compatible 1.0 FP8 block checkpoint established that the FreeToken native MoE route worked on the machine. The setup then moved to 1.5 after reviewing the newer release and its training approach. The 1.0 files and local profile were preserved as a fallback. This history explains the choice; it is not a controlled comparison of 1.0 and 1.5 quality.

The official 1.5 FP8 checkpoint used per-channel quantization, which did not match the native FP8 expert kernels in the installed FreeToken version. We therefore converted the official BF16 weights directly into the supported block format. This avoids trying to reconstruct higher precision from an already lower-bit checkpoint.

| Choice | Reason in this setup |
| --- | --- |
| Ornith 1.5 35B-A3B | The requested newer agentic model, with an existing MoE architecture |
| FP8 block weights | Higher bit depth than a 4-bit format, with working native loader support |
| RAM offload | The complete checkpoint exceeds 16 GB VRAM |
| Text-only serving | Concentrates the available memory on chat and coding |
| 98,304-token conversation limit | Reduces the serving memory requirement relative to the advertised upstream context |
| Local tool API | Lets a coding client use shell commands through the model's structured tool calls |

FP8 was selected for compatibility and the requested precision preference. We have not established that this conversion outperforms a particular 4-bit quantization on downstream tasks.

## Tested machine and software

| Component | Tested configuration |
| --- | --- |
| Operating system | Windows |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM |
| System memory | 64 GB installed; approximately 61.6 GiB usable |
| FreeToken engine | `0.1.3+gfa0e9537d` |
| FreeToken Desktop | `0.2.0-beta.23` |
| Conversion runtime | Python 3.12, PyTorch and safetensors in the installed FreeToken environment |
| Coding client checked | Codex CLI `0.160.0` |

The 16 output weight shards total **36,614,026,256 bytes**, approximately **36.61 GB / 34.10 GiB**. This is the weight-file size, not the total runtime memory requirement. The source BF16 weight shards total approximately 71.90 GB.

Observed system use while serving was approximately 10â€“11 GiB GPU memory and 50â€“51 GiB RAM. These readings include other processes and vary with cache allocation and workload. They are observations from this machine, not minimum requirements or a benchmark.

## How the experts fit

The model already has 256 routed experts per decoder layer and selects eight per token. This conversion keeps the 40 main decoder layers and their expert banks. It does not create four new experts or reduce the architecture to four experts.

FreeToken's offload strategy holds expert weights in system RAM and caches a portion on the GPU. In the tested configuration, the expert banks needed about 30.0 GiB of pinned host memory. Setting `FREETOKEN_PIN_BUDGET_GB=30.5` allowed those banks to load; the default Windows budget on this machine was about 24.6 GiB and was insufficient.

The tested FP8 expert path had no CPU executor fallback. Offload here means storing expert weights in RAM and transferring them for GPU execution, rather than treating the CPU as a substitute FP8 inference device.

Four planner/coder/researcher/reviewer roles are client instructions that can share this one serving model. They are separate from the model's learned experts. The server allows up to four concurrent requests; the prepared local coding profile allows up to three agent threads. Those limits were configured, but sustained concurrent agent workloads were not evaluated.

## Conversion details and provenance

The source is pinned to revision [`10fbf86fed7ecee4a061f8b499a618f46001cac1`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1). All 16 source shard SHA-256 hashes were checked before conversion.

Each quantized matrix uses symmetric absmax scaling per 128 Ã— 128 block. The scale is `max(abs(block)) / 448`; an all-zero block uses scale 1. Values are represented as `float8_e4m3fn`. Scales are stored as F32, and the installed native expert loader casts them to BF16 for execution.

Stacked BF16 expert tensors were split into individual expert `gate_proj`, `up_proj` and `down_proj` weights. Embeddings, the output head, normalization, routers, convolution, vision weights and the small linear-attention `a`/`b` projections retain their source precision according to the included quantization configuration.

The optional `mtp.*` speculative prediction head is omitted, and `text_config.mtp_num_hidden_layers` is set to zero. The main decoder is retained. Multimodal configuration and some source assets remain in the checkpoint, but the tested serving recipe explicitly disables multimodal execution.

The repository includes:

- [Conversion source](conversion_source.py), [pinned source metadata](ornith15-base-metadata.json) and [block configuration](ornith15-block-config.json).
- [Quantization manifest](quantization-manifest.json) with source/output hashes and sampled numeric errors.
- [Structural and numeric verification](conversion-verification.json).
- [Local inference and integration verification](local-verification.json).
- [Original upstream model card](README-upstream.md), [MIT license](LICENSE) and [license provenance](license-provenance.json).

The conversion source accepts `--source` and `--output`. Its companion metadata/config files must remain beside the script. Conversion needs the pinned BF16 download, PyTorch with the specified float8 dtype, safetensors, and sufficient disk space for both input and output. The default paths are examples and can be overridden. Re-running conversion can copy the upstream README into the output; retain this community card separately when reproducing it.

## Download and serve

Install and configure FreeToken separately. This repository contains model weights and supporting documentation, rather than the engine itself. The version above is the one verified; other runtime versions may handle formats or flags differently.

```powershell
hf download qualitymaterial/Ornith-1.5-35B-A3B-FP8-BLOCK-FreeToken `
  --local-dir C:\Models\Ornith15-FP8
```

Confirm that all 16 weight shards and `model.safetensors.index.json` are present. Keep all the configuration, tokenizer and chat-template files from the repository.

The following direct-engine recipe uses the same model and serving arguments as the tested daemon launch. The tested integration itself used FreeToken's daemon-managed engine. Run this recipe only when the existing engine is stopped and port 1919 is available; do not launch a second copy beside a loaded 35B model.

```powershell
$ft = Join-Path $env:LOCALAPPDATA 'FreeToken\bin\freetoken.exe'
$previousBudget = $env:FREETOKEN_PIN_BUDGET_GB
try {
  $env:FREETOKEN_PIN_BUDGET_GB = '30.5'
  & $ft serve --model-path C:\Models\Ornith15-FP8 `
    --host 127.0.0.1 --port 1919 `
    --served-model-name Ornith-1.5-35B-A3B-FP8-BLOCK `
    --moe-strategy offload --moe-cache-size 512 `
    --num-tokens 131072 --max-seq-len-override 98304 `
    --max-running-requests 4 --memory-ratio 0.90 `
    --text-model-only --enable-cache-report `
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 `
    --cors-origins tauri://localhost,http://tauri.localhost,http://localhost:1420
} finally {
  $env:FREETOKEN_PIN_BUDGET_GB = $previousBudget
}
```

This environment setting is scoped to the shell and its child process. It must be present when the engine or supervising daemon starts. Changing it in a different terminal does not change an already-running daemon's environment. The 30.5 GiB setting is specific to the tested 64 GB machine; leave room for the operating system, non-expert weights and other programs.

Wait for actual readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:1919/health
```

Proceed when `status` is `ok` and the model name matches. `/v1/models` can respond before the weights finish loading.

For FreeToken Desktop, import the checkpoint folder from the Models page. The tested local setup linked the existing folder rather than duplicating the weight files, started the engine through its daemon at `127.0.0.1:19000`, and used the inference API at `127.0.0.1:1919/v1`. If an imported model is absent from the cached library, reloading the Models page can trigger a rescan. Once the engine became ready, Chat selected 1.5 and displayed a green running indicator.

A daemon restart on Windows must wait for the old process to exit and release its PID-file lock. The HTTP listener can close first, especially while Desktop log streams remain attached. The tested local launcher was corrected to wait for that shutdown rather than immediately launching a competing daemon.

## API and coding clients

Chat Completions, automatic function calls, tool-result round trips and the Responses API passed local smoke tests. Configure compatible clients with:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:1919/v1` |
| Model | `Ornith-1.5-35B-A3B-FP8-BLOCK` |
| Endpoint family checked | Chat Completions and Responses |
| Local API authentication | No OpenAI account authentication required by the tested provider |
| Conversation limit in this recipe | 98,304 tokens |
| Shared token-cache pool | 131,072 tokens |

The prepared Codex profile used the Responses wire API, disabled WebSockets, set a 48,000-token automatic compaction threshold, and left the user's default hosted-model profile intact. A model catalog was provided so the CLI recognized the local model and its context limit. This describes a verified CLI integration, not a claim that every desktop coding client supports an arbitrary custom provider.

The model only proposes tool calls. The client executes them under its own permissions and returns the results. A working tool parser is therefore part of the integration, not just output formatting.

## What passed, and what remains unmeasured

| Check | Result |
| --- | --- |
| Main checkpoint layout | 62,636 tensors checked against the working native layout |
| Block scales | All 30,970 scale tensors positive and finite |
| Numeric reconstruction samples | 256 samples; mean relative L2 2.6458%, maximum 2.6546% |
| Basic chat | Returned the expected arithmetic answer |
| Automatic tool call | Produced the expected function and arguments |
| Tool-result round trip | Returned the expected supplied marker |
| Responses API | Returned the expected answer |
| Real Codex CLI file task | Read a 45-byte marker, wrote an exact copy, verified matching bytes/hashes |
| Installed local profile | Selected the intended provider/model and returned the expected answer |
| Saved launcher restart | Reloaded the model and passed a fresh inference request |
| FreeToken Desktop | 1.5 selected in Chat with a green running indicator |

The file task included one failed PowerShell syntax attempt that the model corrected before completing the task. Five shell commands succeeded. That is evidence of a functioning tool loop, not perfect command generation.

The relative L2 figures measure sampled weight reconstruction before execution-time scale casting. They are not accuracy loss percentages. No SWE-bench, Terminal-Bench, throughput comparison, full 96K conversation, vision evaluation or sustained multi-agent stress test was run for this conversion. Upstream benchmark scores belong to the authors' evaluated checkpoints and harnesses, not automatically to this FP8 conversion.

The initial setup used a 64K conversation limit. On October 6, 2026, the server and local client profiles were raised to **96K (98,304 tokens)** while retaining the 131,072-token shared cache pool. A synthetic request with **81,834 input tokens** returned the expected short answer. See [96K capacity verification](context-96k-verification.json). This checks capacity and basic inference, not long-range retrieval accuracy or performance at a full window. Input and output share the conversation limit; multiple requests also share the token-cache pool.

## Sampling and self-improvement

The [official serving guidance](https://github.com/ornith-ai/Ornith-1#quickstart) recommends temperature 1.0 for general use and 0.6 for precise coding, with `top_p=0.95` and `top_k=20`. Presence penalty is 1.5 for general tasks and 0 for coding. Clients choose these values; the zero-temperature arithmetic smoke checks were verification fixtures.

Ornith 1.5's [training approach](https://github.com/ornith-ai/Ornith-1) generates tasks, scaffolds and solution rollouts for reinforcement learning. Loading this checkpoint does not start that training process or update weights from chat history. A client can separately run a generate â†’ test â†’ repair loop to improve a solution. New weight training requires a separate training workflow.

The upstream advertised context is 262,144 tokens. This setup serves a smaller window to fit the desktop's memory budget. Source configuration or a library badge may still display the upstream limit; use the actual server configuration when setting client context limits.

## Troubleshooting

- **Pinned-memory budget error:** ensure the 30.5 GiB setting reached the newly started daemon/engine. Confirm adequate free system RAM.
- **Insufficient VRAM:** unload other models and GPU-heavy workloads first. The checkpoint is not intended to fit entirely in 16 GB VRAM.
- **Another daemon holds the PID lock:** wait for the previous daemon to finish shutting down. Do not assume a closed HTTP port means the process has exited.
- **Model listed but not usable:** check `/health`, complete shard download and engine logs.
- **Raw tool-call text instead of structured calls:** check the `qwen3_coder` tool parser and the client's supported tool schema.
- **Image controls or 256K badge appear:** the tested recipe is text-only and capped at 96K regardless of inherited metadata.

## Attribution

Original weights: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). Original project: [ornith-ai/Ornith-1](https://github.com/ornith-ai/Ornith-1). Community conversion and local integration: `qualitymaterial`.

The unchanged MIT license was obtained from the official project repository because the pinned source model files referenced a license file without including it. The exact pinned license URL is recorded in [license-provenance.json](license-provenance.json). See [README-upstream.md](README-upstream.md) for the original authors' full model card.
