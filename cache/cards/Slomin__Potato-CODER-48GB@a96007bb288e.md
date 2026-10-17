---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - flash-next
  - moe
  - potato-coder
  - potato
  - coding
  - long-context
  - 262k
  - iq4_xs
  - pruned
  - reap
  - gguf
  - llama.cpp
  - 48gb
thumbnail: https://huggingface.co/Slomin/Potato-CODER-48GB/resolve/main/Potato-CODER-48GB-header.jpg
---


<!-- v1.0 card, first published 2 Oct 2026 while the last confirmation runs finish; the "being measured" lines are filled in as results land. -->

![Potato-CODER-48GB banner](./Potato-CODER-48GB-header.jpg)

## Temporarily withdrawn — 11 October 2026

I found a degradation in Potato-CODER-48GB, including repetitive reasoning loops during normal use. I am temporarily removing the model files from Hugging Face until the issue is fixed and the model has been re-tested.

The model files are unavailable. The documentation below describes the withdrawn release.

---


# Potato-CODER-48GB (v1.0)


## Read this first

**What it uses, with the serving command below, on Linux (2 × RTX 3090, headless):**

| resource | used | of |
|---|---|---|
| **VRAM, both cards together** | **peak 23,706 + 23,754 MiB** during a 241,540-token prompt with the MTP head and an image | 2 × 24,576 MiB |
| **System RAM** | about 40 GB (27 GB PLE lookup table + llama-server); exact figure being re-measured for v1.0 | 48 GB minimum, 64 GB comfortable |
| Context | 262,144 tokens, 16-bit KV cache | full |
| Extras on board | speculative-decoding head, vision | both on |
| Speed | decode **120 / 93 / 72 tok/s**, prefill 1,421 / 911 / 765 tok/s at 60k / 190k / 260k (one request, MTP on) | |

The files total about 64 GB; the GPUs hold everything that does real work, RAM holds the PLE lookup table that is cheap to keep there.

**The serving parameters below are part of the model.** They were tuned together with the weights to fill both cards. Change the
context size, cache type, batch size or tensor split and it either stops fitting or leaves speed on the table. Start with the exact
command, then adjust.

Developed by a professional software developer who uses it every day as a coding agent behind `llama-server` and an agent harness,
and measured the same way: long real sessions, tool calls, images, 262k contexts.

A coding model that fits **two 24 GB NVIDIA cards at the full 262,144-token context**, with the multi-token-prediction head and vision
still on board. It is **Qwen3.8-Flash-Next** with the routed experts pruned from 512 to **256 per layer**, keeping the experts that
coding and agentic sessions in 16 language families actually route to. Nothing was post-trained.

| | |
|---|---|
| architecture | qwen4exp (Qwen3.8-Flash-Next hybrid DeltaNet / full attention, hyper-connections, PLE n-gram table), 48 blocks |
| experts | 256 of 512 per layer, selected by routing statistics on our coding corpus (16 language families, agentic and tool use) |
| quantization | IQ4_XS routed experts (our importance matrix, collected on the unpruned model); 610 non-expert tensors at Q6_K/Q8_0 |
| size | 60.99 GiB main file (two shards) + 1.3 GiB MTP head + 0.6 GiB vision projector |
| context | 262,144 tokens on 2 × 24 GB with f16 KV cache, MTP head and vision |

## How good it is

All numbers come from one pre-registered confirmation, run on held-out tasks that were sealed before the build was chosen.
Reference: the **full model** with all 512 experts, in our own IQ4_XS.

**Languages, against the unpruned model** (20 sealed + 40 development tasks × 4 samples per family; pass rate, %).
Rule: in every family the one-sided 95 % upper bound of (full model − v1.0) must stay under 15 points. **16 of 16 pass.**

| language | full model | v1.0 |
|---|---|---|
| bash | 97.1 | 98.8 |
| c | 100.0 | 99.6 |
| cpp | 93.3 | 92.5 |
| csharp | 97.1 | 99.6 |
| dart | 97.5 | 96.7 |
| flutter (widget tests) | 95.8 | 94.6 |
| go | 97.9 | 99.6 |
| java | 96.2 | 100.0 |
| javascript | 94.2 | 96.7 |
| kotlin | 97.9 | 96.2 |
| php | 92.5 | 87.1 |
| python | 95.4 | 92.9 |
| ruby | 89.6 | 94.6 |
| rust | 97.5 | 95.4 |
| swift | 85.8 | 94.2 |
| typescript | 92.9 | 95.0 |

PHP is the one family more than 5 points below the full model (5.4; upper bound 12.5).

**Verdict: the confirmation passed (3 Oct 2026).**

**Against NVIDIA's official 4-bit release of the full model.** Half the experts, two gaming cards — and it holds its own against the
full 512-expert model in NVIDIA's own NVFP4 format (`nvidia/Qwen3.8-Flash-Next-NVFP4`, run on a 128 GB DGX Spark), on our development-screen sets (the same
items, prompts and graders for both):

| | **v1.0** — 256 experts, IQ4_XS, 2 × RTX 3090 | NVIDIA NVFP4 — 512 experts, DGX Spark |
|---|---|---|
| LiveCodeBench (144) | **125** | 118 |
| 16 programming languages incl. Dart and Flutter (640 tasks) | 616 | 621 |
| 8 human languages (160) | 157 | 159 |
| tool-call suite (18) | 17 | 18 |
| HumanEval+ / MBPP+ (264) | 229 | 238 |
| BFCL v3 (200) | 148 | 164 |

- **LiveCodeBench: +7 problems** for v1.0 — on competitive coding the pruned model solved more than the full one.
- **Languages: within 1 %** of the full model across all 16 programming languages (616 vs 621 of 640) with half of its experts.
- **Fits 48 GB** at the full 262k context with MTP and vision, at 120 tok/s decode, on two consumer cards.

Descriptive, not a pre-registered gate: different engines (llama.cpp vs vLLM), quantisations (IQ4_XS vs NVFP4) and machines; on vLLM
the 16k thinking budget is not applied (thinking runs to the 32k cap) and 6 requests ran in parallel instead of 4; one run per benchmark.


## Files

`IQ4_XS/Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf` + `-00002-of-00002.gguf` (the pruned model, two shards; point llama.cpp at the
first), `MTP/mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf` (3-bit MTP draft head), `mmproj-Potato-CODER-48GB-Q8_0.gguf` (vision
projector), `serve-potato.bat` (Windows start script, see below), `LICENSE`, `NOTICE`, `SHA256SUMS`.

## Running it (llama.cpp)

**Use our llama.cpp branch: [github.com/potato-os/llama.cpp](https://github.com/potato-os/llama.cpp), release potato-v1.1.0 or
later.** It is stock llama.cpp `master` plus the two additions this model needs. Nothing else differs, and the branch is kept rebased
on upstream.

| addition | what you get |
|---|---|
| MTP for Qwen3.8-Flash-Next: upstream native qwen4exp MTP (#29761) + Potato patches | the bundled `MTP/` draft head works: about 25 % faster generation, same output quality |
| `--mmproj-swap-draft` (ours) | vision **and** the MTP head **and** the full 262k context on two 24 GB cards. The projector stays off the GPU; when an image arrives, the server lends the draft head's VRAM to a temporary GPU projector, encodes, and hands it back. Measured on 2× RTX 3090: about 0.5 s of swap overhead per image on top of the ~1 s encode, nothing on text turns. |

### 1. Get the binaries

Download from the [releases page](https://github.com/potato-os/llama.cpp/releases) and unpack; no install step.

| your system | binaries | CUDA runtime (unpack into the same folder) |
|---|---|---|
| Linux x64, NVIDIA driver 570+ | `llama-potato-bin-linux-cuda-12.8-x64.tar.gz` | `cudart-llama-potato-bin-linux-cuda-12.8-x64.tar.gz`, only if no CUDA 12 runtime is installed |
| Linux arm64, NVIDIA DGX Spark (GB10) | `llama-potato-bin-linux-cuda-13.0-arm64.tar.gz` | `cudart-llama-potato-bin-linux-cuda-13.0-arm64.tar.gz`, only if no CUDA 13 runtime is installed |
| Windows x64, RTX 20xx to 40xx, driver 551+ | `llama-potato-bin-win-cuda-12.4-x64.zip` | `cudart-llama-potato-bin-win-cuda-12.4-x64.zip` |
| Windows x64, RTX 50xx, driver 580+ | `llama-potato-bin-win-cuda-13.4-x64.zip` | `cudart-llama-potato-bin-win-cuda-13.4-x64.zip` |

The `cudart-` archives hold NVIDIA's CUDA runtime libraries (cuBLAS and friends). If one is missing from a release, the `cudart-llama-bin-...` archive of the same CUDA version from the [upstream llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) contains the same files. Building from source is described in `POTATO.md` in the repository.

### 2. Serve

Put the two model shards, the `mmproj` file and the `MTP` head in one folder. The layout below is the one the model was validated with: two 24 GB cards, dense layers / KV cache / output head on the first card, experts spread over both, f16 KV cache, head and vision on.

**Linux**

```bash
./llama-server -m IQ4_XS/Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf \
  -c 262144 -ngl 99 -ts 1.1,1 -ub 512 -b 4096 -fa on -ctk f16 -ctv f16 \
  --mmproj mmproj-Potato-CODER-48GB-Q8_0.gguf --no-mmproj-offload --mmproj-swap-draft --image-max-tokens 2048 \
  --spec-type draft-mtp -md MTP/mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf --spec-draft-n-max 3 -devd CUDA1 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --chat-template-kwargs '{"preserve_thinking": false, "reasoning_effort": "medium"}' \
  --reasoning-budget 16384 \
  --reasoning-budget-message $'\n\nThinking budget reached. Stop deliberating now: state the approach in one or two sentences and write the complete final code.\n'
```

**DGX Spark (arm64, one GB10, 128 GB)**: the Linux command without `-ts 1.1,1` and `-devd CUDA1`. Decode 53 / 42 / 38 tok/s
at 60k / 190k / 260k.

**Windows, step by step.** The script `serve-potato.bat` is part of this repository.

1. Download this repository into one folder and keep its layout (the model files stay in their subfolders).
2. From the [releases page](https://github.com/potato-os/llama.cpp/releases), download `llama-potato-bin-win-cuda-12.4-x64.zip` (RTX 20xx to 40xx) or `llama-potato-bin-win-cuda-13.4-x64.zip` (RTX 50xx), **and** the matching `cudart-` zip. Unzip both **into the same folder**, next to `serve-potato.bat`. Stock llama.cpp builds do not work with this script: they lack `--mmproj-swap-draft`.
3. Double-click `serve-potato.bat`. It starts a **local server on this PC** in a console window; the model is ready when the window shows `listening on http://127.0.0.1:8080`. Keep the window open while you work; close it or press Ctrl+C to stop the server.
4. In your agent (OpenCode, Cline, ...) add an OpenAI-compatible provider with the base URL `http://127.0.0.1:8080/v1`. After each request the server window prints its speed (prompt and generation tokens per second).

The folder then looks like this:

```text
Potato-CODER-48GB\
├── serve-potato.bat
├── IQ4_XS\Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf
├── IQ4_XS\Potato-CODER-48GB-IQ4_XS-00002-of-00002.gguf
├── MTP\mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf
├── mmproj-Potato-CODER-48GB-Q8_0.gguf
└── llama-server.exe, ggml-cuda.dll, cudart64_12.dll, ...   (from the two zips)
```

If a file is missing, the script says which one and waits for a key instead of closing the window. To keep the model somewhere else, change the `MODELS` line at the top of the script.

Two things matter on Windows:
- **Turn off the driver's RAM spill-over.** NVIDIA Control Panel → Manage 3D settings → Program Settings → add `llama-server.exe` → *CUDA - Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**. With the default setting a full card quietly spills into system RAM and generation drops to a few tokens per second instead of reporting an error.
- **Context size.** Windows keeps part of each card for the desktop, so the script starts at `-c 212992` (an estimate, not a measurement). If the server starts cleanly and `nvidia-smi` shows headroom, raise it towards `262144`; if it fails to allocate at start, lower it in steps of 16384 (about 1 GB across the two cards per step).

Start the server with the script rather than typing the command into PowerShell: older PowerShell versions mangle the quotes of the JSON argument.

The server listens on `http://127.0.0.1:8080` with an OpenAI-compatible API; point your agent harness at it. `reasoning_effort` accepts `low`, `medium`, `xhigh`; `preserve_thinking: false` is the intended agent setting.

**Thinking budget (both commands above; recommended for agent use).** Each thinking block is capped at 16,384 tokens; when the cap is reached, the server adds one wrap-up sentence and the model moves on to its answer or tool call. Every v1.0 measurement above was taken with this budget. For unbounded thinking, delete the two budget lines (Linux) or the `--reasoning-budget` line (Windows); clients can override the budget per request. On Windows the script hands the message to llama.cpp through the `LLAMA_ARG_THINK_BUDGET_MESSAGE` environment variable, with its line breaks intact.

### Without our branch (stock llama.cpp)

The model loads and serves text and vision, without the head: drop `--spec-type`, `-md`, `--spec-draft-n-max`, `-devd` and `--mmproj-swap-draft`, and expect roughly 25 % lower decode speed. Vision at 262k then needs the projector on the CPU (`--no-mmproj-offload` alone, slow images) or a smaller context (about 68 KiB saved per token). Upstream's native MTP does not load this shared head yet.

**Validation status:** potato-v1.1.0 passes the full deploy check (14 gates: the 262k fit, MTP drafting, the image swap, tool calls) on 2× RTX 3090 and on DGX Spark. The Windows binaries and `serve-potato.bat` are tested automatically on every build, but the model has not yet been run on a two-card Windows machine; reports are welcome.

## How it was made

1. Source: Qwen's official Qwen3.8-Flash-Next safetensors, converted to a BF16 GGUF.
2. Corpus: coding, agentic and tool-use material in 16 language families (C, C++, C#, Go, Java, JavaScript, TypeScript, Python,
   Rust, Kotlin, Swift, Ruby, PHP, Bash, Dart, Flutter), with a per-language token count checked before selection and
   decontaminated against every benchmark used here.
3. Selection: routing statistics per expert over that corpus, collected with the full model; the top 256 experts per layer are kept,
   and the expert tensors, router rows and biases sliced accordingly. Two selections were built and screened; the one without
   per-language quotas won.
4. Quantization: IQ4_XS with an importance matrix collected on the unpruned model over the same corpus (keyed by original expert
   IDs, so pruning does not shift it); then the non-expert tensors whose quantization error mattered most were raised to Q6_K/Q8_0
   until a 0.80 GiB budget was used (610 tensors).
5. Confirmation: one pre-registered run on sealed tasks (languages, LiveCodeBench, Aider polyglot, BFCL, tools, long context,
   the 262k fit, a deploy check and a memory stress test), with every gate stated before any result.

## Limits

- Behaviour on safety-sensitive prompts was not measured here and nothing was done to change it in either direction; the weights are
  Qwen's official ones.
- Pruned experts are gone: prompts far from coding and agentic work (poetry, rare natural languages) may suffer more than the coding
  benchmarks show.
- "Fits 48 GB" was measured on two RTX 3090s with the layout above; other cards and drivers differ by a few hundred MiB.

## Licence and attribution

Qwen3.8-Flash-Next is released under the **Qwen Community License 1.0** (see `LICENSE`), which this derivative inherits. In short: free
to use, modify and redistribute with the notice kept; a separate licence from Qwen is required for commercial Model-as-a-Service or
AI-coding-assistant businesses, and products above 100M monthly active users or US$20M monthly revenue must display the model name.
Credits: Qwen (base model), unsloth (MTP head and projector conversions), Slomin (corpus, expert selection, importance matrix,
quantization, packaging). See `NOTICE`.
