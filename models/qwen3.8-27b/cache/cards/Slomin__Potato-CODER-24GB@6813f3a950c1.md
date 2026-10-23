---
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model:
  - ukisai/Swift-Qwen3.8-27b
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - 27b
  - potato-coder
  - potato
  - coding
  - long-context
  - 262k
  - iq4_xs
  - gguf
  - llama.cpp
  - 24gb
  - amd
  - vulkan
thumbnail: https://huggingface.co/Slomin/Potato-CODER-24GB/resolve/main/Potato-CODER-24GB-header.png
---

![Potato-CODER-24GB banner](./Potato-CODER-24GB-header.png)

# Potato-CODER-24GB (v1.0-rc1)

A 27B coding model that runs on **one 24 GB card at the full 262,144-token context** (NVIDIA with CUDA, or AMD Radeon with Vulkan), with multi-token-prediction speculation
and vision still on. It is [Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) (Qwen3.8-27B post-trained by UkisAI for
shorter reasoning), requantized with a coding-calibrated importance matrix. Nothing was trained here: this is a quantization and a
deployment recipe, measured end to end.

| | |
|---|---|
| architecture | qwen35 hybrid (DeltaNet + full attention every 4th layer), 64 blocks + MTP block, 262,144 context |
| quantization | recipe "B3": `output.weight` Q5_K, every other quantized tensor IQ4_XS, coding imatrix incl. the output head; 4.31 bits/weight |
| size | 13.7 GiB model (MTP head inside) + 0.6 GiB Q8_0 vision projector |
| VRAM | peak 23,956 MiB on an RTX 3090 at a 261,808-token turn plus an 8 MP image (169 MiB headroom) |
| speed (one RTX 3090, MTP on) | decode 86 tok/s short, 64 / 55 at 33k / 75k (thinking), 43 at 190k, 29 at 262k; prefill 990 tok/s at 33k, 470 at 262k |
| speed (one RX 7900 XTX, Vulkan, MTP on) | decode 96 tok/s short, 58 / 50 at 33k / 75k (thinking), 39 at 190k, 24 at 262k; prefill 746 tok/s at 33k, 371 at 262k; peak VRAM 22,664 MiB at 262k (1.9 GiB headroom) |
| coding | LiveCodeBench-80 (medium effort, 16k cap): 63/80 (hard tier 8/20); a stock-recipe IQ4_XS of the same family with q8 cache scores 60/80, which is within the ±5-point noise of this benchmark; executable tasks after real 190k / 250k sessions: 8/8 and 6/8 (f16-cache control on two cards: 8/8, 7/8) |

## Files

`IQ4_XS/Potato-CODER-24GB-IQ4_XS.gguf` (the model, MTP head embedded), `mmproj-Potato-CODER-24GB-Q8_0.gguf` (vision projector, Q8_0 of
UkisAI's F16 projector; the vision tower is unchanged from Qwen3.8-27B), `serve-potato.bat` (Windows start script, NVIDIA, see below), `serve-potato-amd.bat` (Windows start script, AMD Radeon), `LICENSE` (Swift Open License), `LICENSE-APACHE-2.0` (base model), `NOTICE`, `SHA256SUMS`.

## Running it (llama.cpp)

**Use our llama.cpp branch: [github.com/potato-os/llama.cpp](https://github.com/potato-os/llama.cpp).** It is stock llama.cpp `master` plus what this model needs to fill one 24 GB card completely. Nothing else differs, and the branch is kept rebased on upstream.

| addition | what you get |
|---|---|
| `--mmproj-swap-draft` (ours) | vision **and** the MTP head **and** the full 262k context on one 24 GB card. The projector stays off the GPU; when an image arrives, the server lends the draft context's VRAM to a temporary GPU projector, encodes, and hands it back. About half a second of swap overhead per image, nothing on text turns. |
| q8_0 / q5_0 flash-attention kernels compiled in | full decode speed with the 8-bit-key / 5-bit-value KV cache this model is served with; stock binaries run that pair, but slower |

The model's MTP head is embedded in the GGUF and already supported by stock llama.cpp.

### 1. Get the binaries

Download from the [releases page](https://github.com/potato-os/llama.cpp/releases) and unpack; no install step.

| your system | binaries | CUDA runtime (unpack into the same folder) |
|---|---|---|
| Linux x64, NVIDIA driver 570+ | `llama-potato-bin-linux-cuda-12.8-x64.tar.gz` | `cudart-llama-potato-bin-linux-cuda-12.8-x64.tar.gz`, only if no CUDA 12 runtime is installed |
| Windows x64, RTX 20xx to 40xx, driver 551+ | `llama-potato-bin-win-cuda-12.4-x64.zip` | `cudart-llama-potato-bin-win-cuda-12.4-x64.zip` |
| Windows x64, RTX 50xx, driver 580+ | `llama-potato-bin-win-cuda-13.4-x64.zip` | `cudart-llama-potato-bin-win-cuda-13.4-x64.zip` |

The `cudart-` archives hold NVIDIA's CUDA runtime libraries. If one is missing from a release, the `cudart-llama-bin-...` archive of the same CUDA version from the [upstream llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) contains the same files. Building from source is described in `POTATO.md` in the repository.

### 2. Serve

The full context fits one 24 GB card only with **all** of these: 8-bit keys and 5-bit values in the KV cache, MTP draft depth 3, micro-batch 256, the projector at Q8_0 and swapped in per image.

**Linux**

```bash
./llama-server -m IQ4_XS/Potato-CODER-24GB-IQ4_XS.gguf \
  -c 262144 -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -ub 256 -b 2048 --no-context-shift \
  --mmproj mmproj-Potato-CODER-24GB-Q8_0.gguf --no-mmproj-offload --mmproj-swap-draft \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --chat-template-kwargs '{"preserve_thinking": false, "reasoning_effort": "medium"}'
```

**Windows, step by step.** The script `serve-potato.bat` is part of this repository.

1. Download this repository into one folder and keep its layout (the model files stay in their subfolders).
2. From the [releases page](https://github.com/potato-os/llama.cpp/releases), download `llama-potato-bin-win-cuda-12.4-x64.zip` (RTX 20xx to 40xx) or `llama-potato-bin-win-cuda-13.4-x64.zip` (RTX 50xx), **and** the matching `cudart-` zip. Unzip both **into the same folder**, next to `serve-potato.bat`. Stock llama.cpp builds do not work with this script: they lack `--mmproj-swap-draft`.
3. Double-click `serve-potato.bat`. It starts a **local server on this PC** in a console window; the model is ready when the window shows `listening on http://127.0.0.1:8080`. Keep the window open while you work; close it or press Ctrl+C to stop the server.
4. In your agent (OpenCode, Cline, ...) add an OpenAI-compatible provider with the base URL `http://127.0.0.1:8080/v1`. After each request the server window prints its speed (prompt and generation tokens per second).

The folder then looks like this:

```text
Potato-CODER-24GB\
├── serve-potato.bat
├── IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf
├── mmproj-Potato-CODER-24GB-Q8_0.gguf
└── llama-server.exe, ggml-cuda.dll, cudart64_12.dll, ...   (from the two zips)
```

If a file is missing, the script says which one and waits for a key instead of closing the window. To keep the model somewhere else, change the `MODELS` line at the top of the script.

Two things matter on Windows:
- **Turn off the driver's RAM spill-over.** NVIDIA Control Panel → Manage 3D settings → Program Settings → add `llama-server.exe` → *CUDA - Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**. With the default setting a full card quietly spills into system RAM and generation drops to a few tokens per second instead of reporting an error.
- **Context size.** Windows keeps part of the card for the desktop, so the script starts at `-c 196608` (an estimate, not a measurement). If the server starts cleanly and `nvidia-smi` shows headroom, raise it towards `262144`; if it fails to allocate at start, lower it in steps of 16384 (about 600 MiB per step).

Start the server with the script rather than typing the command into PowerShell: older PowerShell versions mangle the quotes of the JSON argument.

<details><summary>The script, for reference (<code>serve-potato.bat</code>)</summary>

```bat
@echo off
rem ============================================================================================
rem  Potato-CODER-24GB: starts a LOCAL llama.cpp server on this PC.
rem   - It is ready when this window shows "listening on http://127.0.0.1:8080".
rem   - Point your agent (OpenCode, Cline, ...) at http://127.0.0.1:8080/v1 (OpenAI-compatible API).
rem   - Keep this window open while you work. Ctrl+C or closing the window stops the server.
rem   - After each request this window prints its speed (prompt and generation tokens per second).
rem  Put everything in ONE folder, next to this file (no edits needed):
rem     serve-potato.bat                          (this file)
rem     IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf        (the model, keep the IQ4_XS subfolder)
rem     mmproj-Potato-CODER-24GB-Q8_0.gguf          (vision projector)
rem     llama-server.exe and its DLLs               (unzip llama-potato-bin-win-cuda-*.zip + cudart-*.zip here)
rem  Stock llama.cpp does not work with this script (no --mmproj-swap-draft); use github.com/potato-os/llama.cpp
rem  To keep the model somewhere else, change the MODELS line below (no trailing backslash).
rem ============================================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "MODELS=%HERE%"
set "LLAMA=%HERE%\llama-server.exe"

if not exist "%LLAMA%" goto :no_llama
set "F=%MODELS%\IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\mmproj-Potato-CODER-24GB-Q8_0.gguf"
if not exist "%F%" goto :no_model

echo Starting Potato-CODER-24GB on http://127.0.0.1:8080 ... keep this window open.
"%LLAMA%" -m "%MODELS%\IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf" ^
  -c 196608 -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -ub 256 -b 2048 --no-context-shift ^
  --mmproj "%MODELS%\mmproj-Potato-CODER-24GB-Q8_0.gguf" --no-mmproj-offload --mmproj-swap-draft ^
  --spec-type draft-mtp --spec-draft-n-max 3 ^
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --chat-template-kwargs "{\"preserve_thinking\": false, \"reasoning_effort\": \"medium\"}"

echo.
echo The server has stopped. If you did not press Ctrl+C, read the messages above.
pause
exit /b 0

:no_llama
echo.
echo ERROR: llama-server.exe not found in %HERE%
echo Unzip the Windows llama.cpp release AND its cudart- zip into this folder, then run this file again.
echo Download both from https://github.com/potato-os/llama.cpp/releases
echo.
pause
exit /b 1

:no_model
echo.
echo ERROR: model file not found: %F%
echo Keep the files and subfolders exactly as downloaded from Hugging Face.
echo.
pause
exit /b 1
```

</details>

The server listens on `http://127.0.0.1:8080` with an OpenAI-compatible API; point your agent harness at it.

Notes:
- The 262k layout leaves under 200 MiB free: the card must be otherwise empty (headless). With a desktop on the same GPU, lower `-c` (about 38 MiB per 1,000 tokens).
- **KV cache:** keys stay at q8_0. Values at q5_0 cost 1.3-1.4x the q8 divergence at 100-150k tokens and no measurable loss on
  executable coding at 250k. With `-ctv q8_0` the one-card limit with the MTP head is about 212k.
- `reasoning_effort` accepts `low`, `medium`, `xhigh`; `preserve_thinking: false` is the intended agent setting.

### Without our branch (stock llama.cpp)

The model loads, and text plus the MTP head work; drop `--mmproj-swap-draft`. Decode with the q8_0 / q5_0 cache is slower unless you build with `-DGGML_CUDA_FA_QUANTS="...;q8_0-q5_0"`. Vision at 262k then needs the projector on the CPU (`--no-mmproj-offload` alone, slow images), a smaller context, or text-only serving.

**Validation status:** the CI-built Linux binary of this branch was validated end to end on one RTX 3090 (context filled to 261,780 of 262,144 tokens, retrieval at 60k / 190k / 250k, MTP, seven image turns all encoded on the GPU). The Windows binaries and `serve-potato.bat` are tested automatically on every build (missing binaries, missing model files, and a full start with every flag accepted). A first user has run the model on Windows with this branch; Windows speed has not been measured by us yet, and reports are welcome.

## Running it on AMD Radeon (Vulkan)

Use the Vulkan build of our branch: release [`potato-amd-v1.0.0`](https://github.com/potato-os/llama.cpp/releases/tag/potato-amd-v1.0.0)
(branch `potato-xtx`). Besides `--mmproj-swap-draft` it has a faster flash-attention path for RDNA3 cards, which makes
prompt processing with this model's q8_0/q5_0 cache 1.4-1.8x faster at 32k-250k context. It was validated on one RX 7900 XTX
(Linux, Mesa RADV 25.2) with the same checks as the RTX 3090; the model files are the same.

| your system | binaries |
|---|---|
| Linux x64, Mesa RADV | `llama-potato-bin-linux-vulkan-x64.tar.gz` |
| Windows x64, AMD Adrenalin driver | `llama-potato-bin-win-vulkan-x64.zip` (no runtime archive needed) |

**Turn on Resizable BAR** in the BIOS ("Above 4G Decoding" and "Re-Size BAR Support"; AMD calls it Smart Access Memory).
Without it, generation runs at about half speed (20.8 instead of 44.2 tok/s on an RX 7900 XTX, without MTP). If you
cannot turn it on, set `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` before starting the server (42.9 tok/s); `serve-potato-amd.bat`
always sets it (about 3 % slower when Resizable BAR is on).

**Linux**

```bash
./llama-server -m IQ4_XS/Potato-CODER-24GB-IQ4_XS.gguf \
  -c 262144 -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -ub 256 -b 2048 --no-context-shift \
  --mmproj mmproj-Potato-CODER-24GB-Q8_0.gguf --no-mmproj-offload --mmproj-swap-draft \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --chat-template-kwargs '{"preserve_thinking": false, "reasoning_effort": "medium"}'
```

**Windows:** the same steps as for NVIDIA above, with `llama-potato-bin-win-vulkan-x64.zip` (no `cudart-` zip) and
`serve-potato-amd.bat`. It starts at `-c 196608` like the NVIDIA script; raise it towards `262144` if the card has headroom.

| RX 7900 XTX vs RTX 3090, same flags and harness | RX 7900 XTX | RTX 3090 |
|---|---|---|
| prefill at 33k / 75k | 746 / 634 tok/s | 988 / 820 tok/s |
| decode, thinking, at 33k / 75k | 58 / 50 tok/s | 64 / 55 tok/s |
| 262k fit check (retrieval 60k / 190k / 250k, 261,808-token turn, 7 images on the GPU) | pass, 1.9 GiB free | pass, 169 MiB free |

**Validation status (AMD):** the RX 7900 XTX numbers come from a local build of the release commit on Linux; the
CI-built Linux binary was then checked on the same card (same speed, image encoded on the GPU next to MTP). The
Windows binaries and `serve-potato-amd.bat` are tested automatically on every build but have not been run on a Windows AMD
machine by us. Other Radeon cards (RDNA2, RDNA4, 16 GB) are untested; reports are welcome.

## How it was made

1. Source: UkisAI's `Swift-Qwen3.8-27B` Q8_0 GGUF.
2. Importance matrix: computed on those weights over the Potato corpus v2 (290 coding, agentic and tool-use session documents,
   1.54M tokens, 186 chunks of 8,192 tokens), including the output head.
3. Quantization: `llama-quantize --allow-requantize` with `output.weight` at Q5_K and every other quantized tensor at IQ4_XS
   (505 IQ4_XS, 1 Q5_K, 360 F32 norm/conv/gate tensors). Keeping the output head at 5 bits recovers about three quarters of the
   divergence an all-4-bit file loses, for 190 MiB.
4. Validation on one RTX 3090: a 262k fit check (exact retrieval at 60k/190k/250k, a 261,808-token turn, 1080p and 8 MP images,
   no VRAM growth over repeated rounds), full-vocabulary KL divergence of the cache types at 65k and 150k, LiveCodeBench-80 paired
   problem by problem against alternatives, executable coding tasks after 190k and 250k real sessions against an f16-cache control,
   and a speed harness at 33k and 75k.

## Limits

- LiveCodeBench-80 at one seed resolves differences of about 5 points; this file, a sibling built from another Qwen3.8-27B
  derivative, and the f16-cache controls all sit within that band of each other. The measured advantages are context, fit and
  tokens per solved problem, not pass rate.
- The 169 MiB headroom was measured on a headless RTX 3090 with driver 580; other cards and drivers differ by a few hundred MiB.
- The requantization starts from a Q8_0 intermediate, not from full precision.

## Licence and attribution

This is a derivative of Swift-Qwen3.8-27B and is distributed under the **Swift Open License v1.0** (see `LICENSE`, `NOTICE`): free for
personal, research, educational, evaluation and commercial use by individuals and organizations with annual recurring revenue up to
US$1,000,000; above that, commercial use needs a Swift Enterprise License from UkisAI. The base model Qwen3.8-27B is Apache-2.0.
Credits: Qwen (base model), UkisAI (Swift post-training, Q8_0 and projector conversions), Slomin (coding imatrix, quantization
recipe, 262k deployment recipe and measurements).
