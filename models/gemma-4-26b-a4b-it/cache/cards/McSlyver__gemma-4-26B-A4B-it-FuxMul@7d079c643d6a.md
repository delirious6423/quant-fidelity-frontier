---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
quantized_by: McSlyver
license: other
license_name: apache-2.0-weights-agpl-3.0-engine
license_link: https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/blob/main/LICENSING.md
language:
  - en
pipeline_tag: text-generation
tags:
  - gemma4
  - gemma-4
  - moe
  - fuxmul
  - quantization
  - conversational
  - bitplane
  - lut4
  - apple-silicon
  - metal
  - windows
  - amd
  - avx512
  - avx2
  - android
  - adreno
  - mali
  - opencl
  - speculative-decoding
  - mtp
model-index:
  - name: Gemma-4-26B-A4B-IT-FuxMul-A-v3
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: OpenAI GSM8K (main)
          type: openai/gsm8k
          config: main
          split: test
        metrics:
          - type: exact_match
            value: 91.51
            name: Exact Match (%) — 0-shot, greedy, no-think
        source:
          name: FuxMul A-v3 Public Benchmark Report (Mac M1 Pro Metal, 1319/1319)
          url: https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/blob/main/benchmarks/public_gsm8k_a_v3_report.json
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AllenAI ARC-Challenge
          type: allenai/ai2_arc
          config: ARC-Challenge
          split: test
        metrics:
          - type: accuracy
            value: 92.24
            name: Accuracy (%) — 0-shot, greedy, no-think
        source:
          name: FuxMul A-v3 Public Benchmark Report (Mac M1 Pro Metal, 1172/1172)
          url: https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/blob/main/benchmarks/public_arc_challenge_1172_a_v3_report.json
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: OpenAI HumanEval
          type: openai/openai_humaneval
          split: test
        metrics:
          - type: pass@1
            value: 83.54
            name: pass@1 (%) — 0-shot, greedy, no-think, max_tokens=512
        source:
          name: FuxMul A-v3 Public Benchmark Report (Mac M1 Pro Metal, 164/164)
          url: https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/blob/main/benchmarks/public_humaneval_164_a_v3_report.json
---

# Gemma-4-26B-A4B-IT — FuxMul (`A-v3`, 7.53 GiB Direct-BF16 Pack)

**Run Google's 26-billion-parameter `Gemma-4-26B-A4B-IT` Mixture-of-Experts model 100% offline across Apple Silicon macOS, Windows x86-64, and Android ARM64** in a unified **7.53 GiB** (`8,083,193,856 B`) bitplane weight pack compiled **directly from official Google BF16 weights** (`~8.0 GiB` total including the 4-layer Multi-Token Prediction [MTP] speculative drafter) — **zero PyTorch or `llama.cpp` dependencies**.

**Accuracy on complete official test splits (0-shot greedy, no thinking):** **`91.51%` OpenAI GSM8K (`1,207/1,319`)**, **`92.24%` AllenAI ARC-Challenge (`1,081/1,172`)**, and **`83.54%` OpenAI HumanEval (`137/164` `pass@1`)**, plus **100.0%** (`20/20` full / `16/16` core) on our cross-platform executable regression suite across every verified device.

**Verified short-context decode speeds (16-prompt mean):** **82.78 tok/s (`90.83` peak)** on Apple M1 Pro Metal, **58.44 tok/s (`67.17` peak)** on Windows AMD Radeon 890M iGPU (OpenCL + MTP), **24.96 tok/s** on Windows Zen 5 AVX-512 CPU, **5.73 tok/s** on a 2018 4-core Windows AVX2 CPU, **7.08–7.21 tok/s** on a 12 GB Galaxy Tab S9 (Adreno 740), and **5.56 tok/s (`6.21` peak)** on a 2021 Galaxy S21 Ultra (ARM Mali-G78). Ships with a unified offline **Desktop Chat Web UI** (`engine/fuxmul_chat.py`) plus our clean-room **205 KiB Android Chat APK**: **[Download `FuxMulChat.apk`](https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/resolve/main/engine/android_app/FuxMulChat.apk)**.

**Quick Jump:** [Benchmarks](#public-benchmarks) · [Speed Scoreboard](#cross-platform-speed-scoreboard) · [Licensing](#licensing) · [Download](#prerequisites-and-download) · [macOS](#apple-silicon-mac-quickstart) · [Windows](#windows-pc-quickstart) · [Android APK](#android-chat-app-quickstart) · [Architecture](#architecture-inside-the-pack) · [Checksums](#repository-contents-and-checksums) · [Limitations](#current-limitations)

---

## Live On-Device Android Chat

<table>
  <tr>
    <td align="center" width="54%">
      <img src="assets/demo_tabs9_adreno740.png" alt="FuxMul Chat running Gemma-4-26B A-v3 on Samsung Galaxy Tab S9 (Adreno 740)" width="420" /><br/>
      <sub><b>Galaxy Tab S9 (12 GB, Adreno 740, Android 16)</b> — live <code>A-v3</code> logic deduction inside <code>FuxMulChat.apk</code> (6.94 tok/s, 3.36 tok/step)</sub>
    </td>
    <td align="center" width="46%">
      <img src="assets/demo_s21ultra_mali_g78.png" alt="FuxMul Chat running Gemma-4-26B A-v3 on Samsung Galaxy S21 Ultra (ARM Mali-G78)" width="350" /><br/>
      <sub><b>2021 Galaxy S21 Ultra (16 GB, ARM Mali-G78, Android 15)</b> — live <code>A-v3</code> Python generation inside <code>FuxMulChat.apk</code> (5.58 tok/s, 3.63 tok/step, 1.9 GiB free RAM)</sub>
    </td>
  </tr>
</table>

---

## Public Benchmarks

Evaluated on the **complete official Hugging Face test splits** (`2,655` problems total, zero subsets) with open, auditable grading via [`benchmarks/run_public_benchmarks.py`](benchmarks/run_public_benchmarks.py) on Apple Silicon M1 Pro Metal (`FuxMul A-v3`, `7.53 GiB`, `Cascade-32` approximate LM-head shortlist, greedy `temp = 0`, **without** thinking mode; no same-harness BF16 baseline was run):

| Benchmark (Complete Official HF Test Split) | Protocol | Score | Accuracy | Mean Decode | Evidence |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **OpenAI GSM8K** (`openai/gsm8k`, `main` test) | 0-shot greedy `exact_match` (`max_tokens=768`) | **`1,207 / 1,319`** | **`91.51%`** | **79.40 tok/s** (`γ=3`) | [`JSON`](benchmarks/public_gsm8k_a_v3_report.json) |
| **AllenAI ARC-Challenge** (`allenai/ai2_arc`, test) | 0-shot greedy generative single-letter (`A–E`, not log-likelihood `lm-eval`) | **`1,081 / 1,172`** | **`92.24%`** | *(8-tok MCQ)* | [`JSON`](benchmarks/public_arc_challenge_1172_a_v3_report.json) |
| **OpenAI HumanEval** (`openai/openai_humaneval`, test) | 0-shot greedy `pass@1` (`max_tokens=512`, isolated subprocess exec) | **`137 / 164`** | **`83.54%`** | **83.58 tok/s** (`γ=3`) | [`JSON`](benchmarks/public_humaneval_164_a_v3_report.json) |

*Combined summary: [`benchmarks/public_benchmarks_summary_a_v3.json`](benchmarks/public_benchmarks_summary_a_v3.json). Note on raw perplexity: `google/gemma-4-26B-A4B-it` is heavily instruction-tuned for `<|turn>` chat format (untemplated raw `<bos>` WikiText-2 text yields `PPL > 19,000` even on unquantized official BF16), so full official task splits (`GSM8K`, `ARC-Challenge`, `HumanEval`) plus held-out chat-templated `KL(BF16 || A-v3)` (see [Architecture](#architecture-inside-the-pack)) are used instead.*

---

## Cross-Platform Speed Scoreboard

Every row below runs **`FuxMul A-v3`** on the **shipped engines in `engine/`** (greedy decoding, `max_tokens = 384`, MTP speculative depth `γ = 3` on Mac/Android, `γ = 4` on Windows GPU, and `γ = 0` standard single-token decode on Windows CPU), scored by the hardened executable validators in [`benchmarks/gemma4_eval_harness_v2.py`](benchmarks/gemma4_eval_harness_v2.py). Decode columns show **decode-only tokens/sec** (`16p` short-context mean / peak, `all-20p` mean, and `1.8K` long-context needle decode with prompt prefill latency in parentheses).

| Device & Backend | Scope | 16p Mean / Peak (All-20p) | 1.8K Needle Decode (Prefill Time) | Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **MacBook Pro M1 Pro** (32 GB, Metal v3 + MTP `γ=3`) | **20/20** | **82.78 / 90.83 tok/s** (`79.39`) | **65.84 tok/s** (`14.4 s`) | [`JSON`](benchmarks/mac_metal_v3_26b_a_v3_g3_20p_report.json) |
| **Windows PC — AMD Radeon 890M iGPU** (UMA >= 16 GB, OpenCL + MTP `γ=4`) | **20/20** | **58.44 / 67.17 tok/s** (`53.95`) | **35.98 tok/s** (`23.1 s`) | [`JSON`](benchmarks/win_rdna35_gpu_26b_a_v3_20p_report.json) |
| **Windows PC — AMD Ryzen AI 9 HX 470** (12C Zen 5 AVX-512 CPU, `γ=0`) | **20/20** | **24.96 / 25.59 tok/s** (`23.53`) | **17.81 tok/s** (`88.7 s`) | [`JSON`](benchmarks/win_zen5_avx512_26b_a_v3_20p_report.json) |
| **Galaxy Tab S9** (12 GB, Adreno 740, `FuxMulChat.apk` + MTP `γ=3`) | **20/20** | **7.08 / 8.11 tok/s** (`6.56`) | **4.45 tok/s** (`151.9 s`) | [`JSON`](benchmarks/tabs9_adreno740_v3_26b_a_v3_apk_20p_report.json) |
| **Galaxy Tab S9** (12 GB, Adreno 740, `FuxMulChat.apk` + **Cat-Neuro Focus ON**) | **20/20** | **7.21 / 8.04 tok/s** (`6.67`) | **4.48 tok/s** (`151.7 s`) | [`JSON`](benchmarks/tabs9_adreno740_v3_26b_a_v3_catneuro_apk_20p_report.json) |
| **2018 Windows PC — Ryzen 3 PRO 2200G** (4C AVX2 CPU, 16 GB DDR4, `γ=0`) | **16/16** | **5.73 / 6.01 tok/s** *(16p core)* | *(4 long-prefill needles omitted; `1.48 GiB` private RAM + OS `mmap` cache)* | [`JSON`](benchmarks/win_ryzen2200g_avx2_26b_a_v3_report.json) |
| **2021 Galaxy S21 Ultra** (16 GB, Mali-G78, `FuxMulChat.apk` + MTP `γ=3`) | **20/20** | **5.56 / 6.21 tok/s** (`5.31`) | **4.27 tok/s** (`255.8 s`) | [`JSON`](benchmarks/s21ultra_mali_g78_v3_26b_a_v3_apk_20p_report.json) |

*All three Android rows were executed end-to-end directly inside the foreground **`FuxMulChat.apk`** WebView app (`0` OOMs, `Min MemAvailable = 580 MiB` on 12 GB Tab S9 and `1,922 MiB` on 16 GB S21 Ultra). The 2018 4-core AVX2 PC ran the 16-prompt core suite (omitting the 4 slow 1.8K-token CPU prefills) via zero-copy `mmap` (`1.48 GiB` private working set plus OS file-page cache for the `7.53 GiB` pack).*

---

## Licensing

| Scope | Files | License |
| :--- | :--- | :--- |
| **Model Weights, Tokenizer, Checksums, Benchmarks & Demo Images** | Root `*.astca`, `*.bin`, `gemma4_tokenizer.json`, `SHA256SUMS.txt`, `benchmarks/**`, `assets/**` | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)) |
| **FuxMul Inference Engine, Kernels, Web UI, CLIs, Android App & Prebuilt Binaries** | Everything under `engine/**` | **GNU AGPL-3.0-or-later** **OR** **Commercial** (`LicenseRef-FuxMul-Commercial`) — see [`LICENSING.md`](LICENSING.md), [`engine/LICENSE`](engine/LICENSE), and [`engine/COMMERCIAL-LICENSE.md`](engine/COMMERCIAL-LICENSE.md) |

---

## Prerequisites and Download

### Supported Platforms and Minimum Specs (Read Before Downloading 8.0 GiB)

- **macOS:** Apple Silicon (`M1` / `M2` / `M3` / `M4`, ARM64 only — Intel Macs are not supported). **16+ GB RAM** recommended for Metal GPU (`8 GB` Macs supported via CPU `--backend neon`).
- **Windows:** Windows 10/11 x86-64 (Windows ARM64 is not supported; native Linux prebuilts are not shipped yet). **16+ GB RAM** (AVX2/AVX-512 CPU runs on virtually any x86-64 PC since 2013; OpenCL GPU requires >= 8.5 GiB accessible GPU memory).
- **Android:** Android 12+ (API 31+) ARM64 with **12+ GB RAM**, public vendor `libOpenCL.so`, and **>= 16.5 GiB free storage** during one-time GPU setup (reclaims back to **8.0 GiB** afterward).

### Python Environment and Download (Python 3.9+)

Create a clean virtual environment and install the 3 lightweight host packages (`numpy`, `tokenizers`, `huggingface_hub`):

**macOS / Linux (`bash` / `zsh`):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U numpy tokenizers huggingface_hub
hf download McSlyver/gemma-4-26B-A4B-it-FuxMul --local-dir ./gemma4-fuxmul
cd ./gemma4-fuxmul
```

**Windows (`PowerShell`):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U numpy tokenizers huggingface_hub
hf download McSlyver/gemma-4-26B-A4B-it-FuxMul --local-dir .\gemma4-fuxmul
cd .\gemma4-fuxmul
```

> **Download note:** Total download is approx. **8.0 GiB** (`engine/`, `benchmarks/`, and `assets/` add approx. `5.0 MiB`). `hf download` writes files without execute (`+x`) bits on POSIX, so invoke shell scripts with `bash engine/.../*.sh` as shown below.

---

## Apple Silicon Mac Quickstart

- **Metal GPU (`--backend metal`, default):** Tested on **32 GB** Apple Silicon. Total Metal + staging footprint during initialization peaks near **9.5 GiB** (`~8.9 GiB` steady-state with MTP). On a **16 GB Mac**, close heavy browsers/IDEs before running, or pass `--gamma 0` (disables speculative decoding and skips the 434 MiB MTP drafter).
- **NEON CPU / 8 GB Macs (`--backend neon`):** Uses multiplier-free 4-bit LUT4 on CPU (`γ = 0`, single-digit tok/s) and supports optional 4096B-aligned `ExpertBlob` streaming.

### Run on Metal GPU (Default)

```bash
# Launch the offline dark-mode local Chat Web UI (auto-opens http://127.0.0.1:8765; supports --gamma 0, --port, --no-browser)
python3 engine/fuxmul_chat.py

# Multi-turn interactive terminal chat (default max-tokens=384, max-seq-len=2560; type /continue to finish a truncated reply, /clear to reset)
python3 engine/fuxmul_mac.py --chat

# Single prompt (add --thinking for Gemma 4's thinking channel with graceful budget auto-close)
python3 engine/fuxmul_mac.py --prompt "Write a Python function for binary search."

# Fast parity + 4-prompt executable Python unit-test verification
python3 engine/fuxmul_mac.py --quick-verify

# Full verification: Full-262K vs Cascade-32 parity + 64-tok B=1 vs MTP check + 20-prompt hardened suite
python3 engine/fuxmul_mac.py --verify
```

### Run on ARM64 NEON CPU (or 8 GB Apple Silicon Mac via `ExpertBlob`)

```bash
# Run standard .astca directly on CPU NEON:
python3 engine/fuxmul_mac.py --backend neon --prompt "What is 2 + 2?"

# Optional (recommended on 8 GB Macs; requires ~7.53 GiB temporary free disk space for one-time 4K-aligned repack):
python3 engine/mac_neon/repack_expertblob.py \
  gemma4_26b_a4b_fuxmul_a_v3.astca \
  gemma4_26b_a4b_fuxmul_a_v3_blob.astca

python3 engine/fuxmul_mac.py --backend neon \
  --model gemma4_26b_a4b_fuxmul_a_v3_blob.astca \
  --chat
```

### Rebuilding the macOS Libraries Locally (Optional / Gatekeeper)

Prebuilt `.dylib` files are ad-hoc signed. If macOS Gatekeeper blocks loading a downloaded `.dylib`, either clear quarantine (`xattr -d com.apple.quarantine engine/mac_*/*.dylib`) or rebuild locally with Xcode Command Line Tools (`xcode-select --install`):

```bash
bash engine/mac_metal/build_mac_metal.sh
bash engine/mac_neon/build_mac_neon.sh
```

---

## Windows PC Quickstart

Works out-of-the-box on Windows 10/11 x86-64 with **zero CUDA, ROCm, HIP, Vulkan, or C++ Redistributable installation** — all three prebuilt DLLs in `engine/windows/` are statically linked (importing only `KERNEL32.dll` + Windows UCRT), and `--backend auto` picks the fastest safe backend automatically:

1. **OpenCL GPU + 4-Layer MTP (`fuxmul_win_gpu.dll`):** Dynamically loads Windows `OpenCL.dll` on OpenCL 1.2+ GPUs with >= 8.5 GiB accessible VRAM/UMA (`max_seq_len <= 4096`). Verified live on **AMD Radeon 890M (RDNA 3.5 `sudot4`, `58.44 / 67.17 tok/s`)**; also includes an untested AMD RDNA 2 (`sdot4`) tier and portable OpenCL 1.2 fallback for NVIDIA GeForce/RTX & Intel Arc/Iris GPUs (community reports welcome).
2. **AVX-512 VNNI CPU (`fuxmul_win_avx512.dll`):** Zero-copy `mmap` (`24.96 tok/s` verified on Ryzen AI 9 HX 470, `0.69 s` cold init) — only loaded after CPUID confirms `AVX512F + BW + DQ + VL + VNNI`.
3. **Universal x86-64 AVX2 CPU (`fuxmul_win_avx2.dll`):** Zero AVX-512 instructions — runs on virtually any Windows PC made since 2013 (Intel Haswell+ / AMD Zen 1+), verified at `5.73 tok/s` (`16/16`) on a 2018 4-core Ryzen 3 PRO 2200G with 16 GB DDR4 and no discrete GPU.

### Run on Windows (Auto-Detect Backend)

```powershell
# Launch the offline dark-mode local Chat Web UI (auto-opens http://127.0.0.1:8765; auto-detects GPU -> AVX-512 -> AVX2)
python engine\fuxmul_chat.py

# Or run the interactive terminal CLI (supports /continue, /clear, --thinking, --backend auto|gpu|avx512|avx2)
python engine\fuxmul_win.py --chat

# Fast parity + 4-prompt Python unit-test verification
python engine\fuxmul_win.py --quick-verify

# Full parity + 20-prompt hardened suite
python engine\fuxmul_win.py --verify
```

### Rebuilding the Windows DLLs Locally (Optional — MinGW-w64 GCC; `avx2` Also Supports MSVC)

```powershell
powershell -ExecutionPolicy Bypass -File engine\windows\build_win_gpu.ps1
powershell -ExecutionPolicy Bypass -File engine\windows\build_win_avx512.ps1
powershell -ExecutionPolicy Bypass -File engine\windows\build_win_avx2.ps1
```

---

## Android Chat App Quickstart

Chat directly with **Gemma-4-26B-A4B-IT (`A-v3`)** on your Android phone or tablet using our clean-room **205 KiB** offline APK — **[Download `FuxMulChat.apk`](https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/resolve/main/engine/android_app/FuxMulChat.apk)** — featuring a built-in pure-C Gemma 4 BPE tokenizer (`30/30` parity across ASCII, UTF-8/emoji, code, math, and all 24 Gemma 4 special tokens), one-time in-app GPU weight optimizer, live token streaming, Markdown + Code Copy, Thinking Mode with graceful budget auto-close, seamless in-bubble `Continue ▸`, and optional zero-reload `Cat-Neuro Focus` (100% offline; zero external network requests).

### Requirements

- **Verified Hardware:** **Samsung Galaxy Tab S9** (`Adreno 740`, 12 GB RAM, Android 16) and **Samsung Galaxy S21 Ultra** (`Exynos 2100 / ARM Mali-G78 MP14`, 16 GB RAM, Android 15).
- **OS & Driver:** **Android 12+ (API 31+)** with public vendor `libOpenCL.so` (declared via `<uses-native-library>`).
- **Memory & Storage:** **12+ GB RAM** (approx. 7–8 GiB free RAM when loading; close heavy background apps on 12 GB devices) and **>= 16.5 GiB free storage** during the one-time 20 s GPU optimization. The in-app checkbox **"Delete source `.astca` after setup"** is enabled by default, reclaiming storage down to **8.0 GiB** automatically once optimization finishes.

### Install (ADB or Manual Sideload)

**Option A — One-Command ADB Install (Fastest):** Requires downloading the repo above, Android Platform Tools (`adb`), **Developer Options -> USB Debugging** enabled on the device, and Bash (macOS/Linux terminal or **Git Bash** on Windows):

```bash
cd ./gemma4-fuxmul
bash engine/android_app/install_android_app.sh
```

**Option B — Manual Sideload (No ADB):** Note that macOS requires `adb` or an MTP tool like OpenMTP to transfer files to Android, and sideloading a self-signed APK requires enabling *"Install unknown apps"* (dismissing Play Protect's unknown-developer warning):

1. Sideload [`FuxMulChat.apk`](https://huggingface.co/McSlyver/gemma-4-26B-A4B-it-FuxMul/resolve/main/engine/android_app/FuxMulChat.apk) and open **FuxMul Chat** once.
2. Place the 4 model files (`gemma4_26b_a4b_fuxmul_a_v3.astca`, `a_v3_manifest.bin`, `mtp_gemma4_26b_flat.bin`, `gemma4_tokenizer.json`) into `Download/gemma4-fuxmul/` and tap **Allow Downloads Access** in the app (or copy via USB into `Android/data/com.fuxmul.chat/files/fuxmul/` if accessible on your device).
3. Tap **Optimize for GPU (approx. 20s)** once (keep *"Delete source `.astca` after setup"* checked to reclaim approx. 8 GiB), then tap **Load 26B Model** and start chatting!

### Rebuilding the Android App from Source (Android NDK r25+ & SDK API 35)

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r27
export ANDROID_HOME=/path/to/android-sdk
bash engine/android_app/build_jni.sh
bash engine/android_app/build_apk.sh
```

---

## Android CLI Harness

### Requirements and Caveats

- **Device Requirements:** Same hardware, Android 12+, `12+ GB RAM` (>= 8.0 GiB idle `MemAvailable`), and **>= 16.5 GiB** temporary storage requirements as the Android Chat App above, plus OpenCL 2.0/3.0 with `cl_khr_subgroups`.
- **CLI Modes:** Supports **parity, profiler, and benchmark modes (`paris` [parity] | `smoke` | `profile` | `full`)** using pre-tokenized prompts in `benchmarks/eval_20p_prompts.txt` (for interactive on-device chat, use `FuxMulChat.apk` above).
- **Host Tools:** Requires `adb` with USB Debugging enabled and Bash (macOS/Linux or **Git Bash** on Windows; WSL does not forward USB natively without `usbipd-win`). If multiple Android devices are connected, prefix commands with `ANDROID_SERIAL=<device_id>`.

### Deploy and Verify

```bash
# Make sure you downloaded the repo first (see Prerequisites and Download above)
cd ./gemma4-fuxmul

# 1. Push binaries + 7.53 GiB .astca, run atomic tile-64 repack, delete temp .astca, and run smoke/parity check:
bash engine/android_adreno/run_android.sh smoke

# 2. Run the full 20-prompt executable benchmark (~15-20 min including four 1.8K needles; keep device awake & plugged in):
bash engine/android_adreno/run_android.sh full

# 3. Score the pulled JSONL output (pure-Python scorer; works identically via fuxmul_mac.py or fuxmul_win.py on any OS):
python3 engine/fuxmul_mac.py --score-jsonl benchmarks/android_20p_out.jsonl
```

### Rebuilding the Android CLI Binaries from Source (Standard Android NDK r25+, API 31+)

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r27
bash engine/android_adreno/build_android.sh
```

---

## Highlights

- **Compact 8.0 GiB Unified Pack:** Compresses the 26B MoE model to **7.53 GiB** (`gemma4_26b_a4b_fuxmul_a_v3.astca`) plus **434 MiB** for the 4-layer MTP speculative drafter (`mtp_gemma4_26b_flat.bin`).
- **Bitplane Execution Across 4 GPU Vendors + ARM64/x86-64 CPUs:** Every quantized linear layer uses unitary **FWHT-64** (Fast Walsh-Hadamard Transform, block size 64) orthogonalization with 2-plane **Syzygy-4x4** (4-bit bitplane) on the attention/shared-FFN **Spine** and 1-plane **Syzygy-4** (2-bit bitplane) across all **3,840 MoE experts** (`30 layers × 128 experts`).
  - **CPU & Mobile Execution:** Executes quantized Spine & MoE bitplane linear layers via **multiplier-free 4-bit LUT4** (Apple Silicon ARM64 NEON, x86-64 AVX2, and Android Adreno/Mali OpenCL) and **8-bit integer-dot `vpdpbusd`** (x86-64 AVX-512 VNNI); router projections, LM head, MTP drafter, and per-block scales use standard FP32/FP16 FMA.
  - **Desktop GPU Execution:** Windows `fuxmul_win_gpu.dll` dynamically loads Windows `OpenCL.dll` (zero CUDA/ROCm/HIP/Vulkan SDK required; verified on AMD Radeon 890M RDNA 3.5 `sudot4`, with an untested portable OpenCL 1.2 fallback for AMD RDNA 2, NVIDIA GeForce/RTX, and Intel Arc/Iris GPUs); Apple Metal executes native FP16 `fma()`.
- **High MTP Speculative Acceptance:** Direct-BF16 `A-v3` reaches **`86.7%–100.0%`** draft acceptance on M1 Pro Metal (`γ = 3`) and **`75.0%–100.0%`** on Windows Radeon 890M GPU (`γ = 4`), driving `3.63–3.64 tok/step` on Adreno 740 & Mali-G78 mobile GPUs.
- **Unified Offline Desktop Chat Web UI (`engine/fuxmul_chat.py`) & 205 KiB Android APK (`FuxMulChat.apk`):** Shared dark-mode chat experience with live SSE streaming, `Stop`, Markdown + Code Copy, Thinking Mode with graceful `<channel|>` auto-close & auto-collapse, seamless in-bubble **`Continue ▸`**, localhost token + CSP + `X-Frame-Options` hardening on desktop, and an opt-in zero-reload **`Cat-Neuro Focus`** toggle on Android (`0` extra `.astca` bytes; verified `20/20` at `7.21 tok/s` on Tab S9 with `-5.6%` fewer tokens).
- **Turnkey Standalone Execution & Hardened 20-Prompt Cross-Device Suite:** Prebuilt standalone native binaries for macOS, Windows, and Android plus full build scripts and a 20-prompt executable regression suite (Python unit tests with restricted builtins + timeout, multi-step math, strict JSON schema, logical reasoning with anti-flip-flop guards, and four 1.8K-token needle prompts across the 1,024-token Sliding-Window Attention [SWA] boundary) plus 17 adversarial validator self-tests.

### Measurement and Parity Notes

- **Apple Silicon Metal (`M1 Pro 32 GB`, `A-v3`):** Running `python3 engine/fuxmul_mac.py --verify` locally checks exact Full-262K vs `Cascade-32` vs micro-batch parity, 3-run determinism (`min_cos >= 0.9999`), **16-token exact Full-262K vs `Cascade-32` verbatim parity**, **64-token `B=1` vs MTP `γ=3` verbatim match**, and all 20 hardened prompts (`all-20p mean = 79.39 tok/s`).
- **Android Adreno 740 (`Tab S9`, 12 GB) & ARM Mali-G78 (`S21 Ultra`, 16 GB):** All 30 transformer layers (`5.99 GiB` tile-64 Spine + all `3,840` experts across 8 OpenCL slabs + `253.1 MiB` compact KV cache + `434 MiB` MTP drafter) stay **GPU-resident** with zero host fences during decode `verify_batch`; the `704.5 MiB` Q8 embedding / LM head (`tok_q8.bin`) is CPU-`mmap`'d across performance NEON cores. S21 Ultra (`cl_khr_subgroups`, no QCOM extensions, `~16 s` in-app repack, `Min MemAvailable = 1,922 MiB`) matches Tab S9's `20/20` hardened score (`3.63` vs `3.64 tok/step`). *(Log transparency note: in the baseline Tab S9 APK JSON, `needle_03`'s `output_preview` captured the preceding WebView scroll window though the 32-token reply completed cleanly; the clean 32-token `needle_03` reply preview is visible in both the Tab S9 Cat-Neuro and S21 Ultra APK logs.)*
- **Apple Silicon NEON (`--backend neon`):** Default progressive LM-head cascade prioritizes Latin/ASCII/code/math/punctuation (`use_lm_head_cascade=False` in Python API enables full 262K multilingual vocab).
- **Cross-Device Floating-Point Order:** Because Metal FP16 FMA and OpenCL LUT4/FP32 accumulate in different reduction orders, long greedy generations can differ slightly in length or phrasing across backends.

---

## Architecture Inside the Pack

1. **Asymmetric Bitplane Budget (`A-v3`, Direct-BF16):**
   - **Attention & Shared-FFN Spine (`2P Syzygy-4x4`, `~4.25 bpw`):** 16 uniform signed quantization levels per block of 64 weights (`FWHT-64` orthogonalized, 1 shared FP16 scale/64), compiled directly from BF16 with absorbable Cauchy-Schwarz SmoothQuant (`α = 0.5`) and diagonal Hessian Lloyd refinement (`28.08 dB` on middle-layer `blk.15` vs `23.85 dB` uncalibrated direct-BF16 `Syzygy-4x4` and `20.02 dB` `Q4_0`-derived `Syzygy-4x4`).
   - **Routed MoE Experts (`3,840 × 1P Syzygy-4`, `~2.25 bpw`):** Encodes all 128 experts across all 30 layers in `5.99 GiB` with shared `pre_ffw_norm_2` SmoothQuant, per-expert GeGLU intermediate gauge equilibration (`λ_{e,j}`), Rademacher/LPT block balancing, regularized `11×11` block-scale LSQ (`7.83 → 10.53 dB` on unseen tokens), and layerwise closed-form per-channel Wiener norm calibration (`ffn_down_exps.scale`, `post_ffw_norm_1`, `post_ffw_norm_2`, `post_ffw_norm`) — boosting final-layer (`L29`) trajectory cosine against BF16 from **`0.7656` (`A-v2`) → `0.9488` (`A-v3`)** (`4.58×` smaller `1 - cos` directional drift).
   - **Embeddings & LM Head (`Q8`):** Row-wise symmetric INT8 + FP16 scale (`704.5 MiB`) directly from BF16, paired with a **`Cascade-32`** sign-sketch prefilter (`n_top_blocks = 32`, verified exact against full-262K Top-1 argmax on 2 code/reasoning prompts; exact full-262K evaluation remains available in the Python engine API via `head_mode=1`).
2. **All-GPU 4-Layer MTP Speculative Drafter (`mtp_gemma4_26b_flat.bin`):**
   - Repacked from Google's `gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` into a single `434 MiB` flat Q8 binary loaded by Metal, Windows OpenCL, Adreno, and Mali OpenCL. Every draft token is verified by the target model (verified verbatim-identical on a 64-token `B=1` vs MTP check).
3. **Universal Android OpenCL Execution Design (Adreno 740+ & ARM Mali-G78+):**
   - One-time on-device tile-64 transposition (`fuxmul_repack_t64` or in-app optimizer) aligns 64-row tiles for coalesced GPU loads and extracts standalone `tok_q8.bin` so the 7.53 GiB source `.astca` can be deleted (`~8.0 GiB` final device footprint).
   - In-GPU router bucketing partitions active experts by batch multiplicity (`c1`, `c2`, `cN` — experts hit by 1, 2, or N tokens in a speculative batch) without host readbacks during decode `verify_batch`.
   - Dynamically detects Qualcomm `cl_qcom_reqd_sub_group_size` (`64`-wide subgroups on Adreno) and falls back automatically to portable `cl_khr_subgroups` + `__local` reductions on ARM Mali-G78 and other OpenCL GPUs.

### Direct-BF16 Calibration Progression (`512` Disjoint Chat-Templated Tokens)

Compared to our earlier archived `Q4_0`-derived pack (`A-v2`, retained in `benchmarks/` for reference), compiling directly from BF16 with the zero-extra-byte calibration stack cuts mean tokenwise `KL(BF16 || *)` by **`29.4%`** at the exact same `7.528 GiB` (`8,083,193,856 B`) footprint:

| Pack | Size | Mean `KL(BF16 \|\| *)` (nats) | Top-1 Match vs. BF16 | Hardened 20-Prompt Suite |
| :--- | :---: | :---: | :---: | :---: |
| `A-v2` (archived, from `Q4_0`) | `7.53 GiB` | `1.84508` | `57.62%` | `18/20` (Mac Metal, 384 tok) |
| **`A-v3` (Direct BF16 + Calibration Stack)** | **`7.53 GiB`** | **`1.30215` (`-29.4%` vs `A-v2`)** | **`67.38%` (`+9.76 pp`)** | **`20/20 (100.0%)` (Mac, Windows & Android)** |

---

## Repository Contents and Checksums

| Path | Size | Description | License |
| :--- | ---: | :--- | :--- |
| `gemma4_26b_a4b_fuxmul_a_v3.astca` | 7.53 GiB | Unified Direct-BF16 `A-v3` Gemma-4-26B-A4B-IT bitplane + Q8 pack | Apache 2.0 |
| `mtp_gemma4_26b_flat.bin` | 434 MiB | Unified 4-layer Gemma 4 MTP Q8 drafter + Cascade tables | Apache 2.0 |
| `a_v3_manifest.bin` | 44 MiB | Binary `ASTMAN01` offset manifest for Android repacker & runtime | Apache 2.0 |
| `gemma4_tokenizer.json` | 31 MiB | Official Google Gemma 4 `tokenizers` JSON (24 added special tokens) | Apache 2.0 |
| `SHA256SUMS.txt` | 1 KiB | Complete SHA-256 checksums for all weight files & prebuilt native binaries | Apache 2.0 |
| `benchmarks/` | 700 KiB | Official public benchmark runner & full logs (`GSM8K`, `ARC-Challenge`, `HumanEval`) + 20-prompt cross-device suite | Apache 2.0 |
| `assets/` | 676 KiB | Cropped on-device `A-v3` Android app screenshots (`Tab S9` & `S21 Ultra`) | Apache 2.0 |
| `engine/fuxmul_chat.py` + `fuxmul_chat_ui.html` | 72 KiB | Unified offline browser Chat Web UI for macOS & Windows (SSE streaming, Thinking auto-close, `Continue ▸`) | AGPL-3.0-or-later / Commercial |
| `engine/fuxmul_mac.py` | 52 KiB | Apple Silicon Metal & NEON CLI + cross-OS JSONL scorer | AGPL-3.0-or-later / Commercial |
| `engine/fuxmul_win.py` | 96 KiB | Windows OpenCL GPU, AVX-512 & Universal AVX2 CLI (`--backend auto`, `--ui`, `--verify`) | AGPL-3.0-or-later / Commercial |
| `engine/fuxmul_format.py` + `fuxmul_tokenizer.py` | 28 KiB | Shared `.astca` binary reader and Gemma 4 chat/tokenizer helpers (`engine/LICENSE`, `COMMERCIAL-LICENSE.md`, `NOTICE`) | AGPL-3.0-or-later / Commercial |
| `engine/mac_metal/` | 288 KiB | Apple Metal ObjC++ host, MSL shader, prebuilt ad-hoc signed `.dylib`, build script | AGPL-3.0-or-later / Commercial |
| `engine/mac_neon/` | 204 KiB | ARM64 NEON LUT4 C library, runtime, `ExpertBlob` repacker, ad-hoc signed `.dylib` | AGPL-3.0-or-later / Commercial |
| `engine/windows/` | 1.54 MiB | Windows OpenCL GPU + AVX-512 + Universal AVX2 C sources, PowerShell build scripts, static zero-dep DLLs | AGPL-3.0-or-later / Commercial |
| `engine/android_app/` | 884 KiB | Offline Android Chat App (self-signed `FuxMulChat.apk`, pure-C BPE, JNI server, Java/WebView source, NDK/SDK build scripts) | AGPL-3.0-or-later / Commercial |
| `engine/android_adreno/` | 408 KiB | Universal Adreno + Mali-G78 OpenCL CLI host, kernels, atomic tile-64 repacker, `build_manifest.py`, prebuilt ARM64 ELFs | AGPL-3.0-or-later / Commercial |

### Verify Checksums (`SHA256SUMS.txt`)

Every weight pack, manifest, tokenizer `.json`, and prebuilt native binary (`.astca`, `.bin`, `.dylib`, `.dll`, `.so`, `.apk`, and Android ELFs) is covered in [`SHA256SUMS.txt`](SHA256SUMS.txt) — or you can delete all prebuilt binaries and rebuild every target cleanly from the included source scripts:

```bash
# macOS / Linux (or use `sha256sum -c SHA256SUMS.txt` on minimal Linux installs)
shasum -a 256 -c SHA256SUMS.txt
```

```powershell
# Windows PowerShell
Get-Content SHA256SUMS.txt | ForEach-Object { $h,$f = $_ -split '\s+',2; "$((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -eq $h)  $f" }
```

---

## Current Limitations

- **Text-Only (No Vision Tower):** The Gemma 4 multimodal vision encoder (`mmproj`) is omitted from this `7.53 GiB` text-generation pack.
- **Greedy Decoding & English-Only Evaluation:** All engines currently run deterministic greedy argmax (`temp = 0`); sampling controls (`temperature`, `top-p`, `repeat-penalty`), an OpenAI-compatible `/v1/chat/completions` endpoint, and non-English / multilingual benchmarks are not yet included.
- **Long-Prompt Prefill Latency on CPU & Mobile:** While short-context prompts respond immediately, 1.8K-token prompt prefill takes `14.4 s` on M1 Pro Metal, `23.1 s` on Radeon 890M GPU, `88.7 s` on Zen 5 CPU, `151.9 s` on Adreno 740, and `255.8 s` on Mali-G78.
- **Context Defaults:** Default context is `2,560` tokens (`--max-seq-len` scales to `16K–32K+` on Mac Metal & Windows CPU; currently capped at `4,096` on Windows OpenCL GPU and fixed at `2,560` in the Android APK).
- **Platform & Compiler Scope:** Turnkey prebuilt engines cover macOS (Apple Silicon), Windows x86-64, and Android ARM64 (native Linux builds are not shipped yet). The inference engines (`engine/**`) are open-source (`AGPL-3.0-or-later`); the offline BF16 → `.astca` compiler is not included in this repository.

---

## Acknowledgments and Credits

- **Google DeepMind** — for releasing `Gemma-4-26B-A4B-IT` and the 4-layer `Gemma-4-26B-A4B-IT-QAT-Q4_0-Unquantized-Assistant` under Apache 2.0. *(Note to the Google DeepMind / Gemma & LiteRT teams: royalty-free Apache 2.0 / CLA licensing of `engine/**` is available on request — see [`engine/COMMERCIAL-LICENSE.md`](engine/COMMERCIAL-LICENSE.md).)*
- **Google Gemini** — special thanks to Google Gemini as my AI engineering assistant throughout architecture exploration, debugging, and live hardware verification.
- **[Elephant-Goldfish Model (EGM)](https://github.com/google/elephant-goldfish-model) Team ([`egm.dev`](https://egm.dev))** — huge thanks to the EGM team for building the open-source multi-persona Goldfish review framework that helped me vibe-code the Fux out of this project — ruthlessly stress-testing the Metal, NEON, Adreno/Mali OpenCL kernels, pure-C BPE tokenizer, BF16 compiler, and Android chat app before release.
- **McSlyver** — creator and maintainer of the FuxMul quantization format, inference engine, and on-device ports.
