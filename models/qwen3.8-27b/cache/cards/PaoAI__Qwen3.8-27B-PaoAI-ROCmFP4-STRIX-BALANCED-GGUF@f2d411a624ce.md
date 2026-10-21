---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- quantized
- imatrix
- strix-halo
- rocmfp4
- gfx1151
- qwen3
pretty_name: Qwen3.8-27B PaoAI ROCmFP4 STRIX BALANCED (GGUF)
---

# Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED-GGUF

**The balanced Qwen3.8-27B quant for AMD Strix Halo (Ryzen AI Max+ 395 / gfx1151).** Measured, not marketed: one 15 GB file on the iGPU, the full 262,144-token window integrity-checked, three serving profiles — **reading 101–410 t/s and writing 14.8–49.8 t/s** across the window (depending on how full it is, what you ask for, and the profile).

*Reading = processing your prompt (prefill). Writing = producing the answer (decode). t/s = tokens per second.*

**Brand: PaoAI** · community quant, not affiliated with Alibaba or the Qwen team.

> **Updated 2026-09-29 — new engine version + three serving profiles, weights unchanged.** The latest version runs on
> [paoai-strix-engine `strix-v4`](https://github.com/guevae2/paoai-strix-engine/tree/strix-v4) (`e9b93d56a`) with three ready-made settings: **Chat** (default), **Code** (with a
> small helper model) and **Long** (for long documents and long code). Same file, same sha256. The speed tables and the quality bench were
> measured on the published engine; the coding exam and the 256K needle ran on the same code a day earlier (noted where they appear).
> The previous version's numbers are kept only in the "against the previous version" table. See [Update log](#update-log).

---

## What is this? (30 seconds)

Qwen3.8-27B is a hybrid: 48 cheap linear-attention layers and only 16 full-attention layers that actually re-read your context. We compressed it **part by part**: the finer 4.5-bit FP4 layout where it matters most (attention K/V and the linear-attention qkv), the faster 4.25-bit layout for the bulk, 6-bit token embeddings, every norm at full precision — guided by an importance matrix. One 15 GB file, no offload flags, speculative decoding built in, checked from 8K to the full 256K window.

## The recipe (what's inside — read from the file)

| Model part | What it does | Format (share of its weights) | Size |
|---|---|---|---|
| FFN gate/up + down (64 layers) | the knowledge — the biggest block | ROCmFP4-FAST (4.25 bits) | 9.09 GB |
| Linear attention (48 gated-delta layers) | the fast path — constant-size state | ROCmFP4-FAST (4.25 bits) 55 % + ROCmFP4 (4.5 bits) 45 % | 3.03 GB |
| Full attention Q/K/V/O (16 layers) | reads your prompt, tracks context | ROCmFP4-FAST (4.25 bits) 90 % + ROCmFP4 (4.5 bits) 10 % | 0.90 GB |
| Token embeddings | vocabulary lookup | Q6_K (6.56 bits) | 1.04 GB |
| Output head | picks the next word | ROCmFP4-FAST (4.25 bits) | 0.68 GB |
| Draft head (blk.64, MTP) | guesses the next tokens for speculative decoding | ROCmFP4-FAST (4.25 bits) 85 % + Q8_0 12 % + ROCmFP4 (4.5 bits) 2 % + FP32 (norms, <1 %) | 0.25 GB |
| Norms and small state (conv, decay) | keep signals correct | FP32 | 0.01 GB |

**Total: 15.01 GB, 27.3 B weights, 4.40 bits per weight on average.** *(Measured from the published file's tensor table. Our earlier card listed the draft head as Q8_0 and the whole of full attention as one format; the file has always been as above — same sha256.)*

## Which profile? (the same file, three ways to serve it)

| Profile | Use it for | What it switches on |
|---|---|---|
| **Chat** (default) | chat, writing, mixed use | the model's own draft head (MTP, 4 guesses) + a cheaper guess (the draft head scores only the 65,536 most-used tokens) + a faster check of the guesses |
| **Code** | coding agents, up to ~64K of context | a separate small helper model ([z-lab DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), 2.1 GB) that guesses 3–7 tokens at a time |
| **Long** | long documents and long code (64K+) | the Code helper + a full-precision (f16) context memory laid out so reads spread over all memory channels |

All three give the same answers the model would give without any speed-up, within rounding (checked below). Pick by what you do most.

## Coding exam — "chain test" (3 runs per profile, fresh server each — latest version, 2026-09-28)

A fixed coding exam graded by **running the code**, not by opinion: the model reads deceptive code and must trace it, implements a novel state machine, and writes a thread-safe ring buffer that survives a 5,000-operation stress test. **Three identical runs; we report medians with the spread** (identical runs vary — single scores are unreliable).

| Task (what it measures) | Chat — median (runs) | Code — median (runs) | Long — median (runs) |
|---|---|---|---|
| **Hunt** — read lying code: trace it + find the planted bug | **100** (100 · 100 · 40) | **100** (100 · 100 · 100) | **100** (100 · 100 · 100) |
| **Automaton** — implement novel rules + exact state trace | **100** (100 · 100 · 100) | **100** (100 · 100 · 100) | **100** (100 · 100 · 100) |
| **Ring buffer** — multi-threaded code under stress | **20** (100 · 20 · 20) | **100** (100 · 100 · 100) | **100** (100 · 100 · 100) |

**Code and Long scored 100 on every task in all three runs.** Chat's run totals were 100 · 73 · 53: in two of them the ring-buffer code crashed on `Condition.notify_one()`, a method Python does not have (the hunt miss in the third run was a trace step). We checked it with more runs, rules fixed before running: over **10 Chat runs** the ring buffer failed **3** times (2× `notify_one`, 1× notify without holding the lock); with this card's previous settings it failed 0 of 6 (those runs slipped on other tasks instead). The **original full-precision model** (BF16, no quantization, a cloud server, 5 runs) wrote the same `notify_one` crash 1 time and failed the ring buffer 2 times in all — so it is the base model's habit, not something the quant or the Chat settings add. The gap between 3 of 10 and 0 of 6 is within chance (Fisher p ≈ 0.25).

### Writing speed per phase (tokens/second, range over the 3 runs)

| Phase | Chat | Code | Long |
|---|---|---|---|
| Spec (planning) | **26.6 – 28.9** | **25.8 – 26.7** | **25.5 – 27.3** |
| Hunt (analysis) | **29.8 – 33.0** | **31.3 – 38.5** | **31.5 – 34.7** |
| Automaton (code writing) | **34.1 – 39.6** | **36.0 – 40.4** | **36.6 – 42.6** |
| Ring buffer (code writing) | **27.8 – 28.9** | **27.6 – 31.4** | **27.3 – 34.3** |
| Publish (long output) | **34.0 – 34.3** | **29.8 – 34.1** | **31.1 – 34.4** |

*Short prompts (a few hundred tokens). Exam settings: 64K window, thinking budget 1,500 tokens, each profile on its own settings. Run 2026-09-28 on our test build of the exact engine code that was then published as `strix-v4` (same source tree, checked).*

### Does quantization hurt? (the file itself, unchanged by this update — measured 2026-09-17)

**Plain words:** the chain test above can only tell you if the model still *passes*. This test asks a different question: **did compressing the weights change what the model actually predicts?** We ran the industry-standard check — llama.cpp's own `llama-perplexity` tool — comparing this 15 GB FP4 file against a full-precision **F16 copy of the same model** (built by us from Qwen's official BF16 release, sha-verified shards), reading the same ~295,000 tokens of standard Wikipedia text through both at 8K windows.

| Metric | What it means | Result |
|---|---|---|
| **Perplexity delta** | how much worse the quant predicts text vs its own F16 parent (lower = better) | **−1.0 %** (6.587 vs 6.654 — statistically identical) |
| **Mean KL divergence** | average "extra surprise" per token vs the parent — 0 = identical | **0.086** (median 0.029) |
| **Same top token** | how often both models pick the exact same next word | **88.8 %** of positions |

**Plain read:** on standard text, the FP4 cut is **statistically indistinguishable from the full-precision model** — no measurable quality loss. The small difference that exists is concentrated in a tail of rare positions (~1 % of tokens shift noticeably), and the chain test above shows those shifts didn't cost task success. **Not lossless in the mathematical sense — lossy in a way we cannot measure on standard text.**

*Reproducible: llama.cpp `llama-perplexity`, engine commit `11bfe8a`, wikitext-2-raw, `-c 8192`, 36 chunks (~295K tokens). Reference F16 converted by us from Qwen's official BF16 release (sha-verified shards). Full data in our results file.*

### Do the speed-ups change the answers? (latest version, 2026-09-28)

Speculative decoding should never change what the model writes: the big model checks every guessed token. We checked it anyway, on 22 fixed short and code questions at temperature 0, against the same engine with **no** speed-up:

| Check | Result |
|---|---|
| Answers that changed | 2–5 of 22, depending on the speed-up tried (Chat's setup: 2, both the same as a fresh no-speed-up run; the no-speed-up setup itself changes a few between runs) |
| Why they changed | every change was a near tie: the model's own second choice, with the top two words 0.0005–0.18 nats apart (rounding in a different-size batch) |
| Changes that pointed to a bug | **0** of 29 changed answers (8 setups tried, including the Chat, Code and Long settings) |
| Long profile's f16 context memory vs the default q8_0 (llama-perplexity, wikitext-2, 12 × 8,192 tokens) | perplexity 6.4384 vs 6.4370 (**+0.02 %**), KL 0.0008, same top word 99.81 % — the same size as the difference from only changing the batch size (99.87 %); with and without the row pad: identical to every digit |

## Deep-context sweep — what happens when the window fills (8K → 256K)

We filled the window in five steps and measured reading and writing at each — a fresh server and a fresh prompt at every point (no prompt-cache reuse), three kinds of answer per depth, one pass, **for each profile**.

![Qwen3.8-27B PaoAI STRIX BALANCED — v4 profiles, speed vs filled context](context-sweep.png)

**Chat profile**

| Filled context | Reading t/s | Writing t/s — short answer | Writing t/s — code | Writing t/s — long writing | Guesses accepted (short · code · long) | Answer check (17×23) |
|---|---|---|---|---|---|---|
| 8K | **404** | **35.5** | **42.4** | **36.9** | 0.61 · 0.78 · 0.65 | ✓ |
| 32K | **321** | **37.2** | **38.0** | **27.1** | 0.76 · 0.78 · 0.49 | ✓ |
| 64K | **239** | **31.3** | **34.1** | **27.0** | 0.71 · 0.79 · 0.58 | ✓ |
| 128K | **160** | **23.7** | **28.0** | **20.0** | 0.63 · 0.80 · 0.50 | ✓ |
| 256K | **101** | **18.2** | **20.3** | **16.1** | 0.67 · 0.79 · 0.58 | ✓ |

**Code profile**

| Filled context | Reading t/s | Writing t/s — short answer | Writing t/s — code | Writing t/s — long writing | Guesses accepted (short · code · long) | Answer check (17×23) |
|---|---|---|---|---|---|---|
| 8K | **399** | **43.8** | **49.7** | **28.6** | 0.77 · 0.78 · 0.52 | ✓ |
| 32K | **323** | **36.6** | **41.0** | **24.9** | 0.71 · 0.75 · 0.51 | ✓ |
| 64K | **255** | **30.0** | **39.9** | **30.2** | 0.67 · 0.79 · 0.66 | ✓ |
| 128K | **181** | **28.8** | **30.9** | **19.5** | 0.76 · 0.75 · 0.52 | ✓ |
| 256K | **115** | **17.7** | **25.0** | **16.1** | 0.63 · 0.79 · 0.56 | ✓ |

**Long profile**

| Filled context | Reading t/s | Writing t/s — short answer | Writing t/s — code | Writing t/s — long writing | Guesses accepted (short · code · long) | Answer check (17×23) |
|---|---|---|---|---|---|---|
| 8K | **410** | **49.8** | **48.9** | **34.8** | 0.85 · 0.77 · 0.63 | ✓ |
| 32K | **357** | **44.0** | **45.0** | **27.6** | 0.82 · 0.78 · 0.56 | ✓ |
| 64K | **301** | **30.0** | **38.5** | **23.1** | 0.66 · 0.75 · 0.52 | ✓ |
| 128K | **230** | **27.4** | **33.3** | **26.4** | 0.71 · 0.78 · 0.67 | ✓ |
| 256K | **152** | **22.2** | **25.8** | **14.8** | 0.76 · 0.79 · 0.52 | ✓ |

*The three answer kinds:* **short answer** = "What is 17 times 23 plus N? Answer with the number only." (up to 300 tokens, thinking on — the method of every earlier table on this card; the answer is checked); **code** = "write a complete Python module … with tests" and **long writing** = "write a long, detailed explanation … 900+ words", each exactly 1,000 tokens, thinking off. Reading = the short-answer prompt.
*Measured 2026-09-29 on one Strix Halo, latest version (binary built from `8ddb7c4d3`; the branch tip `e9b93d56a` only adds README fixes), window `-c` 73,728 up to 64K, 139,264 at 128K, 262,144 at 256K (the window size does not change the speed at a given fill), prompt cache off, `-ub 256` at 256K. Achieved fills: 8,238 / 32,813 / 65,594 / 131,127 / 260,056 tokens. Speeds are the server's own timings.*

**Against the previous version** (engine `4905ff417`, the settings on this card before, short-answer method — the only kind the old table has):

| Filled context | Reading t/s: previous → Chat · Code · Long | Writing t/s: previous → Chat · Code · Long |
|---|---|---|
| 8K | 294.6 → **404** · **399** · **410** | 26.4 → **35.5** · **43.8** · **49.8** |
| 32K | 246.9 → **321** · **323** · **357** | 32.8 → **37.2** · **36.6** · **44.0** |
| 64K | 186.4 → **239** · **255** · **301** | 21.3 → **31.3** · **30.0** · **30.0** |
| 128K | 122.3 → **160** · **181** · **230** | 17.1 → **23.7** · **28.8** · **27.4** |
| 256K | 79.3 → **101** · **115** · **152** | 11.1 → **18.2** · **17.7** · **22.2** |

*Previous-version numbers: this card's 2026-09-20 table (same short-answer question, one Strix Halo, previous engine and settings). A short answer is a few hundred tokens, so its writing speed moves more from run to run than the 1,000-token rows.*

### Full window — 262,144 tokens (latest version)

The model's own metadata declares a **262,144-token** window (`qwen35.context_length`). A **needle test** passed on every profile: one sentence ("Note: The vault code is 7291-ALPHA.") hidden in the middle of the filled window → asked for the code: Chat `7291-ALPHA` at 260,018 tokens; Code `7291-ALPHA` at 260,018 tokens; Long `7291-ALPHA` at 260,018 tokens. (Run 2026-09-28 on our test build of the same engine code, before the branch was published; every other number in this section is from the published build.)

**Serve it with `-c 262144` for the full window, and add `-ub 256` for prompts past ~200K** — the default batch can lose the Vulkan device there (`vk::DeviceLostError`, reproduced at ~213K); with `-ub 256` all three profiles were clean at 260K.

## Quality bench (PaoAI Quality Bench v1.3 — 55 tasks, 7 subjects, Chat profile)

Overall **83.2** (code counts ×2) · one run per task · temperature 0 · one request at a time · 87.1 min total · latest version.

| Subject | Tasks | Score | Reading t/s | Writing t/s, median (min–max) | Draft guesses accepted |
|---|---|---|---|---|---|
| Code | 22 | 79.0 | 186 *(short prompt)* | 33.4 (30.8–47.0) | 57 % |
| Long context (131K-token book) | 1 | 100.0 | 161 | 26.6 (26.6–26.6) | 74 % |
| Instructions | 11 | 90.9 | 106 *(short prompt)* | 36.0 (30.0–47.4) | 54 % |
| Math | 6 | 50.0 | 155 *(short prompt)* | 36.7 (31.5–41.0) | 57 % |
| Sanity | 3 | 100.0 | 65 *(short prompt)* | 37.5 (36.7–49.3) | 77 % |
| Creative | 6 | 100.0 | 120 *(short prompt)* | 38.7 (33.5–42.2) | 59 % |
| Real world | 6 | 66.7 | 228 *(short prompt)* | 33.6 (31.0–36.3) | 54 % |

> **Short prompt:** most tasks send 60–270 tokens, so their reading speed is mostly start-up time; real reading speed = the long-context row and the sweep above.
>
> **Out of budget:** this model thinks at length. On 6 of 55 tasks it used its whole token budget (code 2, math 2, realworld 2) and scored 0 — still thinking, or its answer was cut off. That is how the model behaves at these budgets (math 2,048 · real world 8,192 · code 12,288–16,384 tokens), not a crash. Same budgets for every model we test, no retries.
>
> **Settings:** Chat profile, thinking budget 12,000 (the bench standard; the rest of this card uses 1,500), `-c 262144`, latest version.

## Facts

| Metric | Result |
|---|---|
| File | `Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf` — **15.01 GB (13.98 GiB)**, 15,013,963,328 B, single file, 4.40 bits per weight (27.3 B weights) |
| sha256 | `3727594804afef8140165e36c0919c70a98b43751a3153d43dd10385c1f100fd` |
| Helper (Code and Long profiles) | `Qwen3.8-27B-DFlash2-Q8_0.gguf` from z-lab, 2.06 GB, sha256 `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |
| Word list (Chat profile) | `paoai/draft-vocab-v2-65536.bin` in the engine repo, 262,144 B, sha256 `fc5ae6310e92489fc2394bfbc18262151ec6349bc430282f2bbd086928d2e703` |
| Load | full GPU, no offload flags. System memory still free after loading at the full 262,144 window: Chat ~93 GiB · Code ~91 GiB · Long ~82 GiB (of ~121 GiB free before loading, 128 GB box) |
| Context | **full 262,144-token window served and checked on every profile** (5/5 depths + needle at 260K) |
| Weights | **unchanged by this update** — same sha256 as the original publish |

## Requirements — which llama.cpp build (latest version)

This model uses two things stock llama.cpp does not have: the ROCmFP4 tensor types (types 100 and 101) and, for the Chat profile, the draft word list switch. The `qwen35` architecture itself is in upstream llama.cpp.

- **Latest version:** [guevae2/paoai-strix-engine, branch `strix-v4`](https://github.com/guevae2/paoai-strix-engine/tree/strix-v4) @ `e9b93d56a` (`e9b93d56a30ea24001ac4380209c6daca6df9a66`). The speed numbers were measured on a build of `8ddb7c4d3`; the later commits change only the README. It is our `strix-main` (`afd1088e8`, the engine behind the previous version, a fork of [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp)) plus three changes: `LLAMA_FA_VERIFY_PACK` (faster check of the guesses — **on by default** in this build, `=0` turns it off), `LLAMA_MTP_DRAFT_VOCAB` (the word list — off unless set) and `GGML_VK_FA_STAGING` (experimental, off unless set, no profile uses it). `LLAMA_KV_ROW_PAD` was already in `strix-main`.
- **Previous version:** the same repo, branch `strix-main` @ `4905ff417` — the build this card's earlier numbers were measured on. This file runs on it unchanged (the Chat profile, minus the word list and the faster check).

| Build | What you see |
|---|---|
| upstream llama.cpp (no ROCmFP4 types) | `tensor 'output.weight' has invalid ggml type 101. should be in [0, 43)` |
| a build without the word-list switch (e.g. `strix-main`) | the Chat profile still runs; the log lacks `MTP draft word list ON` and writing is slower |

## How to run (latest version, measured)

**What you need:** an AMD Strix Halo PC (Ryzen AI Max+ 395, Radeon 8060S), running **Linux** (this engine has no Windows build), and
about **20 GB** of free disk (model 15.0 GB + helper 2.1 GB + engine build 0.6 GB). Memory: at the full 262,144 window the server
used about 28 GiB (Chat), 30 GiB (Code) and 40 GiB (Long) — measured on a 128 GB machine; we have not tested a 64 GB one.

**1. Install the tools** (once).

Fedora:

```bash
sudo dnf install -y ca-certificates cmake curl gcc gcc-c++ git glslc python3 spirv-headers-devel vulkan-loader-devel \
  mesa-vulkan-drivers
```

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install -y build-essential ca-certificates cmake curl git glslc libvulkan-dev spirv-headers \
  mesa-vulkan-drivers python3-venv
```

Then, on both:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash     # the hf download tool
export PATH="$HOME/.local/bin:$PATH"               # so this terminal finds hf (new terminals do it by themselves)
```

**2. Build the engine** (about 10 minutes, no sudo):

```bash
git clone -b strix-v4 https://github.com/guevae2/paoai-strix-engine
cd paoai-strix-engine
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-server
cd ..
```

It ends with `[100%] Built target llama-server`.

**3. Get the files** (anonymous, no login needed). The helper is only needed for the Code and Long profiles:

```bash
hf download PaoAI/Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED-GGUF Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf --local-dir .
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q8_0.gguf --local-dir .
```

**4. Run it** (from the folder with the files; pick ONE profile):

Chat (the default — chat, writing, mixed use):

```bash
LLAMA_FA_VERIFY_PACK=1 LLAMA_MTP_DRAFT_VOCAB=paoai-strix-engine/paoai/draft-vocab-v2-65536.bin \
paoai-strix-engine/build/bin/llama-server -m Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf \
  -ngl 99 -fa on -c 262144 --jinja --reasoning-budget 1500 \
  -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 4 \
  --host 0.0.0.0 --port 8080
```

Code (coding agents, up to ~64K context):

```bash
LLAMA_FA_VERIFY_PACK=1 \
paoai-strix-engine/build/bin/llama-server -m Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf \
  -ngl 99 -fa on -c 262144 --jinja --reasoning-budget 1500 \
  -ctk q8_0 -ctv q8_0 -md Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-p-min 0.10 --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 \
  --host 0.0.0.0 --port 8080
```

Long (long documents and long code, 64K+):

```bash
LLAMA_FA_VERIFY_PACK=1 LLAMA_KV_ROW_PAD=256 \
paoai-strix-engine/build/bin/llama-server -m Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf \
  -ngl 99 -fa on -c 262144 --jinja --reasoning-budget 1500 \
  -ctk f16 -ctv f16 -md Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-p-min 0.10 --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7 \
  --host 0.0.0.0 --port 8080
```

After about 20 seconds the log says `listening on http://0.0.0.0:8080`. Then open **http://localhost:8080** in a browser for the
built-in chat page, or point any OpenAI-compatible app at `http://localhost:8080/v1`. The server also answers other computers
on your network (that is what `--host 0.0.0.0` does); use `--host 127.0.0.1` to keep it to this PC only.

**If something goes wrong**

| You see | Do this |
|---|---|
| `Could NOT find Vulkan` / `glslc` / `SPIRV-Headers` during the build | step 1 was skipped or failed — run your system's line again |
| `hf: command not found` | run the `export PATH=…` line from step 1, or open a new terminal |
| `The virtual environment was not created successfully` (Ubuntu) | install `python3-venv` (it is in the Ubuntu line of step 1), then run the `hf` installer line again |
| no GPU found (`--list-devices` shows no `Vulkan0`) | install `mesa-vulkan-drivers`; on a remote login add yourself to the `render` and `video` groups |
| `vk::DeviceLostError` with a very long prompt | add `-ub 256` (needed past ~200K tokens) |
| `Warning: You are sending unauthenticated requests to the HF Hub` | normal — the files are public; no login needed |

*Steps 1–4 were run exactly as printed, on 2026-09-29, on a clean **Fedora 44** and a clean **Ubuntu 24.04** (fresh systems, a normal user, the GPU passed in): both built the engine, downloaded both files with no login (checksums matched), served all three profiles, and answered correctly.*

**Optional: check the files** (the sums must match):

```bash
sha256sum Qwen3.8-27B-PaoAI-ROCmFP4-STRIX-BALANCED.gguf   # 3727594804afef8140165e36c0919c70a98b43751a3153d43dd10385c1f100fd
sha256sum Qwen3.8-27B-DFlash2-Q8_0.gguf                   # c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e
```

## Known limits (honest)

- **Depth costs speed:** reading and writing slow down as the window fills (tables above). The 16 full-attention layers re-read the history for every token.
- **The Code helper is for code:** on long free text at 8K–32K it writes slower than Chat (8K: 28.6 vs 36.9 t/s). Pick Chat for chat and writing, Code for code, Long for long inputs.
- **The model sometimes writes `notify_one()`** (a C++/Java name) in Python threading code — see the coding exam. The full-precision model does it too; check threaded Python it writes.
- **Vulkan only:** every number here is from the gfx1151 Vulkan backend on a Strix Halo iGPU. CPU builds and other GPUs are not comparable.
- **Long thinking:** a bigger `--reasoning-budget` means noticeably longer waits before the answer starts.
- **Not adopted, and why:** adaptive drafting on the built-in draft head (measured slower); wide batches `-ub 2048` (no gain, and it lost the GPU device at depth).

## Serving features explained (plain words)

Every switch in the run commands above, from this build's own `llama-server --help` and source:

| Setting | What it does |
|---|---|
| `LLAMA_FA_VERIFY_PACK=1` | when the guessed tokens are checked, reads the context memory once for the whole batch instead of once per token (on by default in `strix-v4`; written out so the command is explicit) |
| `LLAMA_MTP_DRAFT_VOCAB=…bin` | the built-in draft head only scores the 65,536 most-used tokens, so each guess is cheaper; the model still checks every guess against its full vocabulary |
| `LLAMA_KV_ROW_PAD=256` | lays the f16 context memory out so reads spread over all memory channels (without it f16 reads 2–3× slower at 64K+) |
| `--spec-type draft-mtp --spec-draft-n-max 4` | the model's own draft head guesses 4 tokens ahead; the model checks them in one pass |
| `-md … --spec-type draft-dflash … --spec-draft-adaptive --spec-draft-n-min 3 --spec-draft-n-max 7` | the helper model guesses 3–7 tokens at a time, more when its guesses are being accepted |
| `--spec-draft-p-min 0.10` | the helper does not offer guesses it is less than 10 % sure of |
| `-ctk/-ctv q8_0` vs `f16` | context memory in 8-bit (half the size) or 16-bit (bigger, faster to read at depth with the row pad) |
| `-fa on` | flash attention — faster reading and writing |
| `--reasoning-budget 1500` | caps thinking before the answer; `-1` = unlimited, `4000` = deeper, `12000` = deep work (long waits) — every number here uses 1500 except the quality bench (12000) |
| `-c 262144` | the full window; a smaller `-c` uses less memory |
| `-ngl 99` / `-ngld 99` | put every layer of the model / the helper on the GPU |
| `--jinja` | use the model's own chat template (on by default; written out so the command is explicit) |
| `-ub 256` | smaller processing batches — needed only for prompts past ~200K tokens (see Full window) |
| prompt caching (on by default) | a follow-up turn re-uses the part of the conversation already read; our speed tables turn it off so every point reads its whole prompt |

## Update log

| Date | What changed |
|---|---|
| **2026-09-29** | New engine version [`strix-v4`](https://github.com/guevae2/paoai-strix-engine/tree/strix-v4) (`e9b93d56a`) with three serving profiles (Chat, Code, Long). Re-measured on the PaoAI Card Test v1: depth sweep 8K→256K × 3 answer kinds × 3 profiles, needle at 260K × 3, coding exam × 3 per profile, quality bench v1.3; answer checks and perplexity for the new settings; build + download + run steps tested from a clean Fedora 44 and Ubuntu 24.04. Recipe table corrected to what the file actually contains (read from its tensor table). Weights unchanged — same sha256. |
| **2026-09-20** | Re-swept 8K→256K on the latest version (paoai-strix-engine `4905ff417`): first full-window 256K integrity pass (261,519 tokens), decode faster at every depth from 32K up, chart regenerated. Weights unchanged — same sha256. |
| **2026-09-18** | Added quant-loss measurement (perplexity + KL vs own F16 parent): no measurable loss. Card text only — weights untouched. |
| **2026-09-18** | Changed the recommended engine to our fork [paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine) (`6d1d80e3f`), a fork of `11bfe8a` with a few of our patches. Re-ran the battery on it: same speed, same quality. Weights untouched. |
| **2026-09-17** | Numbers re-measured on the STRIX Vulkan fork (`11bfe8a`). Prefill +11–33 % at matching fills; quality unchanged (3 runs 100/100/100); 128K holds with integrity; 192K bonus row. Weights untouched — same sha256, same 15.01 GB. |
| 2026-09-16 | Model published: 15.01 GB single file, MTP draft head, sweep measured on the ROCmFPX engine. |

Earlier engine numbers are not repeated here — they were measured on a different build; the tables above are the current measurement. If you reproduce a number, quote the engine commit with it.

## Fleet recipes

Serving configs and our benchmark battery: **[paoai-strix-recipes](https://github.com/guevae2/paoai-strix-recipes)** (GitHub).

## Credits & license

- **Qwen team (Alibaba)** — base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **ggml-org** — [llama.cpp](https://github.com/ggml-org/llama.cpp): the engine, the quant formats, the Vulkan backend, DFlash and adaptive drafting
- **Laurent Zuijdwijk** — [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) (`master` @ `11bfe8a`): the Strix-Halo-tuned Vulkan fork paoai-strix-engine is built on, including the adaptive draft length the Code and Long profiles use (written with his coding agent [agention-marshall](https://github.com/LaurentZuijdwijk/agention-marshall))
- **nathanw** — [Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp): ROCmFP4 types, qwen3.5/MTP work, and the coopmat GEMM rewrite ([`e68dd31db`](https://github.com/Nathanw1014/llama.cpp/commit/e68dd31db)) carried in this engine
- **kingjones30 / charlie12345** — [kingjones30/ROCmFPX](https://github.com/kingjones30/ROCmFPX) · [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX): ROCmFPX format & recipes
- **vcruz305** — [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp): converter branch work (qwen3_5 support path)
- **z-lab** — [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) (Apache-2.0): the DFlash2 helper model the Code and Long profiles use
- **FR-Spec authors** — [FR-Spec: frequency-ranked speculative sampling](https://arxiv.org/abs/2502.14856), and **Shreesh-Sree** — [llama.cpp PR #29143](https://github.com/ggml-org/llama.cpp/pull/29143) (draft-vocabulary trim for qwen35 MTP): the idea behind the Chat profile's draft word list
- **Unsloth** — [huggingface.co/unsloth](https://huggingface.co/unsloth): imatrix methodology reference
- **deepu105** — [Qwen3.8-27B benchmarks on Strix Halo](https://www.reddit.com/r/StrixHalo/comments/1vsstxm/qwen3827b_benchmarks_on_strix_halo_q8_0_mtp_21_ts/) · [27B and Flash Next on 128GB Strix Halo](https://www.reddit.com/r/StrixHalo/comments/1wd7lfx/qwen_38_27b_and_flash_next_on_128gb_strix_halo/) (r/StrixHalo): same-box Strix Halo proof and serving-config intel
- **antirez** — [gguf-tools](https://github.com/antirez/gguf-tools): GGUF layout reference
- **PaoAI** — [huggingface.co/PaoAI](https://huggingface.co/PaoAI) · [paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine) · [paoai-strix-recipes](https://github.com/guevae2/paoai-strix-recipes): the STRIX recipe (per-tensor cut/keep table), imatrix-guided builds, the three profiles, the draft word list, the KV row pad, sweep benchmarking, chain test

Apache-2.0, same as the base model. Not affiliated with Alibaba or the Qwen team.
