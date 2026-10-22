---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
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
- gfx1151
- qwen3
- moe
- mixture-of-experts
- speculative-decoding
pretty_name: Qwen3.8-Flash-Next PaoAI STRIX BALANCED-2 (GGUF)
---


# Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF

**The balanced Qwen3.8-Flash-Next quant for AMD Strix Halo (Ryzen AI Max+ 395 / gfx1151).** Measured, not marketed: one 78.27 GB file on ONE Strix Halo box — **reading 301–870 t/s and writing 17.5–46.7 t/s** across the full window (depending on how full it is and what you ask for), MTP speculative decoding active, and **integrity-checked at every depth to the full 262,144-token window**.

*Reading = processing your prompt (prefill). Writing = producing the answer (decode). t/s = tokens per second.*

**Brand: PaoAI** · community quant, not affiliated with the Qwen team.

> **Updated 2026-09-27 — new engine + new file.** The latest version runs on a **ROCm/HIP engine** (pwilkin's Strix Halo
> llama.cpp branch + one PaoAI fix) with a new file, **`…BALANCED-2.1.gguf`** — the same recipe with the dense weights stored in
> 8-bit (78.27 GB, perplexity −0.16 % vs the previous file). Against the previous version: **reading 2.0–2.6× faster at every
> depth, writing faster at every depth for every kind of answer**, coding-exam median **100** (was 86.7). The previous file stays
> in this repo for the previous (Vulkan) engine — see "Previous version" under How to run.

> **Known issue on PaoAI engine v1 and older llama.cpp:** rarely, a chat can stop mid-answer; press regenerate. Fixed in llama.cpp
> (PR #29161) and in the new PaoAI engine 1.0.
>
> **Newer version (recommended):** [Qwen3.8-Flash-Next Q2-1.0 and Q4-1.0](https://huggingface.co/PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-GGUF)
> on PaoAI engine 1.0, a ready-made download with this fix. New users: start there.

---

## What is this? (30 seconds)

We took the 180B-parameter BF16 Qwen3.8-Flash-Next master (360 GB — 48 blocks, 512 experts, hybrid linear attention, a PLE n-gram table and a NextN/MTP head) and compressed it **organ by organ**: the parts that need precision keep it, the parts that don't get compressed harder. The BALANCED-2 mix fits one box with no offload flags, and pairs with an MTP draft sidecar that makes speculative decoding actually pay.

## The recipe (what's inside)

| Model part | What it does | BALANCED-2.1 (latest) | BALANCED-2 (previous) |
|---|---|---|---|
| 👁 Dense weights — attention, shared expert, linear-attention output, layer-mixing (hyper-connection) weights | reads your prompt, tracks context | **Q8_0** | F16 |
| 🧠 Experts gate/up (512 experts) | the knowledge (most of the size) | IQ2_XXS (imatrix) | same |
|  Experts down | expert down-projection (shape-forced) | IQ4_NL | same |
| 📡 PLE n-gram table | the big lookup organ | IQ4_NL | same |
|  Output head | picks the next word | **Q8_0** | F16 |
|  Token embeddings, norms, router, attention indexer, small gates | keep signals correct | F32 / F16 / BF16 | same |
| ✍️ MTP draft head | speculative guessing — **separate sidecar file** | Q8_0 (unsloth's public sidecar) | Q8_0 |

**BALANCED-2.1** = BALANCED-2 with 423 tensors (the dense weights + output head) moved from F16 to 8-bit Q8_0; the experts and the PLE table are
byte-identical to the previous file. 80.78 → **78.27 GB** (3.65 → 3.54 bits per weight). Perplexity on wikitext-2 (8K context):
4.6695 → **4.6618 (−0.16 %, within error)**. The 8-bit-dense idea comes from Halogen.

**Which file?** `…BALANCED-2.1.gguf` (78.27 GB) is the recommended file — same mix, always-on parts in 8-bit: smaller, same quality, faster. `…BALANCED-2.gguf` (80.78 GB) is the previous file, kept for existing links and for the numbers labelled "previous version". BALANCED-2.1 also runs on the previous (Vulkan) engine.

## Coding exam — "chain test" (3 runs, fresh server each — latest version, 2026-09-26)

A fixed coding exam graded by **running the code**, not by opinion: the model reads deceptive code and must trace it, implements a novel state machine, and writes a thread-safe ring buffer that survives a 5,000-operation stress test. **Three identical runs; we report medians with the spread** (identical runs vary — single scores are unreliable).

| Task (what it measures) | Median of 3 | Runs |
|---|---|---|
| **Automaton** — implement novel rules + exact state trace | **100** | 100 · 60 · 100 |
| **Hunt** — read lying code: trace it + find the planted bug | **100** | 100 · 100 · 100 |
| **Ring buffer** — multi-threaded code under stress | **100** | 100 · 100 · 100 |

**Per-task median: 100/100** (run totals 100 · 87 · 100). The one 60: the model wrote the correct final state but left out the
`simulate()` function the task asks for — its own slip; the answer finished normally (not cut off).
Previous version (2026-09-20): medians 100 · 60 · 100 → **86.7** (run totals 87 · 53 · 87).

### Writing speed per phase (tokens/second)

| Phase | previous version | **latest version** |
|---|---|---|
| Spec (planning) | 16.8 – 19.8 | **23.7 – 29.1** |
| Hunt (analysis) | 20.8 – 24.7 | **33.5 – 38.1** |
| Automaton (code writing) | 23.1 – 27.9 | **39.9 – 40.7** |
| Ring buffer (code writing) | 18.6 – 20.3 | **27.3 – 30.8** |
| Publish (long output) | 27.1 – 29.1 | **29.0 – 38.6** |

*Short prompts (a few hundred tokens). Exam settings for both versions: 64K window, thinking budget 1,500 tokens, 4 drafted tokens, each version on its own recommended server settings.*

## Deep-context sweep — what happens when the window fills (8K → 256K)

We filled the window in five steps and measured reading and writing at each — a fresh server and a fresh prompt at every point (no prompt-cache reuse), three kinds of answer per depth, one pass.

![Qwen3.8-Flash-Next STRIX BALANCED-2.1 — writing, reading and MTP guess rate vs filled context](context-sweep.png)

| Filled context | Reading t/s | Writing t/s — short answer | Writing t/s — code | Writing t/s — long writing | Draft guesses accepted (short · code · long) | Answer check (17×23) | GPU memory (GTT) |
|---|---|---|---|---|---|---|---|
| 8K | **857** | **38.3** | **46.7** | **30.1** | 0.85 · 0.85 · 0.48 | ✓ | 67.9 GiB |
| 32K | **870** | **41.0** | **42.9** | **29.9** | 0.80 · 0.79 · 0.49 | ✓ | 69.5 GiB |
| 64K | **710** | **37.3** | **38.3** | **26.6** | 0.78 · 0.76 · 0.46 | ✓ | 69.5 GiB |
| 128K | **485** | **34.2** | **34.2** | **22.9** | 0.83 · 0.80 · 0.46 | ✓ | 69.5 GiB |
| **256K** | **301** | **25.2** | **27.7** | **17.5** | 0.76 · 0.83 · 0.45 | ✓ | 69.5 GiB |

*The three answer kinds:* **short answer** = "What is 17 times 23 plus N? Answer with the number only." (up to 300 tokens,
thinking on — the method of every earlier table on this card; the answer is checked); **code** = "write a complete Python module …
with tests" and **long writing** = "write a long, detailed explanation … 900+ words", each exactly 1,000 tokens, thinking off.
Reading = the short-answer prompt.
*Measured 2026-09-26 on one Strix Halo, latest version, `-c 262144`, prompt cache off. Achieved fills: 8,238 / 32,813 / 65,594 /
131,127 / 260,056 tokens. Speeds are the server's own timings, cross-checked against its log lines (all 15 match).*

**Against the previous version** (Vulkan engine `4905ff417` + the previous file):

| Filled context | Reading t/s | Writing — short answer | Writing — code | Writing — long writing |
|---|---|---|---|---|
| 8K | 431 → **857** | 33.2 → **38.3** | 35.9 → **46.7** | 22.1 → **30.1** |
| 32K | 384 → **870** | 19.3 → **41.0** | 24.2 → **42.9** | 21.8 → **29.9** |
| 64K | 294 → **710** | 26.6 → **37.3** | 27.2 → **38.3** | 17.9 → **26.6** |
| 128K | 195 → **485** | 28.2 → **34.2** | 22.3 → **34.2** | 17.1 → **22.9** |
| 256K | 117 → **301** | 19.1 → **25.2** | 19.8 → **27.7** | 15.6 → **17.5** |

*Previous-version numbers: short answer = this card's 2026-09-20 table; code and long writing = our measurements of that engine on
2026-09-25/26 (medians of 2–6 runs). All were measured while that box's drive was nearly full, which slows this model — some may be
a little low.*

**Plain read:** reading is **2.0–2.6× faster** at every depth, and the gap grows as the window fills. Writing is faster in every cell.
Code writes fastest because the draft head guesses code well (0.76–0.85 of its guesses accepted); long free text writes slowest
(~0.46 — prose is harder to guess). All five answer checks are right; memory barely moves with depth (67.9 → 69.5 GiB).

### Full window — 262,144 tokens (latest version)

The model's own metadata declares a **262,144-token** window (`qwen4exp.context_length`). The table above fills it to 260,056
tokens with the answer check right, and a **needle test** passed at the same depth: one sentence ("Note: The vault code is
7291-ALPHA.") hidden in the middle of **260,018 tokens** of text → asked for the code, the model answered `7291-ALPHA`.

**Serve it with `-c 262144` for the full window.** The latest version needs no special batch settings for deep context.

## Quality bench (PaoAI Quality Bench v1.3 — 55 tasks, 7 subjects)

Overall **75.7** (code counts ×2) · one run per task · temperature 0 · one request at a time · 81.8 min total · latest version.

| Subject | Tasks | Score | Reading t/s | Writing t/s, median (min–max) | Draft guesses accepted |
|---|---|---|---|---|---|
| Code | 22 | 74.1 | 165 *(short prompt)* | 39.0 (34.7–46.9) | 59 % |
| Long context (131K-token book) | 1 | 100.0 | **475** | 35.7 | 84 % |
| Instructions | 11 | 90.9 | 114 *(short prompt)* | 42.6 (38.4–46.8) | 66 % |
| Math | 6 | 66.7 | 152 *(short prompt)* | 43.8 (38.8–46.4) | 65 % |
| Sanity | 3 | 100.0 | 71 *(short prompt)* | 41.1 (35.7–53.1) | 67 % |
| Creative | 6 | 83.3 | 126 *(short prompt)* | 42.8 (40.0–46.1) | 65 % |
| Real world | 6 | 16.7 | 196 *(short prompt)* | 36.8 (32.8–40.8) | 56 % |

> **Short prompt:** most tasks send 60–270 tokens, so their reading speed is mostly start-up time; real reading speed = the
> long-context row and the sweep above.
>
> **Out of budget:** this model thinks at length. On 10 of 55 tasks it used its whole token budget (math 2,048 · real world 8,192 ·
> code 12,288–16,384): 6 were still thinking and gave no answer, 4 wrote an answer that was cut off — all score 0. That is how the
> model behaves at these budgets, not a crash. Real world: 4 of its 6 tasks ran out of budget, 1 answered wrong, 1 right.
> Same budgets for every run, no retries.

## Facts

| Metric | Result |
|---|---|
| File | `Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf` — **78.27 GB** (78,271,751,360 B), single file, 3.54 bits per weight |
| sha256 | `d3257e4fa5a243fcc7e1a7f4f1e99e6ba5c7094f70a00fbb5c3ff262ae7aff86` |
| Draft sidecar | unsloth `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 4.14 GB (separate download, see How to run) |
| Load | full GPU, no offload flags; **67.9–69.5 GiB GPU memory (GTT)** across the full window |
| Context | **full 262,144-token window served and checked** (5/5 depths) + needle at 260K |
| Served with | ROCm/HIP engine, MTP speculative decoding (4 drafted tokens), flash attention, KV cache f16, batch 16384 |

## Requirements — which build (latest version)

**Engine: [guevae2/paoai-qwen38fn-rocm-engine](https://github.com/guevae2/paoai-qwen38fn-rocm-engine)** — build it with
`scripts/paoai/build-strix-halo.sh` (user space, no sudo). It is
**[pwilkin's llama.cpp, branch `strix-halo`](https://github.com/pwilkin/llama.cpp/tree/strix-halo)** (MIT) at
`b0f31f5876ef3856b55f5bb88072cc96e5effafe`, plus one PaoAI fix `b8fe9e80d5b33a30415d6756a694029bc8c28738` (lets the draft-check
step use a fast GPU routine instead of a slow library path on RDNA3.5), built for HIP with
**[TheRock](https://github.com/ROCm/TheRock) ROCm 10** and **[pwilkin's ROCm runtime](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments)**
at `7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1` (retained-PM4 graphs).

⚠️ **Use a build with a working MTP path.** We measured a wiring defect in other builds of this architecture where the hidden-state export collapses to one row while the draft head reads the whole batch — draft acceptance then sits at **0–2%** instead of the 45–98% a correct build delivers, and speculative decoding becomes a pure slowdown (writing falls to ~10 t/s). This engine measures **0.76–0.85 accepted on short answers and code** and **0.45–0.49 on long free text** — if your numbers look like 2%, you are on the wrong build; swap binaries before blaming the quant.

| Build | What you see |
|---|---|
| a `qwen4exp` build without a working MTP path | loads and runs, but the draft sidecar is accepted at ~2% instead of 45–98% |
| upstream llama.cpp | now has `qwen4exp` (September 2026); **we have not measured this file on it** — check the acceptance number first |

## How to run (latest version, measured)

**What you need:** an AMD Strix Halo PC (Ryzen AI Max+ 395) with **128 GB** of memory, running **Linux** (this engine has no
Windows build), and about **100 GB** of free disk (model 78.3 GB + draft 4.1 GB + engine build 13 GB).

**1. Install the tools** (once). The package lists are the ones pwilkin's own installer uses for these same build steps.

Fedora:

```bash
sudo dnf install -y ca-certificates cmake curl elfutils-libelf-devel gcc gcc-c++ git libcurl-devel libdrm-devel \
  libglvnd-devel libstdc++-devel libzstd-devel make ninja-build numactl-devel openssl-devel pciutils \
  pkgconf-pkg-config python3 python3-pip python3-devel tar vim-common zlib-devel
```

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install -y build-essential ca-certificates cmake curl git libcurl4-openssl-dev \
  libdrm-dev libdw-dev libelf-dev libgl-dev libnuma-dev libpciaccess-dev libssl-dev libudev-dev libzstd-dev \
  ninja-build pciutils pkg-config python3 python3-pip python3-venv xxd zlib1g-dev
```

Then, on both:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash     # the hf download tool
export PATH="$HOME/.local/bin:$PATH"               # so this terminal finds hf (new terminals do it by themselves)
```

**2. Build the engine** (about 10 minutes plus a 1.8 GB download; no sudo, everything goes into `~/.local/share/paoai-qwen38fn`):

```bash
git clone https://github.com/guevae2/paoai-qwen38fn-rocm-engine
cd paoai-qwen38fn-rocm-engine && scripts/paoai/build-strix-halo.sh && cd ..
```

It ends with `LINK OK` and `BUILD DONE`. What the script does is under Requirements above.

**3. Get the two files** (anonymous, no login needed):

```bash
hf download PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf --local-dir .
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --local-dir .
mv MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf .    # hf keeps the repo's MTP/ folder; the command below expects the file here
```

**4. Run it** (from the folder with the two files):

```bash
source ~/.local/share/paoai-qwen38fn/env.sh   # written by the engine build: library path + the 4 switches explained below
llama-server -m Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf \
  -dev ROCm0 -ngl 999 -fa on -fit off --load-mode none --lazy-mode on-direct \
  -ctk f16 -ctv f16 -c 262144 -b 16384 -ub 16384 --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-model mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --spec-draft-device ROCm0 --spec-draft-ngl 99 --spec-draft-n-max 4 \
  --host 0.0.0.0 --port 8080
```

After about a minute the log says `listening on http://0.0.0.0:8080`. Then open **http://localhost:8080** in a browser for the
built-in chat page, or point any OpenAI-compatible app at `http://localhost:8080/v1`. The server also answers other computers
on your network (that is what `--host 0.0.0.0` does); use `--host 127.0.0.1` to keep it to this PC only.

**If something goes wrong**

| You see | Do this |
|---|---|
| `missing command: ninja` (or git, cmake …), or `Could NOT find OpenGL` / another `Could NOT find …` | step 1 was skipped or failed — run your system's line again |
| `hf: command not found` | run the `export PATH=…` line from step 1, or open a new terminal |
| no GPU found, or `permission denied` on `/dev/kfd` | `sudo usermod -aG render,video $USER`, then log out and back in |
| `Warning: You are sending unauthenticated requests to the HF Hub` | normal — the files are public; no login needed |
| a download stopped halfway (network drop, `Killed`) | run the same line again; it starts that file over. The broken part stays in `.cache/huggingface/download/` (a `.incomplete` file up to 78 GB): delete it to get the space back |
| draft acceptance near 2 % in the log | wrong engine build (see the warning under Requirements) |

*Steps 1–4 were run exactly as printed, on 2026-09-27, on a clean **Fedora 44** and a clean **Ubuntu 24.04** (fresh systems, a normal user, the GPU passed in): both built the engine, downloaded both files with no login (checksums matched), served, and answered correctly.*

**Optional: check the files** (the sums must match):

```bash
sha256sum Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf   # d3257e4fa5a243fcc7e1a7f4f1e99e6ba5c7094f70a00fbb5c3ff262ae7aff86
sha256sum mtp-Qwen3.8-Flash-Next-Q8_0.gguf                    # cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e
```

- **Draft head:** download `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (4,137,429,120 B, sha256
  `cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e`). Without it the model still serves, just without speculation.
- **What `env.sh` sets** (the engine's build script writes it): the library path to pwilkin's ROCm runtime + the engine's `bin/`,
  `HSA_OVERRIDE_GFX_VERSION=11.5.1` (marks the GPU as gfx1151), `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` (lets it use system memory),
  `ENABLE_RETAINED_PM4=1` + `DEBUG_HIP_GRAPH_PM4=1` (switch on the retained-PM4 graph path of that runtime). Built somewhere else with
  `WORK=<dir>`? Source `<dir>/env.sh` instead.
- **Keep the file on a fast NVMe drive that is not nearly full.** `--lazy-mode on-direct` reads the 26.8 GiB PLE table from the file
  while writing instead of holding it all in memory. On our box, runs wrote up to a third slower while its drive (a DRAM-less SSD)
  was 86 % full, and ran clean at 45 % full.

**Output budget — the one setting that changes answers, not just speed.** This model thinks hard before it writes; the thinking is billed against the same output budget as the answer. Nothing is baked into the file — set it in your client (`max_tokens`), not in the model:

| Value | Behaviour |
|---|---|
| 12,000 | hard tasks can truncate mid-thought and return **no answer text** |
| **32,000** | **recommended on hard tasks** — a ring-buffer prompt that scored 20 at 12K scored 100/100 here (measured on the previous version) |
| 64,000+ | long multi-file work; watch the wall clock (thinking runs at writing speed) |

<details>
<summary><b>Previous version — Vulkan engine + the original 80.78 GB file (still in this repo)</b></summary>

The original file `Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.gguf` (80.78 GB, sha256 `b552c44269035b25a638be75a03043eddeafcd5f52ca4937a30c648d6fcd19b3`) runs on our Vulkan fork **[paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine)** at commit `4905ff417` — every "previous version" number on this card was measured there.

```bash
git clone https://github.com/guevae2/paoai-strix-engine && cd paoai-strix-engine
git checkout 4905ff417
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

llama-server -m Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.gguf \
  --host 0.0.0.0 --port 8080 -ngl 999 -c 262144 --parallel 1 \
  -b 2048 -ub 2048 \
  -fa on --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --ctx-checkpoints 0 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf
```

On that engine: `-b 2048 -ub 2048` is free prefill; **do not raise it** (`-ub 8192` takes the GPU down with `vk::DeviceLostError`); the fork steps the batch down by itself past ~55K and ~200K depth; `--ctx-checkpoints 0` is required (checkpoint restore crashes otherwise).

</details>

## Serving features explained (plain words)

| Flag | What it does | Cost |
|---|---|---|
| `--spec-type draft-mtp --spec-draft-n-max 4` + `--spec-draft-model …` | the draft head guesses 4 tokens ahead and the model checks them in one pass — writing measured 17.5–46.7 t/s across the fill | needs the unsloth Q8_0 sidecar (4.1 GB) |
| `--spec-draft-device ROCm0 --spec-draft-ngl 99` | runs the draft head on the GPU too | none |
| `-dev ROCm0 -ngl 999` | whole model on the GPU (Strix Halo's memory is shared, so it fits) | none |
| `-fa on` | flash attention — faster reading and writing | none |
| `-ctk f16 -ctv f16` | memory for the conversation kept at full precision; 8-bit gave no speed gain on this engine | 67.9–69.5 GiB total at the full window |
| `-b 16384 -ub 16384` | reads your prompt in big 16K-token chunks — the fast reading numbers above | none measured; no step-down needed at 256K |
| `--load-mode none --lazy-mode on-direct` | reads the 26.8 GiB PLE table from the file only as rows are needed, instead of holding it in memory | the drive is read while writing — keep the file on a fast drive (How to run) |
| `-fit off` | uses exactly these settings instead of letting the server shrink them to fit | none |
| `--parallel 1` | one conversation gets the whole 262,144-token window | one user at a time |
| `--jinja` | uses the model's own chat template (thinking on by default) | none |

## Known limits (honest)

- **Long thinking is the real caveat.** The model reasons heavily and the thinking length varies run-to-run (4K → 30K tokens on an identical prompt). With a small output cap you get truncation — mid-thought, sometimes with **no answer text at all**. Give it **32K+ output tokens** on hard tasks.
- **Writing speed depends on how well the draft guesses:** 0.76–0.85 accepted on code and short answers, ~0.46 on long free text —
  so code writes about 1.4–1.6× faster than prose at the same depth.
- **ROCm/HIP numbers:** the latest numbers are from the HIP backend (TheRock ROCm 10) on a Strix Halo iGPU; the previous version's
  are Vulkan. Other backends and boxes are not comparable.
- **The drive matters on this engine** (see How to run): the PLE table is read from the file while writing.
- **Speculative decoding does NOT change your output.** Measured on the previous version's build: the same prompt at temperature 0, run once with the draft head and once without, produced **byte-identical** answers. The draft head buys speed only — you can leave it on.

## Update log

| Date | What changed |
|---|---|
| 2026-09-27 (evening) | How to run is now a from-zero guide: what you need, install the tools (Fedora / Ubuntu), build, download, run, where to open it, and a fix-it table. Every step was run as printed on a clean Fedora 44 and Ubuntu 24.04. Docs only; no number or file changed. |
| 2026-09-27 (later) | How to run: copy-paste download commands for both files + the `source env.sh` step (docs only; no number or file changed). |
| **2026-09-27** | **New engine + new file.** ROCm/HIP engine (pwilkin `strix-halo` `b0f31f58` + PaoAI fix `b8fe9e80`) and `…BALANCED-2.1.gguf` (dense weights + output head F16 → Q8_0, 78.27 GB, perplexity −0.16 %), unsloth draft sidecar. Re-measured: reading 2.0–2.6×, writing faster at every depth, coding-exam median 100. The previous file stays in the repo. |
| 2026-09-20 | Engine fork updated (paoai-strix-engine `4905ff417`): deep-context auto-switch fixed and re-verified 2/2; coding exam re-run: identical per-task medians; weights unchanged. |
| 2026-09-17 | Numbers re-measured on our own build of the current engine (fork commit `50c271f8`, v0.7.6 lineage). |
| 2026-09-16 | Model published: 80.78 GB single file (3.65 bpw), MTP sidecar, full 262,144-token window measured. |

## Fleet recipes

Serving configs, ring self-heal infra, and our benchmark battery: **[paoai-strix-recipes](https://github.com/guevae2/paoai-strix-recipes)** (GitHub).

## Credits & license

- **Qwen team** — base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0)
- **pwilkin (Piotr Wilkin)** — [llama.cpp `strix-halo` branch](https://github.com/pwilkin/llama.cpp/tree/strix-halo) (MIT) and [ROCm runtime](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments) the latest version runs on
- **Unsloth** — [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): BF16 source, imatrix, and the MTP Q8_0 draft sidecar
- **Halogen** — [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server): the 8-bit dense-weights idea behind BALANCED-2.1
- **nathanw** — [Nathanw1014/llama.cpp](https://github.com/Nathanw1014/llama.cpp): `qwen4exp` port, the working MTP hidden-state handoff, and the Vulkan engine fork of the previous version
- **kingjones30 / charlie12345** — [kingjones30/ROCmFPX](https://github.com/kingjones30/ROCmFPX) · [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX): ROCmFPX format & recipes
- **AMD ROCm** — [TheRock](https://github.com/ROCm/TheRock) (ROCm 10 build system)
- **ggml-org** — [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **PaoAI** — [huggingface.co/PaoAI](https://huggingface.co/PaoAI) · [paoai-qwen38fn-rocm-engine](https://github.com/guevae2/paoai-qwen38fn-rocm-engine) · [paoai-strix-engine](https://github.com/guevae2/paoai-strix-engine): hybrid recipe, BALANCED-2.1 re-quant, the RDNA3.5 draft-check fix, fleet benchmarking, coding exam, PaoAI Card Test v1

**Qwen Community License 1.0** (the base model's own license — `license: other`, `qwen-community-1.0`), inherited by this derivative quant. Not affiliated with the Qwen team.
