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
pretty_name: Qwen3.8-Flash-Next PaoAI STRIX (GGUF)
---

# Qwen3.8-Flash-Next-PaoAI-STRIX-GGUF

**Qwen3.8-Flash-Next for AMD Strix Halo (Ryzen AI Max+ 395 / gfx1151), in two sizes, with a ready-made engine.** One file on ONE Strix Halo PC with
128 GB: **Q2-1.0** (78.3 GB) or **Q4-1.0** (102.8 GB). The engine is one container download (`podman pull`); nothing to build.

*Reading = processing your prompt (prefill). Writing = producing the answer (decode). t/s = tokens per second.*

**Brand: PaoAI** · community quant, not affiliated with the Qwen team.

| You want | Download |
|---|---|
| the full 261,888-token context with room to spare, or the smaller download | **Q2-1.0** (78.3 GB, 3.54 bits per weight) |
| answers closest to the original model | **Q4-1.0** (102.8 GB, 4.65 bits per weight) |

**Speed** (both files, 8K to 256K of context, measured): reading your prompt **1,375–1,717 t/s**; writing the answer
**40.3–62.9 t/s**. Every test is under *Q2-1.0 vs Q4-1.0 — measured* below.

## What is this? (30 seconds)

We took the 180B-parameter BF16 Qwen3.8-Flash-Next master and compressed it **part by part**: the parts that need precision keep it (8-bit), the
512 experts that hold most of the size are compressed harder. Both files fit one box with no offload flags and pair with a small draft head
that makes writing faster (speculative decoding). The PaoAI engine 1.0.0 runs both.

### Words on this page

- **Token** — a piece of a word, about three-quarters of an English word on average. Speeds are in tokens per second (t/s).
- **Context** (the chart calls it the *window*) — how much text the model holds at once: your prompt, the chat so far and its answer.
  Both files can hold 261,888 tokens.
- **Reading / writing** — reading = taking in your prompt (prefill); writing = producing the answer (decode).
- **Q2-1.0 / Q4-1.0, bits per weight** — how hard each file is compressed. Fewer bits = a smaller file whose answers are a little further
  from the original.
- **Experts** — this model is a *mixture of experts*: 512 small expert networks, of which only some work on each word. They hold most of
  the size.
- **KL · same top word** — how close a file's next-word choices are to the original model's. Lower KL is closer; *same top word* = how
  often both pick the same next word.
- **Perplexity** — how well the model predicts a standard text (wikitext-2). Lower is better; the original scores 2.0809.
- **Draft head (MTP)** — a small extra file that guesses the next few words; the model checks the guesses, so writing gets faster.
  *MTP guess rate* = the share of guesses it kept.
- **GTT memory** — the part of your system memory the GPU is using.
- **n-gram table (PLE)** — a 26.8 GiB lookup table inside this model that the engine reads from the drive while it writes, so drive
  speed matters.
- **IQ2_XXS, IQ4_NL, Q4_K, Q8_0, F16, F32, BF16** — the compression formats of each part (the recipe table).

## Which file?

**Pick Q2-1.0 if you want…**

- a smaller download (78.3 GB, 24.5 GB less)
- the full 261,888-token context even with a 16 GiB desktop
- coding check 28 / 30
- faster writing: ahead in 12 of 15 writing points (code at every context size)

**Pick Q4-1.0 if you want…**

- closer to the original (KL 0.324 vs 0.551)
- coding chain 100 in all 3 runs (Q2-1.0: 93.3 over 6)
- quality bench 85.2 vs 83.1 (following instructions, math)

| | Q2-1.0 | Q4-1.0 |
|---|---|---|
| Size | 78.3 GB · 3.54 bits per weight | 102.8 GB · 4.65 bits per weight |
| Experts (most of the file) | 2-bit (IQ2_XXS) | 4-bit (Q4_K) |
| Context by default | the full 261,888-token context | a 65,536-token context, so a desktop keeps room to work |
| Closeness to the original (KL, lower is closer) | 0.551 · same top word 78.3 % | **0.324** · same top word 84.8 % |
| Perplexity (wikitext-2, 4K; the original: 2.0809) | 2.8465 (+36.8 %) | **2.4045** (+15.5 %) |

*Closeness and perplexity: measured 2026-10-05 with llama.cpp's perplexity tool against the BF16 original (4 chunks of 4,096 tokens).*

![Speed as the context fills: reading and writing speed of both files from 8K to 256K](context-sweep.png)

## How to run (PaoAI engine 1.0.0)

A ready-made engine for the PaoAI builds of **Qwen3.8-Flash-Next** on AMD Strix Halo, in one container image. You download the
image and one model file, run one line, and chat in your browser. Nothing to build.

**What you need:** an AMD Strix Halo PC (Ryzen AI Max+ 395, GPU gfx1151) with **128 GB** of memory, running **Linux** (there is no
Windows build), and free disk for the model you pick: about **85 GB** for Q2-1.0 or **110 GB** for Q4-1.0 (model + the 4.1 GB draft
head + the engine image, about 1 GB).

**Tested host:** Fedora 44, kernel `7.2.9-200.fc44`, booted with these 3 kernel settings (sized for 128 GB):
`amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856`.

<details markdown="1"><summary>What each setting does</summary>

- `amd_iommu=off` — turns off the memory address-translation layer; we run every test with it off.
- `amdgpu.gttsize=126976` — lets the GPU use up to 124 GiB of the system memory (the number is in MiB).
- `ttm.pages_limit=32505856` — the same 124 GiB limit for the kernel's memory manager (counted in 4 KiB pages).

</details>

**0. Let the GPU use up to 124 GiB of your 128 GB** (once per PC). Linux lets the built-in GPU use only part of the memory unless 3 kernel settings say otherwise. Skip this step if `cat /proc/cmdline` already shows all 3; otherwise run the lines for your system, then restart:

Fedora (kernel settings):
```bash
sudo grubby --update-kernel=ALL --args="amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856"
sudo reboot
```

Ubuntu (kernel settings):
```bash
sudo apt-get -y install linux-generic-hwe-24.04   # the newer kernel; Ubuntu Desktop already has it
sudo mkdir -p /etc/default/grub.d
echo 'GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856"' | sudo tee /etc/default/grub.d/99-paoai-strix.cfg
sudo update-grub
sudo reboot
```

*Ubuntu: AMD supports this GPU on Ubuntu 24.04.5 with its newer 7.0 kernel and on 26.04.1; we tested the GPU itself on Fedora 44,
and these commands on fresh Ubuntu 24.04 virtual PCs.*

<details markdown="1"><summary>Check, undo, and how we tested these lines</summary>

After the restart, `cat /proc/cmdline` shows all 3, and they stay when the kernel updates. To undo, on Fedora run
`sudo grubby --update-kernel=ALL --remove-args="amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856"`, on Ubuntu run
`sudo rm /etc/default/grub.d/99-paoai-strix.cfg` and `sudo update-grub`; then restart. *Tried on fresh Fedora 44 and Ubuntu 24.04
virtual PCs on 2026-10-21 (set twice, restart, a kernel update, undo); the Ubuntu lines with the newer-kernel line again on 2026-10-22. A virtual PC has no AMD graphics chip, so this checks the
commands, not the 124 GiB.*

</details>

**1. Install the tools** (once).

Fedora:
```bash
sudo dnf install -y podman python3
```

Ubuntu / Debian:
```bash
sudo apt update && sudo apt install -y podman curl python3-venv
```

Then, on both:
```bash
curl -LsSf https://hf.co/cli/install.sh | bash     # the hf download tool
export PATH="$HOME/.local/bin:$PATH"               # so this terminal finds hf (new terminals do it by themselves)
```

**2. Get the engine** (one download, no login):
```bash
podman pull ghcr.io/guevae2/paoai-engine:1.0.0
```

**3. Get the files** (anonymous, no login needed). Pick **one** model; both run on the same engine.

Q2-1.0 (78.3 GB; the full 261,888-token context by default):
```bash
hf download PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-GGUF Qwen3.8-Flash-Next-PaoAI-STRIX-Q2-1.0.gguf --local-dir ~/paoai-models
```

Q4-1.0 (102.8 GB; a 65,536-token context by default, so a desktop keeps room to work):
```bash
hf download PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-GGUF Qwen3.8-Flash-Next-PaoAI-STRIX-Q4-1.0.gguf --local-dir ~/paoai-models
```

Then the draft head (it makes writing faster; the engine checks it is the exact file we tested):
```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --local-dir ~/paoai-models
```

**4. Run it:**
```bash
podman run --rm -it --device /dev/kfd --device /dev/dri --group-add keep-groups --ipc=host --ulimit memlock=-1:-1 \
  -v ~/paoai-models:/models:ro,z -p 127.0.0.1:8080:8080 ghcr.io/guevae2/paoai-engine:1.0.0
```

What the flags do: `--device /dev/kfd --device /dev/dri` give the container the GPU; `--group-add keep-groups` keeps your own GPU
permission inside it; `--ipc=host` and `--ulimit memlock=-1:-1` are the container settings the engine is tested with (when podman runs as your own user, `--ulimit` cannot raise your locked-memory limit; the engine is tested at Fedora's default, 8 MiB);
`-v ~/paoai-models:/models:ro,z` shows it your model folder, read-only (`z` lets the container read it on systems with SELinux, such as Fedora; elsewhere it does nothing); `-p 127.0.0.1:8080:8080` serves on this PC only.

It prints `PaoAI engine 1.0.0 · FN engine e42b10d38`, finds your model, and after about a minute the log says
`listening on http://0.0.0.0:8080` (that address is inside the container). Then open **http://localhost:8080** in a browser for the
built-in chat page, or point any OpenAI-compatible app at `http://localhost:8080/v1`. Stop it with Ctrl+C.

**Answer budget:** the built-in chat page sets no limit on how long an answer can be. In other apps, set `max_tokens` to 32,000 for hard tasks: this model thinks before it answers, and a smaller budget can stop it mid-thought (see *Output budget* below).

**5. Check it answers** (in a second terminal):
```bash
curl -s --compressed http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 391*7? Reply with only the number."}],"max_tokens":3000}'
```
The reply's answer is `2737`.

**If something goes wrong**

| You see | Do this |
|---|---|
| `hf: command not found` | run the `export PATH=…` line from step 1, or open a new terminal |
| `permission denied` on `/dev/kfd`, or no GPU found | `sudo usermod -aG render,video $USER`, then log out and back in |
| `no supported model in /models` (exit code 2) | the model file is not in `~/paoai-models`, or its name or size differs: download it again with the step 3 line |
| `several supported models in /models` (exit code 2) | keep one there, or pick one: add `-e PAOAI_MODEL=<file name>` before the image name |
| `draft head differs from the tested file` (exit code 3) | download it again with the step 3 line (it pins the tested revision) |
| `draft head not found: serving without it` | the model still serves, only writing is slower; get the head with the step 3 line |
| `Warning: You are sending unauthenticated requests to the HF Hub` | normal — the files are public; no login needed |
| a download stopped halfway (network drop, `Killed`) | run the same line again; it starts that file over. The broken part stays in `~/paoai-models/.cache/huggingface/download/` (a `.incomplete` file): delete it to get the space back |

To check the GPU, memory settings and kernel in one go:
```bash
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups -v ~/paoai-models:/models:ro,z ghcr.io/guevae2/paoai-engine:1.0.0 doctor
```
It prints PASS, WARN or FAIL for the gfx1151 GPU, the GPU memory size, device access, the model files and the draft head, and the kernel it runs on.

**Settings** (add `-e NAME=value` before the image name):

| Setting | What it does |
|---|---|
| `PAOAI_CTX` | the context length in tokens. On a PC that only serves the model (no desktop session), Q4-1.0 can use `PAOAI_CTX=261888` |
| `PAOAI_API_KEY` | asks every client for this key |
| `PAOAI_MODEL` | which file in `/models` to serve, when there are several |
| `PAOAI_PORT` | the port inside the container (default 8080); change the `-p` line to match |
| `PAOAI_CORS_ORIGINS` | which other web site may call the engine from a browser (default `localhost`: only pages from this PC; the built-in chat page and apps outside a browser are not affected). To use a chat web site at another address, set its origin, for example `PAOAI_CORS_ORIGINS=https://chat.example.com` (one origin); `*` allows every site (then also set `PAOAI_API_KEY`) |
| `PAOAI_LOOP_GUARD` | `0` turns off the loop guard (on by default: when the model's thinking repeats one short piece many times, the engine ends the thinking and the answer follows) |
| `PAOAI_PREREAD` | Q2-1.0 only: `1` turns on a background read that warms the model's 26.8 GiB n-gram table after start (off by default: it made no difference in our tests) |

**Optional: check the files** (the sums must match):
```bash
sha256sum ~/paoai-models/Qwen3.8-Flash-Next-PaoAI-STRIX-Q2-1.0.gguf          # 5afc4e88807e23b7bd48b1a40f402476627bc9a923912f2d9b5bcfc8f737858d
sha256sum ~/paoai-models/Qwen3.8-Flash-Next-PaoAI-STRIX-Q4-1.0.gguf          # 0d491aa8caa206b641af92745c473fca25b437e7f751b18948ee5b70ee88bb03
sha256sum ~/paoai-models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf                # cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e
```
`podman run … ghcr.io/guevae2/paoai-engine:1.0.0 verify` checks the same sums from inside the engine.

**Docker** (not tested by us): the same line with `docker run` and `--group-add video --group-add render` in place of
`--group-add keep-groups`.

**Keep the model on a fast NVMe drive that is not nearly full.** The engine reads the 26.8 GiB n-gram table from the file while it
writes instead of holding it all in memory. On our box, runs wrote up to a third slower while its drive was 86 % full, and ran clean
at 45 % full.

**Output budget — the one setting that changes answers, not just speed.** This model thinks before it writes, and the thinking
counts against the same output budget as the answer. Set `max_tokens` in your client: 12,000 can cut a hard task off mid-thought
with no answer text; **32,000 is our pick for hard tasks**.

## Q2-1.0 vs Q4-1.0 — measured (PaoAI Card Test v2)

*Both files on the same Strix Halo PC with the PaoAI engine 1.0.0. **Ahead** = the file that did better (bold); speeds within 3 % of each
other are "about the same". Reading = processing your prompt; writing = producing the answer.*

### The short version

| Section | Ahead | In plain words |
|---|---|---|
| 1 · The file (the Which file? table above) | Split | Q2-1.0 is 24.5 GB smaller; Q4-1.0 is closer to the original (KL 0.324 vs 0.551). |
| 2 · Reading speed | About the same | 5 of 8 reading points (sections 2 and 4) are within 3 %; Q4-1.0 is faster at 2, Q2-1.0 at 1. |
| 3 · Writing speed | Q2-1.0 | Q2-1.0 writes faster at 12 of 15 points, and at every window size for code. |
| 4 · Long and real text | About the same | Both found the code hidden in a 260,018-token prompt. Q2-1.0 got the 256K sum wrong (a speed row, not scored). |
| 5 · Coding chain | Q4-1.0 | Q4-1.0 scored 100 in all 3 runs; Q2-1.0 passed with 93.3 over 6 runs (automaton is its weak task). |
| 6 · Quality bench | Q4-1.0, small | 85.2 vs 83.1: only 3 of 55 tasks scored differently on the middle of 3 passes. |
| 7 · Coding check | Q2-1.0, small | 28 vs 27 of 30: the files differ on 1 question. |
| 8 · Room for your desktop | Q2-1.0 | With a 16 GiB desktop, Q2-1.0 keeps the full 261,888-token context; Q4-1.0 131,072. |
| 9 · Test health | Both fine | 0 errors, GPU faults or safety stops on either file. |

<details markdown="1"><summary><b>All test results</b> — every test, both files side by side (click to open)</summary>

### 2 · Reading speed (t/s)

*How fast it takes in your prompt — higher is better.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Reading at 8K | 1,375 | **1,517** | Q4-1.0 · 10 % faster |
| Reading at 32K | 1,625 | 1,643 | about the same |
| Reading at 64K | 1,657 | **1,717** | Q4-1.0 · 4 % faster |
| Reading at 128K | 1,655 | 1,700 | about the same |
| Reading at 256K | 1,581 | 1,617 | about the same |

*Q2-1.0's first 8K reading was under the test's bar, so the test read it 4 more times: 1,425–1,459 t/s (still under Q4-1.0).*

### 3 · Writing speed (t/s)

*How fast it answers — higher is better.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Writing code at 8K | **61.8** | 58.4 | Q2-1.0 · 6 % faster |
| Writing code at 32K | **62.9** | 57.8 | Q2-1.0 · 9 % faster |
| Writing code at 64K | **60.0** | 57.7 | Q2-1.0 · 4 % faster |
| Writing code at 128K | **59.4** | 54.2 | Q2-1.0 · 10 % faster |
| Writing code at 256K | **56.0** | 51.2 | Q2-1.0 · 9 % faster |
| Writing long text (prose) at 8K | 40.9 | **42.5** | Q4-1.0 · 4 % faster |
| Writing long text (prose) at 32K | **45.2** | 42.1 | Q2-1.0 · 7 % faster |
| Writing long text (prose) at 64K | **47.0** | 41.6 | Q2-1.0 · 13 % faster |
| Writing long text (prose) at 128K | **47.3** | 40.3 | Q2-1.0 · 17 % faster |
| Writing long text (prose) at 256K | 40.9 | 41.4 | about the same |
| Writing short answer at 8K | 46.3 | **49.4** | Q4-1.0 · 7 % faster |
| Writing short answer at 32K | **59.5** | 55.1 | Q2-1.0 · 8 % faster |
| Writing short answer at 64K | **58.3** | 54.2 | Q2-1.0 · 8 % faster |
| Writing short answer at 128K | **55.8** | 46.5 | Q2-1.0 · 20 % faster |
| Writing short answer at 256K | **52.5** | 48.8 | Q2-1.0 · 8 % faster |

*Most writing, including chat with thinking on, falls between the long-text (prose) and code rows. Very regular output (lists, tables, JSON) can be faster; other languages or a high temperature can be slower.*

**Details per point** (the same runs):

| Filled context | MTP guess rate Q2-1.0 (short · code · writing) | MTP guess rate Q4-1.0 | Sum right? Q2-1.0 · Q4-1.0 | Memory (GTT) Q2-1.0 · Q4-1.0 |
|---|---|---|---|---|
| 8K | 0.76 · 0.86 · 0.55 | 0.83 · 0.87 · 0.55 | ✓ · ✓ | 70.7 GiB · 93.6 GiB |
| 32K | 0.90 · 0.87 · 0.58 | 0.97 · 0.88 · 0.54 | ✓ · ✓ | 72.2 GiB · 95.1 GiB |
| 64K | 0.87 · 0.83 · 0.59 | 0.93 · 0.88 · 0.55 | ✓ · ✓ | 72.2 GiB · 95.1 GiB |
| 128K | 0.86 · 0.85 · 0.61 | 0.76 · 0.83 · 0.54 | ✓ · ✓ | 72.2 GiB · 95.1 GiB |
| 256K | 0.84 · 0.82 · 0.52 | 0.87 · 0.80 · 0.59 | ✓ · ✓ | 72.2 GiB · 95.1 GiB |

*Measured 2026-10-19 (Q2-1.0) and 2026-10-20 (Q4-1.0) on one Strix Halo PC with the PaoAI engine 1.0.0 image, started with the README's `podman run` line; a fresh container and a fresh prompt per point, prompt cache off; one pass. Short answer = a sum question after the filler (300 tokens, thinking on, thinking budget 1,500); code and long text = exactly 1,000 tokens, thinking off. Reading = the short-answer prompt.*

### 4 · Long and real text

*The same checks on both files.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Real book-length text at 32K: reading t/s | 1,664 | 1,661 | about the same |
| Real text at 32K: sum right? | ✓ | ✓ | tie |
| Real book-length text at 128K: reading t/s | 1,654 | 1,664 | about the same |
| Real text at 128K: sum right? | ✓ | ✓ | tie |
| Real book-length text at 256K: reading t/s | **1,537** | 1,467 | Q2-1.0 · 5 % faster |
| Real text at 256K: sum right? | ✗ (264,830, should be 265,130) | ✓ | Q4-1.0 right · not scored |
| Needle in a haystack: a code hidden in a 260,018-token prompt | found | found | tie |

*The needle prompt was read at 1,573 t/s on Q2-1.0 and 1,616 t/s on Q4-1.0.*

*One real-text answer was wrong: at 256K Q2-1.0 answered 264,830 instead of 265,130 (it took 17 × 23 as
91). That row measures reading speed; the answer is shown as it came.*

### 5 · Coding chain

*3 hard tasks, fresh server per run; pass mark 90.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Automaton — new rules + an exact state trace | 80 (runs 100 · 60 · 0 · 100 · 100 · 60) | **100** (runs 100 · 100 · 100) | Q4-1.0 · 20.0 points |
| Hunt — find the planted bug in misleading code | 100 (runs 100 · 40 · 100 · 100 · 100 · 100) | 100 (runs 100 · 100 · 100) | tie |
| Ring buffer — multi-threaded code under stress | 100 (runs 100 · 100 · 100 · 100 · 100 · 100) | 100 (runs 100 · 100 · 100) | tie |
| Chain score (middle of each task, averaged) | 93.3 (6 runs) | **100** (3 runs) | Q4-1.0 · 6.7 points |

*Why 6 coding runs for Q2-1.0: runs 1–3 averaged 86.7, under our pass mark of 90, mostly on one task (automaton) that this file
passes in about half of its runs. We then wrote a rule — 3 more runs, judged on the middle of all 6 — before running them; the
table shows all 6 (93.3). Q4-1.0 ran 3 times.*

### 6 · Quality bench

*55 tasks, 3 passes, the middle score per task; overall = the 7 groups averaged, coding counted twice.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Coding (22 tasks) | **91.5** | 87 | Q2-1.0 · ≈ 1 task |
| Following instructions (11 tasks) | 81.8 | **90.9** | Q4-1.0 · ≈ 1 task |
| Math (6 tasks) | 66.7 | **83.3** | Q4-1.0 · ≈ 1 task |
| Creative writing (6 tasks) | 83.3 | 83.3 | tie |
| Real-world tasks (6 tasks) | 50 | 50 | tie |
| Long context (1 task) | 100 | 100 | tie |
| Sanity checks (3 tasks) | 100 | 100 | tie |
| **Quality bench overall** | 83.1 (passes 83.1 · 83.1 · 83.1) | **85.2** (passes 85.2 · 85.2 · 85.2) | Q4-1.0 · 2.1 points |

*Each group gap above is about one task (0.99, 1.00, 1.00).*

*All 3 passes gave the same score on every task (the bench asks each question at temperature 0).*

### 7 · Coding check

*30 Python questions the model never saw; each answer runs against its own tests (no AI judge).*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Questions passed (all of their own tests) | **28 / 30** | 27 / 30 | Q2-1.0 · 1 more |
| Hard questions passed | **23 / 24** | 22 / 24 | Q2-1.0 · 1 more |
| Finish-with-an-answer questions passed | 5 / 6 | 5 / 6 | tie |
| Single tests passed | **263 / 276** | 259 / 276 | Q2-1.0 · 4 more |
| Writing speed while coding (middle, t/s) | **51.0** | 46.9 | Q2-1.0 · 9 % faster |
| Stuck loops · freezes | 0 · 0 | 0 · 0 | tie |

*The two files differ on 1 question.*

*30 Python questions the model was never trained on; each answer is run against its own tests (no AI judge). Measured 2026-10-18 on the same PC with the engine build just before 1.0.0, which gives the same answers word for word.*

### 8 · Room left for your desktop

*The longest context that still fits while your desktop uses this much memory.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Desktop using 8 GiB | not measured (16 GiB already fits the full window) | 261,888 tokens | — |
| Desktop using 16 GiB | **261,888 tokens** | 131,072 tokens | Q2-1.0 (more room) |
| GPU memory in use at 256K | **72.2 GiB** | 95.1 GiB | Q2-1.0 uses 22.9 GiB less |

### 9 · Test health

*The test stops itself on heat, memory or errors.*

| Test | Q2-1.0 | Q4-1.0 | Ahead |
|---|---|---|---|
| Errors, GPU faults, safety stops | 0 | 0 | tie |

</details>

## The recipe (what's inside, read from each file's header)

| Model part | Q2-1.0 | Q4-1.0 |
|---|---|---|
| 🧠 Experts gate / up (512 experts) — the knowledge, most of the size | IQ2_XXS ×96 | Q4_K ×96 |
| Experts down | IQ4_NL ×48 | IQ4_NL ×48 |
| 📡 PLE n-gram table — the big lookup organ | IQ4_NL ×1 | IQ4_NL ×1 |
| 👁 Attention and linear attention | Q8_0 ×156 / F32 ×108 / F16 ×72 | F32 ×180 / Q8_0 ×156 |
| Shared expert | Q8_0 ×144 / F32 ×48 | Q8_0 ×144 / F32 ×48 |
| Output head — picks the next word | Q8_0 ×1 | Q8_0 ×1 |
| Token embeddings | F16 ×1 | F16 ×1 |
| Router | F32 ×48 | F32 ×48 |
| Norms | F32 ×184 | F32 ×184 |
| Attention indexer | BF16 ×24 | BF16 ×24 |
| Layer-mixing (hyper-connection) weights | Q8_0 ×194 / F16 ×96 | Q8_0 ×194 / F32 ×96 |
| PLE helpers (key, value, conv) | IQ2_XXS ×2 / F16 ×1 | IQ2_XXS ×2 / F16 ×1 |

*Tensor types per part (×N = how many tensors). Besides the experts, the two files differ only in 168 small tensors kept at F16 in Q2-1.0 and at
F32 in Q4-1.0. Both files: 1,224 tensors, 176,943,899,520 weights.*

## Facts

| | Q2-1.0 | Q4-1.0 |
|---|---|---|
| File | `Qwen3.8-Flash-Next-PaoAI-STRIX-Q2-1.0.gguf` | `Qwen3.8-Flash-Next-PaoAI-STRIX-Q4-1.0.gguf` |
| sha256 | `5afc4e88807e23b7bd48b1a40f402476627bc9a923912f2d9b5bcfc8f737858d` | `0d491aa8caa206b641af92745c473fca25b437e7f751b18948ee5b70ee88bb03` |
| Draft head (both) | unsloth `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, sha256 `cd87e5d1a4dadaeed63e35929f3b2f28d13e081b4cd32e00f2835095ec09351e` (pinned in How to run) | same |

## Known limits (honest)

- **Long thinking is the real caveat.** The model thinks before it answers and the thinking counts against the output budget: set `max_tokens`
  to 32,000 for hard tasks, or a hard task can stop mid-thought with no answer.
- **Writing speed depends on the kind of text:** code writes faster than long prose at the same depth (see the tables).
- **The drive matters:** keep the model on a fast NVMe drive that is not nearly full (see How to run).
- **Linux only;** no Windows build of the engine.

## Credits & license

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — **Qwen Community License 1.0** (`license: other`,
  `qwen-community-1.0`), inherited by these files. Not affiliated with the Qwen team.
- Draft head: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (`MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`).
- Engine: the PaoAI engine 1.0.0, free to use under its own licence (`EULA.md`, inside the download). It is built on llama.cpp; the notices for
  llama.cpp and the other parts made by others are inside the download (`THIRD-PARTY-NOTICES.md`).
- Quants, engine and tests: [PaoAI](https://huggingface.co/PaoAI).
