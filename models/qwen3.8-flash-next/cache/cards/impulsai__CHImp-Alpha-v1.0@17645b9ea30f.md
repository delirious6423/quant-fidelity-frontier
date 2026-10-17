---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
quantized_by: impulsAI
tags:
  - gguf
  - llama.cpp
  - moe
  - mixture-of-experts
  - qwen4exp
  - speculative-decoding
  - mtp
  - impulsAI
language:
  - en
  - de
---

<div align="center">

<img src="assets/hero.png" alt="CHImp Alpha v1.0 — Qwen3.8-Flash-Next on one 32 GB GPU: 41 tok/s, HumanEval+ 94.5 %, 93 % same next token as the 8-bit reference, 92 GiB of weights instead of 175" width="100%">

<br>

[![base model](https://img.shields.io/badge/base-Qwen3.8--Flash--Next-1f6feb)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
![params](https://img.shields.io/badge/model-125B%20MoE%20%2B%2051B%20n--gram%20table%20%C2%B7%2010%20of%20512%20experts%20active-555)
![size](https://img.shields.io/badge/download-92.2%20GiB%20%2B%201.8%20GiB%20draft%20head-555)
![hardware](https://img.shields.io/badge/runs%20on-one%2032%20GB%20GPU%20%2B%20128%20GB%20RAM-555)
![decode](https://img.shields.io/badge/decode-41%20tok%2Fs-199e70)
![humaneval](https://img.shields.io/badge/HumanEval%2B-94.5%25-199e70)
![kld](https://img.shields.io/badge/same%20next%20token%20as%208--bit%20reference-93%25-199e70)
[![runtime](https://img.shields.io/badge/runtime-llama.cpp%20b10909%20(unsloth%20CUDA--13%20bundle)-d95926)](https://github.com/ggml-org/llama.cpp)
[![tooling](https://img.shields.io/badge/build%20repo-zurd46%2FQwen3.8FlashNextTQ1-8f8e86)](https://github.com/zurd46/Qwen3.8FlashNextTQ1)

**Powered by [impulsAI.ch](https://impulsai.ch)**

</div>

# CHImp Alpha v1.0


**Qwen3.8-Flash-Next** is Qwen's 125-billion-parameter mixture-of-experts language model (6B active per token) with a
51-billion-parameter n-gram table — 177 billion weights in one file. **CHImp Alpha v1.0 is that model, compressed and
arranged so that it runs on one 32 GB GPU with 128 GB of system RAM** — no multi-GPU box, no cloud — at **41 tokens per
second**, picking the same next token as the 8-bit reference **93 % of the time**.

Every number on this page was measured on this exact build, on one machine (llama.cpp b10909, RTX Pro 4500 32 GB, Ryzen 9
5950X, 128 GB DDR4), with thinking off. Nothing is estimated and nothing comes from other builds. The reference throughout is
the **8-bit conversion of Qwen3.8-Flash-Next** (Q8_0, 175.3 GiB) — the closest version of the original that runs anywhere
near this hardware. Text only: the original model can also read images; this build cannot.

## ✅ Why it is good

| | measured on this build |
|---|---|
| **Runs on one 32 GB card** | 92.2 GiB of weights: 24.9 GiB on the GPU, 40.5 GiB read constantly from RAM, 26.8 GiB left on disk and read in small pieces. At the 64k-context default 31.0 GiB of the card's 31.9 GiB were in use — including ≈ 0.9 GiB the desktop took on the test machine. |
| **Keeps the reference's quality** | On 131 072 tokens of Wikipedia text the compressed model picks the **same next token as the 8-bit reference in 93.0 %** of positions; where it differs, the shift is small (KL divergence **0.041** on average, 0.011 for the median token) and its perplexity is only **0.58 % higher** (3.838 vs 3.816). |
| **Fast** | **41 tok/s** (average of code 44, JSON 48, German prose 32) with the model's built-in look-ahead helper — a separate 1.8 GiB file whose guesses the main model checks, so the text is unchanged; 29.5 tok/s without it. Reading a prompt: 132 tok/s on the 60–440-token coding prompts, 212 tok/s on the ~370-token tool-calling prompts — a 400-token prompt takes 2–3 seconds. |
| **Strong on real tasks — with thinking off** | HumanEval **95.7 %** / HumanEval+ **94.5 %**, MBPP **91.8 %** / MBPP+ **78.6 %**, Terminal-Bench Core <!--tb-->**52.5 %** (42 of 80 tasks)<!--/tb-->, BFCL function calling **73.7 %** single-turn / **60.5 %** multi-turn base. |
| **Tool calling and long context built in** | OpenAI-style function calling through `llama-server --jinja` (that is how BFCL was run), 65 536-token context by default, thinking can be switched on. |
| **Half the download** | 92.2 GiB instead of the 175.3 GiB 8-bit conversion, in three GGUF files plus the 1.8 GiB helper. |

## 🧪 Task benchmarks

Public benchmark harnesses with their own scoring — nothing graded in-house. The model was served by `llama-server`
with the **Run** flags below (64k context, look-ahead helper on, **thinking off**), temperature 0 (BFCL: its default
0.001), one attempt per task (pass@1). The 8-bit reference was not run on these benchmarks (it does not fit this machine).

<div align="center"><img src="assets/benchmarks.png" alt="Task benchmarks of CHImp Alpha v1.0 rendered from the harness result files; pending rows are drawn empty" width="100%"></div>

| benchmark | what it tests | CHImp Alpha v1.0 |
|---|---|---|
| **HumanEval / HumanEval+** (EvalPlus 0.3.1) | 164 Python functions — pass@1 on the original tests / on EvalPlus's ~80× larger test sets (the '+' score) | **95.7 % / 94.5 %** (157 / 155 of 164, 0 empty answers) |
| **MBPP / MBPP+** (EvalPlus 0.3.1) | 378 Python problems — pass@1 on the original tests / on the extended tests (the '+' score) | **91.8 % / 78.6 %** (347 / 297 of 378, 0 empty answers) |
| **Terminal-Bench Core 0.1.1** (terminal-bench 0.2.18, Terminus-2 agent) | 80 real tasks in a Docker terminal — fix a build, wrangle data, recover a repo, configure a service; a task counts only if its hidden tests pass | **52.5 %** (42 of 80 tasks resolved) |
| **BFCL v4** (bfcl-eval 2026.3.23) | native OpenAI-style function calling — single-turn (choose and fill the right call, detect when no tool fits) and state-based multi-turn tool use | single-turn **73.7 %** (2574 of 3491 calls in 11/13 categories; plain mean of the category scores 73.9 %) · multi-turn **60.5 %** (121 of 200 scenarios, 1/4 categories) |

BFCL coverage: 11 of 13 single-turn categories ran (Java and JavaScript `simple` did not) and 1 of 4 multi-turn categories (`multi_turn_base`, 200 scenarios; `multi_turn_miss_func` was stopped after 41 of 200 and is not scored). Every call counts once in the percentages.

Thinking was off for every scored run; nothing on this page was measured with thinking on.

<details><summary>EvalPlus: the failed tasks</summary>

- HumanEval+ failed on the original tests: `HumanEval/116`, `HumanEval/129`, `HumanEval/130`, `HumanEval/132`, `HumanEval/145`, `HumanEval/147`, `HumanEval/32`; passed those but failed the extra tests: `HumanEval/39`, `HumanEval/91`.
- MBPP+ failed on the original tests: `Mbpp/124`, `Mbpp/138`, `Mbpp/160`, `Mbpp/235`, `Mbpp/260`, `Mbpp/286`, `Mbpp/310`, `Mbpp/311`, `Mbpp/398`, `Mbpp/415`, `Mbpp/430`, `Mbpp/437`, `Mbpp/448`, `Mbpp/462`, `Mbpp/468`, `Mbpp/473`, `Mbpp/580`, `Mbpp/583`, `Mbpp/590`, `Mbpp/602`, `Mbpp/603`, `Mbpp/615`, `Mbpp/635`, `Mbpp/722`, `Mbpp/755`, `Mbpp/765`, `Mbpp/769`, `Mbpp/773`, `Mbpp/777`, `Mbpp/780`, `Mbpp/787`; passed those but failed the extra tests: `Mbpp/102`, `Mbpp/109`, `Mbpp/113`, `Mbpp/119`, `Mbpp/126`, `Mbpp/129`, `Mbpp/16`, `Mbpp/161`, `Mbpp/223`, `Mbpp/244`, `Mbpp/255`, `Mbpp/267`, `Mbpp/278`, `Mbpp/287`, `Mbpp/294`, `Mbpp/300`, `Mbpp/305`, `Mbpp/410`, `Mbpp/427`, `Mbpp/440`, `Mbpp/451`, `Mbpp/459`, `Mbpp/556`, `Mbpp/559`, `Mbpp/577`, `Mbpp/579`, `Mbpp/589`, `Mbpp/593`, `Mbpp/597`, `Mbpp/620`, `Mbpp/622`, `Mbpp/630`, `Mbpp/639`, `Mbpp/7`, `Mbpp/735`, `Mbpp/739`, `Mbpp/74`, `Mbpp/745`, `Mbpp/748`, `Mbpp/757`, `Mbpp/759`, `Mbpp/771`, `Mbpp/785`, `Mbpp/790`, `Mbpp/792`, `Mbpp/794`, `Mbpp/800`, `Mbpp/806`, `Mbpp/84`, `Mbpp/99`.

</details>

<details><summary>Terminal-Bench per task</summary>

Resolved (42): `blind-maze-explorer-5x5`, `blind-maze-explorer-algorithm`, `blind-maze-explorer-algorithm.easy`, `blind-maze-explorer-algorithm.hard`, `conda-env-conflict-resolution`, `configure-git-webserver`, `count-dataset-tokens`, `crack-7z-hash`, `crack-7z-hash.easy`, `crack-7z-hash.hard`, `csv-to-parquet`, `eval-mteb`, `eval-mteb.hard`, `extract-safely`, `fix-pandas-version`, `fix-permissions`, `git-workflow-hack`, `grid-pattern-transform`, `hello-world`, `heterogeneous-dates`, `incompatible-python-fasttext`, `incompatible-python-fasttext.base_with_hint`, `modernize-fortran-build`, `new-encrypt-command`, `oom`, `openssl-selfsigned-cert`, `organization-json-generator`, `processing-pipeline`, `prove-plus-comm`, `pytorch-model-cli`, `pytorch-model-cli.easy`, `pytorch-model-cli.hard`, `sanitize-git-repo`, `sanitize-git-repo.hard`, `simple-web-scraper`, `solana-data`, `sqlite-db-truncate`, `sqlite-with-gcov`, `swe-bench-astropy-1`, `swe-bench-astropy-2`, `swe-bench-langcodes`, `tmux-advanced-workflow`

Unresolved (38): `build-initramfs-qemu`, `build-linux-kernel-qemu`, `build-tcc-qemu`, `cartpole-rl-training`, `chess-best-move`, `create-bucket`, `cron-broken-network`, `decommissioning-service-with-sensitive-data`, `download-youtube`, `extract-moves-from-video`, `fibonacci-server`, `fix-git`, `get-bitcoin-nodes`, `git-multibranch`, `gpt2-codegolf`, `hf-model-inference`, `intrusion-detection`, `jupyter-notebook-server`, `nginx-request-logging`, `password-recovery`, `path-tracing`, `path-tracing-reverse`, `play-zork`, `polyglot-c-py`, `polyglot-rust-c`, `qemu-alpine-ssh`, `qemu-startup`, `raman-fitting`, `raman-fitting.easy`, `reshard-c4-data`, `run-pdp11-code`, `security-vulhub-minio`, `simple-sheets-put`, `super-benchmark-upet`, `swe-bench-fsspec`, `train-fasttext`, `vim-terminal-task`, `write-compressor`

</details>

<details><summary>BFCL v4 per category</summary>

| category | accuracy | correct / total |
|---|---|---|
| `simple_python` | **94.5 %** | 378 / 400 |
| `simple_java` | not run | – |
| `simple_javascript` | not run | – |
| `multiple` | **93.0 %** | 186 / 200 |
| `parallel` | **65.5 %** | 131 / 200 |
| `parallel_multiple` | **67.0 %** | 134 / 200 |
| `irrelevance` | **57.9 %** | 139 / 240 |
| `live_simple` | **76.0 %** | 196 / 258 |
| `live_multiple` | **75.6 %** | 796 / 1053 |
| `live_parallel` | **62.5 %** | 10 / 16 |
| `live_parallel_multiple` | **75.0 %** | 18 / 24 |
| `live_irrelevance` | **64.8 %** | 573 / 884 |
| `live_relevance` | **81.2 %** | 13 / 16 |
| `multi_turn_base` | **60.5 %** | 121 / 200 |
| `multi_turn_miss_func` | stopped at 41/200, not scored | – |
| `multi_turn_miss_param` | not run | – |
| `multi_turn_long_context` | not run | – |

</details>

Per-test report with harness commands, raw verdict files and the published numbers of the original
Qwen3.8-Flash-Next: [`BENCHMARKS_CHImp.md`](https://github.com/zurd46/Qwen3.8FlashNextTQ1/blob/main/BENCHMARKS_CHImp.md)
in the build repository.

## 📏 Quality against the 8-bit reference

The compressed model and the 8-bit reference conversion of Qwen3.8-Flash-Next each read the same 131 072 tokens of
Wikipedia text (wikitext-2 test set, 64 chunks × 2048 tokens), and their next-token predictions were compared position by
position:

| metric | CHImp Alpha v1.0 | in plain words |
|---|---|---|
| same top-1 token | **93.0 %** | in 93 of 100 positions the compressed model's most likely next token is the one the reference would pick |
| KL divergence, mean / median / 99th percentile | **0.041** / 0.011 / 0.48 | how much its next-token probabilities are shifted from the reference's (0 = identical): tiny for the typical token (0.011), small on average, noticeable only for the 1 % of hardest positions |
| perplexity | **3.838** vs 3.816 (**+0.58 %**) | it is 0.6 % less sure of the true next token than the reference |

The reference itself was not run on the task benchmarks below (it does not fit this machine), so the card cannot say how
many benchmark points the compression costs; the 93 % figure is the direct comparison. Raw file:
`benchmarks/results/perplexity-htc2d.json` in the build repository (same model files under their working name).

## ⚡ Speed

<div align="center"><img src="assets/mtp.png" alt="Decode speed by workload, plain vs with the MTP draft: code 29.5 to 44.4 tok/s (1.5x), German prose 29.6 to 32.3 (1.1x), JSON 29.4 to 48.0 (1.6x)" width="100%"></div>

Qwen3.8-Flash-Next comes with a small look-ahead head (its MTP head). CHImp runs it as a **helper next to the main model**:
the helper guesses up to three tokens ahead, the main model checks the guesses in one pass and keeps only those it would
have produced itself — so the text is exactly what you would get without the helper, it just arrives faster. Measured on
512-token generations at temperature 0 and 8k context: **44 tok/s on code, 48 on JSON, 32 on German prose** (the helper's
guesses were accepted 86 % / 97 % / 48 % of the time) against 29.5 tok/s for all three without it; 41 tok/s is the average
of the three. For German prose `--spec-draft-n-max 2` is the better setting (35.0 tok/s). The task benchmarks and the
31.0 GiB VRAM figure were taken at the 64k default, where the same line measures 38.8 tok/s.

**Tuned line (2026-09-21, same files, flags only).** Pinning one thread per physical core, letting the helper stop when it
is unsure (`--spec-draft-p-min 0.3`), keeping whole layers 16–47 in RAM instead of splitting every layer between RAM and
the card, and an 8-bit KV cache add up to **41.4 tok/s at 64k context (code 45.9, German 31.6, JSON 48.9)** and free
1 GiB of VRAM. Reading a prompt goes from ~250 tok/s to **~490 tok/s** on prompts of 3k tokens and more when the
RAM-side experts are streamed to the card for each prompt batch from pinned memory (`-lm none --op-offload -ub 1024`):
a 13k-token document takes 26 s instead of 52. The price: about a minute longer to start, and prompt batches of
32–400 new tokens take ~1 s more than on the CPU. The full 22-row measurement is in `BENCHMARKS.md` §5 of the
[repository](https://github.com/zurd46/Qwen3.8FlashNextTQ1/blob/main/BENCHMARKS.md).

## 🧠 How it fits on one GPU

<div align="center"><img src="assets/placement.png" alt="Memory placement: 24.9 GiB of weights plus the 1.8 GiB look-ahead helper on the GPU, 40.5 GiB read constantly in RAM, 26.8 GiB n-gram table on disk" width="100%"></div>

A mixture-of-experts model does not use all of its weights for every token: in each of its 48 layers a token is routed to
10 of 512 experts. CHImp is laid out around that:

- **On the GPU (24.9 GiB of weights + the 1.8 GiB helper):** everything every token needs — attention, the recurrent
  DeltaNet layers, the shared expert, the hyper-connections and the language-model head (3.8 GiB) — plus the experts' output
  weights (21.1 GiB), which are read selectively too but fit on the card, so they go there.
- **In RAM (40.5 GiB, read constantly):** the experts' input weights (39.8 GiB, 4.25 bits per weight) and the token
  embedding. For each token only the 10 chosen experts per layer are read — about 0.8 GiB of the 39.8 GiB.
- **On disk (26.8 GiB):** the 51-billion-parameter n-gram table, memory-mapped. A token touches 16 rows of it, so only those
  are read; with 128 GB of RAM the operating system caches the touched parts, and nothing is read from disk while generating.

That adds up to 31.0 GiB in use on the 32 GB card at 64k context (0.9 GiB of it the desktop), 40.5 GiB of RAM in constant
use and 67.3 GiB mapped in total. Less than 128 GB of RAM was not tested.

## 🚀 Run

<details open>
<summary><b>llama-server</b> — tested only with unsloth's llama.cpp bundle <code>b10909-mix</code> (CUDA 13); any llama.cpp that knows the <code>qwen4exp</code> model type and the <code>draft-mtp</code> option should work, but no other build was tried</summary>

```bash
llama-server -m CHImp-Alpha-v1.0-00001-of-00003.gguf -c 65536 -np 1 --jinja \
  -ngl 99 -t 16 -fa on --no-op-offload --fit off \
  -ot 'blk\.\d+\.ffn_gate_exps\.weight=CPU,blk\.\d+\.ffn_up_exps\.weight=CPU,per_layer_token_embd\.weight=CPU' \
  -md mtp-CHImp-Alpha-v1.0-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3 \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'   # drop these two to think
```

| flag | what it does |
|---|---|
| `-ot …=CPU` | keeps the experts' input weights and the n-gram table off the card — **without it the model does not fit** |
| `--no-op-offload` | computes the RAM-side experts on the CPU instead of copying them to the GPU for every prompt batch |
| `--fit off` | stops llama.cpp from silently moving weights around when VRAM gets tight; the placement above is deliberate |
| `-t 16` | set to your CPU's number of physical cores (16 on the test machine) |
| `-c 65536 -np 1` | 64k context, one request at a time — the setting the task benchmarks and the 31.0 GiB VRAM figure used (speed was measured at 8k, perplexity at 2k) |
| `-md … --spec-type draft-mtp` | the look-ahead helper: 1.4× decode, same text; leave it out to run plain (29.5 tok/s, 2.3 GiB less VRAM) |
| `--reasoning-budget 0` + `enable_thinking:false` | thinking off — the setting all benchmarks here used; drop both to let the model think |

**Tuned line** — the same files, measured 2026-09-21 (see ⚡ Speed): +7 % decode, 1.9× prompt reading, 96k context on the
32 GB card. Start takes about a minute longer (42 GiB of RAM are page-locked so PCIe copies run at full speed):

```bash
llama-server -m CHImp-Alpha-v1.0-00001-of-00003.gguf -c 98304 -np 1 --jinja \
  -ngl 99 -t 16 -fa on --fit off --op-offload -lm none -b 2048 -ub 1024 \
  -ot 'blk\.(1[6-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps\.weight=CPU,per_layer_token_embd\.weight=CPU' \
  -ctk q8_0 -ctv q8_0 --cpu-range 0-15 --cpu-strict 1 --poll 100 \
  -md mtp-CHImp-Alpha-v1.0-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}'
```

| flag | what it does |
|---|---|
| `-ot 'blk\.(1[6-9]\|[2-4][0-9])\.ffn_(gate\|up\|down)_exps…=CPU'` | layers 16–47 keep all three expert matrices in RAM, layers 0–15 live entirely on the card: same RAM traffic, a third fewer CPU↔GPU hand-offs per token, 0.85 GiB less VRAM |
| `-lm none --op-offload -ub 1024` | prompt batches of 32 tokens and more stream the RAM-side experts to the card from page-locked memory (26 GB/s over PCIe 4.0): ~490 tok/s on long prompts; use `--no-op-offload` instead if your prompts are mostly short |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV cache — no measurable speed change, 1 GiB of VRAM back, which is what pays for ub 1024 and 96k context |
| `--cpu-range 0-15 --cpu-strict 1 --poll 100` | one thread per physical core, threads spin instead of sleeping between the ~100 CPU ops of a token (+6 % decode) |
| `--spec-draft-p-min 0.3` | the helper stops guessing when its top probability is under 0.3 (+4 % decode, mostly on prose) |
| `-c 98304` | the largest context that fits with the helper (31.8 GiB in use on a 34k-token prompt); 128k does not |

</details>

<details>
<summary><b>Context length and thinking</b></summary>

- **Context**: 64k is what the benchmarks used (31.0 GiB of the card in use with the helper). The tuned line above runs
  96k with `-ctk q8_0 -ctv q8_0` (31.8 GiB on a 34k-token prompt); 128k did not fit with the helper on this card.
- **Thinking**: Qwen3.8-Flash-Next thinks by default. Nothing on this page was measured with thinking on. To use it, start
  the server without the two thinking-off flags; expect longer answers.

</details>

## 📦 Files

| file | GiB | content |
|---|---|---|
| `CHImp-Alpha-v1.0-00001-of-00003.gguf` | 64.8 | the 48 transformer blocks + token embedding |
| `CHImp-Alpha-v1.0-00002-of-00003.gguf` | 0.6 | language-model head |
| `CHImp-Alpha-v1.0-00003-of-00003.gguf` | 26.8 | n-gram table |
| `mtp-CHImp-Alpha-v1.0-shared-Q4_K_M.gguf` | 1.8 | the look-ahead helper (the model's MTP head, unsloth's conversion) |
| `CHImp-Alpha-v1.0.llama-args.txt` | | the `llama-server` line above |
| `CHImp-Alpha-v1.0.json` | | shard and format summary |

```bash
hf download impulsai/CHImp-Alpha-v1.0 --local-dir CHImp   # llama.cpp finds shards 2 and 3 next to shard 1
```

*The four GGUF files are uploaded once the benchmark run on the build machine is through; until then this repo holds the
card, the graphics and the metadata (see the status line at the top).*

## 🔧 How it was built

From the 8-bit (Q8_0) conversion of Qwen3.8-Flash-Next by unsloth and its importance matrix, with the tooling in
[zurd46/Qwen3.8FlashNextTQ1](https://github.com/zurd46/Qwen3.8FlashNextTQ1) (recipe `configs/htc2.yaml`). The 80 billion
expert input weights — the bulk of the model — got the smallest storage format that kept the KL divergence against the
8-bit reference below 0.05 (IQ4_XS, 4.25 bits per weight); the expert output weights use the 4.5-bit format their row
width allows; everything else keeps 6.6–8.5 bits or the source's full precision. Each group is stored where it is read
from. The measurements on this page have their raw files in the repository's `benchmarks/results/` and are summarised
with commands and per-test lists in
[`BENCHMARKS_CHImp.md`](https://github.com/zurd46/Qwen3.8FlashNextTQ1/blob/main/BENCHMARKS_CHImp.md) — the repository is
not public yet (status line above).

## ⚖️ License

The weights derive from Qwen/Qwen3.8-Flash-Next and are distributed under the Qwen Community License 1.0; the helper file is
unsloth's conversion of the same model's MTP head. Use is subject to the base model's terms.

<div align="center"><sub>CHImp Alpha v1.0 · Powered by <a href="https://impulsai.ch">impulsAI.ch</a> · 2026</sub></div>
