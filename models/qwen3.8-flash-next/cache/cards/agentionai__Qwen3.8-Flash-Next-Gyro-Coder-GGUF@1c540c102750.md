---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8-flash-next
- qwen4exp
- code
- experimental
- agentionai
---

<p>
<a href="https://www.agention.ai/"><img src="./gyro-coder-header.png" width="100%" alt="AgentionAI Gyro Coder 3.8 Flash Next"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"><b>Gyro (general use)</b></a> &bull; <a href="https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF"><b>Signal 2 (agentic coding)</b></a> &bull; <a href="https://marshall.agention.ai"><b>Marshall coding agent</b></a> &bull; <a href="https://www.agention.ai/"><b>Website</b></a> &bull; <a href="#work-with-agentionai"><b>Custom quants</b></a>
</p>

<h1 align="center">Qwen3.8-Flash-Next Gyro-Coder</h1>

<p align="center"><b>A 125B-parameter coding model on one 24 GB card.</b> Gyro-Coder brings Qwen3.8-Flash-Next to a single RTX 3090, 4090 or 7900 XTX in 19.9 GiB of GPU memory; Gyro-Coder-L does the same for 32 GB cards.</p>

<p align="center">
  <a href="#pick-your-file"><b>Pick your file</b></a> · <a href="#quick-start">Quick start</a> · <a href="#quality">Quality</a> · <a href="#how-the-experts-were-chosen">How the experts were chosen</a> · <a href="#changelog">Changelog</a>
</p>

<p>
<img class="dark:hidden" src="./assets/stats-light.svg" width="100%" alt="19.92 GiB of GPU weights: a 125B coding model on one 24 GB card (RTX 3090, 4090 or 7900 XTX). 94.5% top-1 agreement with the full model on C++ (Python 93.1%). Gyro-Coder-L is closer to the full model on C++ than Gyro-M with 21% less memory (KLD 0.077 in 27.54 GiB against 0.082 in 34.98 GiB). Gyro-Coder-L v1.2: 0 broken tool-call tags in 100 sampled sites; 292 tool calls, 0 format errors.">
<img class="hidden dark:block" src="./assets/stats-dark.svg" width="100%" alt="19.92 GiB of GPU weights: a 125B coding model on one 24 GB card (RTX 3090, 4090 or 7900 XTX). 94.5% top-1 agreement with the full model on C++ (Python 93.1%). Gyro-Coder-L is closer to the full model on C++ than Gyro-M with 21% less memory (KLD 0.077 in 27.54 GiB against 0.082 in 34.98 GiB). Gyro-Coder-L v1.2: 0 broken tool-call tags in 100 sampled sites; 292 tool calls, 0 format errors.">
</p>

**Only the experts that code, without the usual damage.** Each layer keeps the 256 of its 512 experts that matter for
programming, which leaves 65B parameters on the GPU, stored at 2.125 bits per weight with our own trellis encoder: the
same one that puts Gyro-M ahead of IQ3_XXS and UD-Q2_K_XL quants that need 20-25% more memory. On held-out Python it
picks the same next token as the full model 93% of the time. Give it clear instructions (implement this, fix that,
refactor, explain) and it delivers. For open-ended work and general knowledge,
[Gyro-S](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) or Gyro-M is the better pick.

## Pick your file

**The file follows your GPU memory.**

| Your hardware | File | GPU memory (weights) |
|---|---|---:|
| 24 GB card (RTX 3090 / 4090, 7900 XTX) | **`Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`** | 19.92 GiB |
| 32 GB card (RTX 5090, R9700), code | **`Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf`** | 27.54 GiB |
| 32 GB, general use as well as code | [Gyro-S](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 27.64 GiB |
| 48 GB+, general use as well as code | [Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 34.98 GiB |

**Gyro-Coder-L** keeps the same experts at about 3.2 bits per weight: 27.5 GiB of GPU memory, and closer to the full
model than Gyro-Coder on every set we measure. On C++ it is closer than Gyro-M, which keeps all 512 experts.

> [!IMPORTANT]
> **Stock llama.cpp cannot load these files.** The `TQ2_0` / `TQ3_0` in the names are only size classes for the Hub.
> Runs in our llama.cpp build (Docker, see Quick start). Gyro-Coder-L also runs in our Strata branch (see below).

## Quick start

1. **Download Gyro-Coder** (70.6 GiB on disk; Gyro-Coder-L 78.2 GiB; put it on a fast NVMe drive).
2. **Run it** with our [llama.cpp build](https://gitlab.com/agentionai-group/llama.cpp) (Vulkan, in Docker).
3. **Point your coding agent or editor plugin** at `http://127.0.0.1:8080/v1` (web chat at http://127.0.0.1:8080).

```bash
pip install -U huggingface_hub
hf download agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf --local-dir ~/models
# 32 GB card: Qwen3.8-Flash-Next-Gyro-Coder-L-TQ3_0.gguf (and use that name below)

# AMD (and Intel) GPUs
docker run --rm -it --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v ~/models:/models -p 8080:8080 agentionai/agention-llama:server \
  -m /models/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf -ngl 999 -fa on --jinja --ngram-on-disk \
  -c 65536 -ctk q8_0 -ctv q8_0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05
```

**NVIDIA:** replace `--device /dev/dri --group-add ...` with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`.

<details>
<summary><b>Use it in Marshall</b>, our open-source coding agent</summary>

Marshall runs a small team of agents (planner, coder, reviewer) on local or hosted models, with every action
approval-gated by default.

```bash
npm install -g @agentionai/marshall-cli && marshall
```

Inside Marshall, run `/model` and pick your model from the selection menu.

</details>

<details>
<summary>Strata</summary>

Gyro-Coder-L runs on our Strata branch,
[agentionai-group/strata `main`](https://gitlab.com/agentionai-group/strata/-/tree/main), tested on AMD Strix Halo;
build it from source as in its `docs/GYRO.md`. Upstream [Strata](https://github.com/Niko1221/Strata) 0.2.18 includes
the kernels for these types, but we have not tested this file there yet. Gyro-Coder (the 24 GB file) has not been
tested in Strata yet: use our llama.cpp build.

</details>

### Recommended settings

| Setting | Value |
| --- | --- |
| Sampling | `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05` (the settings all our tests used) |
| n-gram table | keep `--ngram-on-disk`, with the model on fast NVMe: the 50.7 GiB n-gram table stays on disk, which is why the file (70.6 GiB) is much larger than its GPU footprint |
| Prompts | give it clear instructions; it does best on well-specified tasks |

## Quality

**Code stays close to the full model.** Half the experts are gone, so general text and knowledge lose the most
(Gyro-M: held-out 0.291); code stays close to the full model.

<p>
<img class="dark:hidden" src="./assets/top1-light.svg" width="100%" alt="Top-1 agreement with the full model by language, Gyro-Coder against Gyro-Coder-L: C++ 94.5% and 95.5%, TypeScript/JS 92.5% and 93.7%, SQL 92.7% and 93.6%, Python 93.1% and 94.0%, Rust 88.7% and 89.5%, shell 80.9% and 82.3%.">
<img class="hidden dark:block" src="./assets/top1-dark.svg" width="100%" alt="Top-1 agreement with the full model by language, Gyro-Coder against Gyro-Coder-L: C++ 94.5% and 95.5%, TypeScript/JS 92.5% and 93.7%, SQL 92.7% and 93.6%, Python 93.1% and 94.0%, Rust 88.7% and 89.5%, shell 80.9% and 82.3%.">
</p>

| Set | Gyro-Coder KLD ↓ | top-1 ↑ | Gyro-Coder-L KLD ↓ | top-1 ↑ |
|---|---:|---:|---:|---:|
| C++ (held-out source files) | 0.118 | 94.5% | **0.077** | **95.5%** |
| web (TypeScript / JS) | 0.142 | 92.5% | **0.108** | **93.7%** |
| SQL | 0.145 | 92.7% | **0.114** | **93.6%** |
| Python | 0.165 | 93.1% | **0.132** | **94.0%** |
| Rust | 0.223 | 88.7% | **0.189** | **89.5%** |
| shell | 0.417 | 80.9% | **0.371** | **82.3%** |
| knowledge (descriptions of everyday things) | 0.301 | 80.8% | **0.237** | **83.6%** |
| held-out (2026 technical writing and code) | 0.579 | 70.8% | **0.521** | **72.0%** |

KL divergence and top-1 agreement against the full model (Q8_0, all 512 experts), `-c 2048`.

<details>
<summary>Coding tests</summary>

8 auto-graded tasks: LRU cache, interval merging, bug fix from a trace, refactor, bash script, SQL, CSV parsing in JS,
multi-file change; 2 seeds each, thinking on, 64k tokens. Gyro-Coder passes 13 of 16; the same expert set at 8 bits
passes 15 of 16. Gyro-Coder-L v1.2 passes 13 of 16 and the bash task 3 of 10 (4 runs hit the 64k token limit).

</details>

<details>
<summary><b>Gyro-Coder-L v1.2:</b> 0 broken tool-call tags</summary>

Thanks to a detailed community report, three experts that matter for tool calls and symbols are back in the expert
set, in place of three that did less. Tool calls no longer break under sampling: across 100 `<parameter=` sites in real
agent sessions, v1.0 wrote a broken tag at 4 and v1.2 at none. Roman numerals are fixed too, and KLD is unchanged on
every set. v1.2 replaces both v1.0 and the v1.1 preview.

| Test | v1.0 | v1.2 |
|---|---:|---:|
| Broken `<parameter=` tag, per site, sampled (temp 0.6, top-p 0.95, top-k 20, min-p 0.05) | 0.69% | **0%** |
| Sites with any broken-tag risk, out of 100 | 4 | **0** |
| `int_to_roman` written correctly, 5 runs | 3 | **4** |
| Bare 13-pair Roman table, greedy | wrong | **correct** |
| Agent sessions (tool calls): code tasks / Roman task | — | **16/16 / 4/4**, 292 calls, 0 format errors |
| KLD, all 8 sets above | — | within 0.001 of v1.0 |

Symbol tables are still a weak spot of the pruned files (Greek letters: 2 of 5 runs correct); for those, use Gyro-M.

</details>

**Speed:** Strix Halo (Vulkan, MTP draft): 11–22 tok/s per stream with two streams running at once (measured while
generating long answers).

## How the experts were chosen

**A needle in a very large haystack.** Each layer keeps 256 of its 512 experts: about 10¹⁵² possible choices for one
layer, and about **10⁷³²⁸** for the whole model.

<details>
<summary>The search</summary>

Written out, 10⁷³²⁸ fills about four pages. If you had tested a billion candidate models every second since the Big
Bang, you would have covered about 10⁻⁷³⁰¹ of them.

The layers don't act alone either: what one layer keeps changes what every later layer sees, and each candidate is
a full model that has to be built and tested before you know whether it is any good.

We started by mapping which experts carry which domains. From there, an evolutionary search proposed and scored
about 2,200 candidate expert sets across 14 runs. The best were built as full models and put through coding, loop
and knowledge tests. This is the one that passed.

</details>

## Smaller download: files without the n-gram table

**Already have the table? Download about a third.** The n-gram (PLE) table is the base model's and the same for every
Qwen3.8-Flash-Next quant.

| | Full file | `-noPLE` file | Table it ships |
|---|---:|---:|---|
| Gyro-Coder | 75.8 GB | 21.4 GB | Q8_0 |
| Gyro-Coder-L | 84.0 GB | 29.6 GB | Q8_0 |

<details>
<summary>How to load a <code>-noPLE</code> file</summary>

Get the table once from
[Delgon/Qwen3.8-Flash-Next-PLE-GGUF](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF) and load it with `--ple`
(details: [Gyro card](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF#smaller-download-files-without-the-n-gram-table)).

```bash
hf download agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0-noPLE.gguf --local-dir ~/models
hf download Delgon/Qwen3.8-Flash-Next-PLE-GGUF Qwen3.8-Flash-Next-PLE-Q8_0.gguf --local-dir ~/models
llama-server -m ~/models/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0-noPLE.gguf --ple ~/models/Qwen3.8-Flash-Next-PLE-Q8_0.gguf -ngl 999 -fa on --jinja ...
```

`--ple` works in agentionai/llama.cpp main from 2026-11-05 (and containers built after that) and in strix-llama.cpp master; Strata takes
the table with `--ple-gguf`. Older builds, LM Studio and Ollama cannot load a `-noPLE` file: use the full file there.

</details>

## Changelog

**The model files last changed on 2026-10-29 (Gyro-Coder-L v1.2).** If you downloaded after that, there is nothing to
download again.

- **2026-11-21: card only.** New layout; same numbers, commands and files.
- **2026-11-05: new optional files** `-noPLE` for Gyro-Coder and Gyro-Coder-L: the same weights without the n-gram
  table. Full files unchanged.
- **2026-10-29: Gyro-Coder-L v1.2** replaces v1.0 under the same file name (and the v1.1 preview): three experts
  swapped back in after a community report on broken tool-call tags; tool calls and Roman numerals fixed, KLD unchanged.
- **2026-10-26: Gyro-Coder-L v1.1 preview** (improved encode of the same experts).

Experimental, like the Gyro files: the format and kernels may change.

## Work with AgentionAI

<p align="center">
  <b>Want results like these on your own model?</b><br>
  <sub>We tune and quantize models for your agents and hardware: fewer tokens, less GPU memory, quality measured against the original. Every engagement ends with a full measurement report. These quants are released freely; AgentionAI is a one-person team, and sponsoring funds the next release and the benchmarking behind it.</sub>
</p>

<p align="center">
  <a href="mailto:hello@agention.ai"><img src="./assets/btn-custom.svg" alt="Custom quants and tunes: hello@agention.ai" height="44"></a>&nbsp;&nbsp;
  <a href="https://github.com/sponsors/agentionai"><img src="./assets/btn-sponsor.svg" alt="Sponsor AgentionAI on GitHub" height="44"></a>&nbsp;&nbsp;
  <a href="https://x.com/laurent_zw"><img src="./assets/btn-follow.svg" alt="Follow @laurent_zw on X" height="44"></a>
</p>

<p align="center"><sub>Or <a href="https://buymeacoffee.com/agentionai">buy me a coffee</a> or some GPU time.</sub></p>

## Credits

- **Hardware:** built and tested on AMD Ryzen AI Max (Strix Halo). Hardware supported by AMD.
- **Format:** trellis-coded quantization with Hadamard rotation, in the family of QTIP (Tseng et al., 2024). Encoder, trellis types and kernels: AgentionAI. The Gyro format is open: specification, reference decoder and encoder, and bit-exact tests (Apache 2.0) at [agentionai-group/agention-gyro](https://gitlab.com/agentionai-group/agention-gyro). The released files come from our own production encoder.
- **Base model:** Qwen/Qwen3.8-Flash-Next. **Runtimes:** llama.cpp, Strata, strix-llama.cpp.
- Thanks to the community tester whose report on broken tool-call tags led to v1.2.
