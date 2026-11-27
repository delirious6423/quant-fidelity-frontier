---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- gguf
- qwen3.8-flash-next
- qwen4exp
- experimental
- agentionai
---

<p>
<a href="https://www.agention.ai/"><img src="./gyro-header.png" width="100%" alt="AgentionAI Gyro 3.8 Flash Next"></a>
</p>

<p align="center">
  <a href="https://www.agention.ai/"><b>Website</b></a> &bull; <a href="https://agention.ai/models/qwen3.8-flash-next/"><b>Guide</b></a> &bull; <a href="https://marshall.agention.ai"><b>Marshall coding agent</b></a> &bull; <a href="https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF"><b>Signal 2</b></a> &bull; <a href="https://www.fieldworkledger.com"><b>Fieldwork Ledger</b></a> &bull; <a href="#work-with-agentionai"><b>Custom quants</b></a>
</p>

<h1 align="center">Qwen3.8-Flash-Next Gyro</h1>

<p align="center"><b>Qwen3.8-Flash-Next (125B MoE) on one GPU.</b> Gyro-S for a 32 GB card (27.6 GiB), Gyro-M for 48 GB (35 GiB), Gyro-L for 64 GB (50 GiB). Released freely.<br><b>Gyro-S and Gyro-M updated to v2 (Gyro-M on 2026-11-26).</b> Same files, same sizes, closer to the original: 27% less drift on tool calls for both, with agentic coding on par.<br>Running a coding agent or looking for a speedup? <a href="https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF"><b>Signal 2</b></a> is Gyro-S trained on real agentic coding work.</p>

<p align="center">
  <a href="#pick-your-file"><b>Pick your file</b></a> · <a href="#quick-start">Quick start</a> · <a href="#quality">Quality</a> · <a href="#speed">Speed</a> · <a href="#smaller-download-files-without-the-n-gram-table">Smaller download</a> · <a href="#work-with-agentionai">Work with us</a>
</p>

<p>
<img class="dark:hidden" src="./assets/stats-light.svg" width="100%" alt="Gyro-S v2 runs on a 32 GB card in 27.64 GiB, up to 128k context. Gyro-S v2 and Gyro-M v2 drift 27% less on tool-call transcripts than v1 at the same size. 221 tok/s on an RTX 5090 with Strata (Gyro-S, JSON output; prose 140, code 209). Gyro-M v2 uses 20% less GPU memory than ISTA IQ3_XXS and drifts 20% less (KLD 0.249 against 0.311).">
<img class="hidden dark:block" src="./assets/stats-dark.svg" width="100%" alt="Gyro-S v2 runs on a 32 GB card in 27.64 GiB, up to 128k context. Gyro-S v2 and Gyro-M v2 drift 27% less on tool-call transcripts than v1 at the same size. 221 tok/s on an RTX 5090 with Strata (Gyro-S, JSON output; prose 140, code 209). Gyro-M v2 uses 20% less GPU memory than ISTA IQ3_XXS and drifts 20% less (KLD 0.249 against 0.311).">
</p>

## Pick your file

**The file follows your GPU memory.** Each one runs fully on the GPU; the n-gram table stays on disk.

| Your hardware | File | GPU memory (weights) |
|---|---|---:|
| 32 GB card (RTX 5090, R9700) | **`Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`** (v2, updated 2026-11-24) | 27.64 GiB |
| 32 GB card, for coding agents | [Signal 2](https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF): Gyro-S tuned on real agentic coding work | same size as Gyro-S |
| 48 GB (one card or 2×24), 64 GB unified memory | **`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`** (v2, updated 2026-11-26) | 34.98 GiB |
| 64 GB unified memory (Strix Halo): more quality | **`Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf`** | 50.21 GiB |
| 24 GB card, for coding | [Gyro-Coder](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF): 256 code experts per layer | 19.92 GiB |
| 16–24 GB card | Gyro-S with Strata's expert cache | |
| Strix Halo / unified memory, 64 GB+ | Gyro-S or Gyro-M | |

- **Tuned for agentic coding:** [Signal 2](https://huggingface.co/agentionai/Signal-2-Flash-Next-Gyro-GGUF) is Gyro-S with an agentic-coding tune merged in. On our coding-agent gate it passes the same tasks as Gyro-S with 27% fewer output tokens, so each run finishes sooner.
- **Abliterated** Gyro-S and Gyro-M: [agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF).
- **Gyro-S v2** replaces Gyro-S v1 under the same file name: same size and tensor types. Tested with agentionai/llama.cpp and strix-llama.cpp master (the full file, and the `-noPLE` file with `--ple`, give identical output), and in Strata on AMD (HIP) at its default settings. Strata's setup still downloads v1 until its catalog is updated; v1 stays available as revision `bdd06693`.
- **Gyro-M v2** replaces Gyro-M v1 under the same file names: same size and tensor types, tested with agentionai/llama.cpp. Strata's setup still downloads v1 until its catalog is updated; v1 stays available as revision `cd23c647`.
- Each file also comes as a smaller `-noPLE` download: see [Smaller download](#smaller-download-files-without-the-n-gram-table).

> [!IMPORTANT]
> **Upstream llama.cpp cannot load these files yet.** They run in:
> - [Strata](https://github.com/Niko1221/Strata) 0.2.0 or newer: app, Terminal and engine for NVIDIA and AMD RX 7000 / RX 9000 / R9700
> - [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) master: CPU, Vulkan and HIP, made for AMD Strix Halo
> - [agentionai/llama.cpp](https://gitlab.com/agentionai-group/llama.cpp) and its container: CPU, Vulkan and CUDA; also runs the MTP draft

<details>
<summary>About the file names and format</summary>

The routed experts are stored in a rotated basis with our own rotor code (Agention Precision Rotor, APR). The
`TQ1_0` / `TQ2_0` / `TQ3_0` in the names is only a size class for the Hub (the experts use the trellis types TQK6/TQK7,
TQ2_T and TQP5-7).

</details>

## Quick start

1. **Download a file** for your hardware, plus the MTP draft and vision projector if you want them.
2. **Start the server** with our container (below), Strata or strix-llama.cpp.
3. **Open http://localhost:8080**, or point your app or coding agent at it.

```bash
hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf \
  mtp-Qwen3.8-Flash-Next-draft.gguf mmproj-F16.gguf --local-dir ~/models

# AMD (and Intel) GPUs
docker run --rm -it --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v ~/models:/models -p 8080:8080 agentionai/agention-llama:server \
  -m /models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf -ngl 999 -fa on --jinja --ngram-on-disk \
  -c 65536 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 \
  --mmproj /models/mmproj-F16.gguf
```

**Gyro-L:** use `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` in place of the Gyro-S file (pull the image again first).

<details>
<summary><b>NVIDIA</b></summary>

Replace `--device /dev/dri --group-add ...` with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`
(Vulkan), or build [agentionai/llama.cpp](https://gitlab.com/agentionai-group/llama.cpp) with `-DGGML_CUDA=ON` for full speed.

</details>

<details>
<summary><b>Strata</b> (NVIDIA, AMD RX 7000 / RX 9000 / R9700): the fastest way on an RTX 5090</summary>

[Strata](https://github.com/Niko1221/Strata) is the fastest way to run Gyro on an RTX 5090, and its expert cache keeps
the most-used experts on the GPU, which makes 24 GB cards practical. Since 0.2.0 it is an app, a Terminal and an
engine, and the installers carry the engine. Pick the file for your card on the
[release page](https://github.com/Niko1221/Strata/releases/latest), or install the Terminal:

```bash
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
strata launch                                                             # suggests a model for your memory and serves it
strata launch --model-path ~/models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf  # files you already have
```

Web chat at http://127.0.0.1:8080, OpenAI-compatible API at `http://127.0.0.1:8080/v1`.
Measured with Strata (tok/s, prose / JSON / code): RTX 5090 **140 / 221 / 209**.

**Gyro-L** runs in Strata 0.2.10 or newer (community reports in discussions
[#10](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/discussions/10) and
[#12](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/discussions/12)). Our Strata branch
[agentionai-group/strata `main`](https://gitlab.com/agentionai-group/strata/-/tree/main)
(Strata 0.2.4 + Gyro-L / Coder-L kernels + a `tool_choice` fix) also runs it, built from source as in its `docs/GYRO.md`;
Gyro-Coder-L: use our branch for now.
On Strix Halo (HIP, MTP draft): Gyro-S / M / L decode 57 / 58 / 54 tok/s on code, all experts on the GPU, 131k context (262k tested for S and L).
On an RTX 5090 (Windows, community report in [discussion #6](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF/discussions/6)):
Gyro-L passes the kernel parity checks and holds about 100 tok/s decode at 100k tokens of context with Strata's expert cache.

</details>

<details>
<summary><b>strix-llama.cpp</b> (AMD Strix Halo; CPU, Vulkan, HIP)</summary>

```bash
git clone https://github.com/halo-box/strix-llama.cpp && cd strix-llama.cpp
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j --target llama-server   # HIP: -DGGML_HIP=ON -DGPU_TARGETS=gfx1151
./build/bin/llama-server -m ~/models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf -ngl 999 -fa on --jinja --ngram-on-disk \
  -c 65536 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
```

Tested with Gyro-S and Gyro-M (Vulkan); Gyro-L not tested yet.

</details>

<details>
<summary><b>Use it as a coding agent</b> (Marshall)</summary>

Marshall, our open-source terminal coding agent, runs a small team of agents (planner, coder, reviewer) on local or
hosted models, with every action approval-gated by default.

```bash
npm install -g @agentionai/marshall-cli && marshall
```

Inside Marshall, run `/model` and pick your model from the selection menu.

</details>

### Recommended settings

| Setting | Value |
| --- | --- |
| Sampling | `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05` (Qwen's thinking settings plus min-p) |
| n-gram table | keep `--ngram-on-disk` (fast direct reader, Linux and Windows), with the model on fast NVMe |
| KV cache | `-ctk q8_0 -ctv q8_0` |

<details>
<summary>MTP draft, older builds and Windows</summary>

- MTP draft: `--spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-draft.gguf --spec-draft-n-max 6 --spec-draft-n-min 2 --spec-draft-mtp-vocab 32768`
  (adds ~3.75 GiB: on a 32 GB card, 24k context with the draft, 128k without, or the draft on a second GPU).
  The draft was updated on 2026-10-05 to upstream llama.cpp's tensor names (weights unchanged): it needs
  agentionai/llama.cpp main or the container from 2026-10-05 or later; older builds fail to load it, so update.
- Older Windows builds: leave `--ngram-on-disk` out, and the table is read on demand (`--lazy-mode on`).
- Container images from before 2026-10-09 could answer HTTP 500 when several requests run at once with the MTP draft:
  pull the current `agentionai/agention-llama:server` from Docker Hub (on an older image, add `-e LLAMA_GRAPH_SLOTS=0`).
  The image is mirrored on Docker Hub while `ghcr.io/agentionai/agention-llama` is not being updated.
- CUDA builds from before 2026-10-04 could crash with several long prompts at once: update agentionai/llama.cpp (on an older build, `GGML_CUDA_TQ_MMQ=0` or `--parallel 1`).

</details>

## Quality

**Gyro-M v2: same size, closer to the original.** Against Gyro-M v1 it drifts 14% less on held-out text, 27% less on tool-call transcripts (24% less with the system prompts left out of the score), 9% less on code, 10% less on out-of-domain text and 6% less on web text, and it picks the original's top token as often or more often on every text. Against ISTA-DASLab's IQ3_XXS, a 3-bit quant that needs 25% more GPU memory (43.80 GiB), it drifts 20% less on held-out text (0.249 against 0.311).

<p>
<img class="dark:hidden" src="./assets/kld-gmv2-light.svg" width="100%" alt="Mean KL divergence to the original model by text set, lower is closer. Gyro-M v2 / Gyro-M v1: held-out 0.249 / 0.291; tool-call transcripts 0.377 / 0.519; code 0.060 / 0.066; web 0.115 / 0.122; out-of-domain 0.126 / 0.141. Both 34.98 GiB.">
<img class="hidden dark:block" src="./assets/kld-gmv2-dark.svg" width="100%" alt="Mean KL divergence to the original model by text set, lower is closer. Gyro-M v2 / Gyro-M v1: held-out 0.249 / 0.291; tool-call transcripts 0.377 / 0.519; code 0.060 / 0.066; web 0.115 / 0.122; out-of-domain 0.126 / 0.141. Both 34.98 GiB.">
</p>

| Text (mean KL divergence, lower is closer) | Gyro-M v2 | Gyro-M v1 |
|---|---:|---:|
| Held-out | **0.249** | 0.291 |
| Tool-call transcripts | **0.377** | 0.519 |
| Code | **0.060** | 0.066 |
| Web | **0.115** | 0.122 |
| Out-of-domain | **0.126** | 0.141 |
| Top-1 agreement, held-out | **79.41%** | 78.75% |

Both measured on the same llama.cpp build and text sets; none of the texts was used to build the quants.

**Agentic coding: Gyro-M v2 passed 140 of 152 Marshall coding tasks, against 135 for Gyro-M v1** (the same 38 tasks, 4 seeds each, on the same machine and build). On a 30-problem LiveCodeBench v6 subset it is on par with Gyro-M v1 (29 and 28 of 30, same problems and settings).

**Gyro-S v2: same size, closer to the original.** Against Gyro-S v1 it drifts 18% less on held-out text, 27% less on tool-call transcripts, 14% less on code and 11% less on web text. Against ISTA-DASLab's IQ2_XS, which is 16% larger, it drifts 21% less on held-out text, 30% less on tool calls and 20% less on code, and is on par on out-of-domain text.

<p>
<img class="dark:hidden" src="./assets/kld-v2-light.svg" width="100%" alt="Mean KL divergence to the original model by text set, lower is closer. Gyro-S v2 / Gyro-S v1 / ISTA-DASLab IQ2_XS: held-out 0.331 / 0.401 / 0.420; tool-call transcripts 0.451 / 0.617 / 0.641; code 0.094 / 0.109 / 0.118; web 0.190 / 0.214 / 0.196; out-of-domain 0.205 / 0.239 / 0.202. File sizes 58.5, 58.5 and 68.0 GB.">
<img class="hidden dark:block" src="./assets/kld-v2-dark.svg" width="100%" alt="Mean KL divergence to the original model by text set, lower is closer. Gyro-S v2 / Gyro-S v1 / ISTA-DASLab IQ2_XS: held-out 0.331 / 0.401 / 0.420; tool-call transcripts 0.451 / 0.617 / 0.641; code 0.094 / 0.109 / 0.118; web 0.190 / 0.214 / 0.196; out-of-domain 0.205 / 0.239 / 0.202. File sizes 58.5, 58.5 and 68.0 GB.">
</p>

| Text (mean KL divergence, lower is closer) | Gyro-S v2 | Gyro-S v1 | ISTA-DASLab IQ2_XS |
|---|---:|---:|---:|
| Held-out | **0.331** | 0.401 | 0.420 |
| Tool-call transcripts | **0.451** | 0.617 | 0.641 |
| Code | **0.094** | 0.109 | 0.118 |
| Web | **0.190** | 0.214 | 0.196 |
| Out-of-domain | 0.205 | 0.239 | 0.202 |
| File size | 58.5 GB | 58.5 GB | 68.0 GB |

All three measured on the same llama.cpp build and text sets; none of the texts was used to build the quants.

**Agentic coding: on par with Gyro-S v1.** Each file ran the Marshall coding agent on the same 38 tasks, 4 seeds each (152 sessions), on the same machine and build.

<p>
<img class="dark:hidden" src="./assets/marshall-v2-light.svg" width="100%" alt="Marshall coding tasks passed of 152: Gyro-S v2 134, Gyro-S v1 135, ISTA-DASLab IQ2_XS 133.">
<img class="hidden dark:block" src="./assets/marshall-v2-dark.svg" width="100%" alt="Marshall coding tasks passed of 152: Gyro-S v2 134, Gyro-S v1 135, ISTA-DASLab IQ2_XS 133.">
</p>

| Marshall coding tasks | Gyro-S v2 | Gyro-S v1 | ISTA-DASLab IQ2_XS |
|---|---:|---:|---:|
| Passed (of 152) | 134 | 135 | 133 |

**More model per gigabyte.** Each Gyro file stays closer to the original model than the other quants we measured at a
similar size: Gyro-M v2 drifts 20% less than ISTA-DASLab IQ3_XXS with 20% less GPU memory, and Gyro-L is closer to the original than
ggml-org's 68 GiB IQ4_NL at three quarters of its size.

<p>
<img class="dark:hidden" src="./assets/quality-light.svg" width="100%" alt="KL divergence against GPU memory for Qwen3.8-Flash-Next quants. Gyro-S v2 0.331 at 27.64 GiB, Gyro-S v1 0.401 at 27.64 GiB, Gyro-M v2 0.249 at 34.98, Gyro-M v1 0.291 at 34.98, Gyro-L 0.146 at 50.21. Others: ISTA IQ2_XS 0.420 at 36.52, ISTA IQ3_XXS 0.311 at 43.80, unsloth UD-Q2_K_XL 0.334 at 46.63, unsloth UD-IQ3_XXS 0.264 at 49.50, ISTA IQ3_S 0.204 at 51.04, AtomicChat Q4_K_M 0.228 at 52.26, unsloth UD-Q3_K_XL 0.185 at 56.98, ggml-org IQ4_NL 0.200 at 68.21, AP-Q4_K_XL reference 0.107 at 67.38.">
<img class="hidden dark:block" src="./assets/quality-dark.svg" width="100%" alt="KL divergence against GPU memory for Qwen3.8-Flash-Next quants. Gyro-S v2 0.331 at 27.64 GiB, Gyro-S v1 0.401 at 27.64 GiB, Gyro-M v2 0.249 at 34.98, Gyro-M v1 0.291 at 34.98, Gyro-L 0.146 at 50.21. Others: ISTA IQ2_XS 0.420 at 36.52, ISTA IQ3_XXS 0.311 at 43.80, unsloth UD-Q2_K_XL 0.334 at 46.63, unsloth UD-IQ3_XXS 0.264 at 49.50, ISTA IQ3_S 0.204 at 51.04, AtomicChat Q4_K_M 0.228 at 52.26, unsloth UD-Q3_K_XL 0.185 at 56.98, ggml-org IQ4_NL 0.200 at 68.21, AP-Q4_K_XL reference 0.107 at 67.38.">
</p>

| File | GPU memory | KLD ↓ | top-1 ↑ |
|---|---:|---:|---:|
| **Gyro-S v2** (updated 2026-11-24) | **27.64 GiB** | **0.331** | **76.56%** |
| Gyro-S v1 (2026-10-10) | 27.64 GiB | 0.401 | 75.29% |
| **Gyro-M v2** (updated 2026-11-26) | **34.98 GiB** | **0.249** | **79.41%** |
| Gyro-M v1 (2026-10-11) | 34.98 GiB | 0.291 | 78.75% |
| **Gyro-L** (new 2026-10-12) | **50.21 GiB** | **0.146** | **83.87%** |
| ISTA-DASLab GSQ-RCO IQ2_XS | 36.52 GiB | 0.420 | 74.22% |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 43.80 GiB | 0.311 | 77.68% |
| unsloth UD-Q2_K_XL | 46.63 GiB | 0.334 | 76.61% |
| AtomicChat AD-4.27bpw Q4_K_M | 52.26 GiB | 0.228 | 80.58% |
| ggml-org IQ4_NL | 68.21 GiB | 0.200 | 81.29% |
| Agention AP-Q4_K_XL (reference) | 67.38 GiB | 0.107 | **85.52%** |

**Benchmarks** (first Gyro-M release):
MMLU-Pro 80.0 against 81.4 for our 67 GiB AP-Q4_K_XL; LiveCodeBench v6 30/30 given 128k tokens to reason.

<details>
<summary>Gyro vs EXL3 (same build and box)</summary>

Same llama.cpp build (strix-llama.cpp with the EXL3 Vulkan PRs #201/#204), same Strix Halo box, same
held-out and code KLD sets, no MTP; turboderp's
[Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) quants, GPU memory without the
n-gram table for both:

| File | GPU memory | KLD ↓ | top-1 ↑ | code KLD ↓ | decode tok/s |
|---|---:|---:|---:|---:|---:|
| **Gyro-S v1** | 27.64 GiB | **0.399** | **75.33%** | **0.109** | **35.2** |
| EXL3 1.66 bpw | 26.52 GiB | 0.443 | 73.78% | 0.130 | 22.5 |
| **Gyro-M v1** | 34.98 GiB | **0.291** | **78.65%** | **0.066** | **29.5** |
| EXL3 2.18 bpw | 34.93 GiB | 0.300 | 78.28% | 0.073 | 21.7 |
| EXL3 2.05 bpw | 32.92 GiB | 0.315 | 77.83% | 0.075 | 23.2 |
| Gyro-L | 50.21 GiB | 0.146 | 83.94% | **0.026** | **28.4** |
| **EXL3 3.18 bpw** | 49.33 GiB | **0.142** | **84.21%** | 0.028 | 20.8 |

Comparable quality: Gyro is slightly ahead at the S and M sizes and on code, EXL3 slightly ahead on general text at L.
Gyro decodes 1.3-1.6x faster in the same build (EXL3's Vulkan kernels are new; prefill is equal, ~600-700 tok/s).
Both formats build on QTIP; credit to turboderp for EXL3.

</details>

<details>
<summary>Benchmarks in detail</summary>

MMLU-Pro: Gyro-M 80.0, Gyro-S 78.1 (both first release), AP-Q4_K_XL 81.4,
all within noise at our 70-question sample. LiveCodeBench v6 (30-problem subset, one seed): Gyro-M 90% at a 64k
token limit and 30/30 when its long answers get 128k; Gyro-S 67% at 64k, 83% at 128k (both first release; Gyro-S v2 not re-run). Gyro-M v2 on a different 30-problem
subset at a 120k token limit: 29 of 30, on par with Gyro-M v1 on the same problems (28 of 30).

We record every number on this card, with its raw source, in [Fieldwork Ledger](https://www.fieldworkledger.com).

</details>

## Speed

**Up to 221 tok/s on one RTX 5090.** Batch size 1, tok/s:

| Hardware | Engine | Gyro-S decode | Gyro-S prefill | Gyro-M decode |
|---|---|---:|---:|---:|
| RTX 5090, 32 GB | llama.cpp fork main, CUDA | **113** (MTP: JSON 210, code 180, copy 218) | ~3,000 (pp2048) | |
| RTX 5090, 32 GB | Strata `rc1` (release candidate) | prose **140**, code 209, JSON 221 | 4,883 (16k prompt) | prose 114 |
| RTX A6000, 48 GB | llama.cpp fork main, CUDA | 57.4 | 740 (pp2048) | 52.3 |
| Radeon AI PRO R9700, 32 GB | llama.cpp fork, Vulkan | 57.8 (code with MTP: 70.6) | 1,246 (pp512) | |
| AMD Strix Halo | llama.cpp fork, Vulkan | 31.7 (MTP: code 50, JSON 55) | | 26.4 |
| AMD Strix Halo | strix-llama.cpp master, Vulkan | 36.1 | 700 (pp512) | 30.1 |

Speed rows were measured on Gyro-S v1 and Gyro-M v1; the v2 files have the same size and tensor types. RTX 5090 rows: Ryzen 9 9950X host; decode tg128 or greedy content-type prompts with the MTP draft; llama.cpp fork main from 2026-10-04.

## Smaller download: files without the n-gram table

**Already have the table? Download about half.** The n-gram (PLE) table is the base model's, the same for every
Qwen3.8-Flash-Next quant, so it does not need to come with every file.

| | Full file | `-noPLE` file | Table it ships |
|---|---:|---:|---|
| Gyro-S | 58.5 GB | 29.7 GB | IQ4_NL |
| Gyro-M | 92.0 GB | 37.6 GB | Q8_0 |
| Gyro-L | 108.3 GB | 53.9 GB | Q8_0 |

<details>
<summary>How to load a <code>-noPLE</code> file</summary>

About half of each file is the n-gram (PLE) table, which stays on disk and is read row by row. The `-noPLE` files have
the same weights without it; get the table once from
[Delgon/Qwen3.8-Flash-Next-PLE-GGUF](https://huggingface.co/Delgon/Qwen3.8-Flash-Next-PLE-GGUF) and load it with `--ple`.

Delgon's repo has the table in IQ4_NL (28.8 GB), Q8_0 (54.4 GB) and BF16 (102.4 GB); any of them works with any Gyro
file. Q8_0 is what Gyro-M and Gyro-L ship. Gyro-S ships its own IQ4_NL rounding of the table, so Gyro-S with Delgon's
IQ4_NL uses the same table with a different 4-bit rounding: not bit-identical to the full file.

```bash
hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF Qwen3.8-Flash-Next-Gyro-M-TQ2_0-noPLE.gguf --local-dir ~/models
hf download Delgon/Qwen3.8-Flash-Next-PLE-GGUF Qwen3.8-Flash-Next-PLE-Q8_0.gguf --local-dir ~/models
llama-server -m ~/models/Qwen3.8-Flash-Next-Gyro-M-TQ2_0-noPLE.gguf --ple ~/models/Qwen3.8-Flash-Next-PLE-Q8_0.gguf \
  -ngl 999 -fa on --jinja -c 65536 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
```

`--ple` works in agentionai/llama.cpp main from 2026-11-05 (and containers built after that) and in strix-llama.cpp master. Strata takes
the table with `--ple-gguf` (see its `docs/GYRO.md`). Older builds, LM Studio and Ollama cannot load a `-noPLE` file:
use the full file there. Without `--ple` the file stops at load (in agentionai/llama.cpp main with "this model file has no PLE n-gram table"; other builds report a missing tensor).

</details>

## Changelog

**The model files last changed on 2026-11-26 (Gyro-M v2).** If you downloaded Gyro-M before 2026-11-26 or Gyro-S before 2026-11-24,
download it again (and its `-noPLE` file, if you use that). Gyro-L is unchanged since 2026-10-13.

- **2026-11-26: Gyro-M updated to v2.** Same file names (`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` and `-noPLE`), size and tensor
  types; closer to the original: KLD 0.249 (held-out), 0.060 (code), 0.377 (tool-call transcripts). The `-noPLE` file still loads
  with the same Q8_0 table. Download again; v1 stays available as revision `cd23c647`.
  SHA-256: `4791f759c7fbbf4742e8c07249583233d038415d9964f70d4ff476990842e770` (full), `77f29aa35e1378d51877e10825365c0e22b27d7950923cb0bd73fff0832b824e` (`-noPLE`).
- **2026-11-24: Gyro-S updated to v2.** Same file names (`Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` and `-noPLE`), size and tensor
  types; closer to the original: KLD 0.331 (held-out), 0.094 (code). The `-noPLE` file still loads with
  `Qwen3.8-Flash-Next-PLE-IQ4_NL-GyroS.gguf`. Download again; v1 stays available as revision `bdd06693`.
  SHA-256: `1a8267c29026748ac7445e37b2f9d50c27aa30426be8eab70a30c7fa019264f0` (full), `e33a245502a4ae0083ac2064b019f948f5a2cfec6152b1dd6854dd014aee66bb` (`-noPLE`).
- **2026-11-21: card only.** New layout: pick your file first, a quality chart, details folded away. Same numbers,
  commands and files.
- **2026-11-18: new optional file** `Qwen3.8-Flash-Next-PLE-IQ4_NL-GyroS.gguf`: Gyro-S's own n-gram table as a separate
  file, for loading `Qwen3.8-Flash-Next-Gyro-S-TQ1_0-noPLE.gguf` with `--ple`. Model files unchanged.
- **2026-11-05: new optional files** `-noPLE` for Gyro-S, Gyro-M and Gyro-L: the same weights without the n-gram table,
  about half the download. See [Smaller download](#smaller-download-files-without-the-n-gram-table). Full files unchanged.
- **2026-10-24: card only.** Gyro vs EXL3 comparison; the Gyro format specification and reference code are open (Apache 2.0).

<details>
<summary>2026-10-10 to 2026-10-21</summary>

- **2026-10-21: Gyro-Coder released.** Flash-Next for coding on one 24 GB card (19.92 GiB): each layer keeps the
  256 of its 512 experts that matter for code. [agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF).
- **2026-10-19: more engines.** Gyro runs in [Strata](https://github.com/Niko1221/Strata) 0.2.0 and in
  [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) master, besides our own build. Files unchanged.
- **2026-10-13: Gyro-L file updated** with an 8-bit n-gram table: same GPU memory (50.21 GiB), KLD 0.146 (held-out), 0.027 (code).
  The table stays on disk; the download is now 108 GB. Download the file again.
- **2026-10-12: Gyro-L added** (`Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf`, 50.21 GiB GPU weights).
  Needs the `agentionai/agention-llama:server` container from 2026-10-12 or later, or a current build of
  [agentionai/llama.cpp](https://gitlab.com/agentionai-group/llama.cpp) main; older builds cannot load it. Tested on Strix Halo (Vulkan).
- **2026-10-11: Gyro-M updated.** Same file name, size, format and speed; better weights from an improved encode.
  KLD 0.291 (held-out), 0.066 (code).
  Download the file again; the first release stays available as revision `ae0277db`.
- **2026-10-10: Gyro-S updated.** Same file name, size, format and speed; better weights from an improved encode.
  KLD 0.401 (held-out), 0.109 (code).
  Download the file again; the first release stays available as revision `aba94138`.

</details>

Tested on Linux with Strix Halo, R9700, RTX 5090, RTX A6000 and RTX 3090.

## Work with AgentionAI

<p align="center">
  <b>Want results like these on your own model?</b><br>
  <sub>We tune and quantize models for your agents and hardware: fewer tokens, less GPU memory, quality measured against the original. Every engagement ends with a full measurement report. Gyro itself is released freely; AgentionAI is a one-person team, and sponsoring funds the next release and the benchmarking behind it.</sub>
</p>

<p align="center">
  <a href="mailto:hello@agention.ai"><img src="./assets/btn-custom.svg" alt="Custom quants and tunes: hello@agention.ai" height="44"></a>&nbsp;&nbsp;
  <a href="https://github.com/sponsors/agentionai"><img src="./assets/btn-sponsor.svg" alt="Sponsor AgentionAI on GitHub" height="44"></a>&nbsp;&nbsp;
  <a href="https://x.com/laurent_zw"><img src="./assets/btn-follow.svg" alt="Follow @laurent_zw on X" height="44"></a>
</p>

<p align="center"><sub>Or <a href="https://buymeacoffee.com/agentionai">buy me a coffee</a> or some GPU time.</sub></p>

## Credits

- **Hardware:** built and tested on AMD Ryzen AI Max (Strix Halo). Hardware supported by AMD.
- **Format:** trellis-coded quantization with Hadamard rotation, in the family of QTIP (Tseng et al., 2024). Encoder, trellis types and kernels: AgentionAI. The Gyro format is open: specification, reference decoder and encoder, and bit-exact tests (Apache 2.0) at [agentionai-group/agention-gyro](https://gitlab.com/agentionai-group/agention-gyro). The released files come from our own production encoder, which we are not publishing at this stage.
- **Base model:** Qwen/Qwen3.8-Flash-Next. **Runtimes:** llama.cpp, Strata, strix-llama.cpp.
- **Comparison quants:** ISTA-DASLab (GSQ-RCO), unsloth, AtomicChat, ggml-org, turboderp (EXL3).

