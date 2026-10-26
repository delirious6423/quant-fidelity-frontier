---
language:
- en
- zh
license: mit
library_name: vllm
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- exl3
- quantized
- openseek
- rtx-pro-6000
- experimental
inference: false
---

# DeepSeek-V4.1-Flash-1017 · OpenSeek EXL3 2.7bpw

![DeepSeek 1017 2.7bpw eager benchmark: 10.64 tok/s, 5.31-second TTFT and actual memory observations](assets/deepseek-1017-2.7bpw-card.png)

[Download 1600×900 PNG](assets/deepseek-1017-2.7bpw-card.png) · [Editable HTML source](assets/deepseek-1017-2.7bpw-card.html) · [GitHub runbook](https://github.com/LichAmnesia/openseek-recipe/blob/969d0e87d688218d48125417bdceaf85eff40e60/docs/deepseek-v41-flash-1017.md)

**A measured single-card eager port: 10.64 tok/s, 5.31 s TTFT on one RTX PRO 6000 Blackwell Max-Q 96 GB + 128 GB host RAM.**

OpenSeek made sfxnz's mixed-bit 2.7bpw checkpoint run with Diffbot's host-expert
stack on this machine: cold experts in RAM, Engram tables on NVMe, mixed-bit
MCG kernels, and a repaired MXFP8 output head. This release packages the weights,
patches, fixed sources, complete runbook and raw measurements together.

**Experimental first gate.** It passed arithmetic and parsed tool-call smoke
tests, but missed our speed targets. KL, SWE-bench, full 256K quality and real
coding-agent traffic have not been accepted. The weights were quantized by
**sfxnz**; OpenSeek performed no new training, calibration or quantization.

[OpenSeek Recipe](https://github.com/LichAmnesia/openseek-recipe) ·
[Runnable snapshot](recipe/docs/deepseek-v41-flash-1017.md) ·
[Measured evidence](recipe/results/2026-10-26-deepseek-2.7bpw-eager/SUMMARY.md) ·
[Original quantizer's card](upstream/README.md) ·
[Files](https://huggingface.co/LichAmnesia/DeepSeek-V4.1-Flash-1017-OpenSeek-EXL3/tree/main)

## Actual results — 2026-10-26

Hardware: **RTX PRO 6000 Blackwell Max-Q, 96 GB, 300 W**, x86_64 Linux,
128 GB host RAM (125.06 GiB reported), local NVMe, driver 610.57.04.

| Workload, concurrency 1 | Input / output tokens | Decode tok/s | TTFT seconds |
|---|---:|---:|---:|
| C1 trial 1 | 1550 / 256 | 10.5752 | 8.4490 |
| C1 trial 2 | 1550 / 256 | 10.6374 | 5.3096 |
| C1 trial 3 | 1550 / 256 | 10.6641 | 5.2358 |
| **C1 median** | **1550 / 256** | **10.6374** | **5.3096** |
| 34K warmed prefix + 6K new tokens | 40000 / 256 | 10.9388 | 16.2935 |

Local streamed `/completions`, repeated text, `ignore_eos`; decode is
255 divided by elapsed time between the first and last text events. All four
streams returned 256 tokens / 256 text events, `finish_reason=length`.
These numbers describe the reference throughput harness, not useful coding-task
completion speed. Arithmetic `17+25=42` and a parsed `get_weather` call passed.
Prefix-hit counters increased by 33,920, but API `cached_tokens` was absent:
cache activity is confirmed; the request-level reused amount is not.

| Resource observation | Actual value |
|---|---:|
| Inference GPU sampled peak | 97,280 MiB = 95.00 GiB |
| Container kernel peak, includes loading/file cache | 100.000008 GiB |
| Inference sampled container current peak | 99.9990 GiB |
| Inference anonymous-memory peak | 96.2308 GiB |
| Host minimum MemAvailable / total | 16.3907 / 125.0604 GiB |
| Container swap / OOM events / OOM kills | 0 / 0 / 0 |

1,074 one-second resource records; 233 in the correctness/speed phase.
The container had a 100 GiB limit. Its 968,412 `memory.events.max` events are
limit pressure, **not OOM events**. Host total minus available is not model RSS;
do not add peaks taken at different times. [Raw records and provenance](recipe/results/2026-10-26-deepseek-2.7bpw-eager/).

The original gates were decode >=20 tok/s and warmed agent-shaped TTFT <=6 s.
This port **missed both**. It demonstrates that the checkpoint can run on this
configuration; it is not yet a qualified resident coding-model replacement.

## Where the GPU memory, RAM and SSD go

In plain English, this run uses **95 GiB GPU memory + about 100 GiB host RAM + NVMe SSD** together.

| Location | Amount | What lives there |
|---|---:|---|
| GPU memory | **95 GiB** sampled peak | About **79 GiB** of expert weights in the allocation plan, plus other model layers, KV cache and compute workspace |
| Host RAM | About **100 GiB** container peak, including loading/file cache | About **92 GiB** for **8,220 experts**, plus runtime buffers and file cache |
| NVMe SSD | About **405 GB** for the complete checkpoint | Includes about **203 GB of Engram tables**, retained on SSD and read on demand |

The RAM tier stores expert weights; **the GPU still performs their computation**,
accessing those weights over PCIe. CPU-expert compute was disabled in this run.
The expert trellis allocation plan was 78.8599 GiB on GPU and 91.9995 GiB in RAM;
these components exclude small scales and are separate from the measured total peaks.
The GPU's total also includes the 1.1 GiB KV allocation and other runtime memory.

**16.39 GiB in the figure is remaining available host RAM.** The host reported
125.0604 GiB total; total minus minimum available was 108.6697 GiB (about109 GiB),
including the OS and other workloads. The model container's measured peak was
about100 GiB, including file cache and earlier loading. Peaks are observed independently.

The full checkpoint is 405.05 GB (377.23 GiB); its two Engram shards total
203.07 GB (189.13 GiB), already included in that total. Other weight files also
remain on SSD after loading. Docker images/build caches require additional space.
Engram rows are read on demand from the NVMe-backed files; the OS can cache rows.
Measured container **swap was0**: SSD participates through explicit model-table reads.
These are the archived run's measurements and allocation plan, not a new benchmark.

中文速读：本次同时使用约95 GiB显存、约100 GiB电脑内存和NVMe SSD。
约92 GiB内存放8,220个专家，仍由GPU计算；16.39 GiB是剩余可用内存。
405 GB模型文件已包含203 GB Engram，SSD按需提供模型数据，容器swap为0。

[GitHub allocation notes](https://github.com/LichAmnesia/openseek-recipe/blob/667fcf54b49e1e8b0e23b379b0869f7ef1a1d53b/docs/deepseek-v41-flash-1017.md#where-the-gpu-memory-ram-and-ssd-go).

## What OpenSeek adds

1. **Mixed-bit host-expert port.** Per-layer gate/up versus down widths,
   K0/K4 MCG support and host allocation for the sfxnz 2.7bpw layout.
2. **Disk Engram path.** Explicit disk allocation, loader skip and eager lookup;
   the large tables stay on NVMe to fit the 128 GB host.
3. **MXFP8 output-head repair.** Correct namespace/scale allocation and Diffbot
   sm120 W8A16 compute. The first full load failed with
   `There is no module or parameter named 'lm_head' in DeepseekV41ForCausalLM`.
   After repair, an exact 188,246-key mapping check changed only two head names;
   1,024 real checkpoint head rows passed the numeric gate (max relative L2
   0.002588 versus the BF16 dequantized reference).
4. **Reproduction and evidence.** 152 SHA256-locked small source/config files,
   the complete 55-file checkpoint manifest, preparation/build/serve scripts,
   numeric probes, archived benchmark and memory trace. The recipe's full
   offline suite passed 37 tests when packaged.

The successful model run was on the original machine. Portable packaging has
offline checks, not a second fresh-machine GPU qualification.

## Run this release

Prerequisites: NVIDIA Docker on x86_64 Linux, CUDA13-capable driver, Python3.10+,
`git`, `curl`, `nvidia-smi`, Hugging Face CLI, and sufficient NVMe space for
**405,050,217,567 source bytes (377.2 GiB)** plus build/images/cache.
Install the HF CLI using its official instructions if absent. The tested
downloader was `huggingface-hub==2.0.0`, `hf-xet==1.6.0`, four workers.
Do not run competing downloaders; restarting an unfinished huge shard with that
HF version loses its UUID temporary transfer progress.

```bash
export PACK="$HOME/models/DeepSeek-V4.1-Flash-1017-OpenSeek-EXL3"
export DSV41_ROOT="$HOME/.local/share/openseek/dsv41-2p7-eager"
hf download LichAmnesia/DeepSeek-V4.1-Flash-1017-OpenSeek-EXL3 \
  --revision v0.1.0-eager --max-workers 4 --local-dir "$PACK"

export EXP="$PACK/recipe/experiments/deepseek-v41-flash-1017"
python3 "$EXP/prepare.py" --root "$DSV41_ROOT" --dry-run
python3 "$EXP/prepare.py" --root "$DSV41_ROOT"

# Use this release's mapped manifest: the quantizer's README is upstream/README.md.
# This hashes every LFS object, including all 48 weight shards, without loading tensors.
python3 "$DSV41_ROOT/scripts/verify_weights.py" \
  "$PACK/release-model-files.json" "$PACK" \
  "$DSV41_ROOT/logs/weight-verification.json"
```

Then follow **sections 3–6** of the bundled
[complete runbook](recipe/docs/deepseek-v41-flash-1017.md): build, numeric gates,
normal idle GPU handoff, serve, measure, release GPU and restore the prior
service on success or failure. Skip its upstream download section when using
the verified copy above. The main `./recipe` CLI is still Qwen-specific;
DeepSeek uses this standalone experimental entry point.

```bash
# After the runbook's build, handoff and numeric gates:
bash "$EXP/serve.sh" --dry-run
bash "$EXP/serve.sh"
```

`serve.sh` runs in the foreground and refuses another GPU compute process,
insufficient available RAM, failed verification or an existing same-name
container. It does not stop another service. It uses localhost:8010, model
`dsv41-1017-2p7-test`, benchmark key `local-benchmark-only`, and retains its
container on exit. Use an SSH tunnel for a remote client. First load on the
measured host took about **14 minutes**; `48/48 shards` does not mean API ready.

For the already prepared original host, reuse its verified weights and image;
do not download another 405 GB or restart an old automated gate unnecessarily.
Host-specific HeyDreaming credentials/cloud checks are not included.

## Exact serving scope

| Setting | Measured configuration |
|---|---|
| Parallelism / execution | TP1, eager; CUDA graphs disabled |
| Experts in host RAM | 8,220; 91.9995 GiB trellis; deterministic bootstrap, extra IDs unprofiled |
| Engram | FP8 tables on local NVMe |
| Activations / KV | BF16 / FP8; fixed KV allocation 1.1 GiB |
| Max context / batch / sequences | 262144 / 512 / 1 |
| Container memory / swap | 100 GiB / 0 |
| Disabled paths | Vision, DSpark, host cache, CPU experts, DMA, decode-once |

`EXL3_HOST_COLD_PACK` must be empty; `0` is interpreted as a filename.
Setting max context to 256K does not validate output quality at 256K.
No short100 KL, SWE-bench, CUDA graphs, optimized routing or real-agent
acceptance is claimed. The local serving image is not published/pullable;
the pinned base and source lock support rebuilding, but apt/compiler artifacts
are not fully locked. A new build needs its own numeric/API gates.

## Frozen inputs and downloadable artifacts

| Component | Revision / identity |
|---|---|
| sfxnz checkpoint | `a8006e99931e7d67c4d218c66543f478b521b857` |
| Diffbot serving recipe | `2c3797578722747dad3593531fd447b1d3d4c732` |
| sfxnz helpers | `19e0c56620a7b9748fd901cdf3ef533a0f6ed917` |
| OpenSeek Recipe base commit | `3d1a2fc3683a9335df25960968f14e7cf30d2c3a` |
| Measured local image ID | `sha256:5c5a4d3e1bbf46aa24db3c6da126cfa3c628076137dccdf5935d81b912443664` |

The DeepSeek changes were local/uncommitted at publication. The actual portable
snapshot is hosted **inside this HF repository**; this card does not assume
those new files are already on GitHub main. Per-file hashes freeze the snapshot.

- [runtime-lock.json](runtime-lock.json): source revisions, base digest, measured image identity and release settings.
- [release-manifest.json](release-manifest.json): 55 original assets, mapped paths, storage identities and portable snapshot hashes.
- [release-model-files.json](release-model-files.json): runtime verifier input for this distribution.
- [SHA256SUMS](SHA256SUMS): original assets and added release files; `sha256sum -c SHA256SUMS` checks all payloads.
- [recipe/](recipe/): runnable snapshot, patches, probes and raw measured evidence.

## Attribution and licenses

Original model: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
Quantized checkpoint: [sfxnz](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3/tree/a8006e99931e7d67c4d218c66543f478b521b857).
Host-expert serving foundation: [Diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/tree/2c3797578722747dad3593531fd447b1d3d4c732).

All 55 source files are preserved byte-for-byte. The source README is at
`upstream/README.md` and its original Git attributes at `upstream/.gitattributes`.
The root `.gitattributes` retains the original rules plus HF's new PNG LFS rule;
this root card is OpenSeek's documentation. All 48 weight shards retain their
original LFS SHA256. The original checkpoint [MIT LICENSE](LICENSE)
is unchanged. Model metadata retains its original capabilities, but this runtime
was tested text-only with vision disabled.

The model's MIT label does not relicense the runtime. The mixed-bit plugin patch
and generated plugin modifications are **AGPL-3.0-only**; other components retain
their respective licenses. See [NOTICE.md](NOTICE.md),
[runtime notices](recipe/experiments/deepseek-v41-flash-1017/THIRD_PARTY_NOTICES.md)
and [AGPL text](recipe/experiments/deepseek-v41-flash-1017/LICENSE-AGPL-3.0.txt).
