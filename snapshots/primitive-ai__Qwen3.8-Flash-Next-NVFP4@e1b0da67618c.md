---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- nvfp4
- quantized
- vllm
- modelopt
- qwen3.8
- flash-next
- single-gpu
- speculative-decoding
thumbnail: https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4/resolve/main/assets/banner.png
---

<p align="center">
  <img src="assets/banner.png" alt="Primitive — Qwen3.8-Flash-Next NVFP4" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/size-186%20GB-F98424?style=flat-square" alt="size 186 GB">
  <img src="https://img.shields.io/badge/formats-NVFP4%20experts%20+%20BF16-F98424?style=flat-square" alt="formats NVFP4 experts + BF16">
  <img src="https://img.shields.io/badge/runtime-vLLM%20image,%20stock-F98424?style=flat-square" alt="runtime vLLM image, stock">
  <img src="https://img.shields.io/badge/hardware-one%2096%20GB%20Blackwell-F98424?style=flat-square" alt="hardware one 96 GB Blackwell">
  <img src="https://img.shields.io/badge/license-Qwen%20Community%201.0-F98424?style=flat-square" alt="license Qwen Community 1.0">
  <img src="https://img.shields.io/badge/fits-1x%2096%20GB%20GPU%20+%20100%20GB%20RAM-3DD13D?style=flat-square" alt="fits 1x 96 GB GPU + 100 GB RAM">
  <a href="https://primitive.com"><img src="https://img.shields.io/badge/web-primitive.com-lightgrey?style=flat-square" alt="primitive.com"></a>
</p>

<h3 align="center">The 180B Flash-Next on one GPU.</h3>

<p align="center">
Qwen3.8-Flash-Next is 360&nbsp;GB in BF16 and needs two data-center GPUs in FP8.<br>This build serves it on <b>one 96&nbsp;GB Blackwell</b>: 88.8&nbsp;GiB of VRAM, the 51B n-gram table in host RAM, and no patched runtime.
</p>

---

## Why this quant

* 🖥️ **One 96 GB GPU.** 88,828 MiB of VRAM at serve: NVFP4 experts plus the BF16 tail. The n-gram table lives in host RAM (~100 GB) with async prefetch. Any checkpoint that keeps the table in BF16 can take this path, the W4A16 builds measured further down included; the FP8-table checkpoints (official FP8, RadixArk, Inferact) cannot on the shipped image.
* 🎯 **92.2 knowledge on a 1,170-item, 9-benchmark suite** (n=2 runs: 92.1, 92.2), 99.4% finished, zero request errors. GSM8K 98.0, MMLU-Pro 90.0.
* 🔧 **84.6 call / 56.7 abstain** on the 200-item tool-calling suite (n=3). It makes correct calls at the same rate as much smaller strong models and never declines to call — the abstention items are what pull a pooled number down, which is why both halves are printed.
* ⚡ **74.4 tok/s single-stream (12.2 ms/token), 483.8 tok/s at concurrency 32**, measured prefix-cache-free with distinct seeds; the two seeds agreed within 0.04%.
* 🔀 **MTP speculative decoding preserved.** All 31 MTP tensors byte-identical to the source; `{"method":"mtp","num_speculative_tokens":3}` works as Qwen documents.
* 🧩 **Stock image, no patches.** `vllm/vllm-openai:qwen38-flash-next` exactly as published. The serve block below carries the two flags that make single-GPU work — without them the server hangs silently or times out. One optional one-file overlay (further down) fixes a flaky upstream startup race that the image predates.

---

## Serve it

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  vllm/vllm-openai:qwen38-flash-next \
  --model primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-seqs 36 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Two flags are load-bearing on a single GPU. `--distributed-executor-backend mp`: the default single-GPU executor never starts the n-gram offload worker, and the first forward waits on it forever — the server looks healthy and hangs. `VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`: the worker loads a 95 GB table before serving and the 600 s default can expire first. Host needs about 100 GB of free RAM. The reasoning parser and tool-call parser in this block are validated on this exact checkpoint: thinking lands in `reasoning` with no markup in `content`, and tool calls arrive as structured `tool_calls` with valid JSON arguments and `finish_reason: tool_calls`. If you have seen other NVFP4 cards warn that the automatic MoE backend picks `FLASHINFER_TRTLLM` and corrupts output: on this image on an RTX PRO 6000 the automatic choice is `FLASHINFER_CUTLASS`, printed at boot, and every number on this card went through it. Measured on the mixed sibling (same expert kernel), `--moe-backend marlin` decodes 10% faster single-stream (93.2 vs 84.6 tok/s) and 1–3% slower at concurrency 32, with TTFT up 12%; tool-calling under Marlin 79.5 (n=1). Marlin does not combine with MTP on this image: the BF16 MTP draft inherits the backend and refuses it, and the draft-scoped override in `--speculative-config` is ignored by Model Runner V2 until vllm-project/vllm#54788 lands. Choose MTP for interactive use, Marlin only without speculation. On vLLM nightly builds from early September the SM120 selector picks Marlin by default anyway ([vllm#54788](https://github.com/vllm-project/vllm/pull/54788#issuecomment-5554777896)); the flag matters on this image. Two more flags are in the block since 2026-09-06: without them the image's defaults do not boot on a 96 GB card. A [field report](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/discussions/3) found it first and we reproduced both failures on this box the same day. Without MTP the default `--max-num-seqs` (1024) exceeds the Mamba state blocks left at 0.92 utilization (598 here) and CUDA graph capture aborts with `Engine core initialization failed`; with MTP at the native 262,144 context the KV cache needs 7.57 GiB and 5.95 GiB is left after the draft head. 32,768 / 36 is what every number on this card was measured with; the report ran 135,168 / 64 with MTP on a 64 GB host.

---

<!-- pledisk:begin -->
## Not enough host RAM? Put the table on NVMe

The serve command above wants ~100 GB of free host RAM for the n-gram table. With fast local
storage you can skip that: this repo ships a one-file overlay ([`worker_image_disk.py`](./worker_image_disk.py))
that maps the table from a file instead. First boot writes 95.4 GB into the store directory;
every later boot maps it instantly and skips the table's checkpoint reads.

```bash
hf download primitive-ai/Qwen3.8-Flash-Next-NVFP4 worker_image_disk.py --local-dir .
mkdir -p pledisk_store

docker run --gpus all --ipc=host -p 8000:8000 \
  -v $PWD/worker_image_disk.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
  -v $PWD/pledisk_store:/pledisk_store \
  -e VLLM_PLE_DISK_OFFLOAD_DIR=/pledisk_store \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  vllm/vllm-openai:qwen38-flash-next \
  --model primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-seqs 36 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Measured on the [mixed sibling](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8) — both repos ship byte-identical BF16 tables and the same worker path, so the disk behavior transfers; absolute tok/s columns are the sibling's. 8K in / 512 out, prefix-cache-free, two seeds per cell (shown a / b):

| config | boot | tok/s @ 1 | tok/s @ 32 | median TTFT @ 1 |
|---|---|---|---|---|
| table in RAM (command above) | 302 s | 84.5 / 84.4 | 516.8 / 523.6 | 569 / 573 ms |
| disk, container capped to 48 GB RAM — **recommended** | 263 s | 79.4 / 76.8 | 427.0 / 435.8 | 571 / 573 ms |
| disk, uncapped 176 GB host | 303–344 s | 50.4–62.1 | 196.7–396.7 | 1.8–2.9 s |
| disk, cold page cache | 404 s | 40.8 / 37.0 | 134.2 / 290.3 | 4.6 / 5.2 s |
| disk, first boot (writes the file) | 504 s | — | — | — |

Net cost of the disk path, run capped: **−8% single-stream, −17% at concurrency 32, TTFT
parity** with the RAM baseline. The counterintuitive row is the uncapped one, and it
reproduces across two boots and four seeds: without a container memory cap, the boot's own
172 GB checkpoint streaming flows through the global page cache and evicts the table it is
about to need, so gathers fault back to NVMe mid-decode. A memory cap makes reclaim
cgroup-local — the container's checkpoint reads can only evict the container's own cache, and
the table stays resident. So on the disk path, **always cap the serving container** (48 GB is
what we validated; `--memory 48g --memory-swap 48g`).

On a host with less than about 100 GB of RAM plus swap, also set `sysctl vm.overcommit_memory=1`. The offload worker constructs the 95 GB BF16 table parameter before either overlay swaps in the disk map or the quantized store, and the kernel's default heuristic refuses a single allocation larger than RAM + swap (`DefaultCPUAllocator: can't allocate memory` from the PLE worker). Our 48 GB validation was a cgroup cap on a 176 GB host, which exercises page-cache residency but not this check; the failure was reported from a 64 GB host ([discussion](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/discussions/3)). For the quantized tables this is fixed: the current `ple_layer_quant.py` in the [PLE-quant repo](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) builds that parameter on the meta device, so only the BF16 disk path still needs the sysctl.

Accuracy is unaffected — the mapping serves the same bytes. Inside the 48 GB cap the 200-item
tool-calling suite scored 78.5 with zero request errors and zero truncations (repeat spread on
this suite: 78.0–80.5), and the generation-sanity gate passed on the first-boot and capped
configurations. Cold cache is a floor, not a steady state: the two cold @ 32 runs went
134 → 290 tok/s back to back as the cache refilled. Boot times share one caveat: all were
measured with the checkpoint at least partially page-cache-resident; a truly cold first read
of the 172 GB weights adds its own disk time to any of them.

The overlay targets this exact image. The same change is a draft PR to vLLM —
[vllm-project/vllm#54070](https://github.com/vllm-project/vllm/pull/54070), branch
[`feat/ple-disk-offload`](https://github.com/jagat-primitive-org/vllm/tree/feat/ple-disk-offload) — stacked on the PLE CPU-offload PR (vllm-project/vllm#53899).

Where upstream stands (2026-09-03): the architecture merged into vLLM main on 2026-08-31
([#53896](https://github.com/vllm-project/vllm/pull/53896)), so nightly builds serve this model with the table on GPU, which means TP2
or more. The CPU-offload path ([#53899](https://github.com/vllm-project/vllm/pull/53899)) is still open, so single-GPU serving still
means this image or that branch. A second upstream route, [#54129](https://github.com/vllm-project/vllm/pull/54129), memory-maps the
FP8 table straight out of the checkpoint shards; its thread reports 87.3 tok/s single-stream at
TP1 on an RTX PRO 6000 with RadixArk's NVFP4, on par with the table resident in RAM on that box.

**Known image bug, fix included.** The current image (`sha256:fc120ece…`, unchanged as of
2026-09-03) predates an upstream
fix for a startup race in the PLE offload path: vLLM can hang right after CUDA graph capture,
looping `No available shared memory broadcast block found in 60 seconds`
([vllm-project/vllm#53960](https://github.com/vllm-project/vllm/issues/53960); fixed on
2026-08-29 in the offload PR branch, #53899, which is still unmerged, so the issue stays
open). This repo ships the fixed connector as
[`connector_mrv2.py`](./connector_mrv2.py) — mount it alongside any of the serve commands here,
with or without the other overlays:

```bash
  -v $PWD/connector_mrv2.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro ```

Verified on this build with the fix mounted: normal boot, sanity gate passed, accuracy and
throughput unchanged.

**Same symptom, different cause: `pidfd_getfd` permissions.** The CUDA IPC handshake between the GPU worker and the offload subprocess imports a file descriptor with `pidfd_getfd(2)`. Where that call is denied the boot hangs the same way (after graph capture, `shm_broadcast` timeouts, no error naming the cause). Two reports on the offload PR: a managed container host whose seccomp profile denies it without `CAP_SYS_PTRACE` ([details and a ten-line preflight probe](https://github.com/vllm-project/vllm/pull/53899#issuecomment-5448518741)), and bare metal with the Yama default `kernel.yama.ptrace_scope=1` ([details](https://github.com/vllm-project/vllm/pull/53899#issuecomment-5553027248)). Fixes: `--cap-add=SYS_PTRACE` on the `docker run`, or on bare metal `sysctl kernel.yama.ptrace_scope=0` or `setcap cap_sys_ptrace+ep` on the Python binary. Neither our box nor the 64 GB-host field report needed any of this with the plain `docker run` above.

---

## Quantized PLE tables: 49 / 32 / 28.8 GB instead of 95 GB

The table itself also quantizes well. We publish it in **FP8 per-row (49 GB)**, **INT4
group-16 (32 GB)**, and **NVFP4-style e2m1 group-16 (28.8 GB)**, served memory-mapped by a
two-file overlay — host RAM cost becomes page cache only, no container cap needed. Accuracy
holds on both suites for all three (knowledge 92.2 / 92.9 / 92.2 vs 92.2 BF16; tool-calling
n=3 means 77.7 / 78.2 / 78.7 vs 79.2, one ±1.5 band), throughput lands within 5–6% of the
in-RAM BF16 baseline, and MTP keeps most of its speed-up (129.6 tok/s single-stream with the
INT4 table, 128.8 with NVFP4, vs 142.6 in-RAM). Validated end to end inside a 48 GB container:
tool-calling 80.5, 79.4 / 486 tok/s — a 64 GB-RAM host serves this model. Tables, overlay
files, serve command, format spec, and the full measurement table:
[primitive-ai/Qwen3.8-Flash-Next-PLE-quant](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant).

---

## Speculative decoding (MTP)

The MTP tensors are preserved byte-identical, so vLLM's built-in draft path works — add:

```bash
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Real-prompt A/B (measured on the mixed sibling), single stream, thinking on (60-item subset of our eval,
decode rate = output tokens over wall time — not comparable to the bench-serve numbers above,
and measured on real prompts because random-token benches overstate speculative gains):

| speculative config | decode tok/s | strict score |
|---|---|---|
| none | 91.2 | 91.7 |
| `num_speculative_tokens: 1` | does not boot | — |
| `num_speculative_tokens: 2` | 133.2 | 93.3 |
| `num_speculative_tokens: 3` | **142.6 (+56%)** | 93.3 |
| `num_speculative_tokens: 3`, table on NVMe | 77.5–82.3 | 93.3 |

Score differences are single-run noise on 60 items; the speed difference reproduces.
`num_speculative_tokens: 1` hangs this image at startup — the engine core blocks in
`shm_broadcast` before the API server binds, reproduced on three boots (one waited 80 minutes).
Use 3.

One caveat from a field report ([Marzero, 2026-09-06](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench)): with thinking on and MTP on, grammar-constrained tool calls logged xgrammar `Failed to advance FSM` at the `</think>` to `<tool_call>` boundary (33 times over about 2,100 requests; every request still returned HTTP 200 and the tool-eval error rate stayed 0.0), and with MTP off the count was 0 in 2,093. On the same build `tool_choice="required"` was not enforced in streaming with thinking off. Both are vLLM-side; if you constrain tool-call output with a grammar, measure before turning MTP on.

---

## The other 4-bit quants, measured on the same box

When we first ran this (2026-08-28), every vLLM-lane quant of this model was experts-only 4-bit
and these were the four public ones; nota-ai's calibrated NVFP4 was added on 2026-09-03 in a
separate sitting. Same protocol for all: one RTX PRO 6000, 200-item tool-calling suite,
cache-free 8K/512 bench, table in host RAM.

| build | expert format | tok/s @ 1 | tok/s @ 32 | TTFT @ 1 | tool-calling |
|---|---|---|---|---|---|
| our mixed build | NVFP4 g16 + FP8 attention | **84.4** | **520** | **570 ms** | 79.2 (n=3) |
| [Intel AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | INT4 g128 sym, iters=200 | 82.6 | 482 | 683 ms | 80.5 (n=1) |
| [lvkaokao RTN](https://huggingface.co/lvkaokao/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound) | INT4 g128 sym, RTN | 82.4 | 481 | 683 ms | 80.0 (n=1) |
| [cyankiwi AWQ](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) | INT4 g32 asym, AWQ | 81.6 | 422 | 680 ms | 78.0 (n=1) |
| [wtdcode AWQ](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) | INT4 g128 sym, AWQ | 82.6 | 484 | 682 ms | 76.0 (n=1) |
| [nota-ai NVFP4](https://huggingface.co/nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4) (added 2026-09-03) | NVFP4 g16, calibrated W4A4 | 74.5 | 483 | 624 ms | 78.0 (n=1) |
| [nvidia NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (added 2026-09-06) | NVFP4 g16, static-scale W4A4 | 71.5 | 451 | 665 ms | 78.3 (n=2) |

Single tool-calling runs carry a ±1.5 spread, so the middle of this column is one band; the
AWQ-vs-RTN-family gap at the edges is larger than that. Two patterns hold regardless: every
4-bit expert format decodes at ~82 tok/s single-stream with ~680 ms TTFT — the speed
difference in the top row is the FP8 attention, not the expert format — and the W4A16 builds
run on pre-Blackwell GPUs, which NVFP4 does not serve natively. All four keep the n-gram table
in BF16, so the [quantized-table sidecars](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)
apply to them as well.

The nota-ai row is the one calibrated NVFP4 in the field (512 conversations of 4K tokens, W4A4 with
a calibrated activation global scale) and the only one we also ran through the knowledge suite:
92.1 strict / 92.6 completed, 99.3% finished, zero errors, against 92.2 for our uncalibrated
round-to-nearest NVFP4; tool calling 83.8 call / 55.0 abstain against our 84.6 / 56.7. Speed is identical to our plain build (74.5 vs 74.4 tok/s single-stream,
483 vs 484 at concurrency 32) because the tensors that move at decode are the same format. On this
suite, at this precision, calibration did not buy anything measurable. Their card's greedy-decoding
table shows NVFP4 at or above BF16 on three of five benchmarks; they call that sampling noise, and
our numbers agree with the reading.

NVIDIA's own NVFP4 build, added on 6 September, is the third calibration strategy on the same
experts: static per-expert activation scales fitted by minimising squared error on CNN/DailyMail
and Nemotron post-training text, against nota-ai's calibrated dynamic scales and our
round-to-nearest weights with a trivial activation scale. All three land in the same place:
knowledge 92.1 / 92.1 / 92.2 strict, tool calling 78.3 / 78.0 / 79.0, each on this box under this
protocol. Their published table compares NVFP4 against the FP8 release across nine harder
benchmarks and lands at or above it on five, which reads the same way. What does move between
these builds is speed, and it comes from what else is quantized rather than from how the experts
were fitted.

Two notes on that row. Their card documents an 8-GPU deployment, because the checkpoint keeps
the n-gram table in FP8 and vLLM's host-side offload accepts only BF16 tables on the image these
numbers come from. We served it on one card by supplying the table from the
[INT4 sidecar](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) instead, which holds the same values; the row is therefore their experts
with our table, and their MTP module (FP8, 128-block) was not loaded. And at 71.5 tok/s
single-stream it is the slowest build in this table, about 4% behind the other two NVFP4 builds
at concurrency 1 and 7% behind at concurrency 32, reproduced across two seeds that agree to
0.1%. We have not established why. Not yet measured here:
[btbtyler09's GPTQ g32](https://huggingface.co/btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit) (INT4 on the
experts, the shared experts and the QSA projections; BF16 table) and [codegyver's NVFP4 + FP8 weight-only QSA/GDN
build](https://huggingface.co/codegyver/Qwen3.8-Flash-Next-NVFP4-FP8-PBWO-DGX-Spark) for DGX Spark
(FP8 table, so not single-GPU on this image). One more idea from the field:
[hampsonw](https://huggingface.co/hampsonw/Qwen3.8-Flash-Next-W4A16-AutoRound-MTP-INT4-Experts)
ships an INT4 delta for the 512 experts inside the MTP head (5.2 GB of draft weights down to 1.5 GB
per GPU) on top of the Intel build, with acceptance unchanged in their runs. The MTP head here stays
BF16.

One kernel finding from the same sitting, on our mixed build: the image auto-selects the
`FLASHINFER_CUTLASS` NVFP4 MoE kernel on this card, and `--moe-backend marlin` (weight-only dequant
into a BF16 GEMM) decodes faster single-stream: 93.2 tok/s against 84.6, two seeds each, at a cost
of 1–3% at concurrency 32 (502–512 vs 516) and 12% on TTFT (635 vs 567 ms). Tool-calling under
Marlin: 79.5 (n=1; suite band ±1.5). Details in the serve section above.

MTP and the disk-backed BF16 table do not combine well: speculation multiplies table-gather
traffic, the working set outgrows the page cache, and the +56% collapses to roughly the
no-speculation rate (capped 77.5, uncapped 82.3). With the BF16 table in RAM, use MTP; on a
low-RAM host, pair MTP with the
[INT4 quantized table](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)
instead, which keeps 129.6 tok/s.
<!-- pledisk:end -->

---

## Measured

**1,370 items across fourteen public benchmarks.** A 1,170-item knowledge suite (MMLU-Pro, ARC-Challenge, HellaSwag, WinoGrande, CommonsenseQA, BoolQ, OpenBookQA, GSM8K, MATH-500) and a 200-item tool-calling suite (BFCL v4, xLAM/APIGen, ToolACE, Glaive v2, nvidia When2Call), under one fixed protocol: `temperature 0.6` / `top_p 0.95` / `top_k 20`, thinking forced on, a 16,384-token budget, no reasoning parser, the last `ANSWER:` scored. Concurrency 32 on the same single RTX PRO 6000 Blackwell the fit numbers come from. Auto-scored, no LLM judge.

| build | size | overall | knowledge | call | abstain | runs k/a | finished | out/answer | tok/s @ 32 | tok/s @ 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **this repo** | **186 GB** | **90.2** | 92.2 | 84.6 | 56.7 | 2/3 | 99.4% | 664 tok | 483.8 | 74.4 |

**`overall` is one number over both suites**: the 1,170 knowledge and 200 tool-calling items pooled as 1,370, weighted **85.4%** and **14.6%** by item count. `call` is accuracy on the 160 tool-calling items that require a call; `abstain` is the 40 whose correct action is to call **nothing** — they are never pooled into one number on our cards, because a model can be strong at one and weak at the other, and this one is exactly that: solid call accuracy, zero abstentions in the system-prompt tool format. Through the native `tools=` API it does abstain (validated above), so weight the two columns by how your application passes tools.

Same-box numbers for the community W4A16 builds are in the comparison table further up; the section below covers what single-GPU serving requires of a checkpoint.

Knowledge is a mean of 2 runs (92.1, 92.2); tool-calling of 3 (78.5, 80.0, 78.5 pooled). Throughput is prefix-cache-free — `--random-prefix-len 0`, a distinct seed per run, warm-up seeded apart — and two seeds agreed within 0.04%. On our other models this suite's repeat spread runs to ±0.5 on knowledge and ±1.5 on tool calling; treat gaps inside that as ties.

---

## Single-GPU serving: what a checkpoint needs

On the shipped image, `VLLM_PLE_CPU_OFFLOAD` builds the CPU-side n-gram table as plain BF16, so
serving on one 96 GB card there needs a checkpoint that ships the table in BF16. This build does,
and so do the community W4A16 builds measured above and [nota-ai's NVFP4](https://huggingface.co/nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4), which we
booted on this card the same way (394 s to healthy, 88,828 MiB, the same footprint as this build;
their own card lists two cards as the minimum). With about 100 GB of free host RAM they can all
take the single-card path (or far less RAM with the
[quantized-table sidecars](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant): 49 / 32 / 28.8 GB, page-cache only). Checkpoints that store
the table as FP8 with a scale tensor, the official FP8 release and the NVFP4 conversions derived
from it, need TP2 or more on this image.

That constraint is loosening upstream. The offload PR branch ([#53899](https://github.com/vllm-project/vllm/pull/53899)) has loaded FP8
and NVFP4 global-scale tables since 2026-08-30, and its thread confirms RadixArk's NVFP4 booting at
TP1 on an RTX PRO 6000 with no patches on that branch. Outside vLLM, garnermccloud's
[SGLang fork](https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream) streams RadixArk's FP8 table from SSD and reports 164.7 tok/s single-stream
with MTP on one RTX PRO 6000 (their prompts, 1,024-token completions, not our protocol). llama.cpp
has had the architecture on main since 2026-08-27, so the GGUF lane is real now; GGUF and MLX
conversions still do not serve on vLLM.

---

## What's quantized to what

| tensors | format |
|---|---|
| all 48 layers' routed experts (`gate/up/down_proj`, 120.8B params) | NVFP4 (group 16) |
| n-gram embedding table (51.2B, 128 shards) | BF16, pre-scaled — the only table format the shipped image's offload worker loads (the PR branch has since added FP8 and NVFP4 global-scale tables) |
| attention, GDN linear-attention, shared experts, routers, MTP, vision, embeddings, norms | BF16, byte-identical to the source |

Weights-only round-to-nearest, no calibration. The n-gram table carries the FP8 release's values, materialized in BF16 — the same numbers every runtime materializes at load.

## Comparable with our other models

Accuracy numbers move for reasons that have nothing to do with the model: a shorter token budget, a
different `temperature`, or whether the model was allowed to reason at all. So every number in this
table, on this card and on our other cards, comes from one fixed protocol.

The same 1,370 items: a 1,170-item knowledge suite (MMLU-Pro, ARC-Challenge, HellaSwag, WinoGrande,
CommonsenseQA, BoolQ, OpenBookQA, GSM8K, MATH-500) and a 200-item tool-calling suite (BFCL v4,
xLAM/APIGen, ToolACE, Glaive v2, nvidia When2Call). `temperature 0.6`, `top_p 0.95`, `top_k 20`,
thinking forced on, a 16,384-token budget, no reasoning parser, scoring the last `ANSWER:` in the
reply. Concurrency 32 on one RTX PRO 6000 Blackwell, each model's rows in one sitting. Auto-scored,
no LLM judge. Both halves are means of at least three runs per build.


| model | shape | size | overall | knowledge | call | abstain | finished | out | tok/s @ 32 |
|---|---|---|---|---|---|---|---|---|---|
| [Laguna-XS-2.1](https://huggingface.co/primitive-ai/Laguna-XS-2.1-mixed-NVFP4-MXFP8) | 31 B MoE | 19.3 GiB | 81.7 | 83.8 | 68.4 | 73.5 | 98.9% | 1097 tok | 1523 |
| [Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8) | 30 B MoE+Mamba | 19.2 GiB | 87.1 | 87.9 | 85.4 | 70.5 | 97.9% | 1429 tok | 2204 |
| [Ornith-1.5-35B-A3B](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8) | 35 B MoE | 22.6 GiB | 88.7 | 91.7 | 74.4 | 60.0 | 99.3% | 760 tok | 1469 |
| [Muse-Glimmer-30B](https://huggingface.co/primitive-ai/Muse-Glimmer-30B-mixed-NVFP4) | 30 B MoE | 20.4 GiB | 86.6 | 88.8 | 78.6 | 54.5 | 99.7% | 800 tok | 1176 |
| [Qwen3.8-27B](https://huggingface.co/primitive-ai/Qwen3.8-27B-mixed-NVFP4-FP8) | 27 B dense | 20.7 GiB | 88.8 | 90.4 | 85.5 | 54.5 | 99.7% | 651 tok | 908 |
| [Laguna-S-2.1](https://huggingface.co/primitive-ai/Laguna-S-2.1-mixed-NVFP4-MXFP8) | 110 B MoE | 64.0 GiB | 84.3 | 87.1 | 64.6 | 81.0 | 97.3% | 995 tok | 670 |
| [Qwen3.8-Flash-Next](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8) (mixed sibling) | 180 B MoE (6 B active) | 183.7 GB | 90.3 | 92.2 | 84.8 | 56.7 | 99.5% | 686 tok | 816 |

The Flash-Next row is the mixed sibling's sitting; this repo's own numbers under the same protocol
are in Measured above (90.2 / 92.2 / 84.6 / 56.7), a tie with the mixed build.

**Read `overall` with `finished`.** `overall` scores an answer that overran the token budget as wrong,
but it cannot say whether the model needed the room or failed to stop; `finished` and `out` separate
those. A gap under **1.0** is a tie. The `tok/s` column comes from each model's own sitting and drifts
a few percent between sittings, so read it as a bracket.

**`call` and `abstain` are the tool-calling suite's two halves, reported separately.** `call` is
accuracy on the 160 items that require a tool call; `abstain` is the 40 whose correct action is to
call nothing. They used to be pooled into one `agentic` number, and the pooling misled: a model with
ordinary call accuracy and unusual abstention discipline outscored models that are better at
actually making calls. Weight them by your own workload's mix.

<details>
<summary>Per-benchmark detail, both halves</summary>

| benchmark | Laguna-XS-2.1 | Nemotron-3.5-Lightning-30B-A3B | Ornith-1.5-35B-A3B | Muse-Glimmer-30B | Qwen3.8-27B | Laguna-S-2.1 | **Qwen3.8-Flash-Next** |
|---|---|---|---|---|---|---|---|
| **knowledge** | | | | | | | |
| mmlu_pro | 79.0 | 82.0 | 89.5 | 89.0 | 89.5 | 81.0 | 89.0 |
| math_500 | 76.0 | 80.0 | 88.0 | 74.0 | 82.0 | 83.0 | 86.0 |
| gsm8k | 97.0 | 96.0 | 99.0 | 98.0 | 98.0 | 99.0 | 97.0 |
| arc_challenge | 94.0 | 97.3 | 98.7 | 98.0 | 98.0 | 96.0 | 98.7 |
| hellaswag | 70.7 | 78.0 | 86.7 | 83.3 | 82.0 | 76.7 | 88.7 |
| winogrande | 84.7 | 89.3 | 89.3 | 90.7 | 89.3 | 86.0 | 95.3 |
| commonsenseqa | 80.8 | 83.3 | 86.7 | 82.5 | 84.2 | 85.8 | 86.7 |
| boolq | 90.0 | 92.0 | 92.0 | 90.0 | 91.0 | 92.0 | 94.0 |
| openbookqa | 94.0 | 99.0 | 98.0 | 97.0 | 97.0 | 94.0 | 96.0 |
| **tool calling** | | | | | | | |
| bfcl_v4 | 83.3 | 96.7 | 76.7 | 86.7 | 96.7 | 90.0 | 96.7 |
| xlam | 64.4 | 73.3 | 55.6 | 66.7 | 77.8 | 48.9 | 82.2 |
| toolace | 37.8 | 71.1 | 71.1 | 64.4 | 68.9 | 48.9 | 73.3 |
| glaive | 87.5 | 97.5 | 92.5 | 100.0 | 100.0 | 85.0 | 100.0 |
| when2call | 75.0 | 77.5 | 52.5 | 55.0 | 55.0 | 75.0 | 55.0 |

</details>

---

<p align="center">
  <br>
  <img src="assets/primitive-logo.png" alt="Primitive" width="34"><br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>

