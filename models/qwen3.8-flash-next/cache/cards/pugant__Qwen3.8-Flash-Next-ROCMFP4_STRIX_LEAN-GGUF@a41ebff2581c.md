---
library_name: llama.cpp
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - gguf
  - rocmfp4
  - rocmfpx
  - rocm
  - strix-halo
  - gfx1151
  - quantized
  - 2-bit
pipeline_tag: text-generation
language:
  - en
  - multilingual
version: '1.3'
date: 2026-09-04
quantized_by: pugant
---
> **Update — 2026-09-03: the ROCmFP2 (2-bit) build was re-quantized (v2).** The PLE n-gram
> table now ships at Q5_1 served disk-resident (`--ple-disk`), and the dense trunk was
> upgraded to an FP4/Q3 mix. Measured on our [bare-metal Strix Halo](https://github.com/pugant/strix-nebulosa/blob/main/BARE-METAL.md):
> Dante 8k ppl **37.90 -> 2.07**, EN ppl **7.29 -> 3.27** (within ~2% of the ROCmFP4 LEAN),
> tg128 **-5.5%**, resident memory at ctx 131k **55.94 GiB**. The two `ROCmFP2-STRIX_LEAN/`
> shards were **replaced in place**: if you run the 2-bit build, **re-download both shards**
> (same filenames, new bytes). The file is larger than before (75.87 GiB vs 53.12) — with the
> n-gram offload the memory boundary is the measured 55.94 GiB, not the file size. Full
> numbers and protocol: the lab note
> [2026-09-03-fp2mix-v2-requant.md](https://github.com/pugant/strix-nebulosa/blob/main/docs/experiments/2026-09-03-fp2mix-v2-requant.md).


# Qwen3.8-Flash-Next for AMD Strix Halo — two builds (GGUF)

Two quantizations of **Qwen/Qwen3.8-Flash-Next** (arch `qwen4exp`, 180B total: 125B base / 6B active + 51B PLE n-gram table + 4B MTP) for AMD Strix Halo (Radeon 8060S, gfx1151) unified memory — one per memory variant, in one repo:

- **ROCmFP4 STRIX_LEAN** — the recommended build for the **128 GB** variant: full quality, 98.491 GiB, 4.78 bpw.
- **ROCmFP2 STRIX_LEAN (2-bit mix)** — an FP2/FP4/Q3 mix with a disk-resident Q5_1 PLE for the **64 GB** variant: 75.87 GiB on disk (3.68 bpw effective), ~56 GiB resident — fits the full production stack.

Both are quantized from the native BF16 export with the full unsloth importance matrix.


> **Note on the GGUF headers**: the `general.file_type` metadata field declares the closest *standard* types (`Q4_0` / `Q2_K`) so that the Hub file browser lists both builds side by side; the actual format is defined by the per-tensor types (`Q4_0_ROCMFP4` / `Q2_0_ROCMFPX` from the ROCmFPX fork), which are the authoritative ones.

## TL;DR — which build for which machine

| Machine | Build | Files | Quality |
|---|---|---|---|
| **128 GB** Strix Halo | **ROCmFP4 STRIX_LEAN** (recommended) | 3 shards, 44.9 + 44.7 + 16.1 GB — 98.491 GiB, 4.78 bpw | Full |
| **64 GB** Strix Halo | **ROCmFP2 STRIX_LEAN (2-bit mix)** | 2 shards, 41.91 + 39.56 GB — 75.87 GiB on disk, 3.68 bpw | Quality: [measured numbers](#quality--measured-numbers) |

- **128 GB**: use the [STRIX_LEAN build](#the-recommended-build-rocmfp4-strix_lean-128-gb) below — full quality, agent-ready on a runtime that includes patch `0012`.
- **64 GB**: the [2-bit mix build](#the-64-gb-build-rocmfp2-strix_lean-2-bit-mix-v2) fits the whole production stack (external MTP drafter, vision, KV q5_1, ctx 131k) inside a 60 GiB memory boundary — **55.94 GiB measured host peak** — with the PLE n-gram table read from disk on demand. Its perplexity is 1.76× the LEAN's on the memorized Italian holdout and within ~2% of the LEAN's on the English calibration corpus; tg128 is −5.5% vs the previous FP2MIX build. The trade for fitting 64 GB is now the disk footprint (75.87 GiB) and the holdout gap, not English-corpus quality.

---

## The recommended build: ROCmFP4 STRIX_LEAN (128 GB)

> **New build (2026-09-02) — re-download recommended.** Requantized from the native BF16 export with the full unsloth importance matrix. Measured perplexity: **−0.6%** (Italian holdout) / **−4.9%** (English calibration) vs the previous build. Same tensor recipe and file sizes (±800 B of metadata).

ROCmFP4 quantization of **Qwen/Qwen3.8-Flash-Next**, tuned for AMD Strix Halo (Radeon 8060S, gfx1151) unified memory — **98.491 GiB, 4.78 bpw, 3 shards in the `ROCmFP4-STRIX_LEAN/` folder**. This is the build this repo is named after. On a 64 GB machine, see [the 64 GB build](#the-64-gb-build-rocmfp2-strix_lean-2-bit-mix-v2) below.

> **Memory note — the 98.5 GiB file size is not the memory bill.** With `--ple-disk`, the
> 35.76 GiB Q5_1 PLE n-gram table is read from disk block-by-block on demand and never
> loaded: the measured resident envelope is **~64 GiB**, leaving **~36 GB free on a 128 GB
> machine** for context, vision, the speculative drafter and the OS. The same offload the
> 2-bit mix build relies on (see its [boundary numbers](#the-60-gib-boundary-v2-measured-2026-09-03)).

### Run it with our engine

These tensor types need the [ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) family — stock llama.cpp cannot load them.
**Recommended runtime: our lab build** ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) —
full source, Dockerfile and replication guides. On top of the fork it adds:

- the qwen4exp architecture port + external MTP drafter (`-md`), incl. the rollback
  fixes the drafter requires (qwen4exp series, patch `0012` or later);
- `--ple-disk` — the PLE table stays on disk: this quant fits 128 GB with ~36 GB of RAM to spare;
- `--cache-disk-persist` — the prompt cache survives server restarts (a 107k-token
  context: 1.57 s restore, 14.3 s end-to-end vs the measured 920 s cold re-prefill —
  64×; token-exact boundary with the drafter);
- vision (mmproj) together with the drafter; the reasoning-budget warn window;
- graph-reuse + dense-decode patches (lab series optim-camp, 2026-09-02): plain tg512
  +17.5% HIP — measured on this build.

Guides: [PLE disk-offload](https://github.com/pugant/strix-nebulosa/blob/main/docs/guide/qwen38-flash-next-ple-disk.md) · [persistent prompt cache](https://github.com/pugant/strix-nebulosa/blob/main/docs/guide/qwen38-flash-next-prompt-cache-disk.md).

### Recipe (ftype 106, `Q4_0_ROCMFP4_STRIX_LEAN`)

| Tensor group | Quant |
|---|---|
| MoE experts `ffn_*_exps` (48×512) | ROCmFP4 (type 101) |
| Shared experts `ffn_*_shexp` | ROCmFP4 |
| Attention | ROCmFP4 / ROCmFP4-FAST mix |
| **PLE n-gram table** `per_layer_token_embd` (51.2B) | **Q5_1** (`--tensor-type` override) |
| `token_embd` | Q5_K |
| `output` (lm head) | Q6_K |
| Norms / hash constants | F32 / U64 |

- **Provenance (2026-09-02 build): quantized directly from the native BF16 export** ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), 354 GB, 8 shards) with the **full unsloth importance matrix** (580 MB, 926 entries, exact tensor-name matching — covers 878/915 = 96.0% of the quantizable 2D+3D tensors; the 5 never-calibrated embedding/head tensors and one `ple_conv1d` take the unweighted path, as in any imatrix build). Command: `llama-quantize --imatrix --output-tensor-type Q6_K` at preset 106.
- The previous build (served until the 2026-09-02 swap) reached the same recipe the hard way: FP8 official checkpoint → converter (lossy FP8→F32 dequant) → Q8_0 intermediate → LEAN (`--allow-requantize`), with a partial probe-imatrix computed on a Q4_K intermediate. Going back to the native BF16 weights with a complete importance matrix removes both handicaps — see the measured perplexity in [Validation](#validation).
- Format recipe deliberately unchanged: per-tensor type multiset **identical to the previous build (1224/1224 tensors, zero diff on type + shape + bytes)**, so kernels and speed are unchanged by construction.
- MTP draft head not included (converter drops it, as vLLM does); pair it with an external MTP drafter via `-md` — measured below with `agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF` (the community Q8_0 MTP head).
- **Size: 98.491 GiB (105,753,531,776 bytes), 4.78 BPW** — fits the same ~64 GiB resident envelope as the reference build (Δ 800 B of header metadata vs the previous file).

### Serving (Strix Halo)

```bash
llama-server -m ROCmFP4-STRIX_LEAN/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-00001-of-00003.gguf \
  --n-gpu-layers 999 --flash-attn on --ctx-size 8192 --threads 16 --jinja
```

- **Keep mmap ON** (default): the PLE table is a sparse lookup — `--no-mmap` OOMs via cgroup on unified memory with no server-log evidence.
- Point `-m` at the **first** shard.

### Validation

Requant quality, measured 2026-09-02 on the new build (`llama-perplexity`, full corpora, ctx 8192, `-fa on`, `--no-mmap`, 999/999 layers, no drafter, Vulkan build of the fork):

| corpus | previous build | **this build** | Δ |
|---|---|---|---|
| Italian holdout (Dante) | 1.1843 | **1.1776** | **−0.6%** |
| English calibration (agentic / tool-call prose) | 3.3601 | **3.1961** | **−4.9%** |

The Italian holdout is *memorized* by the base model (PPL near 1 is impossible on unseen literary text), so that number is not a quality benchmark — it is a pipeline check: broken weights (e.g. mishandled scales) would blow it up by orders of magnitude (the random-init toy of the same architecture measures ~250k). The English calibration corpus carries the usable signal: the direct BF16 → LEAN path with the full importance matrix buys **−4.9%** perplexity at identical format and size.

- Previous-build pipeline checks, kept for the record: perplexity of the Q4_K intermediate on the Italian holdout, upstream PR #27742 runtime, CPU: **PPL = 1.156 ± 0.006** (64 chunks); cross-runtime agreement through the ROCmFPX fork runtime read **PPL = 1.095** (same file, two independent implementations).

### Benchmarks

On-device, Radeon 8060S (gfx1151) via ROCm 7.2.4 on the ROCmFPX fork, 999/999 layers, flash-attn on, median of 3 (`llama-bench -t 16 -fa 1 -r 3`):

| test | tok/s | flags |
|---|---|---|
| tg128 | **22.03 ± 0.23** | `-b 2048` |
| pp2048 | 131.12 ± 21.74 | `-b 2048 -ub 512` |
| tg128 | 20.41 ± 0.17 | default batch |
| pp512 | 85.99 ± 4.42 | default batch |

- Reference build by kingjones777 (same preset, **no imatrix**): tg 22.6, pp 345 — our tg is on par while carrying an importance-matrix calibration; pp has headroom left in batch/kernel tuning (first-cut port).
- Keep mmap ON: the PLE table is a sparse lookup, resident stays ~64 GiB at 8k ctx like the reference.

### Speculative decoding — external MTP drafter (measured on-device)

Pair this target with the external MTP head [`agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF) (Q8_0, 3.85 GiB — Q8 or better; lighter quants collapse draft acceptance). Requires our lab build (`main`, qwen4exp series, patch `0012` or later) — see [Run it with our engine](#run-it-with-our-engine) above. On older builds the drafter works, but every rollback round degrades the output.

```bash
llama-server -m <this-LEAN-shard-1> -md Qwen3.8-Flash-Next-MTP-Q8_0.gguf -ngl 999 --spec-draft-ngl all -fa on --jinja -c 8192 -fit off --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
```

Dedicated GPU, ctx 8192, temp 0, median of 3 (`/v1/chat/completions`, 600 max tokens), same container image for both arms:

| workload | plain | + MTP n=3 | + MTP n=5 | Δ (n=3 / n=5) |
|---|---|---|---|---|
| deterministic (counting 1-200) | 22.1 | 46.0 | **50.2** | **+108% / +127%** |
| deterministic (alphabet) | 20.9 | 32.0 | 32.6 | +53% / +56% |
| open prose (Italian) | 22.6 | 22.8-25.4 | — | +1% to +12% |

- Draft acceptance: 95.7% of tokens, mean accepted length 3.24 at n-max 3 (per-position 1.00 / 0.72 / 0.53); on deterministic text it stays high out to position 5 (5.20/6) and throughput still climbs with n.
- Backend matrix: the port runs unchanged on HIP (ROCm 7.2.4) and Vulkan (RADV) — Vulkan plain is +4-11% vs HIP; with MTP the two backends converge (Vulkan MTP boost +9% to +95%).
- Where the money is: the round bottleneck is the **batched verify on the hybrid trunk** (KV + GDN scan + QSA index + PLE lookup), not the drafter (~16 ms/round, ~5% of wall). Deterministic work converts acceptance into speed almost 1:1; open prose decays after position 1 and gains marginally.
- Boot with the draft takes ~4-8 min (graph reserve + checkpoint buffers). Perplexity sanity on our port: identical to the pre-MTP build (PPL 1.0952 on the memorized holdout — trunk untouched).
- Vision (`--mmproj`) works **together with** the MTP drafter (2026-08-28 fix, same patch lineage): image chunks are not replayed on the MTP draft context — the drafter still sees the image through the trunk hidden state, and the target's verify keeps the output exact. Acceptance on vision requests measured 98.3%.
- RS ring-salvage rollback enabled for qwen4exp (2026-08-28, same patch lineage): partial verify rejections roll back per-row instead of a full ~115-148 MiB checkpoint restore, so the drafter never stalls at long agent contexts (`speculative replay stalled` gone). An earlier note here described a "one-round acceptance dip after each rollback" as the ring's trade-off: that dip was actually the symptom of the missing conv/PLE ring-slot writer, fixed by patch `0012` (2026-08-30) — with the fix the dip is gone, draft acceptance rises from ~0.74 to **0.91-0.95**, and decode gains ~+70% at n=6 (41-44 tok/s on code workloads, Vulkan). The ring itself costs ~7.2 GiB of buffer.

### Quality on agent workloads — cleared (root-caused 2026-08-30, fixed)

Earlier agent sessions with this quant showed output-integrity incidents (reasoning
breaking mid-sentence, digit loss inside hex/number runs, malformed tool calls, premature
stops). **Root cause: a runtime bug in our fork, not the quant.** The qwen4exp
conv-history writer restored the GDN conv state and the PLE history from never-written
ring slots after every speculative rollback — plain decoding never touched that path,
which is why non-speculative builds of the same quant were clean. Fixed by patch `0012`
(branch `qwen4exp-conv-ring-slots`): after the fix, identical agent-style probes produce
zero anomalies, draft acceptance is 0.91-0.95 and decode reaches 41-44 tok/s at n=6 on
Vulkan (see the spec-decoding section). **This quant is recommended for agent deployments
too — on a runtime that includes patch 0012.**

### Backend: Vulkan (RADV) beats ROCm/HIP by ~20% (measured, 2026-08-29)

Same machine (Radeon 8060S, Strix Halo), same fork build (commit `2c6309e3b`), same protocol (4 prompts × 3 reps, greedy, external MTP drafter Q8_0, `--parallel 1`, `--no-mmap`, ctx 8192, p_min 0.75, n_max 3):

| tok/s (median) | ROCm/HIP | Vulkan/RADV | delta |
|---|---|---|---|
| prose (Roma) | 26.0 | **31.8** | +22% |
| prose (Rinascimento) | 25.9 | **31.6** | +22% |
| counting 1-200 | 44.0 | **47.0** | +7% |
| alphabet | 30.4 | **36.7** | +21% |

- Speculative acceptance is identical on both backends (mean accepted length 2.98 vs 2.94): the gap is pure backend speed, not drafter behavior.
- Rep-to-rep variance collapses on Vulkan (about ±10% down to ±0.6% with `--no-mmap`): with mmap the weights page in lazily and occasionally stall generation; `--no-mmap` pins them up front. On ROCm `--no-mmap` measured neutral.
- **Recommendation: serve this model on Vulkan/RADV on gfx1151.** For context, on the dense Qwen3.8-27B the same test harness measured the two backends at parity — the Vulkan advantage shows on this PLE-hybrid MoE architecture.
- n_max: 6 is the measured optimum. 8 regresses (~35 vs ~42 tok/s on code workloads, re-measured after patch 0012 — the marginal draft positions 7-8 do not pay for themselves); n_max 8 is clean output-wise on fixed builds but slower.

### Quant comparison: STRIX_LEAN vs Unsloth UD-IQ4_XS (measured, 2026-08-29)

Same fork build (commit `2c6309e3b`, Vulkan/RADV), same protocol as the backend test above (4 prompts × 3 reps, greedy, external MTP drafter Q8_0, n_max 3, p_min 0.75, ctx 8192, `--no-mmap`, `-fit off`), same machine. Arms differ only in the target GGUF: this STRIX_LEAN vs `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ4_XS (87.2 GiB sharded / 93.7 GB merged):

| tok/s (median) | STRIX_LEAN | UD-IQ4_XS | Δ (UD vs LEAN) |
|---|---|---|---|
| prose (Roma) | **31.5** | 27.8 | -11.7% |
| prose (Rinascimento) | **31.0** | 26.5 | -14.5% |
| counting 1-200 | **45.5** | 44.0 | -3.3% |
| alphabet | **36.0** | 34.7 | -3.6% |

- Speculative acceptance is identical across quants (mean accepted length 3.0-3.25): the gap is pure decode speed of the quantized kernels, not drafter behavior.
- Engine note (updated 2026-08-30): a clean rebuild of master (`c841aeeb8`) loads the unsloth GGUF fine — sharded or merged, with or without `-fa`/`--no-mmap`; the earlier failure was a stale local build, not upstream and not the quant. What does NOT work on official today is the speculative path: with `-md <MTP head> --spec-type draft-mtp` the target itself fails to load (`check_tensor_dims: blk.0.hc_attn_norm.weight not found` in ms). So: plain serving of Flash-Next quants works on official, the external-drafter setup requires our fork.
- Verdict: STRIX_LEAN keeps the throughput edge on every workload (prose included — the one that matters for agents) and remains the only path with integrated vision (mmproj), context checkpoints and reasoning-budget on our fork.

---

## The 64 GB build: ROCmFP2-STRIX_LEAN (2-bit mix, v2)

**Qwen3.8-Flash-Next for Strix Halo with 64 GB.** A deterministic FP2/FP4/Q3 mix of **Qwen/Qwen3.8-Flash-Next** — **75.87 GiB on disk, 3.68 bpw effective, ~56 GiB resident** — small enough in *memory* to run the full production stack (external MTP drafter, vision, disk-resident PLE, KV q5_1, on-disk prompt cache) inside the **64 GB** unified-memory variant of AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151). The disk file is larger than the previous build (75.87 GiB vs 53.12): the PLE n-gram table now ships at **Q5_1 on disk** and is read block-by-block on demand via `--ple-disk` — it is never loaded, so the memory boundary does not move with the file size. The quality numbers for this build are measured and documented below. If you have 128 GB, use [the STRIX_LEAN build above](#the-recommended-build-rocmfp4-strix_lean-128-gb).

### Read this first

- **Fork-specific tensor types** (`Q2_0_ROCMFPX` type 107, `Q4_0_ROCMFP4`, `Q3_0_ROCMFPX` type 104): stock llama.cpp cannot load this file. See [Runtime](#runtime).
- **The PLE table is Q5_1** (as in the LEAN): `--ple-disk` serves it through the original Q5_1 PLE-store path. The fork's ple-store v2 (FP2 PLE support) is in `main` but no longer required for this file — it was required by the previous build, whose PLE was FP2.
- **Quality scope.** Perplexity and boundary numbers are re-measured on the v2 (below); the task-level (multi-round agentic, vision) results quoted in the history section were measured on the v1 and have **not** been re-measured on the v2.

### Quality — measured numbers

Perplexity, full corpora, `llama-perplexity`, ctx 8192, `-fa on`, `--no-mmap`, 999/999 layers, no drafter, Vulkan build (protocol identical across all rows):

| GGUF | Italian holdout (Dante) | EN calibration (agentic prose) |
|---|---|---|
| STRIX_LEAN (4.78 bpw, reference — 2026-09-02 build) | 1.1776 | 3.1961 |
| **FP2MIX v2 (this, 3.68 bpw)** | **2.0676 ± 0.011 (1.76×)** | **3.2689 ± 0.015 (1.02×)** |

What the v2 changed relative to the v1 it replaced (v1 numbers kept in the [history section](#v1-numbers-kept-for-the-record)):

1. **The PLE recall is back.** The v1's 37.90 / 7.29 was not quantization noise — it was the FP2 PLE table losing the fine-grained n-gram recall (the holdout is memorized text; the LEAN reads 1.18 there by recall). Restoring the table to Q5_1 brings the holdout to 2.07 (**×18.3 better than the v1**) and the English corpus to 3.27 — within ~2% of the LEAN, i.e. at parity given the corpus.
2. **The residual gap is attributable to trunk damage.** With the PLE fixed, the remaining 1.76× on the memorized holdout is the structural 2-bit error of the FP2 trunk (the `ssm_*` band and the FP2 experts), consistent with the codebook-floor analysis of the v1.
3. **Task-level behavior: not re-measured.** The v1's failure modes (round-2 hallucinated data in a 3-round tool smoke; vision reasoning degenerating into a loop) were PLE-recall symptoms, and the recall is restored — but the v2 has not been re-run through those smokes. Perplexity parity with the LEAN on the English corpus is the measured claim; task parity is the plausible, unmeasured one.

### Throughput

Same-harness `llama-bench` tg128 (Vulkan, `-p 0 -n 128 -fa on -mmp 0 -ngl 999 -t 8 --ple-disk --ple-cache-mib 4096`, both arms on the same image, measured 2026-09-03):

| tg128 | tok/s |
|---|---|
| FP2MIX v1 (superseded) | 34.05 ± 0.38 |
| **FP2MIX v2 (this)** | **32.18 ± 0.46** (**0.945×** — the FP4/Q3 trunk upgrade costs −5.5%) |

The v1's cell-by-cell comparison vs the LEAN (+3.6% to +18.5% across prose/deterministic/pp) is kept in the [history section](#v1-numbers-kept-for-the-record); the v2 was not re-measured cell-by-cell against the LEAN. <a id="v1-tg"></a>

### The 60 GiB boundary (v2, measured 2026-09-03)

Full production stack — external MTP drafter at n6 (p_min 0.75), vision projector (mmproj, `--image-min-tokens 1024`), KV q5_1, ctx **131072**, `--ple-disk --ple-cache-mib 4096 --no-mmap`, `-fit off`, docker `--memory=60g`:

| measure | value |
|---|---|
| host-memory peak (Δfree, 6-min multi-turn load) | **55.94 GiB** |
| OOM | none |
| PLE disk-resident marker | `PLE table disk-resident: 35.76 GiB externalized (Q5_1, 320001536 rows), cache 4.00 GiB` |
| boot → healthy | 42 s |

The disk-resident Q5_1 PLE behaves like the LEAN's: blocks are read on demand from the GGUF, the 4 GiB `--ple-cache-mib` window is the only resident part. Measurement caveat that generalizes from the v1 work: the container's own RSS does not measure this backend's weights (VK/GTT-resident); host-level Δ is the only metric that gates the boundary.

### Files and composition

The model is uploaded as **2 shards** in the `ROCmFP2-STRIX_LEAN/` folder — `ROCmFP2-STRIX_LEAN/Qwen3.8-Flash-Next-Q2_0_ROCMFP2_STRIX_LEAN-00001-of-00002.gguf` (**41.91 GB**) and `ROCmFP2-STRIX_LEAN/Qwen3.8-Flash-Next-Q2_0_ROCMFP2_STRIX_LEAN-00002-of-00002.gguf` (**39.56 GB**) — point `-m` at the **first** shard. Total: **81,469,812,288 bytes = 75.87 GiB, 3.68 bpw effective**.

Measured composition (per-tensor reader over the final file, 1224 tensors):

| Type | Tensors | Bytes | Notes |
|---|---|---|---|
| `Q5_1` (6.0 bpw) | 1 | 35.763 GiB | **the PLE n-gram table** (51.2 B params) — disk-resident via `--ple-disk` |
| `Q2_0_ROCMFPX` (2.5 bpw) | 235 | 31.173 GiB | the `ssm_*` band + the FP2 expert blocks |
| `Q3_0_ROCMFPX` (3.5 bpw) | 13 | 4.443 GiB | `ffn_*_exps` of blocks {40,41,42,44,46} — the surgery (+1.89 GiB) |
| `Q4_0_ROCMFP4` (4.5 bpw) | 582 | 3.345 GiB | all dense projections: attention, hyper-connections, shared experts, indexer, `ple_key/value` |
| `Q6_K` / `Q5_K` | 1 each | 0.486 / 0.407 GiB | `output`; `token_embd` |
| F32 | 388 | 0.244 GiB | routers, conv1d, norms — as in the LEAN |
| `Q4_0_ROCMFP4_FAST` / F16 | 2 / 1 | 3.5 MB / 82 KB | output hc heads; `ple_conv1d` |

- 372 of the 1224 tensors changed vs the v1: 1 PLE → Q5_1, 358 dense → FP4, 13 expert blocks → Q3. Everything else is bit-identical in mapping.
- The 48 `ffn_gate_inp` routers and 36 `ssm_conv1d` convolutions are **structurally unquantizable** (quantizer allowlist) and stay F32 — a tool property, not a mapping choice.

### Runtime

Like the LEAN build, these tensor types need the [ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) family — stock llama.cpp cannot load this file.
**Recommended runtime: our lab build** ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) —
full source, Dockerfile and replication guides. On top of the fork it adds the qwen4exp
architecture port, the external MTP drafter with its rollback fixes, `--ple-disk` /
`--cache-disk`, and the ple-store v2 generalization (FP2 PLE tables — required by the
v1, not by this build whose PLE is Q5_1).

Graph-reuse + dense-decode patches (lab series optim-camp, 2026-09-02): plain tg512
+18.6% HIP / +32.5% Vulkan (measured).

```bash
llama-server -m ROCmFP2-STRIX_LEAN/Qwen3.8-Flash-Next-Q2_0_ROCMFP2_STRIX_LEAN-00001-of-00002.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q8_0.gguf --spec-type draft-mtp --spec-draft-ngl all \
  --spec-draft-n-max 6 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 \
  -ngl 999 -fa on --jinja -c 131072 -b 2048 --ubatch-size 512 -fit off \
  --ple-disk --ple-cache-mib 4096 --no-mmap \
  --cache-type-k q5_1 --cache-type-v q5_1 --threads 16
```

This is the configuration the 55.94 GiB boundary above was measured with — the measured run also set `--spec-draft-type-k/v q8_0`, `--parallel 1`, `--memory=60g --memory-swap=60g` and the vision projector (`--mmproj … --image-min-tokens 1024`).

- `--ple-disk --no-mmap` are the measured boundary configuration (mmap on Vulkan collapses pp ~3×; `--no-mmap` pins the weights up front). `--ple-cache-mib 4096` is the measured PLE cache window.
- **KV cache q5_1** (`--cache-type-k q5_1 --cache-type-v q5_1`) — the configuration the boundary numbers were measured with.
- **MTP drafter**: pair with the external head [`agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF) (Q8_0; Q8 or better — lighter draft quants collapse acceptance).
- Vision (`--mmproj`) works together with the drafter on our build; the 55.94 GiB peak includes the projector.
- AMD RDNA 3.5 (gfx1151 / Strix Halo) target, Vulkan/RADV backend. Tested on Radeon 8060S, not elsewhere.

### How it was built (v2)

- **Source: the native BF16 export** by [unsloth](https://huggingface.co/unsloth) (354 GB, 8 shards) with the **full unsloth importance matrix** (580 MB, 926 entries).
- **Deterministic mapping, not a preset**: a generator reads the v1's as-built per-tensor table and emits **1224 explicit first-match-wins rules** (anchored, escaped) for the fork's `--tensor-type-file`: PLE → Q5_1 (rule 1), every non-`ssm_*` dense tensor → `Q4_0_ROCMFP4`, the `ffn_*_exps` of blocks {40,41,42,44,46} → `Q3_0_ROCMFPX`; all 1224 tensors are governed, none falls back to the preset.
- Quantized with `llama-quantize --tensor-type-file <rules> --imatrix --output-tensor-type Q6_K` (flags before the positionals — the fork's parser is index-based), 32 threads, **18.1 min**, zero silent fallbacks.

### v1 numbers, kept for the record

The v1 (57.04 GB, PLE at FP2) was replaced in place by the v2 on 2026-09-03. Its measured numbers, kept for the record — the diagnosis story below documents why they looked the way they did:

| measure | v1 value |
|---|---|
| ppl Dante 8k / EN 8k | 37.9027 (32× LEAN) / 7.2928 (2.17×) |
| tg128 same-harness | 34.05 ± 0.38 |
| vs LEAN (4 prompts × 3 reps, n3, same image) | +8.5% / +18.5% prose, +5.8% / +5.9% deterministic; pp2048 +4.0%, tg128-bench +3.6%, 1,943-token real prefill +7.6% |
| 60 GiB boundary (ctx 32k) | host peak 55 GiB with drafter+vision; 8/8 multi-turn sessions |
| task smoke | 3-round tool calling 1/3; vision: correct perception, no answer (660-token thinking loop) |

(Multiples vs the pre-2026-09-02 LEAN: 1.1843 / 3.3601 — the LEAN in this repo was requantized on 09-02; vs that build the v1 EN multiple is 2.28x.)

- **The error was structural, not a bug.** Per-family relative RMSE against the BF16 source put every FP2 family at 0.317-0.359, bracketing the **theoretical floor of 0.347** that the 2-bit codebook {-4, -1, +1, +4} × block scale admits for a Gaussian source. The quantizer was optimal-for-format; the recall loss lived in the format.
- **Memorized text amplified it.** The Italian holdout is memorized (the LEAN's 1.18 is recall): FP2 noise on the PLE table (cosine 0.949 with the source) broke that recall and the loss compounded non-linearly to 32×; the English corpus carried the "true" 2-bit damage (2.17×).
- **Why publish it anyway** — the diagnosis detour is documented in the lab note ([`2026-09-02-flashnext-fp2-64gb-and-lean-requant.md`](https://github.com/pugant/strix-nebulosa/blob/main/docs/experiments/2026-09-02-flashnext-fp2-64gb-and-lean-requant.md)): a Python probe reported the PLE tables of two same-source builds as nearly orthogonal — "two different tensors" — and the story reproduced across three test rounds before block-level forensics against the fork's `block_q5_1` C struct showed the bug was in the shared decoder (the high-bit array `qh` precedes the nibbles `qs`; classic decoders assume the opposite order). The transferable rule — validate any bit-level decoder against the C struct of record — is why the v1 shipped with its own bad numbers rather than a quiet deletion. The v2 carries the fix: measured, published, and replacing the file in place.

**Benchmark environment:** one bare-metal AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB) — full dated configuration and measurement policy: [BARE-METAL.md](https://github.com/pugant/strix-nebulosa/blob/main/BARE-METAL.md)

## Tooling & credits

- **Primary runtime for both builds**: our lab build ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa)) — qwen4exp port, external MTP drafter with its rollback fixes, `--ple-disk` / `--cache-disk`, and the ple-store v2 extension that made an FP2 PLE servable (v1) and the Q5_1 PLE disk-resident at 35.76 GiB (v2). Upstream fork: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) — the llama.cpp fork that defines the ROCmFP4 / ROCmFPX tensor types both files are built from.
- **[unsloth](https://huggingface.co/unsloth)** for the native BF16 export and the published importance matrix — quantization source and calibration for both builds (the LEAN recipe and both FP2/FP4 mappings).
- **[agentionai](https://huggingface.co/agentionai)** for the community Q8_0 MTP drafter both builds are measured and served with.
- Architecture support ported from upstream [ggml-org/llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (danielhanchen) to the ROCmFPX fork (converter + quantizer side); MTP/NextN draft-head support ported from [PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836) (same author, draft stage).
- Reference quant by kingjones777 (same preset, no imatrix) used as compatibility baseline for the LEAN build.
- Quantization pipeline, mapping and measurements by **pugant**; work assisted by GLM by z.ai.

## License

Qwen Community License 1.0 (inherited from the base model). See LICENSE.

---

> Everything here is experimental and provided as-is, at your own risk.
