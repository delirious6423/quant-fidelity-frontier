---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
tags:
  - qwen4_exp
  - moe
  - uint3
  - sub-4-bit
  - quantization
  - modelopt
  - mtp
  - speculative-decoding
  - long-context
  - dgx-spark
  - gb10
---

# Qwen3.8-Flash-Next — uint3 routed experts (group 64, bf16 scales)

A **sub-4-bit routed-expert quantization** of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
built to fit the full 262,144-token context of a 125B-A6B multimodal MoE into a **single
128 GB unified-memory GB10 (DGX Spark)** with room left over — and to stay fast while doing it.

**98.74 tok/s** single-stream at 30 tokens of code context, **93.90 tok/s at 255,297 tokens**,
**67.48 GiB steady VRAM**, full **256K** context, MTP k=3 speculative decoding intact.

> **Read the [Quality](#quality--the-honest-numbers) section before you use this.** The 3-bit cut
> costs a measured, statistically significant **−2.66 pp** pooled accuracy, and the damage is
> **concentrated in code** (−4.88 pp HumanEval). This is not a free win, and this card will not
> pretend it is.

---

## What was actually quantized

Only the **routed MoE experts** were touched. Everything else is carried through unchanged.

| component | precision | note |
|---|---|---|
| **routed MoE experts** (48 layers × 512 experts × 3 projections) | **uint3, group 64, bf16 group scales** | the cut |
| dense tail (attention, shared experts, norms) | `W4A16_NVFP4` | unchanged from the source build |
| MTP drafter (`mtp.*`) | NVFP4, group 16 | deliberately untouched — see below |
| PLE n-gram table (51.2 B params) | bf16 | memory-mapped from disk, not resident |
| `lm_head`, vision tower | unchanged | byte-identical |

Re-quantized **from the original BF16 weights**, not from an existing NVFP4 build — quantizing a
quantized tensor compounds the error. The 96 expert shards were streamed through a 12-shard ring
with consumed shards deleted, so peak disk stayed bounded well under the full BF16 footprint.

```
156,094 tensors across 196 files, 51.50 GiB of parameters
48 expert entries -> uint3 g64 ; 581 other config entries byte-identical
mtp.layers.48.mlp.experts = {'quant_algo': 'NVFP4', 'group_size': 16}
```

### Why bf16 group scales, not fp8

The obvious choice is `float8_e4m3` group scales. It fails here: at group 64 on these experts,
**99.6 % of the scales land in the e4m3 subnormal range**, where the representation error of the
scale itself swamps the 3-bit weight error.

Measured scale representation error, same export, same tensors:

| scale dtype | `scale_relerr_max` |
|---|---:|
| `float8_e4m3` | 0.14 – 0.20 |
| **`bfloat16`** | **0.002591** |

bf16 scales cost ~0.5 GiB over fp8 and buy back two orders of magnitude of scale fidelity. At
3.125 effective bits per weight the scales are no longer a rounding detail.

### Export audit (all 73,728 exported tensors)

| statistic | value |
|---|---:|
| relative Frobenius error — median / p01 / p99 / max | 0.21234 / 0.20966 / 0.22889 / 0.29639 |
| cosine similarity — median / min | 0.97809 / 0.95758 |
| zero scales | 0 |
| max scale relative error | 0.002591 |
| spread across gate / up / down projections | 0.00231 |
| spread across layers (L0 max, L40 min) | 0.00906 |

The error is remarkably **flat** — across projections (0.2 % spread) and across depth (0.9 %
spread). There is no layer-depth signal, which is worth knowing: schemes that spend more bits on
edge layers and fewer in the middle have no support in this model's error profile.

---

## Performance

Measured on **GB10 (Grace Blackwell), 128 GB unified memory, 48 SM**, TP1.
Medians over 8 timed repetitions, first rep discarded (the uint3 kernels JIT-compile on first
launch). Acceptance rate τ read from the server's own speculative-decoding counters, not modelled.

| cell | tok/s | τ | T_iter | accept | TTFT |
|---|---:|---:|---:|---:|---:|
| 30-token code prompt, **greedy** | **98.74** | 3.6725 | 37.111 ms | 0.8908 | 0.148 s |
| 30-token code prompt, **vendor non-thinking preset** | **97.96** | 3.7860 | 38.555 ms | 0.9287 | 0.150 s |
| **255,297-token prompt**, greedy | **93.90** | 3.4952 | 38.022 ms | 0.8317 | 1.711 s warm |

* **Cold prefill at 255,297 tokens: 55.16 s = 4,628 tok/s.** Prefill roughly doubled versus the
  4.5-bit build, because the weight stream shrank.
* **ΔT_iter from 34 → 255,297 tokens is +0.911 ms** — across a 7,500× context increase. Decode
  cost here is O(sparse-attention indexer budget), not O(context), and the 3-bit export does not
  break that property.
* **Speculative acceptance went *up* after the cut**, τ 3.53 → 3.67. A 3-bit weight cut degrading
  the drafter's tracking was the main unquantified risk going in; it did not happen. This is a
  large part of why the build beat its own projection.
* The sampling preset costs **0.8 %**, and it *tightens* the run-to-run spread (96.45–100.13 vs
  87.45–103.80 greedy) because it stabilises τ.

### Memory

| | 4.5-bit NVFP4 experts | **uint3 experts** | Δ |
|---|---:|---:|---:|
| model load | 69.55 GiB | **52.04 GiB** | −17.51 |
| **steady-state VRAM** | 85.30 GiB / 91.58 GB | **67.48 GiB / 72.46 GB** | **−17.82 GiB** |

Steady state means steady: identical at 27 and 43 minutes of uptime. A reading taken 13 minutes
in reports 62.95 GiB and is **4.5 GiB low** — the allocator is still climbing. If you benchmark
this build's memory, wait.

With an 8 GiB KV budget this serves **292,481 KV tokens**, i.e. 1.12× the full 262,144-token
context in a single request.

---

## Quality — the honest numbers

### Accuracy: a real, significant regression, concentrated in code

GSM8K + MATH-500 + HumanEval, n = 200 / 200 / 164 = 564 items, concurrency 8. Both arms ran the
**same items**, so this is a **paired (McNemar)** test — pairing tightens the standard error from
1.52 to 0.92 pp and flips the verdict from "not significant" to significant. An unpaired
comparison of two accuracy rates would have understated this.

| benchmark | reference | this build | ref-only right | this-only right | Δ | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.9750 | 0.9650 | 4 | 2 | −1.00 pp | 0.683 |
| MATH-500 | 0.9250 | 0.9000 | 8 | 3 | −2.50 pp | 0.228 |
| **HumanEval** | 0.9268 | 0.8780 | 9 | 1 | **−4.88 pp** | **0.0269** |
| **POOLED** | **0.9433** | **0.9167** | **21** | **6** | **−2.66 pp** (SE 0.92) | **0.0071** |

One-sided 95 % upper bound on pooled harm: **4.18 pp**.

**The distribution of the damage matters more than the pooled number.** The pre-registered
projection for this cut was −1.1 to −1.6 pp on GSM8K; measured −1.00 pp, inside the range. But
no code number was ever projected, and **HumanEval lost roughly 5× the GSM8K hit and is the only
benchmark significant on its own.** If your workload is code, weight this accordingly.

### Long context: no loss at all

**63/63** on a long-context replay that reproduces its baseline bit-for-bit in construction —
same filler templates, needles, codes, depths, seeds and wrappers. 36/36 needle-in-haystack +
27/27 RULER-subset (multikey, multivalue, 4-hop variable tracking) across ~3.3K → 60K tokens at
10 / 50 / 90 % depths, plus 3/3 controls. **Nothing was lost at depth.**

### Token-level agreement: fails, and that is expected

Against a lossless reference the top-1 disagreement is 0.0639 (tolerance 0.005), decisive
disagreement 0.0152 (tolerance 0.002), mean KL 0.149 nats (tolerance 0.01).

**These tolerances were written for a lossless transform and this is not one.** A 3-bit weight
cut was never going to clear a 0.005 top-1 gate. Reported for completeness, not as a defect.
The informative slice is agreement stratified by reference confidence:

| reference confidence | n | agreement |
|---|---:|---:|
| p < 0.50 | 17,394 | 0.539 |
| 0.50 – 0.90 | 54,424 | 0.800 |
| 0.90 – 0.99 | 36,944 | 0.955 |
| **p ≥ 0.99** | 144,950 | **0.992** |

Where the reference model is confident — where a flip is damage rather than noise — agreement
holds at 99.2 %.

---

## Repository contents

Everything needed to load, serve, reproduce and re-derive this build is in this repo.

```
├── *.safetensors               196 files, 51.5 GiB — the quantized model
├── ple/
│   └── qwen38-flash-next.ngram.bf16.bin   95.4 GiB — the PLE n-gram table, mmap layout
├── vllm_patches/               8 modified vLLM source files — REQUIRED to load this checkpoint
│   └── NOTICE                  Apache-2.0 attribution + statement of changes
├── tools/
│   ├── quantize_experts.py     the exporter that produced this checkpoint
│   ├── repack_ple.py           rebuilds the PLE .bin from base-model weights
│   ├── build_draft_vocab.py    builds the FR-Spec pruned draft vocabulary
│   └── bench_r6.py             the throughput harness the perf table was measured with
├── fr_spec/
│   └── draft_vocab_65536.json  the pruned draft vocabulary (26.39 % of vocab)
└── audit/
    ├── export-audit-summary.json    export audit summary — backs the table above
    └── export-audit-per-tensor.jsonl  per-tensor audit, all 73,728 exported tensors
```

## Requirements — read this before downloading

**This checkpoint does not load on stock vLLM.** Three things are required, and all three ship
here:

1. **uint3 grouped-MoE kernels** with bf16 group scales. The `quant_algo` block in `config.json`
   is a mixed-precision map; a loader without the matching dequant path will either refuse or
   load silently wrong. Supplied in `vllm_patches/`.
2. **PLE offload.** The 51.2 B-parameter n-gram table is not resident — it is memory-mapped from
   an SSD and demand-faulted a row at a time. Without that path you need ~100 GB more memory.
   Table in `ple/` (hosted here), loader in `vllm_patches/ple_layer.image.py` +
   `ple_offload_worker.image.py`. **Read [The PLE table](#the-ple-table) before serving** — where
   you put this file determines whether the model is fast or unusable.
3. **MTP with `num_speculative_tokens: 3`** to reproduce the quoted throughput. Without
   speculative decoding, expect roughly τ× fewer tokens per second.

Hardware floor: **~68 GiB of VRAM or unified memory at steady state**, plus **95.4 GiB of free
space on a local NVMe SSD** for the memory-mapped PLE table. NVMe is a hard requirement, not a
recommendation — see below. Developed on a 128 GB GB10; **not** tested on discrete-GPU or
multi-GPU topologies.

Total download: **~147 GiB** (51.5 GiB checkpoint + 95.4 GiB PLE table).

### Applying the patches

The files in `vllm_patches/` are modified copies of vLLM source files, mounted over their
originals. Against a vLLM image with `qwen4_exp` support:

```bash
V=/usr/local/lib/python3.12/dist-packages/vllm
docker run ... \
  -v $PWD/vllm_patches/humming_utils.image.py:$V/model_executor/layers/quantization/utils/humming_utils.py:ro \
  -v $PWD/vllm_patches/fused_humming_moe.image.py:$V/model_executor/layers/fused_moe/experts/fused_humming_moe.py:ro \
  -v $PWD/vllm_patches/modelopt.image.py:$V/model_executor/layers/quantization/modelopt.py:ro \
  -v $PWD/vllm_patches/ple_layer.image.py:$V/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v $PWD/vllm_patches/ple_offload_worker.image.py:$V/v1/ple_offload/worker.py:ro \
  -v $PWD/vllm_patches/hyperconnection.image.py:$V/models/qwen3_8_flash_next/nvidia/hyperconnection.py:ro \
  -v $PWD/vllm_patches/model.image.py:$V/models/qwen3_8_flash_next/nvidia/model.py:ro \
  -v $PWD/vllm_patches/mtp_frspec_fix.image.py:$V/models/qwen3_8_flash_next/nvidia/mtp.py:ro \
  ...
```

Confirm the mounts actually took — a mount that silently did not take is the single most common
way to get a wrong-but-running server. The log must show
`Using HummingIndexedExperts Humming MoE backend.` and 48 `-> HUMMING uint3 g64` lines **per
process** (96 total at TP1 with `mp`, because the PLE offload worker builds the config too).

### The PLE table

`ple/qwen38-flash-next.ngram.bf16.bin` — **102,400,491,520 bytes (95.4 GiB)**, hosted here.

It is the base model's per-layer-embedding n-gram table in **flat bf16, row-major, laid out for
`mmap`** — Qwen's data bit-for-bit, stored raw rather than as safetensors because the loader maps
it directly. It carries no quantization and nothing of ours; it is repackaged, not modified.

```bash
huggingface-cli download HamboneLabs-AI/Qwen3.8-Flash-Next-uint3-g64 \
  --include "ple/*" --local-dir ./
```

<details>
<summary>Prefer to build it yourself from the base repo?</summary>

The file is a pure function of Qwen's published weights, so you can regenerate it and skip the
95 GiB download from here:

```bash
huggingface-cli download Qwen/Qwen3.8-Flash-Next --local-dir ./base-ple --include "*ngram*" "*ple*"
python tools/repack_ple.py ./base-ple ./ple/qwen38-flash-next.ngram.bf16.bin --dtype bfloat16
```

It should come out byte-identical. Verify against the hosted copy:

```
sha256  f90a8563ba5939e5c0b0f6a57bd3a5f94c39126932f4f10eae98c3585e81434d
```
</details>

#### How it loads — and why it has to be an SSD

**The table is never loaded.** `ple_layer.py` maps it with `np.memmap(path, mode="r")` and the
embedding parameter *is* that mapping. Nothing materializes a 95 GiB tensor at any point:
resident memory is bounded by the page cache, not by the table size. That single fact is what
lets a 125B-A6B model with a 51.2 B-parameter n-gram table fit in 128 GB alongside 67.48 GiB of
weights and KV.

The consequence is that **the storage device is on the token generation path.** Every decoded
token gathers 16 n-gram rows out of the mapping with `torch.index_select`. A row is ~320 bytes,
so the gather is far below torch's `at::parallel_for` grain size and runs on **one thread** —
each row not already in the page cache costs a *serial* fault. On the NVMe this was developed on
that is ~110 µs, and it lands directly on the GPU critical path (measured: 24–50 faults per k=3
iteration on first-time prose, 3.3–5.5 ms mean, p99 8.3–10 ms, regression slope 0.95–1.01 with
no knee — the latency is fully exposed, not absorbed).

Scale that fault cost by your device and the requirement explains itself:

| backing store | ~cost per cold row | verdict |
|---|---|---|
| **local NVMe** | ~110 µs | what this was built and measured on |
| SATA SSD | ~0.5–1 ms | roughly 5–10× the decode stall; expect a large regression |
| spinning disk | ~10 ms | unusable — a single iteration can stall for hundreds of ms |
| NFS / SMB / FUSE / network block | highly variable | **do not** — mmap over a network filesystem turns every fault into an RPC |

Two deliberate choices in the loader follow from the access pattern:

* **`madvise(MADV_RANDOM)`** on the mapping. Rows are a few hundred bytes and the ids are
  scattered, so the kernel's default 128 KiB readahead would fetch ~400× the bytes actually
  needed on every fault.
* **`mode="r"`** — every page stays clean and therefore instantly reclaimable. Under memory
  pressure the kernel drops PLE pages instead of swapping, which is why this coexists with a
  67 GiB resident model on a 128 GB box.

**Prefetch.** `VLLM_PLE_PREFETCH=1` recomputes the row ids in numpy *before* the gather, maps
them to the 4 KiB pages they occupy, and hands the kernel a readahead list with
`posix_fadvise(POSIX_FADV_WILLNEED)`. The call returns immediately, so the reads are in flight
while `index_select` starts faulting and the device services them concurrently instead of one at
a time. Measured on this box (2 GiB scratch file, pages evicted with `FADV_DONTNEED` between
trials, 64 fresh random rows/trial, 40 trials):

| strategy | cold total | warm total |
|---|---:|---:|
| demand-fault only | 2.833 ms | 0.002 ms |
| `pread` pool ×16 | 2.063 ms | 0.810 ms |
| `madvise(WILLNEED)` per page | 0.333 ms | 0.027 ms |
| **`posix_fadvise(WILLNEED)`** | **0.289 ms** | **0.022 ms** |

WILLNEED recovers ~90 % of the cold fault cost for 0.02 ms warm, which is why it is always-on
rather than gated behind a fault-counter. The `pread` thread pool an earlier revision shipped is
**gone**: it recovered only 27 % of the cold cost while adding 0.81 ms to every *hot* iteration —
more than the 0.55–0.65 ms it was meant to save. The prefetch is semantically a no-op;
`index_select` over the mapping remains the only source of truth, so wrong ids or a short read
cost speed and nothing else.

**Warm-up is real.** On a cold page cache the first requests pay the full fault cost and the
model will feel slow. The working set converges fast — n-gram rows follow token frequency, so a
small fraction of the 95 GiB carries most traffic — but budget a warm-up pass before you
benchmark, and leave the box as much free RAM as you can spare for page cache.
`VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800` exists because the offload worker's first-time setup can
take a while on cold storage; raise it rather than lowering it.

Environment:

```bash
VLLM_PLE_CPU_OFFLOAD=1
VLLM_PLE_MMAP_PATH=/path/to/ple/qwen38-flash-next.ngram.bf16.bin
VLLM_PLE_PREFETCH=1                  # opt-in; off by default
VLLM_PLE_PREFETCH_MODE=fadvise       # or `willneed` (madvise on a second private mapping)
VLLM_PLE_PREFETCH_MAX_TOKENS=64      # skip prefetch above this; prefill already parallelises
VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
```

`VLLM_PLE_PREFETCH_THREADS` and `VLLM_PLE_PREFETCH_ADAPTIVE` are accepted and **ignored** with a
warning — they configured the removed pool and its controller. Setting
`VLLM_PLE_PREFETCH_VERIFY=1` re-gathers with the recomputed ids and compares against the layer's
own output; it doubles the gather, so debug only.

> The loader refuses a size mismatch rather than reading garbage: if the file is not exactly
> 102,400,491,520 bytes it raises at startup. A truncated download fails loudly, not silently.

### FR-Spec draft vocabulary

`fr_spec/draft_vocab_65536.json` prunes the drafter's output vocabulary to the 65,536 most
frequent tokens (26.39 % of 248,320). It is lossless by construction — the drafter only proposes,
the target model always verifies against the full vocabulary. Set
`VLLM_MTP_DRAFT_VOCAB=/path/to/draft_vocab_65536.json`; the log will confirm
`MTP FR-Spec draft vocabulary ACTIVE: N=65536 of 248320`.

### Serving

```bash
vllm serve /path/to/this/checkpoint \
  --served-model-name qwen3.8-flash-next-uint3 \
  --dtype bfloat16 \
  --quantization modelopt_mixed \
  --moe-backend marlin \
  --distributed-executor-backend mp \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-memory 8589934592 \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

Notes that are load-bearing, not cosmetic:

* `--moe-backend marlin` **stays** even though the routed experts are uint3 — it serves the dense
  tail and the drafter's NVFP4 bank. The routed experts dispatch through the uint3 path
  independently.
* `--distributed-executor-backend mp` is required at TP1 for the PLE offload worker.
* `--kv-cache-dtype fp8` is **not** supported by this build.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks the PLE offload path (the worker
  imports shared CUDA buffers via `pidfd_getfd`, which expandable segments defeat under default
  container seccomp / `ptrace_scope=1`).
* Discard your first timed run. The uint3 kernels JIT-compile, costing ~2 ms on run 1.

---

## What did *not* work

Recorded because negative results save other people the box time.

**uint2 is dominated at every point on the frontier.** A 1,296-point study on real dequantized
experts from this model: at 3.000 effective bits the best uint2 configuration (group 16,
Hadamard, asymmetric) errs **32.67 %**, while uint3 at 3.125 effective bits errs **20.01 %**.
Hadamard rotation recovers only 3–5 %, not the QuIP#-class halving hoped for. Asymmetric
quantization *hurts* at group 32 and 64 — the zero point costs more bits than it saves. And
there is no depth-stratification signal (layers 0 / 24 / 47 err 19.99 / 20.03 / 20.00 %), so
mixed W3-edges / W2-middle schemes have no basis in this model.

**More speculative tokens do not help.** Measured across 8 repetitions × 4 context depths with
only k changed: marginal T_iter is **+5.04 ± 0.12 ms per k**, and τ(k=4)/τ(k=3) is only 1.108.
Net effect at k=4 is −0.29 tok/s on short context and −2.0 to −2.4 at depth. Extrapolated k=6
lands below k=3. **k=3 is the optimum** for this model on this hardware.

**Narrow-N GEMMs run at 58–63 % of achievable bandwidth on a 48-SM part.** Measured in-container
with CUDA graphs captured and distinct weights per launch to defeat L2: a GEMM needs
N ≳ 2048 to reach streaming bandwidth here. The gated-DeltaNet input projection (4096×2560) hits
242 GB/s — full rate — while the hyper-connection down-projection (336×10240) manages 140.9 GB/s.
The only remedy for the occupancy problem at N=336 (~6 thread blocks across 48 SMs) is split-K,
a kernel change, not a model change.

---

## Known limitations

* **Code regression** of −4.88 pp on HumanEval (p = 0.0269). The identified fix — keeping the
  gate/up projections at uint3-g64 while moving the down projection to uint4-g64, for
  3.458 effective bits — is measured to cut relative Frobenius error from 20.01 % to 16.64 % at a
  cost of roughly 3 tok/s. It is **not** applied in this build.
* Quoted throughput requires MTP speculative decoding; τ is workload-dependent and the numbers
  here are code and long-context prompts specifically.
* The vision tower is carried through unquantized and functional, but **no multimodal benchmark
  was run on this build**. Treat vision quality as unmeasured. Note also that the MTP drafter
  takes text-only draft inputs on image prompts, so τ — and therefore throughput — may fall on
  multimodal requests.
* Single-node, single-GPU, 128 GB unified memory only. Untested elsewhere.
* Benchmarked at concurrency ≤ 8. Behaviour at higher concurrency is uncharacterised.

## Methodology notes

Anyone reproducing these numbers should know:

* **Never quote a single-run tok/s.** T_iter noise is ±1.2 %, but single-cell tok/s noise is
  **±19 %**, essentially all of it τ. Report median T_iter and median τ over ≥8 repetitions.
* **Never subtract milliseconds across sessions.** The same build read 42.93 / 43.27 / 43.53 /
  44.30 / 44.43 / 44.72 ms for the same cell across six sessions, and −0.5 to −0.8 ms of drift
  was measured *inside a single container* with no configuration change. Any lever worth less
  than ~1.5 ms cannot be validated on its own.
* **Always rotate expert sets in an MoE microbenchmark.** L2 is 24 MiB; an unrotated uint2 read
  reports 307 GB/s, which is above the chip's ceiling.
* Always state task × preset × depth together, and tag every number as measured or projected.

## License

Released under the **Qwen Community License 1.0**, inherited from the base model — **not** Apache
2.0. Note in particular that Model-as-a-Service and "AI work assistant" businesses require a
separate license from the Qwen team; internal use is exempt while nothing is exposed to third
parties. Read the license before deploying commercially.

## Acknowledgements

Built on [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen
team. Quantization performed with a NVIDIA ModelOpt-compatible mixed-precision export.
