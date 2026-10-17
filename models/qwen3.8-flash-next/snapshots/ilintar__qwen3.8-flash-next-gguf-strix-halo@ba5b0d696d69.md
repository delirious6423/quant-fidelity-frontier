---
base_model: Qwen/Qwen3.8-Flash-Next
language:
- en
library_name: gguf
license: apache-2.0
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- rocm
- strix-halo
- gfx1151
---

# Qwen3.8-Flash-Next — IQ4_NL GGUF for Strix Halo

A quantization of Qwen3.8-Flash-Next (177 B parameters) built to run on a single
AMD Ryzen AI Max+ with a Radeon 8060S (gfx1151) and 128 GB of unified memory.

| | |
|---|---|
| Weights | IQ4_NL, 93.16 GiB across 9 shards |
| Parameters | 176.94 B |
| Draft head | separate 2.6 GB Q8_0 file, shared embeddings |
| Trained context | 262 144 |
| Architecture | `qwen4exp` — gated delta-net recurrence, lightning-indexer sparse attention, per-layer embeddings |

## Files

- `Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-*-of-00009.gguf` — the target model. Point
  `llama.cpp` at shard `00001`; it finds the rest.
- `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` — the MTP draft head, exported with
  `--mtp-shared-embd` so it borrows the target's token embeddings, output norm and LM head
  instead of carrying its own copy of a 27.5 GB embedding table.

`PROJFIX` marks a re-export that corrects the projection tensor layout; earlier
local builds without it are not interchangeable.

## What it needs

This model uses architecture support and ROCm kernels that are **not in upstream
llama.cpp**. It needs the `strix-halo` branch of
[pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp):

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pwilkin/strix-halo/main/install-flash-next.sh)
```

That builds the pinned ROCr, HIP and llama.cpp revisions into your home directory and
installs a launcher with the right flags. Nothing is written to `/opt/rocm` and nothing
needs root.

Two flags matter more than the rest:

```
--load-mode none --lazy-mode on-direct
```

The per-layer-embedding table is 27.5 GB. `on-direct` serves its rows with explicit
`pread()`s from a thread pool instead of demand-faulting them through an mmap, and
`--load-mode none` avoids holding a second copy of every weight in page cache during
load. Under unified memory with everything offloaded, the weights land in *managed*
(anonymous) memory, so all 93 GiB already competes with the KV cache and compute
buffers — without these two flags the model does not fit.

## Measured throughput

Radeon 8060S, 128 GB unified, ROCm 10.0 with the retained-PM4 runtimes,
`-b 24576 -ub 24576 -p 16384 -n 128 -r 3`:

| Context depth | Prefill (t/s) | Generation (t/s) |
|---|---:|---:|
| 0 | 1151.77 ± 8.12 | 24.13 ± 0.29 |
| 40 000 | 1059.82 ± 4.46 | 15.43 ± 1.77 |

Prefill keeps 92% of its rate at 40 000 tokens of context, which is the sparse attention
path working as intended. Generation is the weaker half and is still short of what the
same kernels reach out of tree; that gap is open.

A commit-by-commit account of how prefill got from 191 t/s to this, including the
measurements that came out flat and the two bugs the work surfaced, is at
[pwilkin.github.io/strix-halo/journey.html](https://pwilkin.github.io/strix-halo/journey.html).

## Caveats

- Experimental. The ROCm kernels are selected by environment variables that the installed
  launcher sets for you; run `llama-server` directly and you get the generic paths.
- The 24576-token ubatch that these numbers use **cannot run on upstream llama.cpp** on this
  hardware at all: upstream's `mm_ids_helper` sizes its shared memory as `n_tokens × 4`
  bytes, which is 98 KB against 64 KB of LDS on gfx1151.
- Quality was checked by paired perplexity against a reference stack rather than by public
  benchmark scores.
