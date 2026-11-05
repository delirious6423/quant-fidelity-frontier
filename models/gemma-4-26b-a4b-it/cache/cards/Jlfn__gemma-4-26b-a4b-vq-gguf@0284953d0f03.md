---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- multilingual
tags:
- gemma
- gemma-4
- google
- gguf
- vector-quantization
- vq
- mixture-of-experts
- moe
- 262k-context
- ds4
- ds4-gemma4
- vq-tools
---

# Gemma-4 26B A4B VQ (R13+8)

A text-only dense-VQ conversion of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, built with recipe
`gemma-4-26b-a4b-family-r8` and served by the `gemma4_vq` family of the ds4
runtime (CUDA only; there is no Metal path for this family). Standard
llama.cpp, Ollama and Transformers loaders do not read this format: the
release is a directory whose `release.json` names three GGUF parts.

The routed experts carry a 13-bit dense-VQ stage (an E4M3FN codebook of 8,192
words, coded as a 12-bit stream plus a 1-bit plane) and a GVR8 residual plane
(a 256-word E4M3 book with one index byte per 8-weight vector), plus one f16
gain per row. Reconstruction is `bf16(gain * (C1 + C2))`. Attention and the
dense FFN projections stay Q4_K, the dense down matrices Q8_0, and the norms,
layer scalars, router and token table BF16.

The release is 9.95 GiB, 23,805 tensors, release format 2, at 2.65 bits per
weight (1.6498 stage + 1.0 residual). Weight-space reconstruction error over
the 60 routed-expert stacks is 23.39 percent mean relL2 (23.3931 on an
independent `vq-quality` run, 23.3772 from the builder). This is a
weight-space figure, not a logits or quality verdict.

| Part | Role | Tensors | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `gemma4-vq8-trunk.gguf` | 30 layers, tokenizer, chat template | 23,756 | 8,588,461,824 (8.00 GiB) | `0b25b7cb284be1fb59c1bed759e775b589bcae178a9faf18903657a7244d936e` |
| `gemma4-vq8-embedding-BF16.gguf` | BF16 token table, 262,144 x 2,816, read in rows from the host | 1 | 1,476,395,648 (1.38 GiB) | `2043b3362749e781ca47b567d24af70f5282db1c2b93904867f54980b1789636` |
| `gemma4-vq8-mtp-Q4_K.gguf` | optional assistant drafter, Q4_K/BF16 | 48 | 621,995,136 (0.58 GiB) | `f85b05b4b16204aff464c35ffce7b18d054748f7eecaffa5463e9cd3d1d3bb57` |
| **All three parts** | | **23,805** | **10,686,852,608 (9.95 GiB)** | |

`release.json` carries the part roles, sizes, hashes and source revisions;
`files.json` and `SHA256SUMS` cover the rest. Pass the directory to the
runtime, never a single part and never a concatenation.

## Geometry

Read from the checkpoint `config.json`, which the runtime validates against:

| Field | Value |
|---|---|
| Layers | 30: 25 sliding at window 1,024, 5 full at layers 5/11/17/23/29 |
| Sliding layers | 16 query heads, 8 KV heads, head_dim 256, RoPE theta 1e4 |
| Full layers | head_dim 512, 2 KV heads, proportional RoPE (40 of 128 pairs), theta 1e6, `attention_k_eq_v` |
| Hidden / dense FFN | 2,816 / 2,112 |
| Experts | 128 per layer, top-8, FF 704 |
| Vocabulary | 262,144, tied to the head |
| Logit softcap | 30 |
| RMS eps | 1e-6 |
| Native context | 262,144 (`max_position_embeddings`) |

The launcher's family id (`gemma4_vq`, variant `gemma4-a4b-vq8`) and the whole
geometry above are read from the checkpoint directory at open; none of it is
compiled into the runtime.

## What the runtime gives you

**OpenAI-compatible serving.** `/v1/chat/completions`, `/v1/completions`,
`/v1/messages` and `/v1/responses`, streaming included, with live tool calls
(the tool result is carried back into the transcript in the canonical shape)
and the thought channel delivered separately from the answer. The checkpoint
carries no official Jinja template, so the prompt is rendered from the token
grammar. Context:
262,144 tokens on a GB10, 65,536 resident on a 12 GiB card, and 131,072 on the
same card with the expert slot pool.

**Two attention geometries and one KV format per lane choice.** The sliding
and full layers run their own tensor-core prefill kernels; decode uses split-K
attention with a configurable split span (`--gemma4-attn-split-span`, default
64, 16..4096), and the five full layers keep either bf16 rows (default on
GB10, tensor-core) or E4M3 rows (one 2,056-byte combined row per token instead
of 4,096; the same tensor-core kernels since the port, decoding each element on
the way into the shared-memory tile).

**Expert placement, the three llama.cpp-shaped knobs.**
`DS4_GEMMA4_CPU_MOE=1` (`--gemma4-cpu-moe`, llama.cpp's `-cmoe` shape) keeps
every expert host-cold in the mmap and runs the top-8 on the CPU with an AVX2
R13+8 GEMV.
`DS4_GEMMA4_MOE_CPU_LAYERS=N` (`--gemma4-moe-cpu-layers`, the `-ncmoe` shape)
sends layers below N to the host and keeps the rest device-resident (accepted
0..999, clamped to the 30 routed layers, so 30, 99 and 999 all mean every
layer), with the per-layer dispatch chosen at graph build.
`DS4_GEMMA4_MOE_CACHE_MIB=N` (`--gemma4-moe-cache-mib`, the
`--moe-cache-mib` shape) keeps every expert host-cold and fills N MiB of VRAM
slots from the mapping as the router asks, optionally from host memory
(`DS4_GEMMA4_MOE_HOST=1`) and with a per-decode-miss policy
(`DS4_GEMMA4_MOE_MISS=evict|cpu`, evict by default).
`DS4_GEMMA4_MOE_TRACE=1` prints a per-layer routing histogram at exit
(top-8/16/32/64 shares and the eight hottest ids), which is the sizing input
for the pool.

**The expert fill.** A pinned staging ring fed by queued copies, a per-layer
slot map and a warm-at-open pass that touches every expert record once. With
those, a 7,239-token prompt at ctx 8192 went from 22.0 s TTFT (329 tok/s
prefill) to 5.71 s (1,279 tok/s), with the reply md5 unchanged; the fill runs
at 8.27 GiB/s once the mapping is warm, at 43.5 ms to submit and 1.4 ms to
drain 6.51 GiB.

**The serving quote and the boot admission.** The quote prices the KV rows,
the session workspace (activation set + linear workspace + routed-expert
scratch), the expert slot pool and the device-resident spans, and the server
refuses to boot when the serial session cannot keep 384 MiB free at peak,
naming the deficit:

```
ds4: Gemma-4 session does not fit: ctx=8192 chunk=4096 need=2.61 GiB
     (KV 1.05 GiB + workspace 1596.5 MiB: activation 594.3 + linear 32.1
     + moe 970.1) + margin 384 MiB > free 2823.75 MiB (deficit 237.1 MiB)
```

**A typed knob surface.** 28 arithmetic options parse in Rust with strict
ranges and errors that name the flag or variable, travel through the engine
options struct, and are echoed by `--check-config` and the plan JSON. The
`DS4_GEMMA4_*` environment path is kept only for the deprecated C hosts.

**Session KV parking.** `--kv-disk-dir` (launcher `DS4_GEMMA4_KV_DIR`) parks a
session's whole cache on disk and restores it on a later request instead of
re-prefilling. On the 4070 SUPER the first turn pays about 1.10 s to write a
989 MB checkpoint; a returning conversation restores in 0.52-0.95 s instead of
7.69 s, and an identical 62K prompt returns in 1.0-2.2 s instead of 57.08 s.

**Tooling.** `ds4-advisor` runs `ds4-bench` on every candidate arm at one
context, samples the card's peak memory, prunes arms that cannot fit, sizes the
expert pool from the VRAM free at that moment, and ranks by prefill, decode or
their harmonic mean (`--prefer balanced|prefill|decode`); it writes CSV and
JSON and refuses to rank an arm that eats the measured free-memory margin.
`run-advisor.sh` is the 128K sweep that stops and restores the launcher's
endpoint; `run-gemma4.sh` does `serve start|stop|status|logs`, `plan`, `chat`
and `bench`.

**A linear host tokenizer.** The BPE finds the lowest merge rank with a
min-heap instead of rescanning the symbol list, and the token stream is
byte-identical to the scan it replaced:

| Request on a GB10 | Before | After |
|---|---:|---:|
| 61,872 tokens, cold | 98.20 s TTFT | 62.92 s TTFT (62.84 s of prefill) |
| 61,871 tokens, fully cached | 34.46 s TTFT | 0.12 s TTFT |
| 60 KB tokenize | 7.43 s | 0.27 s |

## Performance

### GB10 (DGX Spark), live endpoint

Build `8137cb52`, bf16 full-layer KV, 2,048-row prefill chunk, persistent MoE,
greedy 32 tokens, cold prefix, measured 2026-11-05. The busiest SM was observed
at 2,398 MHz during the run, so the 2,200 MHz clock policy was not in force on
that host and these rows are not policy-normalized.

| Sweep ctx | Prompt tokens | TTFT | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,725 | 7.95 s | 1,098.5 | 35.2 |
| 32,768 | 35,547 | 34.24 s | 1,039.7 | 31.5 |
| 61,440 | 63,832 | 65.44 s | 976.9 | 28.0 |
| 131,072 | 137,101 | 160.99 s | 853.0 | 22.1 |

### RTX 4070 SUPER, 12 GiB

Cold server per arm, greedy 32 tokens, empty prefix cache. TTFT is the first
token; VRAM is boot/peak.

| Arm | ctx | Prompt | TTFT | Decode | VRAM |
|---|---:|---:|---:|---:|---|
| resident | 8,192 | 7,555 | 5.87 s | 54.8 tok/s | 10,121 / 10,861 MiB |
| resident | 32,768 | 31,909 | 26.45 s | 52.5 tok/s | 10,121 / 11,113 MiB |
| CPU MoE (`DS4_GEMMA4_CPU_MOE=1`) | 8,192 | 7,555 | 362.8 s | 17.3 tok/s | 3,617 / 3,673 MiB |
| cache 4 GiB, `miss=cpu` | 8,192 | 7,555 | 22.1 s | 28.0 tok/s | 7,713 / 7,773 MiB |

The served 128K arm, measured through the OpenAI API: ctx 131,072, E4M3
full-layer KV, chunk 512, expert pool 5,856 MiB (3,130 of 3,840 records),
`DS4_GEMMA4_MOE_MISS=evict`, build `8137cb52`, measured 2026-11-05. Cold
prefix, greedy 32 tokens; TTFT, prefill and decode come from the response's
`timings` object, VRAM from `nvidia-smi` during the run.

| Prompt tokens | TTFT | Prefill tok/s | Decode tok/s | VRAM used |
|---:|---:|---:|---:|---:|
| 8,725 | 8.86 s | 989.4 | 36.6 | 10,749 MiB |
| 35,547 | 35.09 s | 1,014.9 | 34.4 | 10,761 MiB |
| 63,832 | 68.19 s | 937.6 | 31.9 | 10,766 MiB |
| 125,666 | 161.79 s | 777.7 | 32.9 | 10,916 MiB |

### The expert fill

One prompt (7,239 tokens, ctx 8192, chunk 2048) with the reply md5
`57a0da50adbfbc90ca06b5e73cb6d842` in every row, so the changes are
token-identical:

| Step | TTFT | Prefill tok/s |
|---|---:|---:|
| cache 4 GiB, chunk 512, linear scan | 22.0 s | 329 |
| + chunk 2048 | 13.8 s | 524 |
| + pool 7 GiB | 12.2 s | 594 |
| + per-layer slot map | 10.1 s | 718 |
| + warm the mapping at open | 5.71 s | 1,279 |

## Quickstart

Build the runtime (`make cuda CUDA_ARCH=sm_89` on a 12 GiB card, `sm_121` on a
GB10), then serve the release directory:

```sh
# GB10: bf16 full-layer KV, the model's own 262,144-token context
./ds4-server --cuda -m <model-dir> -c 262144 --mem-floor-gb 0 --host 127.0.0.1 --port 8000

# 12 GiB card: E4M3 full-layer KV, 65,536 tokens resident
DS4_GEMMA4_FULL_KV=fp8 \
  ./ds4-server --cuda -m <model-dir> -c 65536 --mem-floor-gb 0 --host 127.0.0.1 --port 8000

# 12 GiB card, 131,072 tokens: experts in a 5,856 MiB VRAM slot pool
DS4_GEMMA4_FULL_KV=fp8 DS4_GEMMA4_MOE_CACHE_MIB=5856 \
  ./ds4-server --cuda -m <model-dir> -c 131072 --mem-floor-gb 0 --host 127.0.0.1 --port 8000
```

`<model-dir>` is the extracted release directory and the advertised
`/v1/models` id is that directory's name. Environment knobs used above:
`DS4_GEMMA4_FULL_KV=bf16|fp8`, `DS4_GEMMA4_MOE_CACHE_MIB`, `DS4_GEMMA4_CPU_MOE`,
`DS4_GEMMA4_MOE_CPU_LAYERS`, `DS4_GEMMA4_PREFILL_CHUNK`,
`DS4_GEMMA4_ATTN_SPLIT_SPAN`. Every one of them also has a typed
`--gemma4-*` flag.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
reply = client.chat.completions.create(
    model="<model-dir-name>",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

## Limits and open work

- The MTP drafter (P8) is still in flight. The drafter part ships and is
  validated by the inventory, and `--mtp-draft` is declared and gated, but the
  served default is plain split decode; at 13,447 tokens the drafter is a net
  loss.
- The vision tower is a sidecar part: inventoried and refused, never encoded.
  This release is text only.
- No KLD or perplexity against the original checkpoint is measured. The
  evidence is the weight-space reconstruction (23.39 percent relL2) and a
  two-probe reference gate against an fp32 transformers forward, not a corpus.
- The E4M3 full-layer lane runs on tensor cores since `6dfe55cd`: at 32,768
  tokens it prefills at 746.6 tok/s against bf16's 804.6 and decodes at 19.27
  tok/s against 18.71. The 9.6x prefill and 61x decode figures at the same
  context belong to the scalar fallback (`--gemma4-attn-mma 0` with
  `--gemma4-attn-split-mma 0`), not to the lane.
- The sweep's 65,536 frontier is missing: the byte-calibrated prompt lands a
  few tokens over the booted context and is refused with "prompt exceeds
  context", so both committed 2K-64K CSVs end at 63,488.
- The session context is right-sized below the advertised size on mixed lanes:
  with `DS4_GEMMA4_MOE_CPU_LAYERS` between 1 and 29, or with a 2-4 GiB
  `DS4_GEMMA4_MOE_CACHE_MIB` pool, a 7,555-token prompt at ctx 8192 is refused
  while the same prompt is accepted at N=0 and N=30.
- The mix lane keeps each layer's own accumulation order, so token identity has
  to be defined against `DS4_GEMMA4_PERSIST=0`; the persistent MoE decode is
  near-exact (26 of 30 layers bit-identical), and that is the parity the
  placement change keeps.
- The serial session has no ring-resume guard: a divergent continuation that
  reaches back past the rows the sliding ring still holds can read overwritten
  slots. The banked lane is exact.
- `--ssd-streaming` on this family still takes the shared GLM-5.3 branch rather
  than the Gemma discrete map; the expert slot pool is the streaming lane that
  works today.

## Links

- Runtime and pull request (draft):
  [Baekpica/ds4-dfm-rs#141](https://github.com/Baekpica/ds4-dfm-rs/pull/141)
- Engine, format and serving document:
  [docs/gemma4-vq.md](https://github.com/Baekpica/ds4-dfm-rs/blob/feat/gemma4-family/docs/gemma4-vq.md)

The conversion method is the dense 8-dimensional VQ line from YoungAi (MIT);
the container is GGUF and the Q4_K/Q8_0 layouts come from llama.cpp. Gemma is a
trademark of Google; this is an unofficial quantization, not a Google release.
