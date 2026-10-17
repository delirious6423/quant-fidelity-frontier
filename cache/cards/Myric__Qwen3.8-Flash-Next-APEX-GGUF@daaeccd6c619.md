---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - moe
  - apex
  - quantized
  - qwen
  - llama.cpp
  - multimodal
---

# Qwen3.8-Flash-Next — APEX GGUF

GGUF conversions of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
— a 180 B-total / ~6.8 B-active multimodal MoE (48 layers, 512 routed experts, top-10
routing, hybrid gated-delta-net with full attention every 4th layer, QSA sparse attention,
per-layer n-gram embeddings, 262 K context).

Two files are usable directly: **MIDDLE** for inference, **BF16** as a quantization source.

## Files

| File | Size | bpw | Notes |
|------|------|-----|-------|
| `qwen3.8-flash-next-MIDDLE.gguf` | 91.7 GB | 4.144 | mixed recipe, 1224 tensors |
| `qwen3.8-flash-next-BF16.gguf` | 354.0 GB | 16.0 | lossless conversion, quantization/imatrix source |
| `qwen3.8-flash-next-mmproj-F16.gguf` | 0.9 GB | — | vision projector, pass with `--mmproj` |

The BF16 is a **lossless** conversion of the tensors it contains: the checkpoint is already
`bfloat16` on disk (`config.json` → `"dtype": "bfloat16"`), so the GGUF holds identical bit
patterns. There is no reason to run it for inference.

## llama.cpp compatibility

**qwen4exp is in mainline.** [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
merged 2026-08-27 — about nine hours after these files were first published — followed by
[#27880](https://github.com/ggml-org/llama.cpp/pull/27880),
[#28023](https://github.com/ggml-org/llama.cpp/pull/28023),
[#28123](https://github.com/ggml-org/llama.cpp/pull/28123) and
[#27941](https://github.com/ggml-org/llama.cpp/pull/27941).

Verified against a **stock, unpatched** build (`b2276-22397c31a`, 2026-09-09) rather than
inferred from the merge: the published MIDDLE file loads and generates correctly, and
`--parallel 4` served four concurrent requests without the indexer/attention cache assert that
the pre-merge build hit. Perplexity below was measured on that same stock build.

If your build predates the merge, use
[brywil/llama.cpp @ `qwen4exp-pr27742-spill-fix`](https://github.com/brywil/llama.cpp/tree/qwen4exp-pr27742-spill-fix).

**Converting this model yourself still needs that fork.** The converter has to spill large
tensors to disk and dequantize them in row bands instead of materialising them whole; without
that it cannot produce these files at all. That fix is not upstream. Running the published files
does not need it.

## Running it

MIDDLE needs ~85.4 GiB resident for weights, plus KV cache and compute buffers.

```
llama-server \
  --model qwen3.8-flash-next-MIDDLE.gguf \
  --mmproj qwen3.8-flash-next-mmproj-F16.gguf \
  -ngl 999 -fa on -c 131072 \
  --parallel 1 \
  --cache-type-k f16 --cache-type-v f16 \
  -b 4096 -ub 2048 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0
```

Three of those flags are **not tuning choices**:

- **`--parallel 1`.** The QSA sparse-attention indexer keeps its own KV cache alongside the
  attention cache, and multi-slot serving can desync the two. Serving with the default 4 slots
  aborted a benchmark run at task 2 of 9 on
  `GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv())`.
- **f16 KV only.** `q8_0` KV crashes or emits garbage on this architecture.
- **Vendor sampling.** `temp 1.0 / top_k 20 / top_p 0.95 / min_p 0` is this checkpoint's own
  `generation_config.json` default. Substituting another model's numbers (an 0.6 carried over
  from an unrelated card) made a whole benchmark run incomparable.

KV cache is cheap here — only 12 of the 48 layers are full attention (2 KV heads, head_dim 256),
so **24 KiB/token**: 3.0 GiB at 131072, 6.0 GiB at the full 262144. The 36 gated-delta-net
layers hold a constant-size recurrent state instead. Note that the indexer maintains a second
cache of its own, so trust measured resident size over a hand-computed figure.

## Measured

DGX Spark (GB10), stock llama.cpp `b2276-22397c31a`, MIDDLE, f16 KV, single slot:

| | value |
|---|---|
| generation | 31.1 t/s (8192 ctx) |
| prompt | 106.8 t/s at 24 tokens |
| decode, 131072 ctx (fork build) | 19.9-20.5 t/s |
| prefill | 69 t/s at 20 tokens, 594 t/s at 23k |

Under matched flags (8192 ctx, f16 KV, single slot, same prompt and seed), generation measured
31.1 t/s on the stock mainline build against 28.3 t/s on the pre-merge fork build.

### Vision

MIDDLE plus `qwen3.8-flash-next-mmproj-F16.gguf`, stock llama.cpp, five perception tasks put to
the server directly (read a placard, transcribe a handwritten spec, name a function and its buggy
line from a screenshot, read a value off a bar chart, count shapes):

| arm | passed |
|---|---|
| with the image | **5 of 5** |
| identical questions, image withheld | **0 of 5** |

The second row is the control: every answer is unguessable by construction, so a blind pass would
mean the task was measuring text priors rather than the projector. None passed blind.

Note the sampling requirement this exposed: this model's chat template accepts reasoning effort
`xhigh`, `medium` or `low` and **raises on `high`**, returning HTTP 500 for every request. A
harness defaulting to `high` gets a 100% failure rate from a model that loads perfectly.

### Agentic coding

MIDDLE on the frontier tier of an agentic coding suite (9 tasks, real tool-use loop, graded by a
fresh pytest run), one run, DGX Spark (GB10), stock llama.cpp, `--reasoning medium`, 65536 ctx,
single slot:

| | result |
|---|---|
| tasks solved | **9 of 9** |
| individual tests passed | **154 of 154** |
| tasks lost to harness limits (timeout or output cap) | **none** |
| mean output tokens per task | 4,429 |
| total wall time | 0.53 h |

Per task: backtracking_regex 59 of 59 tests (5,552 output tokens), btree_insert_delete 14 of 14
(23,361), lru_ttl_cache 12 of 12 (1,724), mini_sql_executor 14 of 14 (1,911), or_set_crdt 9 of 9
(1,616), rate_limiter_fix_and_extend 8 of 8 (1,738), tiered_rate_limiter 6 of 6 (1,115),
txn_kv_store 12 of 12 (1,557), weighted_interval_scheduling 8 of 8 (1,284).

A clean sweep on this tier is a floor rather than a ranking: the suite is built so that a
competent coder clears it, and once it saturates the discriminating measurement is output tokens
at equal score, which needs several runs per configuration. This is one run.

### Perplexity

**Not quoted here yet.** A wikitext-2 run on this box returned 4.2801 +/- 0.02495
(`--ctx-size 2048 --chunks 145 --seed 1234`), but the corpus file used has sha256
`aca2f467...` and is not the canonical `wikitext-2-raw-v1` release artifact
(`173c87a5...`). `n_chunk` is a function of tokenised byte count, so a different copy of the
corpus changes the divisor; the figure is therefore not comparable to perplexities published
for other builds of this model, and is left out rather than shown next to them. It will be
restored once measured against the release artifact.

## What MIDDLE actually contains

Measured by reading the published file's tensor index — 1224 tensors, 176.944 B parameters,
91,655,638,816 bytes, **4.144 bpw**.

| tensors | params | type |
|---|---|---|
| `per_layer_token_embd` (1) | 51.200 B | Q4_0 |
| `ffn_down_exps` (48) | 40.265 B | Q4_0 |
| `ffn_gate_exps` + `ffn_up_exps` (96) | 80.531 B | IQ4_XS (40) · IQ3_XXS (56) |
| attention, dense FFN, shared experts, `ssm_out`, `attn_gate` | 3.646 B | Q6_K · Q8_0 · F32 |
| `token_embd`, `output` (2) | 1.272 B | Q6_K |
| `ssm_alpha` / `ssm_beta` (72) | 0.009 B | Q6_K |
| `indexer.q_proj` / `indexer.k_proj` (24) | 0.020 B | BF16 |

The two big Q4_0 blocks are not a recipe preference — see the shape constraint below.

## Not included: the MTP head

**MIDDLE and BF16 are backbone-only.** Two blocks of the 180.000 B are absent:

| | params | where it went |
|---|---|---|
| vision tower | 0.449 B | the `mmproj` file above |
| **MTP head** | **2.607 B** | **dropped — in neither file** |

That leaves 176.944 B. The MTP head is a 1-layer multi-token-prediction block for speculative
decoding; the converter sets `supports_mtp_export = False` for this architecture and discards its
31 tensors **silently** — no warning appears in the conversion log. `--mtp` is refused outright
(`--mtp / --no-nextn are not supported for Qwen4ExpForConditionalGeneration`). The closed
[PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) implemented both the export and the
runtime graph, and its author deferred to #27742 with the intent that the useful parts be
reconciled — so this is a scope cut, not a limitation of the format.

**If you need the MTP head, or full-model reproducibility, start from the original safetensors,
not from these files.**

## Notes for quantizers

Two tensor shapes make this model awkward, and both are shape properties that a name-based
recipe will not notice:

- `ffn_down_exps` rows are **640** wide (`moe_intermediate_size`), and
- `per_layer_token_embd` rows are **160** wide.

Neither is divisible by 256, so K-quants and every 256-block I-quant (`IQ4_XS`, `IQ3_XXS`,
`IQ2_S`, …) are illegal on them and silently fall back to 32-block types. That is **91.5 B of
176.9 B params — 51.7% of the file — with a floor near 4.5 bpw**, which flattens the size/quality
curve fast: pushing the remaining gate/up experts down to `IQ2_XXS` moves the total only from
~112 GB to ~77 GB. The expert gate/up tensors are unaffected (their rows are 2560 wide), which is
why the recipe above can band them at IQ4_XS/IQ3_XXS.

`per_layer_token_embd` is a 51.2 B-param sparse lookup table — only a few rows are read per token
— and at `Q4_0` it is 28.8 GB, roughly a third of the file.

On a **discrete-GPU** system it is worth keeping in host RAM with
`-ot "per_layer_token_embd=CPU"`, since host memory is a separate and usually larger pool and the
access pattern is cheap over PCIe.

On a **unified-memory** system (DGX Spark / GB10, Apple silicon) that does nothing: CPU and GPU
share one pool, so there is nowhere to offload *to*. Budget the full file size as the footprint
there, plus KV cache and compute buffers.

## Provenance

- Source: `Qwen/Qwen3.8-Flash-Next` safetensors, 144 files, 360.0 GB, all verified byte-exact
  against the Hub after download.
- Converter: `convert_hf_to_gguf.py` from PR #27742, `--outtype bf16`, with one extra fix needed
  to convert this model at all (see the compatibility section).
- MIDDLE was quantized against an importance matrix carrying 926 entries over 4,000 calibration
  chunks.
- Unofficial community conversion — **not affiliated with or endorsed by Qwen.** Provided as-is,
  without warranty.
