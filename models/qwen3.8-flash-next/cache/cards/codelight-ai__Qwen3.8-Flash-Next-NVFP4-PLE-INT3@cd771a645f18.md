---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- RadixArk/Qwen3.8-Flash-Next-NVFP4
- Qwen/Qwen3.8-Flash-Next
tags:
- qwen4_exp
- NVFP4
- int3
- PLE
- gb10
- dgx-spark
- experimental
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next — NVFP4 experts + INT3 PLE

**Express goal: fit Qwen3.8-Flash-Next (~180B) on a single NVIDIA GB10 / DGX Spark (128 GB).**

**STATUS: RUNS AND GENERATES CORRECTLY.** Validated end-to-end on a single GB10 on
2026-08-27. Short-form quality is intact; long-form/agentic quality is still unmeasured,
so treat as experimental.

```
"What is 2 + 3?"               -> "5"        finish=stop   (35 completion tokens)
"Capital of France, one word"  -> "Paris"    finish=stop
"17 sheep, all but 9 run away" -> "9 left"   correct at low/medium/xhigh effort
tool calling                   -> 4/4 cases pass (see below)
prefill 2048-in                -> ~1017 tok/s total
decode                         -> ~16 tok/s  (--enforce-eager, no CUDA graphs: a FLOOR)
```

For reference, RadixArk's own FP8 smoke test on the same prompt recorded **35 completion
tokens, finish_reason=stop** — this INT3 build matches that exactly.

**Tool calling** (`--tool-call-parser qwen3_xml`): single-tool call with valid JSON args;
correct tool chosen when two are offered; multi-turn tool-result handling; and
`tool_choice="none"` correctly suppressed. **Reasoning** (`--reasoning-parser qwen3`):
`low`/`medium`/`xhigh` all work (**not** `high` — the model rejects it). Note
`reasoning_content` comes back empty even though `usage.reasoning_tokens` is 104-141, so
the thinking text is stripped but not exposed.

See [`RUNNING.md`](RUNNING.md) for the serving config. It will **not** load on stock vLLM.

## Why this exists

`RadixArk/Qwen3.8-Flash-Next-NVFP4` is 135 GB and needs **126 GiB resident**. A GB10 has
**119 GiB of unified memory shared by CPU and GPU**. It does not fit, and no serving flag
changes that — the weights alone exceed the machine.

The reason is not the experts. It is the **PLE n-gram embedding table**: 20,000,000 rows x
160 dims x 16 n-gram heads = **51.2B parameters**, 38% of the checkpoint, shipped at FP8 and
never quantized below 8 bits by anyone. Taking it to 3.2 bits is what makes the model fit.

| | RadixArk NVFP4 | **this checkpoint** |
|---|---|---|
| routed experts (48x512) | NVFP4, 68.0 GB | **NVFP4, 68.0 GB — inherited byte-identical** |
| PLE n-gram tables | FP8, 51.2 GB | **INT3 asym per-row, 20.5 GB** |
| attention / GDN / QSA / mHC / vision | BF16, 16.0 GB | BF16, 10.8 GB |
| MTP head (31 tensors) | BF16, 5.2 GB | **removed** |
| **on disk** | 135.2 GB | **99.3 GB** |
| **resident** | 126.0 GiB — does not fit | **91.9 GiB — fits, ~20 GiB left for KV** |

## Post Training Quantization

Starting from `RadixArk/Qwen3.8-Flash-Next-NVFP4`, three changes were made, all targeting
GB10 residency:

**1. PLE n-gram tables: FP8 (51.2 GB) -> INT3 (20.5 GB).** All 128 shards, quantized from the
**BF16 source** in `Qwen/Qwen3.8-Flash-Next` (not from RadixArk's FP8, to avoid double
quantization). Scheme is **asymmetric, per-row, with an MSE-optimal clip ratio searched per
row** over 0.55–1.0 in steps of 0.025 — absmax scaling is notably worse at 3 bits because
outliers stretch the grid. Rows are only 160-dim, so a per-row scale is already fine-grained;
explicit grouping (g40/g32/g16) cost more bits than it bought. Storage is
3 + 32/160 = **3.20 bits/weight**: 60 packed bytes + bf16 scale + bf16 zero-point per row.
This is **weight-only** (activations stay BF16) and required **no calibration data**.

**2. Routed experts: unchanged.** RadixArk's NVFP4 W4A4 tensors are copied byte-identical.
Their recipe used activation calibration captured from live SGLang serving on GB300s; there
was no reason to redo it and no way to improve on 4.5 bits/weight. Measurements below confirm
NVFP4 is at the practical floor for these tensors.

**3. MTP head removed** (31 tensors, 5.21 GB — `mtp.layers.0.mlp.experts.*` is 97% of it).
The main model skips these entirely; they are only loaded when speculative decoding is
configured with `method="mtp"`. Speculative decoding is therefore **not available** in this
checkpoint. To restore it, re-assemble using RadixArk's `model-bf16-*` shards, which are
unmodified there.

Not touched: attention, QSA, GDN, mHC, shared experts, **routers**, embeddings, LM head, and
the vision tower all remain BF16. Routers especially — 48x[2560,512] is only 0.13 GB, and a
routing error changes *which experts fire*, a discrete failure that does not average out.
No KV-cache quantization metadata.

## Measured quantization error (PLE table, weight-space)

| scheme | bits | rel-err | cos | PLE size |
|---|---|---|---|---|
| FP8 e4m3 per-tensor (RadixArk ship this) | 8.00 | 0.0265 | 0.999654 | 51.2 GB |
| INT6 asym per-row | 6.20 | 0.0241 | 0.999710 | 39.7 GB |
| INT4 asym per-row | 4.20 | 0.0958 | 0.995406 | 26.9 GB |
| **INT3 asym per-row (this)** | **3.20** | **0.1854** | **0.982685** | **20.5 GB** |

`rel-err = ||Wq-W||_F / ||W||_F` against BF16. **Weight-space proxies — end-to-end quality has
not been measured.** INT3 is ~7x the error of the FP8 it replaces. Note INT6 beats FP8 in
*less* memory: per-row asymmetric scaling with MSE-optimal clipping places 6.2 bits better
than per-tensor FP8 places 8.

For reference, on the same tensors at equal 4.5 bits, NVFP4 measures 0.0952 and INT4-g16
0.0850 — a ~1 dB difference. INT4-g128 (4.12 bits) is 0.1182, i.e. worse than NVFP4.

## Format

```
...ngram_embedding.shard_N.weight_packed       uint8 [rows, 60]
...ngram_embedding.shard_N.weight_scale        bf16  [rows, 1]
...ngram_embedding.shard_N.weight_zero_point   bf16  [rows, 1]

dequant : unpack3(packed) * scale + zero_point
packing : 8 values -> 3 bytes; 160 dims/row -> exactly 60 bytes/row, no padding
```

`config.json` declares `text_config.ple_embedding_dtype = "int3_asym_perrow"` plus a top-level
`ple_quantization` block with the full layout. **A conforming loader must fail loudly on an
unrecognised value**, never fall back to unquantized — the upstream checkpoint's own notes warn
that a naive FP8 PLE loader serves wrong embeddings silently.

Implementation note: the PLE gathers **16 n-gram heads at once**, so the packed tensor reaching
the embedding op is `[tokens, 16, 60]`, not `[tokens, 60]`. Unpack must preserve leading dims.

## Serving on a GB10

Resident 91.9 GiB, leaving ~20 GiB for KV at util 0.85. Only 12 of 48 layers are full
attention — the other 36 are GDN with constant per-sequence state — so KV costs ~12.4 KiB/token
and a 262k-token sequence needs only ~3.4 GB.

Read [`RUNNING.md`](RUNNING.md) first.

**The required vLLM changes ship with this repo** as
[`vllm-qwen4-exp-fixes.patch`](vllm-qwen4-exp-fixes.patch). Apply it on top of PR #53896:

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git remote add pr https://github.com/peakcrosser7/vllm.git
git fetch pr release/qwen38next && git checkout -b qwen4-exp pr/release/qwen38next
git am < /path/to/vllm-qwen4-exp-fixes.patch
TORCH_CUDA_ARCH_LIST="12.0" pip install -e .    # ~1h on a GB10
```

It carries three fixes. **Two are bugs in PR #53896 itself** and block loading for anyone,
with or without this quantization; the third adds the INT3/4/6 PLE path.

## License

Qwen Community License 1.0 (see `LICENSE`), inherited from `Qwen/Qwen3.8-Flash-Next`.
Clause 2 requires a separate license from Qwen for commercial Model-as-a-Service or AI Work
Assistant use; internal use is exempt.
