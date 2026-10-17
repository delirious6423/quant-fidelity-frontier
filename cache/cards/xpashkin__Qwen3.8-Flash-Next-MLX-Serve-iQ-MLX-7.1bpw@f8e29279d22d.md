---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
- imatrix
---

# Qwen3.8-Flash-Next iQ-MLX 7.1 bpw

Imatrix-calibrated quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(133B MoE, Qwen4: 48 layers, 512 experts/layer, hybrid GDN + sparse QSA attention)
in the [mlx-serve](https://github.com/ddalcu/mlx-serve) format (validated on 26.10.2 / MLX 0.32.3).
Target platform: 128 GB Apple Silicon Mac.

Higher-fidelity sibling of the [6.6 bpw pack](https://huggingface.co/xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw):
the entire attention path and recurrent state are held in **bf16**, trading
≈8 GB of weight mass for long-context fidelity.

## Requires a patched mlx-serve — read `BUILDING_DYLIB.md` first

**This pack does not load on stock mlx-serve.** The attention projections
(`self_attn.q/k/v/o_proj`) are stored as dense **bf16** weights beside the
quantized experts. Stock v26.10.2 (the current release) unconditionally
requires quantization scales/biases for those tensors when the model config
declares a quantization, and aborts at load:

```
MISSING WEIGHT: language_model.model.layers.3.self_attn.k_proj.scales
```

Before the first run, build mlx-serve per **`BUILDING_DYLIB.md`** (included in
this repo): clone upstream **v26.10.2**, apply `mlx_qwen4_dense_attn.patch`
(also included here — it applies cleanly to the tag, 31+/9− in
`src/transformer.zig`), link the server against the Homebrew release's own
dylibs (no C++ stage, no Xcode — Command Line Tools are enough), `zig build`.
≈10 minutes on an M-class Mac. The document also covers optional **libmlx
allocator patches** (`patches/`) recommended for long serving sessions of this
model class.

The 6.6 bpw sibling loads on stock builds — only this pack needs the patch.

## Layout and sizes

| Block | Format | Size |
|---|---|---|
| Experts (48 layers × 512) | imatrix-weighted affine, gs64: 85 groups **6x64** + 11 `gate_up` groups **8x64** (error-mass driven) | 102.8 GB |
| MTP head (1 layer, inline in the trunk) | affine 6-bit (plain) | 2.2 GB |
| Non-expert (QSA q/k/v/o + indexer, GDN in_qkv/z/out, hyper-connections, shared expert, router, ple, lm_head) | **bf16 (dense)** | 7.4 GB |
| embed_tokens | affine 8-bit | 0.7 GB |
| Vision tower (Qwen3-VL ViT, 333 tensors) | bf16 (passthrough) | 0.9 GB |
| **N-gram PLE table (51.2B entries)** | **8-bit, gs32** | **57.6 GB** |
| **Total on disk** | | **172.8 GB (160.9 GiB)** |

Resident on a 128 GB Mac: **≈115 GB**; at 240–257k-token contexts wired peaks
at ≈120–121 GB — inside the 126 GB limit.
The n-gram table stays in the page cache (not resident); do **not** pass
`--ple-gpu` — it would pin all 57.6 GB of the table in GPU memory.

The KV cache is cheap: **7.28 GB for the full 262k context** (the 36 GDN
layers carry a constant SSM state; only the 12 QSA layers have per-token KV,
≈28 KB/token) → unquantized bf16 KV.

Why bf16 attention: in this architecture the sparse-attention projections,
the GDN recurrent-state path, and the hyper-connection gates carry the
long-context behavior of the model; at 6-bit they are the first tier to
degrade. At bf16 the whole tier costs only ≈7 GB, which is what the 7.1 bpw
budget buys.

The imatrix comes from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(`imatrix_unsloth.gguf_file`, 829k-token corpus), remapped into HF-named
safetensors — same calibration as the 6.6 bpw pack (clean width comparison).

## Running (macOS, Apple Silicon)

One time: install mlx-serve via Homebrew, then **replace its binary with the
build from `BUILDING_DYLIB.md`** (the document's install section does exactly
that: backup, rpath, re-sign, swap).

Server (auto-download from HF; validated configuration, full 262144 context,
bf16 KV, prefix cache on SSD):

```bash
mlx-serve --model xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-7.1bpw \
    --serve --port 11234 --prefix-cache-mem 2GB --prefix-cache-disk 60GB \
    --os-reserve-gib 2 --ctx-size 262144
```

- Weights are ≈8 GB heavier than the 6.6 sibling, so the validated config
  keeps only **2 GB** of the prefix cache resident in RAM and puts the rest
  on SSD (`--prefix-cache-disk`, persists across restarts, restored instead
  of recomputed). This maximizes long-context headroom: validated up to a
  **256,719-token** prompt. On machines with more RAM, a 10 GB RAM cache is
  also fine for shorter contexts.
- The KV admission gate protects the last few percent of the context: with a
  warm cache, prompts above ≈250k can be rejected with HTTP 400
  (`...needs ~13GB ... available + reclaimable — rejecting`). Start a fresh
  server and the same prompt loads; this is graceful degradation, not a crash.
- MTP stays on by default (recommended — it speeds up decode and outputs are
  identical, being speculative).
- Batched decode is on by default; "parallel N" for MLX packs is set via the KV
  budget `--prefix-cache-mem` (the engine clamps it to `iogpu.wired_limit` itself).
- The MTP head is picked up automatically (72 `language_model.mtp.*` tensors in
  the trunk); do not pass `--no-mtp`.
- API: OpenAI-compatible — `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages` (Anthropic), tools, vision, reasoning,
  `logprobs`, per-request `kv_quant` override.

**Metal object-count guard (long sessions).** The Metal allocator counts every
live MTLBuffer — weights, hot prefix-cache KV/SSM states, MTP snapshots, and
the buffer-reuse pool — against a device object-count limit. On macOS 26+ that
limit falls back to a stale hardcoded 499,000 (the sysctl that reported the
real limit was removed), so long multi-turn sessions can exhaust it: requests
get abandoned as "out of GPU memory" even with plenty of free memory.
Discussed upstream (ml-explore/mlx#3849, ml-explore/mlx#4464); maintainers keep
the guard as an intentional canary. If your session log shows
`Resource limit (499000) exceeded`, build the patched libmlx per
`BUILDING_DYLIB.md` §3b (patches included in `patches/` here).

Local checkout instead of HF: `--model /path/to/pack`.

## Performance (M4 Max 128 GB, patched mlx-serve 26.10.2)

Unique random prompts (no prefix-cache hits); prefill/decode taken from the
server's own measurements; generation capped at 96 tokens.

| ctx (actual) | prefill tok/s | decode tok/s |
|---|---|---|
| 1.5k | 436 | 86 |
| 6k | 780 | 95 |
| 12k | 755 | 67 |
| 24k | 736 | 70 |
| 48k | 721 | 70 |
| 98k | 662 | 62 |
| 197k | 650 | 69 |
| **257k (max, cold start)** | 633 | 56 |

Prefill holds ≈630–780 tok/s across the whole range (a mild dip past ≈100k);
decode runs 56–95 tok/s and is nearly context-independent (the sparse
attention works as designed). Memory: wired ≈113 GB at small contexts,
peaking at ≈120–121 GB at the max context. Parallel 2 (2×48k concurrently):
prefills serialize, decode holds ≈67–68 tok/s per request.

## Conversion recipe (reproduction)

Same pipeline, scripts, and calibration as the
[6.6 bpw pack](https://huggingface.co/xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw)
(steps 0–2 of that README: environment, sources, imatrix shim — unchanged).
All on an ordinary 12-core Linux box (89 GB RAM, ≈1 TB NVMe). Total budget:
≈7 hours. The only differences are the allocation and convert flags:

### 3. Allocate (≈1 min)

```bash
cd ~/mlx-serve
venv313/bin/python tests/qwen38_flash_next_iq_allocate.py allocate \
    --errors ~/iq/errors.json --budget-gb 102.8 --floor 6x64 \
    --out ~/iq3/alloc.json
```

Result: **85 groups 6x64 + 11 `gate_up` groups 8x64** (layers 5, 19, 20, 23,
26, 27, 28, 33, 35, 36, 38 — the highest per-group quantization error mass);
all `down` groups stay 6x64. ≈102.8 GB spent.

### 4. Conversion (≈65 min, 12 CPU)

```bash
venv313/bin/python tests/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next \
    --imatrix ~/iq/imatrix.safetensors --alloc ~/iq3/alloc.json \
    --bits 6 --nonexpert-bits 16 --embed-bits 8 --ngram-bits 8 \
    --ngram-link ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw/ngram_table.bin \
    --jobs 10 \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-7.1bpw
```

- `--bits 6` — pin for the MTP head; experts follow alloc.json;
- `--nonexpert-bits 16` — the bf16 tier: all 2-D non-expert weights
  (attention projections, GDN projections, hyper-connections, shared expert,
  router, ple, lm_head) stay dense;
- `--embed-bits 8`;
- `--ngram-link` — hard-link the existing 8-bit n-gram table from the 6.6 bpw
  pack instead of re-quantizing 57.6 GB (drop the flag to re-quantize from
  the bf16 source);
- resumable: progress in `.convert_state.json`, kill/restart is safe.

### 5. Vision pass (no re-stream)

```bash
venv313/bin/python tests/convert_qwen38_flash_next.py --add-vision \
    --src ~/models/Qwen3.8-Flash-Next \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-7.1bpw
```

→ `model-vision.safetensors` (0.9 GB), `config.json: language_model_only: false`.

### 6. Preprocessors (the converter does not copy these)

```bash
cp ~/models/Qwen3.8-Flash-Next/preprocessor_config.json \
   ~/models/Qwen3.8-Flash-Next/video_preprocessor_config.json \
   ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-7.1bpw/
```

Without them the server still runs, but vision preprocessing is not the
HF-standard one.

### 7. Final checklist (hard gate — run it before declaring the pack ready)

1. `config.json`: `language_model_only: false` + `vision_config` present;
2. `model.safetensors.index.json`: 333 `model.visual.*` keys →
   `model-vision.safetensors`;
3. file list of the pack == the reference pack (same 113 files);
4. preprocessors copied (step 6);
5. md5 of every file: source machine == destination machine
   (including the small metadata files);
6. after server start: log shows `input_modalities` including `image`
   (`['text', 'image', 'video']`), and a one-request vision smoke passes.

## Repo contents

| File | What |
|---|---|
| `model-*.safetensors` + `model.safetensors.index.json` | the pack (trunk 114.3 GB) |
| `ngram_table.bin` | 8-bit n-gram PLE table (57.6 GB) |
| `config.json` | mlx-serve model config (bf16 KV, MTP, ctx 262144) |
| `preprocessor_config.json`, `video_preprocessor_config.json` | HF-standard vision preprocessing |
| **`BUILDING_DYLIB.md`** | **required**: build the mlx-serve that can load this pack |
| `mlx_qwen4_dense_attn.patch` | the loader patch, against upstream v26.10.2 (applies cleanly) |
| `patches/` | optional libmlx allocator patches + build/swap tooling |
