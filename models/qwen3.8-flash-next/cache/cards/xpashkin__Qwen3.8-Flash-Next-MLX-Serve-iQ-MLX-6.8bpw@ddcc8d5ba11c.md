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

# Qwen3.8-Flash-Next iQ-MLX 6.8 bpw

Imatrix-calibrated quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(133B MoE, Qwen4: 48 layers, 512 experts/layer, hybrid GDN + sparse QSA attention)
in the [mlx-serve](https://github.com/ddalcu/mlx-serve) format (validated on 26.10.2 / MLX 0.32.3).
Target platform: 128 GB Apple Silicon Mac.

## Layout and sizes

| Block | Format | Size |
|---|---|---|
| Experts (96 gate_up groups) | imatrix-weighted affine, gs64, **6-bit on 88 groups, 8-bit on 8** (layers 5, 20, 23, 26, 27, 28, 33, 38) | 101.5 GB |
| MTP head (1 layer, inline in the trunk) | affine 6-bit (plain) | ≈2.2 GB |
| Non-expert (GDN qkv/z/a/b/out, QSA q/k/v/o + indexer, hyper-connections, shared expert, router, ple) | imatrix-weighted affine **8-bit** | ≈4.6 GB |
| lm_head | affine 8-bit | ≈0.7 GB |
| embed_tokens | affine 8-bit | ≈0.7 GB |
| Vision tower (Qwen3-VL ViT, 333 tensors) | bf16 (passthrough) | 0.9 GB |
| **N-gram PLE table (51.2B entries)** | **8-bit, gs32** | **57.6 GB** |
| **Total on disk** | | **≈156 GiB** (index 109.8 GB + ngram 57.6 GB) |

Calculated bits-per-weight: **6.788** (trunk 108.9 GB × 8 / 128.351 B parameters).

Resident on a 128 GB Mac: **≈110 GB** (measured 109.8 GB; wired peak ≈113 GB at a
260k context — inside the 126 GB GPU limit).
The n-gram table stays in the page cache (not resident); do **not** pass `--ple-gpu` —
it would pin all 53.6 GB of the table in GPU memory.

The KV cache is cheap: **7.28 GB for the full 262k context** (the 36 GDN layers carry a
constant SSM state; only the 12 QSA layers have per-token KV, ≈28 KB/token) → unquantized
bf16 KV for two full 262k contexts ≈ 14.6 GB. Max validated prompt: **260 451 tokens**.

The imatrix comes from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
(`imatrix_unsloth.gguf_file`, 829k-token corpus), remapped into HF-named safetensors
(coverage 866/866 keys: experts, shared expert, router, GDN, QSA, hyper-connections, ple;
MTP is intentionally not covered).

## Running (macOS, Apple Silicon)

One time:

```bash
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew trust ddalcu/mlx-serve
brew install mlx-serve        # CLI + server only, no GUI
# if dyld complains about libmlx.dylib:
# ln -sfn /opt/homebrew/opt/mlx /opt/homebrew/Cellar/mlx-serve/lib/mlx
```

Server (auto-download from HF; validated configuration, full 262144 context,
bf16 KV, SSD prefix cache for multi-turn reuse):

```bash
mlx-serve --model xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.8bpw \
    --serve --port 11234 --prefix-cache-mem 2GB --prefix-cache-disk 60GB \
    --os-reserve-gib 2 --ctx-size 262144
```

MTP stays on by default (recommended — it speeds up decode and outputs are
identical, being speculative). The MTP head is picked up automatically
(72 `language_model.mtp.*` tensors in the trunk).

**Stock loader.** The non-expert surface is fully quantized, so this pack loads
with stock mlx-serve — no loader patch required.

**Stock-MLX note (Metal object-count guard).** The Metal allocator counts
every live MTLBuffer — weights, hot prefix-cache KV/SSM states, MTP
snapshots, and the buffer-reuse pool — against a device object-count limit.
On macOS 26+ that limit falls back to a stale hardcoded 499,000 (the sysctl
that reported the real limit was removed), so long multi-turn sessions with
several hot-cache entries can exhaust it: requests get abandoned as
"out of GPU memory" even with plenty of free memory. Discussed upstream
(ml-explore/mlx#3849, ml-explore/mlx#4464) — maintainers keep the guard as
an intentional canary; there is no API to raise it.

With unpatched mlx-serve the workarounds are: cap the hot-cache entry count
(`--prefix-cache-entries <n>`), shrink the buffer-reuse pool
(`MLX_SERVE_CACHE_LIMIT=2GB`), or, as a last resort, `--no-mtp`.

The serving instance used here runs a locally built libmlx with two allocator
patches: count-based LRU eviction of the reuse pool when the object count
approaches the guard, and the stale fallback raised 499,000 → 4,000,000
(byte limits — wired ceiling, `--os-reserve-gib`, `--prefix-cache-mem` —
remain the real OOM protection). Validated over a full day of agent-style
sessions and full-context prefill up to a 260,451-token prompt: zero
resource-limit errors, wired peak 113 GB of the 126 GB limit.

Local checkout instead of HF: `--model /path/to/pack`.

- Batched decode is on by default; "parallel N" for MLX packs is set via the KV
  budget `--prefix-cache-mem` (the engine clamps it to `iogpu.wired_limit` itself).
- API: OpenAI-compatible — `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages` (Anthropic), tools, vision, reasoning,
  `logprobs`, per-request `kv_quant` override.

## Performance (M4 Max 128 GB, mlx-serve 26.10.2)

Unique random prompts (no prefix-cache hits); prefill/decode taken from the
server's own measurements; generation capped at 96 tokens; MTP adaptive (default).

| ctx (actual) | prefill tok/s | decode tok/s |
|---|---|---|
| 1.5k (1524) | 481 | 78 * |
| 6k (6004) | 779 | 73 |
| 12k (12069) | 752 | 71 |
| 24k (24179) | 743 | 84 * |
| 48k (48430) | 732 | 76 |
| 98k (97985) | 716 | 72 |
| 197k (197047) | 676 | 59 |
| **260k (max)** | 597 | 60 |

\* short generation (the model ended at 47–48 tokens) — the decode figure is noisier.

Prefill holds ≈597–779 tok/s across the whole range; decode runs 59–84 tok/s and is
nearly context-independent (the sparse attention works as designed).
Parallel 2 (2×48k concurrently): decode 67/98 tok/s per request, both responses complete.

MTP depth: adaptive (the default) is the best overall. For long-context-heavy setups,
pinned depth 4 (`--mtp-min-depth 4 --mtp-max-depth 4`) lifts 98k decode 72.3 → 77.3 tok/s;
at 24k adaptive still wins (84.1 tok/s).

## Quality

Scored on a fixed 322-line reference corpus (cold server, greedy decoding, two runs
bit-for-bit identical — the server is deterministic): **top-1 93.5%, mean KLD 0.058**.
Long-context recall: a single fact placed at the 10/50/90% positions of 32k, 128k and
256k-token prompts is retrieved 3/3 at all three context sizes.

## Conversion recipe (reproduction)

All on an ordinary 12-core Linux box (89 GB RAM, ≈1 TB NVMe). Total budget:
≈7 hours (5.5 h download + ≈1 h conversion).

### 0. Environment

```bash
# IMPORTANT: the PyPI `mlx` wheel for Linux is a stub without libmlx.so
# (on both cp313 and cp314). You need python 3.13 + the full manylinux bundle:
uv venv --python 3.13 ~/venv313
uv pip install --python ~/venv313/bin/python \
    numpy scipy safetensors huggingface_hub gguf tqdm "mlx[cpu]"
git clone https://github.com/ddalcu/mlx-serve ~/mlx-serve   # tests/ = the whole pipeline
```

### 1. Sources

```bash
hf download Qwen/Qwen3.8-Flash-Next --local-dir ~/models/Qwen3.8-Flash-Next        # 360 GB bf16, 144 files
hf download unsloth/Qwen3.8-Flash-Next-GGUF imatrix_unsloth.gguf_file \
    --local-dir ~/iq/base                                                            # 580 MB
```

### 2. Shim: GGUF imatrix → safetensors

The GGUF imatrix carries `<weight>.in_sum2` + `<weight>.counts` pairs (f32,
new format), 1852 tensors, 829k-token corpus. Key details:

- gguf-py's `.data` lies on bulk reads — read via
  `np.frombuffer(bytes, f4, count=n, offset=t.data_offset).reshape(dims[::-1])`;
- experts: in_sum2 loads C-contiguous as `[E=512, in]` (GGUF dims `[in, E]` are
  reversed); counts = per-expert ncall, normalize per row;
- fusion: gate+up (shared input x) = mean of the normalized vectors;
  indexer q+k → `index_qk_proj`;
- GGUF→HF name mapping: `attn_q/k/v/output` → `self_attn.{q,k,v,o}_proj`
  (QSA layers 3,7,…,47), `attn_qkv` → `linear_attn.in_proj_qkv`,
  `attn_gate` → `in_proj_z`, `ssm_alpha/beta` → `in_proj_a/b`,
  `ssm_out` → `out_proj` (GDN = the other 36 layers),
  `hc_{attn,ffn}_{down,up,inject}` → `{attn,mlp}_hyper_connection.*`,
  `ple_key/value` → `layers.1.ple.{key,value}_proj`.

**Correctness gate** (mandatory before conversion): structural check — the imatrix
vectors of single-input groups must correlate: router↔experts-mean↔shared-expert,
GDN qkv↔z↔a↔b → r = 0.992–1.000. (The classic "quant-search gain" does not work as
a signal here: the affine (s,b) search is barely sensitive to channel weights —
the effect is ≈1%.)

### 3. Measure + allocate (`tests/qwen38_flash_next_iq_allocate.py`)

```bash
cd ~/mlx-serve
# error curves for 8 widths (2x64…8x64) over 96 groups (42 layers × gate_up/down), ~23 min:
venv313/bin/python tests/qwen38_flash_next_iq_allocate.py measure \
    --src ~/models/Qwen3.8-Flash-Next --imatrix ~/iq/imatrix.safetensors \
    --jobs 10 --out ~/iq/errors.json
# greedy allocator: 101.6 GB expert budget, floor 6x64, 2 tail layers pinned to 6x64:
venv313/bin/python tests/qwen38_flash_next_iq_allocate.py allocate \
    --errors ~/iq/errors.json --budget-gb 101.6 --floor 6x64 \
    --tail-layers 2 --tail-width 6x64 --out ~/iq/alloc.json
```

Result: **88 groups at 6x64, 8 groups at 8x64** (101.50 GB spent) — the greedy
allocator promotes the groups with the highest size-weighted error mass. The
trunk budget (≈109.1 GB) is: experts 101.6 + dense 8-bit 4.6 + embed 8-bit 0.68
+ MTP 6-bit 2.2 + norms bf16 0.02.

### 4. Conversion (≈66 min, 12 CPU)

```bash
venv313/bin/python tests/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next \
    --imatrix ~/iq/imatrix.safetensors --alloc ~/iq/alloc.json \
    --bits 6 --nonexpert-bits 8 --embed-bits 8 --ngram-bits 8 \
    --ngram-link ~/models/...-6.6bpw/ngram_table.bin \
    --jobs 10 \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.8bpw
```

- `--alloc` — per-group expert widths from step 3 (floor 6x64);
- `--nonexpert-bits 8` — all 2-D non-expert weights (rows≥32, cols%64==0) at 8-bit;
- `--embed-bits 8` — the embedding at 8-bit;
- `--ngram-bits 8` — 8-bit n-gram table → `ngram_table.bin` 57.6 GB + `config.json:
  quantization.calibration = "imatrix-weighted affine (iQ-MLX)"`;
- `--ngram-link` — hard-link an existing 8-bit table of this model (identical
  across the iQ-MLX releases; e.g. from the 6.6bpw/7.1bpw repos) instead of
  re-quantizing 57.6 GB (drop the flag to re-quantize from the bf16 source);
- resumable: progress in `.convert_state.json`, kill/restart is safe.

### 5. Vision pass (no re-stream)

```bash
venv313/bin/python tests/convert_qwen38_flash_next.py --add-vision \
    --src ~/models/Qwen3.8-Flash-Next \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.8bpw
```

→ `model-vision.safetensors` (0.9 GB), `config.json: language_model_only: false`.

## Pipeline artifacts (for reproduction)

| File | What |
|---|---|
| `imatrix.safetensors` (328 MB) | shim output, imatrix in HF key names |
| `errors.json` | error curves, 96 groups × 8 widths |
| `alloc.json` | per-group widths (88× 6x64, 8× 8x64) |
| measure/convert/vision logs | pipeline trace |
| bf16 checkpoint (360 GB) | can be deleted after the pack is shipped |
