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

# Qwen3.8-Flash-Next iQ-MLX 6.6 bpw

Imatrix-calibrated quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(133B MoE, Qwen4: 48 layers, 512 experts/layer, hybrid GDN + sparse QSA attention)
in the [mlx-serve](https://github.com/ddalcu/mlx-serve) format (validated on 26.10.2 / MLX 0.32.3).
Target platform: 128 GB Apple Silicon Mac.

## Layout and sizes

| Block | Format | Size |
|---|---|---|
| Experts (42 free layers + 6 tail layers) | imatrix-weighted affine, gs64, **6-bit** (all 96 groups at 6x64) | 98.3 GB |
| MTP head (1 layer, inline in the trunk) | affine 6-bit (plain) | ≈2.5 GB |
| Non-expert (GDN qkv/z/a/b/out, QSA q/k/v/o + indexer, hyper-connections, shared expert, router, ple) | imatrix-weighted affine **8-bit** | ≈8 GB |
| lm_head | affine 8-bit | ≈0.7 GB |
| embed_tokens | affine 4-bit | ≈0.3 GB |
| Vision tower (Qwen3-VL ViT, 333 tensors) | bf16 (passthrough) | 0.9 GB |
| **N-gram PLE table (51.2B entries)** | **8-bit, gs32** | **57.6 GB** |
| **Total on disk** | | **≈153 GiB** (index 105.2 GB + ngram 57.6 GB) |

Resident on a 128 GB Mac: **≈100–112 GB** (weights 98 GB in the GPU budget;
at 240–260k contexts wired peaks at ≈112–122 GB depending on hot-cache
occupancy — inside the 126 GB limit).
The n-gram table stays in the page cache (not resident); do **not** pass `--ple-gpu` —
it would pin all 53.6 GB of the table in GPU memory.

The KV cache is cheap: **7.28 GB for the full 262k context** (the 36 GDN layers carry a
constant SSM state; only the 12 QSA layers have per-token KV, ≈28 KB/token) → unquantized
bf16 KV for two full 262k contexts ≈ 14.6 GB.

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
bf16 KV):

```bash
mlx-serve --model xpashkin/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw \
    --serve --port 11234 --prefix-cache-mem 10GB --os-reserve-gib 2 --ctx-size 262144
```

MTP stays on by default (recommended — it speeds up decode and outputs are
identical, being speculative).

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

This pack is served here on a locally built libmlx with two allocator
patches: count-based LRU eviction of the reuse pool when the object count
approaches the guard, and the stale fallback raised 499,000 → 4,000,000
(byte limits — wired ceiling, `--os-reserve-gib`, `--prefix-cache-mem` —
remain the real OOM protection). Validated on long agent-style sessions and
full-context prefill up to a 260,410-token prompt: zero resource-limit
errors, wired peak 112 GB of the 120.8 GB ceiling.

Local checkout instead of HF: `--model /path/to/pack`.

- Batched decode is on by default; "parallel N" for MLX packs is set via the KV
  budget `--prefix-cache-mem` (the engine clamps it to `iogpu.wired_limit` itself).
- The MTP head is picked up automatically (72 `language_model.mtp.*` tensors in the
  trunk); do not pass `--no-mtp` unless using the workaround above.
- API: OpenAI-compatible — `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages` (Anthropic), tools, vision, reasoning,
  `logprobs`, per-request `kv_quant` override.

## Performance (M4 Max 128 GB, mlx-serve 26.10.2)

Unique random prompts (no prefix-cache hits); prefill/decode taken from the
server's own measurements; generation capped at 96 tokens.

| ctx (actual) | prefill tok/s | decode tok/s |
|---|---|---|
| 1.5k | 474 | 106 |
| 6k | 780 | 84 |
| 12k | 758 | 76 |
| 24k | 749 | 87 |
| 48k | 736 | 52 |
| 98k | 722 | 68 |
| 197k | 648 | 67 |
| **260k (max)** | 634 | 49 |

Prefill holds ≈630–780 tok/s across the whole range (a mild dip past ≈100k);
decode runs 49–106 tok/s and is nearly context-independent (the sparse
attention works as designed). Memory: wired peaks at ≈112 GB at a 260k
context — inside the 126 GB GPU limit.
Parallel 2 (2×32k concurrently): prefills serialize (≈44 s each), decode
holds 83–89 tok/s per request, both responses complete in ≈92 s.

## Conversion recipe (reproduction)

All on an ordinary 12-core Linux box (89 GB RAM, ≈1 TB NVMe). Total budget:
≈7 hours (5.5 h download + 1.3 h conversion).

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
# greedy allocator: 98.3 GB budget, floor 6x64, 6 tail layers pinned to 6x64:
venv313/bin/python tests/qwen38_flash_next_iq_allocate.py allocate \
    --errors ~/iq/errors.json --budget-gb 98.3 --floor 6x64 \
    --tail-layers 6 --tail-width 6x64 --out ~/iq/alloc.json
```

Result: **all 96 groups at 6x64** (≈98.3 GB spent). The error analysis
justifies the 6x64 floor: letting early-layer groups drop to 5-bit
concentrates ≈41% of the total expert quantization error mass in those
groups, while the flat per-layer error curves mean point 8-bit upgrades
never beat a floor raise at this budget.

### 4. Conversion (≈75 min, 12 CPU)

```bash
# ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw = this repo, downloaded as-is
# (its ngram_table.bin is reused via hard link); rebuild into a fresh directory:
venv313/bin/python tests/convert_qwen38_flash_next.py \
    --src ~/models/Qwen3.8-Flash-Next \
    --imatrix ~/iq/imatrix.safetensors --alloc ~/iq/alloc.json \
    --bits 6 --tail-layers 6 --nonexpert-bits 8 --ngram-bits 8 \
    --ngram-link ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw/ngram_table.bin \
    --jobs 10 \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw-rebuild
```

- `--bits 6` — pin for tail layers and MTP; experts follow alloc.json (all 6x64);
- `--nonexpert-bits 8` — all 2-D weights (rows≥32, cols%64==0) with imatrix
  (embed_tokens stays 4-bit via the separate `--embed-bits` default);
  imatrix activation-energy analysis: 91% of the non-expert error mass sits in the
  hyper-connection gates, which is why the whole non-expert tier is 8-bit
  (only ≈0.9 GB);
- `--ngram-link` — hard-link the existing 8-bit table from another pack instead of
  re-quantizing 57.6 GB (drop the flag to re-quantize from the bf16 source);
- 8-bit n-gram table → `ngram_table.bin` 57.6 GB + `config.json:
  quantization.calibration = "imatrix-weighted affine (iQ-MLX)"`;
- resumable: progress in `.convert_state.json`, kill/restart is safe.

### 5. Vision pass (no re-stream)

```bash
venv313/bin/python tests/convert_qwen38_flash_next.py --add-vision \
    --src ~/models/Qwen3.8-Flash-Next \
    --dst ~/models/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-6.6bpw
```

→ `model-vision.safetensors` (0.9 GB), `config.json: language_model_only: false`.

## Pipeline artifacts (for reproduction)

| File | What |
|---|---|
| `imatrix.safetensors` (328 MB) | shim output, imatrix in HF key names |
| `errors.json` | error curves, 96 groups × 8 widths |
| `alloc.json` | per-group widths (all 6x64) |
| measure/convert/vision logs | pipeline trace |
| bf16 checkpoint (360 GB) | can be deleted after the pack is shipped |
