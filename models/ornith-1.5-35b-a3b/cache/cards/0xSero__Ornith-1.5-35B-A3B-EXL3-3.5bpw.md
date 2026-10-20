---
library_name: transformers
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- exl3
- exllamav3
- quantized
- moe
- tensor-parallel
pipeline_tag: text-generation
---

# Ornith-1.5-35B-A3B-EXL3-3.5bpw

Community EXL3 quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
Not an official Ornith release.

## Why this one is different

Most EXL3 quants of MoE models use the `-hq` switch, which gives attention and shared
experts a small bitrate bump (5 bpw). On this architecture that is the wrong place to
save bits: the model is 90% routed-expert parameters (256 experts, top-8 routing),
which is highly redundant and cheap to compress, while the attention backbone — in
particular the GatedDeltaNet (linear attention) projections that form the only
information pathway in 30 of 40 layers — is small but extremely sensitive.

Layer map used here (popularized by brandonmusic for GLM-5.2):

- **Routed expert matrices: 3.5 bpw**, MCG codebook
- **Everything else BF16**: GatedDeltaNet in/out projections, full-attention
  q/k/v/o, shared experts, embeddings, lm_head, norms, router
- MTP draft layer at 4 bpw, vision tower BF16

## Quantization

- exllamav3 1.4.2 `convert`, MCG codebook, stock calibration (250 rows x 2048 tokens)
- Exact per-module bitrates in `quantization_config.json`
- Artifact size: 20 GB

## Base-vs-quantization check

`exllamav3 eval/model_diff` vs the BF16 base, 100 rounds x 2048 tokens (wikitext):

| Checkpoint | KL(A->B) | Top-1 agreement |
|---|---|---|
| **Ornith-1.5-35B-A3B-EXL3-3.5bpw** | **0.0540** | 90.5% |
| EXL3 2.75 bpw | 0.1005 | 87.1% |
| EXL3 3 bpw | 0.0732 | 88.9% |
| EXL3 3.5 bpw | 0.0540 | 90.5% |
| reference: plain -hq 3 bpw | 0.2509 | 79.9% |

BF16 base perplexity on the same data: 8.717; all three unpruned rungs are within
noise of it (8.698 / 8.711 / 8.719).

## Runtime — tested recipe for 4x RTX 3090 (24 GB)

Served with TabbyAPI (exllamav3 backend), tensor-parallel across 4 GPUs:

```yaml
model:
  model_dir: /path/to/models
  model_name: Ornith-1.5-35B-A3B-EXL3-3bpw   # match your local dir name
  backend: exllamav3
  max_seq_len: 262144        # native context
  cache_mode: Q4
  max_batch_size: 16         # recurrent-state slots; keep >= expected concurrency
  tensor_parallel: true
  gpu_split_auto: true
```

Notes, all learned the hard way:

- **exllamav3 1.4.3 runtime patch required for this layer map.** The TP import for
  *unquantized* GatedDeltaNet projections has a copy-paste bug: in
  `exllamav3/modules/quant/fp16.py`, `tp_import_split_n`, change
  `id_w = exported["suh"]` to `id_w = exported["weight"]`. Without it, TP loading
  crashes with `KeyError: 'suh'`. Single-GPU loading is unaffected.
- `max_batch_size` must cover your concurrency: GatedDeltaNet layers keep per-sequence
  recurrent state; the default of 4 slots exhausts quickly under parallel requests.
- Keep agent-harness context budgets consistent with the server:
  `max_input_tokens + max_output_tokens <= max_seq_len` (with margin).
- Throughput at 4x 3090 TP4: ~50 tok/s single stream, ~180 tok/s aggregate at
  concurrency 4; weights occupy ~4.5-5.5 GB per GPU.
- vLLM with `--quantization exl3` also works; TabbyAPI is what we validated.

## Reproducing this quant (exact pinned recipe)

### Conversion environment

| Component | Version |
|---|---|
| exllamav3 (convert) | **1.4.2** |
| Python | 3.14.7 |
| torch | 2.11.0+cu128 |
| triton | 3.6.0 |
| CUDA toolkit | 13.3 (V13.3.73) |
| NVIDIA driver | 610.57.04 |
| Hardware | 4x RTX 3090 (TP4 conversion) |

### Conversion command

```
python convert.py \
  -i <bf16_model_dir> -o <out_dir> -w <work_dir> \
  -b 3.5 -hb 16 -mb 4 -cb mcg \
  -cr 250 -cc 2048 \
  -d 0,1,2,3
```

The layer map is forced with this patch, imported before `convert` runs
(monkey-patches `create_q_strategy` in `exllamav3.conversion.allocation` and the
reference imported by `exllamav3.conversion.convert_model`):

```python
# hq16_patch.py
import re as _re
from exllamav3.conversion import allocation as _alloc
from exllamav3.conversion import convert_model as _cm

_orig = _alloc.create_q_strategy
_PAT = (_re.compile(r"\.linear_attn\."), _re.compile(r"\.self_attn\."),
        _re.compile(r"\.shared_expert"))

def _patched(*args, **kwargs):
    f_targets, fb = _orig(*args, **kwargs)
    n = 0
    for k in list(f_targets.keys()):
        if any(p.search(k) for p in _PAT):
            f_targets[k] = 16
            n += 1
    print(f"[hq16] forced {n} modules to 16 bits")
    return f_targets, fb

_alloc.create_q_strategy = _patched
_cm.create_q_strategy = _patched
```

Expected confirmation in the conversion log: `[hq16] forced 257 modules to 16 bits`
(unpruned) — 90 linear-attention projections + 40 full-attention + 120 shared-expert
+ 7 MTP modules.

Conversion note: with triton 3.6.0, GatedDeltaNet kernels can trip a Triton
autotuner re-entrancy bug (nested autotuned kernels clobbering `nargs`); if the
conversion crashes inside `triton/runtime/autotuner.py`, capture
`nargs` into a local before the `benchmark()` closure.

### Runtime environment (tested)

| Component | Version |
|---|---|
| TabbyAPI | git `4a4f9f44820303593844f092d424bb7506008733` (2026-08-24) |
| exllamav3 (runtime) | **1.4.3+cu128.torch2.9.0** prebuilt wheel |
| Python | 3.12.14 |
| torch | 2.9.0+cu128 |

Required runtime patch for TP loading (exllamav3 1.4.3 bug, hit only when
GatedDeltaNet projections are stored unquantized):

```diff
--- exllamav3/modules/quant/fp16.py
-        id_w = exported["suh"]        # in tp_import_split_n
+        id_w = exported["weight"]
```

## Attribution

- **brandonmusic** — the GLM-5.2 EXL3 TR3 3bpw recipe (BF16 backbone, 3bpw routed
  experts) that this layer map follows.
- **turboderp** / turboderp-org — the EXL3 format and exllamav3, both conversion
  and runtime.
- **CerebrasResearch** — REAP expert pruning (pruned variants).
- **ornith-ai** — the base model. This quant inherits its capabilities, limitations,
  and MIT license; refer to the base model card before deployment.

If these quants are useful to you, consider supporting the work:
[donate.sybilsolutions.ai](https://donate.sybilsolutions.ai)
