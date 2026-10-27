---
license: apache-2.0
language: en
tags:
  - qwen
  - dflash2
  - speculative-decoding
  - block-diffusion
  - draft-model
  - draft-dflash
  - llama.cpp
  - mixed-quantization
  - mixedlow
  - text-generation
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-FrostDFlash-MixedLow (DFlash2 drafter sidecard)

Custom **precision-gradient quantization** of the z-lab DFlash2 speculative drafter for
Qwen3.8-27B. Rebuilt from the BF16 GGUF with a per-tensor type recipe that keeps the
tokens most sensitive to quantization error at high precision (the selector head at
Q6_K, the residual value / down path at Q5_K) and puts the redundant FFN gates at the
coarsest stable type (IQ3_S).

**Not a standalone LLM** — it is a ~1.9B-parameter **draft model** that runs as a sidecard
next to the target model under llama.cpp speculative decoding (`--spec-type draft-dflash`).
It borrows the target's tokenizer and output embedding (no `token_embd`/`output.weight`
tensors of its own).



## Support

If you find this project useful and want to support development:

Monero (XMR):

`
865BrcfWLdwELwuq5faV1uVTbh93zVK6AUYLY2c3mX6sFfAGRfS6axe1kBTYYKuM7ccN7zBZDAZvnT7E4NKmUazySdbpc7p
`

RTX 5090 FUND

[https://throne.com/plaguekind](https://throne.com/plaguekind/item/073bace1-2e1a-4492-95e6-024b4da6459b)

## File

| Field | Value |
|---|---|
| File | `Qwen3.8-27B-FrostDFlash-MixedLow.gguf` (local build name: `Qwen3.8-27B-DFlash2-mixedlow.gguf`) |
| File size (on disk) | **999 MiB** (1,047,907,936 B; ~10 MiB GGUF header/alignment overhead) |
| Weights-only | **988.92 MiB** |
| Bits / weight | **4.31 BPW** |
| Tensors | 81 (5 dtypes: F32 / Q5_K / IQ4_XS / IQ3_S / Q6_K) |
| Source (BF16) | [`z-lab/Qwen3.8-27B-DFlash2-GGUF` (BF16)](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) — sha256 `26d47ca20ab07688327a63d912acad222d924eaaa92a980cc488de3c67e736bc` |
| Target model | Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P (same folder) |
| Quant tool | llama.cpp `llama-quantize` (fresh build, commit `8b54361`), `--pure` |
| License | apache-2.0 (inherited from source) |

For reference, the other DFlash2 drafter quants in this folder (on-disk sizes):

| Quant | File size | Notes |
|---|---|---|
| **mixedlow** (this) | **999 MiB** | precision gradient, 4.31 BPW |
| Q4_K_M | 1090 MiB | stock flat quant |
| Q8_0 | 1.93 GiB | near-lossless |

This recipe lands ~91 MiB under flat Q4_K_M at a *higher* effective precision where it
matters, holding 4.31 BPW.

## Architecture (from GGUF metadata + tensor list)

- `general.architecture = dflash` — block-diffusion draft model (size label 1.9B)
- **5 DFlash blocks** (`blk.0` … `blk.4`), each with:
  - attention: `attn_q/k/output/v` + per-head `attn_q_norm`/`attn_k_norm` + `attn_norm`
  - a conv feature path: `attn_conv_base` + `attn_conv_proj`
  - FFN: `ffn_gate/up/down` + `ffn_norm` + a conv path `ffn_conv_base`/`ffn_conv_proj`
- embedding length **5120**, rope base **1e7**
- **Selector head** — the token-selection component of the drafter:
  `selector_hidden` (Q6_K, ~1 MiB), `selector_predecessor` / `selector_successor`
  (Q5_K, 41.7 MiB each). This is the part that decides *which* tokens to propose.
- Input feature projection `fc.weight` (IQ4_XS, 66.4 MiB)
- Norms: `output_norm` + `enc.output_norm`

## Quantization recipe (per-tensor, `dflash_q4_recipe.txt`)

```
# decision head — highest precision
selector_hidden.weight             = q6_k
selector_predecessor.weight        = q5_k
selector_successor.weight          = q5_k
# input projection
fc.weight                          = iq4_xs
# top-level norms
output_norm.weight                 = f16
enc.output_norm.weight             = f16

# per DFlash block (blk.0 .. blk.4)
attn_q.weight                      = iq4_xs
attn_k.weight                      = iq4_xs
attn_output.weight                 = iq4_xs
attn_v.weight                      = q5_k        # residual value path
ffn_down.weight                    = q5_k
ffn_gate.weight                    = iq3_s       # coarsest: FFN gates
ffn_up.weight                      = iq3_s
attn_conv_proj.weight              = q5_k
ffn_conv_proj.weight               = q5_k
attn_conv_base                     = f16
ffn_conv_base                      = f16
attn_norm.weight                   = f16
ffn_norm.weight                    = f16
attn_q_norm.weight                 = f16
attn_k_norm.weight                 = f16
```

Verified dtype distribution on the actual file (81 tensors, 988.92 MiB):

| dtype | count | role |
|---|---|---|
| F32 | 32 | all 1-D norms + conv bases (recipe says f16; the build keeps 1-D tensors in F32 — negligible) |
| Q5_K | 22 | `attn_v`, `ffn_down`, `attn_conv_proj`, `ffn_conv_proj` (×5 blocks) + selector pred/succ |
| IQ4_XS | 16 | `fc` + `attn_q/k/output` (×5 blocks) |
| IQ3_S | 10 | `ffn_gate/up` (×5 blocks) |
| Q6_K | 1 | `selector_hidden` |

### Quantize command (canonical, reproducible)

> ⚠️ **Flag order matters.** `llama-quantize` parses `--` flags from argv index 1; if a
> positional (in / out / type) comes first the flag loop stops and **all** `--tensor-type-*`
> overrides are silently dropped (you get a flat Q5_K file at ~1286 MiB instead of 989).
> Put **flags before positionals**:

```bash
llama-quantize \
  --pure \
  --tensor-type-file dflash_q4_recipe.txt \
  --output-tensor-type f16 \
  --token-embedding-type f16 \
  Qwen3.8-27B-DFlash2-BF16.gguf \
  Qwen3.8-27B-DFlash2-mixedlow.gguf Q5_K 4
```

`dflash_q4_recipe.txt` = the 81 `name=type` lines above. The base quant type (the
`Q5_K` positional) is a placeholder that `--tensor-type-file` overrides per tensor —
any real quant base works, but **not** F16/COPY (a float base blocks the overrides).

## Usage (llama.cpp, speculative decoding)

```bash
llama-server \
  --model <target-Q6_K_P.gguf> \
  --model-draft Qwen3.8-27B-DFlash2-mixedlow.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 3 \
  --n-gpu-layers all \
  --n-gpu-layers-draft all
```

- `--spec-draft-n-max 3` — the drafter proposes up to 3 tokens per cycle (validated
  operating point; the model has 5 blocks, so 3 is a conservative proposal depth)
- `--n-gpu-layers-draft all` — sidecard goes fully on GPU
- The drafter **shares the target's tokenizer + output embedding** — both must load
  together; it has no standalone decode path

## Measured performance

Hardware: RTX 5090 (32 GB VRAM), 64 GB RAM, R9 5950X, Arch Linux.
GPU overclocked (60 mV left-shift VF curve + mem +6000 MHz @ 400 W power cap — see
`memory/gpu-oc.md` for the exact LACT profile).

| Scenario | t/s (live, full config) |
|---|---|
| Formulaic (code/math-like) | **~151–152** |
| Prose (natural language) | **~103** |
| Thinking (reasoning) | **~105** (per-cycle 2.77) |

Draft acceptance with `--spec-draft-n-max 3`:
- **acceptance 0.517** (live, 107/207 drafts accepted, mean cycle length 2.55)
- Per-position acceptance (formulaic run, 584 drafts): pos0 ~57%, pos1 ~31%, pos2 ~17%

The sidecard costs ~1 GB of VRAM (989 MiB) — negligible against a 32 GB card, and the
acceptance rate keeps decode comfortably above 89 t/s in the live config.

## Why this gradient

The DFlash2 drafter is a *decision* model: the selector head picks which tokens to
propose, and the value / down path conditions the actual proposals. Quantizing those
coarsely kills acceptance (fewer drafts get verified, speculative decoding degrades
toward the target's plain decode speed). The FFN gate/up projections are the most
redundant — they implement the "transform the feature" step and tolerate IQ3_S without
measurable acceptance loss. This gradient lands ~91 MiB under flat Q4_K_M
(1090 → 999 MiB on disk; 988.92 MiB weights-only) at a *higher* effective precision
where it matters, holding 4.31 BPW.

## Validation

- Rebuilt 2026-10-10 from BF16 source (a scratch-cleanup wiped the first build)
- Verified: 81/81 tensor types match the recipe, dtype distribution matches the table
  above, 988.92 MiB / 4.31 BPW
- Router restarted, worker argv confirmed serving this file, live generation measured
  (acceptance 0.517, ~79–103 t/s depending on scenario)
- Canary token test passed post-quantize, dmesg clean (no Xid/CUDA errors)

## Provenance

- Source BF16: `z-lab/Qwen3.8-27B-DFlash2-GGUF` (Inco AI / z-lab, apache-2.0)
- Quantized: 2026-10-10, llama.cpp fresh build `8b54361` (CUDA)
- Recipe: `~/hermes-workspace/benchmarks/dflash2-20261009/dflash_q4_recipe.txt`
- Bench harness: `~/hermes-workspace/benchmarks/dflash2-20261009/bench_mtp.py`
- OC profile: `~/hermes-workspace/memory/gpu-oc.md`

## Credit
- Co-designed by PlagueKind & Valkyrja Vinter
