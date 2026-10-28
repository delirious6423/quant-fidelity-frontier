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
  - d2t
  - llama.cpp
  - mixed-quantization
  - mixedlow
  - text-generation
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-FrostDFlash (DFlash2 drafter sidecards)

Custom **precision-gradient quantization** of the z-lab DFlash2 speculative drafter for
Qwen3.8-27B. This repo holds three drafter sidecards: the full-vocab **Mixed-Low** draft,
and two **D2T** variants that trim the draft's output vocabulary down to the 32K most
useful tokens. Every recipe keeps the tensors most sensitive to quantization error at high
precision (the selector head at Q6_K, the residual value / down path at Q5_K) and puts the
redundant FFN gates at the coarsest stable type (IQ3_S — Q5_K in the quality-tier variant).

**Not a standalone LLM** — each is a ~1.9B-parameter **draft model** that runs as a sidecard
next to the target model under llama.cpp speculative decoding (`--spec-type draft-dflash`).
The full-vocab draft borrows the target's tokenizer and output embedding (no `token_embd`/`output.weight`
tensors of its own); the D2T drafts carry their own reduced 32K head.



## Support

If you find this project useful and want to support development:

Monero (XMR):

`
865BrcfWLdwELwuq5faV1uVTbh93zVK6AUYLY2c3mX6sFfAGRfS6axe1kBTYYKuM7ccN7zBZDAZvnT7E4NKmUazySdbpc7p
`

RTX 5090 FUND

[https://throne.com/plaguekind](https://throne.com/plaguekind/item/073bace1-2e1a-4492-95e6-024b4da6459b)

## Files

| File | On disk | Weights-only | Bits/weight | Tensors | Draft vocab | FFN gate/up | attn / fc |
|---|---|---|---|---|---|---|---|
| `Qwen3.8-27B-FrostDFlash-Mixed-Low.gguf` | **999 MiB** (1,047,907,936 B) | 988.92 MiB | **4.31** | 81 | full (248,320) | IQ3_S | IQ4_XS |
| `Qwen3.8-27B-FrostDFlash-D2T-Mixed-Low.gguf` | **1.17 GiB** (1,226,428,096 B) | 1159.17 MiB | **4.65** | 83 | 32,768 (d2t) | IQ3_S | IQ4_XS |
| `Qwen3.8-27B-FrostDFlash-D2T-Q5K.gguf` | **1.40 GiB** (1,467,682,496 B) | 1389.25 MiB | **5.57** | 83 | 32,768 (d2t) | Q5_K | Q4_K |

All three are built from the z-lab DFlash2 drafter:

| Field | Value |
|---|---|
| Source (BF16) | [`z-lab/Qwen3.8-27B-DFlash2-GGUF` (BF16)](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) — sha256 `26d47ca20ab07688327a63d912acad222d924eaaa92a980cc488de3c67e736bc` |
| Mixed-Low | rebuilt from the BF16 source (2026-10-10) |
| D2T variants | built from the near-lossless Q8_0 quant of the same source (1.93 GiB, local build), plus the target brain's `output` head rows (32K subset) and the FastMTP-32K d2t mapping (2026-10-24) |
| Target model | any quant of Qwen3.8-27B-Uncensored-HauhauCS-Aggressive (same merge — the drafts share the target's tokenizer; the D2T reduced head comes from the Parasyte brain's `output`, see `Plaguekind/Qwen3.8-27B-Parasyte-GGUF`) |
| Quant tool | llama.cpp `llama-quantize` (fresh build, commit `8b54361`), `--pure` + `--tensor-type-file` |
| License | apache-2.0 (inherited from source) |

Reference sizes for the full-vocab draft: flat Q4_K_M ≈ 1090 MiB, near-lossless Q8_0 ≈
1.93 GiB (local build). Mixed-Low lands ~91 MiB under flat Q4_K_M at a *higher* effective
precision where it matters, holding 4.31 BPW.

## d2t — the draft-token trim (the D2T variants)

- The draft's **output vocabulary is trimmed to 32,768 tokens** — the 32K most-useful
  tokens (reused from the FastMTP-32K mapping). The draft can only propose tokens from that
  subset, which covers the vast majority of draft hits.
- Two tensors are added to the Mixed-Low draft:
  - `d2t` — I64, [32768]: the draft-vocab → target-token map (32,768 unique ids; verified
    order-equal *and* set-equal to the FastMTP-32K mapping)
  - `output.weight` — Q8_0, [5120, 32768]: the draft's own reduced head, rows taken from the
    target brain's `output` (verified, 0 mismatches)
- The **selectors stay full-vocab** (256 × 248,320) and the input side stays full-vocab.
- Why: the output head is the single biggest tensor in a full-vocab draft; trimming it
  shrinks the draft and activates the GPU-resident d2t path in the runtime.
- **Requires a patched build:** the D2T files need a llama.cpp build that carries the d2t
  logic (patched `qwen35`/`dflash` model code). Stock llama.cpp will not run them.

`-Q5K` is the quality tier of the D2T draft: FFN gate/up bumped IQ3_S → Q5_K (the Parasyte
bulk tier) and attn/fc bumped IQ4_XS → Q4_K — 26 tensors differ from `-D2T-Mixed-Low`,
everything else is identical.

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
- Input feature projection `fc.weight` (IQ4_XS / Q4_K depending on variant, 66.4 MiB)
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

The D2T variants add the `d2t` tensor + reduced head (see above); `-D2T-Q5K` additionally
bumps `ffn_gate`/`ffn_up` to `q5_k` and `attn_q/k/output` + `fc` to `q4_k`.

Verified dtype distributions, read from the actual files:

**Mixed-Low** (81 tensors, 988.92 MiB):

| dtype | count | role |
|---|---|---|
| F32 | 32 | all 1-D norms + conv bases (recipe says f16; the build keeps 1-D tensors in F32 — negligible) |
| Q5_K | 22 | `attn_v`, `ffn_down`, `attn_conv_proj`, `ffn_conv_proj` (×5 blocks) + selector pred/succ |
| IQ4_XS | 16 | `fc` + `attn_q/k/output` (×5 blocks) |
| IQ3_S | 10 | `ffn_gate/up` (×5 blocks) |
| Q6_K | 1 | `selector_hidden` |

**D2T-Mixed-Low** (83 tensors, 1159.17 MiB):

| dtype | count | role |
|---|---|---|
| F32 | 32 | all 1-D norms + conv bases |
| Q5_K | 22 | as Mixed-Low |
| IQ4_XS | 16 | as Mixed-Low |
| IQ3_S | 10 | as Mixed-Low |
| Q6_K | 1 | `selector_hidden` |
| Q8_0 | 1 | `output.weight` — the reduced 32K head |
| I64 | 1 | `d2t` — draft-vocab → target-token map |

**D2T-Q5K** (83 tensors, 1389.25 MiB):

| dtype | count | role |
|---|---|---|
| F32 | 32 | all 1-D norms + conv bases |
| Q5_K | 32 | the 22 above + `ffn_gate/up` (×5 blocks) |
| Q4_K | 16 | `fc` + `attn_q/k/output` (×5 blocks) |
| Q6_K | 1 | `selector_hidden` |
| Q8_0 | 1 | `output.weight` — the reduced 32K head |
| I64 | 1 | `d2t` — draft-vocab → target-token map |

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
  --model <target.gguf> \
  --model-draft Qwen3.8-27B-FrostDFlash-Mixed-Low.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 3 \
  --n-gpu-layers all \
  --n-gpu-layers-draft all
```

- `--spec-draft-n-max 3` — the drafter proposes up to 3 tokens per cycle (validated
  operating point; the model has 5 blocks, so 3 is a conservative proposal depth)
- `--n-gpu-layers-draft all` — sidecard goes fully on GPU
- The full-vocab draft **shares the target's tokenizer + output embedding** — both must
  load together; it has no standalone decode path
- The D2T variants use the same flags, plus a llama.cpp build carrying the d2t logic —
  they carry their own reduced head and the `d2t` map, so the draft no longer borrows the
  target's output embedding

Which file: **stock llama.cpp** → `Mixed-Low`; **patched (d2t) build** → `D2T-Mixed-Low`
(smallest) or `D2T-Q5K` (quality tier).

## Measured performance

Hardware: RTX 5090 (32 GB VRAM), 64 GB RAM, R9 5950X, Arch Linux.
GPU overclocked (60 mV left-shift VF curve + mem +6000 MHz @ 400 W power cap).

**Mixed-Low (live, full config):**

| Scenario | t/s (live, full config) |
|---|---|
| Formulaic (code/math-like) | **~151–152** |
| Prose (natural language) | **~103** |
| Thinking (reasoning) | **~105** (per-cycle 2.77) |

Draft acceptance with `--spec-draft-n-max 3`:
- **acceptance 0.517** (live, 107/207 drafts accepted, mean cycle length 2.55)
- Per-position acceptance (formulaic run, 584 drafts): pos0 ~57%, pos1 ~31%, pos2 ~17%

**D2T-Q5K (CPU A/B, 2026-10-25)** — the GPU was held by the live brain, so this is a CPU
run (same prompt, `-n 96`):

| Run | t/s |
|---|---|
| Target brain alone | 1.7 |
| Target + D2T-Q5K draft | **3.9** (~2.3×) |

Coherent output in both runs; the d2t path confirmed firing in the actual spec context
(`DFlash using d2t mapping (draft_vocab_size = 32768)`). GPU speedup not yet measured.

The sidecard costs ~1–1.5 GB of VRAM — negligible against a 32 GB card, and the
acceptance rate keeps decode comfortably above 89 t/s in the live config.

## Why this gradient

The DFlash2 drafter is a *decision* model: the selector head picks which tokens to
propose, and the value / down path conditions the actual proposals. Quantizing those
coarsely kills acceptance (fewer drafts get verified, speculative decoding degrades
toward the target's plain decode speed). The FFN gate/up projections are the most
redundant — they implement the "transform the feature" step and tolerate IQ3_S without
measurable acceptance loss. This gradient lands ~91 MiB under flat Q4_K_M
(1090 → 999 MiB on disk; 988.92 MiB weights-only) at a *higher* effective precision where
it matters, holding 4.31 BPW. The `-Q5K` tier spends the slack back on the gate/up bulk
(Q5_K) and the attention/fc path (Q4_K) for users who want headroom over the IQ3_S floor.

## Validation

**Mixed-Low:**
- Rebuilt 2026-10-10 from BF16 source (a scratch-cleanup wiped the first build)
- Verified: 81/81 tensor types match the recipe, dtype distribution matches the table
  above, 988.92 MiB / 4.31 BPW
- Router restarted, worker argv confirmed serving this file, live generation measured
  (acceptance 0.517, ~79–103 t/s depending on scenario)
- Canary token test passed post-quantize, dmesg clean (no Xid/CUDA errors)

**D2T variants (2026-10-24/25):**
- Verified structure: 83/83 tensors per the tables above; `d2t` I64 [32768] with 32,768
  unique ids, order-equal and set-equal to the FastMTP-32K mapping; reduced-head rows
  match the target brain's `output` with 0 mismatches; selectors full-vocab
- Verified functional (pure CPU, GPU hidden): both load clean (d2t tensor + reduced head
  accepted, no assert failure), generate coherent output, clean exit
- End-to-end spec verification: the ~2.3× CPU A/B above, d2t path confirmed firing

## Provenance

- Source BF16: `z-lab/Qwen3.8-27B-DFlash2-GGUF` (Inco AI / z-lab, apache-2.0)
- Mixed-Low quantized: 2026-10-10, llama.cpp fresh build `8b54361` (CUDA)
- D2T variants built: 2026-10-24, from the local Q8_0 DFlash source (near-lossless quant
  of the same z-lab source) + the target brain's `output` head rows + the FastMTP-32K d2t
  mapping; end-to-end spec verification 2026-10-25
- Recipe: `~/hermes-workspace/benchmarks/dflash2-20261009/dflash_q4_recipe.txt`
- Bench harness: `~/hermes-workspace/benchmarks/dflash2-20261009/bench_mtp.py`

## Credit
- Co-designed by PlagueKind & Valkyrja Vinter
