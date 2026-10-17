---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - mlx-community/Qwen3.8-27B-bf16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - mlx
  - quantized
  - qwen3.8
  - long-context
  - lean-mlx
library_name: mlx
---

# Qwen3.8-27B (LEAN, mixed 3/4/6-bit)

## What this is

A lean MLX build of Qwen3.8-27B for long-context work on Apple Silicon:
4.178 bits per weight, 13.10 GiB on disk, holding a 20-cell long-context
accuracy grid from 8K to 131K tokens against an 8-bit reference.

Unlike the other Yooz lean builds, whose leanness comes from a *uniform*
quantization policy with zero silent 8-bit upgrades, this one is deliberately
mixed: 3-bit on the MLP gate and up projections, 6-bit on the attention value
projections and the output head, 4-bit everywhere else.
The lean promise is the same, the policy is just a different shape:
the weights are exactly as small as the declared policy says, every exception
is listed below, and all of it is verifiable from `config.json` alone.

**The interesting part:** the stock uniform 4-bit MLX build of this model
(`mlx-community/Qwen3.8-27B-4bit`) is *larger* than this one and *fails*
long-context variable tracking, scoring 0.700 at 131K.
This build spends fewer average bits and scores 1.000.
The difference is not the bit budget, it is which tensors get which bits.

![Variable-tracking accuracy from 8K to 131K: the community 4-bit MLX build falls from 1.00 to 0.70 while this build holds 1.00, at fewer bits per weight](assets/placement.png)

## Why this build exists

Yooz needed a 27B-class model that could hold 131K tokens of real context on a
Mac, at a weight footprint that leaves room for the KV cache.
Three attempts failed before this one, and the failures are the reason the
recipe looks the way it does.

The obvious option, `mlx-community/Qwen3.8-27B-4bit`
(plain affine, group size 64; 14.98 GiB, 4.78 bits per weight), fails in a
specific and instructive way.
Retrieval is fine: needle-in-a-haystack is 1.000 everywhere.
But variable tracking, following a chain of variable assignments through a long
document, degrades monotonically with length:
1.000 at 8K, 0.900 at 32K, 0.850 at 65K, 0.700 at 131K.
The model can still find things; it loses the ability to carry state across them.

The next move, spending fewer bits more cleverly, made it worse.
A 3-bit/6-bit mixed build at 3.97 bits per weight dropped multi-round
coreference resolution (MRCR) at 32K to 0.905 and pinned variable tracking at 0.900.
Training the quantization scales and biases on top of it
(distillation-aware weight quantization, DWQ) moved that cell to 0.923,
against a 0.947 threshold. Still a fail.
Rescaling cannot recover information destroyed when the codes were assigned.

What worked was copying a *placement*, not a bit budget.
Unsloth's dynamic GGUF build of the same model, UD-Q3_K_XL, passes the same grid.
Reading its per-tensor allocation out of the file shows it is not uniformly 3-bit at all:
it spends 3 bits only on the MLP gate and up projections,
holds attention value projections and the output head at 5 bits,
and leaves everything else at 4.
Transplanting that allocation onto MLX's own affine rungs, with no calibration data
and none of GGUF's K-quant machinery, recovers every cell.

For this architecture, **where** the bits go dominates **how many** there are
and how cleverly they were rounded.

## Intended use

This build exists for one job:
**reading a lot of text on a Mac, correctly, without sending it anywhere.**

Concretely, it is the right pick when:

- **Long-document question answering.**
  A 300-page contract, a thesis, a deposition transcript, a year of meeting notes,
  loaded whole rather than chunked and retrieved.
- **Whole-repository code understanding.**
  Tens of thousands of lines in the window at once,
  answering questions that need several files held simultaneously.
- **Verbatim recall and citation from long inputs.**
  MRCR stays at 0.9937 to 0.9999 across the whole range,
  so it reproduces requested passages rather than paraphrasing them from memory.
- **Agent loops with large tool-output histories.**
  Accumulated search results, file dumps, and command output that push past the
  point where a 32K model has to start forgetting.
- **Privacy-bound work.**
  Legal, clinical, financial, or personal material that cannot go to a hosted API.

It is a good pick when the input is large and the output is small:
ask a long question, get a short, accurate answer.

## When to use something else

Stated plainly, because the numbers below say so:

- **Interactive chat where typing speed matters.**
  Decode runs 15.9 to 17.2 tokens/s at 8K context and falls to 4.9 to 11.5 tokens/s
  at 131K on an M4 Pro. Long answers at long context feel slow.
  This is a 27B dense model on a laptop; no quantization fixes that.
- **Machines under 32 GB.** The full 131K context peaks at 21.50 GiB, so
  32 GB is the floor; 16 GB will not hold this model at any useful context.
- **Short-context general chat, reasoning, or coding assistance.**
  We did not measure this build on knowledge, math, or code-generation benchmarks.
  A smaller model will be faster and probably just as good at those.
- **Vision.** The base model is multimodal; this build was converted and evaluated as text-only.
- **Languages other than English.** The evaluation grid is English-only.

## Provenance

- **Source checkpoint:** `mlx-community/Qwen3.8-27B-bf16`
- **Base model:** `Qwen/Qwen3.8-27B` (Apache 2.0), a 27B dense hybrid:
  64 layers, 48 GatedDeltaNet linear-attention plus 16 full-attention layers,
  grouped-query attention with 4 KV heads, head dimension 256,
  native context 262,144
- **Conversion tool:** `mlx-lm` 0.31.3, affine quantization, group size 64
- **Placement reference:** `unsloth/Qwen3.8-27B-GGUF`, file
  `Qwen3.8-27B-UD-Q3_K_XL.gguf`, revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`.
  No weights were copied from it; only the per-tensor bit-width assignment was
  read and mirrored.
- **No calibration set, no importance matrix, no training.** Deterministic
  round-to-nearest, reproducible from the policy table below.

## Quantization policy

| tensor role | bits | group size | count |
|---|---|---|---|
| `mlp.gate_proj`, `mlp.up_proj` | 3 | 64 | 128 |
| `self_attn.v_proj`, `lm_head` | 6 | 64 | 17 |
| everything else | 4 | 64 | 353 |

Two deviations from the placement reference, both forced by the target format:
6 bits stands in for the reference's 5, because MLX affine has no 5-bit rung;
the embedding table sits at 4 rather than the reference's 3,
as the nearest safe rung above it.

Effective 4.178 bits per weight over all quantized tensors.
The `quantization` block in `config.json` carries exactly the three groups above
and no others.

## Size

![Download size and decode speed against the community 4-bit MLX build: 13.10 vs 14.98 GiB, and faster decode at both 32K and 131K](assets/efficiency.png)

| Build | Lane | bits/weight | Size (GiB) |
|---|---|---|---|
| **This build** | MLX | 4.18 | **13.10** |
| `mlx-community/Qwen3.8-27B-4bit` | MLX | 4.78 | 14.98 |
| Mixed 3/6 (our failed attempt, unpublished) | MLX | 3.97 | 12.44 |
| `unsloth` UD-Q3_K_XL | GGUF | 3.99 | 12.52 |
| `unsloth` Q4_K_M | GGUF | 5.08 | 15.93 |
| `unsloth` Q8_0 (reference) | GGUF | 8.63 | 27.05 |

Bits per weight is derived uniformly from on-disk bytes over 26.94B parameters,
so the column is comparable across rows.
The GGUF figures are marginally inflated because those files retain a
multi-token-prediction block that the MLX source has stripped.

## Runtime memory and speed (measured, not estimated)

![Memory at every context bin on both lanes, all of it under the 32 GB ceiling, and the peak cost of the default prefill chunk](assets/memory.png)

Apple M4 Pro, 64 GB unified memory, macOS 26.6.1, mlx-lm 0.31.3, KV cache q8_0,
`prefill_step_size=512` (see the note below -- the default costs 10 GiB for
nothing). Memory comes from MLX's own allocator counters, not from the
process's resident size or footprint: `footprint(1)` counts MLX's pool of
freed-but-retained buffers, which at long context is larger than the model.

| Context | Decode (tokens/s) | Working set | Peak |
|---|---|---|---|
| 8K | 15.9 - 17.2 | 13.49 GiB | 14.74 GiB |
| 32K | 12.3 - 14.9 | 14.56 GiB | 16.72 GiB |
| 65K | 9.1 - 14.7 | 15.43 GiB | 17.86 GiB |
| 131K | 4.9 - 11.5 | 17.41 GiB | 21.50 GiB |

Working set is what stays resident while you talk to it (weights plus the KV
cache); peak includes the transient scratch of chunked prefill. Decode ranges
span the five evaluation tasks at each length.

**Both fit a 32 GB machine, including the full 131K context.** On 64 GB there
is room to spare.

**Set your prefill chunk to 512.** mlx-lm defaults to 2048, and the transient
scratch is proportional to chunk size times cache length, so the default costs
1.9 GiB at 8K rising to 10.4 GiB at 131K (peak 31.88 vs 21.50 GiB). It buys
nothing: prefill measured 72.44 tokens/s at 512 against 71.36 at 2048 on the
same prompt, a difference smaller than the run-to-run spread. With `mlx_lm`:

```python
generate(model, tokenizer, prompt=prompt, prefill_step_size=512)
```

Time to first token is dominated by prefill and is measured in minutes at long
context, not seconds: prefill runs at about 72 tokens/s, so a filled
129,514-token prompt takes **29 minutes** before the first output token.
Short context is unremarkable by comparison: 0.89 s to first token and
16.7 tokens/s at 4K.
If you send the same long prefix repeatedly, make sure your server reuses its
prompt cache; re-prefilling a 130K context every turn is the difference between
a usable setup and an unusable one.
This is a model for questions worth waiting for.

This build is also faster than the uniform 4-bit MLX build at every comparable
cell (MRCR at 32K: 14.9 vs 8.7 tokens/s; variable tracking at 65K: 9.1 vs 6.6),
which is what 1.9 GiB less resident weight buys.

## Validation

![The full twenty-cell grid, five tasks at four context lengths, this build against the community 4-bit MLX build](assets/accuracy.png)

RULER-style long-context tasks plus MRCR 2-needle,
at four context lengths, five tasks each, twenty cells total.
n = 10 / 10 / 5 / 5 samples per cell at 8K / 32K / 65K / 131K.
KV cache quantized to q8_0, chat-wrapped prompts, greedy decoding, fixed seeds,
deterministic graders.
Machine: Apple M4 Pro, 64 GB, macOS 26.6.1, mlx-lm 0.31.3.

The tasks:

- **NIAH single / multikey**: retrieve a planted fact from a long distractor haystack.
- **VT** (variable tracking): follow a chain of variable assignments;
  scored as the fraction of the chain recovered.
- **FWE** (frequent word extraction): aggregate over the whole context
  rather than retrieve from one spot.
- **MRCR 2-needle**: reproduce a specific earlier passage verbatim from a long
  multi-turn history.

### This build, full grid

| Task | 8K | 32K | 65K | 131K |
|---|---|---|---|---|
| NIAH single | 1.000 | 1.000 | 1.000 | 1.000 |
| NIAH multikey | 1.000 | 1.000 | 1.000 | 1.000 |
| VT | 1.000 | 0.975 | 1.000 | 1.000 |
| FWE | 1.000 | 1.000 | 1.000 | 1.000 |
| MRCR 2-needle | 0.9937 | 0.9968 | 0.9996 | 0.9999 |

Sixteen of the twenty cells match the placement reference to four decimal places.
The single sub-1.000 variable-tracking cell (39 of 40 chain links at 32K)
does not worsen with length, which is what distinguishes sampling noise from the
degradation signature that condemned the uniform 4-bit build.

### Against every other build of this model

Same twenty cells, same samples, same graders.
GGUF rows were served by llama.cpp, MLX rows by mlx-lm.

| Cell | **This build** | MLX 4-bit | Mixed 3/6 | UD-Q3_K_XL | Q4_K_M | Q8_0 |
|---|---|---|---|---|---|---|
| VT @ 8K | 1.000 | 1.000 | n/a | 1.000 | 1.000 | 1.000 |
| VT @ 32K | **0.975** | 0.900 | 0.900 | 1.000 | 1.000 | 1.000 |
| VT @ 65K | **1.000** | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 |
| VT @ 131K | **1.000** | 0.700 | 0.900 | 1.000 | 1.000 | 1.000 |
| MRCR @ 8K | 0.9937 | 0.9938 | n/a | 0.9938 | 0.9999 | 0.9999 |
| MRCR @ 32K | **0.9968** | 0.9998 | 0.9048 | 0.9968 | 0.9998 | 0.9998 |
| MRCR @ 65K | 0.9996 | 0.9996 | n/a | 0.9996 | 0.9996 | 0.9996 |
| MRCR @ 131K | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| FWE @ 8K | 1.000 | 1.000 | n/a | 0.967 | 1.000 | 0.967 |
| FWE @ 32K | 1.000 | 1.000 | n/a | 0.967 | 1.000 | 1.000 |
| NIAH, all cells | 1.000 | 1.000 | n/a | 1.000 | 1.000 | 1.000 |

Reading it honestly:

- Against **MLX 4-bit**, this build is smaller, faster, and strictly better at long context.
  It gives up 0.0001 at MRCR 8K and 0.003 at MRCR 32K, and gains 0.1, 0.15, and 0.3
  on variable tracking at 32K, 65K, and 131K.
  There is no reason to prefer the uniform 4-bit build over this one.
- Against our **mixed 3/6** attempt, it costs 0.66 GiB more and repairs the two cells
  that build fails.
- Against **UD-Q3_K_XL**, accuracy is a wash: identical to four decimals on sixteen cells,
  0.025 behind on one variable-tracking cell, 0.0001 behind on one MRCR cell,
  and 0.033 ahead on two aggregation cells.
  The GGUF build is 0.58 GiB smaller.
  This build's advantage is that it runs natively on MLX with no llama.cpp process.
- Against **Q4_K_M and Q8_0**, it gives up 0.006 at MRCR 8K and 0.003 at MRCR 32K,
  for 2.8 GiB and 14 GiB less on disk respectively. Everything else is equal.

The FWE cells where this build scores 1.000 and the GGUF 3-bit and 8-bit builds score
0.967 are, as far as we can tell, an artifact of the llama.cpp serving path rather than
a quantization effect: two independent MLX builds beat the GGUF lane's own 8-bit score
on the same samples.

## Lean class

Yooz calls a build **lean class** at its bit tier when it passes all four axes
the benchmark harness measures, and when the Apple Silicon and Linux builds of
that tier perform the same by measurement rather than assumption.
This build's certification:

| Axis | Result |
|---|---|
| Size on disk | 13.10 GiB, 1.88 GiB under the uniform 4-bit MLX build |
| Memory at context | 13.49 GiB working set at 8K to 17.41 GiB at 131K; peak 14.74 to 21.50 GiB. Fits a 32 GB machine at every context including 131K |
| Speed | 15.9 - 17.2 tokens/s at 8K, 4.9 - 11.5 at 131K; faster than the uniform 4-bit build at every comparable cell |
| Task performance across context bins | 20 of 20 cells pass against the reference |
| Cross-lane parity | Equal within noise to the GGUF lane build (`unsloth` UD-Q3_K_XL), which was itself verified bitwise-identical across CUDA and Metal |

Comparisons are within-lane; parity is verified by measurement, never assumed.

## What we did not measure

- General knowledge, reasoning, math, or code-generation benchmarks.
- Vision inputs (converted text-only from a multimodal base).
- Languages other than English.
- Contexts beyond 131K. The base model's native window is 262,144 tokens and
  extends further with YaRN; we have not evaluated this build there.
- Safety, refusal, or bias behavior beyond what the base model brings.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("YoozLabs/Qwen3.8-27B-lean-4bit-mlx")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Summarize the attached contract in three bullets."}],
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True))
```

For long-context work, quantize the KV cache; the numbers above are all at q8_0.
At 131K the KV cache, not the weights, is the larger half of the memory bill.

## About Yooz Labs

Maintained by [Yooz Labs](https://yooz.live), sovereign, on-device AI for macOS.
Everything we ship runs locally; nothing goes to a cloud.
This build comes out of our long-context research line, where weight footprint is
the budget that decides how much context fits on a real machine,
which is why the lean conversions exist.

Questions, issues, or corrections: dev@yooz.info

## License and attribution

Apache 2.0, inherited from the base model `Qwen/Qwen3.8-27B`.
This is an independent, unofficial conversion; it is not distributed, endorsed,
or certified by Alibaba or the Qwen team.
The per-tensor bit-width placement was read from Unsloth's UD-Q3_K_XL GGUF build
and is gratefully credited; no weights were copied from it.
