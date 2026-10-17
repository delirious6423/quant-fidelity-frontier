---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - apple-silicon
  - qwen3
  - moe
  - mixed-precision
  - quantized
  - multimodal
language:
  - en
---

# Qwen3.8-Flash-Next-mixed-3_8bit (MLX)

**Format: MLX — not GGUF.** This is a mixed-precision MLX quantization of
Qwen3.8-Flash-Next — an experimental preview of Qwen4's architecture: a
125B-total / 6B-active MoE plus 51B of hashed n-gram (PLE) embeddings, with a
built-in vision encoder (image-text-to-text). This build runs the full model on
a single 128 GB Apple-Silicon Mac with long-context (100k+) headroom, and is
85 GB (4.094 bpw). Run it with `mlx_vlm` (the multimodal server) or LM Studio's
MLX runtime — the `Q*_K_*` GGUF naming does not apply here (see "Naming" below).

## TL;DR

- **What:** one custom MLX mixed 3-bit/8-bit quant of Qwen3.8-Flash-Next.
- **Why mixed, not uniform 3-bit:** a uniform 3-bit build of this model
  degenerated into token repetition from the first turn under greedy decoding.
  Keeping the routing path and attention indexer at full precision and spending
  bits where they matter fixes that while staying inside a 128 GB budget.
- **Who it's for:** Apple-Silicon users who want an "opus-tier" local model with
  room for very long contexts, not a laptop-sized model.

## Naming

MLX quants are named by bit-width, not by the GGUF k-quant scheme:

- `mixed-3_8bit` follows the MLX `mixed_<low>_<high>bit` convention: the
  low-precision tier is **3-bit**, the high-precision tier is **8-bit**.
- The trailing `8bit` is the **high**-precision tier (not a uniform Q8 build).
- Group sizes and per-tensor assignments are intentionally left out of the repo
  name (as is standard for MLX repos) and documented in the recipe below.

## Quantization recipe

Mixed-precision allocation, following pipenetwork's published mixed-precision
layout for this model but dropping the routed experts one further bit:

| Component | Precision |
|---|---|
| Routers / MoE gates | unquantized (full precision) |
| QSA attention indexer | unquantized (full precision) |
| PLE n-gram table | 4-bit, group 32 |
| Routed experts | 3-bit, group 64 |
| Everything else | 8-bit, group 64 |

The router path and the attention indexer are the parts a uniform low-bit quant
damages first; keeping them exact is what removes the repetition failure of the
uniform 3-bit build. The experts — the bulk of the weights — carry the
aggressive 3-bit tier so the whole model fits with long-context headroom.

## Benchmark — IFEval (instruction-following adherence)

IFEval measures **instruction-following adherence only** (does the model obey
explicit format/content constraints such as "answer in exactly 3 sentences",
"do not use the word X"). It is *not* a general-capability score — no knowledge,
reasoning, or coding is measured. Scoring is programmatic (no judge model), so
the numbers are reproducible.

**Harness:** lm-evaluation-harness 0.4.13, `local-chat-completions`, full 541
prompts, 0-shot, greedy (temperature 0).
**Thinking-off** rows use `max_gen_toks=1280`. **Thinking-on** rows use
`max_gen_toks=8192` so the reasoning trace cannot truncate the final answer; the
runtime routes the trace into a separate `reasoning_content` field, leaving the
scored `content` clean (0 empty answers, 0 `<think>` leakage in 541 samples).

Compared against `unsloth/…-UD-Q3_K_XL` (a public GGUF quant of the same base
model) as an external reference point, run through the same harness and config:

| Build | thinking | prompt_strict | prompt_loose | inst_strict | inst_loose |
|---|---|---|---|---|---|
| **nirecom/…-mixed-3_8bit** (MLX) | off | 0.8447 | 0.8743 | 0.8969 | 0.9185 |
| unsloth/…-UD-Q3_K_XL (GGUF)     | off | 0.8521 | 0.8872 | 0.9017 | 0.9269 |
| **nirecom/…-mixed-3_8bit** (MLX) | on  | 0.9279 | 0.9464 | 0.9484 | 0.9628 |
| unsloth/…-UD-Q3_K_XL (GGUF)     | on  | 0.9261 | 0.9427 | 0.9472 | 0.9604 |

**Reading the numbers:**

- **Thinking-off**, the two quants are within ~1 point on every metric — the
  quantization choice barely moves adherence (the GGUF is marginally ahead).
- **Turning thinking on** lifts this MLX build by **+4.4 to +8.3 points** on
  every metric (prompt_strict 0.8447 → 0.9279) — an order of magnitude larger
  than the quant-to-quant gap. For this model, **enabling thinking is the
  dominant lever for instruction-following**, far more than which quant you pick.
- **Bound the reasoning trace when thinking is on.** On the hardest constraint
  prompts, an unbounded reasoning trace can run to the token limit and leave the
  answer empty (which scores as a failure). A positive reasoning-token cap
  avoids this, and both runtimes support one — MLX `--thinking-budget N`,
  llama.cpp `--reasoning-budget N`. The MLX thinking-on row used
  `--thinking-budget 4096` (0 empty answers); the GGUF thinking-on row used
  `--reasoning-budget 4096` (2 empty answers out of 541, vs ~31 without a cap).

_Benchmarks were run on the plain build (no MTP / speculative-decoding drafter)._

## How to run

With `mlx_vlm` (OpenAI-compatible server):

```
mlx_vlm.server --host 127.0.0.1 --port 8080 \
  --model nirecom/Qwen3.8-Flash-Next-mixed-3_8bit
```

Thinking is off by default. To measure or use the thinking-on behavior above,
start the server with `--enable-thinking` (optionally `--thinking-budget 4096`
to bound the reasoning trace), or pass `enable_thinking: true` per request.

Also loadable in LM Studio via its MLX runtime.

## Hardware tested

Apple M5 Max, 128 GB unified memory. The build is sized so the full model plus a
long-context KV cache stays within the Metal working-set limit on a 128 GB
machine.

## Intended use & limitations

- **Intended:** local, single-Mac inference where long context and strong
  instruction-following matter more than raw parameter count.
- **Adherence:** competitive with a public Q3 GGUF of the same base at
  thinking-off, and markedly higher with thinking on than the same build with
  thinking off (see IFEval above).
- **Not measured here:** knowledge, reasoning, coding, multilingual, or safety
  behavior. IFEval covers instruction-following only.
- **Vision inherited, not evaluated:** the base is multimodal (image/video →
  text) and those weights are carried through this quant, but no vision
  benchmark was run here — the effect of the low-bit tiers on image
  understanding is unverified. All numbers above are text-only.
- **MoE / memory:** this is a 125B-total MoE (plus 51B PLE embeddings); despite
  6B active params it needs the full weights resident. Plan for ~128 GB-class
  hardware.

## Credits

- Base model: **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**
  (Qwen team) — an experimental preview of the Qwen4 architecture, under the
  Qwen Community License 1.0.
- Mixed-precision layout adapted from **pipenetwork**'s published recipe for this
  model (routed experts dropped one further bit here).
- **unsloth** UD-Q3_K_XL used as the external GGUF reference in the IFEval table.
- Quantization and evaluation: **nirecom**.
