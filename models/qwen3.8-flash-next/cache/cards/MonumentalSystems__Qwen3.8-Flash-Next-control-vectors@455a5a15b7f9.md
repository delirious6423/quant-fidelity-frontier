---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - control-vector
  - activation-steering
  - representation-engineering
  - gguf
---

# Control vectors for Qwen3.8-Flash-Next

Behavioural control vectors derived from **Qwen3.8-Flash-Next** activations,
in llama.cpp-compatible GGUF.

A control vector is a *contrastive mean difference* — no gradients, no
optimiser, no fine-tune. For each layer you take the mean activation over a
positive corpus, subtract the mean over a matched negative corpus, and
normalise. Applying it is a few FLOPs per token on the residual stream, and no
weight is ever read or written — so the model's quantisation (NVFP4, GGUF,
bf16) is irrelevant to the arithmetic.

## ⚠️ Read this first: two files per vector, and they are not interchangeable

Each vector ships **two GGUFs of the same direction** in different magnitude
conventions. Using the wrong one is the single most likely way to conclude
these vectors don't work.

| file | rows | use with | why |
|---|---|---|---|
| `vector.gguf` | unit-normalised | **`project`** | `h -= s·(h·v)v` uses direction only; loaders fold the row norm into the per-layer scale, so magnitude cancels |
| `vector-add.gguf` | raw `\|mean-diff\|` | **`add`** | `h += s·row` applies the row *verbatim* under **one global scalar**, so it needs the per-layer magnitude |

`|mean-diff|` varies ~11–29× across layers. A unit file under `add` therefore
forces one scalar to dose every layer at once — wrong at nearly all of them.
Use `vector.gguf` with `add` at scale 1.0 and you will get `useruseruser…`.

Each file states which operator it is for in its metadata:

```
controlvector.magnitude = unit   # vector.gguf      -> project
controlvector.magnitude = raw    # vector-add.gguf  -> add
```

Atlas refuses the wrong pairing at load rather than serving a meaningless dose,
and detects it by *measuring* the rows as well as by reading the key, so files
predating this convention are caught too.

## Choosing the operator

**This follows from the behaviour, not from taste.**

`project` (`h -= s·(h·v)v`) is **quadratic in `v`** — negating the file changes
nothing, because both `h·v` and `v` flip and their product does not. It
*ablates an axis*; it cannot travel along one.

`add` (`h += s·v`) has a sign and *traverses*.

So use `project` for a feature you want **deleted** (refusal — there is no
useful "more refusal") and `add` for an axis you want to **move along in both
directions** (verbosity, comment density).

Measured on the verbosity vector, same direction and layers, one serve:

| operator | scales | result |
|---|---|---|
| `add` | −0.10 … +0.05 | **monotone through zero, −90% → +225%**, 6/6 sign agreement per arm |
| `project` | 0.5 / 1.0 / 2.0 | +49.5% / +25.8% / +40.2% — all *longer*, never ordered by dose |

`project` never shortened at **any** scale, including `s = 2`, which reflects
the component and in theory should. What it measures instead is that
**perturbation lengthens output** — an unrelated refusal vector moved length
+8.6% the same way.

## Scales

The two operators do **not** share a scale range.

| operator | usable range (Qwen3.8-Flash-Next, layers 4–44) |
|---|---|
| `project` | 0.5 – 2.0; **4.0 destroys the model** |
| `add` | see each vector below; roughly ±0.02 – ±0.15 |

`add` needs a much smaller scale because the displacement is applied at **every**
active layer and compounds, while projection is self-limiting — it can only
remove what is present.

---

## `verbosity/` — output length

Steers how much the model elaborates. Derived from 200 line-by-line paired
prompts (same request, verbose vs terse framing), 8,013 / 5,701 tokens.

**Coherence:** adjacent-layer cosine **0.9794**, all-pairs off-diagonal 0.7429.
`|mean-diff|` 0.20–2.13 over layers 4–44, a steady 27–38% of the stream norm at
every depth.

**Measured** — median Δ `completion_tokens` vs unsteered, 6 prompts, temp 0,
paired per prompt:

| scale (`add`) | Δ tokens | notes |
|---|---:|---|
| −0.10 | −90.0% | **too far** — answers end mid-sentence *with `finish_reason: stop`* |
| **−0.05** | **−54.9%** | **best concise setting** — complete, dense, high quality |
| −0.02 | −20.8% | mild tightening |
| +0.02 | +63.5% | more thorough, still on topic |
| +0.05 | +225.4% | essay territory |
| +0.10 | ≥ +484% | saturates the token cap and begins script-mixing |

Strictly monotone through zero, 6/6 sign agreement in every arm, no truncation
in the usable band.

At −0.05 the model produced a *better* answer than baseline in 53% of the
tokens: iterative solution with commented code, complexity, a recursive
alternative, a worked example. Genuine compression, not omission.

**Note the asymmetry.** Lengthening is unbounded; shortening has a floor at
zero tokens. So the negative side saturates faster — −0.10 starts returning
empty answers while +0.25 is still producing coherent prose.

---

## `comment-density/` — comments in generated code

Steers how heavily generated code is commented. Derived from 200 generated
paired prompts (same task, "comment every line" vs "no comments"), 6,518 /
6,166 tokens, balanced across Python, Rust, bash, CUDA, JavaScript, Go, SQL, C,
Java and TypeScript.

**Coherence:** adjacent-layer cosine **0.9667**, all-pairs off-diagonal 0.6937.
`|mean-diff|` 0.042–1.218 over layers 4–44 — a weaker contrast than verbosity
(13% of the stream norm vs 33%), so it wants a correspondingly larger scale.

**Why this one exists.** Verbosity is scored on `completion_tokens`, and almost
any sufficiently strong steering lengthens output — so a length metric is
partly forgeable. Comment density is scored as a **ratio** (comment lines ÷
total lines inside fenced blocks), which rambling cannot move: it adds to
numerator and denominator alike.

**Measured** — median comment ratio over 8 held-out coding tasks (none from the
derivation corpus), temp 0:

| scale (`add`) | comment ratio | code blocks produced |
|---|---:|---|
| −0.25 | — | **4/8 — stopped writing code**, disqualified |
| **−0.15** | **0.000** | 8/8 — clean uncommented code |
| −0.10 | 0.023 | 8/8 |
| −0.05 | 0.039 | 8/8 |
| *unsteered* | *0.051* | 8/8 |
| **+0.05** | **0.236** | 8/8 — well-documented code |
| +0.10 | 0.278 | 7/8 |
| +0.15 | 0.290 | 5/8 |
| +0.25 | 0.341 | 4/8 |

Monotone across every arm that still produced code. **Usable band: −0.15 to
+0.05**; past +0.05 the model increasingly omits code blocks entirely.

Quality holds at both ends. At −0.15 a CUDA kernel came back in 74 tokens,
correct, with the bounds guard intact and nothing missing but the prose. At
+0.05 the same kernel came back with Doxygen parameter docs and substantive
inline comments ("guard against out-of-bounds access when n is not a multiple
of block size") — not filler.

**Two honest caveats.**

*This direction is correlated with verbosity, not orthogonal to it.* At +0.05
total code lines also grow (one task went 28 → 84 lines) and the model expands
scope — it added a whole host-side wrapper no other arm produced. The ratio
still rises by more than the line count does, so the comment effect is real,
but do not expect "same code, more comments."

*The metric is not perfectly forgery-proof either.* The unrelated refusal
vector moved the ratio 0.051 → 0.072. That is the perturbation floor. The
comment-density vector reaches 0.185 above baseline — **8.8× the floor** — so
the signal is well clear of it, but the floor is not zero.

**Projection does not work for this vector**, and this is the cleanest evidence
in the repo for the operator rule above, because a ratio metric is immune to
the perturbation-lengthens artifact that muddies length measurements:

| arm | ratio |
|---|---:|
| `project` 0.5 | 0.074 |
| `project` 1.0 | 0.074 |
| `project` 2.0 | 0.052 |
| `project` 4.0 | no code blocks — destroyed |
| **`refusal` `project` 1.0 (unrelated direction)** | **0.072** |
| unsteered | 0.051 |

Projecting this direction (0.074) is indistinguishable from projecting an
unrelated one (0.072), and scales 0.5 and 1.0 give *identical* results —
no dose response at all. Use `add`.

---

## Credit

Method and reference artifact:
**[Cudecnik/Qwen3.8-Flash-Next-refusal-projection](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection)**
— the refusal-projection control vector, which is also the unrelated-vector
control used in the measurements above. The refusal vector is **not**
redistributed here; get it from Cudecnik's repo.

Vectors in this repo were derived natively from Atlas activations at
TP=2 × EP=2 on 2× NVIDIA GB10, NVFP4 weights.

## Measurement notes, if you plan to evaluate these

Things that produced a confidently wrong number during this work:

- **A fully-truncated arm has no measurable length.** When every response hits
  `max_tokens`, the delta becomes a function of the *baseline alone*, so every
  saturating arm reports the same median. Three did, to one decimal place.
- **`finish_reason: stop` does not mean the answer finished.** Over-driven
  brevity emits EOS mid-sentence and reports clean.
- **Degeneracy has several shapes.** Repetition (low unique-word ratio),
  multilingual token salad (*high* unique-word ratio — sails past a repetition
  test), and whitespace-free repetition (`useruseruser…` splits into one
  enormous "word", so any check gated on a word count silently skips).
- **Always run an unrelated-vector control** at the same operator, scale and
  layers. It measures the perturbation floor for that configuration.
- **Pair per prompt.** Between-prompt spread here is ~110% of the median, which
  swamps the arm difference entirely.
