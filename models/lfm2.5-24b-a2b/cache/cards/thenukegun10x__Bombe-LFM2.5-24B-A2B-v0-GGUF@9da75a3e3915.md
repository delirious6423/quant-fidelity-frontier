---
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
pipeline_tag: text-generation
language:
- en
tags:
- bombe
- sdht
- gguf
- ternary
- quantization
- distillation
- moe
- lfm2.5
---

# Bombe v0 — LFM2.5-24B-A2B

**SDHT ternary GGUF · 1.84 BPW · 5.52 GB · 24B MoE / 2.45B active**

Bombe v0 is a compressed derivative of
[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).
It combines fitted ternary quantization, smooth discrete Hartley transforms,
selective precision allocation and quantization-aware distillation to fit the
model's weights into a 5.52 GB file.

**Release tag:** `Bombe-LFM2.5-24B-A2B-v0-GGUF`

**Version:** `v0`, the first Bombe checkpoint, based on the final update-10 repair.

[Download the GGUF (5.52 GB)](https://huggingface.co/thenukegun10x/Bombe-LFM2.5-24B-A2B-v0-GGUF/resolve/main/Bombe-LFM2.5-24B-A2B-v0-SDHT-TQ1_0_128.gguf?download=true)
· [Release files and checksums](https://huggingface.co/thenukegun10x/Bombe-LFM2.5-24B-A2B-v0-GGUF/tree/main)

**Required runtime:**
[Thenukegun10x/llama.cpp-SDHT](https://github.com/Thenukegun10x/llama.cpp-SDHT).
The custom runtime implements `TQ1_0_128` and the SDHT transform metadata.
AMD HIP and NVIDIA CUDA backends have been tested; see the runtime's
[build instructions](https://github.com/Thenukegun10x/llama.cpp-SDHT#readme).

The full pinned benchmark suite is complete. This first version remains
experimental, with substantial room to improve instruction following, exact tool
calls and task accuracy. The measurements below describe this checkpoint;
the development history and repair experiments follow the usage instructions.

## Benchmarks

Full evaluation completed **21 October 2026**: **4,452 scored cases** across
**3,280 unique dataset examples**. ARC-Challenge appears in two evaluation modes.
All scores below use the complete pinned splits.

| Benchmark / metric | Bombe v0 | BF16 baseline | Difference |
|---|---:|---:|---:|
| IFEval — strict prompt | **83.55%** | 94.27% | -10.72 pp |
| IFEval — strict instruction | **88.49%** | 95.80% | -7.31 pp |
| IFEval — loose prompt | **86.69%** | 95.38% | -8.69 pp |
| IFEval — loose instruction | **90.53%** | 96.52% | -6.00 pp |
| IFBench — strict prompt | **49.33%** | 74.67% | -25.33 pp |
| IFBench — strict instruction | **53.78%** | 77.62% | -23.84 pp |
| IFBench — loose prompt | **54.00%** | 76.33% | -22.33 pp |
| IFBench — loose instruction | **58.43%** | 79.07% | -20.64 pp |
| ARC-Challenge — chat exact match | **56.57%** | 80.29% | -23.72 pp |
| ARC-Challenge — likelihood accuracy | **26.62%** | 44.62% | -18.00 pp |
| ARC-Challenge — normalized likelihood accuracy | **29.01%** | 48.63% | -19.62 pp |
| WinoGrande — likelihood accuracy | **52.64%** | 57.54% | -4.89 pp |

The BF16 column is our saved **LFM2.5-24B-A2B BF16 reproduction**, evaluated
with the same sample prompts and task configurations. These are measured local
results, rather than the original model author's published scores. The runs use
different hardware, operating systems, runtime builds and dependency versions;
the full gap cannot be attributed exclusively to quantization. Exact baseline
chat-template content identity is not established by the saved hashes.

### Evaluation protocol

| Task | Cases | Protocol |
|---|---:|---|
| IFEval | 541 | Greedy, 8,192 tokens including thinking; final-answer-only grading |
| IFBench | 300 | Greedy, 8,192 tokens including thinking; final-answer-only grading |
| ARC-Challenge chat | 1,172 | Forced answer prefix; 100-token generation limit |
| ARC-Challenge likelihood | 1,172 | Raw continuation likelihood, with raw and length-normalized accuracy |
| WinoGrande likelihood | 1,267 | Raw continuation likelihood |

Strict prompt accuracy requires **every instruction in a prompt** to pass.
Instruction accuracy grades each instruction individually. The likelihood tests
and forced-prefix ARC chat task should not be interpreted as unrestricted
reasoning benchmarks.

The harness revision is `d6de81643928d653435c431bae19945d41d32520`. All
4,452 sample prompt hashes and normalized dataset documents match the BF16 run.
Generation used four 16,384-token slots, with prompt caching and context
checkpoints disabled.

**Generation audit:** 12/541 IFEval and 25/300 IFBench responses reached the
8,192-token limit; 15 and 26, respectively, had empty final answers. All remain
included in the scores. BF16 reached the limit on 4 IFEval and 7 IFBench cases.

IFEval retains more of the BF16 score than ARC chat or IFBench. Dedicated
parallel-tool correctness remains outside this suite. Complete metrics, dataset
revisions, generation audits and source hashes are in
[BENCHMARKS.json](BENCHMARKS.json).

## The checkpoint

| Property | Value |
|---|---|
| Model name | **Bombe v0 — LFM2.5-24B-A2B** |
| Release tag | `Bombe-LFM2.5-24B-A2B-v0-GGUF` |
| Original checkpoint | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| Pinned original revision | `1769516ac588c4c558089c94f27d96b0dc67b183` |
| Architecture | Approximately 24B total parameters, 2.45B active per token |
| Blocks | 40: 30 short-convolution blocks and 10 grouped-query attention blocks |
| Expert routing | 64 experts per MoE block, top 4 selected per token |
| Release filename | `Bombe-LFM2.5-24B-A2B-v0-SDHT-TQ1_0_128.gguf` |
| Evaluated source file | `checkpoint-step-00000010.gguf` |
| File size | 5,515,420,736 bytes; approximately 5.52 GB / 5.14 GiB |
| Tensor payload | 5,507,138,048 bytes |
| Average payload bitrate | **1.8378884583 bits per original source parameter** |
| Dominant format | `TQ1_0_128`: ternary values, 128 weights per block |
| Tied embedding/output table | `Q4_K` |
| Inherited context limit | 131,072 tokens |
| Status | Experimental v0 checkpoint; full pinned benchmark suite completed 21 October 2026 |

SHA-256:

```text
b04c17880de37dd7d91da808a6ba57fc434b5a2a03ec105b8d5a7f32ab111f29
```

The file size describes stored weights and metadata. KV cache, activations and
runtime workspaces require additional memory. A 128K serving allocation was
verified; long-context answer quality has not been established by that check.

"Update 10" identifies the final ten-update repair run. Earlier PTQ experiments
and QAD stages are part of this checkpoint's history too.

## Running the model

Download the GGUF using the link above or the Hugging Face CLI:

```powershell
hf download thenukegun10x/Bombe-LFM2.5-24B-A2B-v0-GGUF Bombe-LFM2.5-24B-A2B-v0-SDHT-TQ1_0_128.gguf --local-dir .
```

Compile `llama-server` from
[llama.cpp-SDHT](https://github.com/Thenukegun10x/llama.cpp-SDHT), following its
[build instructions](https://github.com/Thenukegun10x/llama.cpp-SDHT#build-on-windows-with-amd-hip).
The AMD setup was tested on Windows with HIP and an RX 9070 XT (`gfx1201`).
The NVIDIA CUDA backend has also been tested by the maintainer and confirmed
to work well. Choose the build for your hardware.

The example uses the release filename. If you are running the original training
artifact, substitute `checkpoint-step-00000010.gguf`; it contains the evaluated
Bombe v0 weights.

From the custom build's `bin` directory, a single-slot local server can be
started with:

```powershell
.\llama-server.exe `
  -m "<path-to-model>\Bombe-LFM2.5-24B-A2B-v0-SDHT-TQ1_0_128.gguf" `
  -ngl 99 -c 131072 -np 1 `
  -b 4096 -ub 512 -t 4 -tb 4 `
  -fa on -ctk f16 -ctv f16 --fit off `
  --jinja --host 127.0.0.1 --port 12345
```

Select the appropriate HIP device for your system. On the development machine,
the native llama.cpp runtime uses `HIP_VISIBLE_DEVICES=1` for the RX 9070 XT;
that index should not be copied blindly to another machine or PyTorch runtime.

Use the embedded original chat template. The server can return thinking in
`reasoning_content` and the final answer in `content`; preserving that separation
matters for clients and evaluation. Calls still require validation before use.

## The journey

### 1. The first ternary models collapsed

The early R3/R4 models were extremely small, but failed basic arithmetic, copying
and ordinary prose. On a matched eight-chunk, context-512 WikiText diagnostic,
R4 reached **115,428.0986 perplexity**.

The first task was to distinguish a broken representation from a broken runtime.
CPU and GPU generation both failed. Sampled encodings regenerated byte for byte,
and synthetic quantization/transform checks passed. The evidence pointed toward
quantization quality rather than gross pack/decode corruption.

The largest early problem was scale selection. The old ternary encoder used the
largest absolute weight in a block as its scale. With only three reconstruction
levels, that pushed too many smaller weights to zero: the sampled median zero
fraction was approximately **83.4%**.

### 2. Fitting the ternary scales changed everything

R5 replaced maximum-based scale selection with an iterative, importance-weighted
fit. For fixed ternary codes, the scale update is:

```text
d = sum(h_i * w_i * q_i) / sum(h_i * q_i^2)
q_i in {-1, 0, +1}
```

Here `h_i` comes from calibration statistics. The encoder alternates codes and
scales, tries multiple starting points, and scores the scale after actual FP16
rounding. The original maximum-based result remains a candidate, preventing a
worse local fit from being accepted automatically.

On the same eight-chunk diagnostic, perplexity fell from **115,428.0986 to
184.8731**, with the **same tensor payload size**. Across 448 paired calibration
samples, median output reconstruction RMSE fell by about 45.2%. The sampled
median zero fraction fell to approximately 45.7%.

This was a pivotal result: a dramatically better model could occupy exactly the
same number of weight bytes.

### 3. Reallocate precision instead of raising every tensor's bitrate

The next gains came from treating different components differently.

The tied embedding/output table moved into a measured `Q4_K` recipe. Routers were
stored in BF16 when their F32 values could be restored exactly from BF16, saving
9.5 MiB. Most expert weights remained ternary, while sensitive attention,
convolution and dense projections received conventional K-quant formats.

A measured allocation reinvested **8.765625 MiB** of the router savings into
selected early dense-FFN and convolution projections. Full-file WikiText PPL
reached **90.1075**, below the original R5 payload ceiling.

Even source-exact router values did not guarantee identical native results:
changing storage type could select different arithmetic paths. Runtime controls
were needed alongside weight-byte checks.

### 4. Refine the scales without changing the ternary codes

Calibration-fitted row gains and bounded block-scale adjustments recovered more
quality. Projection families responded differently: gate/up and down projections
did not benefit equally from the same correction strength.

The retained combination used the strength-5 gate/up recipe with strength-3 down
scales. These labels describe extrapolations of small fitted corrections; they
do **not** mean multiplying the weights by 500 or 300. Stored block scales stayed
bounded to 0.8–1.25 times their original R5 values.

Selected full-file WikiText milestones were:

| Historical stage | Full-file PPL |
|---|---:|
| Fitted R5 | 183.2284 |
| Q4 embedding/body reallocation recipe | 120.4847 |
| Router-budget dense-FFN reallocation | 90.1075 |
| Uniform strength-5 scale recipe | 41.8071 |
| Strength-5 gate/up, strength-3 down | **41.3621** |

These use the recorded context-512, 575-complete-chunk protocol with 146,625
scored predictions. Payloads are close but not identical across all stages.
This is an adaptively selected development history, not an independent benchmark.
The earlier eight-chunk R4/R5 numbers belong to a separate, smaller diagnostic.

### 5. Recover behavior through native quantization fields

Lower perplexity helped, but repetition, thinking boundaries and tool calls
remained difficult. The next stage used **quantization-aware distillation (QAD)**
to adapt the quantized representation to an original-model teacher.

The learner updated stored floating-point quantization fields and existing input
diagonals while keeping integer/trit codes, tensor types and payload size fixed
within each recovery run. This trains the compressed representation directly;
it does not train a full BF16 copy of all 24B weights.

A separate precision repair re-encoded the final expert down-projection bank as
standard `Q2_K`, using original BF16 weights in the current transformed basis.
That cost **22,020,096 additional payload bytes** and improved targeted parallel
calls. It is included in the update-10 lineage. The average payload bitrate
became approximately **1.84 BPW**; a blanket move to 2 BPW was not required.

GGUF assigns one quantization type to the entire expert bank, so this precision
upgrade covers the final down bank's 64 experts. The later sparse-row experiment
keeps its existing tensor type to avoid requiring a new mixed-row runtime format.

### 6. Train on the situations where the student actually fails

Correct teacher prefixes can be too easy: low reference-token loss does not
ensure that the student reaches those prefixes during generation.

The recovery workflow therefore harvested failures from the student's own
training prompts. The BF16 teacher received the exact student prefix and
produced a corrected continuation. Only verified continuations became targets;
the conditioning prefix itself was unscored.

A reserve of 1,300 training seeds provided room for rejection. Collection examined
905 samples to harvest 302 target failures. The final correction selection had
240 examples: 60 math, 30 single-tool, 100 two-call and 50 three-call repairs,
including verified earlier repairs. Development and retention sources stayed
outside this harvest.

The final run combined these repairs with equal clean replay, preserving the
2:1 two-call/three-call ratio. It completed ten updates with accumulation 48:
**240 repair microbatches and 240 clean microbatches**. These are training
exposures, not a claim that every example in the larger corpus bank was consumed.

## Techniques behind the recipe

### Calibration-aware ternary encoding and LDLQ

The encoder uses activation second moments as a local Hessian approximation.
LDLQ error feedback compensates subsequent quantization decisions for earlier
reconstruction errors. The feedback consumes the reconstruction of the actual
stored codec, including rounded scales.

The initial calibration covered **1,179,648 tokens** in 2,304 sequences of length
512. All 7,296 expert-projection entries met the collector's routing minimum;
the minimum observed routed-row count was 217. Route-weighted pooled Hessians
and bounded per-expert activation reservoirs made the MoE calibration practical.
Coverage alone does not guarantee ideal statistics for every task.

### Smooth discrete Hartley transforms

The SDHT path combines a diagonal input scaling with a normalized, blockwise
discrete Hartley transform of width 128. Applying corresponding transforms to
activations and weights preserves the linear operation in exact arithmetic,
while changing the basis in which quantization occurs.

The exporter folds the reciprocal of the exact rounded input diagonal that the
runtime will read. It also transforms the calibration Hessian consistently.
Matching the weight basis, activation basis and calibration basis is essential.
This implementation uses a **Hartley** basis; it should not be described as a
generic Hadamard rotation merely because some historical metadata uses that name.

### A mixed-precision model, with a mostly ternary payload

`TQ1_0_128` stores 128 ternary values in 26 code bytes plus a two-byte FP16 scale:
**1.75 BPW for that block format**. The whole model averages 1.8378884583 BPW
because other tensors and metadata have different costs.

| Stored type | Tensor count | Payload bytes |
|---|---:|---:|
| `TQ1_0_128` | 113 | 4,976,541,696 |
| `Q4_K` | 41 | 206,438,400 |
| `Q2_K` | 21 | 171,343,872 |
| `Q3_K` | 46 | 140,574,720 |
| BF16 | 38 | 9,961,472 |
| F32 | 169 | 1,415,680 |
| F16 | 210 | 862,208 |

The 638 tensors include transform companions and small floating-point tensors.
The tied embedding/output table remains `Q4_K`; an F16 embedding expansion was
investigated separately and rejected as the deployment direction.

### Sparse teacher targets with protected structural tokens

Teacher caches retain top-128 log probabilities normalized over the full
vocabulary, plus the excluded tail mass. Eleven protected token IDs cover
thinking/tool boundaries, endings and punctuation even when those tokens fall
outside the teacher's top 128.

This prevents rare but structurally important alternatives from disappearing
from the distillation target. The resulting KL is a coarsened distributional
objective, rather than exact full-vocabulary KL.

The original BF16 teacher ran on H100 SXM for trace generation and probability
caching. Student recovery stayed local on the AMD RX 9070 XT. Cache versions
record teacher backend and corpus identity; scores from different teacher
arithmetic or sparse partitions are not treated as interchangeable.

### Component-normalized repair objectives

An objective audit found that math final answers received only about **0.264%**
of mean normalized token weight in an earlier configuration. Increasing a few
token multipliers was insufficient when long reasoning dominated the loss.

Later objectives normalized components per conversation. The final repair run
used 75% teacher KL, 12.5% format/transition cross-entropy and 12.5% final-answer/
tool-body cross-entropy, alongside equal clean and repair replay.

This explicitly gives a short answer or structural transition a meaningful
training contribution while retaining a teacher anchor.

### Train the values that will survive export

Two implementation lessons mattered as much as the objective:

- A scale-offset defect initially pointed TQ1 tuning at code bytes. The real
  FP16 scale begins at byte **26**, after the packed trits. Tests against the
  actual native codec were needed because an old mock fixture shared the defect.
- Finite FP32 optimizer deltas can remain below an FP16 rounding boundary.
  One saved update-4 state contained extensive nonzero deltas but exported
  unchanged fields. That observation did not prove that gradients were absent.

The workflow checks zero-update byte identity, exported/reloaded native loss,
actual field movement and free generation. By final update 10, **32,985,864 native
FP16 fields and 89,689 input-diagonal fields** differed from its starting point.
Payload size remained fixed during that run.

Streaming, activation checkpointing, bounded packed-weight caches and preserved
Adam shards made this recovery practical on the local GPU. A memory investigation
reduced measured replay peak allocation from 10.617 to 6.984 GiB, with additional
compute time. Resume checks preserved both parameters and optimizer moments.

## The recovery data

The teacher corpus deliberately covers more than mathematics:

| Category | Conversations |
|---|---:|
| Math/reasoning | 605 |
| Single-tool conversations | 637 |
| Exactly two parallel calls | 800 |
| Exactly three parallel calls | 400 |
| **Total** | **2,442** |

The initial mixed reasoning/tool collection contains 1,242 conversations. The
parallel-call collection adds 1,200, with a deliberate 2:1 two-call/three-call
ratio. Their combined bank has **2,189 training and 253 development conversations**
and 605,010 scored tokens in complete examples up to 2,048 tokens.

Accepted traces undergo mathematical or typed tool-call checks, schema validation
and structural validation. Tool calls are parsed as data, not executed. The seed
banks exceed the required accepted counts so invalid or unsupported examples can
be rejected without lowering the acceptance standard.

Data preparation also exposed subtle sources of misleading loss. Raw whitespace
before thinking closure had to be restored in **2,094 conversations**. Removing
that whitespace changed the token context and could make a valid close appear
much less likely. A separate audit found **942 contradictory default annotations
across 506 prompts**; later recovery data removed those annotations while keeping
declared types and references. Historical evaluation variants were preserved so
their scores did not silently change.

Source groups, normalized dialogues and complete contexts were checked across
splits. Benchmark-overlap screens and separate retention data reduce leakage
risk; they do not prove the absence of every semantic duplicate or exposure in
the original model's pretraining.

## What update 10 actually improves

The following comparisons use the same **reused development protocols**, against
the precision-repaired QAD model that started the final run:

| Native measurement | Starting checkpoint | Update 10 |
|---|---:|---:|
| Broad syntax passes | 48/56 | **53/56** |
| Exact tool calls | 26/43 | **29/43** |
| Correct math with valid formatting | 8/13 | **10/13** |
| WikiText PPL, **32 chunks** | 33.1415 | **32.7966** |
| Assistant-target PPL | 1.66485 | **1.63866** |
| Tool-target PPL | 1.04801 | **1.01268** |
| Reasoning-body PPL | 1.74436 | **1.72547** |
| Native teacher KL, same sparse target partition | 0.180295 | **0.174155** |

The 32.7966 WikiText result is a **32-chunk guard**, not a full-file update-10
measurement. It must not replace the separately identified full-file historical
results above. Teacher-forced component PPL measures prediction of references;
it does not establish free-generation success at the same rate.

Aggregate gains came with individual losses. Update 10 changed a previously
correct construction-cost answer from `106000` to `105000`, and replaced a valid
profile-call opening `[` with `(`. A further strict-test failure involved a
schema that declared an integer while supplying the fractional default `0.7`.
The original every-prior-pass retention gate therefore **did not pass**.

The checkpoint was retained for testing and larger evaluation with those results
visible. It should not be presented as an unconditional improvement on every
case. Some exact-call mismatches also consist only of explicitly including
schema-listed optional defaults; syntactic validity, schema validity and exact
reference matching are reported as distinct questions.

### Thinking formatting looks substantially healthier

Two known duplicate-`</think>` failures still reproduce on the older parent,
while update 10 passes both.

A later investigation tested 32 natural generations, split between greedy and
sampled decoding, plus 32 teacher-boundary continuations. **No structural
thinking-tag failures appeared in those 64 checks.** The natural runs share
16 prompts; the boundary checks use another 16 sources. These are not 64
independent unseen tasks, and they do not establish universally correct reasoning.

Two earlier apparent missing-close failures were 512-token cutoffs. They closed
under the same protocol with a 1,536-token budget. A cutoff is a different target
from a duplicated, nested or malformed tag.

The native template itself ends the generation prompt with `<think>`. The opening
is consequently part of the input and need not appear again in the generated
stream. Clients must account for that prefill when interpreting `</think>`.

## Beyond update 10: a tiny, goal-directed repair experiment

The remaining profile-call error motivated a separate process-of-elimination
experiment: define the exact current and expected behavior, test interventions,
then retain the smallest tested repair that survives native generation.

The witness was a concrete malformed opening after the tool-call marker. At its
exact native history, the competing tokens were `[` and a merged `(create`
token. Comparing `[` only against a standalone `(` would have diagnosed the
wrong decision.

Original-weight substitutions narrowed a useful contributor to **zero-based
layer 39, expert 20, gate/w1 output channel 804**. Here "neuron" means an entire
projection row of 2,048 weights, not one scalar weight. This establishes a local
causal contribution under the tested intervention, rather than a universal
tool-calling neuron or the unique global cause.

An original-weight oracle could rescue that channel, but deploying the same
repair in the existing ternary codec required re-encoding **rows 804 and 301**.
Both individual same-codec ablations failed the frozen decision; the pair passed.
A matched random two-row control failed. The correct-minus-wrong native logit
margin moved from approximately **-1.547 to +1.228**, and the complete original
generation became a valid exact call.

Only **188 weight bytes differed**. Weight payload growth was **zero**; all other
637 tensors remained exact, including the `Q4_K` embedding/output table. The
candidate gained one syntax and exact-tool pass on the 56-case development check,
with no lost passes there and unchanged printed WikiText-32 PPL.

The transfer result is just as important: on a newly frozen 24-case pilot,
baseline, targeted repair and random repair all scored **21/24 exact calls**.
There was no demonstrated broad transfer benefit. The remaining errors used
wrong function aliases despite correct call counts and literal grammar.

This is a **specific behavioral repair workflow** that can follow PTQ or QAD.
The implemented backend performs discrete PTQ postprocessing; it adds no new
runtime format or training stage. The evidence supports a small local repair,
not a claim of a novel general quantization algorithm.

**This two-row candidate is separate from the update-10 file described here.**
Its SHA-256 is:

```text
461b642545f01936d6803766fd6d7fc87c5a06d2d08657a38b55a6d38bee89ea
```

## Lessons and current limits

The strongest lesson from this journey is that the value of a byte depends on
where it is spent and how its values are encoded. Fitted scales rescued a
collapsed model without adding weight bytes. A few megabytes reallocated into
sensitive components made another large difference. Later, a two-row experiment
could repair one precise decision without changing the bitrate at all.

Those results coexist with remaining weaknesses: wrong tool names or arguments,
occasional omitted calls, incorrect math, possible repetition and unverified
long-context behavior. Neither low reference PPL nor clean thinking tags alone
establish reliable agent behavior.

Failed experiments remain part of the account: joint refits could overfit,
broader precision upgrades could lose previous passes, and a locally successful
sparse repair did not improve the prospective transfer set. Retaining those
controls makes the successful changes easier to assess and reproduce.

The original model and tokenizer are credited to **Liquid AI**. The compression,
HIP integration, recovery experiments and behavioral diagnostics are work from
the **hip-quant / hip_inference** project. The original checkpoint is supplied
under the [LFM Open License v1.0](LICENSE). See [NOTICE](NOTICE) for the derivative
release attribution and modification notice.

Machine-readable measurements, source-report hashes and the exact artifact
identity are recorded in [EVIDENCE.json](EVIDENCE.json). This README describes
an experimental derivative checkpoint; it does not claim official Liquid AI
benchmark scores or a general recovery guarantee.
