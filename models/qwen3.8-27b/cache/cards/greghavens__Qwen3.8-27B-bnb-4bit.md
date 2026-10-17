---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - bitsandbytes
  - nf4
  - 4-bit
  - qlora
  - multimodal
  - vision-language
language:
  - en
---

# Qwen3.8-27B — bitsandbytes NF4

4-bit NF4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for
`transformers` + `bitsandbytes` — the format for QLoRA fine-tuning, or for running the model on a
single 24 GB card. If you are on Blackwell and only need inference, the
[NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) or
[FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) builds will be faster. NF4's advantages are that
it runs anywhere bitsandbytes runs and is the standard base for `peft` adapters.

**The vision tower is deliberately left in bfloat16**, as are `lm_head` and the input embeddings.
Only the language model's linear layers are quantized. Measured against an otherwise identical
all-NF4 control, keeping the tower in bf16 buys **no detectable VQA accuracy** — treat it as a 3.7%
size cost, not a quality claim. See [why](#why-the-vision-tower-stays-in-bf16) and
[the measurement](#vision---three-way-vqa).

**The MTP head is included, in bfloat16.** Earlier revisions of this repo were missing all 15
`mtp.*` tensors. `transformers` has no MTP implementation for this architecture and discards those
weights at load time (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]`), so a
`from_pretrained` -> quantize -> `save_pretrained` round-trip silently dropped them. They are now
grafted back from upstream, byte-identical and unquantized, and `quantize.py` asserts full tensor
inventory parity against the source so this cannot recur. The head is inert under `transformers`;
it matters for vLLM / SGLang speculative decoding. Verified: logits bit-identical to the previous
revision (max |delta| 0.000e+00, 100% argmax agreement) -- this revision adds weights and changes
no behaviour.

**If you are building an agent, read [the BFCL section](#tool-calling---bfcl).** The one regression
this quant has that is both statistically significant and mechanically explained is that it declines
irrelevant prompts less reliably than bf16 — it emits a spurious tool call on 27.0% of no-op prompts
against the base's 23.8%. Tool-call *construction* is at parity.

| | |
|---|---|
| Size | 18.60 GB, 6 shards |
| Base revision | [`1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/commit/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| Quantized | 496 language `Linear4bit` modules |
| Kept bf16 | vision tower (0.46 B params), `lm_head`, `embed_tokens`, all norms |
| wikitext-2 ppl | **6.5571** vs 6.4833 bf16 (**+1.14%**) |
| IFEval (prompt strict) | **0.8189**, identical to bf16 |
| BFCL single-turn | **0.8113** vs 0.8218 bf16 (**-1.04pp**) |
| NoveltyBench distinct_10 | **4.99** vs 4.75 bf16 (no detectable loss) |

## Usage

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "greghavens/Qwen3.8-27B-bnb-4bit"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
proc = AutoProcessor.from_pretrained(model_id)

messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://example.com/cat.jpg"},
    {"type": "text", "text": "What is in this image?"},
]}]
inputs = proc.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=256)
print(proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The quantization config travels in `config.json`, so do **not** pass a `BitsAndBytesConfig` —
just `from_pretrained`. Requires `bitsandbytes` and a supported accelerator.

### QLoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
))
```

Because the vision tower is bf16 rather than `Linear4bit`, adapters targeting vision modules train
against unquantized weights.

## Recipe

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
    llm_int8_skip_modules=["model.visual", "lm_head"],
)
```

One trap worth writing down: `transformers` matches skip patterns with an **anchored** `re.match`
against the full module path, so the intuitive `"visual"` silently fails to match
`model.visual.blocks.0.attn.qkv` and the tower gets quantized anyway. The pattern must be
`"model.visual"`. Assert it rather than trusting it:

```python
from bitsandbytes.nn import Linear4bit
assert not [n for n, m in model.named_modules()
            if isinstance(m, Linear4bit) and n.startswith("model.visual")]
```

NF4 is data-free — blockwise absmax, no calibration set — so this is reproducible and was in fact
produced on CPU in 302 s. See [`quantize.py`](./quantize.py), [`verify.py`](./verify.py),
[`vision_delta.py`](./vision_delta.py).

## Why the vision tower stays in bf16

The usual justification is "vision weights are more sensitive to NF4". On this model that is
**false**. Measured round-trip error `‖dequant(w) − w‖ / ‖w‖` on the actual weights:

| Tensor group | Relative error |
|---|---|
| vision `attn.qkv` | 10.05% |
| vision `mlp.fc1` | 9.61% |
| vision merger | 9.26% |
| language `mlp.down_proj` | 9.38% |
| language `self_attn.q_proj` | 9.32% |

Essentially identical. The argument is about where the error *lands*, not how large it is. Running
three public photos through the tower with every Linear round-tripped through NF4, versus bf16:

- **22.15%** relative L2 change in the image embeddings handed to the language model — about 2.4×
  the weight-level error, so it does compound across the 27 layers
- cosine similarity mean 0.9983, but **min 0.8536** — a minority of patches move a lot

Those embeddings are the language model's *input*. Unlike quantization error inside the LM, which is
injected into a residual stream that still carries the original signal, this perturbation has no
path back to the unquantized representation.

**That embedding shift does not show up in task accuracy.** It has now been measured, and the
result is negative: against an otherwise identical all-NF4 control, the bf16 tower bought
**-0.20pp on DocVQA, -0.33pp on ChartQA and -0.07pp on VQAv2** over 1500 items each, every CI
spanning zero. See [vision - three-way VQA](#vision---three-way-vqa). The perturbation is real —
it flips 6.2-7.8% of answers — it just is not directionally harmful at these settings.

So this stays a **cost-based** decision, not a proven win. The vision tower is 0.46 B params, so
bf16 costs **+0.68 GB, about 3.7% of the checkpoint** (measured: 17.30 vs 16.67 GiB resident). At
that price I would rather not carry an unquantized variable through the language-model error
budget, but anyone tighter on VRAM should quantize it — on this evidence they give up nothing
detectable. Two limits on that evidence: the CIs rule out a benefit larger than roughly 0.8pp and
cannot resolve anything smaller, and everything was run at a pinned 1600 `* 28 * 28` pixel cap.
A higher-resolution regime leans harder on the tower and was not tested.

## Verification

Every claim was checked against the artifact:

- Reloads in 2 s to 18.58 GB VRAM; after reload **0** vision `Linear4bit`, 110 vision bf16 `Linear`,
  496 language `Linear4bit`
- Tensor-level audit of the shards: vision = 333 BF16 tensors, **zero** packed-4bit, **zero**
  `absmax`/`quant_map` sidecars; `embed_tokens` and `lm_head` BF16
- Text generation coherent; vision generation correctly reads a test image
- wikitext-2 perplexity **6.5571** vs **6.4833** bf16 (+1.14%), same harness, 4096-token windows
- All 9 config/tokenizer/processor files **byte-identical** to the base repo. `save_pretrained`
  re-serializes the tokenizer and flips the ByteLevel decoder's `add_prefix_space`, `trim_offsets`
  and `use_regex` from `False` to `True` while dropping `add_bos_token`, `added_tokens_decoder` and
  `additional_special_tokens`; those files were restored from base.

## Benchmarks

### IFEval - instruction following

541 prompts, paired against the bf16 base: identical prompts, identical order, greedy decoding,
batch size pinned across both models (batching changes padding and therefore numerics), thinking
disabled via the chat template. Scored with the official IFEval checkers.

| metric | bf16 base | NF4 | delta |
|---|---|---|---|
| prompt-level strict | 0.8189 | 0.8189 | +0.00pp |
| prompt-level loose | 0.8595 | 0.8447 | -1.48pp |
| instruction-level strict | 0.8741 | 0.8717 | -0.24pp |
| instruction-level loose | 0.9053 | 0.8933 | -1.20pp |

Paired McNemar on prompt-level strict: 28 discordant each way, p = 1.0000. Loose: 30 vs 22,
p = 0.3317. Neither significant.

The two models disagree on ~10% of prompts, but *symmetrically* - NF4 loses 28 prompts the base
passes and wins 28 the base fails. That is noise-like perturbation, not capability loss.

**Statistical power.** With 56 discordant pairs the 95% CI on the delta is +/-2.7pp. This rules out
degradation worse than ~2.7pp and cannot resolve anything smaller. "No significant difference" here
is not "no difference".

Per-instruction-type deltas span -14.8pp to +8.3pp, but each type has only n=12-52 across 21 types,
so they are consistent with multiple-comparison noise: the spread is symmetric and the aggregate is
exactly zero. NF4 responses run 3.7% shorter on average (1243 vs 1291 chars).

Two harness defects had to be fixed before these numbers meant anything (see `ifeval_score.py`):

- IFEval's `letter_frequency` checker rejects any non-`a-z` character and silently substitutes
  `random.choice(ascii_letters)`. Dataset row 1122 specifies `#`, so scoring one file advanced the
  global RNG and the next file was checked against a *different* letter. Scoring the **same** file
  twice differed by 0.18pp - the same order of magnitude as the effect being measured. Fixed by
  reseeding immediately before each file is scored.
- `langdetect`, used by three checkers, is non-deterministic unless `DetectorFactory.seed` is set,
  and lm-eval never sets it.

Raw generations for both models are included as `ifeval_bf16.jsonl` / `ifeval_nf4.jsonl`.

### Vision - three-way VQA

The bf16 vision tower was an argued decision, not a measured one. This is the experiment that
tests it. Three checkpoints, differing only in the vision tower:

| Config | Vision tower | Language weights | Resident |
|---|---|---|---|
| bf16 base | bf16 | bf16 | 50.96 GiB |
| **shipping** (this repo) | **bf16** | NF4 | 17.30 GiB |
| control | **NF4** | NF4 | 16.67 GiB |

The control is built by `python quantize.py --quantize-vision`. Shipping vs control is the only
pair whose sole difference is the vision skip, so it is the only comparison that can answer the
question; bf16 base is the reference ceiling. 1500 items per task, greedy, `max_new_tokens=32`,
and `min_pixels`/`max_pixels` pinned at 256/1600 `* 28 * 28` - Qwen-VL uses dynamic resolution, so
an unpinned processor hands different configs different visual token counts and confounds
everything. Items are prepared once and cached to disk so all three read identical bytes.

**Decisive comparison - does the bf16 tower buy anything?**

| Task | Metric | shipping | control | delta | 95% CI |
|---|---|---|---|---|---|
| DocVQA | ANLS | 0.9559 | 0.9539 | -0.20pp | [-0.71, +0.30] |
| ChartQA | relaxed acc | 0.8353 | 0.8320 | -0.33pp | [-1.13, +0.47] |
| VQAv2 | VQA acc | 0.8453 | 0.8447 | -0.07pp | [-0.78, +0.64] |

**No measurable benefit.** Every CI spans zero (paired bootstrap, 10k resamples). ChartQA McNemar:
shipping wins 22 items, control wins 17, p = 0.52. Quantizing the tower is not inert - it flips
6.2-7.8% of answers - but it wins and loses in equal measure.

Against the bf16 base, neither NF4 config is measurably worse on any of the three:

| Task | bf16 base | shipping delta | control delta |
|---|---|---|---|
| DocVQA | 0.9492 | +0.67pp [+0.13, +1.25] | +0.47pp [-0.13, +1.11] |
| ChartQA | 0.8287 | +0.67pp [-0.60, +1.93] | +0.33pp [-0.93, +1.60] |
| VQAv2 | 0.8458 | -0.04pp [-0.84, +0.73] | -0.11pp [-1.00, +0.76] |

**Do not read the DocVQA row as a quantization gain.** Its CI excludes zero, but item-level win
counts are 28 vs 31 - a coin flip, sign test p = 0.80 - so the effect is ANLS magnitude, not more
correct answers. Inspecting the items where the two cross the ANLS 0.5 threshold, most are the base
choosing a *shorter answer span*, which ANLS at tau = 0.5 zeroes outright: ground truth
`wills lifestyle` scores 0.00 for `Wills`, `data from NHANES` scores 0.00 for `NHANES`. There is no
mechanism by which 4-bit reads documents better; this is a scoring cliff plus a handful of genuine
base-model misses.

VQAv2 is in the set as a falsifier rather than a headline: it needs no fine detail, so if the three
configs separated there too the cause would be the pipeline, not vision precision. They do not -
the spread across all three is 0.11pp.

Two asymmetries between the checkpoint directories were ruled out before trusting any of this. The
quantized dirs carry a `processor_config.json` the base snapshot lacks, and the control's
`tokenizer_config.json` serializes differently. Feeding the same batch through all three processors
produces byte-identical `input_ids` and byte-identical `pixel_values` with identical
`image_grid_thw`, so both are cosmetic and the weights are the only variable.

See [`vqa_gen.py`](./vqa_gen.py), [`vqa_score.py`](./vqa_score.py). Raw generations are included as
`vqa_{task}_{bf16,shipping,control}.jsonl`.

### Tool calling - BFCL

Berkeley Function Calling Leaderboard v4, all 13 single-turn categories, 3641 prompts, paired
against the bf16 base under the same protocol as IFEval: identical prompts from a pre-rendered
file, identical order, greedy decoding, batch size pinned, thinking disabled.

| | bf16 base | NF4 | delta | 95% CI | McNemar |
|---|---|---|---|---|---|
| **overall** | 0.8218 | 0.8113 | **-1.04pp** | [-1.90, -0.19] | 143 / 105, p = 0.0186 |
| minus `live_irrelevance` | 0.8404 | 0.8375 | -0.29pp | [-1.20, +0.62] | 84 / 76, p = 0.5801 |
| `live_irrelevance` alone | 0.7636 | 0.7296 | -3.39pp | [-5.43, -1.24] | 59 / 29, p = 0.0018 |

**The overall loss is one category, not broad degradation.** Removing `live_irrelevance` leaves
-0.29pp with a CI spanning zero and 84-vs-76 discordant pairs, which is a coin flip. Across all 13
category tests under Holm-Bonferroni, `live_irrelevance` (p = 0.0018 vs alpha = 0.0038) is the only
survivor; nothing else is close.

**The failure mode is over-calling, not malformed calls.** On `live_irrelevance` no offered tool
fits and the correct behaviour is to decline:

| | bf16 | NF4 |
|---|---|---|
| emitted a call | 210 / 884 (23.8%) | 239 / 884 (27.0%) |

Counting flips directly: 58 prompts where bf16 declined and NF4 called, against 29 the other way.
Those reconstruct the 59/29 McNemar counts, so the category score is not measuring anything else.
This is computed from raw response text, so it does not depend on the AST parser. 4-bit erodes the
discrimination needed to recognise that nothing applies, and the model falls back on its prior to
call *something*.

Call *construction* is untouched. Every AST-correctness category - `simple` (Python, Java,
JavaScript), `multiple`, `parallel`, `parallel_multiple`, and their `live_` counterparts - is flat.
The effect is also absent from the synthetic `irrelevance` set (+0.42pp, 14 vs 15 flips); it appears
only on real user-submitted prompts, which are messier.

**If you use this checkpoint as an agent, this is the number that matters.** Roughly one extra
spurious call per 30 no-op prompts. Guard it the way you would guard any tool loop - validate that a
proposed call is applicable before executing it - rather than assuming the model will always decline.

Per-category deltas are in the table below. Treat the small ones as noise: `live_relevance` shows
+12.50pp on **n = 16** (two items, p = 0.5000) and is not evidence of anything.

| category | n | bf16 | NF4 | delta | p |
|---|---|---|---|---|---|
| `irrelevance` | 240 | 0.7417 | 0.7458 | +0.42pp | 1.0000 |
| `live_irrelevance` | 884 | 0.7636 | 0.7296 | -3.39pp | 0.0018 |
| `live_multiple` | 1053 | 0.8082 | 0.8015 | -0.66pp | 0.4101 |
| `live_parallel` | 16 | 0.8125 | 0.8125 | +0.00pp | 1.0000 |
| `live_parallel_multiple` | 24 | 0.7500 | 0.7500 | +0.00pp | 1.0000 |
| `live_relevance` | 16 | 0.7500 | 0.8750 | +12.50pp | 0.5000 |
| `live_simple` | 258 | 0.8798 | 0.8760 | -0.39pp | 1.0000 |
| `multiple` | 200 | 0.9700 | 0.9750 | +0.50pp | 1.0000 |
| `parallel` | 200 | 0.8250 | 0.8450 | +2.00pp | 0.5572 |
| `parallel_multiple` | 200 | 0.8350 | 0.8100 | -2.50pp | 0.3593 |
| `simple_java` | 100 | 0.6500 | 0.6400 | -1.00pp | 1.0000 |
| `simple_javascript` | 50 | 0.7800 | 0.7800 | +0.00pp | 1.0000 |
| `simple_python` | 400 | 0.9700 | 0.9650 | -0.50pp | 0.6875 |

**Harness notes.** Two things had to be right before these numbers meant anything:

- Generation uses the model's **own** tool-calling chat template, not BFCL's stock `QwenHandler`.
  That handler targets the older `<tool_call>{"name":..,"arguments":..}</tool_call>` JSON format;
  Qwen3.8 emits an XML dialect (`<function=NAME><parameter=KEY>`). Scoring through the wrong
  template would have measured format mismatch and called it quantization damage.
- `bfcl_eval` serves function docs in two flavours. The hinted docs rewrite every Java/JS parameter
  type to lowercase `"string"` - that is what the model is shown - but the checker must receive the
  **unhinted** docs, because it looks the declared type up in `JAVA_TYPE_CONVERSION`, which has no
  lowercase `"string"` key. Feeding it hinted docs raises `KeyError` on every Java and JavaScript
  item. An early version of `bfcl_score.py` swallowed that in a blanket `except`, scoring both
  models 0 on those categories and reporting perfect agreement - a broken harness manufacturing a
  confident null. Checker crashes are now counted and printed separately from genuine decode
  failures; this run had zero.

Both response files and per-item verdicts are included: `bfcl_bf16.jsonl`, `bfcl_nf4.jsonl`,
`bfcl_verdicts.jsonl`, with `bfcl_prep.py` / `bfcl_gen.py` / `bfcl_score.py`.

NF4 generation hit two GPU faults on this host (Xid 8 "GPU is probably locked", then
`cudaErrorLaunchTimeout`), both in the bitsandbytes dequant path with ~80 GB of VRAM free, while
bf16 ran 2h16m clean on the same card. That is a driver interaction, not a property of the
checkpoint, and it is why `bfcl_gen.py` journals each completed batch. Resume is keyed by batch
offset and skips whole batches only, so replayed batch composition - and therefore padding and
reduction order - matches an uninterrupted run; verified byte-identical on a truncated-journal test.

### Output diversity - NoveltyBench

Everything above is a correctness measure under greedy decoding. Correctness and *diversity* are
orthogonal failure modes: a quantized model can reproduce the argmax token perfectly while its
sampling distribution sharpens, so repeated sampling at temperature returns the same answer over
and over. Nothing in IFEval, BFCL or the VQA runs can see that. This section measures it with
[NoveltyBench](https://github.com/yimingzhang/novelty-bench) (NB-Curated, 100 prompts, k=10),
scored by their **unmodified** `partition.py` / `score.py` / `summarize.py` - the numbers below are
upstream's own metric, not a local invention.

`distinct_k` is the number of equivalence classes among k samples, partitioned by their
`deberta-v3-large-generation-similarity` classifier. `utility_k` is the discounted cumulative
reward over those classes under `Skywork-Reward-Gemma-2-27B-v0.2`.

**Three arms, not two.** This benchmark samples at temperature (1.0, top_p 0.95, top_k 20, 512 new
tokens), so two runs of the *same* model disagree by some amount. Without measuring that floor, any
NF4-vs-bf16 gap is uninterpretable. The third arm is the bf16 base re-run under a different seed.

| arm | distinct_10 | utility_10 |
|---|---|---|
| bf16 base, seed 0 | 4.75 | 5.210 |
| **NF4, seed 0** (this repo) | **4.99** | **5.359** |
| bf16 base, seed 1 (noise floor) | 4.91 | 5.228 |

Paired per-prompt deltas against bf16 seed 0, 10k-resample bootstrap CIs:

| comparison | distinct_10 | utility_10 |
|---|---|---|
| NF4 | **+0.240** [-0.060, +0.540] | **+0.149** [-0.125, +0.432] |
| bf16 reseeded (noise floor) | +0.160 [-0.200, +0.530] | +0.018 [-0.316, +0.360] |
| NF4 vs bf16 seed 1 | +0.080 [-0.290, +0.450] | - |

NF4 scores *higher* than the base on both metrics - the opposite direction from degradation - and
every interval spans zero. Exact sign test on `distinct_10`: 41 prompts better, 31 worse,
p = 0.2888. Under Holm-Bonferroni across the three NF4-vs-bf16 tests reported here, nothing is
significant (smallest adjusted p = 0.17). This is a null result, and the +0.240 should not be read
as an improvement either: it is the same size as the reseed floor.

**Statistical power.** With n=100 the minimum detectable effect is 0.43 distinct classes at 80%
power (0.40 for `utility_10`), and the worst-case CI lower bound across both bf16 reference draws
is -0.29. Degradation larger than roughly 0.3-0.4 classes out of 10 is ruled out; anything smaller
is below this benchmark's resolution at this sample size and is *not* excluded. NB-WildChat (1000
prompts) would tighten the bound to ~0.14; it was not run.

**The floor arm earned its place.** Broken out by category, the two bf16 seeds differ by 1.20
distinct classes on `Random Generation & Selection` (5.27 vs 6.47, n=15) - larger than any
NF4 effect anywhere in the table. Per-category deltas at n=7-28 are not interpretable here, in
either direction.

**Mode collapse is a base-model property, not a quantization artifact.** Severe collapse - 2 or
fewer distinct outputs from 10 samples - is the failure mode that matters if you are sampling a
model repeatedly for synthetic data:

| | prompts collapsing to <=2 classes |
|---|---|
| bf16 base, seed 0 | 20 / 100 |
| NF4, seed 0 | 12 / 100 |
| bf16 base, seed 1 | 17 / 100 |

NF4 collapses on *fewer* prompts, but the comparison that matters is the bf16 pair: reseeding the
same model moves 20 to 17, and only 6 of the original 20 collapse again. Which prompts collapse is
mostly sampling luck, not a stable property of a checkpoint. Only 4 prompts collapse in all three
arms, and they are inherently constrained ("Name one person who was involved in the American
Revolution."). The sharpest case is `curated-57`, "Name one UFC fighter.": both bf16 and NF4
returned "Conor McGregor" ten times out of ten, byte-identical across the two checkpoints - one
class out of ten. That ceiling belongs to Qwen3.8-27B, and quantization neither caused it nor
worsened it. Practical consequence: if you are generating many samples from one prompt, NF4 is not
your limiting factor, and the mitigation is prompt-side rather than checkpoint-side.

**Harness notes.** Only the sampling is local (`novelty_gen.py`); all measurement is upstream's.
Four deliberate deviations from their reference `inference.py`:

- `AutoModelForImageTextToText`, not `AutoModelForCausalLM` - this model is multimodal and does not
  load under the causal-LM class.
- `skip_special_tokens=True`. Their batch path decodes with `skip_special_tokens=False`, which
  leaves `<|im_end|>` on every generation; that text is fed to the equivalence classifier, adding a
  constant shared suffix to every response on both models.
- Sampling parameters passed explicitly rather than inherited from each checkpoint's
  `generation_config.json`. The two configs are byte-identical, but relying on that would make the
  comparison silently config-dependent rather than weights-dependent.
- Seeded per (run, prompt); the reference implementation is unseeded. On a stochastic benchmark
  reproducibility is the only way to separate a real gap from resampling noise.

Thinking is disabled, consistent with the IFEval and BFCL runs. As on the BFCL run, NF4 generation
hit three GPU faults in the bitsandbytes dequant path while both bf16 arms ran clean on the same
card; `novelty_gen.py` journals each completed prompt and derives its seed from the prompt index
alone, so a resumed run reproduces an uninterrupted one.

Full per-prompt records - generations, partition assignment, per-generation reward, and utility -
are included for all three arms as `novelty_bf16_s0.jsonl`, `novelty_nf4_s0.jsonl`,
`novelty_bf16_s1.jsonl`, with `novelty_gen.py` and `novelty_compare.py`.

### Logit divergence vs bf16

`divergence.py` runs both models on identical inputs in one process. Hidden-state drift accumulates
smoothly with depth - relative L2 0.025 at layer 1 to 0.40 at layer 64, cosine 1.000 to 0.907 - with
no pathological layer.

Distributional metrics badly overstate the damage on this model, which is worth knowing if you plan
to evaluate it that way. Tool-calling prompts showed mean KL 0.93 (30x plain text) and 12% of
*high-confidence* tokens flipping top-1, yet the generated tool calls were functionally identical:
same function selected, same argument values, same decision to parallelise. Every divergence was a
synonym swap inside the reasoning trace ("I should use" -> "I'll use"). Score this model on
verifiable outcomes, not on KL or token agreement - BFCL above is that measurement, and it puts
call construction at parity while isolating the real regression to declining irrelevant prompts.

### Not measured

- **Multi-turn and agentic tool use.** BFCL's single-turn categories were run; the multi-turn and
  agentic categories, which test state tracking across a session, were not.
- **Thinking-mode behaviour.** IFEval, BFCL and NoveltyBench were all run with thinking disabled.
- **Diversity below ~0.3 distinct classes.** The NoveltyBench run bounds diversity loss but cannot
  resolve effects smaller than its n=100 detection threshold; NB-WildChat was not run.

## Provenance

Base weights are `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, checked
against the repo's own `crc32.txt`. Five of eight listed files matched; `chat_template.jinja`,
`generation_config.json` and `tokenizer_config.json` did not, because Qwen re-uploaded exactly those
three about two hours *after* the manifest was generated (commits `412f8b6b`, `dbdc473d`, `452438ec`
on 2026-08-13). Those were re-verified byte-for-byte against live `main` instead.

Built with `transformers` 5.10.1, `bitsandbytes` 0.49.2, `torch` 2.10.0+cu128.

## License

Apache 2.0, inherited from the base model; `LICENSE` is included.
