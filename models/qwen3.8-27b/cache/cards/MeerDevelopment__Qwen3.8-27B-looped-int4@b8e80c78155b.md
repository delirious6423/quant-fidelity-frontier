---
library_name: transformers
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - qwen3_5
  - looped-transformer
  - recurrent-depth
  - int4
  - gptq
  - marlin
  - 4-bit
  - image-text-to-text
---

> **This repository carries no model weights.** It is an overlay release: the int4 checkpoint it starts from is
> Qwen3.8-27B in W4A16 AutoRound form, and `scripts/build_looped_dir.sh` builds the loadable model directory from it
> in a few seconds, without duplicating the 16 GB. What is here is the whole proof of concept: the vendored forward
> that implements the loop, the reference `config/config.json`, the benchmark harness, the specs, and every result row
> the tables below are computed from, under `bench/results/`.

# Looped Qwen3.8-27B, int4, one RTX 3090

To get a loadable directory, start from the int4 base checkpoint and run the build script:

```bash
git clone https://huggingface.co/MeerDevelopment/Qwen3.8-27B-looped-int4
cd Qwen3.8-27B-looped-int4
scripts/build_looped_dir.sh <base-gptq-dir> ~/looped-model
```

`<base-gptq-dir>` is a Qwen3.8-27B int4 W4A16 checkpoint in GPTQ layout, the layout `scripts/convert_ct_to_gptq.py`
produces from a compressed-tensors export. The output directory is symlinks into that checkpoint plus two files that
differ: the patched `config.json` and `vendor/modeling_qwen3_5_looped.py`. Pass `--copy` for a standalone directory at
the cost of the 16 GB. `config/config.json` in this repo is the reference output, so you can diff yours against it.

A proof of concept built by replaying part of an existing model at inference, with no retraining and no new weights. Qwen3.8-27B in int4 (W4A16) has 64 stored decoder layers; this build applies layers 16 through 47 a second time, so one forward runs 96 layer applications from the 64 stored modules.

Both arms fit on a single RTX 3090 and sit at about 16 GB of its 24 GB, at 32768 tokens of context, so the benchmark compares them at equal footprint. The only variable is the loop.

This is not a better Qwen3.8-27B. Read the limits first.

## Limits

These are the measured results, and they are mostly negative. Nothing here is a claim of improvement.

- **It is much worse on GSM8K.** The looped arm scores 0.210 against 0.620 for the unmodified int4 base. This is not a truncation artefact. Restricted to the 147 items it finished it scores 0.259 against 0.623 over 199, and ten truncated items re-run at four times the token budget scored exactly what they scored before. Worse, the loop costs accuracy even where it does not degenerate: the 135 GSM8K traces that never repeat score 0.230, against 0.622 over 196 clean traces for the base arm.
- **On MATH-500 it is better only where it converges, and it often does not converge.** 96 of its 200 items ran to the 4096-token cap, and most of those traces fall into long repeating loops; a re-run at 16384 tokens left four of five still looping and changed no score. Among the 104 items it did finish it scores 0.692 against 0.562 over 194 for the base arm, and the 97 traces that never repeat score 0.660 against 0.558. Its table score of 0.450 is real: the loop helps on hard multi-step problems and then collapses into repetition on the other half.
- **BFCL is nearly a tie.** Both arms match the called function name on all 200 single-function items, and once the arguments are compared the base arm leads by 15 calls out of 200, 0.925 against 0.910. It is not a capability signal in either direction.
- **The loop helps exactly where nothing is generated.** On GPQA diamond, a multiple-choice task scored by log-probability, the looped arm is better: 0.343 against 0.293. That is the one clear gain, and it is a gain in scoring a fixed answer, not in producing one.
- **It roughly doubles the work per item.** GSM8K took 25 minutes of task wall clock for the base arm and 66 for the looped arm; MATH-500 took 97 against 198. Its speed is not a claim, and it is the plain Hugging Face path with no vLLM integration, no CUDA graphs and no MTP head.
- **The context is 32768 tokens, not the base model's 262144.** The long-context path was not recovered for the replay.
- **The loop count is fixed at two.** There was no sweep over blocks or repeat counts, and a third application is untested.
- **This is one card and one quantisation.** Every number is from a single RTX 3090, 24 GB, with int4 W4A16 weights in GPTQ layout loaded through gptqmodel's Marlin kernels. Results on another card or another quant are not implied.
- **Not measured:** aider polyglot, multi-turn BFCL, SWE-bench, and any vision benchmark, including the vision half of this hybrid model. Out of scope, not passed.

Reproducibility evidence rather than capability evidence: setting `repeats` to 1 makes the looped schedule identical to the stored layer order, and the two arms then match on logits to below 1e-5. The surgery is the only difference between them.

## What the model is

- Base: `Qwen/Qwen3.8-27B`, a hybrid text and vision model. 64 decoder layers, hidden size 5120, 24 query and 4 KV heads, vocabulary 248320, `full_attention_interval` 4.
- Weights: int4 W4A16, group size 128, symmetric, stored in GPTQ layout, loaded with `gptqmodel` Marlin kernels. The embeddings are stored at 8 bits and dequantised on load. No weight was requantised for this build.
- Loop: layers 16 through 47 applied twice, layers 0 through 15 and 48 through 63 applied once. 64 stored modules, 96 applications per forward.
- Licence: Apache-2.0, inherited from `Qwen/Qwen3.8-27B`.

The 48 Gated DeltaNet layers and the 16 full-attention layers keep the attention class they were trained with. The loop does not change which mask a stored layer uses: the mask comes from `layer_types[stored_index]`, never from the application index.

## How the loop works

`config.json` carries the schedule beside the layer count:

```json
"loop": {"start": 16, "end": 47, "repeats": 2}
```

The vendored forward builds one schedule from it: layers 0 to 15 once, then the block 16 to 47 twice, then layers 48 to 63 once. The checkpoint's `auto_map` points at that forward:

```json
"auto_map": {
  "AutoModelForImageTextToText": "modeling_qwen3_5_looped.LoopedQwen3_5ForConditionalGeneration"
}
```

### Why 16 to 47, and why twice

Both numbers are a priori choices from the architecture, not the result of a sweep on this data. The numbers below are what they were chosen for, and the benchmarks test the choice rather than justify it.

**The middle half, because layers are not interchangeable.** Indices are 0-based, so 16 to 47 is 32 of the 64 layers. Layers 0 to 15 are the prelude: closest to the token embeddings, they form the features everything above consumes, and replaying them would mean re-deriving the input representation of a token that is already represented. Layers 48 to 63 are the coda: they were trained to be last, sitting immediately before the final norm and the language head, so replaying them applies output-forming weights twice, which is the least defensible edit to make to a model that is not retrained.

**Those two boundaries, because they respect the attention cadence.** `layer_types` puts a full-attention layer every fourth index, at 3, 7, 11, 15, 19 and so on to 63. The block 16 to 47 therefore holds exactly 8 full-attention and 24 Gated DeltaNet layers, which is the model's own 8 to 24 ratio, so the replay keeps the balance between the two attention classes that the weights were trained under. Both edges also fall on multiples of four, and the block ends on index 47, a full-attention layer. That last detail is what lets the cache rule split cleanly: the final replayed layer writes its KV entry once, and everything from 48 onwards is untouched.

**Twice, because that is the smallest loop that is a loop.** With `repeats: 1` the schedule is identical to the stored order, which is exactly how the two arms are proven to be the same model: one repeat, and the logits match to below 1e-5. So one is the parity baseline and two is the first real step. Three is untested and this release does not claim either direction for it. The residual bridge between passes is a single tensor add that keeps the second pass's input distribution near what the layers were trained on, and that argument gets harder to make with each additional pass.

The measured results explain why more passes are unlikely to be the interesting axis. Two passes already help where the answer is scored rather than generated, and already collapse into repetition on long generation. More passes would plausibly sharpen both effects rather than remove the collapse. Changing `repeats` is a one-line edit to `config.json`, and it costs a full re-run of both harness stages per arm to measure.

Two cache rules make the replay work, and they differ by layer class:

- A replayed full-attention layer writes its KV entry on the first pass only. A second write would append a duplicate for a token already in the cache, doubling the cache length and corrupting every later step.
- A replayed Gated DeltaNet layer advances its recurrent state on both passes. That state is what carries one pass into the next, so freezing it would make the second pass a no-op.

Inside the block a linear layer's recurrent state therefore advances twice per generated token, and outside it once. The extra pass changes the block's contribution at that token; it does not shift the state forward by one token position.

Between the two passes the block's input hidden state is added to the current hidden state once, at the first application of the second pass. It is one tensor add, not a learned parameter.

The full-attention KV cache length equals the number of tokens generated. The forward asserts that after every step and raises naming the application if it drifts. The harness that produced these numbers also guards the loader: it forces the remote code rather than transformers' native `qwen3_5` class, asserts the loaded class name, requires every projection to be a Marlin kernel rather than a stock `torch.nn.Linear`, requires the quantisation config to declare 4 bits, and refuses a resident footprint over 20 GB. Without the class assertion the looped arm would silently load the unmodified model and the two arms would answer identically.

## How it was built

1. Start from the int4 W4A16 AutoRound checkpoint of Qwen3.8-27B.
2. Rewrite it to GPTQ layout with `scripts/convert_ct_to_gptq.py`. The conversion is a pure transpose of the packed weights; the scales carry over, the zero points are the symmetric `0x88888888`, and the dequantised result matches compressed-tensors' own unpack at a maximum absolute difference of 0.0.
The int4 weights are a third-party AutoRound W4A16 export of Qwen3.8-27B. Its Hub repository could not be identified at the time of writing, so it is not named here rather than guessed. No weight was quantised for this build, and the loop changes no tensor: the licence is Qwen3.8-27B's own, Apache-2.0.

3. Run `scripts/build_looped_dir.sh BASE_DIR OUT_DIR` to build the model directory: copy the base config and patch in the `loop` block and the `auto_map` entry, link or copy the weights and tokenizer files, and place the vendored `modeling_qwen3_5_looped.py`.
4. Load with `transformers` and `gptqmodel`. The measured stack is `transformers 5.20.2`, `gptqmodel 7.12.2` and `torch 2.13.0` with CUDA 13 wheels, running fp16 activations.

Nothing is retrained, fine-tuned or requantised. The same stored tensors are replayed by a different forward.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

path = "path/to/looped/model"
model = AutoModelForImageTextToText.from_pretrained(
    path, trust_remote_code=True, device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(path)
```

`trust_remote_code=True` is required: the loop lives in the checkpoint's own modeling file. Loading needs the `gptqmodel` kernels; a stock `transformers` load has no int4 path for these tensors.

## What was measured

Both arms, one RTX 3090, same context (32768) and decoding. Budgets: 1536 new tokens on GSM8K, 4096 on MATH-500 and AIME, 2048 on GPQA, 512 on BFCL. Greedy everywhere except BFCL, which uses 0.7 with a recorded seed and carries a couple of points of run-to-run variance.

| task | items | base | looped | truncated base / looped |
|---|---|---|---|---|
| GSM8K | 200 | 0.620 | 0.210 | 1 / 53 |
| MATH-500 | 200 | 0.555 | 0.450 | 6 / 96 |
| AIME 2024 | 30 | 0.233 | 0.000 | 20 / 28 |
| AIME 2025 | 30 | 0.233 | 0.000 | 19 / 28 |
| GPQA diamond | 198 | 0.293 | 0.343 | 0 / 0 |
| BFCL simple_python | 200 | 0.925 | 0.910 | 0 / 0 |

The AIME base column is a floor: re-running only its capped items at twice the budget recovers five of thirty-nine, which would put it at 0.333 and 0.300. The looped arm scores zero on all sixty AIME items and recovers nothing; twenty-five to twenty-seven of its thirty traces per set are repetition loops.

What separates the arms is whether the answer is generated at length, not how hard the task is.

- GPQA scores a multiple-choice answer by log-probability, so nothing is generated and the loop cannot degenerate. The looped arm is better there, 0.343 against 0.293.
- MATH-500 needs a chain of reasoning. Among the 97 looped traces that never repeat it scores 0.660 against 0.558, then half its traces collapse into repetition and the table figure lands at 0.450.
- GSM8K needs a short chain, and the loop costs accuracy outright: its 135 clean traces score 0.230 against 0.622.
- AIME needs the longest chain, and the looped arm almost never produces a usable trace at that length.

BFCL matches the function name on all 200 items for both arms; the difference is arguments alone, 185 full calls against 175.

The MATH-500 column is the task's `math_verify` metric, not its `exact_match`. That filter wants an answer ending in "The answer is", which this model does not produce, so `exact_match` reads 0.0 for both arms on every item. The harness prefers `math_verify` and falls back to `exact_match` only when it is absent.

The looped arm is much slower. GSM8K took 25 minutes of task wall clock for 200 items against 66 for the looped arm, and MATH-500 took 97 minutes against 198, so the loop roughly doubles the work per item.

Not measured: aider polyglot, multi-turn BFCL, SWE-bench, and every vision benchmark. AIME is 30 items per set, so one item is 3.3 points.

## Hardware

One RTX 3090, 24 GB, about 16 GB resident for either arm, one arm on the card at a time. The project exists because the 27B fits that card at int4 and the interesting question is what the loop does to capability at that footprint. GPU1 on the host is never used by this project.
