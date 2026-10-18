---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- deepseek
- deepseek-v4
- deepseek-v4-flash
- moe
- mixture-of-experts
- quantized
- 2-bit
- 4-bit
- iq2_xxs
- q2_k
- q4_k
- ds4
- dwarfstar
- imatrix
- apple-silicon
- metal
---

# DeepSeek-V4-Flash-0731: GGUF for ds4 (mixed 2+4 bit)

GGUF builds of **[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)**
(the official release of DeepSeek V4 Flash: 304B total parameters, hybrid CSA+HCA
attention, 1M context) for the
**[ds4 / DwarfStar](https://github.com/antirez/ds4)** inference engine. Runs fully
resident on a single 128 GB Apple Silicon machine.

Quantized in the asymmetric style of
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf): crush the
routed experts (they are almost all of the weights), keep the decision-making parts
high precision. **The filename is the spec.**

> 📊 **The imatrix is published here too**, not just the models it produced:
> [`imatrix/DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-1p5m.dat`](./imatrix/DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-1p5m.dat)
> (430 MB), collected **on the `0731` weights themselves**, 2729 prompts / 1.5M tokens /
> 387M routed-expert observations, 129 entries (43 layers × gate/up/down, full coverage,
> zero missing tensors). Reuse it with `deepseek4-quantize --imatrix` to build your own
> mix, or to check mine.

> **This is not my recipe.** The recipe, the quantizer (`gguf-tools/deepseek4-quantize`),
> the imatrix pipeline and the engine are all [**antirez**](https://github.com/antirez)
> and the DwarfStar contributors. All I did was run that toolchain against the newer
> `0731` checkpoint and collect a fresh imatrix on those weights.

> ✅ **antirez has now shipped official `0731` builds** (2026-08-01), including
> `...Layers37-42Q4KExperts-...-imatrix-fixed-0731.gguf`, the same recipe as the file
> here. **Get them from
> [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)**, that is
> the reference.
>
> What this repo still adds: **the imatrix collected on the `0731` weights, published as a
> file**. As of writing, his `imatrix/` folder publishes only the preview-checkpoint `.dat`
> (last touched 2026-05-12; the `fixed` in his filenames refers to a May "fixed routed-mid
> imatrix build", not to anything `0731`-specific). What he actually built his `0731` files
> with is not something I can verify; he may well have collected one without publishing it.
> So: this is the only `0731`-collected imatrix I know to be *available as a file*, not a
> claim that his builds lack one.
>
> And read [the analysis below](#why-the-imatrix-buys-so-little--reading-the-dat-itself)
> before assuming it matters much: measured on general text, using it changes nothing
> significant, and there is a structural reason why.

> ⚠️ **Needs ds4 / DwarfStar.** This is not a generic GGUF. It will not load in
> llama.cpp, Ollama or LM Studio: the tensor layout, quant mix and metadata are
> specific to the DS4 engine.

> Not affiliated with DeepSeek or antirez. Weights © DeepSeek, released under MIT.

## Installation

```bash
git clone https://github.com/antirez/ds4
cd ds4
make                  # macOS Metal
```

Then download one of the files below into `gguf/` and point `-m` at it.

## Files

| File | Size | imatrix | status |
|------|------|---------|--------|
| `DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` | 97.6 GB | none | ✅ available |
| `DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | 97.6 GB | ✅ collected on `0731` | ✅ available |
| `imatrix/DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-1p5m.dat` | 430 MB | n/a | ✅ available |

**Either works.** On held-out wikitext the two are statistically indistinguishable
(−1.36 %, p ≈ 0.098; [full numbers](#does-the-imatrix-actually-help)). The imatrix build
is the one to prefer *a priori*, since its statistics come from the `0731` weights, but I
have no measurement proving it on general text. The plain build is the intermediate the
imatrix was collected on, published so the comparison can be reproduced rather than taken
on trust.

## Usage

```bash
./ds4 -m gguf/DeepSeek-V4-Flash-0731-...-imatrix.gguf \
  -p "Explain Redis streams in one paragraph."

./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Sampling, per DeepSeek: `temperature = 1.0`, `top_p = 0.95` for agentic scenarios,
`top_p = 1.0` otherwise.

> **`0731` introduces a `reasoning_effort` parameter** with three levels (`low`,
> `high`, `max`) controlling how much the model deliberates before answering. The
> preview's card documented no such parameter (it exposed `thinking_mode` only), so
> this is an addition. `0731` is also reported to spend substantially more tokens
> thinking than the preview; budget context and output limits accordingly, or use
> `--nothink`.

## Quantization recipe

Unchanged from antirez's mixed 2+4 bit build:

| Tensor class | Quant | Notes |
|---|---|---|
| `blk.37..42.ffn_{gate,up,down}_exps` | `Q4_K` | last 6 layers, most sensitive |
| `blk.*.ffn_{gate,up}_exps` | `IQ2_XXS` | routed expert gate/up |
| `blk.*.ffn_down_exps` | `Q2_K` | K-quant: `down` is far more sensitive |
| `blk.*.ffn_{gate,up,down}_shexp` | `Q8_0` | shared experts |
| `blk.*.attn_q_a/q_b/kv/output_a/output_b` | `Q8_0` | MLA + low-rank output |
| `output.weight` | `Q8_0` | |
| `token_embd.weight` | `F16` | |
| `blk.*.ffn_gate_inp` | `F16` | router, never touch this |
| router bias, `attn_sinks`, all `*_norm` | `F32` | |
| `blk.*.ffn_gate_tid2eid` | `I32` | hash-routing tables, first 3 layers |
| `attn_compressor_*`, `indexer_*`, `hc_*` | `F16` / `F32` | DSv4-specific blocks |

Routed experts are the overwhelming majority of the parameters, but each individual
expert only sees a fraction of the tokens, so aggressive quantization there costs
little on average. Everything every token passes through stays high precision.

## How this was built

### The source weights are already 4-bit

Worth stating plainly, because it is easy to assume otherwise: **DeepSeek never
published a BF16 version of this model.** The routed experts ship at FP4 because
that is how they were *trained*: the paper ([2606.19348](https://arxiv.org/abs/2606.19348),
§5.2.1) applies **FP4 quantization-aware training** to the MoE expert weights and the
indexer QK path during post-training, and states that "the routed expert parameters
utilize FP4 precision".

Reading the safetensors headers across all 48 shards:

| dtype | tensors | bytes | what it is |
|---|---|---|---|
| `I8` | 35 328 | 148.18 GB | FP4 weights, **two per byte** |
| `F8_E8M0` | 35 718 | 9.26 GB | block scales, one per 32 weights |
| `F8_E4M3` | 390 | 6.30 GB | |
| `BF16` | 445 | 2.97 GB | attention / shared / embeddings |
| `F32` | 433 | 0.15 GB | norms, biases |
| **total** | **72 317** | **166.88 GB** | |

Declared shapes are the *packed* shapes. Taken literally they sum to 165B parameters.
Unpacking gives the real count. Note these are **bytes on the left, parameters on the
right**:

```text
148.18 GB of I8   -> 148.18e9 bytes x 2 weights/byte = 296.36e9 FP4 weights
                   +   1.48e9 BF16  +  6.30e9 F8_E4M3
                   = 304.14e9 parameters
```

which is the advertised count.

**The scale count confirms the unpacking independently.** If `I8` really holds two FP4
weights per byte in blocks of 32, there must be `296.36e9 / 32 = 9.26e9` block scales,
and `F8_E8M0` is exactly 9.26 GB, i.e. 9.26e9 one-byte scales. The two numbers are derived from
different fields of the header and agree, so this isn't a guess about the layout.

Two different bit figures follow, and they should not be confused:

| | |
|---|---|
| FP4 expert path | **4.25 bits/weight** (4 bits + one 8-bit scale per 32 weights) |
| whole model on disk | **4.39 bits/param** (`166.88 GB × 8 / 304.1B`, including the BF16/F32 tensors that were never quantized) |

So this is a **4-bit → 2-bit requantization**: one lossy step, not two. The tensors
crushed to `IQ2_XXS` are exactly the ones DeepSeek trained at FP4. The quantizer
unpacks FP4/FP8 before re-encoding.

### Two passes, because the imatrix needs a running model

Collecting an imatrix means *running* the model: the DwarfStar collector hooks the
layer-major Metal prefill graph and accumulates `sum(x[column]^2)` per routed expert.
That needs a loadable GGUF, which doesn't exist yet before the first quantization:

1. **Quantize without imatrix** → the plain file above.
2. **Run it** over the calibration corpus to collect the imatrix.
3. **Re-quantize with `--imatrix`** → the final file.

Budget disk for it: the 167 GB of source safetensors have to stay available for step 3,
and each GGUF is 97.6 GB, so plan for ~360 GB free if you want to keep both builds
around. Drop the intermediate after step 2 if you don't.

antirez collects on the Q4 build for Flash, but Q4 for `0731` is ~165 GB and won't stay
resident on 128 GB. So this imatrix was collected on the **2-bit** build, which is
exactly what the upstream README does for DeepSeek-V4-Pro, for the same reason.

That's defensible because of the recipe itself: the router is `F16`, attention
projections and shared experts are `Q8_0`. Only the routed experts are degraded, so
the *routing decisions* observed during collection are essentially the full-precision
model's.

### Why a fresh imatrix and not the preview's

The collector records, for `down` tensors, the routed SwiGLU row **after route
weighting**, so the statistics depend directly on which experts fire. `0731` moves a
lot versus the preview on exactly that axis (DeepSWE 7.3 → 54.4, Terminal-Bench
61.8 → 82.7), and the calibration corpus is heavily agent/code weighted (1106 of 4690
prompts are `agent`, 2074 are `source`). A stale imatrix would be most wrong precisely
on `ffn_down_exps`, the tensor the recipe protects with `Q2_K` because it's the most
fragile. So it was recollected.

**That reasoning turned out to be only half right.** It holds for `ffn_down_exps`, but the
collector records something else entirely for `gate`/`up`. See
[why the imatrix buys so little](#why-the-imatrix-buys-so-little-reading-the-dat-itself),
which measures the consequence and largely deflates this argument.

### Calibration corpus

Upstream's, unmodified: `gguf-tools/imatrix/dataset/rendered_prompts.txt` from
[antirez/ds4](https://github.com/antirez/ds4). Not mine, and not regenerated:

| | |
|---|---|
| ds4 commit | `54b36ed` |
| last commit touching the dataset | `b166a73` |
| `rendered_prompts.txt` | 12 MB, sha256 `1159b0e7f1eff1c9…` |
| prompts | 4690 |
| tokens | ~2.92M (bytes/4 estimate, per `manifest.json`) |

Category mix, straight from upstream's `manifest.json`: `source` 2074, `agent` 1106,
`language` 1024, `translation` 180, `eval_reasoning` 150, `programming` 48, `general`
40, `long_context` 36, `algorithms` 32.

> **Use the tracked file, don't regenerate it.** `build_ds4_imatrix_dataset.py` builds
> part of the corpus *from the ds4 repository's own C/Metal sources*, so regenerating
> at a different commit yields a different corpus and a non-comparable imatrix. The
> tracked file was used verbatim, which is what makes this reproducible: check out
> `54b36ed`, verify the sha256, and you have the same input.

### Why the Hub sidebar says 284B

The model tree widget reads the GGUF metadata, which comes from the `--template` file,
a preview-checkpoint build. So it reports `284B params` and architecture `deepseek4`,
the preview's numbers. The tensor data is `0731` (that is what `--compare-tensor`
verifies), and the 304.1B figure derived from the safetensors headers above is the
correct one for this checkpoint. Fixing the sidebar would mean rewriting the metadata
block; it does not affect inference.

### Template GGUF

`deepseek4-quantize` regenerates tensor bytes from safetensors but takes metadata,
tokenizer, tensor order and logical shapes from an existing DS4 GGUF passed as
`--template`. The preview-checkpoint GGUF was used. Safe here because the tokenizer is
**byte-identical** across the two releases: `tokenizer.json`, `tokenizer_config.json`
and `generation_config.json` share the same blob hashes on the Hub. The only
`config.json` differences are four added DSpark keys (`dspark_block_size`,
`dspark_noise_token_id`, `dspark_target_layer_ids`, `dspark_markov_rank`).

### Pre-flight checks

Run before writing anything, as the upstream README requires:

```text
--dspark-manifest   dspark_stages=3   unknown_dspark_tensors=0
--dry-run           n_tensors=1328    type_changes=0    97 591 747 168 bytes
--compare-tensor    blk.0.attn_q_a.weight → bytes match, hashes differ
```

`type_changes=0` proves the recipe was reproduced exactly from the template. The
`--compare-tensor` mismatch is the *expected* result and the whole point of the check:
identical byte counts prove the shape and target type are right, differing hashes prove
the new `0731` weights are actually being read and not the template's.

`0731` also carries **4 705 `mtp.*` tensors** for the DSpark speculative decoding module
that the preview template knows nothing about. The dry-run confirms they're correctly
excluded from the main model (1328 tensors out). Convert them separately with
`--dspark-support` if you want `--dspark`.

## Does the imatrix actually help?

**Honest answer: not measurably, on general text.** Three runs, each widening the scored
window on the same held-out corpus:

| ctx | tokens scored | ppl plain | ppl imatrix | delta | significance |
|---|---|---|---|---|---|
| 512 | 480 | 5.580490 | 5.250970 | −5.90 % | 0.9 σ |
| 8192 | 8160 | 5.020238 | 4.847609 | −3.44 % | 2.1 σ |
| **32768** | **32736** | **4.848056** | **4.782034** | **−1.36 %** | **1.7 σ, p ≈ 0.098** |

The gap **shrinks as the sample grows**: −5.90 → −3.44 → −1.36 %. That is the signature
of an effect collapsing toward zero, not of a real one being measured more precisely. At
32k tokens it is not significant at the conventional threshold.

So the imatrix build is **not demonstrably better on Wikipedia prose**. It is not worse
either. If you were hoping for a number that justifies picking one file over the other on
general text, this measurement does not provide it, and I am not going to dress it up.

Method, so it can be repeated:

```sh
./ds4 -m <build>.gguf --perplexity-file wikitext2_test_300kb.txt -c 32768
```

- **Corpus**: wikitext-2 **raw test** split, first 300 KB. Deliberately *not*
  `rendered_prompts.txt`: the imatrix was collected on that, measuring there would be
  circular. Verified: zero overlap with the calibration corpus.
- Identical file, context length and tokenizer on both runs, 32736 tokens scored out of
  68186 read. Teacher-forced NLL, no sampling, so no seed to fix.
- Significance uses a conservative per-token σ ≈ 1.5 and treats the two runs as
  independent. They are actually **paired** (same corpus, same tokens, same order), so a
  proper paired test on per-token log-probs would have more power. `ds4 --perplexity-file`
  only returns the aggregate NLL, so that test could not be run here. The figures above
  are therefore a lower bound on significance, not an upper one.

⚠️ **Not comparable to published wikitext perplexities.** `ds4 --perplexity-file` scores a
single context window, not a sliding window over the whole test set like
`llama-perplexity`. These numbers are valid *against each other* and nothing else, and
this GGUF cannot be loaded by `llama-perplexity` at all.

**What this does not test.** The imatrix targets the routed-expert distribution seen in
agent and code work, and it records the routed SwiGLU row *after route weighting*. General
prose exercises a different mix of experts. A held-out code/agent slice would be the
measurement that matters for the case this build is actually meant for; it has not been
run yet. Until it is, treat the two builds as equivalent and pick either.

## Why the imatrix buys so little: reading the .dat itself

The perplexity result above says the imatrix changes almost nothing on general text. Rather
than leave that as a shrug, I opened the `.dat` and looked at what it actually contains.
All of the following is reproducible from the published file with ~30 lines of numpy.

### The imatrix does not discriminate on two thirds of the tensors

For each expert, the imatrix holds one importance value per input column. If that vector is
flat, the quantizer has no information to act on. Measuring its effective width
(`exp` of the Shannon entropy, so it is scale-free):

| tensor | columns | median effective width | reading |
|---|---|---|---|
| `ffn_gate_exps` | 4096 | 3487 – 3922 | **85–96 % flat, near-useless** |
| `ffn_down_exps` | 2048 | 370 – 1295 | 18–63 %, genuinely structured |

This follows from how the collector is built. Upstream documents it plainly: for gate/up it
records *"the squared FFN-normalized input activation"*, the input to the FFN block, which
is **identical for all 256 experts because it precedes routing**, while for `down` it
records the routed SwiGLU row *after route weighting*, i.e. what each expert actually saw.

The consequence is arithmetic. `gate` and `up` are **two thirds** of the routed-expert
weights and they are the ones crushed hardest, to `IQ2_XXS`. That is exactly where the
imatrix has nothing to say. The remaining third, `down`, is the only place it informs, and
it is already the best-protected, at `Q2_K`.

**So the imatrix carries usable signal on one third of the weights, and that third is the
one quantized least aggressively.** A small measured gain is the expected outcome, not an
anomaly. This is a property of the collector, not of the checkpoint or the recipe.

### Per-expert importance is dominated by single outliers

Summing importance per expert on `ffn_down_exps`, the deep layers look extraordinarily
concentrated, until you look at why:

| layer | dominant expert | share of importance | max / median |
|---|---|---|---|
| 30 | #176 | 99.9 % | 3.0e5 |
| 29 | #126 | 99.8 % | 2.9e5 |
| 36 | #104 | 99.8 % | 2.0e5 |
| 34 | #33 | 99.8 % | 1.7e5 |

One expert with 300 000× the median. Remove it and the distribution is unremarkable:
effective width climbs back to 150–171 out of 255. This is the *massive activations*
pattern, not specialization. It does not mislead the quantizer, which slices each expert's
own segment, but it does mean per-expert importance sums are not a usable basis for
allocating bits.

### There is no expert "highway"

If some experts mattered everywhere, they could be protected globally. They do not. Overlap
of the top-26 experts between layers runs 4–19 %, against **10.2 % expected from chance**.
Experts never appearing in any top-26 across the 17 deep layers: **43 of 256, versus 41.5
predicted by chance**. Adjacent layers 29 and 30 share exactly one expert out of 26.

Specialization is strictly per-layer; the router redistributes completely at every level.
Good news about the model (its 256 experts are genuinely used), and bad news for anyone
hoping to find a global subset worth extra bits.

### What this implies

- **The `37-42 → Q4_K` boundary is a convention, not a measurement.** Nothing in the imatrix
  singles those six layers out. Note that cross-layer importance totals are *not* comparable
  (activation scale grows with depth), so this analysis cannot propose a better boundary
  either; it can only say the current one is not derived from data.
- **Finer allocation is blocked by the format, not the tooling.** A GGUF tensor carries one
  type, written once; all 256 experts of a layer live in a single tensor. Per-expert types
  would require changing the format *and* the Metal dequantization kernels, not just the
  quantizer.
- **The tractable improvement is upstream, in the collector**: recording, for gate/up, the
  activation each expert actually receives *after* routing, instead of the shared pre-routing
  input. That would give the quantizer real per-expert signal on two thirds of the weights,
  where today it has almost none.

Complementary to this, [nazeshinjite](https://huggingface.co/nazeshinjite/DeepSeek-V4-Flash-0731-ds4-GGUF)
measured weight drift between the preview and `0731` checkpoints and found correlation above
0.99 at every depth, with three quarters of 4-bit values bit-identical: a continued-training
refresh rather than a retrain. Two independent routes to the same conclusion: there was
little for a freshly collected imatrix to recover.

## Performance

Apple M3 Max, 128 GB. Single-run Metal CLI, `--ctx 32768 --nothink --temp 0 -n 256`,
short prompt, the same conditions upstream uses for its own table:

| | imatrix build |
|---|---|
| prefill | **45.08 t/s** |
| generation | **27.01 t/s** |

For reference, upstream reports 58.52 / 26.68 t/s for the **uniform q2** preview build on
the same machine class. Generation matches; prefill is lower here because this is the
mixed 2+4 bit recipe: layers 37-42 stay at `Q4_K`, so there are more bytes to move per
token during prompt processing.

Memory, measured at load:

| | |
|---|---|
| resident model | 90.88 GiB |
| KV @ 32k ctx | 0.61 GiB (raw 0.36 + compressed 0.25) |
| total planned | **91.74 GiB** |
| model residency | ~35-40 s from SSD |

Build cost on the same machine, for anyone reproducing it:

| step | time |
|---|---|
| quantization pass (`--threads 16`, CPU-bound, GPU idle) | ~1 h 05 |
| imatrix collection, 1.5M tokens (Metal, GPU-bound) | ~4 h |
| second quantization pass with `--imatrix` | ~1 h 10 |

## Caveats

- **Community build, not endorsed** by antirez or DeepSeek.
- **Not scored** against the official DeepSeek continuation vectors that antirez uses as
  a release gate (`tests/test-vectors`). The perplexity delta above is a same-engine A/B
  between these two files, not an absolute quality claim against the FP4 original.
- The imatrix was collected on the 2-bit build, not a Q4 one. See above for why that
  is a reasonable compromise, but it *is* a compromise.

## Credits

- **[DeepSeek](https://huggingface.co/deepseek-ai)**: base model and weights (MIT)
- **[antirez](https://github.com/antirez)** and the DwarfStar contributors: recipe,
  quantizer, imatrix pipeline, inference engine. This repo is a straight application
  of their work to a newer checkpoint.
- **llama.cpp / GGML**: quant formats and the groundwork all of the above rests on

## License

MIT, following the base model's release terms.
