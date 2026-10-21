---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek
  - moe
  - reap
  - pruned
  - fp4
  - fp8
  - speculative-decoding
---

# DeepSeek-V4-Flash-0731 REAP128 FP4/FP8

A REAP-pruned build of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
keeping **128 of its 256 routed experts**, in DeepSeek's native FP4/FP8
checkpoint format. It is intended to make the model practical on a single
128 GB DGX Spark while retaining its DSpark draft head.

> **Important quality trade-off:** this is an expert-pruned model, not merely a
> quantized copy. A same-size 2-bit build that retains all 256 experts was
> substantially stronger in the evaluations below. Choose this repository for
> native FP4 execution, lower resident memory, and DSpark speculation—not for
> the best quality per byte.

## What this is

| | |
|---|---:|
| target experts | 128 of 256 |
| DSpark draft-head experts | 128 of 256 |
| active experts per token | 6 |
| on disk | 88.1 GB |
| tensors | 36,989 |

The repository name counts experts rather than a percentage. At this size the
two readings agree — 128 of 256 is half kept and half pruned — but they diverge
for any other count, and `n_routed_experts` in `config.json` is the number a
user can actually check.

This is one capacity profile from one Japanese/English-centered calibration
mixture. It is not a universal language profile and should not be treated as
the recommended expert selection for every task.

## Reproducibility

This checkpoint can be rebuilt from the official one with what is published
here. [`scripts/`](scripts/README.md) contains the actual build, calibration and
evaluation code — the reference REAP path, the byte-copy path that builds any
expert count, the draft-head calibration, the verification passes, and the
harnesses behind every number in [`docs/QUALITY.md`](docs/QUALITY.md).
Those scripts are a snapshot: the canonical copy, with history and issues, is
[`g667300/deepseek-v4-flash-reap-fp4`](https://github.com/g667300/deepseek-v4-flash-reap-fp4) on GitHub.
[`calibration/`](calibration/README.md) contains the inputs and results those
scripts consume: the exact mixture, the 43-layer target saliency, the REAP128
retained sets, and the separately captured MTP selection.

Rebuilding this checkpoint does not require a GPU, a recalibration, or the
568.7 GB BF16 intermediate. Pruning never modifies a surviving expert, so the
whole checkpoint is a byte copy from the official one plus 86 recomputed router
tensors — verified as **17,711 tensors, 0 differing in value** against the full
pipeline's output, in 84 seconds:

```bash
python scripts/build_pruned.py --reference /path/to/DeepSeek-V4-Flash-0731 \
    --retained calibration/reap128-retained-sets.json --out dsv4-reap128
python scripts/carry_mtp.py --src /path/to/DeepSeek-V4-Flash-0731 \
    --dst dsv4-reap128 --score saliency --saliency calibration/mtp-saliency.json
```

Changing the *mixture* is the expensive case, and it is the one that needs the
reference path. See [language and task profiles](calibration/README.md#make-a-language-or-task-profile)
before making a selection for another language or workload.

`calib.pt` itself is deliberately **not** distributed. It is a 3.1 MB sequence
of token IDs derived from third-party datasets and can be decoded with the
tokenizer. `calibration/README.md` gives the exact regeneration configuration
and the SHA-256 of the input used for this release.

The published `target-saliency.json` is one combined default mixture. It has no
per-source contribution split, so it **must not** be reweighted as though it
were a French-, Hindi-, or domain-specific profile. Create a source-labelled
calibration run for that purpose.

## Other builds worth comparing against

| build | target experts | draft head | size | tensors | tested utilization |
|---|---:|---:|---:|---:|---:|
| **this build** | 128 / 256 | 128 | 88.1 GB | 36,989 | 0.75 |
| [REAP152-FP4](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP152-FP4) | 152 / 256 | 128 | 101.9 GB | 43,181 | 0.85 |

[**REAP152-FP4**](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP152-FP4) comes from the same calibration and the same saliency
file — the top 152 experts of each layer instead of the top 128, with the same
calibrated 128-expert draft head. On the same 205-question generative MMLU
sample it scored **58.05%** against this build's **51.22%**, while the small code
evaluations were mixed rather than consistently better. Choose it when roughly
14 GB more storage and a tighter runtime memory budget are acceptable; choose
this one when KV capacity and host headroom are the binding constraints.

**The comparison that matters more is against an unpruned model.** On that same
MMLU sample the 256-expert 2-bit `DeepSeek-V4-Flash-UD-IQ2_M`, which occupies
almost exactly the same 85 GB, scored **84.39%**. Read
[Quality and benchmarks](docs/QUALITY.md) before choosing any of the three.

## Format

| component | representation |
|---|---|
| routed expert weights | FP4 E2M1, 32-element blocks, E8M0 scales |
| other quantized weights | FP8 E4M3, 128 x 128 blocks, E8M0 scales |
| target MoE | 43 layers, 6 active experts per token |
| draft head | 3 MTP blocks, 128 experts each, live-calibrated REAP selection |

The tensor names and quantization layout are the native DeepSeek format. This
is not a GGUF checkpoint.

## Download

```bash
hf download noooop/DeepSeek-V4-Flash-REAP128-FP4 --local-dir dsv4-reap128
```

That is 80 files (~88 GB), including the documentation, the vLLM overlays and
`scripts/`. For the weights alone:

```bash
hf download noooop/DeepSeek-V4-Flash-REAP128-FP4 --local-dir dsv4-reap128 \
    --exclude "docs/*" --exclude "scripts/*" --exclude "patches/*" \
    --exclude "calibration/*"
```

which leaves 33 files: the 23 shards, the shard index, `config.json`,
`generation_config.json`, both tokenizer files, this card, the licences and
`SHA256SUMS`. Add `--dry-run` to list what a command would fetch without
fetching it.

Verify the transfer with the manifest that came with it:

```bash
cd dsv4-reap128 && sha256sum -c SHA256SUMS
```

Two practical notes. **`--include`/`--exclude` patterns must each carry their own
flag**: a bare second pattern is parsed as a filename and the whole filter is
then ignored, with only a warning. And there is no need to download anything up
front — vLLM fetches the model itself:

```bash
vllm serve noooop/DeepSeek-V4-Flash-REAP128-FP4 ...
```

## Serving

The validated backend is vLLM 0.25.1 on a DGX Spark (GB10, SM121). Plain,
non-speculative serving of this build works on the tested stock image:

```bash
vllm serve noooop/DeepSeek-V4-Flash-REAP128-FP4 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8_ds_mla
```

`--max-model-len 65536` is the recommended setting: 64K is both the longest
context RULER was run at and the model's native window before YaRN
interpolation. Measured on one Spark in the tighter speculative configuration it
leaves 279,203 KV tokens and 4.26 concurrent full-length requests. Longer
contexts start — the checkpoint
declares 1,048,576 positions and a 256K server came up — but nothing above 64K
has been measured for quality. See
[context length](docs/SERVING.md#context-length).

DSpark speculative decoding needs:

```text
method: dspark
num_speculative_tokens: 5
kv-cache-dtype: fp8_ds_mla
```

It also needs the included compatibility overlays on the tested SM120/SM121
software stack. Stock vLLM 0.25.1 cannot dispatch the sparse-MLA decode path
with its 256-token allocation pages. Do not use vLLM's `mtp` method: V4-Flash's
draft-head layout is implemented by `dspark`.

See [Serving on DGX Spark](docs/SERVING.md) for the exact patch matrix and
memory cautions.

### Or run it under llama.cpp, with no patches at all

Current upstream llama.cpp implements this model (`deepseek4`) and its DSpark
draft head (`--spec-type draft-dspark`), so none of the overlays in
[`patches/`](patches) are involved. It reads GGUF, so convert once — the target
and the draft head are two separate exports:

```bash
python convert_hf_to_gguf.py --no-mtp --outfile reap128.gguf ./dsv4-reap128
python convert_hf_to_gguf.py --dspark --target-model-dir ./dsv4-reap128 \
    --outfile dspark-128.gguf ./dsv4-reap128

./build/bin/llama-server -m reap128.gguf -md dspark-128.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 \
  -c 65536 -ngl 999 --host 0.0.0.0 --port 8080
```

The target converts to 82.7 GB (FP4 experts repacked to MXFP4, FP8 attention to
Q8_0) and the shipped draft head to 5.8 GB. Build llama.cpp with
`-DCMAKE_CUDA_ARCHITECTURES=121` for GB10, and keep `--spec-draft-n-max 5`: the
default is 3, and 5 is this checkpoint's own `dspark_block_size`.

**This is where the drafter does best.** The same weights accept 63.1% of drafts
under llama.cpp against 53.2% under vLLM — 3.19 accepted drafts per step against
2.66 — for 26.3 tok/s against vLLM's 28.2, since vLLM's individual steps are
faster. Without a drafter the same file serves at 15.4 tok/s. Full recipe,
including the unpruned 256-expert draft head and the KV-cache options, in
[Serving on DGX Spark](docs/SERVING.md#serving-with-llamacpp).

## Measured performance

One DGX Spark, greedy decoding:

| model | engine | resident weights | no speculation | with DSpark |
|---|---|---:|---:|---:|
| REAP128 FP4/FP8 (this build) | vLLM | 77.4 GiB target / 82.7 GiB with drafter | 16.5 tok/s | 28.2 tok/s |
| the same build converted to GGUF | llama.cpp | 82.7 GB + 5.8 GB drafter | 15.4 tok/s | 26.3 tok/s |
| unpruned IQ2_M GGUF | llama.cpp | 84.5 GiB + 10.9 GB drafter | 19.0 tok/s | 35.4 tok/s |

The old 7.1 tok/s IQ2_M result was invalid: it came from a mismatched llama.cpp
implementation and the wrong CUDA architecture target. With current upstream
llama.cpp built for SM121, the 2-bit control is level with this checkpoint
without speculation. DSpark is the measured throughput advantage.

The speculative figures are warm four-prompt runs; the 512-token benchmark puts
this build at 24.0 tok/s. Throughput depends heavily on prompt shape, context,
software build — and on whether the first request after a load is counted, which
costs vLLM up to 1.8x.

### The 2-bit build speculates too, and it is faster — measured here

`unavailable` was a property of the llama.cpp build in use, not of the 2-bit
format. Current llama.cpp implements DSpark (`--spec-type draft-dspark`) and its
converter extracts the draft head from the official checkpoint
(`convert_hf_to_gguf.py --dspark`). Measured on one DGX Spark, same four prompts,
same harness, 8,192-token context, greedy, draft width 5, **warm** (see the two
notes below — both cost more than the differences being measured):

**Same engine (llama.cpp), so the models are comparable:**

| model | draft head | acceptance | accepted drafts / step | decode |
|---|---:|---:|---:|---:|
| 2-bit IQ2_M, 256 experts | 128 | 62.9% | 3.22 | **36.1 tok/s** |
| 2-bit IQ2_M, 256 experts | 256 | 60.6% | 3.05 | 35.4 tok/s |
| **this build (REAP128)** | 128 (shipped) | 63.1% | 3.19 | 26.3 tok/s |
| REAP128, unpruned head | 256 | 66.4% | 3.38 | 27.8 tok/s |
| REAP152 (sibling) | 128 (shipped) | 44.2% | 2.44 | 20.9 tok/s |

The unpruned 2-bit build is **1.37x faster** with matched engine, harness and
draft width, at acceptance within a few points. It also scored 84.39% against
this build's 51.22% on the generative MMLU sample. At this footprint, on this
hardware, pruning is not buying speed either. What this checkpoint offers is
native FP4 execution under vLLM, a smaller resident footprint, and an
OpenAI-compatible vLLM deployment path — not throughput.

**vLLM does not get the same speculation out of the same weights.** Serving
REAP128 under vLLM 0.25.1 with the patched sparse-MLA backend, against the same
prompts and the same draft width, its drafter lands **53.2%** where llama.cpp
lands 63.1% on weights converted from the same checkpoint — 2.66 accepted drafts
per step against 3.19. The unpruned head shows the same shortfall (60.0% against
66.4%). vLLM's individual steps are faster, so end-to-end throughput comes out
similar (28.2 against 26.3 tok/s), but roughly a fifth of the drafting is being
left on the table. **The cause is not known.** KV quantization is ruled out:
llama.cpp with `q8_0` KV reproduces its own f16 acceptance *bit for bit* (570 of
903 either way), and vLLM's DSv4 path refuses anything but fp8, so the two cannot
be matched there. Numbers measured under vLLM should be read as a floor.

**Two ways to get this comparison wrong**, both of which happened here first:

* **`mean len` is not accepted-drafts-per-step.** llama.cpp counts the target's
  own bonus token in it; vLLM's counters do not. Comparing them directly turns a
  19% gap into a fictitious 57% one.
* **Discard the first run after a load.** vLLM's first request is 1.66-1.80x
  slower than steady state while its speculative counters stay bit-identical;
  llama.cpp shows 1.04x of the same effect. Every number here is a warm run.

The comparison is still not perfectly clean: different quantization on both sides
(FP4 and FP8 under vLLM, MXFP4 and Q8_0 in the GGUF), and each engine's own
counters.

## What was pruned

The target experts were selected independently per layer with the reference
REAP score

$$
S_j \;=\; \frac{1}{|T_j|} \sum_{t \in T_j} g_j(t) \, \bigl\lVert f_j(t) \bigr\rVert_2
$$

where \\(T_j\\) is the set of calibration tokens the router sent to expert
\\(j\\), \\(g_j(t)\\) is the gate weight it gave that token and \\(f_j(t)\\) is
the expert's output for it. The score is therefore a per-expert mean over the
tokens that actually reached it, taken over 512 x 2048 calibration tokens. The
mixture was 35% Japanese, 35% English, 26% code, and 4% Chinese. This mixture
matters: multilingual losses were largest for languages absent from
calibration.

The MTP experts were calibrated separately from the live draft head. Reusing a
target layer's expert set is inferior, and selecting the 64 most-used draft
experts broke generation completely (0 accepted tokens). Both this build and
the 152-expert one therefore retain the calibrated top 128 MTP experts.

See [Build, provenance, and validation](docs/PROVENANCE.md) for the full path,
and [Calibration recipe and recorded selections](calibration/README.md) for
the reproducible calibration configuration and the exact selections used here.

## Limitations

- Substantial capability loss is measured against an unpruned same-base control.
- The calibration distribution is narrow and creates language/domain bias.
- Speculative serving on the tested SM121 stack requires version-specific vLLM
  overlays; future vLLM releases may make them unnecessary or incompatible.
- Only single-DGX-Spark serving has been validated.

## License

| what | terms |
|---|---|
| model weights, config, tokenizer | MIT, inherited from `deepseek-ai/DeepSeek-V4-Flash-0731`; see [`LICENSE`](LICENSE) |
| `scripts/`, `calibration/`, `docs/`, this card | MIT, see [`LICENSE-CODE`](LICENSE-CODE) |
| `patches/*.py` | Apache-2.0, modified copies of vLLM 0.25.1 files; each keeps its upstream SPDX header and carries a notice of what was changed |
| the sparse-MLA page re-view inside `patches/flashinfer_sparse.py` | derived from `anemll/dspark-vllm-gx10` (MIT), attributed at its use site |

No calibration or evaluation corpus is redistributed here, so no dataset licence
attaches to these files. The calibration sources are named with their own
licences — two of them share-alike — in
[`calibration/README.md`](calibration/README.md#calibration-sources-and-their-licences),
which is also why `calib.pt` is not published.
