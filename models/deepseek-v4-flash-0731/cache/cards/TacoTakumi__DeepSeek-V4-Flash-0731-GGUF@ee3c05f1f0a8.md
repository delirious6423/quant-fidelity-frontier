---
quantized_by: TacoTakumi
pipeline_tag: text-generation
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- imatrix
- deepseek
- moe
---

# DeepSeek-V4-Flash-0731 - expert-only IQ3 custom mixes (GGUF)

Two imatrix GGUFs of `deepseek-ai/DeepSeek-V4-Flash-0731` built on one idea:
**requantize only the 129 routed-expert tensors, and leave every other tensor
at the precision it already had in the source.** Attention, shared experts,
the router, the indexer and the compressors are untouched in both.

They differ in exactly one thing: whether the expert down-projections get one
extra rung of precision. That single change costs 4.03 GiB and buys 0.0243 of
mean KLD.

Requantized from [`bartowski/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF)
(native MXFP4), which is itself a conversion of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
MIT throughout.

Every number on this page was measured locally. The methodology and the raw
tool output are in this repo, so you can check the claims rather than take
them.

## Which file should I download?

**Take `IQ3_XXS-D_IQ3_S` (111.37 GiB) if it fits comfortably. Take
`IQ3_XXS` (107.34 GiB) if you are tight on memory.** That is the whole
decision. Quality goes one way; size and a little speed go the other; and
nothing here should override fit.

| | `IQ3_XXS-D_IQ3_S` | `IQ3_XXS` |
|---|---|---|
| Size | 111.37 GiB | **107.34 GiB** |
| Mean KLD | **0.238633** | 0.262919 |
| Same top-1 | **84.65%** | 83.91% |
| delta-PPL | **+0.536** | +0.641 |
| vs `unsloth` UD-IQ3_S (109.25 GiB) | +2.12 GiB, better KLD | **-1.91 GiB**, better KLD |

Both beat UD-IQ3_S on mean KLD. The smaller one does it while also being
smaller than UD-IQ3_S, which is the case for it existing at all.

**How to tell if you are "tight".** Add up your VRAM and your system RAM,
then budget the file size plus your KV cache plus a few GiB for the runtime
and the OS. If that total lands within a couple of GiB of your combined
memory, take the smaller file. Concretely: a 128 GB machine (say 32 GB VRAM
plus 96 GB RAM) has about 16 GiB of room left after the 111.37 GiB file, and
that is where people start paging.

**But check the next section first if most of your model ends up on the
CPU.** There is a second failure mode that size does not fix, and on some
machines it matters more than fit does.

**When neither is right for you:**

- **You can fit a larger quant entirely in VRAM.** Then fit a larger quant
  entirely in VRAM. Neither of these files is aimed at you.
- **You are chasing tokens per second above all.** Go smaller, not sideways.
  A flat ~2-bit quant of this model is around 80 GiB and will be far faster
  on a spill-bound box, at a quality tier these files are deliberately not
  competing in.
- **You are on a single card.** Neither file fits. This is a multi-GPU or
  big-RAM proposition.
- **Most of your model will run on the CPU, and your CPU is slow with IQ
  quants.** This is the important one and it is easy to miss. Both files put
  all 129 routed-expert tensors in IQ3 formats. On some CPUs the IQ
  dequantization path is far slower than the K-quant or MXFP4 path, and when
  a large fraction of the experts live on the host that cost dominates
  everything else, including file size. If that is your machine, a mix that
  keeps its big tensors in MXFP4 or K-quants will beat both of these no
  matter how the KLD table looks. See [Speed](#speed) for the measurements
  behind this.

**Where speed comes into it.** The smaller file is also modestly faster
where both fit: measured at **+7.9% decode** on the rig below, at every
depth, because it leaves 4.03 GiB less expert weight on the CPU. That is a
real advantage but a small one next to the fit question, and it shrinks
toward nothing on a machine that holds either file entirely in VRAM. Pick on
fit first and quality second; treat the speed as a tiebreaker, not a reason.

## Reading the filenames

Both are `imat` - built with an importance matrix.

`IQ3_XXS-D_IQ3_S-imat`:
- routed experts (`ffn_gate_exps`, `ffn_up_exps`) at **IQ3_XXS**
- `D_IQ3_S`: the **down**-projections (`ffn_down_exps`) raised one rung to **IQ3_S**

`IQ3_XXS-imat`:
- all 129 routed-expert tensors at **IQ3_XXS**, down-projections included

Neither is a stock `IQ3_S` quant. Note that `unsloth/DeepSeek-V4-Flash-0731-GGUF`
ships a `UD-IQ3_S` at a size between the two (109.25 GiB); it is a different
mix built by a different pipeline, and it is scored against both below.

### Full tensor map

| Tensors | `IQ3_XXS-D_IQ3_S` | `IQ3_XXS` | Source of the type |
|---|---|---|---|
| `ffn_down_exps` (43) | IQ3_S | IQ3_XXS | requantized |
| `ffn_gate_exps`, `ffn_up_exps` (86) | IQ3_XXS | IQ3_XXS | requantized |
| attention (`attn_kv`, `attn_q_a/q_b`, `attn_output_a/b`) | Q8_0 | Q8_0 | source |
| shared experts (`ffn_{down,gate,up}_shexp`) | Q8_0 | Q8_0 | source |
| router `ffn_gate_inp`, compressors, `indexer.proj` | BF16 | BF16 | source |
| APE, `hc_attn_fn`, `hc_ffn_fn`, `output_hc_fn` | F32 | F32 | source |
| token embedding, output tensor | BF16 | BF16 | set explicitly |

Census, `IQ3_XXS-D_IQ3_S`: F32 641 / Q8_0 365 / BF16 190 / IQ3_XXS 86 /
IQ3_S 43 / I32 3 = 1328 tensors.

Census, `IQ3_XXS`: F32 641 / Q8_0 365 / BF16 190 / IQ3_XXS 129 / I32 3 =
1328 tensors.

For both, a type-by-type diff of every non-expert tensor against the source
returns zero differences.

## Download

Download all four shards of whichever you pick and point llama.cpp at shard
1; it finds the rest. Do not mix shards between the two.

### `IQ3_XXS-D_IQ3_S` - 4 shards, 119,585,919,200 bytes (111.37 GiB / 119.59 GB)

| File | Size |
|---|---|
| `DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-00001-of-00004.gguf` | 30.00 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-00002-of-00004.gguf` | 29.88 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-00003-of-00004.gguf` | 29.98 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-00004-of-00004.gguf` | 29.73 GB |

```
hf download TacoTakumi/DeepSeek-V4-Flash-0731-GGUF \
  --include "DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-*" \
  --local-dir ./DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S
```

### `IQ3_XXS` - 4 shards, 115,257,397,472 bytes (107.34 GiB / 115.26 GB)

| File | Size |
|---|---|
| `DeepSeek-V4-Flash-0731-IQ3_XXS-imat-00001-of-00004.gguf` | 28.79 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-imat-00002-of-00004.gguf` | 28.77 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-imat-00003-of-00004.gguf` | 28.77 GB |
| `DeepSeek-V4-Flash-0731-IQ3_XXS-imat-00004-of-00004.gguf` | 28.92 GB |

```
hf download TacoTakumi/DeepSeek-V4-Flash-0731-GGUF \
  --include "DeepSeek-V4-Flash-0731-IQ3_XXS-imat-*" \
  --local-dir ./DeepSeek-V4-Flash-0731-IQ3_XXS
```

## Quality

KL divergence against reference logits generated from the MXFP4 source
itself, so each quant is scored against the thing it was made from.

**Axes, identical for all three rows:** `wiki.test.raw` first 150 chunks
(38250 scored positions), `n_ctx` 512, `batch_size` 2048, `n_seq` 4,
`llama-perplexity` v10155, same reference logits file. The eval corpus is
disjoint from the calibration corpus.

| Quant | Size | Mean KLD | Median KLD | 99% KLD | Max KLD | delta-PPL | Same top-1 |
|---|---|---|---|---|---|---|---|
| **`IQ3_XXS-D_IQ3_S`** | 111.37 GiB | **0.238633 +/- 0.003169** | 0.046533 | **3.199** | 12.53 | **+0.536** | **84.65%** |
| **`IQ3_XXS`** | 107.34 GiB | 0.262919 +/- 0.003423 | 0.051265 | 3.310 | 15.34 | +0.641 | 83.91% |
| `unsloth/...-GGUF` UD-IQ3_S | 109.25 GiB | 0.293623 +/- 0.003664 | 0.060894 | 3.682 | **11.13** | +0.685 | 82.78% |

Reference PPL 5.016233. `IQ3_XXS-D_IQ3_S` 5.551907; `IQ3_XXS` 5.657595;
UD-IQ3_S 5.701592.

Raw `llama-perplexity` output for all three runs is in `eval/`.

### Honest caveats

- **UD-IQ3_S wins on Max KLD** (11.13, against 12.53 and 15.34). Its
  worst-case token is less bad than either of ours even though its mean,
  median, 99th percentile, delta-PPL and top-1 agreement are all worse. If
  tail behaviour matters more to you than average fidelity, that is a real
  trade. Note the flat `IQ3_XXS` has the worst Max KLD of the three.
- **Neither file is a free win over UD-IQ3_S.** The larger one is 2.12 GiB
  bigger. The smaller one is 1.91 GiB smaller and still ahead on mean KLD,
  which is the cleaner result of the two, but it is also the weaker quant in
  absolute terms.
- **The same recipe scored much better on the pre-0731 checkpoint** (mean KLD
  0.177073). It did not reproduce here. That looks like a property of the
  0731 weights rather than of this pipeline: unsloth's own UD-IQ3_S moved
  0.2008 -> 0.2936 across the same checkpoint transition. 0731 appears to be
  harder to quantize at roughly 3 bits, across at least two independent
  pipelines.
- Raising `ffn_down_exps` one rung bought 0.0243 of mean KLD for +4.03 GiB.
  That is the entire difference between these two files. Further single-rung
  changes were not explored.

## Chat template and reasoning_effort

Both files embed `chat_template.jinja` from `unsloth/DeepSeek-V4-Flash-0731`
(13698 chars, sha256 `e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645`),
byte-identical to the file in that repo.

Before embedding, it was run against DeepSeek's own four-case encoding
conformance suite: rendering `test_input_{1..4}.json` produces output
byte-identical to `test_output_{1..4}.txt`. The harness is in `template/`.

All three `reasoning_effort` levels were then verified **on a live backend**
via `/apply-template`, not just offline:

- `high` prepends its canonical 476-char prompt at message index 0
- `max` prepends its canonical 526-char prompt
- `low` adds no prefix
- no leak into chat (non-thinking) mode

Both prompt strings were hash-checked against `encoding_dsv4.py` from
DeepSeek. So `reasoning_effort` works with no `--chat-template-file` override.

Worth knowing: the MXFP4 source GGUF embeds a stale pre-0731 template, which
is why these files carry a replacement rather than whatever they inherited.

## Speed

Measured on a 5-GPU rig (2x 3090, 5060 Ti, 2x 4060 Ti, 96 GiB VRAM total)
with CPU expert spill, 5 reps per depth, against the full MXFP4 source served
at the same placement:

| Depth | `IQ3_XXS` | `IQ3_XXS-D_IQ3_S` | MXFP4 source | smaller/larger | vs source |
|---|---|---|---|---|---|
| 0 | **15.02 t/s** | 13.91 t/s | 9.88 t/s | 1.080x | 1.520x |
| 4096 | **14.62 t/s** | 13.57 t/s | 9.69 t/s | 1.078x | 1.509x |
| 16384 | **14.33 t/s** | 13.26 t/s | 9.51 t/s | 1.081x | 1.507x |

Prefill: `IQ3_XXS` 157-159 t/s, `IQ3_XXS-D_IQ3_S` 137-142 t/s, source 84-86
t/s. Per-depth spread across 5 reps was 0.18-0.28 t/s, so the ~8% gap
between the two mixes is outside run-to-run noise.

Both files were measured with identical flags at identical placement; only
the model path differed. The smaller file's advantage tracks its 4.03 GiB of
extra VRAM headroom, which is exactly the amount of expert weight it keeps
off the CPU.

**These ratios will not transfer to your hardware, and they can invert.**
They say these mixes are faster than the 156 GiB source on a box where
everything spills to CPU. They say nothing about a machine that fits either
one entirely in VRAM, where the gap between the two should close.

### The IQ-on-CPU problem, and how it was found

Two users reported no speedup or an outright slowdown. One, on a 5090 (32 GB
VRAM) plus 96 GB DDR5, measured the larger file at about 11 t/s against
15 t/s for UD-IQ3_S - a file only 2.12 GiB smaller.

My first guess was that 2 GiB had pushed them over a memory cliff. **That
guess was wrong, and they disproved it themselves.** They downloaded a third
quant - `bullerwins/DeepSeek-V4-Flash-0731-GGUF` `IQ3_XXS-Experts-Q8_0`,
which is *smaller* than UD-IQ3_S and also uses IQ3_XXS on the large tensors -
and measured 11-12 t/s, the same as mine. A smaller file being just as slow
rules size out. What the two slow files share is IQ3 expert tensors; what the
fast one has is some of its big tensors in MXFP4.

So the mechanism is the **IQ dequantization path on the CPU**, not fit. That
user had already seen it independently, reporting IQ quants far slower than
K-quants for CPU-offloaded MoE on the same machine.

It does not contradict the table above, and the reason is instructive. On my
rig only ~29 GiB of 111 GiB sits on the host - roughly a quarter - so IQ
dequant cost is a minor term and file size dominates, which is why the
smaller file wins by 8%. On a 32 GB card, ~79 GiB is on the host, so the
same cost dominates instead. **The more of the model that runs on your CPU,
the more IQ format choice matters and the less size does.** These files were
built for the first regime. If you are in the second, they are the wrong
tool and no amount of shaving GiB will fix it.

My thanks to the person who ran that comparison rather than taking my
explanation at face value; it is a better result than anything in my own
bench.

## Running it

What we run, via llama.cpp (swap in the other filename as needed):

```
llama-server \
  -m DeepSeek-V4-Flash-0731-IQ3_XXS-D_IQ3_S-imat-00001-of-00004.gguf \
  --flash-attn on --jinja --parallel 1 \
  --ctx-size 131072 --ubatch-size 512 \
  --temp 0.6 --top-p 0.95
```

Placement flags (`--fit`, `-ngl`, `-ts`) are rig-specific and deliberately
omitted. Backend RSS was 34.5 GiB with about 29 GiB of experts on the host
for the larger file.

**Runtime caveat, and it is not specific to these quants.**
DeepSeek-V4-Flash has open SWA / rollback stall issues in llama.cpp. Both
files were quantized with mainline `llama-quantize` v10155 (`1cbfd1988`) but
are served here on a locally patched build carrying DSV4 stall fixes that are
not in mainline. A user has since run the larger file on stock mainline
b10217 successfully, so it loads and generates without the patches, but we
have not tested long-context stall behaviour there ourselves. If you hit
stalls on long or churning contexts, that is the known upstream issue and it
affects every DSV4 GGUF, not these in particular.

## Reproducing the build

`imatrix`, shared by both: `llama-imatrix` over bartowski's
`calibration_datav3.txt`, 124 chunks at `n_ctx` 512, run against the MXFP4
source. Minimum routed-expert coverage 99.22%; 768 importance-matrix entries;
all 129 expert tensors covered. The file is in this repo.

`quantize`: mainline `llama-quantize` v10155 (`1cbfd1988`). About 98 minutes
for `IQ3_XXS-D_IQ3_S`, 107.3 minutes for `IQ3_XXS`.

```
llama-quantize \
  --allow-requantize \
  --imatrix DeepSeek-V4-Flash-0731-MXFP4-cdv3-ctx512.imatrix.gguf \
  --tensor-type-file recipe/<recipe>.types.txt \
  --token-embedding-type bf16 --output-tensor-type bf16 \
  --keep-split \
  --override-kv general.license=str:mit \
  --override-kv general.source.url=str:https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF \
  --override-kv general.base_model.count=int:1 \
  --override-kv general.base_model.0.name=str:DeepSeek-V4-Flash-0731 \
  --override-kv general.base_model.0.organization=str:DeepSeek \
  --override-kv general.base_model.0.repo_url=str:https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 \
  <source-shard-1> <output>.gguf IQ3_XXS
```

Use `recipe/iq3s-down-iq3xxs-gateup-experts.types.txt` for the larger file
and `recipe/iq3xxs-experts.types.txt` for the smaller one. The two recipes
differ by one line.

The template is applied afterwards with
`gguf-py/gguf/scripts/gguf_new_metadata.py --chat-template-file` on shard 1.
It cannot be done during the pass: `--override-kv` caps string values at 127
characters (`llama_model_kv_override.val_str[128]` in `llama.h`), which no
chat template fits inside.

## What is in this repo

| Path | What |
|---|---|
| `DeepSeek-V4-Flash-0731-MXFP4-cdv3-ctx512.imatrix.gguf` | the importance matrix used for both builds |
| `recipe/iq3s-down-iq3xxs-gateup-experts.types.txt` | the `--tensor-type-file` recipe for `IQ3_XXS-D_IQ3_S` |
| `recipe/iq3xxs-experts.types.txt` | the `--tensor-type-file` recipe for `IQ3_XXS` |
| `eval/kld-this-mix.log` | raw KLD output for `IQ3_XXS-D_IQ3_S` |
| `eval/kld-all-iq3xxs-variant.log` | raw KLD output for `IQ3_XXS` |
| `eval/kld-unsloth-ud-iq3s.log` | raw KLD output for UD-IQ3_S, same axes |
| `template/render_conformance.py` | the conformance harness |
| `template/conformance-unsloth.out` | its output for the embedded template |

## Credits

- [`deepseek-ai`](https://huggingface.co/deepseek-ai) for the model
- [`bartowski`](https://huggingface.co/bartowski) for the MXFP4 conversion this
  was built from, and for `calibration_datav3`
- [`unsloth`](https://huggingface.co/unsloth) for the 0731 chat template
  embedded here, and for UD-IQ3_S as a comparison point
- the `llama.cpp` project
- the r/LocalLLaMA readers who tested these on hardware I do not have and
  reported back; the smaller file is here because of them
