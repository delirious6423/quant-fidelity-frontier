---
library_name: cortiq
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- cmf
- cortiq
- moe
- 4-bit
- 2-bit
---

# DeepSeek-V4-Flash-0731 — CMF

[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(304B, 43 layers, 256 routed experts top-6 + shared) in the
[CMF container](https://github.com/infosave2007/cmf), running on `cortiq`.

**Requires cortiq 0.5.47 or newer.** No configuration: the runtime detects
the GPU and VRAM itself and picks the fast path, including speculative
decode. Older versions needed a dozen environment variables for the same
result — they are no longer necessary.

| variant | size | expert planes | folder |
|---|---|---|---|
| `q2tp` | 112 GB | 2-bit gate/up, 4-bit down | `parts-q2tp-v3/` |
| `q4tp` | 158 GB | 4-bit throughout | `parts-q4tp/` |

(`parts-q2tp-v2` is superseded by `-v3`: same trunk byte for byte, the
draft stack requantized for speculative decode.)

## Install the runtime

```sh
cargo install cortiq-cli   # or a prebuilt binary: github.com/infosave2007/cmf/releases
```

## Download

Files ship in slices that concatenate back byte for byte:

```bash
huggingface-cli download infosave/DeepSeek-V4-Flash-0731-cmf \
  --include 'parts-q2tp-v3/part_*' --local-dir .
cat parts-q2tp-v3/part_* > dsv4-flash-q2tp.cmf

# speculative-decode routing tally — keep it next to the model file
huggingface-cli download infosave/DeepSeek-V4-Flash-0731-cmf \
  dsv4-flash-q2tp.cmf.dspark.tsv --local-dir .
```

The `.dspark.tsv` sidecar tells the draft which experts to keep in VRAM.
Without it everything still runs; speculative acceptance is just lower.

## Run

Get `cortiq` from the [releases](https://github.com/infosave2007/cmf/releases)
or `cargo build --release --features gpu`. Then:

```bash
# one-shot
cortiq run dsv4-flash-q2tp.cmf \
  --prompt $'<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>' \
  --max-tokens 8

# server (OpenAI-compatible)
cortiq serve dsv4-flash-q2tp.cmf --port 8080

# benchmark (greedy, the numbers below)
cortiq bench dsv4-flash-q2tp.cmf --core --tokens 128
```

Prompt format: BOS is required, and ordinary chat closes the reasoning
block **in the prompt** — end with `</think>`. For reasoning mode end with
`<think>` instead. Without BOS the output is noise; that is
out-of-distribution input, not a broken model.

```
<｜begin▁of▁sentence｜><｜User｜>your question<｜Assistant｜></think>
```

**Generating code: use `--temperature 0.6 --rep-penalty 1.0`.** Measured
head to head on a ~6000-token three.js scene: that setting produced a
complete page that ran on the first try with zero console errors. The
default repetition penalty (1.1) is harmful for code — it penalizes the
repeats code is made of (identifiers, digits): at low temperature the
model argues itself into the token cap, at high it writes a tutorial
with placeholders instead of a file. Greedy is close but left a handful
of one-character typos.

## Measured speed

`cortiq bench --core --tokens 128`, `q2tp`, one RTX PRO 6000 Blackwell,
cortiq 0.5.46, no environment variables. The VRAM budget is auto-detected;
the smaller points below were measured on the same card by capping it with
`CMF_GPU_VRAM_MB` to what a card of that size would auto-detect.

| VRAM | tok/s | mode |
|---|---|---|
| 96 GB | **40.2** | speculative decode, 63% acceptance |
| 64 GB | 6.9 | GPU walk, cold experts from mmap |
| 32 GB | 3.9 | GPU walk |
| 16 GB | 3.2 | GPU walk |
| CPU only (48 cores) | 2.7 | host |

Speculation engages automatically when the budget packs the trunk far
enough for the draft's capture layers; below that it declines and the
whole budget goes to expert packs. Plan on system RAM at least the file
size — the model is memory-mapped.

## Quality gate

```bash
cortiq ppl dsv4-flash-q2tp.cmf --file docs/ppl_nat.txt --tokens 128
# PPL = 4.578
```

4.578 on the repository's reference text, identical to the last digit on
CPU and GPU and at every VRAM budget from 16 to 96 GB. Any change that
moves it is a defect. (`cortiq ppl` pins the strict kernels itself;
generation uses the fast ones.)

Known trade-off of `q2tp`: it reasons and writes like `q4tp` but is
weaker at arithmetic (2+2 can come out wrong; `q4tp` answers 4). Do not
restrict routing with `CMF_MOE_MASK` — generation degrades within a few
dozen tokens.

## Provenance

Weights derive from DeepSeek's release and remain under its licence. The
CMF container and the cortiq runtime are Apache-2.0 (see the repository's
LICENSE and PATENTS.md).
