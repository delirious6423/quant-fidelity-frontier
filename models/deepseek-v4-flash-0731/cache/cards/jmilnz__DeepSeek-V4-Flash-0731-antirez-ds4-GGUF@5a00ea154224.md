---
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
language:
- en
license: mit
library_name: gguf
pipeline_tag: text-generation
tags:
- deepseek-v4
- deepseek-v4-flash
- gguf
- ds4
- imatrix
- mlx
- apple-silicon
- m5-max
- 128gb
- mixture-of-experts
- quantized
---

# DeepSeek-V4-Flash-0731 — Mixed Q2/Q4 GGUF for ds4 (128 GB Macs)

Mixed-precision GGUFs of **DeepSeek-V4-Flash-0731**, built for
[antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) on 128 GB Apple Silicon.
Reproduces the canonical antirez "layers 37-42 Q4 / everything else Q2" recipe,
applied to the 0731 weights.

## The file

**`DeepSeek-V4-Flash-0731-Layers37-42Q4K-mixed-realimatrix-v2.gguf`** — 91 GiB
(97,426,596,736 bytes), quantized with a real activation imatrix.

An earlier build of this same recipe used the quantizer's synthetic
weight-energy fallback instead. It has been removed in favour of this one, which
measures better on every metric tested (see *Quality* below). The two had an
identical tensor-type layout and identical runtime cost — they differed only in
*which* quantization codes were chosen for the routed experts.

## Why this exists

`deepseek-ai/DeepSeek-V4-Flash-0731` ships as ~155 GB of fp8 safetensors — too
large for a 128 GB Mac alongside KV cache and the OS. These files compress the
0731 snapshot to ~91 GB using antirez's mixed-precision strategy, so they drop
straight into an existing ds4 setup.

## Quantization recipe

| Tensors | Quant | Rationale |
|---|---|---|
| Routed MoE experts, layers 37–42 (the "hot" layers) | **Q4_K** | Higher fidelity where it matters most |
| Routed MoE experts, all other layers | **IQ2_XXS** (~2-bit) | The cold bulk, aggressively compressed |
| Down projections | **Q2_K** | 2-bit |
| Attention projections | **Q8_0** | Small, fires every token — keep sharp |
| Shared experts | **Q8_0** | Always-on dense path |
| Output head | **Q8_0** | Critical, small |

Verified layout: 18 × Q4_K + 37 × Q2_K + 74 × IQ2_XXS = 129 routed-expert
tensors across 43 layers.

v2 is built in a **single `deepseek4-quantize` pass**, using `--tensor-type`
overrides to promote layers 37-42 rather than quantizing a separate ~153 GiB Q4
donor and splicing it in:

```sh
args=()
for L in 37 38 39 40 41 42; do
  for T in gate down up; do
    args+=(--tensor-type "blk.$L.ffn_${T}_exps.weight=q4_k")
  done
done

gguf-tools/deepseek4-quantize \
  --hf   /path/to/DeepSeek-V4-Flash-0731 \
  --template DeepSeek-V4-Flash-0731-Q2-imatrix.gguf \
  --imatrix  DeepSeek-V4-Flash-0731-routed-moe-real.dat \
  --out  DeepSeek-V4-Flash-0731-Layers37-42Q4K-mixed-realimatrix-v2.gguf \
  "${args[@]}" --threads 16
```

This produces a byte-identical layout to the splice route (same 1328 tensors,
same 18 type changes) while skipping an entire quantize pass.

## The imatrix, and a trap worth documenting

v2 uses a **real 0731-native routed-MoE imatrix** collected with
`ds4 --imatrix-out` (129 entries = 43 layers × {gate, down, up}, 2752 chunks —
comparable calibration depth to antirez's own reference imatrix). v1 used the
quantizer's synthetic weight-energy fallback.

**Do not pass `--imatrix-strict` with a routed-MoE imatrix.** The first v2 attempt
did, and produced a silently truncated 73 GiB file. The flag means *"fail if a
quantized tensor has no matching imatrix vector"* — but a routed-MoE imatrix by
construction only covers routed experts. It therefore `exit(1)`s on
`token_embd.weight`, the first non-expert tensor in write order, at byte offset
77,922,699,264 (72.6 GiB), killing the process mid-write. This is deterministic:
strict mode can *never* succeed with a routed-MoE imatrix. Plain `--imatrix` is
correct usage — the fallback for non-expert tensors is the intended path, not
silent data loss.

Two related gotchas:

- **Template must match the revision.** Templating off the older `chat-v2` GGUF
  instead of the 0731 Q2 shifts the planned size by ~165 MB.
- **The tool prints decimal bytes; `ls -lh` prints GiB.** A correct file is
  97,426,596,736 B, which `ls -lh` shows as `91G`. A real-imatrix build is also
  exactly **288 bytes larger** than a non-imatrix one — those are the four
  `quantize.imatrix.*` provenance keys, not corruption.

## Quality: real imatrix vs the synthetic build it replaced

Scored with `gguf-tools/quality-testing/score_official` against the **100 tracked
DeepSeek API continuations** in `data/flash`, compared via `compare_scores.py`.
The metric is target-token negative log likelihood — how much probability each
GGUF assigns to the official continuation, token by token. This avoids judging
quality from a single sampled answer.

| Metric | v1 (synthetic) | v2 (real imatrix) |
|---|---|---|
| Mean NLL (lower is better) | 0.564260 | **0.522266** (−7.44%) |
| Cases won (of 100) | 38 | **62** (0 ties) |
| First-token match | 44/100 | **53/100** |
| Avg greedy longest-common-prefix | 4.770 | **5.490** tokens |

All four metrics move the same direction, which is stronger evidence than the
mean alone — a mean shift can come from a couple of outliers, but a 62/38 case
split with better first-token agreement and longer greedy prefixes indicates a
broad improvement.

**Honest caveat on strength of evidence:** the 62/38 split is exact-binomial
p = 0.0105 — significant at p < 0.05, but **not** at p < 0.01, and this is a
single 100-case run. Treat it as a real and consistent improvement, not a
dramatic one. v2 still loses 38 cases; it wins more of them and loses them by
less.

## Hardware / performance

Tested on **Apple M5 Max, 128 GB unified memory**:

- Model maps to ~92.9 GB; loads with comfortable headroom for KV cache
- **~34.7 tokens/sec** generation, ~23–58 tokens/sec prefill (ds4, Metal)
- Native ds4 `--ctx 200000` works

Throughput figures were measured on v1. v2 has an identical tensor-type
histogram, so runtime cost is unchanged — the imatrix affects which codes are
chosen, not how many bits are read per weight.

## Usage

```bash
# ds4 (DwarfStar) — intended runtime
./ds4 -m DeepSeek-V4-Flash-0731-Layers37-42Q4K-mixed-realimatrix-v2.gguf --ctx 200000

# As a drop-in for an existing ds4 install:
ln -sfn /path/to/this/file.gguf ds4flash.gguf
./ds4-server --ctx 200000
```

**Not a general GGUF.** Like all antirez-recipe files, this requires the ds4
engine's specific tensor layout — it will not load in a generic llama.cpp/Ollama
build.

## Comparison vs prior revision (0730)

Basic reasoning (greedy, temp=0) is equivalent on math, coding, and knowledge
prompts. The 0731 finetune shows a more verbose structured "show your work"
style. Across four agentic tasks via `ds4-agent` (file read, CSV filtering, log
scanning, and an implement→write→run→verify loop), 0730 and 0731 were
equivalent — 0731's distinguishing behavior was more transparent self-monitoring
and error recovery rather than flawless execution. A rigorous multi-turn agent
benchmark would be needed to confirm 0731's "enhanced agentic capabilities"
claim; that probe only established parity at the basic-to-intermediate level.

## Known caveats

- **DSpark/MTP speculative decoding**: 0731 adds `dspark_*` config fields. Base
  inference works fully; whether ds4's MTP path exploits 0731's DSpark config is
  unverified. The speculative-decoding speedup may not transfer without engine
  work or a DSpark-support companion GGUF.
- Built and tested on M5 Max only; behavior on other Apple Silicon (M3/M4) is
  expected to be equivalent but not separately validated.
- Quality evaluation is a single 100-case NLL run (see caveat above).

## Acknowledgements

Recipe and tooling (`deepseek4-quantize`, the quality-testing harness, the
mixed-precision strategy) are entirely from
[antirez/ds4](https://github.com/antirez/ds4). This is a community quant applying
that tooling to the 0731 weights, not an official DeepSeek or antirez release.
