---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx-vlm
pipeline_tag: image-text-to-text
tags:
- mlx
- moe
- pruning
- reap
- qwen
---

# Qwen3.8-Flash-Next REAP-288 (MLX, 4-bit)

| | Disk | Resident memory | HumanEval pass@1 | Decode (M4 Max) |
| --- | --- | --- | --- | --- |
| Base Q4 (512 experts) | 98 GB | 97 GB | 93.9% | 25 tok/s |
| **This build (288 experts)** | **68 GB (-31%)** | **68 GB, or 39 GB streamed** | **91.5%** | **28-65 tok/s** |

Qwen3.8-Flash-Next with 288 of 512 experts per MoE layer, pruned with REAP
saliency calibrated on the quantized weights, on the machine that serves them.
It is a 180B-class MoE that runs on a 128 GB Mac for Claude Code-style agentic
workloads and scores 91.5% on HumanEval. Decode is runtime-dependent: ~28 tok/s
on stock mlx-vlm today, and 37 to 65 tok/s on the pmlx engine that ships with
the tiered release (see "Decode speed" for the full breakdown). It supports MTP
speculative decoding out of the box. Resident memory is ~68 GB as shipped, or
~39 GB when the n-gram table is streamed from NVMe (see "Running it").

- 180B-parameter class: 125B main model, 51B n-gram embedding table, 48 layers
  alternating Gated DeltaNet and Qwen sparse attention, each with a 288-expert
  MoE routing top-10
- Runs on a 128 GB Mac at ~68 GB resident on mlx-vlm, or ~39 GB with the n-gram
  table streamed from NVMe on omlx
- Multimodal weights (vision tower) are intact but only text quality has been
  evaluated

## Running it

For the ~39 GB streamed footprint, run under omlx v0.6.4 or newer. omlx offloads
the n-gram table to NVMe automatically when the full model would not fit under
the memory ceiling, so the build serves in about 39 GB resident.

```bash
brew install jundot/omlx/omlx

# Put the model where omlx can discover it
hf download sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --local-dir ~/models/Qwen3.8-Flash-Next-REAP-288-MLX-4bit

# A memory ceiling below the 68 GB full-resident size makes omlx stream the
# n-gram table, landing the model at about 39 GB resident
omlx serve --model-dir ~/models --memory-guard-gb 48
```

The server is OpenAI-compatible at `http://localhost:8000/v1`.

mlx-vlm runs the same build today at about 68 GB resident, with the n-gram table
in memory. It streams to about 39 GB once PR #2045 (external PLE storage) lands
in a release, since this repo already ships the `ple-store.json` manifest and
`ple_storage` config that path reads. mlx-vlm needs `qwen4_exp` MTP support (git
main after 2026-08-27, or any release that includes it):

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git

# One-shot generate
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --prompt "Refactor this function to add input validation." \
  --max-tokens 512

# OpenAI-compatible server
python -m mlx_vlm.server \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit --port 8080
```

Speculative decoding with the model's own MTP head, using the companion
drafter
[sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16):

```bash
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit \
  --draft-model sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16 \
  --draft-kind mtp \
  --prompt "..." --max-tokens 512
```

A note on speculative speed: the drafter's acceptance rate is healthy
(~44-68% depending on sampling), but the net speedup depends on how cheaply
your hardware runs the verification pass. M5-class GPUs report 1.5-2.6x;
on M4 it is roughly break-even. Quality is unaffected either way, since the
target model verifies every drafted token.

## Why 288 experts

We measured an eleven-point pruning ladder and 288 is where quality-per-GB
peaks among usable builds. Aggregate metrics (KL divergence against the stock
model) suggest pruning deeper, but sampled rare-token reliability collapses
below 288: a 256-expert build produces an intact rare name in 1 of 10 seeded
generations where this build scores 9 of 10, a failure mode invisible to
greedy evaluation and unrepairable by giving the experts more bits. The full
study and tooling will be made available soon; the per-layer kept-expert
manifest ships in this repo as `reap_kept_experts.json`, which makes the
prune reproducible from the source conversion.

| Build (experts) | Disk | HumanEval pass@1 |
| --- | --- | --- |
| 512 (stock conversion) | 98 GB | 93.9% |
| 384 | 80 GB | 92.1% |
| 320 | 72 GB | 90.9% |
| **288 (this build)** | **68 GB** | **91.5%** |
| 256 | 65 GB | 88.4% |

All legs ran the same harness on the same machine: 164 HumanEval problems,
unit-test verified, one run per build. Routing width is untouched at the
trained top-10; narrowing it costs far more than it saves (top-6 scores
84.8%, top-4 collapses to 63.4%, for at most +2.5 tok/s).

## Pinned-expert manifests (SSD expert streaming)

The `manifests/` folder ships our measured per-layer saliency selection in the
plain layer-id -> expert-id-list JSON shape that SSD-expert-streaming runtimes
consume directly as a pinned source of truth. Expert ids are stock ids (0-511)
and layer keys are the 48 routed MoE layers (0-47), so the maps pin against the
full 512-expert base checkpoint. Each file is a flat
`{"<layer>": [<expert ids>], ...}` map, ids only, well under the 2 MiB limit.

- `manifests/qwen38-flash-next-512e_saliency_pinned.json` - all 512 experts per
  layer in descending saliency order. Pin a prefix of length K to get the top-K
  experts at any budget; the first 288 ids of each layer are exactly the kept
  set of this build.
- `manifests/qwen38-flash-next-reap-288_kept.json` - the REAP-288 kept set,
  ascending ids (same selection as `reap_kept_experts.json`). The exact
  288-expert case.
- `manifests/qwen38-flash-next-reap-288_kept_saliency_ordered.json` - the same
  288 kept ids reordered descending by saliency, so a shorter prefix is a valid
  smaller-budget selection.

Saliency is the same on-device REAP measurement used to choose this build's kept
experts. The maps match the REAP/omlx manifest shape (a plain layer -> expert-id
map, also accepted under `layers` / `pinned_experts` / `kept_experts` wrappers).

## The 39 GB mode

Per token the model reads only a few hundred bytes of the 51B n-gram table, so
the table does not need to be resident. A runtime that streams it row by row
from NVMe drops resident memory from about 68 GB to about 39 GB, with logits
identical to the in-memory path. omlx v0.6.4 or newer does this today; the
commands are in "Running it" above.

mlx-vlm gains the same capability through PR #2045 ("external PLE storage"),
merged to main on 2026-08-28 and awaiting a tagged release. This repo ships the
two files that feature reads, `ple-store.json` and a `ple_storage` block in
`config.json`, so once #2045 lands in a release `python -m mlx_vlm.server`
streams the table to about 39 GB with no extra steps. Until then, use omlx for
the 39 GB footprint; stock mlx-vlm loads the full table at about 68 GB.

## Decode speed

The ~28 tok/s figure above is stock mlx-vlm today, with the full n-gram table
resident at about 68 GB. omlx serves the same weights at about 39 GB by
streaming that table from NVMe. A separate pure-MLX engine, pmlx, runs these
same 4-bit weights faster and releases alongside the upcoming tiered model. It
is not required to run this build, and nothing here waits on it, but the same
download gets faster when it lands.

| Engine | Resident | Decode (M4 Max) | Available |
| --- | --- | --- | --- |
| stock mlx-vlm, table resident | 68 GB | ~28 tok/s | today |
| omlx, table streamed from NVMe | 39 GB | not separately benchmarked | today (v0.6.4+) |
| pmlx, table streamed from NVMe | 39 GB | ~37 tok/s | with the tiered release |
| pmlx, table in RAM | 73 GB | ~65 peak, ~41 sustained at 4K | with the tiered release |

Same 4-bit build in every row. The only measured decode on a public runtime today
is ~28 tok/s on stock mlx-vlm with the table resident. Streaming the table from
NVMe costs some throughput that we have not separately benchmarked under omlx.

The pmlx rows are not needed to serve the model today. Read the ~65 tok/s figure
carefully: it is the RAM-resident table, not the 39 GB streamed path, and it is a
short-context peak. Sustained pmlx decode at 4K context is about 41 tok/s bare and
about 52 tok/s with speculative decoding. The ~37 tok/s streamed figure is the
same engine reading the n-gram table from NVMe.

## Provenance and what was fixed

- `Qwen/Qwen3.8-Flash-Next`: upstream weights
- [Sawfwair/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit):
  MLX affine 4-bit conversion (group size 64; n-gram table group size 32)
- This build: REAP expert pruning 512 -> 288 per layer, calibrated on-device
  over ~686K tokens of agentic-coding traffic

Two defects of the source conversion are corrected in the weights, so no
loader patches are needed: RMSNorm tensors stored un-centered (+1) are
re-centered to the zero-centered convention the runtime's `(1 + w)` norm
expects, and the n-gram table tensors plus their per-tensor quantization
overrides are renamed `shard_N -> shards.N` to match the runtime module
path. Everything else is byte-identical to the pruned source. Stock-runtime
logits on this build match our patched-runtime reference exactly (max abs
diff 0.0 at the final prefill position).

## Limitations

- Calibration reflects one team's agentic-coding distribution. Retention
  numbers should not be read as general-domain; domains far from code may
  degrade more.
- Single-run evaluations, no confidence intervals. Differences of a point or
  two between neighboring builds are within noise.
- Rare-name sampling reliability is 9/10, not 10/10; pruned models benefit
  from clearing context after a visible garbled name, since a corruption that
  enters the context conditions later turns.
- Vision input is untested after pruning.

## License

Qwen Community License 1.0, inherited from the base model; see `LICENSE`.
