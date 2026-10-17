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

# Qwen3.8-Flash-Next REAP-288 Q8E (MLX, 8-bit experts on a 4-bit backbone)

| | Disk | Resident memory | HumanEval pass@1 |
| --- | --- | --- | --- |
| Base Q4 (512 experts) | 98 GB | 97 GB | 93.9% |
| **This build (288 experts, 8-bit expert tensors)** | **100 GB** | **100 GB, or 70 GB streamed** | **90.9%** |

On stock mlx-vlm the n-gram table is resident and this build is ~100 GB, which
does not leave working room on a 128 GB Mac. The 70 GB figure needs a runtime
that streams the table from NVMe: omlx v0.6.4+ does this today (see below).

Qwen3.8-Flash-Next with 288 of 512 experts per MoE layer, pruned with REAP
saliency calibrated on the quantized weights, and the surviving expert
tensors re-quantized to 8-bit. The backbone (attention, DeltaNet, norms,
embeddings) stays at 4-bit, so all of the extra 32 GB over the plain 4-bit
prune goes to expert precision. Loads on stock mlx-vlm with no patches.

- 180B-parameter class: 125B main model, 51B n-gram embedding table, 48 layers
  alternating Gated DeltaNet and Qwen sparse attention, each with a 288-expert
  MoE routing top-10
- Expert tensors (`gate_proj`/`up_proj`/`down_proj` of every MoE layer) at
  8-bit, group size 64; everything else 4-bit (n-gram table group size 32),
  carried as per-module overrides in the quantization config
- Multimodal weights (vision tower) are intact but only text quality has been
  evaluated

## Where this build sits

Be aware of what the extra bits buy: in our measurements the 8-bit experts
scored at or below the plain 4-bit prune on HumanEval (90.9% vs 91.5%) and
tied it on rare-name sampling reliability (9 of 10). The pruning error
dominates the quantization error at this width, so for most users the
recommended download is the smaller
[sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit).
This build exists for users who want expert-precision headroom anyway — the
one place it led in our testing was a single conventions-adherence style
evaluation that we have not replicated, so we publish it as an option, not a
recommendation.

## Run it

For the streamed ~70 GB footprint, run under omlx v0.6.4 or newer. It offloads
the n-gram table to NVMe automatically when the full model would not fit under
the memory ceiling.

```bash
brew install jundot/omlx/omlx
hf download sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX \
  --local-dir ~/models/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX
# A ceiling below the ~100 GB full-resident size makes omlx stream the table
omlx serve --model-dir ~/models --memory-guard-gb 72
```

The server is OpenAI-compatible at `http://localhost:8000/v1`.

mlx-vlm runs the build today at ~100 GB resident, with the n-gram table in
memory. mlx-vlm streaming lands once PR #2045 ships and we add a Q8E
`ple-store.json` (the manifest is int4-only today; see "The NVMe n-gram mode").
mlx-vlm needs `qwen4_exp` MTP support (git main after 2026-08-27, or any release
that includes it):

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git

# One-shot generate
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX \
  --prompt "Refactor this function to add input validation." \
  --max-tokens 512

# OpenAI-compatible server
python -m mlx_vlm.server \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX --port 8080
```

Speculative decoding with the model's own MTP head, using the companion
drafter
[sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16):

```bash
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-288-Q8E-MLX \
  --draft-model sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16 \
  --draft-kind mtp \
  --prompt "..." --max-tokens 512
```

A note on speculative speed: the drafter's acceptance rate is healthy
(~44-68% depending on sampling), but the net speedup depends on how cheaply
your hardware runs the verification pass. M5-class GPUs report 1.5-2.6x;
on M4 it is roughly break-even. Quality is unaffected either way, since the
target model verifies every drafted token.

## The NVMe n-gram mode

Per token the model reads only a few hundred bytes of the 51B n-gram table, so
the table does not need to be resident. A runtime that streams it from NVMe drops
resident memory from ~100 GB to 70 GB (MLX peak 70.4 GiB during a short greedy
run), with logits identical to the in-memory path.

omlx v0.6.4+ does this today: it offloads the n-gram table automatically when the
full model would not fit under the memory ceiling. The commands are in "Run it"
above.

mlx-vlm gained the same capability through PR #2045 (external PLE storage), merged
to main on 2026-08-28 and awaiting a tagged release. That path reads a
`ple-store.json` manifest, which we currently ship only for the int4 build; a Q8E
manifest is a follow-up. Until then, use omlx for the streamed footprint on this
build. The kept-expert manifest ships here as `reap_kept_experts.json`, which
makes the prune reproducible from the source conversion.

## Provenance and what was fixed

- `Qwen/Qwen3.8-Flash-Next`: upstream weights
- [Sawfwair/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit):
  MLX affine 4-bit conversion (group size 64; n-gram table group size 32)
- This build: REAP expert pruning 512 -> 288 per layer, calibrated on-device
  over ~686K tokens of agentic-coding traffic, with the surviving expert
  tensors re-quantized to 8-bit

Two defects of the source conversion are corrected in the weights, so no
loader patches are needed: RMSNorm tensors stored un-centered (+1) are
re-centered to the zero-centered convention the runtime's `(1 + w)` norm
expects, and the n-gram table tensors plus their per-tensor quantization
overrides are renamed `shard_N -> shards.N` to match the runtime module
path. The 8-bit expert tensors and their per-module quantization overrides
pass through untouched. Stock-runtime logits on this build match our
patched-runtime reference (max abs diff 0.0 at the final prefill
position).

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
