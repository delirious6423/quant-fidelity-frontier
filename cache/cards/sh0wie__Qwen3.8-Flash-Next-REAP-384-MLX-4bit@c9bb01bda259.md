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

# Qwen3.8-Flash-Next REAP-384 (MLX, 4-bit)

| | Disk | Resident memory | HumanEval pass@1 |
| --- | --- | --- | --- |
| Base Q4 (512 experts) | 98 GB | 97 GB | 93.9% |
| **This build (384 experts)** | **80 GB (-18%)** | **51 GB** | **92.1%** |

The resident figure is measured with the n-gram table served from NVMe (see
below); loaded fully in memory this build is ~80 GB resident.

Qwen3.8-Flash-Next with 384 of 512 experts per MoE layer, pruned with REAP
saliency calibrated on the quantized weights, on the machine that serves
them. This is the shallowest cut in our pruning ladder that we publish: at
1.8 points off the stock conversion on HumanEval it is the closest-to-stock
option when your disk budget allows 80 GB. If you want the best
quality-per-GB instead, take
[sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit)
(68 GB, 91.5%). Loads on stock mlx-vlm with no patches.

- 180B-parameter class: 125B main model, 51B n-gram embedding table, 48 layers
  alternating Gated DeltaNet and Qwen sparse attention, each with a 384-expert
  MoE routing top-10
- Affine 4-bit throughout (group size 64; n-gram table group size 32)
- Multimodal weights (vision tower) are intact but only text quality has been
  evaluated

## Run it

Requires mlx-vlm with `qwen4_exp` MTP support (git main after 2026-08-27, or
any release that includes it):

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git
```

Generate:

```bash
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit \
  --prompt "Refactor this function to add input validation." \
  --max-tokens 512
```

Serve (OpenAI-compatible):

```bash
python -m mlx_vlm.server \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit --port 8080
```

Speculative decoding with the model's own MTP head, using the companion
drafter
[sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16):

```bash
python -m mlx_vlm.generate \
  --model sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit \
  --draft-model sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16 \
  --draft-kind mtp \
  --prompt "..." --max-tokens 512
```

A note on speculative speed: the drafter's acceptance rate is healthy
(~44-68% depending on sampling), but the net speedup depends on how cheaply
your hardware runs the verification pass. M5-class GPUs report 1.5-2.6x;
on M4 it is roughly break-even. Quality is unaffected either way, since the
target model verifies every drafted token.

## Where 384 sits on the ladder

We measured an eleven-point pruning ladder; 384 is the runner-up, and the
right choice when you are optimizing for quality rather than footprint:

| Build (experts) | Disk | HumanEval pass@1 |
| --- | --- | --- |
| 512 (stock conversion) | 98 GB | 93.9% |
| **384 (this build)** | **80 GB** | **92.1%** |
| 320 | 72 GB | 90.9% |
| 288 (efficiency pick) | 68 GB | 91.5% |
| 256 | 65 GB | 88.4% |

All legs ran the same harness on the same machine: 164 HumanEval problems,
unit-test verified, one run per build. Routing width is untouched at the
trained top-10. The kept-expert manifest ships in this repo as
`reap_kept_experts.json`, which makes the prune reproducible from the source
conversion.

## The NVMe n-gram mode

Per token the model reads only a few hundred bytes of the 51B n-gram table,
so the table does not need to be resident. A row-granular disk-read patch
(ours, not yet upstream in mlx-vlm) serves it from NVMe with logits
bit-identical to the in-memory path, dropping resident memory from ~80 GB
to 51 GB measured (MLX peak 50.6 GiB during a short greedy run).

## Provenance and what was fixed

- `Qwen/Qwen3.8-Flash-Next`: upstream weights
- [Sawfwair/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit):
  MLX affine 4-bit conversion (group size 64; n-gram table group size 32)
- This build: REAP expert pruning 512 -> 384 per layer, calibrated on-device
  over ~686K tokens of agentic-coding traffic

Two defects of the source conversion are corrected in the weights, so no
loader patches are needed: RMSNorm tensors stored un-centered (+1) are
re-centered to the zero-centered convention the runtime's `(1 + w)` norm
expects, and the n-gram table tensors plus their per-tensor quantization
overrides are renamed `shard_N -> shards.N` to match the runtime module
path. Everything else is byte-identical to the pruned source. Stock-runtime
logits on this build match our patched-runtime reference (max abs diff
0.0 at the final prefill position).

## Limitations

- Calibration reflects one team's agentic-coding distribution. Retention
  numbers should not be read as general-domain; domains far from code may
  degrade more.
- Single-run evaluations, no confidence intervals. Differences of a point or
  two between neighboring builds are within noise.
- Vision input is untested after pruning.

## License

Qwen Community License 1.0, inherited from the base model; see `LICENSE`.
