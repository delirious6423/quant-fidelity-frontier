---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- rtx-pro-6000
- blackwell
- sm120
- tensor-parallel
---

# DeepSeek-V4.1-Flash-1017: EXL3 3.0 bpw for 2× RTX PRO 6000 Blackwell (diffbot layout)

This is [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), the October 17 update of
DeepSeek-V4.1-Flash, packed in the layout of
[diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
(the "diffbot pack"). It replaces the weights of that pack and runs on diffbot's recipe unchanged, at the same speed and
memory footprint: two 96 GB cards, about 230 GiB of host RAM, 1M-token context.

**This is a repack, not a new quantization of the routed experts.** The routed experts are
[PixelML's uncalibrated EXL3 3 bpw quant](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw). What this
repo adds is the rest of a servable pack for the 1017 weights: the DSpark drafter and the Engram tables, re-encoded from
DeepSeek's 1017 files with diffbot's scripts, plus measurements.

| | |
|---|---|
| Size | 332 GB in 48 safetensors shards |
| Serving stack | diffbot's recipe at revision [`a1cf1bd2`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000/tree/a1cf1bd227a7) (vLLM + vllm-exl3 + ExLlamaV3, TP2) |
| Decode, 1 stream at 2K | 105 tok/s; 110 tok/s on an agentic turn |
| Prefill at 46K | ~8,600 tok/s with 1 stream, 15,770 total with 4 streams |
| Long context | needle found at 100K, 250K and 448K tokens |
| Quality | GSM8K-200 98.5 %, HumanEval 95.1 % (the diffbot pack on the same machine: 98.0 %, 95.7 %). The experts are **uncalibrated**; no KL measurement yet |

## What's in this repo

| Files | Contents | Source |
|---|---|---|
| `model-00001` to `model-00043` | Routed experts in EXL3 3 bpw (MUL1 codebook, K=3, **uncalibrated**); everything else in DeepSeek's FP8 | [PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw) @ `3d095b72`, shards 1-43, unchanged |
| `model-00044` to `model-00046` | DSpark drafter. Its routed experts are re-encoded from MXFP4 to EXL3 3-bit (MCG codebook, uncalibrated), as in the diffbot pack; its other tensors are DeepSeek's | Made here from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10`, shards 44-46, with diffbot's `quantize_mtp_exl3.py` |
| `model-00047`, `model-00048` | Engram tables in int4 with fp16 scales per 32 values (`int4_fp16s32`), 111 GB instead of 203 GB of FP8 | Made here from the official 1017 shards 47-48 with diffbot's `convert_engram_int4.py` |
| `config.json` | The diffbot pack's config, with only `quantization_config.source` changed. Its model fields are identical to DeepSeek's 1017 config | |
| `model.safetensors.index.json` | Rebuilt from the shard headers: 190,549 tensors, the same names as the diffbot pack | |
| `encoding/`, `tokenizer*`, `LICENSE` | DeepSeek's, from the 1017 release. Prompts are formatted by `encoding/` (there is no Jinja chat template) | |
| `build/` | The assembly script and build notes | |
| `results/` | Raw benchmark and quality logs from the runs below | |

Why a repack is needed: 1017 has the same architecture and tensor names as V4.1-Flash, but DeepSeek retrained every
weight, so **all 48 files changed, including the Engram tables and the drafter**. The diffbot pack's files can't be mixed
with 1017 files. PixelML's repo has the 1017 experts but no Engram tables and an MXFP4 drafter.

## Quick start

You need two sm_120 GPUs with 96 GB each, Docker with the NVIDIA container toolkit, about 250 GiB of free host RAM and
340 GB of disk (the same as the diffbot pack).

```bash
# 1. This pack
hf download Nateemerson/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw

# 2. diffbot's recipe, at the revision tested here (the recipe only, not the old weights)
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --revision a1cf1bd227a7 \
  --include "recipe/*" --local-dir ~/dsv41-recipe
cd ~/dsv41-recipe/recipe
bash docker/build.sh       # serving image dsv41-flash-exl3-3bpw-sm120
bash kernels/build.sh      # the xmoe and xmoe_do MoE kernels

# 3. Serve: OpenAI-compatible API on :8000, model name deepseek-v4.1-flash
PACK=~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw PLAN=plan-blend30-dh.json API_KEY=change-me bash serve/serve.sh
```

- `PACK` must point at this repo's folder, because the recipe isn't inside it.
- `PLAN=plan-blend30-dh.json` is what every number below used. The recipe's default is `plan-agent30-dh.json`.
- On our host we also set `NCCL_P2P_DISABLE=1` (GPU-to-GPU P2P hangs there), `CONTAINER_MEM=236g` (the host has 249 GiB;
  the default is 260g) and `EXTRA_ARGS="--host 127.0.0.1"`. Leave P2P on if it works on yours.
- diffbot's newer recipe revision (`6062be5f`, rebased onto a newer vLLM nightly) was **not** tested with this pack.

## Speed

Same machine, same image, same recipe, same plan. Only the weights differ between the two columns.

| Test (diffbot's `recipe/bench` scripts) | diffbot pack (V4.1-Flash) | **This pack (1017)** |
|---|---|---|
| Short-context sweep / tool calls | 11/11 / OK | 11/11 / OK |
| 2K prompt, 1 stream: decode / prefill (tok/s) | 104.8 / 4,051 | **105.0** / 4,265 |
| 2K prompt, 4 streams: decode / prefill, total (tok/s) | 219.7 / 7,900 | **212.7** / 8,783 |
| 46K prompt, 1 stream: prefill / decode (2 runs) | 8,446-8,489 / 116.0, 100.1 | **8,555-8,697** / 109.6, 100.7 |
| 46K prompt, 4 streams: prefill / decode, total | 15,664 / 208.2 | **15,770** / 209.4 |
| Agentic turn, 1 stream: time to first token / decode / tokens per DSpark step | 1.66 s / 111.2 / 3.25 | 1.62 s / **110.0** / 3.30 |
| Agentic turn, 4 streams: time to first token / decode total / tokens per step | 4.65 s / 162.5 / 3.67 | 4.53 s / **158.0** / 3.48 |
| Needle at 100K / 250K / 448K tokens | found in 12.7 / 25.6 / 42.4 s | **found** in 11.9 / 23.8 / 40.1 s |

- Speed matches the old pack within run-to-run noise. The one visible difference: at 4 streams the retrained drafter's
  guesses are accepted a little less often (3.48 vs 3.67 tokens per step).
- These are lower than the diffbot card's headline numbers. That card measured a different setup: the `gen30` plan
  (faster on synthetic prompts) on 2× Max-Q cards at 300 W. Ours ran the `blend30` plan with P2P off.
- **Memory:** VRAM 91.5 / 90.4 GB on the two cards (the old pack: 90.7 / 90.3). Lowest free host RAM while serving: 12.8 GB.

**Test machine:**
- 2× RTX PRO 6000 Blackwell Workstation Edition (96 GB), power-limited to 450 W, PCIe 5.0 x8 each, P2P off.
- AMD Ryzen 9 9950X, 256 GB DDR5 (249 GiB usable), NVMe.
- Driver 610.57.04; image `dsv41-flash-exl3-3bpw-sm120` built from diffbot's recipe at `a1cf1bd2`.

## Quality

<!-- quality-results -->
Both packs ran diffbot's quality probe (`recipe/bench/quality-probe.py`: GSM8K first 200 and HumanEval 164, greedy,
thinking off, 4 streams, max 4,096 tokens) on the same machine and setup. Only the weights differed.

| | GSM8K-200 | HumanEval-164 |
|---|---:|---:|
| **This pack (1017, uncalibrated experts)** | **98.5 %** (197/200) | **95.1 %** (156/164) |
| diffbot pack (V4.1-Flash, calibrated experts), same machine | 98.0 % (196/200) | 95.7 % (157/164) |
| diffbot pack, as published on its card | 98.0 % | 95.1-96.3 % (4 runs) |

- The two packs are within one item of each other on both tests, and the diffbot pack's run reproduces its published
  numbers, so the setup matches.
- Item by item:
  - GSM8K: 1017 solves 2 problems the old pack misses and misses 1 it solves.
  - HumanEval: 1017 passes 3 the old pack fails and fails 4 it passes.

  Both tests are close to their ceiling for this model, so they show no damage from the uncalibrated experts but can't
  rule out a small loss.
- **HumanEval ran sandboxed.** Each program ran in a fresh container: no network, read-only filesystem, unprivileged
  user, 512 MB, 15 s. That replaces the probe's direct execution on the host; the rest of the script is unchanged. All 8
  failures were wrong answers, not sandbox limits.
- Raw per-item outputs, including the generated code, are in `results/probe-1017-pack.jsonl` and
  `results/probe-diffbot-pack.jsonl`.
<!-- /quality-results -->

What hasn't been measured:
- **KL divergence against the original model.** The reference logits for KL need the 510 GB original, which can't run
  on this machine, and nobody has published a 1017 reference set yet. The diffbot pack measured KL 0.054 / 0.032 / 0.010
  (neutral / code / agentic) for V4.1-Flash with calibrated experts. Expect this pack's to be somewhat higher, because
  PixelML's experts are uncalibrated.
- **DeepSeek's claimed gains for 1017** (Terminal-Bench, DeepSWE and others). These are full-precision numbers and
  weren't checked here.

## Limitations

- **Uncalibrated routed experts** (identity Hessian). When a calibrated 1017 EXL3 3 bpw quant appears (coolbho3k made
  the calibrated one for V4.1-Flash), swapping it into shards 1-43 should be an upgrade.
- **The host-expert plans come from the old model.** They decide which 30% of the experts live in host RAM, and were
  built from V4.1-Flash routing statistics. A 1017-specific plan (`EXL3_EXPERT_STATS` + `recipe/plans/make_plan.py`)
  may decode faster.
- **Tested only with diffbot's recipe at `a1cf1bd2`,** on one machine.
- **The drafter and Engram encodings are lossy,** the same as in the diffbot pack: Engram first-chunk relative error
  0.089 (the script measured 0.093 on V4.1-Flash).

## How it was built

All steps follow diffbot's `recipe/quantize/README.md`, with diffbot's scripts unchanged. `build/` has the assembly
script; the times are from this machine.

1. **Download** PixelML @ `3d095b72` (all files) and DeepSeek 1017 @ `1bae0e10` (shards 44-48, config, index,
   tokenizer, `encoding/`). Every file used was checked against the Hub's sha256.
2. **Drafter:** `quantize_mtp_exl3.py --bits 3 --greedy --beam 16 --batch 8` on shards 44-46, on one GPU: 1,152 matrices in
   419 s. `quantize_mtp_exl3.py` imports `tools/pack_meta.py` from sfxnz's recipe at commit `460ac89c`, the commit
   diffbot's `quantize_mtp_exl3.diff` applies to.
3. **Engram:** `convert_engram_int4.py` on shards 47-48: CPU only, about 9 minutes.
4. **Assembly:** `build/assemble_pack_1017.py`. It is diffbot's `assemble_pack.py` adapted to PixelML's shard layout:
   - PixelML shards 1-43, then the new drafter (44-46) and Engram (47-48) shards;
   - the index is rebuilt from the shard headers, because PixelML's index lists `.mcg` tensors its shards don't contain;
   - `config.json` is the diffbot pack's, with only `quantization_config.source` changed.

   The result has exactly the diffbot pack's 190,549 tensor names, and its drafter and Engram tensors have the same
   dtypes, shapes and sizes as the diffbot pack's.

## Related work

- [cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000):
  an independent recipe for the same stack, published the same night. It runs without Docker (bubblewrap), keeps the
  drafter in MXFP4, and reports GSM8K, MATH-500, HumanEval, MMLU-Pro and C-Eval results.
- [LibertAIDAI](https://huggingface.co/LibertAIDAI) published calibrated 1017 EXL3 quants (2.9 and 3.5 bpw). They
  quantize every layer in ExLlamaV3's native layout, so they don't load in this stack.

## Credits and license

Built entirely on the work of:
- **DeepSeek-AI:** the model and weights ([DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT).
- **PixelML:** the 1017 routed-expert quantization ([PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw](https://huggingface.co/PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw), MIT).
- **diffbot:** the pack layout, the serving recipe, and the drafter, Engram and assembly scripts ([diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), MIT).
- **sfxnz:** the V4.1 EXL3 quantization tooling ([sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark), MIT).
- **turboderp:** ExLlamaV3 and the EXL3 format ([turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), MIT).
- **coolbho3k:** the calibrated V4.1-Flash quant behind the diffbot pack that this one is measured against.

The weights keep DeepSeek's MIT license (`LICENSE`). Repacking adds no restrictions.
