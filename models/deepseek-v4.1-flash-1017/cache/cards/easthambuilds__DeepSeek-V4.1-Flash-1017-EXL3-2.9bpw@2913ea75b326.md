---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
tags:
  - exl3
  - deepseek
  - moe
  - tensorfold
  - dgx-spark
---

# DeepSeek-V4.1-Flash-1017 — EXL3 2.9 bpw, reproducible

An EXL3 quant of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(commit `1bae0e10`) for the TensorFold `dsv41` engines on NVIDIA DGX Spark (GB10). Every step from the pinned source bytes to these
tensors is published, with the checks we ran, so you can rebuild it or audit it.

- **Layout:** the Mia-AiLab 2.9 bpw layout that the TensorFold `dsv41` engines load.
  - EXL3 `mul1` codebook, `out_scales = always`.
  - Per-tensor bit widths from Mia-AiLab's V4.1 2.9 bpw map. Routed experts are 3 bits, except layers 18–22 at 2;
    attention is 5–6; `indexer.wk` is 8; the head is 6; the MTP / DSpark drafter is 4.
  - The vision tower and aligner are unquantized.
- **Engram tables:** not included. Serve with DeepSeek's original shards 47 and 48 next to the quant, as the engines
  expect.
- **Chat template:** Mia-AiLab's `chat_template.jinja` from
  [their V4.1 2.9 bpw quant](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) (`64ba41b6`, MIT). It is
  the template the TensorFold dsv41 engine serves V4.1 with; DeepSeek's own template uses Jinja features the engine's
  template compiler does not accept. 1017's own `encoding/` is included too. The 1017 release ships no template, and
  its `encoding/` and tokenizer are byte-identical to the base V4.1 release's.
- **Size:** 210.9 GB (196.4 GiB, 39 shards). Effective rate 2.90 bpw, excluding the head.

## What is different from other V4.1 EXL3 quants

The converter is exllamav3 [PR #449](https://github.com/turboderp-org/exllamav3/pull/449) with five small patches. Three of them fix
V4.1 conversion bugs in PR #449 itself, which we [reported](https://github.com/turboderp-org/exllamav3/pull/449#issuecomment-6241301205):

1. **All experts are calibrated.** Without the fix, every routed expert's gate/up Hessian is built only from the tokens
   routed to expert 0, and many experts' down projections see too few tokens for a full-rank Hessian. Here every routed
   expert's Hessians see all 262,144 calibration tokens.
2. **The shared KV pool is carried between layers during calibration.** Without the fix, conversion stops at layer 3.
3. **The hyperconnections' carried pre weights are kept between layers during calibration.** Without the fix, the
   first hyperconnection site of every layer after the first passes stream 0 on instead of the model's weighted mix.
   Every layer's Hessians then come from inputs the model never produces, and the converter's own error
   measurement cannot show it.

The other two keep both Engram n-gram tables unquantized, and quantize `indexer.wk` and `engram.wkv` (as the TensorFold
load plan requires).

## Quality

Measured with exllamav3's `eval/model_diff.py` against DeepSeek's FP8 1017 checkpoint: 100 rows × 2,048 tokens of
its default eval text, the same tokens for both quants, Engram tables read from the original shards. Raw output is
in `provenance/logs/kl/`.

| vs DeepSeek-V4.1-Flash-1017 (FP8) | KL(quant ‖ FP8) | KL(FP8 ‖ quant) | Top-1 agreement | Median per-token KL |
| --- | ---: | ---: | ---: | ---: |
| **This quant (2.9 bpw)** | 0.0864 | 0.0914 | 90.8% | 0.0137 |
| LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw | 0.0857 | 0.0905 | 90.9% | 0.0135 |

At the same 2.9 bpw layout, the two are level on this text; the difference is within what we would expect from noise
(no paired interval computed yet).

## Speed on 2 × DGX Spark (TensorFold TP2)

Measured with bertholomus's [`deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig)
engine, unchanged, on two GB10 joined directly by their ConnectX-7 ports, with MTP drafts on. Its release gates all pass:

| Gate | Result |
| --- | --- |
| Concurrent == solo, drafted (12 prompts, burst / staggered) | 12/12, 12/12 |
| Concurrent == solo, drafts off | 12/12, 12/12 |
| Drafted == drafts off | 12/12 |
| Images (6 checks) | 6/6 |

Speed (kit `bench`, tok/s). For scale, the right-hand column is what bertholomus publishes for Mia-AiLab's 2.9 bpw quant
of the base V4.1 Flash on their own pair; it is not a same-hardware comparison.

| | This quant | Published, Mia-AiLab 2.9 |
| --- | ---: | ---: |
| Decode, set a (code / prose / structured) | 84.5 / 56.3 / 101.3 | 89.8 / 59.0 / 106.6 |
| Decode, set b | 94.5 / 61.3 / 138.6 | 103.1 / 64.1 / 143.2 |
| 2 / 4 concurrent streams (total) | 82.8 / 118.6 | 83.1 / 118.9 |
| Sustained | 128 | 142 |
| Prefill, 8K / 32K / 64K / 128K | 1557 / 1597 / 1584 / 1543 | 2433 / 2495 / 2461 / 2354 |

Decode and concurrency are within about 5–8% of the published figures. Prefill is about 35% lower on our pair, and we
have not yet found why. The GPUs reach their full clocks under load, so we are testing other engine builds, and another
2.9 bpw quant, on the same two machines. Raw results are in `provenance/logs/tp2-v1/`.

## Serving

Use **[the recipe for 2 × DGX Spark](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2x-DGX-Sparks-TensorFold)**:
`./start.sh` sets up both Sparks and serves this checkpoint with Bertholomus AI's `dsv41` TensorFold engine, unchanged.

To do it by hand instead, follow the [Bertholomus AI TP2 recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10), with
this repository as `<MODEL_DIR>` and the 1017 shards 47/48 as `<ENGRAM_DIR>`. Use a kernel kit whose
`vision/gate_bias_vl.safetensors` comes from the 1017 checkpoint (`kit/tools/make_bias_vl.py`, sha256
`4745da4b5f24c67170ee24e2c95652e45ba13c0ca0abc2be3cc7a2a3545792d0`). The published kit's copy comes from the base V4.1.

## How it was made

`provenance/` holds the full record (PROVENANCE.md):

- **Source:** the source file list with the Hub's sha256 values.
- **Recipe:** generated from Mia-AiLab's `quantization_config.json` by a deterministic script.
- **Converter:** the exllamav3 patches, and the image build with pinned packages and its `pip freeze`.
- **Distributed conversion:** the tools that spread the work over 12 GB10s, and the checks that this does not change a
  single byte:
  - local and distributed capture give the same Hessians;
  - quantizing a tensor alone or in a batch gives the same result;
  - a full layer quantized either way is byte-identical;
  - during the run, the converter re-checked each worker's results.

Calibration: exllamav3's bundled text mix at 128 × 2048. Conversion: about 6 hours on 12 GB10s.

## Credits

This quant is built on other people's work. Each entry says what we used and what we changed.

- **[DeepSeek](https://huggingface.co/deepseek-ai):** [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (MIT), the weights we quantized. The Engram tables are used as published.
- **[turboderp](https://github.com/turboderp-org):** [exllamav3](https://github.com/turboderp-org/exllamav3) and the
  EXL3 format (MIT): the converter and its quantizer, and `eval/model_diff.py` for the quality numbers.
- **[anoane](https://github.com/anoane):** the DeepSeek-V4.1 port of exllamav3,
  [PR #449](https://github.com/turboderp-org/exllamav3/pull/449) at `c341a8f5`, which we converted with. We modified it
  with five small patches (`provenance/patches/`). Three fix V4.1 conversion bugs, reported on the PR. Two keep both
  Engram tables unquantized and quantize `indexer.wk` and `engram.wkv`. Our tools for spreading the conversion over
  several machines are added on top and do not change its output.
- **[Mia-AiLab / Mia's AI Lab](https://huggingface.co/Mia-AiLab):** the
  [V4.1 2.9 bpw quant](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) (MIT). Its per-tensor bit-width
  map is reproduced exactly from its `quantization_config.json`, its layout is followed, and its `chat_template.jinja`
  is included unchanged. Mia's model cards and
  serving recipes are the model for this card and our recipe.
- **[Bertholomus AI](https://github.com/bertholomus) (Albert Lee):** the `dsv41` TensorFold engines, kernel kit, kit
  tools and TP2 recipe that serve this quant, used unchanged except for the one kit file noted under Serving. The
  release gates and `kit_bench.py` give the speed numbers.
- **[ashhart](https://github.com/ashhart):** [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0) and the
  community gates of [#847](https://github.com/ashhart/TensorFold/issues/847).
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI):** another 1017 EXL3 quant at 2.9 bpw, measured beside ours
  for comparison only.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Bertholomus AI or Mia-AiLab.

## License

MIT, as the base model.
