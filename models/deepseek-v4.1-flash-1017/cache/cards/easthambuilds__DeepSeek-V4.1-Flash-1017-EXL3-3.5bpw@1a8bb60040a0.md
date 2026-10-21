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

# DeepSeek-V4.1-Flash-1017 — EXL3 3.5 bpw, agentic calibration, for three DGX Sparks

An EXL3 quant of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(commit `1bae0e10`), sized for TensorFold TP3 on three NVIDIA DGX Spark (GB10). It is the larger sibling of our
[2.9 bpw quant](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw): the same converter,
patches and checks, with more bits in the routed experts and a calibration set of agentic, tool-use and coding text.
Every step from the pinned source bytes to these files is in `provenance/`.

- **Layout:** EXL3 `mul1` codebook, `out_scales = always`.
  - Routed experts average 3.5 bits: 4 bits in layers 0–14, 16–18 and 21; in layers 20, 22 and 24 the down projection
    is 4 bits and gate/up 3; 3 bits elsewhere. These per-layer widths follow LakyFx's
    [1017 3.5 bpw TensorFold export](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold),
    which follows the Pollard precision map.
  - Every other tensor has the widths of our 2.9 bpw quant (Mia-AiLab's map): attention 5–6, `indexer.wk` 8, head 6,
    MTP / DSpark drafter 4. The vision tower and aligner are unquantized.
- **Calibration:** 256 rows × 2,048 tokens (369,677 tokens) of an agentic / tool-use / coding mix from nine
  permissively licensed Hub datasets, rendered in DeepSeek-V4.1's own prompt format (DSML tool calls). It replaces
  exllamav3's default text. Every routed expert's Hessians see every calibration token. The MTP drafter is quantized by
  the converter as an uncalibrated side model, as in our 2.9 bpw quant.
- **Router bias in F32:** the 43 `ffn.gate.bias` tensors are DeepSeek's F32 values, bit-exact. The converter stores
  them in BF16; we put the source values back. On our 2.9 bpw quant the same change measured neutral.
- **Engram tables:** not included. Serve with DeepSeek's original shards 47 and 48 next to the quant.
- **Chat template:** Mia-AiLab's `chat_template.jinja` (as in our 2.9 bpw quant), plus 1017's `encoding/`.
- **Size:** 253.3 GB (235.9 GiB, 41 shards). Effective rate 3.52 bpw, excluding the head.

## Quality

### Fidelity to the original model

exllamav3's `eval/model_diff.py` against DeepSeek's FP8 1017 checkpoint: 100 rows × 2,048 tokens per set, the same
tokens for every quant, Engram tables read from the original shards. KL(quant ‖ FP8), lower is better, with top-1
agreement. Raw output is in `provenance/logs/tp3a/kl/`.

| Set | This quant (3.5 bpw) | Our 2.9 bpw | LibertAIDAI 2.9 bpw |
| --- | ---: | ---: | ---: |
| Wikitext-2 (exllamav3's default eval text) | **0.0738** · 91.5% | 0.0864 · 90.8% | 0.0857 · 90.9% |
| FineWeb-Edu (English web text) | **0.0415** · 91.9% | 0.0524 · 90.7% | 0.0526 · 90.8% |
| Aya (seven languages) | **0.0808** · 89.8% | 0.1068 · 88.4% | 0.1070 · 88.3% |
| Toucan (MCP tool-use conversations) | **0.3595** · 83.5% | 0.4953 · 80.9% | 0.4952 · 80.5% |
| Agentic, held out (rows never calibrated on) | **0.0813** · 94.3% | 0.1339 · 92.3% | 0.1347 · 92.4% |

KL is 15–39% lower than the 2.9 bpw quants on every set. Two notes:

- FineWeb-Edu, Aya and Toucan are not in the calibration mix. The held-out agentic rows come from the same nine
  datasets as the calibration rows, so that set is in-distribution.
- Part of the gain is the calibration, not the bits. The same recipe calibrated on exllamav3's default text instead
  measures 7–13% lower KL on Wikitext-2, FineWeb-Edu and Aya, and 8–15% higher on Toucan and the agentic rows. We
  chose the agentic calibration for agent and tool-use work.

### Benchmarks

Greedy decoding, over HTTP. Thinking off: the full MMLU (14,042) and GSM8K (1,319). Thinking on (the server's default
effort): fixed seeded subsets of 200 and 100. BFCL v3 prompts (3,473) with our simplified AST checker. Needle in a
haystack at 8K / 32K / 128K / 250K × 3 depths. This quant on TensorFold TP3 (three GB10); our 2.9 bpw on TensorFold TP2
(two GB10). Raw results and the paired comparison are in `provenance/logs/tp3a/suite/`.

| | This quant | Our 2.9 bpw | Paired difference (McNemar p) |
| --- | ---: | ---: | --- |
| GSM8K, thinking off | 97.04% | 97.19% | −0.15 pp (0.75) |
| MMLU, thinking off | 88.41% | 88.21% | +0.20 pp (0.21) |
| BFCL v3 | 81.46% | 81.28% | +0.17 pp (0.67) |
| GSM8K, thinking on | 98.0% | 98.0% | 0 |
| MMLU, thinking on | 94.5% | 92.0% | +2.5 pp (0.06) |
| Needle | 12/12 | 12/12 | |

No difference is significant. These benchmarks sit close to the model's ceiling and do not show the KL gain.
With thinking off, 81 of the 14,042 MMLU replies started working the problem in text instead of giving the letter and
were cut by the 16-token answer limit (4 for the 2.9 bpw quant); they count as wrong.

## Speed and memory on 3 × DGX Spark (TensorFold TP3)

LakyFx's TensorFold TP3 runtime
([deepseek-v41-flash-exl3-tp3-spark](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark) at tag
`v1.2.2TF`, commit `14a6d49`, which pins TensorFold runtime snapshot `61357deb`), on three GB10 joined by their
ConnectX-7 ports. Its engine and kernels are unchanged. We added a
launcher profile for this checkpoint, a launcher option for our three point-to-point links, one exact BF16 → F32 cast
in its rank-cache exporter, and RoPE tables for a 4M-token pool made with its own tool. 12 streams, a 4M-token shared
KV pool, 401,408-token request context. **GPU clocks capped at 2,000 MHz on all
nodes**, for every number here. `kit_bench` medians, tok/s. The right-hand column is our 2.9 bpw quant on two GB10
with TensorFold TP2 at the same 2,000 MHz cap, served through our TP2 recipe.

| | This quant, TP3 | Our 2.9 bpw, TP2 |
| --- | ---: | ---: |
| Decode, set a (code / prose / structured) | 109.2 / 71.2 / 124.7 | 85.7 / 56.1 / 103.4 |
| Decode, set b (384 tokens) | 125.9 / 77.7 / 172.2 | 94.8 / 61.3 / 140.9 |
| Prefill, 8K / 32K / 128K | 2,223 / 2,269 / 2,115 | 2,120 / 2,269 / 2,100 |
| 2 / 4 / 8 concurrent streams (total) | 99–106 / 133–144 / 169–181 | 77–78 / 102–105 / 112–118 |

- The decode gain comes from the third GPU, not from this quant. On the same TP3 runtime, this quant decodes 8–13%
  slower than our 2.9 bpw weights on code and structured text (theirs: set a 118.2 / 71.4 / 143.8, set b 138.4 /
  79.5 / 193.3); prose is level. This quant trades that speed for fidelity.
- The TP2 8-stream figure is limited by the TP2 recipe's 4 parallel slots.
- **Memory:** the model takes 92.5 GiB on the head node. With the pool allocated and idle, the nodes have 17.8 / 19.3 /
  19.2 GiB available. Under a full-pool stress test (ten requests of 389,900 tokens, all completed), the lowest was
  9.9 GB on the head and 19.5 GB on the other two.

Raw results are in `provenance/logs/tp3a/bench/` and `provenance/logs/tp3a/stress/`.

## Serving

This checkpoint is too large for two Sparks. Use **[the recipe for 3 × DGX Spark](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3x-DGX-Sparks-TensorFold)**:
`./start.sh` sets up three Sparks joined point to point by their ConnectX-7 ports and serves this checkpoint with
LakyFx's TP3 runtime, engine and kernels unchanged.

That runtime ships a launcher and export tools for LakyFx's own checkpoint. For this one we made a checkpoint plan
(`provenance/tools/make_plan.py`), exported and verified the three rank caches with LakyFx's tools, and added a
launcher profile. `provenance/PROVENANCE-TP3A.md` lists the steps. The numbers on this card come from that hand-built
setup, which the recipe automates; its README says how far the recipe itself has been tested.

The files are standard EXL3 for exllamav3 with DeepSeek-V4.1 support (PR #449), which can load them wherever the
memory is available.

## How it was made

`provenance/PROVENANCE-TP3A.md` holds this quant's record; `provenance/PROVENANCE.md` holds what it shares with our
2.9 bpw quant:

- **Source:** the source file list with the Hub's sha256 values.
- **Recipe:** `recipe/recipe-dsv41-tp3a-350.yaml`, per tensor.
- **Calibration:** `logs/v2-agentic-cal-768.json` records the mix: every source parquet with its sha256, the examples
  and tokens taken from each, and the seed. This quant uses its first 256 rows. `tools/build_agentic_cal.py` builds it.
- **Converter:** exllamav3 PR #449 at `c341a8f5` with our five patches, the same image as the 2.9 bpw quant.
- **Distributed conversion:** 8 GB10s, about 6.5 hours, with the converter re-checking every worker's results.
- **After conversion:** the F32 router bias (`tools/splice_bias.py`), the Markov head names
  (`tools/fix_markov_names.py`), the chat template, and `avg_bits` in `config.json` (`tools/fix_config_avg_bits.py`).
  We rebuilt the files uploaded here from the converter's output with these tools and checked them byte for byte
  against the checkpoint we served for every test above; only `config.json` differs, by the `avg_bits` key.
  `logs/tp3a/tp3a-pub.sha256` lists every file.

## Credits

This quant is built on other people's work. Each entry says what we used.

- **[DeepSeek](https://huggingface.co/deepseek-ai):** [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (MIT), the weights we quantized. The Engram tables and the router bias are used as published.
- **[turboderp](https://github.com/turboderp-org):** [exllamav3](https://github.com/turboderp-org/exllamav3) and the
  EXL3 format (MIT): the converter, its quantizer and `eval/model_diff.py`.
- **[anoane](https://github.com/anoane):** the DeepSeek-V4.1 port of exllamav3,
  [PR #449](https://github.com/turboderp-org/exllamav3/pull/449) at `c341a8f5`, which we converted with, with five
  small patches (`provenance/patches/`).
- **[LakyFx](https://huggingface.co/LakyFx):** the TensorFold TP3 runtime for three DGX Sparks that serves this quant
  (MIT), and the per-layer routed-expert widths of their 1017 3.5 bpw export, which this recipe follows.
- **[bot-lab-21](https://huggingface.co/bot-lab-21) and [WestWaters](https://github.com/WestWaters/pollard-weights):**
  the Pollard precision map behind those widths.
- **[Mia-AiLab / Mia's AI Lab](https://huggingface.co/Mia-AiLab):** the per-tensor widths of every tensor other than
  the routed experts, the layout, and `chat_template.jinja`, from their V4.1 2.9 bpw quant (MIT).
- **[Bertholomus AI](https://github.com/bertholomus) (Albert Lee):** the `dsv41` TensorFold TP2 engine, kit and
  `kit_bench.py`, used for the TP2 comparison and the speed numbers.
- **[ashhart](https://github.com/ashhart):** [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0).
- **Calibration data:** [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
  (CC-BY-4.0), [SWE-Gym/OpenHands-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories)
  (MIT), [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
  (Apache-2.0), [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) (CC-BY-4.0),
  [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Apache-2.0),
  [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
  (Apache-2.0), [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
  (Apache-2.0), [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)
  (ODC-BY), [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT).
- **Evaluation data:** Wikitext-2 (via exllamav3), FineWeb-Edu, Aya, Toucan, MMLU, GSM8K and BFCL v3.
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI):** another 1017 EXL3 quant at 2.9 bpw, measured for comparison only.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, LakyFx, Bertholomus AI or Mia-AiLab.

## License

MIT, as the base model.
