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

# DeepSeek-V4.1-Flash-1017 — EXL3 3.75 bpw, agentic calibration, for three DGX Sparks

An EXL3 quant of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(commit `1bae0e10`), sized for TensorFold TP3 on three NVIDIA DGX Spark (GB10). It is our
[3.5 bpw quant](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) with more bits in the routed
experts: the same converter, patches, calibration and checks. Every step from the pinned source bytes to these files is
in `provenance/`.

- **Layout:** EXL3 `mul1` codebook, `out_scales = always`.
  - Routed experts average 3.75 bits. The down projection is 4 bits in every layer. Gate and up are 4 bits in layers
    0–18, 21, 23, 27 and 30–32, and 3 bits elsewhere.
  - We chose those widths from the 3.5 bpw quant's own conversion. Starting from its recipe, we raised the 30 routed-expert
    roles whose measured quantization error at 3 bits was highest, one layer's down projection or gate+up pair at a time,
    until the mean reached 3.75 (`provenance/tools/make_recipe_tp3b.py`).
  - Every other tensor is as in the 3.5 bpw quant: attention 5–6, `indexer.wk` 8, head 6, MTP / DSpark drafter 4. The
    vision tower and aligner are unquantized.
- **Calibration:** the 3.5 bpw quant's: 256 rows × 2,048 tokens of an agentic / tool-use / coding mix from nine
  permissively licensed Hub datasets, in DeepSeek-V4.1's own prompt format. With the calibration and every other tensor
  held fixed, the difference between the two quants is the extra routed-expert bits.
- **Router bias in F32:** the 43 `ffn.gate.bias` tensors are DeepSeek's F32 values, bit-exact.
- **Engram tables:** not included. Serve with DeepSeek's original shards 47 and 48 next to the quant.
- **Chat template:** Mia-AiLab's `chat_template.jinja`, plus 1017's `encoding/`.
- **Size:** 270.3 GB (251.7 GiB, 41 shards), 17 GB more than the 3.5 bpw quant. Effective rate 3.76 bpw, excluding the
  head.

## Quality

### Fidelity to the original model

exllamav3's `eval/model_diff.py` against DeepSeek's FP8 1017 checkpoint: 100 rows × 2,048 tokens per set, the same tokens
as for the 3.5 bpw quant, Engram tables read from the original shards. KL(quant ‖ FP8), lower is better, with top-1
agreement. Raw output is in `provenance/logs/tp3b/kl/`.

| Set | This quant (3.75 bpw) | Our 3.5 bpw | Change in KL |
| --- | ---: | ---: | ---: |
| Wikitext-2 (exllamav3's default eval text) | **0.0594** · 92.5% | 0.0738 · 91.5% | −19.5% |
| FineWeb-Edu (English web text) | **0.0347** · 92.6% | 0.0415 · 91.9% | −16.3% |
| Aya (seven languages) | **0.0644** · 91.0% | 0.0808 · 89.8% | −20.4% |
| Toucan (MCP tool-use conversations) | **0.3452** · 84.1% | 0.3595 · 83.5% | −4.0% |
| Agentic, held out (rows never calibrated on) | **0.0762** · 94.4% | 0.0813 · 94.3% | −6.3% |

KL is lower on every set: by 16–20% on general and multilingual text, and by 4–6% on the two tool-use sets.

### Benchmarks

The 3.5 bpw quant's suite, on the same three GB10 and runtime: greedy decoding over HTTP. Thinking off: the full MMLU
(14,042) and GSM8K (1,319). Thinking on (the server's default effort): fixed seeded subsets of 200 and 100. BFCL v3
prompts (3,473) with our simplified AST checker. Needle in a haystack at 8K / 32K / 128K / 250K × 3 depths. Raw results
and the paired comparison are in `provenance/logs/tp3b/suite/`.

| | This quant | Our 3.5 bpw | Paired difference (McNemar p) |
| --- | ---: | ---: | --- |
| GSM8K, thinking off | 97.27% | 97.04% | +0.23 pp (0.51) |
| MMLU, thinking off | 88.41% | 88.41% | 0 (157 / 157 discordant) |
| BFCL v3 | 81.37% | 81.46% | −0.09 pp (0.84) |
| GSM8K, thinking on | 99.0% | 98.0% | +1 question |
| MMLU, thinking on | 93.5% | 94.5% | −2 questions (0.50) |
| Needle | 12/12 | 12/12 | |

No difference is significant: these benchmarks sit near the model's ceiling and do not show the KL gain. With thinking
off, 49 of the 14,042 MMLU replies started working the problem in text and were cut by the 16-token answer limit (81
for the 3.5 bpw quant); they count as wrong.

## Speed and memory on 3 × DGX Spark (TensorFold TP3)

LakyFx's TensorFold TP3 runtime
([deepseek-v41-flash-exl3-tp3-spark](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark) at tag `v1.2.2TF`,
commit `14a6d49`, which pins TensorFold runtime snapshot `61357deb`), engine and kernels unchanged, with the same
additions as for the 3.5 bpw quant (a launcher profile, a launcher option for three point-to-point links, one exact
BF16 → F32 cast in its rank-cache exporter, RoPE tables for the pool). Both quants measured with 12 streams, a 4M-token
shared KV pool and 401,408-token requests, **GPU clocks capped at 2,000 MHz on all three nodes**, no other traffic.
`kit_bench` medians, tok/s.

| | This quant | Our 3.5 bpw |
| --- | ---: | ---: |
| Decode, set a (code / prose / structured) | 111.9 / 67.5 / 129.6 | 109.2 / 71.2 / 124.7 |
| Decode, set b (384 tokens) | 122.6 / 73.8 / 147.6 | 125.9 / 77.7 / 172.2 |
| Prefill, 8K / 32K / 128K | 2,222 / 2,257 / 2,074 | 2,223 / 2,269 / 2,115 |
| 2 / 4 / 8 concurrent streams (total) | 93–102 / 121–136 / 165–181 | 99–106 / 133–144 / 169–181 |

- Decode is 5% slower on prose and 14% slower on set b's structured text; code is level (+2.5% / −2.6%) and set a's
  structured text 4% faster. Prefill is level to 2% slower; 2 and 4 streams are 5–9% slower.
- **Memory:** the model takes 97.8 GiB on the head node (3.5 bpw: 92.5). With the 4M-token pool allocated and idle, the
  nodes had 13.3 / 13.7 / 13.7 GiB available (3.5 bpw: 17.8 / 19.3 / 19.2), and 8.5 / 8.8 / 8.8 GiB at idle after the
  3.4-hour benchmark suite. We did not run the 3.5 bpw quant's full-pool stress test on this configuration.

Raw results are in `provenance/logs/tp3b/bench/` and `provenance/logs/tp3b/memory.md`.

### Recommended settings

Because the nodes have about 4.5 GiB less memory available than with the 3.5 bpw quant, we recommend a **3.3M-token pool**
(LakyFx's qualified pool size), a **1M-token context per request**, and **2 to 4 concurrent sessions**. This is our
recommendation, not a measured configuration: every number above uses a 4M pool, 401,408-token requests and 12 streams.

## Serving

This checkpoint is too large for two Sparks. Use **[the recipe for 3 × DGX Spark](https://github.com/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw-3x-DGX-Sparks-TensorFold)**:
`./start.sh` sets up three Sparks joined point to point by their ConnectX-7 ports and serves this checkpoint with
LakyFx's TP3 runtime, engine and kernels unchanged. Its defaults are the recommended settings above.

That runtime ships a launcher and export tools for LakyFx's own checkpoint. For this one, as for our 3.5 bpw quant, we
made a checkpoint plan (`provenance/tools/make_plan.py`), exported and verified the three rank caches with LakyFx's
tools, and added a launcher profile. `provenance/PROVENANCE-TP3B.md` lists the steps. The numbers on this card come from
that hand-built setup. The recipe ran end to end on 2026-10-22 and built rank caches byte-identical to it; its README
lists what that run covered.

The files are standard EXL3 for exllamav3 with DeepSeek-V4.1 support (PR #449), which can load them wherever the memory
is available.

## How it was made

`provenance/PROVENANCE-TP3B.md` holds this quant's record; `provenance/PROVENANCE-TP3A.md` and
`provenance/PROVENANCE.md` hold what it shares with our 3.5 bpw and 2.9 bpw quants:

- **Source:** the source file list with the Hub's sha256 values.
- **Recipe:** `recipe/recipe-dsv41-tp3b-375.yaml`, per tensor, made by `tools/make_recipe_tp3b.py` from the 3.5 bpw
  recipe and its conversion log.
- **Calibration:** as the 3.5 bpw quant (`logs/v2-agentic-cal-768.json`, first 256 rows).
- **Converter:** exllamav3 PR #449 at `c341a8f5` with our five patches, the same image.
- **Distributed conversion:** 8 GB10s (the converter and 7 workers), about 7 hours, with the converter re-checking every
  worker's results.
- **After conversion:** the F32 router bias, the Markov head names, the chat template and `avg_bits`, with the same tools
  as the 3.5 bpw quant. We rebuilt the files uploaded here from the converter's output and checked them byte for byte
  against the checkpoint we served for every test above; only `config.json` differs, by the `avg_bits` key.
  `logs/tp3b/tp3b-pub.sha256` lists every file.

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
  (MIT), and the per-layer routed-expert widths of their 1017 3.5 bpw export, the starting point of this recipe.
- **[bot-lab-21](https://huggingface.co/bot-lab-21) and [WestWaters](https://github.com/WestWaters/pollard-weights):**
  the Pollard precision map behind those widths.
- **[Mia-AiLab / Mia's AI Lab](https://huggingface.co/Mia-AiLab):** the per-tensor widths of every tensor other than
  the routed experts, the layout, and `chat_template.jinja`, from their V4.1 2.9 bpw quant (MIT).
- **[Bertholomus AI](https://github.com/bertholomus) (Albert Lee):** `kit_bench.py` and the quality and needle tools
  behind the speed and benchmark numbers.
- **[ashhart](https://github.com/ashhart):** [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0).
- **Calibration data:** as the 3.5 bpw quant's card lists: nebius/SWE-agent-trajectories (CC-BY-4.0),
  SWE-Gym/OpenHands-SFT-Trajectories (MIT), NousResearch/hermes-function-calling-v1 (Apache-2.0), nvidia/When2Call
  (CC-BY-4.0), Team-ACE/ToolACE (Apache-2.0), glaiveai/glaive-function-calling-v2 (Apache-2.0),
  open-thoughts/OpenThoughts-114k (Apache-2.0), bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-BY),
  HuggingFaceH4/ultrachat_200k (MIT).
- **Evaluation data:** Wikitext-2 (via exllamav3), FineWeb-Edu, Aya, Toucan, MMLU, GSM8K and BFCL v3.

Not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, LakyFx, Bertholomus AI or Mia-AiLab.

## License

MIT, as the base model.
