---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- unsloth/Qwen3.8-Flash-Next-GGUF
- Qwen/Qwen3.8-Flash-Next
language:
- ja
- en
tags:
- gguf
- moe
- expert-pruning
- strata
- unsloth
---

# Qwen3.8 Flash-Next-UD-IQ1_M-384Exp

**384Exp = 384 experts per layer, down from 512.** The number of experts used per token (10) is unchanged.

This is the UD-IQ1_M file (Unsloth Dynamic v3.1, Nov 4 update) from Unsloth's [Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) with 128 experts removed from every layer. The quantized expert weights and the router rows that are kept are copied byte for byte; nothing is re-quantized. The experts kept are the same list as [Qwen3.8 Flash-Next-IQ3_S-384Exp](https://huggingface.co/kaban-baku/Qwen3.8-Flash-Next-IQ3_S-384Exp-GGUF) (`selection/keep_384Exp.json` in the GitHub repository).

I made it to run Qwen3.8-Flash-Next with [Strata](https://github.com/Niko1221/Strata) on an 8 GB GPU and 64 GB of RAM, with Strata's RAM budget at 24 GB so the PC stays usable for other work.

Tools, the list of kept experts and the evaluation scripts: https://github.com/kaban-baku/Qwen3.8-Flash-Next-IQ3_S-384Exp

## Files

| File | Size | Where |
|---|---|---|
| `Qwen3.8-Flash-Next-UD-IQ1_M-384Exp-00001-of-00003.gguf` | 10.9 MB (index only) | this repository |
| `Qwen3.8-Flash-Next-UD-IQ1_M-384Exp-00002-of-00003.gguf` | 30.4 GB (experts and router) | this repository |
| `Qwen3.8-Flash-Next-UD-IQ1_M-384Exp-00003-of-00003.gguf` | 28.8 GB | identical to `UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf` in Unsloth's repository; download it from there and rename it |

SHA-256:
- 1st: `7cbd297dab47b9119d8983a63df7e2a8f05c7a325cfe0218c9d35f0706d311cd`
- 2nd: `58a3129e6c9a166b286f7338ad33c9d5b7cc32cdacdf95430b7b304fc061ccae`
- 3rd (Unsloth's file, unchanged): `05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6`

## Results

RTX 3070 Ti Laptop 8 GB, 64 GB RAM, Windows 11. Strata 0.2.11 series (engine 0.1.57) built locally with the patch below, `--resident-budget-gib 24`, 128K context, MTP, STRATA_ROUTE_PRIOR=1.5. All evaluations without thinking, temperature 0.

| | IQ3_S (512 experts, ISTA-DASLab) | UD-IQ1_M (512) | UD-IQ1_M-384Exp |
|---|---|---|---|
| Expert file | 54.8 GB | 39.2 GB | 30.4 GB |
| Decode, tokens/s (2 runs) | 22.5 / 20.1 | 46.0 / 49.0 | 48.7 / 51.2 |
| Expert cache hit rate | 68% | 75% | 75% |
| JamC-QA, 748 questions (Japanese knowledge, 4 choices) | 65.9% | 63.5% | 60.7% |
| HumanEval, 164 | 92.7% | 93.9% | 93.3% |
| MBPP, 257 | 87.9% | 89.1% | 85.6% |
| ELYZA-tasks-100 (the three graded blind together by Claude, out of 5) | 3.94 | 3.76 | 3.54 |
| tool-eval-bench, full (69 scenarios) | 89 | 85 | 85 |

- Decode speed is the mean of 4 requests (two short prompts, 1,200 output tokens each).
- ELYZA scores are only comparable within this table (a different grading round gives a different level).
- Strata reads the first file's index and the experts from the second; with a 24 GB RAM budget almost all experts of this file stay in RAM.

## What you need in Strata

- **Unsloth Dynamic v3.1 support**: the 0.2.11 series (round 0214, engine 0.1.57). I used a local build of the `f-0211-rc` branch (c3b3f503); I have not checked whether the released 0.2.11 includes it.
- **Pack options**: `python <Strata>/tools/iq_pack.py --gguf <first file> --out <PACK_DIR> --compat-bf16` (Unsloth stores some matrices as Q8_0), and start the engine with `--native <first file>`.
- **The routing patch for the speed**: up to Strata 0.2.11, STRATA_ROUTE_PRIOR and STRATA_ROUTE_RESIDENT only run on models with 512 experts. The patch is proposed in https://github.com/Niko1221/Strata/issues/5287 and is in the GitHub repository above.
- **Expert profile**: `python tools/remap_profile.py selection/keep_384Exp.json <Strata>/data/expert-profile-384Exp.bin` (same as the IQ3_S-384Exp).
- **Long prompts**: with fewer than 512 experts, Strata 0.2.10 and 0.2.11 stopped with 0xC0000409 on prompts longer than about 3,300 tokens (Strata issue #5104); the workaround is `STRATA_PREFILL_LAYER_MAJOR=0`. Not tested with this file yet.

## How the experts were chosen

Same as the IQ3_S-384Exp: I recorded which experts Strata's router picked while writing (299 prompts in eight groups: code, reading Japanese, writing Japanese, tool use, short technical explanations, general knowledge, reasoning, planning), normalized and weighted the counts per group, and kept the top 384 per layer. None of the evaluation questions were used for the selection. The list was made on the IQ3_S file and reused here as is.

## License

The weights follow the Qwen Community License 1.0 of Qwen3.8-Flash-Next (see `LICENSE`).

## Credits

Qwen (Qwen3.8-Flash-Next), Unsloth (Dynamic v3.1 quantization), Strata, and the authors of JamC-QA, ELYZA-tasks-100 and tool-eval-bench.

---

## 日本語の要約

Unsloth Dynamic v3.1 の UD-IQ1_M（Qwen3.8-Flash-Next）の各層の専門家を 512 から 384 に減らしたものです。残す専門家は IQ3_S-384Exp と同じ一覧です。8GB の GPU・RAM 64GB・Strata の RAM 上限 24GB で、書く速さは IQ3_S（512）の 21 tok/s に対して 50 tok/s、専門家のファイルは 54.8GB → 30.4GB。一般知識（JamC-QA）は 65.9 → 60.7%、コード（HumanEval・MBPP）はほぼ同等、ELYZA は 3.94 → 3.54（3 つを一緒に伏せて採点）。評価はすべて思考なし・温度 0。Strata は v3.1 対応の 0.2.11 系と、issue #5287 の修正が要ります。
