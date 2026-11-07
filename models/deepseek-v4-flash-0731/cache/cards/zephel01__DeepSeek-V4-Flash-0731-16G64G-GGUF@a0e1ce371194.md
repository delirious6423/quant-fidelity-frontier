---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, imatrix, moe, abyss]
---

# DeepSeek V4 Flash 0731 GGUF (for a 16 GB GPU + 64 GB RAM PC) — BETA

A GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT) sized
for **one 16 GB GPU (RTX 5060 Ti 16 GB, RTX 4080, RX 9070 XT ...) plus 64 GB RAM**. **Every expert is kept** (all 256,
no pruning). Speed and quality are measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC, an RTX 5090 limited to 16 GB of VRAM and to a 64 GB PC's RAM share (below), not on a real
16 GB card yet. The file may be replaced by a better build after more measurements (the folder name will change then).
Reports from real 16 GB cards are very welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

Other sizes of the same build: [32G96G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) (96.8 GB),
[32G64G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G64G-GGUF) (72.0 GB),
[DGX Spark](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF).

日本語の説明は下にあります。

## The file

| | download | KL (12 seqs) | KL, Japanese (3) | same top token | 16G64G, 4K prompt | 16G64G, 16K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`L62/`](./L62) (here)** | **61.7 GB** | **0.407** | **0.435** | **73.7%** | **84.5 tok/s** | **89.5 tok/s** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | 0.186 | 0.219 | 81.5% | 21.3 tok/s | 19.6 tok/s |

- unsloth's 96.8 GB UD-Q2_K_XL does not fit a 16 GB GPU + 64 GB RAM PC and waits for the SSD (21.3 tok/s); L62 fits
  and runs **4x faster** (84.5 tok/s) on this PC.
- The price is quality: KL 0.407 against UD-Q2_K_XL's 0.186. In the real-use checks below it did as well as
  UD-Q2_K_XL; tell us where it does not. With 32 GB of VRAM take
  [L72](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G64G-GGUF); with 96 GB RAM
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF).
- A 24 GB GPU + 64 GB RAM is not measured; L62 fits there with room to spare.
- The DSpark drafter (~6.5 GB of VRAM) is not measured at this size and is not recommended on a 16 GB card
  (`./setup-deepseek.sh --no-dspark`).

## Files

`L62/DeepSeek-V4-Flash-0731-L62-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` in the folder.

- Routed experts (all 256 in every layer), re-quantized with LDLQ from unsloth's UD-Q8_K_XL (original MXFP4 experts),
  using a Hessian dump of V4's own chat answers (241 prompts, 104 of them Japanese) plus general text, and our own
  imatrix: gate / up IQ1_S, down IQ1_M.
- Attention `attn_q_b`, `attn_output_a`, `attn_output_b`: Q5_K. Everything else is unsloth UD-Q2_K_XL's bytes; the first
  shard (metadata only, 5 MB) is unsloth's file as is, so its `general.quantized_by` still says Unsloth.
- How it was made and measured in detail: see the [32G96G card](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF).

## Quality

KL against unsloth's UD-Q8_K_XL, 12 sequences x 1,024 tokens (8 English text, 1 code, 3 Japanese), Abyss engine.

Real use (Abyss v1.3.120, RTX 5090 with 16G64G limits):

- Abyss `--check`: **10 of 11 pass (Japanese, three coding tasks, tool call, 8K-31K needle, 11-turn conversation); the one miss is a Japanese explanation answered in Chinese (see below). unsloth's UD-Q2_K_XL: 11 of 11**
- 10 Japanese answers, 5 coding tasks with tests, 8K needle: 9/10 Japanese answers stayed in Japanese, 0 wrong facts,
  5/5 coding tasks pass, needle found (unsloth's UD-Q2_K_XL on the same prompts and limits: 9/10, 0 wrong facts, 5/5).
- V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct); unsloth's GGUF does it too. Whether the original weights do it as well is being checked; a system prompt asking for Japanese answers helps.
- llama.cpp: not checked yet.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.120 on an RTX 5090 + Ryzen AI Max+ 395 PC, limited to 16 GB of VRAM
(`STRATA_GLM_VRAM_GB=16`) and a 64 GB PC's RAM share: RAM tier 40 GB (`STRATA_GLM_RAM_GB=40`) under a 58 GB memory limit.
Temperature 0, 512 tokens after a 4K / 16K prompt, page cache dropped before each run, median of 3.

- A real 16 GB card has less memory bandwidth than a 5090 and this PC's RAM (LPDDR5X) is about 3x a usual DDR5
  desktop's, so expect lower numbers on a usual 16 GB PC; reports welcome.

## Running it

[Abyss](https://github.com/zephel01/Abyss) v1.3.120 or later. Set up DeepSeek V4 once (`./setup-deepseek.sh --no-dspark`),
then pack this GGUF:

```sh
cd Abyss
.venv/bin/python tools/iq_pack.py --compat-bf16 --gguf /path/to/L62/DeepSeek-V4-Flash-0731-L62-00001-of-00003.gguf --out /path/to/L62/pack
```

and point the config's `--glm-pack` at `/path/to/L62/pack`.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep`, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+16G64G+L62+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

DeepSeek V4 Flash 0731 の GGUF です。**GPU 16 GB + RAM 64 GB の PC** 向けです。エキスパートは 256 個すべて残しています。
速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは、RTX 5090 の VRAM を 16 GB に、RAM を 64 GB の PC と同じくらいに制限した 1 台だけです。
本物の 16 GB の GPU ではまだ測っていません。報告をいただけるととても助かります。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | 16G64G、4K | 16G64G、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L62（ここ）** | **61.7 GB** | **0.407** | **0.435** | **73.7%** | **84.5 tok/s** | **89.5 tok/s** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | 21.3 tok/s | 19.6 tok/s |

- unsloth の 96.8 GB は GPU 16 GB + RAM 64 GB に載りきらず、SSD を待ちます（21.3 tok/s）。L62 は全部載るので、この PC では
  **約 4 倍速い**です（84.5 tok/s）。
- そのかわり、品質は元のモデルから離れています（KL 0.407 対 0.186）。下の実際の作業での確認では unsloth と同じくらいの
  結果でしたが、うまくいかない場面があれば教えてください。GPU が 32 GB なら
  [L72](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G64G-GGUF)、RAM が 96 GB なら
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) をおすすめします。
- **実際の作業での確認**（Abyss v1.3.120）：`--check` は 11 項目のうち 10 項目が合格（落ちた 1 つは、日本語の説明を中国語で答えたもの）。日本語 10 問のうち 9 問が日本語のまま、事実の誤りは 0 件、
  コードは 5 問すべて合格、8K の needle も見つかりました（unsloth は同じ条件で日本語 9/10、コード 5/5）。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも起きます。元のモデルでも起きるかは、いま確かめています。システムプロンプトで「日本語で答えて」と書くと減ります。
- GPU 24 GB + RAM 64 GB は測っていませんが、L62 は余裕を持って載ります。
- DSpark（下書きモデル、VRAM を約 6.5 GB 使う）は、16 GB の GPU ではおすすめしません（`./setup-deepseek.sh --no-dspark`）。
- 本物の 16 GB の GPU は 5090 よりメモリが遅く、測った PC の RAM は速い（ふつうの DDR5 の約 3 倍）ので、ふつうの PC では
  遅くなる見込みです。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
