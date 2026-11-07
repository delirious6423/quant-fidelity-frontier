---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, imatrix, moe, abyss]
---

# DeepSeek V4 Flash 0731 GGUF (for a 32 GB GPU + 64 GB RAM PC) — BETA

A GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT) sized
for **one 32 GB GPU (RTX 5090, AI PRO R9700) plus 64 GB RAM**. **Every expert is kept** (all 256, no pruning). Speed and
quality are measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far, with its RAM limited to a 64 GB PC's share (below). The file may be replaced by a
better build after more measurements (the folder name will change then). Reports are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

Other sizes of the same build: [32G96G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) (96.8 GB,
better quality), [16G64G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-16G64G-GGUF) (61.7 GB),
[DGX Spark](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF).

日本語の説明は下にあります。

## The file

| | download | KL (12 seqs) | KL, Japanese (3) | same top token | 32G64G, 4K prompt | 32G64G, 16K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`L72/`](./L72) (here)** | **72.0 GB** | **0.314** | **0.331** | **76.9%** | **90.0 tok/s** | **86.4 tok/s** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | 0.186 | 0.219 | 81.5% | 33.4 tok/s | 34.0 tok/s |
| [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) (32G96G) | 96.8 GB | 0.150 | 0.145 | 84.1% | - | - |

- **Speed depends on whether the file fits in GPU memory + RAM.** unsloth's 96.8 GB UD-Q2_K_XL does not fit a 32 GB
  GPU + 64 GB RAM PC and waits for the SSD every token (33.4 tok/s); L72 fits and runs **2.7x faster** (90.0 tok/s).
- The price is quality: L72 is further from the original than UD-Q2_K_XL (KL 0.314 vs 0.186). In the real-use checks
  below it did as well as UD-Q2_K_XL; tell us where it does not. With 96 GB RAM or more, take
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF).
- The DSpark drafter is not measured at this size; on 32G96G it made decoding slower (it takes ~6.5 GB of VRAM), so
  start without it (`./setup-deepseek.sh --no-dspark`).

## Files

`L72/DeepSeek-V4-Flash-0731-L72-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` in the folder.

- Routed experts (all 256 in every layer), re-quantized with LDLQ from unsloth's UD-Q8_K_XL (original MXFP4 experts),
  using a Hessian dump of V4's own chat answers (241 prompts, 104 of them Japanese) plus general text, and our own
  imatrix: gate / up IQ1_S, down IQ2_S (layers 36-42 IQ3_XXS).
- Attention `attn_q_b`, `attn_output_a`, `attn_output_b`: Q5_K. Everything else is unsloth UD-Q2_K_XL's bytes; the first
  shard (metadata only, 5 MB) is unsloth's file as is, so its `general.quantized_by` still says Unsloth.
- How it was made and measured in detail: see the [32G96G card](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF).

## Quality

KL against unsloth's UD-Q8_K_XL, 12 sequences x 1,024 tokens (8 English text, 1 code, 3 Japanese), Abyss engine.

Real use (Abyss v1.3.120, RTX 5090, 32G64G limits):

- Abyss `--check`: **11 of 11 pass (Japanese, three coding tasks, tool call, 8K-31K needle, 11-turn conversation). unsloth's UD-Q2_K_XL: 11 of 11**
- 10 Japanese answers, 5 coding tasks with tests, 8K needle: 9/10 Japanese answers stayed in Japanese, 0 wrong facts,
  5/5 coding tasks pass, needle found (unsloth's UD-Q2_K_XL on the same prompts and limits: 8/10, 0 wrong facts, 5/5).
- V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct); unsloth's GGUF does it too. Whether the original weights do it as well is being checked; a system prompt asking for Japanese answers helps.
- llama.cpp: not checked yet.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.120 on an RTX 5090 (32 GB) + Ryzen AI Max+ 395 PC, limited to what a
64 GB RAM PC has: RAM tier 40 GB (`STRATA_GLM_RAM_GB=40`) under a 58 GB memory limit. Temperature 0, 512 tokens after
a 4K / 16K prompt, page cache dropped before each run, median of 3. This PC's RAM (LPDDR5X) is about 3x a usual DDR5
desktop's, so expect lower numbers on a usual PC; reports welcome.

## Running it

[Abyss](https://github.com/zephel01/Abyss) v1.3.120 or later. Set up DeepSeek V4 once (`./setup-deepseek.sh --no-dspark`),
then pack this GGUF:

```sh
cd Abyss
.venv/bin/python tools/iq_pack.py --compat-bf16 --gguf /path/to/L72/DeepSeek-V4-Flash-0731-L72-00001-of-00003.gguf --out /path/to/L72/pack
```

and point the config's `--glm-pack` at `/path/to/L72/pack`.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep`, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+32G64G+L72+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

DeepSeek V4 Flash 0731 の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 96 GB ではなく 64 GB の PC** 向けです。
エキスパートは 256 個すべて残しています。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです（RAM を 64 GB の PC と同じくらいに制限して測りました）。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | 32G64G、4K | 32G64G、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L72（ここ）** | **72.0 GB** | **0.314** | **0.331** | **76.9%** | **90.0 tok/s** | **86.4 tok/s** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | 33.4 tok/s | 34.0 tok/s |

- **速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで決まります。** unsloth の 96.8 GB は GPU 32 GB + RAM 64 GB に
  載りきらず、毎トークン SSD を待ちます（33.4 tok/s）。L72 は全部載るので **2.7 倍速い**です（90.0 tok/s）。
- そのかわり、品質は unsloth の UD-Q2_K_XL より元のモデルから離れています（KL 0.314 対 0.186）。下の実際の作業での確認では
  同じくらいの結果でしたが、うまくいかない場面があれば教えてください。RAM が 96 GB 以上あるなら
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) をおすすめします。
- **実際の作業での確認**（Abyss v1.3.120）：`--check` は 11 項目すべて合格。日本語 10 問のうち 9 問が日本語のまま、事実の誤りは 0 件、
  コードは 5 問すべて合格、8K の needle も見つかりました（unsloth は同じ条件で日本語 8/10、コード 5/5）。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも起きます。元のモデルでも起きるかは、いま確かめています。システムプロンプトで「日本語で答えて」と書くと減ります。
- DSpark（下書きモデル）はこの大きさでは測っていません。32G96G では遅くなったので、なしで始めてください
  （`./setup-deepseek.sh --no-dspark`）。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 の約 3 倍）ので、ふつうの PC では遅くなる見込みです。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
