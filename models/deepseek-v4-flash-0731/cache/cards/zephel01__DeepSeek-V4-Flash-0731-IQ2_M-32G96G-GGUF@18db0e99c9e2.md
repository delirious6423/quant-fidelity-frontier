---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- imatrix
- moe
- abyss
- consumer-gpu
- local-llm
- cpu-offload
- japanese
- rtx-5090
- 32gb-vram
- 96gb-ram
language:
- en
- ja
library_name: gguf
---

# DeepSeek V4 Flash 0731 GGUF (for a 32 GB GPU + 96 GB RAM PC) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2% (by bytes), 2.64 bits a weight on average, about IQ2_M.

A GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT) sized
for **one 32 GB GPU (RTX 5090, AI PRO R9700) plus 96 GB RAM**. **Every expert is kept** (all 256, no pruning). Speed and
quality are measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far. The file may be replaced by a better build after more measurements (the folder
name will change then). Reports are very welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

Other sizes of the same build: [32G64G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) (72.0 GB),
[16G64G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) (61.7 GB),
[DGX Spark](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) (this file).

日本語の説明は下にあります。

## Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 61.7 GB | smallest (folder L62) |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | 32 GB GPU + 64 GB RAM | 72.0 GB | folder L72 |
| **this repo** | 32 GB GPU + 96 GB RAM | 96.8 GB | highest quality (folder L97) |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark (128 GB) | 96.8 GB | same files as 32G96G |

Earlier repo name: DeepSeek-V4-Flash-0731-32G96G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## The file

| | download | KL (12 seqs) | KL, Japanese (3) | same top token | RTX 5090, 4K prompt | RTX 5090, 16K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`L97/`](./L97) (here)** | **96.8 GB** | **0.150** | **0.145** | **84.1%** | **69.1 tok/s** | **65.9 tok/s** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | 0.186 | 0.219 | 81.5% | 65.1 tok/s | 63.9 tok/s |

- The same size as unsloth's UD-Q2_K_XL, **closer to the original** (KL 0.150 vs 0.186; in Japanese 0.145 vs 0.219)
  and a little faster on this PC (+6% at 4K, +3% at 16K).
- **Use it without the DSpark drafter on a 32 GB GPU.** The drafter takes about 6.5 GB of VRAM, so fewer experts fit on
  the GPU and more are read from the SSD: 69.1 -> 62.5 tok/s at 4K and 65.9 -> 56.4 at 16K (unsloth's UD-Q2_K_XL:
  65.1 -> 56.7). Answer "no" when Abyss's setup asks about the drafter, or run `./setup-deepseek.sh --no-dspark`.
  On a DGX Spark the drafter does help (see that repo).

## Files

`L97/DeepSeek-V4-Flash-0731-L97-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` in the folder.

- Routed experts (all 256 in every layer), re-quantized with LDLQ from unsloth's UD-Q8_K_XL (whose experts are the
  original MXFP4 weights), using a Hessian dump of V4's own chat answers (241 prompts, 104 of them Japanese, in V4's
  template) plus general text, and our own imatrix (about 1M tokens; unsloth has not published one for V4):
  gate / up IQ2_XS (layers 31-42 IQ2_S, layer 26 IQ3_XXS), down IQ3_XXS (layers 26 and 42 kept in MXFP4).
- Attention `attn_q_b`, `attn_output_a`, `attn_output_b`: Q5_K.
- Everything else (the other dense tensors, router, embeddings, output, hash tables) is unsloth UD-Q2_K_XL's bytes.
  The first shard (metadata only, 5 MB) is unsloth's file as is, so its `general.quantized_by` still says Unsloth.

## Quality

KL against unsloth's UD-Q8_K_XL (original MXFP4 experts; dense tensors as Q8_0), top-k log-probs, 12 sequences x
1,024 tokens (8 English text, 1 code, 3 Japanese), Abyss engine. Same top token: how often the top-1 token matches.

Real use (Abyss v1.3.120, RTX 5090, 32G96G limits):

- Abyss `--check`: **10 of 11 pass (Japanese, three coding tasks, tool call, 8K-31K needle, 11-turn conversation); the one miss is a Japanese explanation answered in Chinese (see below). unsloth's UD-Q2_K_XL: 11 of 11**
- Japanese, code, 8K needle (10 Japanese answers, 5 coding tasks with tests): 7/10 Japanese answers stayed in Japanese,
  0 wrong facts, 4/5 coding tasks pass (one test failed on the model's own code), needle found. unsloth's UD-Q2_K_XL on
  the same prompts: 7/10, 0 wrong facts, 5/5.
- **V4 0731 sometimes answers a Japanese question in Chinese** (the answer itself is correct). This happens with
  unsloth's GGUF too, and **the original weights do it as well**: 8 Japanese prompts x 3 seeds (temperature 0.6), answers that
  switched to Chinese - the original (UD-Q8_K_XL, original MXFP4 experts) 6 of 24, L97 11 of 24, **L97 with the system prompt
  "日本語で答えてください。" 1 of 24**. So it comes from the model; this quantization makes it a little more frequent. Put
  "answer in Japanese" in the system prompt.
- llama.cpp: not checked yet. The tensor types are standard ggml types (IQ2_XS, IQ2_S, IQ3_XXS, MXFP4, Q5_K and
  unsloth's dense types) in unsloth's layout.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.120, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395
(LPDDR5X, about 3x the bandwidth of a usual dual-channel DDR5 desktop), RAM tier 60 GB (`STRATA_GLM_RAM_GB=60`) under an
84 GB memory limit, NVMe SSD. Temperature 0, 512 tokens after a 4K / 16K prompt, page cache dropped before each run,
median of 3.

- On a usual DDR5 desktop the part computed on the CPU side is slower, so expect lower numbers; reports welcome.
- Other GPUs are not measured yet.

## Running it

[Abyss](https://github.com/zephel01/Abyss) v1.3.120 or later. Set up DeepSeek V4 once (`./setup-deepseek.sh`, without
`--download-model` if you only want this file), then pack this GGUF:

```sh
cd Abyss
.venv/bin/python tools/iq_pack.py --compat-bf16 --gguf /path/to/L97/DeepSeek-V4-Flash-0731-L97-00001-of-00003.gguf --out /path/to/L97/pack
```

and point the config's `--glm-pack` at `/path/to/L97/pack`.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+32G96G+L97+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2%（容量の割合）、平均 2.64 ビットで、IQ2_M くらいです。

DeepSeek V4 Flash 0731 の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 96 GB の PC** 向けです。
エキスパートは 256 個すべて残しています（削っていません）。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです。測り直した結果によっては、もっと良い版に入れ替えることがあります
（そのときはフォルダの名前を変えます）。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | RTX 5090、4K | RTX 5090、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L97（ここ）** | **96.8 GB** | **0.150** | **0.145** | **84.1%** | **69.1 tok/s** | **65.9 tok/s** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | 65.1 tok/s | 63.9 tok/s |

- unsloth の UD-Q2_K_XL と同じ大きさで、**元のモデルにより近い**結果です（KL 0.150 対 0.186。日本語では 0.145 対 0.219）。
  この PC では少し速くもなりました（4K で +6%、16K で +3%）。
- **GPU 32 GB では、DSpark（下書きモデル）なしで使ってください。** 下書きモデルが VRAM を約 6.5 GB 使うぶん、GPU に置ける
  エキスパートが減り、SSD から読む分が増えます。4K で 69.1 → 62.5 tok/s、16K で 65.9 → 56.4 tok/s に下がりました。
  Abyss のセットアップで下書きモデルを聞かれたら「いいえ」にするか、`./setup-deepseek.sh --no-dspark` を使ってください。
  DGX Spark では、下書きモデルで速くなります（Spark 用のリポジトリを見てください）。
- **実際の作業での確認**（Abyss v1.3.120、RTX 5090）：`--check` は 11 項目のうち 10 項目が合格（落ちた 1 つは、日本語の説明を中国語で答えたもの。unsloth の UD-Q2_K_XL は 11 項目すべて合格）。日本語 10 問、コード 5 問、8K の needle では、
  事実の誤りは 0 件、コードは 5 問中 4 問合格（1 問はモデルが書いたコードの誤り）、needle は見つかりました。
- **V4 0731 は、日本語で聞いても中国語で答えることがあります**（答えの中身は正しいです）。unsloth の GGUF でも起きます。
  **元のモデルでも起きます。** 日本語の 8 問を seed 3 つで聞いたとき（温度 0.6、24 回答）、中国語になったのは、
  元のモデル（UD-Q8_K_XL、エキスパートは元の MXFP4）が 6 回、L97 が 11 回、**システムプロンプトに「日本語で答えてください。」を
  入れた L97 は 1 回**でした。モデルのくせで、この量子化で少し増えています。システムプロンプトで日本語を指定してください。
- 中身：ルーティングされるエキスパートを、unsloth の UD-Q8_K_XL（エキスパートは元の MXFP4 のまま）から LDLQ で作り直しました。
  Hessian は V4 自身のチャットの答え（241 問、うち日本語 104 問）と一般の文章から取りました。imatrix も自作です（約 100 万トークン）。
  attention の 3 つ（`attn_q_b`、`attn_output_a`、`attn_output_b`）は Q5_K、ほかは unsloth の UD-Q2_K_XL のバイトのままです。
  1 つ目のファイル（メタデータだけ、5 MB）は unsloth のものをそのまま使っているので、`general.quantized_by` は Unsloth のままです。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 のデスクトップの約 3 倍）ので、ふつうの PC では CPU 側の分だけ遅くなる
  見込みです。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
`./abyss.sh --bench --deep` の結果（`abyss-bench.txt`）を、GPU、RAM、SSD と一緒に貼ってもらえると助かります。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 61.7 GB | いちばん小さい（L62） |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | GPU 32 GB + RAM 64 GB | 72.0 GB | L72 |
| **この版** | GPU 32 GB + RAM 96 GB | 96.8 GB | 品質がいちばん上（L97） |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark（128 GB） | 96.8 GB | 32G96G と同じファイル |

以前のリポジトリ名は DeepSeek-V4-Flash-0731-32G96G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
