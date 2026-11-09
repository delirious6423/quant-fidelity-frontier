---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- imatrix
- moe
- abyss
- dgx-spark
- gb10
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

# GLM-5.3-Flash GGUF (for a 32 GB GPU + 96 GB RAM PC, and DGX Spark) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1% (by bytes), 2.26 bits a weight on average, about IQ2_XS.

GGUFs of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) sized for **one 32 GB GPU
(RTX 5090, AI PRO R9700) plus 96 GB RAM**, and for **one DGX Spark (128 GB unified memory)**. Every expert is kept
(no pruning). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far; real-use checks and llama.cpp loading are still in progress and this card will be
updated with them. The file may be replaced by a better build after more measurements (the folder name will change
then). Reports are very welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

**Japanese, measured separately**: our KL set has 5 Japanese sequences and we report them on their own. On them S1 is
a little closer to the original than unsloth's UD-IQ2_XXS (KL 0.251 vs 0.257). A build with more Japanese in its
calibration was tried and did not do better on general Japanese text, so S1 stays. Best Japanese now: the
quality-first [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF).
**日本語は別に測っています**：日本語 5 本の KL は S1 が 0.251 で、unsloth の UD-IQ2_XXS（0.257）より少し良い結果です。
日本語を強めた版も試しましたが、一般的な日本語の文章では差が出なかったので、S1 のままにします。いま日本語がいちばん良いのは
[32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) です。

日本語の説明は下にあります。

## Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 32 GB GPU + 96 GB RAM | 94.1 GB | folder S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | 32 GB GPU + 128 GB RAM | 127.9 GB | higher quality (folder Q120d) |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark (128 GB) | 94.1 GB | same files as 32G96G |

Earlier repo name: GLM-5.3-Flash-32G96G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## Speed depends on whether it fits

**Fast when the whole file fits in GPU memory + RAM; slow when part of it has to be read from the SSD.** Abyss keeps
the most-used experts in VRAM, the rest in RAM, and reads what does not fit from the SSD each token - and those SSD
reads dominate the time. S1 is 94.1 GB so that it fits a 32 GB GPU + 96 GB RAM (75.7 tok/s); unsloth's 101.8 GB
UD-IQ2_XXS does not quite fit and waits for the SSD (53.3 tok/s), and our 127.9 GB quality-first build drops to
23 tok/s on the same PC. With less RAM (e.g. 64 GB) part of S1 is read from the SSD and it is slower; with 128 GB RAM
or more, the quality-first [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) fits too.

## Which one

| | download | KL (14 seqs) | KL, Japanese (5) | same top token | RTX 5090, 4K prompt | RTX 5090, 16K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`S1/`](./S1) (here)** | **94.1 GB** | **0.2534** | **0.251** | **77.3%** | **75.7 tok/s** | **69.8 tok/s** |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | 0.2643 | 0.257 | 76.6% | 53.3 tok/s | 50.5 tok/s |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) (32G128G) | 127.9 GB | 0.1367 * | 0.145 * | 84.9% * | 23.2 tok/s (SSD-bound here) | 21.1 tok/s |

\* Q120d: the older 10-sequence measurement (see Quality); not directly comparable with the 14-sequence numbers.

- **S1** is a little closer to the original than unsloth's UD-IQ2_XXS (KL 0.253 vs 0.264; better in English,
  a little behind on the one code sequence) and **42% faster on a 32 GB GPU + 96 GB RAM PC**: at 94 GB it fits in VRAM + RAM, so it almost
  never waits for the SSD (0.3 ms a token against 3.2 ms).
- A larger, higher-quality file (KL about half) is being evaluated; it needs more than 96 GB RAM to be fast.

## Files

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards; point your engine at `-00001-of-00004.gguf`. `SHA256SUMS` in the
folder.

- Routed experts (layers 3-44), re-quantized with LDLQ from unsloth's UD-Q4_K_XL using a Hessian dump of chat
  transcripts in GLM's template (English, code and Japanese) and unsloth's imatrix:
  gate / up IQ2_XXS (layer 11 IQ2_S), down IQ2_S (layers 11, 12, 44 IQ3_XXS).
- Dense layers: attention output, attention q_b and the dense / shared-expert down Q5_K; the other attention projections,
  dense / shared-expert gate / up and the NextN projection Q4_K.
- Router, embeddings, output and the rest of the NextN (MTP) layer: unsloth UD-IQ2_XXS's bytes.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 14 sequences x 1,024 tokens: 8 English text, 1 code, 5 Japanese),
Abyss engine, re-measured 2026-11-08 with the router's expert bias (`exp_probs_b`) kept in F32 as stored in the GGUF.
Earlier numbers on this card (S1 0.2706, UD-IQ2_XXS 0.2651, 10 sequences) came from an Abyss pack that rounded that
bias to BF16, which hurt S1 more than UD-IQ2_XXS; Abyss keeps it in F32 from the next release. The GGUF files did not
change. Real-use checks: Abyss `--check` passes 12 of 12 (Japanese, three coding tasks, tool call, 8K-63K needle,
11-turn conversation) on an RTX 5090. A llama.cpp run: **in progress, results will be added here**.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.117, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395
(LPDDR5X, about 3x the bandwidth of a usual dual-channel DDR5 desktop), RAM tier 60 GB (`STRATA_GLM_RAM_GB=60`), NVMe
SSD. Temperature 0, 512 tokens, page cache dropped before each run, median of 3 (4K) / 1 run (16K).

- On a usual DDR5 desktop the part computed on the CPU side is slower, so expect lower numbers; reports welcome.
- **DGX Spark** (measured 2026-11-07, Abyss v1.3.119): everything fits in its memory; **27.4 tok/s** (21.4 without
  drafting; unsloth's UD-IQ2_XXS 24.2 / 19.8). See [the DGX Spark repo](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF).
- Other GPUs are not measured yet.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G96G+S1+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.
- **What you would rather have**: a smaller one for 64 GB RAM, a faster one for DGX Spark, other GPUs.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1%（容量の割合）、平均 2.26 ビットで、IQ2_XS くらいです。

GLM-5.3-Flash の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 96 GB の PC** と、**DGX Spark 1 台**向けです。
エキスパートは削っていません。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです。llama.cpp で読めるかの確認はまだ途中で、終わりしだいここに書き足します。
測り直した結果によっては、もっと良い版に入れ替えることがあります（そのときはフォルダの名前を変えます）。

**速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで大きく変わります。全部載れば速く、載りきらずに SSD から読む分が
あると遅くなります。** S1 は、GPU 32 GB + RAM 96 GB に全部載るように 94.1 GB にしました（75.7 tok/s）。同じ PC で、少し
はみ出す unsloth の 101.8 GB は SSD を待って 53.3 tok/s、品質優先の 127.9 GB の版は 23 tok/s です。RAM が少ない PC
（例：64 GB）では S1 も一部を SSD から読むので遅くなります。RAM が 128 GB 以上あれば、品質優先の
[32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) も全部載ります。

| | 大きさ | KL（14 本） | KL（日本語 5 本） | 同じ 1 位 | RTX 5090、4K | RTX 5090、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | **0.2534** | **0.251** | **77.3%** | **75.7 tok/s** | **69.8 tok/s** |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2643 | 0.257 | 76.6% | 53.3 tok/s | 50.5 tok/s |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF)（32G128G） | 127.9 GB | 0.1367 * | 0.145 * | 84.9% * | 23.2 tok/s（この PC では SSD 待ち） | 21.1 tok/s |

\* Q120d は前の 10 本での測り方の値で、14 本の値とはそのまま比べられません。

- **品質**：S1 は unsloth の UD-IQ2_XXS より、元のモデルに少し近い結果です（KL 0.253 対 0.264。英語では良く、
  1 本だけのコードでは少し負けます）。2026-11-08 に、ルーターのエキスパートの補正値（`exp_probs_b`）を GGUF のとおり F32 のまま使って測り直しました。
  前にこのページに書いた値（S1 0.2706、UD 0.2651、10 本）は、Abyss がこの補正値を BF16 に丸めていたときのもので、S1 のほうが
  大きく損をしていました。Abyss は次のリリースから F32 のまま使います。GGUF のファイルは変わっていません。
- **日本語**：日本語 5 本の KL は S1 0.251、unsloth 0.257 です。日本語を強めた版も試しましたが、一般的な日本語の文章では
  差が出なかったので、S1 のままにします。
- **実際の作業での確認**：Abyss の `--check` は RTX 5090 で 12 項目すべて合格です（日本語、コード 3 問、ツール呼び出し、
  8K〜63K の needle、11 ターンの会話）。
- GPU 32 GB + RAM 96 GB の PC では **42% 速い**です。94 GB だと VRAM と RAM にほぼ全部載るので、SSD をほとんど待ちません。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 のデスクトップの約 3 倍）ので、ふつうの PC では CPU 側の分だけ遅くなる
  見込みです。
- DGX Spark では、全部メモリに載ります。2026-11-07 の実測（Abyss v1.3.119）で **27.4 tok/s**（下書きなしで 21.4。unsloth の
  UD-IQ2_XXS は 24.2 / 19.8）です。
- 品質を上げた大きめの版（KL が約半分）も評価しています。速く動かすには RAM が 96 GB より多く必要です。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
`./abyss.sh --bench --deep` の結果（`abyss-bench.txt`）を、GPU、RAM、SSD と一緒に貼ってもらえると助かります。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 32 GB + RAM 96 GB | 94.1 GB | S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | GPU 32 GB + RAM 128 GB | 127.9 GB | 品質重視（Q120d） |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark（128 GB） | 94.1 GB | 32G96G と同じファイル |

以前のリポジトリ名は GLM-5.3-Flash-32G96G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
