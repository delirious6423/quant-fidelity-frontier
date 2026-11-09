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
- local-llm
- japanese
language:
- en
- ja
library_name: gguf
---

# GLM-5.3-Flash GGUF for one DGX Spark (GB10, 128 GB) — BETA

- a mixed-type file - the experts are IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1% (by bytes), 2.26 bits a weight on average, about IQ2_XS.

A GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) that fits **one NVIDIA DGX Spark**
(GB10, 128 GB unified memory) with room left for context. **Every expert is kept** (no pruning). Speed and quality are
measured in [Abyss](https://github.com/zephel01/Abyss), **on a real DGX Spark**.

**BETA**: measured on a rented DGX Spark on 2026-11-07 (below). The speed comes from the engine: newer Abyss releases may
raise it, and this card will follow. The file may still be replaced by a better build (the folder name will change then). Reports are very
welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

**Japanese, measured separately**: on the 5 Japanese sequences of our KL set S1 is a little closer to the original than
unsloth's UD-IQ2_XXS (KL 0.251 vs 0.257). A build with more Japanese in its calibration was tried and did not do better on
general Japanese text, so S1 stays.
**日本語は別に測っています**：日本語 5 本の KL は S1 が 0.251 で、unsloth（0.257）より少し良い結果です。日本語を強めた版も
試しましたが、一般的な日本語の文章では差が出なかったので、S1 のままにします。

日本語の説明は下にあります。

## Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 94.1 GB | folder S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | 32 GB GPU + 128 GB RAM | 127.9 GB | higher quality (folder Q120d) |
| **this repo** | DGX Spark (128 GB) | 94.1 GB | same files as 32G96G |

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## The file

| | download | experts | KL (14 seqs) | KL, Japanese (5) | same top token | DGX Spark decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`S1/`](./S1) (here)** | **94.1 GB** | **all (no pruning)** | **0.2534** | **0.251** | **77.3%** | **27.4 tok/s** (Abyss v1.3.119; 21.4 without drafting) |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | all | 0.2643 | 0.257 | 76.6% | 24.2 tok/s (v1.3.119; 19.8 without drafting) |

- The same file as [`GLM-5.3-Flash-IQ2_XS-32G96G-GGUF`](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) `S1/`, where
  it runs at 75.7 tok/s on an RTX 5090 + 96 GB RAM PC.
- DGX Spark, Abyss v1.3.119: 4K prompt, median of 3 x 512 tokens. Abyss drafts with the model's own NextN block and
  checks the drafts in one batched window on the Spark (same tokens as without; drafts accepted 74-77%); v1.3.119 made
  that window faster (v1.3.118: S1 24.3, UD-IQ2_XXS 22.9 tok/s).
- KL: against unsloth's UD-Q4_K_XL, 14 sequences x 1,024 tokens (8 English, 1 code, 5 Japanese), re-measured 2026-11-08
  with the router's expert bias kept in F32 as stored in the GGUF. Earlier numbers here (S1 0.2706, UD 0.2651) came from
  an Abyss pack that rounded that bias to BF16; the GGUF files did not change. Abyss `--check` passes 12 of 12 (RTX 5090).
- On a Spark the whole file sits in unified memory; decode is bound by its 273 GB/s memory, so speed follows the bytes
  read per token. S1 reads about 8% less than unsloth's UD-IQ2_XXS at a slightly better KL (0.253 vs 0.264), and leaves
  ~8 GB more memory for long context.

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards; point your engine at `-00001-of-00004.gguf`. `SHA256SUMS` in the
folder. What is inside (quant types, how it was made): see the
[32G96G card](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF).

## Running it on a Spark

Abyss supports DGX Spark (aarch64, sm_121) - see
[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) for the build and the settings.

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+DGX-Spark+S1+on+GB10).
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- 形式を混ぜたファイルです。エキスパートは IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1%（容量の割合）、平均 2.26 ビットで、IQ2_XS くらいです。

GLM-5.3-Flash の GGUF を、**NVIDIA DGX Spark 1 台**（GB10、共有メモリ 128 GB）に、長い文脈の余裕を残して載るようにした
ものです。**エキスパートは削っていません**。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で、**本物の DGX
Spark で**測っています。

**BETA です**：2026-11-07 に、借りた DGX Spark で測りました（下の表）。速さはエンジンで決まるので、Abyss の新しい
版で上がることがあり、そのときはこのページも直します。もっと良い版に入れ替えることもあります（そのときはフォルダの名前を変えます）。

| | 大きさ | エキスパート | KL（14 本） | KL（日本語 5 本） | 同じ 1 位 | DGX Spark の生成 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | **全部（削っていない）** | **0.2534** | **0.251** | **77.3%** | **27.4 tok/s**（Abyss v1.3.119。下書きなしで 21.4） |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 全部 | 0.2643 | 0.257 | 76.6% | 24.2 tok/s（v1.3.119。下書きなしで 19.8） |

- [`GLM-5.3-Flash-IQ2_XS-32G96G-GGUF`](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) の `S1/` と同じファイルです。
  RTX 5090 + RAM 96 GB の PC では 75.7 tok/s で動きます。
- DGX Spark、Abyss v1.3.119：4K のあと、3 回の中央値 x 512 トークン。モデル自身の NextN 層で下書きを作り、Spark では
  1 つの窓でまとめて確かめます（出力は下書きなしと同じ。下書きの採用 74〜77%）。v1.3.119 でこの窓が速くなりました
  （v1.3.118 では S1 24.3、UD-IQ2_XXS 22.9 tok/s）。
- KL：unsloth の UD-Q4_K_XL と比べ、14 本 x 1,024 トークン（英語 8、コード 1、日本語 5）。2026-11-08 に、ルーターの
  エキスパートの補正値を GGUF のとおり F32 のまま使って測り直しました。前の値（S1 0.2706、UD 0.2651）は、Abyss がこの
  補正値を BF16 に丸めていたときのものです。GGUF のファイルは変わっていません。Abyss の `--check` は 12 項目すべて合格
  （RTX 5090）です。
- Spark ではファイルが全部共有メモリに載ります。生成の速さはメモリの速さ（273 GB/s）で決まるので、1 トークンで読む量で
  ほぼ決まります。S1 は unsloth の UD-IQ2_XXS より読む量が約 8% 少なく、KL も少し良く（0.253 対 0.264）、長い文脈に使える
  メモリも約 8 GB 多く残ります。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 94.1 GB | S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | GPU 32 GB + RAM 128 GB | 127.9 GB | 品質重視（Q120d） |
| **この版** | DGX Spark（128 GB） | 94.1 GB | 32G96G と同じファイル |

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
