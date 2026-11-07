---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, imatrix, moe, abyss, dgx-spark, gb10]
---

# GLM-5.3-Flash GGUF for one DGX Spark (GB10, 128 GB) — BETA

A GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) that fits **one NVIDIA DGX Spark**
(GB10, 128 GB unified memory) with room left for context. **Every expert is kept** (no pruning). Speed and quality are
measured in [Abyss](https://github.com/zephel01/Abyss), **on a real DGX Spark**.

**BETA**: measured on a rented DGX Spark on 2026-11-07 (below). A faster Spark-tuned build (target 25-30 tok/s) is in the
works and may replace this file (the folder name will change then). Reports are very
welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

**Japanese, measured separately**: on the Japanese text of our KL set S1 is a little behind unsloth's UD-IQ2_XXS (KL 0.280
vs 0.240, one 1,024-token sequence); a build with more Japanese in its calibration is planned.
**日本語は別に測っています**：S1 の日本語の KL は 0.280 で、unsloth（0.240）より少し悪い結果です。日本語を強めた版を作る予定です。

日本語の説明は下にあります。

## The file

| | download | experts | KL (10 seqs) | KL, Japanese | same top token | DGX Spark decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`S1/`](./S1) (here)** | **94.1 GB** | **all (no pruning)** | 0.2706 | 0.280 | 79.1% | **24.3 tok/s** (Abyss v1.3.118; 21.4 without drafting) |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | all | 0.2651 | 0.240 | 79.0% | 22.9 tok/s (v1.3.118; 19.8 without drafting) |

- The same file as [`GLM-5.3-Flash-32G96G-GGUF`](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF) `S1/`, where
  it runs at 75.7 tok/s on an RTX 5090 + 96 GB RAM PC.
- DGX Spark, Abyss v1.3.118: 4K prompt, median of 3 x 512 tokens. v1.3.118 drafts with the model's own NextN block and
  checks the drafts in one batched window on the Spark (same tokens as without; drafts accepted 74%).
- On a Spark the whole file sits in unified memory; decode is bound by its 273 GB/s memory, so speed follows the bytes
  read per token. S1 reads about 8% less than unsloth's UD-IQ2_XXS at about the same quality (KL +0.006), and leaves
  ~8 GB more memory for long context.

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards; point your engine at `-00001-of-00004.gguf`. `SHA256SUMS` in the
folder. What is inside (quant types, how it was made): see the
[32G96G card](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF).

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

GLM-5.3-Flash の GGUF を、**NVIDIA DGX Spark 1 台**（GB10、共有メモリ 128 GB）に、長い文脈の余裕を残して載るようにした
ものです。**エキスパートは削っていません**。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で、**本物の DGX
Spark で**測っています。

**BETA です**：2026-11-07 に、借りた DGX Spark で測りました（下の表）。もっと速い Spark 向けの版（25〜30 tok/s が目標）も
作っていて、できたら入れ替えることがあります（そのときはフォルダの名前を変えます）。

| | 大きさ | エキスパート | KL（10 本） | KL（日本語） | 同じ 1 位 | DGX Spark の生成 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | **全部（削っていない）** | 0.2706 | 0.280 | 79.1% | **24.3 tok/s**（Abyss v1.3.118。下書きなしで 21.4） |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 全部 | 0.2651 | 0.240 | 79.0% | 22.9 tok/s（v1.3.118。下書きなしで 19.8） |

- [`GLM-5.3-Flash-32G96G-GGUF`](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF) の `S1/` と同じファイルです。
  RTX 5090 + RAM 96 GB の PC では 75.7 tok/s で動きます。
- DGX Spark、Abyss v1.3.118：4K のあと、3 回の中央値 x 512 トークン。v1.3.118 は、モデル自身の NextN 層で下書きを作り、
  Spark では 1 つの窓でまとめて確かめます（出力は下書きなしと同じ。下書きの採用 74%）。
- Spark ではファイルが全部共有メモリに載ります。生成の速さはメモリの速さ（273 GB/s）で決まるので、1 トークンで読む量で
  ほぼ決まります。S1 は unsloth の UD-IQ2_XXS より読む量が約 8% 少なく、品質はほぼ同じ（KL +0.006）で、長い文脈に使える
  メモリも約 8 GB 多く残ります。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
