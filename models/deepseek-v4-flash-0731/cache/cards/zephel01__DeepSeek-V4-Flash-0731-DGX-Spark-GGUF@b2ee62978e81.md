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
- dgx-spark
- gb10
- local-llm
- japanese
---

# DeepSeek V4 Flash 0731 GGUF for one DGX Spark (GB10, 128 GB) — BETA

- a mixed-type file - the experts are IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2% (by bytes), 2.64 bits a weight on average, about IQ2_M.

A GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT) that
fits **one NVIDIA DGX Spark** (GB10, 128 GB unified memory) with room left for context. **Every expert is kept** (all
256, no pruning). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA, and the Spark speed below is an estimate**: this file has not been run on a DGX Spark yet. The estimate comes
from unsloth's UD-Q2_K_XL measured on a rented Spark (same size; a Spark's decode follows the bytes read per token).
This card will be updated after a real Spark run. Reports are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

日本語の説明は下にあります。

## The file

| | download | experts | KL (12 seqs) | KL, Japanese (3) | same top token | DGX Spark decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`L97/`](./L97) (here)** | **96.8 GB** | **all (no pruning)** | **0.150** | **0.145** | **84.1%** | **about 26 tok/s (estimate)** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | all | 0.186 | 0.219 | 81.5% | 26.0 tok/s with the DSpark drafter (measured, Abyss v1.3.119) |

- The same file as [`DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF`](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF)
  `L97/`, where it runs at 69.1 tok/s on an RTX 5090 + 96 GB RAM PC.
- The same size as unsloth's UD-Q2_K_XL and closer to the original (KL 0.150 vs 0.186; in Japanese 0.145 vs 0.219).
- **On a Spark, use the DSpark drafter** (`./setup-deepseek.sh --dspark`): the whole model sits in unified memory, so
  the drafter's 6.5 GB costs nothing in speed there (unsloth's UD-Q2_K_XL: 21.0 -> 22.8 tok/s with v1.3.114, 26.0 with
  v1.3.119). On a 32 GB GPU PC it is the other way round.
- KL: against unsloth's UD-Q8_K_XL (original MXFP4 experts), 12 sequences x 1,024 tokens (8 English, 1 code,
  3 Japanese). Abyss `--check` on an RTX 5090: 10 of 11 pass (the miss: a Japanese explanation answered in Chinese).
- V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct); unsloth's GGUF does it too. The original weights do it too (6 of 24 answers on 8 Japanese prompts x 3 seeds; L97 11 of 24); with the system prompt "日本語で答えてください。" L97 switched in 1 of 24, so put "answer in Japanese" in the system prompt.

`L97/DeepSeek-V4-Flash-0731-L97-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` in the folder. What is inside (quant types, how it was made): see the
[32G96G card](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF).

## Running it on a Spark

Abyss supports DGX Spark (aarch64, sm_121) - see
[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) for the build and the settings.

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+DGX-Spark+L97+on+GB10).
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- 形式を混ぜたファイルです。エキスパートは IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2%（容量の割合）、平均 2.64 ビットで、IQ2_M くらいです。

DeepSeek V4 Flash 0731 の GGUF です。**DGX Spark 1 台（GB10、128 GB）** に、文脈の分の余裕を残して載ります。
エキスパートは 256 個すべて残しています。

**BETA です。Spark での速さは見込みの値です**：このファイルは、まだ DGX Spark で動かしていません。同じ大きさの unsloth の
UD-Q2_K_XL を Spark で測った値から見積もりました（Spark の速さは、1 トークンで読むバイト数でほぼ決まります）。Spark で
測ったら、このページを書き直します。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | DGX Spark |
| --- | ---: | ---: | ---: | ---: | ---: |
| **L97（ここ）** | **96.8 GB** | **0.150** | **0.145** | **84.1%** | **約 26 tok/s（見込み）** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | DSpark ありで 26.0 tok/s（実測、v1.3.119） |

- [32G96G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) の `L97/` と同じファイルです。RTX 5090 + RAM 96 GB の
  PC では 69.1 tok/s でした。
- unsloth の UD-Q2_K_XL と同じ大きさで、元のモデルにより近い結果です（KL 0.150 対 0.186。日本語では 0.145 対 0.219）。
- **Spark では DSpark（下書きモデル）を使ってください**（`./setup-deepseek.sh --dspark`）。Spark ではモデル全体がメモリに
  載るので、下書きモデルの 6.5 GB で遅くなることはありません。GPU 32 GB の PC では逆に遅くなります。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも起きます。元のモデルでも起きます（日本語 8 問 × seed 3 つの 24 回答で、元のモデル 6 回、L97 11 回）。システムプロンプトに「日本語で答えてください。」を入れると、L97 で 1 回に減りました。
- Spark でのビルドと設定は [docs/DGX-SPARK.ja.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.ja.md) にあります。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
