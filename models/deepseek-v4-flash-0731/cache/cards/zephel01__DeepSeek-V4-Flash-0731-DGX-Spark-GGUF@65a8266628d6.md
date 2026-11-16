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
language:
- en
- ja
library_name: gguf
---

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | [Same file for a 32 GB GPU + 96 GB RAM PC](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | [Abyss on DGX Spark](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+DGX-Spark+L97+on+GB10) | 日本語は下

# DeepSeek V4 Flash 0731 GGUF for one DGX Spark (GB10, 128 GB) — BETA

- a mixed-type file - the experts are IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2% (by bytes), 2.64 bits a weight on average, about IQ2_M.

A 96.8 GB GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT)
that fits **one NVIDIA DGX Spark** (GB10, 128 GB unified memory) with room left for context. **Every expert is kept**
(all 256, no pruning). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA, and not run on a DGX Spark yet.** This card will get Spark numbers after a real run. Reports from Spark owners
are welcome (see Feedback).

## Model files

`L97/DeepSeek-V4-Flash-0731-L97-0000N-of-00003.gguf`, 3 shards, 96.8 GB. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the folder.

The same files as [`DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF`](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF)
`L97/`. What is inside (quant types, how it was made): see that card.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 61.7 GB | smallest (folder L62) |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | 32 GB GPU + 64 GB RAM | 72.0 GB | folder L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 96.8 GB | highest quality (folder L97) |
| **this repo** | DGX Spark (128 GB) | 96.8 GB | same files as 32G96G |

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief

- The same size as unsloth's UD-Q2_K_XL and closer to the original (KL 0.150 vs 0.186; in Japanese 0.145 vs 0.219).
- On a Spark the whole model sits in unified memory, so nothing is read from the SSD.
- **On a Spark, use the DSpark drafter.** Its 6.5 GB costs nothing there, since everything fits. Measured with
  unsloth's same-size UD-Q2_K_XL on a rented Spark (table below). On a 32 GB GPU PC it is the other way round.

## Run it on a Spark

1. Build the engine as in [docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md), steps 1-2
   (aarch64, sm_121, CUDA 12.9 or newer).
2. The model, the DSpark drafter and the pack:

```sh
cd Abyss
.venv/bin/hf download zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF --include "L97/*" --local-dir ~/models/v4-l97
(cd ~/models/v4-l97/L97 && sha256sum -c SHA256SUMS)
.venv/bin/hf download bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf \
  --revision 57fae967d0c5dac1f32a444ea8884ec46459247b --local-dir ~/models/v4-dspark
.venv/bin/python tools/iq_pack.py --compat-bf16 \
  --gguf ~/models/v4-l97/L97/DeepSeek-V4-Flash-0731-L97-00001-of-00003.gguf --out ~/models/v4-l97/L97/pack
```

3. The config and the server:

```sh
cat > abyss-v4-l97.json <<EOF
{"exe": "build-cuda/strata",
 "args": ["--glm-pack", "$HOME/models/v4-l97/L97/pack", "--max-context", "65536", "--prefill", "auto"],
 "tokenizer": "$HOME/models/v4-l97/L97/pack/tokenizer", "model_name": "deepseek-v4-flash",
 "gpu": [0], "sampling": {"temperature": 1.0, "top_p": 0.95}, "lib_dirs": ["/usr/local/cuda/lib64"],
 "port": 8082, "backend": "cuda",
 "env": {"STRATA_DS4_DSPARK": "$HOME/models/v4-dspark/DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf"}}
EOF
.venv/bin/python serve/server.py --engine strata --config abyss-v4-l97.json --port 8082
```

**llama.cpp**: not checked with this file.

## Quality

KL against unsloth's UD-Q8_K_XL (original MXFP4 experts), 12 sequences x 1,024 tokens (8 English, 1 code, 3 Japanese),
Abyss engine:

| | download | experts | KL (12 seqs) | KL, Japanese (3) | same top token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **L97 (this repo)** | **96.8 GB** | **all (no pruning)** | **0.150** | **0.145** | **84.1%** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | all | 0.186 | 0.219 | 81.5% |

- Abyss `--check` on an RTX 5090: 10 of 11 pass (the miss: a Japanese explanation answered in Chinese).
- V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct). unsloth's GGUF and the
  original weights do it too (original: 6 of 24 answers on 8 Japanese prompts x 3 seeds; L97: 11 of 24). With the
  system prompt "日本語で答えてください。" L97 switched in 1 of 24, so put "answer in Japanese" in the system prompt.

## Speed

This file has not been run on a Spark. What is measured:

| | decode | where |
| --- | ---: | --- |
| this file (L97) | 69.1 tok/s (4K prompt, no drafter) | RTX 5090 32 GB + 96 GB RAM PC, Abyss v1.3.120 |
| unsloth UD-Q2_K_XL, same size | 21.0 tok/s, 22.8 with the DSpark drafter (78% accepted) | rented DGX Spark, Abyss v1.3.114 |
| unsloth UD-Q2_K_XL, same size | 26.0 tok/s with the DSpark drafter (4K prompt, median of 3 x 512 tokens) | rented DGX Spark, Abyss v1.3.119 |
| llama.cpp, DeepSeek V4 0731 | 20.2 tok/s (8K prompt) | the same rented Spark |

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+DGX-Spark+L97+on+GB10).
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). DSpark
drafter GGUF: [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF).
Quantized with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- 形式を混ぜたファイルです。エキスパートは IQ2_XS 41%, IQ3_XXS 39%, IQ2_S 18%, MXFP4 2%（容量の割合）、平均 2.64 ビットで、IQ2_M くらいです。

DeepSeek V4 Flash 0731 の GGUF です。**DGX Spark 1 台（GB10、128 GB）** に、文脈の分の余裕を残して載ります。
エキスパートは 256 個すべて残しています。

**BETA です。このファイルは、まだ DGX Spark で動かしていません。** Spark で測ったら、このページに数字を書き足します。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 |
| --- | ---: | ---: | ---: | ---: |
| **L97（ここ）** | **96.8 GB** | **0.150** | **0.145** | **84.1%** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% |

- [32G96G](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) の `L97/` と同じファイルです。RTX 5090 + RAM 96 GB の
  PC では 69.1 tok/s でした。
- unsloth の UD-Q2_K_XL と同じ大きさで、元のモデルにより近い結果です（KL 0.150 対 0.186。日本語では 0.145 対 0.219）。
- 同じ大きさの unsloth UD-Q2_K_XL を、借りた Spark で測った値：DSpark ありで 26.0 tok/s（Abyss v1.3.119）。
- **Spark では DSpark（下書きモデル）を使ってください。** Spark ではモデル全体がメモリに載るので、下書きモデルの 6.5 GB で
  遅くなることはありません。GPU 32 GB の PC では逆に遅くなります。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも、元のモデルでも起きます（日本語 8 問 × seed 3 つの 24 回答で、元のモデル 6 回、L97 11 回）。システムプロンプトに「日本語で答えてください。」を入れると、L97 で 1 回に減りました。
- Spark でのビルドと設定は [docs/DGX-SPARK.ja.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.ja.md) にあります。
  モデルと pack と設定ファイルのコマンドは、上の英語の節のとおりです。

感想や速さの報告は、Community タブか [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+DGX-Spark+L97+on+GB10) にお願いします。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 61.7 GB | いちばん小さい（L62） |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | GPU 32 GB + RAM 64 GB | 72.0 GB | L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 96.8 GB | 品質がいちばん上（L97） |
| **この版** | DGX Spark（128 GB） | 96.8 GB | 32G96G と同じファイル |

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
