---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- qwen4exp
- moe
- abyss
- japanese
- dgx-spark
- gb10
- local-llm
---

# Qwen3.8-Flash-Next GGUF for one DGX Spark (GB10, 128 GB) — BETA

- a mixed-type file - the experts are IQ3_XXS 56%, IQ4_NL 42%, IQ3_S 1% (by bytes), 3.55 bits a weight on average, about IQ3_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for **one NVIDIA DGX Spark** (GB10,
128 GB unified memory). **All 512 experts are kept** (no pruning). Speed is measured in
[Abyss](https://github.com/zephel01/Abyss) **on a real DGX Spark**.

These are the same files as [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)
(byte for byte, same `SHA256SUMS`). This repo is the Spark entry point: the Spark numbers and how to run it there. How the
file was made and the full quality comparison are on the 16G64G card.

**BETA**: measured on a rented DGX Spark (2026-11-05 to 11-07). The speed comes from the engine: newer Abyss releases may
raise it, and this card will follow. Reports from your Spark are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

日本語の説明は下にあります。

## The file

| | download | experts | KL (10 seqs) | KL, Japanese | DGX Spark decode, 4K prompt | after a 32K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **this file** | **86.9 GB** | **all 512** | **0.083** | **0.176** | **55.8 tok/s** | **53.4 tok/s** |
| unsloth UD-IQ3_XXS (reference) | 82.0 GB | all 512 | 0.154 | 0.317 | - | - |

- DGX Spark, Abyss v1.3.119, the MTP drafter on, 512 tokens after the prompt (v1.3.118: 54.7 / 47.5 tok/s).
- Prompt processing on the Spark: about 1,040-1,100 tok/s from 4K to 64K (Abyss v1.3.113).
- The whole file fits the Spark's memory; the 28.8 GB per-layer embedding table is read a few rows a token from the SSD.
- KL against the NVFP4 model, 10 held-out sequences x 1,023 tokens (code, English, Chinese, Japanese and others).

## Files

`Qwen3.8-Flash-Next-16G64G-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`. `SHA256SUMS` in
the repo. If you already have the 16G64G files, you do not need to download these again.

## Running it on a Spark

Abyss supports DGX Spark (aarch64, sm_121) - see
[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) for the build and the settings. Pack the
GGUF once:

```sh
cd Abyss
.venv/bin/python tools/iq_pack.py --fast --gguf /path/to/Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf --out /path/to/pack-fast
```

and start it with `--glm-pack /path/to/pack-fast`. llama.cpp (arch `qwen4exp`) also reads the file.

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+DGX-Spark+on+GB10).
- **Quality in real use**: empty answers, wrong language, loops. A prompt that reproduces it helps most.

## License and credits

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are
[magnitudedev](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)'s, the non-expert tensors unsloth's.
Quantized with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- 形式を混ぜたファイルです。エキスパートは IQ3_XXS 56%, IQ4_NL 42%, IQ3_S 1%（容量の割合）、平均 3.55 ビットで、IQ3_M くらいです。

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**NVIDIA DGX Spark 1 台**（GB10、共有メモリ
128 GB）向けにまとめたものです。**エキスパートは 512 個すべて残しています**。速さは [Abyss](https://github.com/zephel01/Abyss)
で、**本物の DGX Spark で**測りました。

ファイルは [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) と同じものです
（`SHA256SUMS` も同じ）。このリポジトリは Spark 向けの入口で、Spark での速さと動かし方を載せています。作り方と品質の詳しい比較は、
16G64G のカードにあります。

**BETA です**：借りた DGX Spark で測りました（2026-11-05〜07）。速さはエンジンで変わるので、Abyss の新しい版で上がれば、この
カードも更新します。お手元の Spark での結果を、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) で
教えてもらえると助かります。

| | 大きさ | エキスパート | KL（10 本） | 日本語の KL | Spark の生成（4K のあと） | 32K のあと |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **この版** | **86.9 GB** | **512 個すべて** | **0.083** | **0.176** | **55.8 tok/s** | **53.4 tok/s** |
| unsloth UD-IQ3_XXS（比べる相手） | 82.0 GB | 512 個すべて | 0.154 | 0.317 | - | - |

- DGX Spark、Abyss v1.3.119、MTP の下書きあり、プロンプトのあと 512 トークン（v1.3.118 では 54.7 / 47.5 tok/s）。
- Spark でのプロンプトの処理：4K〜64K で約 1,040〜1,100 tok/s（Abyss v1.3.113）。
- ファイル全体が Spark のメモリに載ります。層ごとの埋め込み表（28.8 GB）は、1 トークンにつき数行だけ SSD から読みます。
- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。16G64G のファイルをお持ちなら、ダウンロードし直す必要は
  ありません。
- 動かし方は上の英語の節のとおりです（[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) でビルドし、
  `iq_pack.py --fast` で pack を作って `--glm-pack` で指定）。
- ライセンスは元のモデルと同じです（qwen-community-1.0）。
