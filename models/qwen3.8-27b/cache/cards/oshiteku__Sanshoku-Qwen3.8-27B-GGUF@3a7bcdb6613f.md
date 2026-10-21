---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
language:
- en
- ja
tags:
- ternary
- 2-bit
- gguf
- llama-cpp
base_model:
- Qwen/Qwen3.8-27B
---

# 🍡 Sanshoku-Qwen3.8-27B-GGUF

[English](#english) | [日本語](#日本語)

<a id="english"></a>

A fully ternary version of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Every weight matrix,
including the embedding, the output head and the MTP block, takes values in {−1, 0, +1}. The model was
trained with QAT against the original model.

This is an experimental model and is not yet suited for practical use.

## Files

All three GGUFs contain the same weights in different packings.

| File | Size | Runtime |
|---|---:|---|
| `Sanshoku-Qwen3.8-27B-Q2_0.gguf` | 7.70 GB | upstream llama.cpp |
| `Sanshoku-Qwen3.8-27B-PQ2_0.gguf` | 7.28 GB | [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp) |
| `Sanshoku-Qwen3.8-27B-PTQ1_0.gguf` | 6.00 GB | PrismML fork of llama.cpp |
| `mmproj-F16.gguf` | 0.93 GB | vision encoder (original, FP16) |

## Quick start

```bash
llama-server -hf oshiteku/Sanshoku-Qwen3.8-27B-GGUF -hff Sanshoku-Qwen3.8-27B-Q2_0.gguf \
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
```

Specify the file with `-hff`, since the tag `:Q2_0` can also match the PQ2_0 file. The MTP head is
included in each GGUF. Tested with llama.cpp `1a67982` and the PrismML fork `bdc23b5`.

## Speed

Measured on an RTX 4000 Ada (20 GB). Prefill and decode are `llama-bench` pp512 and tg128. Decode + MTP
is the average over four chat requests (256 output tokens, 2 draft tokens).

| Format | Runtime | Prefill (tok/s) | Decode (tok/s) | Decode + MTP (tok/s) |
|---|---|---:|---:|---:|
| Q2_0 | [upstream](https://github.com/ggml-org/llama.cpp/tree/1a679828f3312ebc53c9285805cc8bd7c7d90366) | 1012 | 36.5 | 59.2 |
| PQ2_0 | [fork](https://github.com/PrismML-Eng/llama.cpp/tree/bdc23b56b4458b9f1655aec5287f3ab56ee8daaa) | 1040 | 39.1 | 62.4 |
| PTQ1_0 | [fork](https://github.com/PrismML-Eng/llama.cpp/tree/bdc23b56b4458b9f1655aec5287f3ab56ee8daaa) | 1003 | 41.9 | 43.7 |
| [Qwen3.8-27B UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (16.5 GB) | [upstream](https://github.com/ggml-org/llama.cpp/tree/1a679828f3312ebc53c9285805cc8bd7c7d90366) | 923 | 19.2 | 35.5 |

## Evaluation

Small-scale local evaluations. All models use the same prompts and settings.

| | Sanshoku | [Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | [Qwen3.8 UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
|---|---:|---:|---:|
| Size | 7.3 GB | 7.2 GB | 16.5 GB |
| Top-1 agreement with BF16 ¹ | 84.4% | 89.1% | 97.0% |
| KL divergence from BF16 ¹ | 0.297 | 0.153 | 0.014 |
| Math (subsets) ² | 74.7% | 89.0% | 91.0% |
| SWE-bench Verified Mini ³ | 18% | 60% | – |

1. 60 in-house prompts, 32 tokens each, averaged over chat and coding-agent prompts in both thinking and
   non-thinking modes. UD-IQ2_XXS (7.3 GB): 84.0% / 0.311.
2. Average over fixed subsets: 100 of 1,319 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test problems, 50 of 500 [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
   problems and 100 of 250 [MGSM](https://huggingface.co/datasets/juletxara/mgsm) Japanese problems. Thinking mode, output capped at 4,096 tokens.
3. [SWE-bench Verified Mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini), a 50-task subset of SWE-bench Verified, run with mini-SWE-agent.

The model is weakest at long reasoning (math), Japanese writing and long agentic tasks.

## Training

- Initialized by absmean ternarization in groups of 128 weights, then trained with a straight-through
  estimator against the top-32 distribution of BF16 Qwen3.8-27B.
- About 1B tokens, roughly 78% general text and code; the rest is agent, tool-use, math and reasoning data.
- About 31 hours on a single NVIDIA B300.

## License

Apache-2.0

## Acknowledgements

Thanks to the Qwen team for the base model, [PrismML](https://github.com/PrismML-Eng/llama.cpp) for the
PQ2_0 and PTQ1_0 kernels, and the [llama.cpp](https://github.com/ggml-org/llama.cpp) project.

---

<a id="日本語"></a>

# 🍡 Sanshoku-Qwen3.8-27B-GGUF (日本語)

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) の重みをすべて3値にしたモデルです。embedding、出力 head、
MTP ブロックを含むすべての重み行列が {−1, 0, +1} の値をとります。元モデルを教師として QAT で学習しました。

実験的なモデルであり、まだ実用的な品質ではありません。

## ファイル

3つの GGUF は同じ重みを異なる形式で格納しています。

| ファイル | サイズ | 実行環境 |
|---|---:|---|
| `Sanshoku-Qwen3.8-27B-Q2_0.gguf` | 7.70 GB | 通常の llama.cpp |
| `Sanshoku-Qwen3.8-27B-PQ2_0.gguf` | 7.28 GB | [PrismML 版 llama.cpp (fork)](https://github.com/PrismML-Eng/llama.cpp) |
| `Sanshoku-Qwen3.8-27B-PTQ1_0.gguf` | 6.00 GB | PrismML 版 llama.cpp (fork) |
| `mmproj-F16.gguf` | 0.93 GB | 画像エンコーダ (オリジナル、FP16) |

## 使い方

```bash
llama-server -hf oshiteku/Sanshoku-Qwen3.8-27B-GGUF -hff Sanshoku-Qwen3.8-27B-Q2_0.gguf \
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
```

タグ `:Q2_0` では PQ2_0 のファイルが選ばれることがあるため、ファイルは `-hff` で指定してください。MTP head は
各 GGUF に含まれています。llama.cpp `1a67982` と PrismML 版 `bdc23b5` で動作を確認しています。

## 速度

RTX 4000 Ada (20 GB) で測定。prefill と decode は `llama-bench` の pp512 と tg128、decode + MTP は4つの
チャットリクエストの平均です (出力 256 トークン、ドラフト 2 トークン)。

| 形式 | 実行環境 | prefill (tok/s) | decode (tok/s) | decode + MTP (tok/s) |
|---|---|---:|---:|---:|
| Q2_0 | [通常版](https://github.com/ggml-org/llama.cpp/tree/1a679828f3312ebc53c9285805cc8bd7c7d90366) | 1012 | 36.5 | 59.2 |
| PQ2_0 | [fork](https://github.com/PrismML-Eng/llama.cpp/tree/bdc23b56b4458b9f1655aec5287f3ab56ee8daaa) | 1040 | 39.1 | 62.4 |
| PTQ1_0 | [fork](https://github.com/PrismML-Eng/llama.cpp/tree/bdc23b56b4458b9f1655aec5287f3ab56ee8daaa) | 1003 | 41.9 | 43.7 |
| [Qwen3.8-27B UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (16.5 GB) | [通常版](https://github.com/ggml-org/llama.cpp/tree/1a679828f3312ebc53c9285805cc8bd7c7d90366) | 923 | 19.2 | 35.5 |

## 評価

手元で行った小規模な評価です。すべてのモデルで同じプロンプトと設定を使っています。

| | Sanshoku | [Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | [Qwen3.8 UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
|---|---:|---:|---:|
| サイズ | 7.3 GB | 7.2 GB | 16.5 GB |
| BF16 との top-1 一致率 ¹ | 84.4% | 89.1% | 97.0% |
| BF16 との KL divergence ¹ | 0.297 | 0.153 | 0.014 |
| 数学 (サブセット) ² | 74.7% | 89.0% | 91.0% |
| SWE-bench Verified Mini ³ | 18% | 60% | – |

1. 独自の 60 プロンプトで各 32 トークンを評価。チャットとコーディングエージェントのプロンプトについて、
   thinking モードと non-thinking モードの平均です。UD-IQ2_XXS (7.3 GB) は 84.0% / 0.311 でした。
2. 固定したサブセットの平均: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test から 1,319 問中 100 問、[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) から 500 問中 50 問、
   [MGSM](https://huggingface.co/datasets/juletxara/mgsm) の日本語から 250 問中 100 問。thinking モード、出力上限は 4,096 トークンです。
3. SWE-bench Verified から 50 問を抜き出した [SWE-bench Verified Mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini) を mini-SWE-agent で実行。

特に弱いのは、長い推論 (数学)、日本語の文章作成、長いエージェントタスクです。

## 学習

- 重みを 128 個ずつ absmean で3値化して初期化し、BF16 の Qwen3.8-27B の上位 32 トークンの分布を目標に
  straight-through estimator で学習しました。
- 学習データは約 10 億トークンで、約 78% が一般のテキストとコード、残りがエージェント、ツール呼び出し、
  数学、推論のデータです。
- NVIDIA B300 1台で約 31 時間。

## ライセンス

Apache-2.0

## 謝辞

ベースモデルを公開している Qwen チーム、PQ2_0 と PTQ1_0 のカーネルを開発した
[PrismML](https://github.com/PrismML-Eng/llama.cpp)、そして [llama.cpp](https://github.com/ggml-org/llama.cpp) プロジェクトに
感謝します。
