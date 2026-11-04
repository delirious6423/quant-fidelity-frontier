---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
---

# Qwen3.8-Flash-Next-mlx-4bit-e4o8-ngram4

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の MLX 4bit 版。
[e4o8](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8)（エキスパート 4bit・ほか 8bit）の n-gram 埋め込みだけを 4bit にして、
一律 4bit とほぼ同じ大きさに収めたもの。道具の使い方は一律 4bit より良く、e4o8 よりは少し落ちる。

- サイズ: 106 GiB（5.136 bits per weight、モデルサーバーのメモリは最大 115GB を観測）
- 量子化（affine）:
  - ルーティングされるエキスパート（`switch_mlp.{gate,up,down}_proj`）: **4bit g32**
  - n-gram 埋め込み（`ple.ple_embedding.ngram_embedding.shards.*`、128 モジュール・約 51B パラメータ）: **4bit g32**
  - 注意機構・共有エキスパートなど、ほかの線形層: **8bit g32**（ルーターは 8bit g64）
  - 画像部分: 量子化しない（bf16）

## 作り方

e4o8 の n-gram 埋め込みの重みを、[一律 4bit 版](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit)（revision f128069、bf16 から直接 4bit g32）の
同じ名前の重みにそのまま差し替えた（丸め直しはしていない）。`config.json` にはそのモジュールごとに `{"group_size": 32, "bits": 4}` を書いてある。
n-gram 埋め込みは e4o8 の 130 GiB のうち 54 GiB を占める。

## 評価

思考オフ、温度 0.6・top_p 0.95・top_k 20、各ケース 5 回。mlx-vlm 0.7.11 のモデルサーバーを直接使い、MTP なし。
probe は日本語・英語の短い問答 34 問、道具は呼び出しの場面 20 種を日本語と英語でそれぞれ。

| | [8bit](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-8bit) | [一律 4bit](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit) | [e4o8-ngram4](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8-ngram4) | [e4o8](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8) |
|---|---|---|---|---|
| probe 正答 | 100% | 100% | 100% | 100% |
| 日本語の道具 | 97% | 94% | 99% | 99% |
| 英語の道具 | 88% | 80% | 92% | 98% |
| 正しい呼び出し形式（英語） | 100% | 95% | 98.5% | 100% |
| 重みの大きさ | 186 GiB | 104 GiB | 106 GiB | 130 GiB |
| 生成速度（MTP なし） | — | 46.6 tok/s | 37.0 tok/s | 38.5 tok/s |
| 生成速度（MTP あり） | — | 48.7 tok/s | 37.0 tok/s | 39.0 tok/s |

生成速度は M3 Ultra、温度 0・400 トークン × 3 問の平均。

- 一律 4bit の主な失敗: 要らない場面での道具呼び出し、3 件並べて呼ぶべきところを 1 件だけ、設定ファイルを読む道具の使い間違い（いずれも 5 回中 5 回）。
- e4o8-ngram4 は英語で、道具の呼び出しを本文にそのまま書く失敗が 100 回中 2 回あった（e4o8・8bit では 0 回）。
- 8bit が e4o8 より悪い理由は分かっていない（エキスパート以外の重みは両者で同じ）。
- 長い会話・画像入力は試していない。

## 使い方

```
pip install -U mlx-vlm
python -m mlx_vlm generate --model ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8-ngram4 \
    --prompt "この画像を説明してください" --image path/to/image.jpg --max-tokens 512
```

## MTP ドラフター（投機デコード）

ドラフターは別リポジトリ
[Qwen3.8-Flash-Next-MTP-bf16](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16)（mlx-vlm 0.7.0 以降）。
この量子化ではほとんど速くならなかった（上の表）。

```
python -m mlx_vlm generate --model ToPo-ToPo/Qwen3.8-Flash-Next-mlx-4bit-e4o8-ngram4 \
    --draft-model ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16 \
    --draft-kind mtp --prompt "..." --max-tokens 512
```
