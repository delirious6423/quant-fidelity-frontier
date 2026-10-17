---
language:
- en
- ja
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
- mlx
- qwen4_exp
- speculative-decoding
- mtp
---

# Qwen3.8-Flash-Next MLX 4bit (v-l) + MTP サイドカー

Apple Silicon 向けの 4bit 変換に、**MTP (Multi-Token Prediction) の重みを同梱**したもの。
[mlxturbo](https://github.com/) の `flash_spec` 経路で投機デコードが効きます。

素の mlx-lm でもテキスト生成として読めますが、その場合 MTP は使われません。

## 中身

| | |
|---|---|
| アーキテクチャ | `qwen4_exp` (48 層、512 エキスパート / top_k 10、GDN ハイブリッド) |
| 量子化 | 4bit affine、group_size 64 |
| 文脈長 | 262,144 |
| MTP | `mtp.safetensors` (bf16, 5.2GB、深さ 1) |
| 合計 | 約 77GB |

n-gram (PLE) テーブルは本体に含めず、別サイドカーとして扱います。

## 使い方

```bash
pip install mlxturbo
mlxturbo-serve --model /path/to/this/repo --require-runner flash_spec --port 8765
```

`mtp.safetensors` が同じディレクトリにあるので、追加のフラグなしで MTP が自動発見されます。
`--require-runner flash_spec` を付けると、何らかの理由で投機が効かない構成のときに
黙って遅く動かず、起動を拒否します。

## 速度

M3 Max 128GB、貪欲、160 トークン × 10 課題での実測。

| 課題 | 受理率 | 対 非投機 |
|---|---|---|
| 散文 (英語 0.720 ± 0.046 / 日本語 0.682 ± 0.047) | 約 0.70 | 約 1.39x |
| コード | 0.564 ± 0.068 | 約 1.25x |

非投機は 31.14 ms/token。**効くかどうかは言語ではなく課題の種類で決まり、コードが最も低い**
(信頼区間が重ならない)。どの課題でも投機は効きます。

短い試行 (48 トークン) では言語差があるように見えますが、標準誤差が ±0.09 あり有意ではありません。

## メモリ

77GB を常駐させるので、**128GB 以上のマシンを想定**しています。96GB 機では厳しく、
64GB 機では載りません。

## ライセンス

元モデル [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の
Qwen Community License 1.0 を継承します。全文は同梱の `LICENSE` を参照してください。

月間アクティブユーザー 1 億超または月商 2,000 万ドル超の商用製品で使う場合はモデル名の
表示義務が、Model as a Service / AI Work Assistant 事業では別途 Qwen との契約が必要です。
