---
license: other
language:
  - ja
  - en
base_model: LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-BestChoice-6G16G-GGUF

VRAM 6GB / RAM 16GB の PC で動く範囲で「汎用最強」を目指した、LFM2.5-24B-A2B の GGUF です。

## 量子化（独自 UD レイアウト）

- IQ3_XXS ベース。自前 imatrix で較正。
- テンソルの種別ごとに「精度を上げると perplexity がどれだけ改善するか」を実測し、効果のあった場所にだけビットを配分したカスタムレイアウトです。
  - expert の gate/up は序盤（blk2-9）と終盤（blk35-39）を iq3_s に引き上げました。最も費用対効果が高かった箇所です。
  - expert の down projection や、LFM2 系の短い畳み込み演算子（shortconv）は、引き上げても改善が誤差範囲だったためベースの精度のままにし、その分の容量を gate/up に回しました。
- ファイルサイズ: 9,443,769,984 B（約 9.4GB）。

## 重み（蒸留）

- ベースモデルへ LoRA（r16、expert は r4）で蒸留学習。データ 14,674 行 / 約 7.9M トークン。
- 配合: 英語推論・コード・短答に、日本語の数学（NuminaMath-CoT-JA）・論理・会話を加えた汎用配合。
- 教師データ: 公開蒸留コーパス r0b0tlab/qwen3.8-max-glm5.2-distillation-51389 のサブセットほか。

## 推奨設定（llama.cpp、RTX 3060 Laptop 6GB + RAM 16GB 実測）

- `--moe-cache-mib 3072 --moe-cache-cpu-share 0.5 -lm dio`、KV cache q8_0、ctx 32768。
- 生成速度: 約 54 t/s（512 トークン実測、llama-completion）。

## 評価（自前の固定ベンチ 54 問、temp 0、上限 1024 トークン）

日本語/英語の推論・指示追従・コード・知識の 54 問で、完結×正答 **53/54**。
比較対象も同一条件で測定しました。

| モデル | 完結×正答 |
|---|---|
| 本モデル（汎用配合の蒸留＋上記カスタムレイアウト） | 53/54 |
| Abliterated UD-IQ3_XXS（alexwkleung による、ベースモデルの既存量子化） | 49/54 |
| 先行版 1: 英語中心の蒸留版（`hirokidesuyo/LFM2.5-24B-A2B-Distill-GGUF` として公開済み） | 51/54 |
| 先行版 2: 先行版 1 に日本語データを少量追加した試作（未公開） | 51/54 |

日本語の質問では日本語で思考し、54 問すべてが上限内に完結しました。
英語 perplexity（llama-perplexity、96 chunks）: 12.52。

※ 小規模な自作ベンチによる評価です。用途に合わせてご自身でもご確認ください。

## ライセンス

ベースモデルおよび教師データの条件に従い、license は other とします。
