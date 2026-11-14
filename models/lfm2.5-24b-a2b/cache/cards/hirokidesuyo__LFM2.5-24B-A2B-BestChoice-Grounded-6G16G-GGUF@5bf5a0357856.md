---
license: other
language:
  - ja
  - en
base_model: LiquidAI/LFM2.5-24B-A2B
tags:
  - gguf
  - llama-cpp
---

# LFM2.5-24B-A2B-BestChoice-Grounded-6G16G-GGUF

The English description can be found below the Japanese description.

VRAM 6GB / RAM 16GB の PC で動作する LFM2.5-24B-A2B の GGUF です。先行モデル [BestChoice](https://huggingface.co/hirokidesuyo/LFM2.5-24B-A2B-BestChoice-6G16G-GGUF) の重み・量子化レイアウトを土台に、(1) 検索ツールの使用可否と要否に関する行動学習（SFT）、(2) 検索と即答の判断境界の調整（DPO）の2段階の追加学習を行いました。

## ファイル情報

| 項目 | 内容 |
|---|---|
| ファイル名 | `LFM2.5-24B-A2B-BestChoice-Grounded-6G16G.gguf` |
| サイズ | 9,443,769,984 バイト（約 9.4GB） |
| SHA-256 | `0df87f54be40a623d8f194f8439e6ffdbaafed560bd3a38b45ff055a61fc7674` |
| ベースモデル | LiquidAI/LFM2.5-24B-A2B |
| 量子化 | BestChoice と同一の独自 UD レイアウト（IQ3_XXS ベース。詳細は BestChoice のカード参照） |
| ライセンス | other（ベースモデルおよび教師データの条件に従います） |

## 作り方 1: 行動学習（SFT）

### 学習レシピ

- 初期モデル: ベースモデル LiquidAI/LFM2.5-24B-A2B。BestChoice の蒸留データに本段階の行動データを合流させ、ベースから再学習しました（合流再学習）。
- LoRA: r=16 / alpha=32。ただし expert FFN（w1/w2/w3）は r=4 / alpha=8。対象 46 モジュール。
- 最適化: lr 1e-4、cosine スケジュール、1 epoch、4096 トークンパッキング。
- 実績: 最終データ 20,137 行、342 ステップ、train_loss 0.6234。

### 追加データ（行動学習分）

| 区分 | 件数 | 内容 |
|---|---|---|
| 日本語教師データ | 2,350 | 教師 GLM-4.7-Flash が生成。内訳: 検索が必要な質問 700、目的と手段がずれる質問 550、即答すべき質問 500、未知の対象への質問 350、数量の質問 250 |
| 公開ツール利用データ | 3,000 | ToolACE 2,500 件、Glaive 関数呼び出しデータ 500 件を変換 |

学習対象の行動は、未知の固有名詞・最新情報に対する検索ツールの使用、目的と手段がずれる質問での正しい結論、計算・常識で解ける問題の自己解決、および不明な事項に対する不明の表明です。

## 作り方 2: 判断境界の調整（DPO）

### 学習レシピ

- 初期ポリシーおよび参照モデル: 作り方 1 の SFT マージ済みモデル。
- 手法: TRL DPOTrainer、Unsloth 4bit ロード。新規 LoRA r=16 / alpha=32（SFT と同一の対象モジュール群）。
- ハイパーパラメータ: beta 0.3、lr 1e-5、cosine スケジュール、warmup 5%、1 epoch、max_len 1536、有効バッチ 16。
- 実績: 87 ステップ、学習時間 1,984 秒（Google Colab L4）、train_loss 0.3746、最終 rewards/margins 1.64。

### ペアデータ（1,377 件）

| 区分 | 件数 | chosen（好ましい応答） | rejected（避けるべき応答） |
|---|---|---|---|
| fire（検索側） | 332 | 検索ツールの呼び出し | SFT モデル自身による捏造応答 |
| direct（即答側） | 905 | 検証済みの正答 | DPO 前モデル自身による検索ツールの呼び出し |
| admit（認知側） | 140 | 不明であることの表明 | SFT モデル自身による捏造応答 |

rejected 側は原則としてモデル自身の出力を収集して用いました。評価ベンチおよび固定54問ベンチの問題文・固有名詞・数値・シナリオは学習データに含めていません。

## 評価

### 測定条件

- マシン: ノート PC（Core i7-12650H、RTX 3060 Laptop 6GB、RAM 16GB）。
- 推論: llama.cpp b12208 の llama-server。推奨設定（下記）適用。
- 振る舞いベンチ: 自作 42 問（検索発火 26 試行、検索不要 16 試行、目的トラップ 50 試行、数量 18 試行、未知の対象 18 試行／条件）。temperature 0.8、最大生成 1,024 トークン。tools 条件では `web_search` ツールを提示し、ツール呼び出しの有無と応答内容を測定（検索自体は実行しません）。自動スコア後に全境界行を目視で裁定しました。
- 54問ベンチ: 知識・推論・コード・日本語の自作固定 54 問。temperature 0、ルールベースのチェッカーで採点。

### 振る舞いベンチ結果

| 指標 | 本モデル | Grounded（SFT のみ） | BestChoice |
|---|---|---|---|
| 検索発火率（tools） | **96.2%** | 84.6% | 77% |
| 未知の対象への非捏造率（tools） | **94.4%** | 89% | 78% |
| 未知の対象への非捏造率（notools） | 50.0% | 56% | 44% |
| 検索系質問への非捏造率（notools） | **57.7%** | 54% | 27% |
| 既知事実への即答正答率（notools） | **93.8%** | 88% | 94% |
| 直接回答率（tools、参考） | 62.5% | **75%** | 69% |
| 目的トラップ正答率（notools） | **86.0%** | 80% | 54% |
| 目的トラップ正答率（tools、参考） | 48% | 58% | **63%** |
| 数量正答率（notools） | 88.9% | 94% | **100%** |
| 数量正答率（tools、参考） | 78% | 67% | **100%** |

### 54問ベンチ結果（完結×正答）

| モデル | 合計 | コード | 英語推論 | 日本語指示 | 日本語推論 | 知識 |
|---|---|---|---|---|---|---|
| 本モデル | 48/54 | 9/11 | 11/11 | 8/10 | 11/11 | 9/11 |
| Grounded（SFT のみ） | 49/54 | 11/11 | — | — | — | — |
| BestChoice | **53/54** | 11/11 | 11/11 | 9/10 | 11/11 | 11/11 |

### 読み取れること

- tools 条件では、検索すべき場面での発火率と非捏造率が比較モデルの中で最も高く、既知事実への過剰発火も SFT 時点より抑制されています。
- notools 条件では、目的トラップと既知事実の即答が改善する一方、未知の対象への非捏造率は SFT 時点から改善しませんでした。ツールが使用できない環境では、未知の事項を推測で補う傾向が残ります。
- 54問ベンチの正答数は BestChoice を下回ります。本モデルの変更点は知識量ではなく、検索ツールの使用判断を中心とする応答行動です。

## 推奨設定

llama-server の例:

```
llama-server -m LFM2.5-24B-A2B-BestChoice-Grounded-6G16G.gguf \
  -ngl 99 -ncmoe 99 --moe-cache-mib 3072 --moe-cache-cpu-share 0.5 -lm dio \
  -c 32768 -ctk q8_0 -ctv q8_0 --flash-attn on -t 8
```

- 本モデルの行動学習は、関数名 `web_search`（引数 `query`）の検索ツールを提示する環境を前提としています。
- UI から使う場合は最大生成トークンを大きめに設定してください（思考が長めのモデルです）。
- VRAM 6GB / RAM 16GB が前提の配分です。RAM が少ない環境では expert の実駐が崩れて大きく速度が落ちます。

## 制限と注意

- 評価は小規模な自作ベンチによる相対比較です。MMLU 等の標準ベンチではありません。
- 事実・知識系の質問では誤ることがあります。重要な用途では裏取りをしてください。
- 検索ツールが実行されない環境では、ツール呼び出しで応答が終了し、無回答となる場合があります。
- 本モデルは通常版ベースの学習で、abliteration（拒否解除）は施していません。
- 学習データの日本語比率は約2割強で、英語が主軸のモデルです。

## クレジット・出典

- ベースモデル: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- 土台モデル: [hirokidesuyo/LFM2.5-24B-A2B-BestChoice-6G16G-GGUF](https://huggingface.co/hirokidesuyo/LFM2.5-24B-A2B-BestChoice-6G16G-GGUF)
- 教師: GLM-4.7-Flash（Zhipu AI）。公開データとして ToolACE、Glaive 関数呼び出しデータを使用
- ツール: llama.cpp（b12208）、Unsloth、TRL（DPOTrainer）

---

# English description

## LFM2.5-24B-A2B-BestChoice-Grounded-6G16G-GGUF

A GGUF build of LFM2.5-24B-A2B for PCs with 6GB of VRAM and 16GB of RAM. Starting from the weights and quantization layout of the earlier [BestChoice](https://huggingface.co/hirokidesuyo/LFM2.5-24B-A2B-BestChoice-6G16G-GGUF) build, it adds two stages of behavior training: (1) SFT for when and how to use a search tool, and (2) DPO to adjust the boundary between searching and answering directly.

## File info

| Item | Value |
|---|---|
| File name | `LFM2.5-24B-A2B-BestChoice-Grounded-6G16G.gguf` |
| Size | 9,443,769,984 bytes (~9.4GB) |
| SHA-256 | `0df87f54be40a623d8f194f8439e6ffdbaafed560bd3a38b45ff055a61fc7674` |
| Base model | LiquidAI/LFM2.5-24B-A2B |
| Quantization | Custom UD layout (IQ3_XXS-based), identical to BestChoice |
| License | other (subject to the base model and teacher-data terms) |

## Training

### Stage 1: Behavior SFT

- Starting point: the base model, trained on the union of the BestChoice distillation data and this stage's behavior data.
- LoRA r=16 / alpha=32 (expert FFN matrices at r=4 / alpha=8), 46 target modules; lr 1e-4, cosine, 1 epoch, 4096-token packing.
- Final data 20,137 rows, 342 steps, train_loss 0.6234.
- Added behavior data: 2,350 Japanese examples generated by GLM-4.7-Flash (search-required 700, purpose-trap 550, answer-directly 500, unknown-target 350, quantity 250) plus 3,000 converted public tool-use examples (ToolACE 2,500, Glaive function-calling 500).

### Stage 2: DPO

- Policy and reference: the merged Stage 1 model. TRL DPOTrainer with Unsloth 4-bit loading; fresh LoRA r=16 / alpha=32 on the same module set.
- beta 0.3, lr 1e-5, cosine, warmup 5%, 1 epoch, max_len 1536, effective batch 16.
- 87 steps, 1,984 s on a Colab L4, train_loss 0.3746, final rewards/margins 1.64.
- Pairs (1,377): fire 332 (chosen: search tool call; rejected: the SFT model's own fabrication), direct 905 (chosen: verified direct answer; rejected: the pre-DPO model's own tool call), admit 140 (chosen: admission of not knowing; rejected: the SFT model's own fabrication). No evaluation-bench items, names, numbers, or scenarios were used in training data.

## Evaluation

Self-made benches on the target laptop (llama.cpp b12208, recommended settings below). Behavior bench: 42 items across two conditions (with/without a `web_search` tool offered), temperature 0.8; the tool is never actually executed. Fixed 54-question bench (knowledge, reasoning, code, Japanese): temperature 0, rule-based checkers.

| Metric | This model | Grounded (SFT only) | BestChoice |
|---|---|---|---|
| Search firing rate (tools) | **96.2%** | 84.6% | 77% |
| Non-fabrication on unknown targets (tools) | **94.4%** | 89% | 78% |
| Non-fabrication on unknown targets (no tools) | 50.0% | 56% | 44% |
| Non-fabrication on search-type questions (no tools) | **57.7%** | 54% | 27% |
| Correct direct answers to known facts (no tools) | **93.8%** | 88% | 94% |
| Direct-answer rate (tools, reference) | 62.5% | **75%** | 69% |
| Purpose-trap accuracy (no tools) | **86.0%** | 80% | 54% |
| Quantity accuracy (no tools) | 88.9% | 94% | **100%** |
| 54-question bench (completed × correct) | 48/54 | 49/54 | **53/54** |

Notes:

- Under the tools condition, this model has the highest search firing rate and non-fabrication rate of the three, with less over-firing on known facts than the SFT-only model.
- Without tools, accuracy on purpose traps and known facts improved, but non-fabrication on unknown targets did not improve over the SFT-only model; the model still tends to guess when no tool is available.
- The 54-question score is below BestChoice. The changes in this build concern answering behavior — primarily search-tool use — not knowledge.

## Recommended settings

```
llama-server -m LFM2.5-24B-A2B-BestChoice-Grounded-6G16G.gguf \
  -ngl 99 -ncmoe 99 --moe-cache-mib 3072 --moe-cache-cpu-share 0.5 -lm dio \
  -c 32768 -ctk q8_0 -ctv q8_0 --flash-attn on -t 8
```

The behavior training assumes an environment that offers a `web_search` function tool (argument `query`). Set a large max-token budget; the model thinks at length. The layout assumes 6GB VRAM / 16GB RAM; with less RAM the experts no longer stay resident and speed drops sharply.

## Limitations

- Evaluation uses small self-made benches for relative comparison, not standard benchmarks such as MMLU.
- The model can be wrong on factual questions; verify important outputs.
- In environments where the offered tool is never executed, a tool call ends the response with no answer.
- No abliteration has been applied. Japanese accounts for roughly a fifth of the training data; the model is English-first.
