---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- ja
- en
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- quantized
- japanese
- qwen3_5
- low-bit
model_name: Qwen3.8-27B EXL3 1.14514bpw
---

# Qwen3.8-27B EXL3 1.14514bpw

> **迫真量子化部・低ビットの裏技**
>
> お待たせ！27Bしかなかったけどいいかな？

**27Bモデルを、重み約7.83 GBのEXL3に。日本語を重視して校正した、Hent-AIの実験的な量子化モデルです。**

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)を強く圧縮し、EXL3と日本語中心の校正で品質の劣化を抑えることを狙いました。対応GPUを使ったローカル推論や、メモリの限られたエッジ環境を目指しています。

| 先に知りたいこと | 答えはこ↑こ↓ |
|---|---|
| 形式・重み容量 | **EXL3 / 約7.83 GB（7.29 GiB）** |
| 動作確認 | RTX 3090、DGX Spark（GB10）、RTX 5090でテキスト推論 |
| 日本語向けの調整 | Swallow・Dolly日本語を使った量子化校正。**追加学習なし** |
| ベンチ | **全40分割・12,121問完了。6項目平均 39.47%** |

> 実測は1.14515やけど、配布名は1.14514にしとこ……0.00001多くても、まあバレへんか。

## 1. 使い方 — あくしろよ

> まずうちさぁ、EXL3あるんだけど……試してかない？

1. **EXL3対応の実行環境を用意する。**  
   Qwen3.8のアーキテクチャに対応する [ExLlamaV3](https://github.com/turboderp-org/exllamav3#installation) 等を使用してください。テキスト生成はExLlamaV3 **1.5.3**で確認しています。

2. **モデルをダウンロードする。**

   ```sh
   hf download Hent-AI/Qwen3.8-27B-EXL3-1.14514bpw --local-dir ./Qwen3.8-27B-EXL3-1.14514bpw
   ```

3. **取得したフォルダーをモデルとして読み込む。**  
   同梱のチャットテンプレートを使います。掲載ベンチと同じ思考OFFにする場合は `enable_thinking=false` を指定してください。

**7.83 GBは重みファイルの容量です。必要VRAMはこれより増えます。**  
KVキャッシュや推論用バッファーにもメモリを使います。スマートフォン・CPUのみ・8GB VRAM環境での動作は未検証です。

> VRAM不足で起動できない？「お前やりませんねぇスギィ！！！！」

じゃけん、対応環境でローカル推論しましょうね〜。

## 2. ベンチ結果 — そんなに見たけりゃみせてやるよ

**全40/40分割・12,121問を評価済み。** 問題・正解・採点情報を照合し、重複と欠落がないことを確認しました。やりますねぇ！

評価・検証完了：2026-10-04T**19:19**:07+09:00（日本時間）。

| ベンチマーク | スコア | 問題数 |
|---|---:|---:|
| MMLU-Redux | 31.30% | 5,700 |
| MuSR | 33.99% | 756 |
| GSM8K | 16.60% | 1,319 |
| HumanEval+ | 31.71% | 164 |
| IFEval | 60.78% | 541 |
| BFCL v3（13 subset平均） | 62.43% | 3,641 |

**6項目の単純平均：39.47%。** 各ベンチのスコアを同じ重みで平均した値です。全問題をまとめた正答率ではありません。

**これは日本語能力の評価ではありません。** 他の公開スコアと比較する際は、実行環境・生成条件・採点方法をそろえる必要があります。「そうだよ（便乗）」で同じ条件扱いにはしない、はっきりわかんだね。

<details>
<summary><strong>評価条件・採点方法を見る</strong></summary>

| 条件 | 設定 |
|---|---|
| 実行環境 | ExLlamaV3 1.5.3 / EvalScope 1.12.0 |
| GPU | RTX 3090 / DGX Spark（GB10）/ RTX 5090 |
| 生成 | **思考OFF、temperature 0（greedy）**、top_p 1、top_k 20、seed 42 |
| 同時リクエスト | 3090：4問 / DGX：2問 / 5090：4問 |
| MTP先読み | 3090・DGX：最大4トークン / 5090：評価途中で最大4→8に変更 |
| 最大生成長 | MMLU-Redux・MuSR・GSM8K：2,048 / IFEval・BFCL：4,096 / HumanEval+：8,192トークン |

- **HumanEval+：** pass@1。
- **IFEval：** strict promptとstrict instructionの平均。
- **BFCL v3：** single-turnの13 subset（分類）を選び、各subsetのaccuracyを統合してから単純平均。
- **MTP変更：** 別の同一8問でMTP4/6/8の回答一致を確認。全ベンチでの出力一致は未検証です。
- **数値の丸め：** EvalScopeレポートの丸め済みスコアを集約した概数です。

</details>

<details>
<summary><strong>検証内容・評価上の注意を見る</strong></summary>

全40分割の保存済み問題本文・正解・採点メタデータを元データと照合しました。ランダムなメッセージIDを除いたcanonical hashと問題indexで、全問が重複・欠落なく含まれることを検証しています。

本評価の実行条件と、比較するときの注意点です。

- 実行環境は **EvalScope 1.12.0 / ExLlamaV3 1.5.3**です。
- データrevisionや独自修正が異なる評価との、完全な同条件比較は保証していません。
- Geminiによる回答抽出フォールバックは使用していません。

集計値・分割別指標・問題カバレッジ・出典ハッシュは [benchmark-results.json](benchmark-results.json) に収録しています。

</details>

<!-- japanese-benchmark:start -->
### 日本語ベンチ — 同じ8,245問で確認

JCommonsenseQA・JNLI・JSQuAD・MGSM日本語の**全8,245問**を使った独自の0-shot生成評価です。JGLUEや他の公式ランキングの再現ではありません。

| 項目・指標 | 問題数 | Thinking OFF | Thinking ON |
|---|---:|---:|---:|
| JCommonsenseQA 正答率 | 1,119 | 73.01% | 77.57% |
| JNLI 正答率 | 2,434 | 59.78% | 69.60% |
| JNLI balanced accuracy | 2,434 | 40.50% | 60.37% |
| JSQuAD 正規化EM | 4,442 | 85.37% | 86.88% |
| JSQuAD 文字多重集合F1 | 4,442 | 93.04% | 94.32% |
| MGSM日本語 正答率 | 250 | 18.80% | 18.40% |

**共通条件：** 同一の問題・正解・プロンプト・採点。temperature 0、top_p 1、top_k 20、seed 42、MTP4、各GPU4問並列。RTX 3090とDGX Sparkを使用し、DGXの利用期限後は3090で継続します。

**生成上限：** OFFは常識32／文脈推論32／読解128／算数1,024トークン。ONは前三項目2,048・算数4,096トークン（思考を含む総数）、思考予算は前三項目1,024・算数2,048トークンです。予算到達時は思考を閉じて最終回答へ進めます。生成上限も異なるため、思考の有無だけを切り分けた比較ではありません。

採点対象は最終回答です。分類は一意な選択肢・ラベル、算数は最終数値を採点します。JSQuADはNFKC・小文字化・空白とUnicode句読点除去後のEMと文字多重集合F1（複数正解の最大値）を使います。llm-jp-evalの同名指標とは定義が異なります。

| 条件 | 無効回答 | 生成上限到達 | 思考のみで終了 | 思考予算による終了誘導 |
|---|---:|---:|---:|---:|
| OFF | 8 | 32 | 0 | 0 |
| ON | 41 | 66 | 19 | 3403 |

全問の問題hash・条件hash・採点・重複と欠落を検証済み。試走は本番に含めていません。ネットワーク障害を不正解として集計せず、モデルが最終回答を出さなかった場合は無効回答として記録します。

**この評価だけでは、日本語校正による性能向上は実証していません。** 元モデル・別校正との比較は未実施です。データrevision・条件・項目別集計は [日本語ベンチ集計](japanese-benchmark-results.json) に収録しています。

- OFF評価完了：2026-10-05T02:45:14+09:00（日本時間）
- ON評価完了：2026-10-07T08:01:57+09:00（日本時間）
<!-- japanese-benchmark:end -->

## 3. 日本語向けの校正 — サーッ（迫真）

**Swallowの一部データを使い、日本語応答の品質維持を狙って校正しました。**

| 校正データ | 行数 |
|---|---:|
| Swallow日本語 | 100 |
| Dolly日本語 | 100 |
| Qwen self-calibration | 50 |
| **合計** | **250** |

1行あたり最大2,048トークン、有効トークン数は **434,527**。日本語例に属するトークンは **約76.4%**です。例の中のコードなども、この割合に含みます。

**行ったのは量子化校正です。追加学習（FT）はしていません。**  
元モデルや別の校正との比較による、日本語性能の向上は未確認です。「日本語が絶対に強くなった」とまでは言えません。ここ大事、覚えといて。

<details>
<summary><strong>データの選び方を見る</strong></summary>

Swallowの日本語版code / math / STEMから、各データの先頭にある限定サンプルを使用しました。日本語応答を選び、英語の思考過程を除いて、Qwenのチャットテンプレートでトークン化しています。

出典・revision・構成は [calibration-info.json](calibration-info.json) に記録しています。データセット本体や校正トークン列は配布物に含めていません。

</details>

## 4. 仕様と注意点 — これもうわかんねぇな、を解消

### 「1.14514」と「1.14515」の違い

> ちょっと待って！1.14514になってないやん！  
> 1.14514が欲しかったから量子化したの！

- **配布名：** `1.14514bpw`
- **対象重みの正確な平均：** `10627/9280 = 1.145150862… bpw`
- **小数点以下5桁の表示：** `1.14515 bpw`

この平均は **decoder Linearの対象重み**についての値です。埋め込み・出力ヘッド・MTP・vision・スケール等は含みません。

### 使う前に知っておくこと

**強い圧縮による品質の制約があります。** 日本語の少数例では、正しい回答のほか、指示違反・反復・不自然な補足も確認しています。

> 指示と違う回答が返ってきたら──「てめぇ様は、何をしてくれちゃってんだスギ？」

大事な用途では、回答を確認してください。

EXL3や日本語校正によって、劣化がなくなることや元モデルと同じ性能を保つことは保証していません。

**vision / MTPの重みは同梱済みです。** 掲載ベンチはテキストのみで、画像・動画能力は未評価です。

<details>
<summary><strong>量子化の詳細仕様を見る</strong></summary>

| 項目 | 内容 |
|---|---|
| 元モデル | Qwen/Qwen3.8-27B |
| 元モデルrevision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| 形式・変換実装 | EXL3 / ExLlamaV3 1.5.3 |
| decoder Linearの対象重み平均 | 1.145150862… bpw |
| 出力ヘッド / MTP / vision Linear | 6 bit / 4 bit / 6 bit |
| codebook | mul1 |
| 重みファイル | 2ファイル、計7,828,334,162 bytes（約7.83 GB / 7.29 GiB） |

ビット配分はEXL3のgroup境界を維持して調整しました。日本語に敏感な層を比較実験で特定した配分ではありません。

設定と配分は [quantization_config.json](quantization_config.json) / [quantization-recipe.json](quantization-recipe.json) を参照してください。

</details>

<details>
<summary><strong>Hubの「4B params」やconfig警告について</strong></summary>

**圧縮前の基底モデルは27Bです。** Hubの自動テンソル集計では、量子化後のデータから `4B params` と表示されます。

また、Hubの汎用config検査は `bits` に2以上の整数を要求するため、この小数・極低ビット設定に `Configuration Parsing Warning` が表示されます。

量子化仕様や容量は、このカードと同梱のEXL3設定を参照してください。表示だけ見て「ファッ！？」となったら、こ↑こ↓を確認。

</details>

## 5. ライセンスと出典 — ありがとナス！

**モデルは元のQwen3.8-27BのApache License 2.0を継承しています。** 原文は [LICENSE](https://huggingface.co/Hent-AI/Qwen3.8-27B-EXL3-1.14514bpw/blob/main/LICENSE)、量子化の変更内容とデータ出典は [NOTICE](https://huggingface.co/Hent-AI/Qwen3.8-27B-EXL3-1.14514bpw/blob/main/NOTICE) に収録しています。

| 対象 | 出典・著作者 | ライセンス |
|---|---|---|
| 元モデル | [Qwen / Alibaba Cloud](https://huggingface.co/Qwen/Qwen3.8-27B) | [Apache 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |
| Swallow校正データ | [Swallow LLM / tokyotech-llm](https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Dolly日本語校正データ | [kunishou / databricks-dolly-15k-ja](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)、元データDatabricks | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| self-calibration trace | [turboderp / Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) | 配布元の表記を参照 |

校正データと実行ソフトウェアのライセンスは、それぞれの原文に従ってください。モデルのライセンスが、データやソフトウェアのライセンスを一つに置き換えるわけではありません。

本モデルは **Hent-AIによる量子化版**です。QwenまたはSwallowの公式モデルではありません。

> 元モデル・データ・実装を公開してくださった皆さま、ありがとナス！
