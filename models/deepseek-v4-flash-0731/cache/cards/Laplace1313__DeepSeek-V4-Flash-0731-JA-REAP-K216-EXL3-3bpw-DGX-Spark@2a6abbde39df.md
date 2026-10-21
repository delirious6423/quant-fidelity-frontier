---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: other
license_name: mixed-mit-apache-2.0
license_link: https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark/blob/main/LICENSES/README.md
library_name: custom
pipeline_tag: text-generation
inference: false
tags:
- deepseek-v4
- exl3
- exllamav3
- reap
- expert-pruning
- mixture-of-experts
- dgx-spark
- japanese
- tool-calling
- speculative-decoding
---

# DeepSeek V4 Flash JA REAP K216 — DGX Spark向け

DeepSeek V4 Flash-0731を、日本語、ツール呼び出し、コーディング向けに調整したDGX Spark 1台用モデルです。K216 TP1のメインモデル、対応するK64 DSparkドラフトモデル、位置補正アダプター、専用ランタイムをまとめて配布しています。

[GitHubのソース](https://github.com/Laplace1313/deepseek-v4-ja-reap) · [ファイル一覧](https://huggingface.co/Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark/tree/main) · [ライセンス](LICENSES/README.md) · [リリースマニフェスト](RELEASE_MANIFEST.json)

> [!IMPORTANT]
> 付属の専用ランタイムでのみ検証しています。標準のTransformers、vLLM、SGLangでは起動できません。

## 概要

| 項目 | 内容 |
|---|---|
| ベースモデル | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| メインモデル | K216 REAP / EXL3 3.0 bpw / TP1 |
| ドラフトモデル | K64 DSpark / K5 / 確率的サンプリング |
| コンテキスト長 | 256,000トークン |
| 同時処理数 | 1 |
| 対応ハードウェア | NVIDIA DGX Spark / GB10 1台 |
| 配布サイズ | 101.79 GiB |
| API | OpenAI互換 |

## セットアップ

### 1. ダウンロードする

```bash
hf download \
  Laplace1313/DeepSeek-V4-Flash-0731-JA-REAP-K216-EXL3-3bpw-DGX-Spark \
  --local-dir ./DeepSeek-V4-Flash-JA-REAP
```

102 GiB以上の空き容量を用意してください。ダウンロードには、メインモデル、ドラフトモデル、位置補正アダプター、専用ランタイム、オフライン導入用wheel、checksumが含まれます。

### 2. DGX Sparkで起動する

```bash
cd DeepSeek-V4-Flash-JA-REAP
docker compose -f runtime/compose.example.yml up -d
```

初回起動時は、モデルの重み、AOTアーティファクト、CUDAグラフの読み込みに数分かかります。

### 3. 動作を確認する

```bash
curl -fsS http://127.0.0.1:8009/health
curl -fsS http://127.0.0.1:8009/v1/models
```

### 4. リクエストを送る

```bash
curl http://127.0.0.1:8009/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-ja-uncensored-0731",
    "messages": [
      {"role": "user", "content": "日本語で、分散システムの設計原則を簡潔に説明してください。"}
    ],
    "temperature": 0,
    "max_tokens": 512
  }'
```

APIは標準で`127.0.0.1:8009`を使用します。外部から利用する場合は、認証付きプロキシなどを経由してください。

## 性能

生成速度は、OpenAI互換SSEの最初のトークン受信後から完了までを次の式で計算しています。

```text
(completion_tokens - 1) / (finished_at - first_content_token_at)
```

| 処理 | 平均生成速度 | 内容 |
|---|---:|---|
| 日本語の自由文 | **30.02 tok/s** | 日本語の文章生成 |
| コーディング | **33.51 tok/s** | 長めのPython実装 |
| 構造化JSON | **58.31 tok/s** | 制約付きJSON生成 |
| 構造化JSON（参考計測） | **54.97 tok/s** | 5回すべて有効なJSON、最小52.35 tok/s |

構造化JSONは自由文やコードよりドラフトモデルが予測しやすいため、別の処理として掲載しています。54.97 tok/sは、MiaAI-LabのREADMEを参考に再構成した条件での参考値です。

## 品質評価

固定シードで抽出した0-shot評価の結果です。コード生成の評価は、ネットワークを無効にしたリソース制限付きコンテナで実行しています。

| 評価 | 結果 |
|---|---:|
| 日本語プロンプト | **50/50** |
| ツール呼び出し | **6/6** |
| MMLU full 5-shot（全57分野・14,042問） | **86.03%**（12,081/14,042） |
| MMLU 0-shot（200問） | **85.5%**（171/200） |
| ARC-Challenge 0-shot（200問） | **95.0%**（190/200） |
| HellaSwag 0-shot（200問） | **70.5%**（141/200） |
| GSM8K 0-shot（100問） | **96.0%**（96/100） |
| MMLU-ProX 日本語（200問） | **48.5%**（97/200） |
| HumanEval pass@1（50問） | **96.0%**（48/50） |
| MBPP sanitized pass@1（50問） | **84.0%**（42/50） |

MMLU full 5-shotの全条件・カテゴリ別・57分野別結果は[`validation/mmlu-full-5shot-20260827.md`](validation/mmlu-full-5shot-20260827.md)、機械可読データは[`validation/mmlu-full-5shot-20260827.json`](validation/mmlu-full-5shot-20260827.json)に記録しています。その他の評価条件は[`validation/benchmark-summary.json`](validation/benchmark-summary.json)を参照してください。

## ファイル構成

| 構成要素 | サイズ | 用途 |
|---|---:|---|
| K216 TP1メインモデル | 98.83 GiB | 推論用モデル |
| K64 DSparkドラフトモデル | 2.94 GiB | 5トークンの投機的生成 |
| 位置補正アダプター | 0.02 GiB | メインモデルとドラフトモデルの補正 |
| ランタイムとメタデータ | 小容量 | 起動、ライセンス、検証情報 |
| 合計 | 101.79 GiB | 配布全体 |

## 検証済みの設定

```text
Target: japanese-aware-uncensored-tp1
Target format: K216 REAP, EXL3 3.0 bpw, TP1
Draft: dspark-draft-k64
Speculative depth: K5 probabilistic
Position adapter: rank 16, full vocabulary, scale 1.0, activation none
Draft proposal head: row-wise FP8 enabled
CUDA graph capture: exact size 6
Context: 256,000
Max sequences: 1
GPU memory utilization: 0.945 for MMLU full 5-shot; 0.946 for earlier benchmarks; 0.949 for release cold boot
Runtime image: ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
vLLM: 0.15.1+nv26.2
xgrammar: 0.2.4
transformers: 5.13.1
```

## 互換性

次の構成で検証しています。

- NVIDIA DGX Spark / GB10 1台
- 256,000トークンのコンテキスト
- 同時処理数1
- OpenAI互換Chat Completions API
- 日本語応答
- ツール呼び出し
- K64 DSparkによる投機的デコード

標準のvLLM、Transformers `pipeline()`、SGLang、llama.cpp、GGUFには対応していません。TP2、TP4、1Mコンテキストも未検証です。

## 主な変更点

- 各MoE層で、256個のルーティング対象エキスパートから216個を選択
- 日本語、ツール呼び出し、エージェント型のツール利用、拒否傾向を抑えるエキスパートを優先して保持
- ルーター行と3種類のハッシュルーティングテーブルを、新しいエキスパートIDへ再配置
- BackboneとDSpark/MTPにレイヤーごとの保持マップを適用
- メインモデルとドラフトモデルの位置ずれを補正するアダプターを追加
- Attention、埋め込み層、共有エキスパート、mHC、圧縮器、インデクサー、出力層は量子化元の構成を維持

## 制限事項

- 日本語instruction tuningではなく、expert pruningとroutingの調整を中心としたモデルです。
- 日本語の専門的な選択式問題は、英語MMLU subsetより低い結果でした。
- HellaSwagはMMLUとARCより低い結果でした。
- 公開設定を再現するには位置補正アダプターが必要です。
- 今後のvLLMやSparkInferでは、専用ランタイムをそのまま適用できない可能性があります。
- 検証済みのコンテキスト長は256,000トークン、同時処理数は1です。

## 整合性と出典

`RELEASE_MANIFEST.json`には、マニフェスト自身を除く全ファイルのサイズとSHA-256を記録しています。推論時の重みのマニフェストは`rank-sliced-tp1-manifest.json`、エキスパート保持マップは`REAP_K216_PLAN.json`です。

K216プランは、MITライセンスで公開されている[`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark)を基にしています。

## ライセンスとクレジット

- DeepSeek V4 Flash-0731の重み：MIT
- 0xSeroのK216チェックポイントとEXL3形式：上流のMIT条件
- MiaAI-LabのDGX Spark向けレシピ：MIT
- vLLM / SparkInferのランタイムソース：Apache-2.0
- 本プロジェクトの統合コードと位置補正アダプター：MIT

構成要素ごとの対応と著作権表示は[`LICENSE`](LICENSE)、[`LICENSES/README.md`](LICENSES/README.md)、[`NOTICE.md`](NOTICE.md)に記載しています。

DeepSeek、NVIDIA、0xSero、MiaAI-Lab、vLLM、Cerebras、ExLlamaV3による公式リリースではありません。
