---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- ja
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- oddeye
- michiyomi
- street-level-imagery
- urban
- nvfp4
- compressed-tensors
- experimental
model-index:
- name: OddEye 0.1 27B
  results:
  - task:
      type: image-text-to-text
      name: Street photo to structured Japanese observation (prompt v5.2c)
    dataset:
      name: michiyomi blind evaluation, 100 street photos (AI-written reference facts)
      type: michiyomi-blind-100
    metrics:
    - type: reference_fact_score
      name: Reference facts captured (held + 0.5 x partial, higher is better)
      value: 0.534
    - type: contradicted_fact_rate
      name: Reference facts contradicted, % (lower is better)
      value: 11.3
    - type: overconfident_per_photo
      name: Overconfident statements per photo (lower is better)
      value: 0.69
    - type: clear_errors_per_photo
      name: Clear errors per photo (lower is better)
      value: 0.26
  - task:
      type: image-text-to-text
      name: Street photo to structured Japanese observation (prompt v5.2c)
    dataset:
      name: michiyomi evaluation, 140 street photos (internal)
      type: michiyomi-140
    metrics:
    - type: valid_structure_rate
      name: Outputs with all 13 top-level fields, %
      value: 98.6
    - type: throughput_rtx_pro_6000
      name: Images per hour, RTX PRO 6000, 32 concurrent
      value: 1833
---

![OddEye — street photos in, structured observations out](assets/oddeye-banner.png)

# OddEye 0.1 27B (experimental)

**One street photo in, a structured Japanese observation record out.** OddEye is the street-reading model of [michiyomi](https://michiyomi.dev). It is built on Qwen3.8-27B and compressed to 4-bit (NVFP4, about 20 GB), so a single GPU — even an RTX 5090 — can run it.

![OddEye example: a street photo and excerpts of the observation and interpretation fields](assets/oddeye-example.jpg)

It keeps **what it sees** (observations) and **what it infers** (interpretations, each with a confidence label) in separate fields. The record has 13 top-level fields, including objects, text on signs, relations, surfaces, standard road facets (lanes, sidewalks, lighting…), interpretations and a scene summary.

## Results

![Overconfident statements, clear errors, contradicted facts and throughput for the BF16 original, a generic 4-bit version and OddEye](assets/oddeye-results.png)

- **Fewer mistakes than a generic 4-bit version** of the same size: fewer overconfident statements and fewer clear errors.
- **Captures about as many facts as the original model** (0.53 against 0.54).
- **1.8× faster than the original**, and it fits on one RTX 5090 (about 27 s per photo).

### Scores

Blind comparison on 100 street photos, with the recommended prompt.

| | Original (BF16) | Generic 4-bit | **OddEye** |
|---|---:|---:|---:|
| Reference facts captured ↑ | 0.54 | 0.53 | **0.53** |
| Reference facts contradicted ↓ | 9.3% | 12.4% | **11.3%** |
| Overconfident statements per photo ↓ | 0.70 | 0.97 | **0.69** |
| Clear errors per photo ↓ | 0.35 | 0.43 | **0.26** |
| Useful observations beyond the references, per photo ↑ | 1.67 | 1.58 | **1.72** |

### Speed and size

| | Original (BF16) | Generic 4-bit | **OddEye** |
|---|---:|---:|---:|
| Weights | 55.6 GB | 20.6 GB | **20.6 GB** |
| Images per hour, RTX PRO 6000 (32 at once) ↑ | 1,020 | 1,749 | **1,833** |
| RTX 5090 (32 GB) | does not fit | not tested | **27.5 s per photo; ~900 images/h (8 at once)** |

### How it was evaluated

- 100 street photos from Japan, 40 of which were never seen while developing the prompt or the model. None share an image sequence with the training photos.
- Each photo has about 15 reference facts written by AI from the photo. A Claude model (Anthropic) graded all outputs side by side, without knowing which model wrote which. Read the numbers as a comparison between models, not as absolute accuracy.
- "Generic 4-bit" is a publicly distributed NVFP4 version of the same base model. All three models were run with the same settings, including MTP speculative decoding.

## Quick start

```sh
vllm serve finalvent/OddEye-0.1-27B-NVFP4 --served-model-name oddeye \
  --max-model-len 16384 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --limit-mm-per-prompt '{"image":1}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-prefix-caching
# RTX 5090 (32 GB): add --max-num-seqs 8 --gpu-memory-utilization 0.92

python3 example_request.py photo.jpg > observation.json
```

Use the included prompt, `prompt_v5.2c.md`; the model was specialised on it. `example_request.py` (standard library only) sends exactly the request that was evaluated, and warns if an output is cut off at the 8,192-token limit (about 1 in 70). Tested with vLLM 0.27–0.28 on Blackwell GPUs.

## Before you use the output

- It describes only what is visible in that photo, at the time it was taken — not the current state of the place.
- Widths and lane counts are estimates from one image, not measurements. "Not visible" is a real answer; don't fill in what the record does not say.
- Lane counts, sidewalks and small text are where it slips most; check those fields when they matter. Some interpretations are generic and add little.
- It occasionally transcribes a licence-plate number (1 photo in 140, as did the original). Redact plates and personal names before publishing outputs.
- Don't use it as the only basis for safety or accessibility decisions.

## About

- Version 0.1, an experimental release from the michiyomi project for the Tokyo Governor's Cup 2026 Final Stage. It pairs with prompt v5.2c. Later versions will be separate repositories; the full version, OddEye 1.0, will be built on the next Qwen generation.
- Trained only on the base model's own answers (4-bit-aware self-distillation on 400 photos); no outputs of external AI services were used.
- Training photos: street-level imagery from [Mapillary](https://www.mapillary.com) (not included).
- License: Apache-2.0 (same as the base model). No conditions on the outputs. The example figure contains a photo by nashitakahashi via Mapillary and is shared under CC BY-SA 4.0.

---

## 日本語

### OddEye 0.1 27B（実験版）

**街路の写真を1枚入れると、日本語の構造化された観測記録が返ってきます。** OddEyeは、[みちよみ](https://michiyomi.dev)の街路を読むモデルです。Qwen3.8-27Bを元に、4bit（NVFP4、約20GB）に縮めたので、GPU 1枚で動きます。RTX 5090でも動きます。

例の図はページの先頭にあります。**見えたもの**（観測）と、**推し量ったこと**（解釈。それぞれに確度が付きます）を、別々の欄に書きます。記録の最上位の欄は13あり、物、看板の文字、物どうしの関係、表面の状態、道路の標準項目（車線・歩道・街路灯など）、解釈、場面の要約などがあります。

### 結果

![元のモデル・一般的な4bit版・OddEyeの、断定のしすぎ、明らかな誤り、参照factとの矛盾、処理量](assets/oddeye-results-ja.png)

- **同じ大きさの一般的な4bit版より、誤りが少ない**：断定のしすぎと、明らかな誤りが少なくなりました。
- **参照factを捉える度合いは、元のモデルとほぼ同じ**です（0.53と0.54）。
- **元のモデルの1.8倍速く**、RTX 5090 1枚に収まります（1枚あたり約27秒）。

#### 成績

100枚の街路写真での盲検比較です。推奨プロンプトを使っています。

| | 元のモデル（BF16） | 一般的な4bit版 | **OddEye** |
|---|---:|---:|---:|
| 参照factを捉えた度合い（0〜1） ↑ | 0.54 | 0.53 | **0.53** |
| 参照factと矛盾した割合 ↓ | 9.3% | 12.4% | **11.3%** |
| 断定のしすぎ（1枚あたり） ↓ | 0.70 | 0.97 | **0.69** |
| 明らかな誤り（1枚あたり） ↓ | 0.35 | 0.43 | **0.26** |
| 参照にない有用な観測（1枚あたり） ↑ | 1.67 | 1.58 | **1.72** |

#### 速さと大きさ

| | 元のモデル（BF16） | 一般的な4bit版 | **OddEye** |
|---|---:|---:|---:|
| 重みの大きさ | 55.6 GB | 20.6 GB | **20.6 GB** |
| 1時間あたりの画像数（RTX PRO 6000、同時32件） ↑ | 1,020 | 1,749 | **1,833** |
| RTX 5090（32 GB） | 載らない | 試していない | **1枚あたり27.5秒、同時8件で毎時約900枚** |

#### 評価のしかた

- 日本の街路写真100枚です。うち40枚は、プロンプトやモデルを作るあいだに一度も見ていない写真です。どの写真も、学習に使った写真と撮影系列が重なりません。
- 各写真には、AIが写真から書いた参照factが約15件あります。Claude（Anthropic）が、どのモデルの出力かを伏せたまま、すべての出力を並べて採点しました。数字は絶対的な正確さではなく、モデル同士の比較として読んでください。
- 「一般的な4bit版」は、同じ元モデルを配布用にNVFP4へ量子化した版です。3つのモデルは、MTPの先読みを含めて同じ設定で動かしました。

### 使い方

```sh
vllm serve finalvent/OddEye-0.1-27B-NVFP4 --served-model-name oddeye \
  --max-model-len 16384 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --limit-mm-per-prompt '{"image":1}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --enable-prefix-caching
# RTX 5090（32 GB）では --max-num-seqs 8 --gpu-memory-utilization 0.92 を足す

python3 example_request.py photo.jpg > observation.json
```

同梱の推奨プロンプト `prompt_v5.2c.md` を使ってください。モデルはこのプロンプトに特化しています。`example_request.py`（Pythonの標準ライブラリだけで動きます）は、評価と同じ形の要求を送ります。出力が8,192 tokenの上限で途切れたとき（約70回に1回）は、警告を出します。動作は、Blackwell世代のGPUと vLLM 0.27〜0.28 で確かめました。

### 出力を使う前に

- 出力は、その写真に写っているものを、撮影した時点で書いたものです。その場所の現在の状態ではありません。
- 幅や車線数は、写真1枚からの推定値で、測った値ではありません。「見えない」も意味のある答えです。記録にないことを補わないでください。
- 間違えやすいのは、車線数、歩道、小さな文字です。大事な場合は、これらの欄を確かめてください。解釈には、一般的で情報の少ないものも混じります。
- まれにナンバープレートの数字を書き写すことがあります（140枚中1枚。元のモデルも同じ）。出力を公開するときは、ナンバーや個人名を伏せてください。
- 安全やバリアフリーの判断の、唯一の根拠にはしないでください。

### このモデルについて

- 版は0.1で、みちよみプロジェクトの実験版です。都知事杯2026のFinal Stageに向けて公開しました。推奨プロンプト v5.2c と組で使います。次の版は別のリポジトリとして公開します。本格版のOddEye 1.0は、次の世代のQwenで作る予定です。
- 学習には、元のモデル自身の回答だけを使いました（400枚の写真で、4bitを意識した自己蒸留）。外部のAIサービスの出力は使っていません。
- 学習に使った写真は、[Mapillary](https://www.mapillary.com) の街路写真です（このリポジトリには含めていません）。
- ライセンスは、元のモデルと同じApache-2.0です。出力には条件を付けません。例の図には、Mapillaryのnashitakahashiさんの写真を使っていて、図はCC BY-SA 4.0で共有しています。
