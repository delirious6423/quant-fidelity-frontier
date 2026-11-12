---
license: other
language:
  - ja
  - en
base_model: LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-BestChoice-6G16G-GGUF

VRAM 6GB / RAM 16GB の PC で動く範囲で「汎用最強」を目指した、LFM2.5-24B-A2B の GGUF です。
素のモデルをそのまま量子化するのではなく、(1) 実行時チューニング、(2) 感度実測に基づく独自の量子化レイアウト、(3) 日本語を含む汎用配合での蒸留、の3段階を実機（RTX 3060 Laptop 6GB + RAM 16GB）で測定しながら積み上げた成果物です。

## ファイル情報

| 項目 | 内容 |
|---|---|
| ファイル名 | `LFM2.5-24B-A2B-BestChoice-6G16G-GGUF.gguf` |
| サイズ | 9,443,769,984 バイト（約 9.4GB） |
| SHA-256 | `e56585ced40f894eec70d51877fffc1383c651f07cdf3e435d683717288f86f4` |
| ベースモデル | LiquidAI/LFM2.5-24B-A2B |
| 量子化 | 独自 UD レイアウト（IQ3_XXS ベース、下記） |
| ライセンス | other（ベースモデルおよび教師データの条件に従います） |

## 作り方 1: 実行時チューニング（llama.cpp の MoE expert cache）

LFM2.5-24B-A2B は MoE（総パラメータ約 24B、活性約 2.5B）で、expert の大半は RAM 側に置かれます。llama.cpp の expert cache 機能（`--moe-cache-mib` / `--moe-cache-cpu-share` / ロードモード `-lm dio`、PR #30966）を実測で詰めました。

llama-completion（固定プロンプト、生成 512 トークン、temp 0、ctx 32768、KV cache q8_0）での生成速度:

| 条件 | Abliterated 版 | 蒸留 v1（先行版） |
|---|---|---|
| チューニング前 | 23.54 t/s | 16.23 t/s |
| 3072MiB / share 0.5 / dio | **52.87 t/s** | **65.71 t/s** |

推奨値は **moe-cache-mib 3072、moe-cache-cpu-share 0.5、load-mode dio** です。他の GPU プロセスと VRAM を分け合う環境では 2368 程度に下げると安全です。

## 作り方 2: 独自 UD レイアウト（感度実測に基づくビット配分）

Unsloth の UD（dynamic quantization）は imatrix で重要度を測って量子化しますが、本モデルではさらに一歩進めて、**テンソル種別ごとに「精度を上げると perplexity がどれだけ改善するか」を実測**し、効く場所にだけビットを配分しました。

- 較正用 imatrix は自前で作成（英語 wikitext-2、日本語 Wikipedia、推論サンプルを混ぜた較正テキスト、200 chunks）。
- 評価は較正とは分離した英語/日本語の評価テキストで、llama-perplexity 96 chunks に統一。
- 基準（自前 imatrix + 標準 UD の型配分、8,959,327,872 バイト）: 英語 PPL 14.6249 ± 0.264、日本語 PPL 19.3503 ± 0.385。

感度マップ（基準からの変化。PPL は低いほど良い）:

| 変更 | サイズ増 | 英語 PPL | 日本語 PPL | 判定 |
|---|---|---|---|---|
| expert gate/up 序盤（blk2–9）を iq3_s へ | +264MB | −2.56% | −1.84% | **最良の効率** |
| expert gate/up 中盤（blk10–29）を iq3_s へ | +881MB | −1.97% | −2.51% | 効くが高コスト |
| expert gate/up 終盤（blk30–39）を iq3_s へ | +440MB | −2.21% | −3.22% | 日本語に強い |
| expert down 序盤を iq4_xs へ | +123MB | +0.09% | — | 無効 |
| expert down 中盤を iq4_xs へ | +409MB | −0.85% | — | 非効率 |
| expert down 終盤を iq4_xs へ | +204MB | −0.37% | — | 非効率 |
| shortconv を iq3_s へ | +24MB | +3.67%（悪化） | — | 逆効果 |
| shortconv を q8_0 へ | +342MB | +0.32% | +0.32% | 改善なし（iq3_xxs で損失ほぼゼロ） |
| blk7,8 の既存 boost を解除 | −129MB | +0.74%（悪化） | +0.31%（悪化） | boost は妥当 |

採用した最終レイアウト（C2）: 自前 imatrix + gate/up の blk2–9 と blk35–39 を iq3_s に引き上げ、それ以外は標準 UD の配分（expert gate/up の残りは iq2_s、down は iq3_s、attention・dense FFN・埋め込みは q6_k、shortconv は iq3_xxs、router/norm 類は f32）。

- 結果: 英語 PPL 14.0789（−3.73%）、日本語 PPL 18.5336（−4.22%）、サイズ 9,443,769,984 バイト。
- 実機速度への影響: 同一条件の交互測定で 56.3 t/s（標準レイアウト版 57.4 t/s、−1.9%）、VRAM ピークは不変（4,451MiB）。RAM 16GB でも全 expert が実駐します。

※ 上記の感度測定は、同一のマージ済み重みに対する相対比較です。本モデルの出荷重みは、後述のとおりモジュール別に正しい scaling でマージし直したうえで、このレイアウトを適用しています。

## 作り方 3: 蒸留（汎用配合）

### 学習レシピ

- ベース: LiquidAI/LFM2.5-24B-A2B を Unsloth FastLanguageModel で 4bit ロード、max_seq_len 4096。
- LoRA: r=16 / alpha=32。ただし expert FFN（w1/w2/w3）は rank_pattern / alpha_pattern で r=4 / alpha=8。対象は実在モジュールから自動選択した 46 モジュール（attention、dense FFN、expert FFN、shortconv 周辺）。学習可能パラメータ約 2.46M（先行版での実績値）。
- データ整形: system プロンプトを除去し、assistant の思考過程を `<think>...</think>` 形式に結合、チャットテンプレートで描画、プロンプト部は損失マスク（-100）。全トークンを連結して 4096 トークンのブロックにパッキング。
- 最適化: AdamW (fused)、lr 1e-4、cosine スケジュール、warmup 3、batch 8、1 epoch、bf16。
- 実績: 1,918 ブロック = 240 ステップ、Google Colab L4 で 3,489 秒、**train_loss 0.5764**。
- マージ: アダプタをベースへストリーミングマージ。scaling は adapter_config からモジュール別に計算（全モジュール実効 2.0）。F16 GGUF（47,962,094,976 バイト）を経て、上記 C2 レイアウトで量子化。

### データ配合（14,674 行 / 7,856,616 トークン）

| 区分 | 行数 | 出典・内容 |
|---|---|---|
| 英語推論 | 6,000 | 公開蒸留コーパス `r0b0tlab/qwen3.8-max-glm5.2-distillation-51389` の reasoning 系 |
| コード | 3,000 | 同コーパス |
| 英語短答 | 2,250 | 同コーパス（短い応答で完結するもの） |
| 日本語数学 | 2,600 | NuminaMath-CoT-JA |
| 日本語論理 | 314 | テンプレート生成（答えは構成上保証される） |
| 日本語会話 | 510 | 自作の日本語会話セットを増幅 |

教師コーパスのトレース生成元は Qwen3.8-Max-Preview、GLM-5.2、Kimi K3 です（データセットの内訳による）。

フィルタ方針: 「思考が長い」こと自体は悪ではないので一律の長さ切りはせず、**堂々巡りする非収束トレースと、難易度に不釣り合いな過剰長トレースを除去**しました。また評価ベンチとの混入除去として、ベンチ問題文と 20-gram が重なる行を 296 行除外しています。

## 評価

### 測定条件

- マシン: ノート PC（ASUS Vivobook K6501ZM、Core i7-12650H、RTX 3060 Laptop 6GB、RAM 16GB DDR5-4800 1枚）。
- 推論: llama.cpp b12208 の llama-server。上記の実行時チューニング適用、ctx 8192、temperature 0、最大生成 1024 トークン、Chat Completions API。
- ベンチ: 自作の固定 54 問。内訳は日本語推論 11、日本語指示 10、英語推論 11、コード 11、知識 11。コードは実行テスト、その他はルールベースのチェッカーで採点。追加した 24 問は学習データとの 20-gram 重なり 0 件を確認済み。
- 指標: 完結率（finish_reason = stop の割合）、正答率、完結×正答、日本語思考率（日本語 21 問で思考過程が主に日本語だった割合）。

### 総合結果（54 問）

| モデル | 完結 | 正答 | 完結×正答 | 日本語思考 |
|---|---|---|---|---|
| **本モデル** | **54** | **53** | **53** | **21/21** |
| 先行版 2（先行版 1 + 日本語少量追加、未公開） | 53 | 51 | 51 | 5/21 |
| 先行版 1（英語中心の蒸留、`hirokidesuyo/LFM2.5-24B-A2B-Distill-GGUF`） | 53 | 51 | 51 | 5/21 |
| Abliterated UD-IQ3_XXS（alexwkleung） | 49 | 50 | 49 | 1/21 |
| 素のベース + Unsloth UD-IQ3_XXS（`unsloth/LFM2.5-24B-A2B-GGUF`） | 45 | 45 | 45 | 0/21 |

### ドメイン別（完結×正答）

| ドメイン | 本モデル | 先行版 2 | 先行版 1 | Abliterated | 素 + Unsloth |
|---|---|---|---|---|---|
| コード（11） | **11** | 10 | 10 | 8 | 8 |
| 英語推論（11） | **11** | 11 | 11 | 11 | 10 |
| 日本語指示（10） | **9** | 9 | 8 | 8 | 7 |
| 日本語推論（11） | **11** | 10 | 11 | 11 | 10 |
| 知識（11） | **11** | 11 | 11 | 11 | 10 |

従来から使っている 30 問サブセットでも本モデルが首位（29/30。他は素 24、Abliterated 27、先行版 1 が 27、先行版 2 が 28）。

### 読み取れること

- 素のベースモデルは 9 問すべてが「思考が収束せず上限 1024 トークンで打ち切り」で未達でした。蒸留の主効果は知識の追加ではなく、**難易度に見合った長さで考えて答えまで到達する振る舞い**の獲得です。
- 本モデルの唯一の未達は尊敬語への変換（ji1）で、これは 5 モデル全員が落とした問題です（先行版は「走られた」と正しい尊敬語でも丁寧体でないためチェッカー外、本モデルは言い換え自体が不正確）。
- 日本語の質問には日本語で思考します（21/21）。先行版は英語思考が大半でした。

### その他の指標

- 英語 perplexity: **12.5167 ± 0.161**（llama-perplexity、較正と分離した英語評価テキスト、ctx 4096、96 chunks）。日本語 perplexity は信頼できる測定値が取れなかったため、代わりに上記ベンチの日本語ドメインで評価しています。
- 生成速度: **53.81 t/s**（llama-completion、生成 511 トークン、ctx 32768、KV q8_0、実行時チューニング適用）。54 問ベンチの所要時間（サーバー起動込み）も 4 分 03 秒で、比較した蒸留系の中で最速でした。
- VRAM ピーク: 4,451MiB（このレイアウトの実機検証時）。

## 推奨設定

llama-server の例:

```
llama-server -m LFM2.5-24B-A2B-BestChoice-6G16G-GGUF.gguf \
  -ngl 99 -ncmoe 99 --moe-cache-mib 3072 --moe-cache-cpu-share 0.5 -lm dio \
  -c 32768 -ctk q8_0 -ctv q8_0 --flash-attn on -t 8
```

- UI から使う場合は最大生成トークンを大きめに（思考が長めのモデルです。簡単な質問では短く収束しますが、難しい質問では相応に考えます）。
- VRAM 6GB / RAM 16GB が前提の配分です。RAM が少ない環境では expert の実駐が崩れて大きく速度が落ちます。

## 制限と注意

- 評価は小規模な自作ベンチ（54 問、temp 0、単回測定）によるものです。MMLU 等の標準ベンチではありません。数値は「同一条件での相対比較」として読んでください。
- 事実・知識系の質問では誤ることがあります。重要な用途では裏取りをしてください。
- 本モデルは通常版ベースの蒸留で、abliteration（拒否解除）は施していません。拒否挙動はベースモデル＋蒸留データの影響を受けます。
- 学習データの日本語比率は約 23% です。日本語は改善しましたが、英語が主軸のモデルです。

## クレジット・出典

- ベースモデル: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- 教師データ: [r0b0tlab/qwen3.8-max-glm5.2-distillation-51389](https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-distillation-51389)、NuminaMath-CoT-JA
- 比較に使った量子化: [unsloth/LFM2.5-24B-A2B-GGUF](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)
- ツール: llama.cpp（b12208）、Unsloth

---

# English description

## LFM2.5-24B-A2B-BestChoice-6G16G-GGUF

A GGUF build of LFM2.5-24B-A2B that aims to be the strongest *general-purpose* model you can run on a PC with **6GB of VRAM and 16GB of RAM**. Instead of quantizing the stock weights as-is, it stacks three stages, each tuned by measurements on the target hardware (RTX 3060 Laptop 6GB + 16GB RAM): (1) runtime tuning, (2) a custom quantization layout driven by a measured sensitivity map, and (3) distillation on a general-purpose data mix that includes Japanese.

## File info

| Item | Value |
|---|---|
| File name | `LFM2.5-24B-A2B-BestChoice-6G16G-GGUF.gguf` |
| Size | 9,443,769,984 bytes (~9.4GB) |
| SHA-256 | `e56585ced40f894eec70d51877fffc1383c651f07cdf3e435d683717288f86f4` |
| Base model | LiquidAI/LFM2.5-24B-A2B |
| Quantization | Custom UD layout (IQ3_XXS-based, see below) |
| License | other (subject to the base model and teacher-data terms) |

## Step 1: Runtime tuning (llama.cpp MoE expert cache)

LFM2.5-24B-A2B is an MoE model (~24B total, ~2.5B active); most experts live in system RAM. We measured llama.cpp's expert-cache options (`--moe-cache-mib` / `--moe-cache-cpu-share` / load mode `-lm dio`, PR #30966) on the target machine.

Generation speed with llama-completion (fixed prompt, 512 generated tokens, temp 0, ctx 32768, KV cache q8_0):

| Setting | Abliterated build | Distilled v1 (earlier build) |
|---|---|---|
| Before tuning | 23.54 t/s | 16.23 t/s |
| 3072MiB / share 0.5 / dio | **52.87 t/s** | **65.71 t/s** |

Recommended: **moe-cache-mib 3072, moe-cache-cpu-share 0.5, load-mode dio**. If other GPU processes share VRAM, 2368 is a safer value.

## Step 2: Custom UD layout (bits allocated by measured sensitivity)

Unsloth's UD (dynamic quantization) uses an imatrix to weight importance. We went one step further: for each tensor class we **measured how much perplexity actually improves when its precision is raised**, and spent bits only where they pay off.

- The calibration imatrix is our own (200 chunks over a mix of English wikitext-2, Japanese Wikipedia, and reasoning samples).
- Evaluation used held-out English/Japanese texts, llama-perplexity with 96 chunks throughout.
- Baseline (own imatrix + stock UD type assignment, 8,959,327,872 bytes): EN PPL 14.6249 ± 0.264, JA PPL 19.3503 ± 0.385.

Sensitivity map (change vs. baseline; lower PPL is better):

| Change | Size delta | EN PPL | JA PPL | Verdict |
|---|---|---|---|---|
| Expert gate/up, early layers (blk2–9) → iq3_s | +264MB | −2.56% | −1.84% | **Best efficiency** |
| Expert gate/up, middle (blk10–29) → iq3_s | +881MB | −1.97% | −2.51% | Works, expensive |
| Expert gate/up, late (blk30–39) → iq3_s | +440MB | −2.21% | −3.22% | Strong for Japanese |
| Expert down, early → iq4_xs | +123MB | +0.09% | — | No effect |
| Expert down, middle → iq4_xs | +409MB | −0.85% | — | Inefficient |
| Expert down, late → iq4_xs | +204MB | −0.37% | — | Inefficient |
| shortconv → iq3_s | +24MB | +3.67% (worse) | — | Counterproductive |
| shortconv → q8_0 | +342MB | +0.32% | +0.32% | No gain (loss at iq3_xxs is ~zero) |
| Removing the stock blk7,8 boost | −129MB | +0.74% (worse) | +0.31% (worse) | The boost earns its bytes |

Final layout (C2): own imatrix + gate/up of blk2–9 and blk35–39 raised to iq3_s; everything else follows the stock UD assignment (remaining expert gate/up at iq2_s, expert down at iq3_s, attention / dense FFN / embeddings at q6_k, shortconv at iq3_xxs, router/norm tensors at f32).

- Result: EN PPL 14.0789 (−3.73%), JA PPL 18.5336 (−4.22%), size 9,443,769,984 bytes.
- On-device cost: 56.3 t/s vs. 57.4 t/s for the stock-layout build in an interleaved A/B test (−1.9%), with an unchanged VRAM peak (4,451MiB). All experts stay resident in 16GB of RAM.

Note: the sensitivity figures above are relative comparisons measured on one shared set of merged weights. The shipped weights were re-merged with correct per-module scaling (below) before this layout was applied.

## Step 3: Distillation (general-purpose mix)

### Training recipe

- Base: LiquidAI/LFM2.5-24B-A2B, loaded in 4-bit with Unsloth FastLanguageModel, max_seq_len 4096.
- LoRA: r=16 / alpha=32, except expert FFN matrices (w1/w2/w3) at r=4 / alpha=8 via rank/alpha patterns. 46 target modules chosen automatically from what exists (attention, dense FFN, expert FFN, shortconv-adjacent modules); ~2.46M trainable parameters (as measured on the earlier build).
- Formatting: system prompts dropped; assistant reasoning folded into `<think>...</think>` form; rendered with the chat template; prompt tokens masked (−100). All tokens concatenated and packed into 4096-token blocks.
- Optimization: fused AdamW, lr 1e-4, cosine schedule, warmup 3, batch 8, 1 epoch, bf16.
- Run: 1,918 blocks = 240 steps, 3,489 s on a Colab L4, **train_loss 0.5764**.
- Merge: the adapter was stream-merged into the base weights with per-module scaling computed from adapter_config (effective 2.0 for every module), exported to an F16 GGUF (47,962,094,976 bytes), then quantized with the C2 layout above.

### Data mix (14,674 rows / 7,856,616 tokens)

| Slice | Rows | Source / content |
|---|---|---|
| English reasoning | 6,000 | Public distillation corpus `r0b0tlab/qwen3.8-max-glm5.2-distillation-51389`, reasoning split |
| Code | 3,000 | Same corpus |
| English short answers | 2,250 | Same corpus (items that conclude in a short response) |
| Japanese math | 2,600 | NuminaMath-CoT-JA |
| Japanese logic | 314 | Template-generated (answers correct by construction) |
| Japanese conversation | 510 | Self-authored Japanese conversation set, augmented |

The teacher traces in the corpus were generated by Qwen3.8-Max-Preview, GLM-5.2, and Kimi K3 (per the dataset's own breakdown).

Filtering policy: long thinking is not inherently bad, so there was no blanket length cap; instead we removed **non-convergent traces (reasoning loops) and traces whose length was disproportionate to the task**. For decontamination, 296 rows overlapping the evaluation bench at the 20-gram level were removed.

## Evaluation

### Setup

- Machine: laptop (ASUS Vivobook K6501ZM, Core i7-12650H, RTX 3060 Laptop 6GB, 16GB DDR5-4800, single stick).
- Inference: llama-server from llama.cpp b12208, runtime tuning above applied, ctx 8192, temperature 0, max 1024 generated tokens, Chat Completions API.
- Bench: 54 self-made fixed items — Japanese reasoning 11, Japanese instruction-following 10, English reasoning 11, code 11, knowledge 11. Code items are graded by executing tests; the rest use rule-based checkers. The 24 added items were verified to have zero 20-gram overlap with the training data.
- Metrics: completion rate (finish_reason = stop), accuracy, completion × accuracy, and Japanese-thinking rate (share of the 21 Japanese items whose reasoning was primarily in Japanese).

### Overall results (54 items)

| Model | Completed | Correct | Compl. × Acc. | JA thinking |
|---|---|---|---|---|
| **This model** | **54** | **53** | **53** | **21/21** |
| Earlier build 2 (build 1 + a small amount of Japanese, unpublished) | 53 | 51 | 51 | 5/21 |
| Earlier build 1 (English-centric distillation, `hirokidesuyo/LFM2.5-24B-A2B-Distill-GGUF`) | 53 | 51 | 51 | 5/21 |
| Abliterated UD-IQ3_XXS (by alexwkleung) | 49 | 50 | 49 | 1/21 |
| Vanilla base + Unsloth UD-IQ3_XXS (`unsloth/LFM2.5-24B-A2B-GGUF`) | 45 | 45 | 45 | 0/21 |

### Per-domain (completion × accuracy)

| Domain | This model | Build 2 | Build 1 | Abliterated | Vanilla + Unsloth |
|---|---|---|---|---|---|
| Code (11) | **11** | 10 | 10 | 8 | 8 |
| English reasoning (11) | **11** | 11 | 11 | 11 | 10 |
| Japanese instruction (10) | **9** | 9 | 8 | 8 | 7 |
| Japanese reasoning (11) | **11** | 10 | 11 | 11 | 10 |
| Knowledge (11) | **11** | 11 | 11 | 11 | 10 |

On the original 30-item subset this model also leads (29/30; vanilla 24, Abliterated 27, build 1: 27, build 2: 28).

### What the numbers say

- Every one of the vanilla base model's 9 misses was a length cutoff: its thinking never converged within 1024 tokens. The main effect of the distillation is not extra knowledge — it is **learning to think at a length proportionate to the task and actually reach an answer**.
- This model's only miss is an honorific-conversion item (ji1) that all five models missed (the earlier builds produced a valid honorific in plain form, which the checker rejects; this model's conversion itself was inaccurate).
- It thinks in Japanese for Japanese questions (21/21); the earlier builds mostly thought in English.

### Other measurements

- English perplexity: **12.5167 ± 0.161** (llama-perplexity, held-out English eval text, ctx 4096, 96 chunks). We could not obtain a reliable Japanese perplexity value, so Japanese ability is assessed through the bench domains above instead.
- Generation speed: **53.81 t/s** (llama-completion, 511 generated tokens, ctx 32768, KV q8_0, runtime tuning applied). It also finished the 54-item bench fastest among the distilled builds (4m03s including server start).
- VRAM peak: 4,451MiB (measured when validating this layout on-device).

## Recommended settings

llama-server example:

```
llama-server -m LFM2.5-24B-A2B-BestChoice-6G16G-GGUF.gguf \
  -ngl 99 -ncmoe 99 --moe-cache-mib 3072 --moe-cache-cpu-share 0.5 -lm dio \
  -c 32768 -ctk q8_0 -ctv q8_0 --flash-attn on -t 8
```

- When driving it from a UI, give it a generous max-token budget: it is a thinking model. Easy questions converge quickly; hard ones legitimately think longer.
- The layout assumes 6GB VRAM / 16GB RAM. With less RAM the experts stop being resident and speed collapses.

## Limitations

- Evaluation is a small self-made bench (54 items, temp 0, single run) — not MMLU or any standard benchmark. Read the numbers as same-conditions relative comparisons.
- It can still be wrong on factual/knowledge questions. Verify anything important.
- This is a distillation of the *vanilla* base — no abliteration was applied. Refusal behavior follows the base model as modified by the distillation data.
- Japanese is ~23% of the training mix. Japanese improved substantially, but English remains the model's center of gravity.

## Credits and sources

- Base model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
- Teacher data: [r0b0tlab/qwen3.8-max-glm5.2-distillation-51389](https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-distillation-51389), NuminaMath-CoT-JA
- Stock quantization used for comparison: [unsloth/LFM2.5-24B-A2B-GGUF](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF)
- Tooling: llama.cpp (b12208), Unsloth
