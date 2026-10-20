---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- ja
- en
tags:
- deepseek
- deepseek-v4.1-flash
- exl3
- exllamav3
- ds41rt
- mixture-of-experts
- 2-bit
- quantized
- calibration
- distillation
- japanese
- vision
- rtx-pro-6000
- blackwell
- sm120
---

# DeepSeek-V4.1-Flash-1017 EXL3 2.0 bpw ja-calibrated

A 2.0 bpw EXL3 quantization of the routed experts of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
followed by a lightweight layer-wise scale calibration (scale-only distillation of each expert's input/output scales
against the original expert outputs), with calibration data and loss weights tilted toward Japanese, images and agentic use.
It is built to run on **two RTX PRO 6000 Blackwell (96 GB) cards** with
[ds41rt](https://github.com/tpurtell/ds41rt) in its 0-spark (RTX-only) layout.

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) の routed expert を
EXL3 2.0 bpw に量子化し、各 expert の入出力スケールだけを層ごとに原重みの expert 出力へ合わせる軽量な校正 (scale-only の局所蒸留) をかけたものです。
校正データと損失の重みは日本語・画像・エージェント用途に寄せています。
**RTX PRO 6000 Blackwell (96 GB) 2 枚**で、[ds41rt](https://github.com/tpurtell/ds41rt) の 0-spark (RTX だけの) 構成で動かすことを前提にしています。

> **This is a 2-bit model. It is measurably worse than the original weights** (see [Quality](#quality--品質) and [Limitations](#limitations--制約)).
>
> **2bit のモデルなので、原重みより確実に劣ります** (品質・制約の節を参照)。

---

## What was done / 何をしたか

| Part / 部分 | Format in this repo / 形式 | Changed? / 変更 |
|---|---|---|
| Routed experts, layers 0-39 (shards 3-42) | **EXL3 K2, MCG codebook** (2.0 bpw), calibrated `suh`/`svh` | yes / あり |
| Attention, shared experts, router, norms, hyper-connections | official FP8 (E4M3, UE8M0 block 32x32), unchanged | no / なし |
| Embedding, LM head, vision tower + aligner (shards 1, 2, 43) | official, unchanged | no |
| DSpark / MTP draft layers `mtp.0-2` (shards 44-46) | official source precision (`mtp_experts: source`) | no |
| Engram tables, layers 1 and 14 (shards 47, 48, ~101.5 GB each) | official FP8, unchanged | no |

Shards 1, 2 and 43-48 are byte-identical to the official 1017 checkpoint. In shards 3-42 only the routed-expert tensors differ;
`config.json` differs from the official one only in `quantization_config`.

shard 1・2・43〜48 は公式 1017 とバイト単位で同じです。shard 3〜42 で違うのは routed expert のテンソルだけで、
`config.json` も公式との違いは `quantization_config` だけです。

### 1. EXL3 K2 MCG (Viterbi + scale refit)

- Every routed expert projection (`w1`/`w3`/`w2`, 40 layers x 384 experts) is encoded with
  [exllamav3](https://github.com/turboderp-org/exllamav3) 1.5.1 as a 2-bit trellis with the MCG codebook,
  re-encoded from the original MXFP4 weights with tail-biting Viterbi (optimal trellis path) instead of greedy search,
  followed by a refit of the scales with identity Hessian (H = I), following sfxnz's `requant_full.py` recipe.
- Compared with a greedy beam-16 encoding (exllamav3 1.4.9, same format and size), relative weight error dropped
  from 0.378 to 0.262 and teacher-forced KL dropped by about 62% (table below, rows v0 and v1).

- routed expert の各射影 (`w1`/`w3`/`w2`、40 層 x 384 expert) を exllamav3 1.5.1 で 2bit の trellis (MCG codebook) に符号化しました。
  原本の MXFP4 から tail-biting Viterbi で最適な trellis の経路を選び (貪欲探索の代わり)、そのあと H = I で scale を当て直しています (sfxnz の `requant_full.py` の手順)。
- 同じ形式・同じサイズの貪欲 (beam16) 符号化 (exllamav3 1.4.9) と比べて、重みの相対誤差は 0.378 → 0.262、teacher-forced の KL は約 62% 下がりました (下表の v0 と v1)。

### 2. Layer-wise scale calibration (scale-only local distillation)

- **What this is not:** no weights and no discrete codes are trained, and there is no end-to-end (whole-model) distillation;
  each expert's scales are fitted on their own, layer by layer, to the teacher's output for that expert.
  It is a calibration of about 342M scale values (22,272 x 15,360 experts), not full quantization-aware training or distillation.
  The companion repository's scripts and notes call this step "QAD L2" (`quant/harness/qad_l2.py`); there the name means only this scale-only local distillation.
- The discrete EXL3 codes (trellis) are **frozen**. Only the per-expert input/output sign-scale vectors
  `suh`/`svh` of `w1`/`w3`/`w2` (22,272 values per expert) are trained, as a multiplicative correction
  `s = s_v1 * (1 + delta)` with an L2 penalty on `delta`, so the file format and size are unchanged.
- Target: the output of the same expert in the original MXFP4 teacher, for the same captured inputs
  (MoE inputs after `ffn_norm`, teacher top-6 routing and route weights), per expert and per layer.
  Adam, lr 5e-4, 4 epochs, L2 coefficient 0.1. Japanese tokens weighted x2.0, image tokens x1.5.
- Acceptance per expert on held-out sequences: an expert keeps its new scales only if its validation error improves
  (measured on Japanese tokens when the expert sees enough Japanese);
  an **image guard** additionally rejects the update when the image-only validation error gets worse than v1.
  Result: 14,582 of 15,360 experts updated, 51 rejected, 154 rejected by the image guard,
  573 left as-is because they saw too few calibration tokens.

- **しないこと:** 重みや離散コードは学習せず、モデル全体での end-to-end の蒸留もしていません。各 expert のスケールを、層ごとに、その expert の教師出力へ個別に合わせるだけです
  (計約 3.4 億値 = 22,272 x 15,360 expert)。全重みを学習する本来の量子化考慮学習・蒸留 (QAT / QAD) ではありません。
  付属リポジトリのスクリプトやメモではこの段を「QAD L2」(`quant/harness/qad_l2.py`) と呼んでいますが、そこでの QAD はこの scale-only の局所蒸留だけを指します。
- EXL3 の離散コード (trellis) は**固定**です。学習するのは各 expert の `w1`/`w3`/`w2` の `suh`/`svh` (expert あたり 22,272 値) だけで、
  `s = s_v1 * (1 + delta)` の乗法形と `delta` の L2 罰則で学習するので、ファイル形式もサイズも変わりません。
- 目標は、同じ入力 (`ffn_norm` 後の MoE 入力、教師の top-6 routing と重み) に対する MXFP4 教師の同じ expert の出力です。expert ごと・層ごとに合わせます。
  Adam、lr 5e-4、4 epoch、L2 係数 0.1。日本語トークンは 2.0 倍、画像トークンは 1.5 倍の重みです。
- expert ごとに、別に取っておいた系列での検証誤差 (日本語を十分に通る expert では日本語トークンの誤差) が下がったときだけ新しい scale を採用します。さらに**画像ガード**で、画像だけの検証誤差が v1 より悪くなる更新を捨てます。
  結果は 15,360 expert のうち採用 14,582、不採用 51、画像ガードで不採用 154、校正トークンが少なすぎて据え置き 573 です。

### Calibration data / 校正データ (contents not released / 中身は公開しません)

| Set / 集合 | Size / 量 | Composition / 構成 |
|---|---|---|
| Text (`calib256k`) | 79 sequences, 260,849 tokens | Japanese 40%, agentic 34%, code 13%, English 8%, other languages 5% (by tokens) |
| Images (`vl_calib`) | 24 sequences, 13,207 tokens (9,758 image tokens) | Japanese 16 / English 8: OCR, charts, documents, screenshots, VQA |

- Text sources: private agentic/coding request logs from the author's own use, public Japanese corpora
  (Japanese Wikipedia, Aozora Bunko, llm-jp/oasst2-33k-ja, kunishou/databricks-dolly-15k-ja, Japanese technical text)
  and the calibration set of [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41).
  The private logs are not released. The evaluation sets below are disjoint from calibration (time split and 16-gram shingle filtering).
- Image sources: public VQA/OCR/chart benchmarks and synthetic renders. Some of these images cannot be redistributed
  (e.g. DocVQA is research-only, ChartQA is GPL), so **no calibration or evaluation data is included in this repository**.
- テキストの出典は、作者自身の利用から取ったエージェント/コーディング要求のログ (非公開)、公開の日本語コーパス
  (日本語 Wikipedia、青空文庫、llm-jp/oasst2-33k-ja、kunishou/databricks-dolly-15k-ja、日本語の技術文書)、
  [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) の校正セットです。下の評価集合とは時間分割と 16-gram shingle で分けています。
- 画像は公開の VQA/OCR/チャートのベンチマークと合成画像です。再配布できない画像 (DocVQA は研究目的限定、ChartQA は GPL など) を含むので、
  **校正・評価のデータはこのリポジトリに入れていません**。

---

## Quality / 品質

### Teacher-forced KL against the original weights / 原重みとの teacher-forced KL

Teacher = official 1017 weights (MXFP4 experts + FP8 Engram) in a reference harness.
KL(P_teacher || Q_student) approximated on the teacher's top-32, mean per scored token (median in parentheses);
top-1 = student argmax equals teacher argmax. 95% intervals are sequence-bootstrap.
Without quantization the harness scores KL 0.0000 / top-1 100%.

教師は公式 1017 の原重み (MXFP4 expert + FP8 Engram) です。KL は教師の top-32 上で近似した KL(P||Q) の採点トークン平均 (括弧内は中央値)、
top-1 は学生の argmax が教師の argmax と一致する割合です。区間は系列ブートストラップの 95% です。量子化しなければ KL 0.0000 / top-1 100% になります。

| Set / 集合 | Tokens / 採点 tok | v0: greedy EXL3 | v1: Viterbi + refit | **this model (v1 + scale calibration)** | vs v1 [95%] |
|---|---|---|---|---|---|
| agent24: agentic requests | 11,284 | 0.158 / 86.4% | 0.0593 / 91.8% | **0.0462 / 92.5%** | -22.1% [-26.0, -18.6] |
| ja16: Japanese answers to agent requests | 9,014 | 0.206 / 84.6% | 0.0739 / 90.5% | **0.0598 / 91.5%** | -19.1% [-23.0, -16.1] |
| jakl32: Japanese prose (Wikipedia, Aozora, technical, chat) | 52,287 | - | 0.1393 / 87.7% | **0.1196 / 88.5%** | -14.2% [-16.3, -12.3] |
| images, Japanese questions (28) | 3,972 (all images) | 0.267 / 86.4% | 0.1385 / 90.2% | **0.1268 / 91.1%** | -8.5% [-16.1, -3.6] |
| images, English questions (22) | | 0.197 / 86.7% | 0.1084 / 92.1% | **0.1022 / 91.9%** | -5.7% [-12.1, +2.8] |
| images, all (50) | | 0.251 | 0.1318 / 90.7% | **0.1213 / 91.3%** | -8.0% [-13.2, -3.7] |

Cells are KL / top-1. The agentic and Japanese evaluation requests come from private logs and are not released.
セルは KL / top-1 です。エージェントと日本語の評価要求は非公開のログ由来で、公開しません。

### Chinese-only token mass / 中国語専用トークンの確率質量

Low-bit quantization of this model tends to leak Simplified-Chinese-only tokens into Japanese answers.
Below is the student's total probability on Chinese-only tokens relative to the teacher's (1.0 = same as the original), in answer spans, teacher-forced.

このモデルを低ビットにすると、日本語の回答に簡体字専用のトークンが混ざりやすくなります。
下表は回答部分で、中国語専用トークンに乗る確率の合計を教師比で示したものです (1.0 = 原重みと同じ、teacher-forced)。

| Set / 集合 | v0 | v1 | **this model** |
|---|---|---|---|
| ja16 answers | 7.83x | 2.20x | **0.98x** |
| agent24 answers | 14.45x | 3.16x | **1.31x** |
| jakl32 | - | 2.89x | **2.08x** |
| images, Japanese answers | - | 4.11x | ~0.95x (calibration pilot; wide interval) |

Values near 1.0x mean about the same as the original weights; they are not a claim of less leakage than the original.
1.0 倍前後は原重みと同程度という意味で、原重みより混入が少ないことを示すものではありません。

### Generation checks on ds41rt / ds41rt での生成の確認

| Check / 項目 | v1 | **this model** |
|---|---|---|
| Image QA accuracy, 245 questions (ja 145 / en 100), greedy, thinking off | 87.9% | **88.0%** (paired diff +0.1 [-1.2, +1.4]) |
| - Japanese / English | 85.8% / 91.0% | 85.5% / 91.7% |
| Simplified Chinese in Japanese image answers | 0 / 145 | 0 / 145 |
| Chinese mixing in Japanese text generation (200 prompts, T=1.0, 1 seed) | 1.19% | 2.35% |
| Japanese reply, parallel tool calls | OK | OK |

The text-generation mixing rates differ by 2-4 answers and are statistically indistinguishable at this sample size;
the teacher-forced mass above is the more sensitive measure. Image questions are drawn from JMMMU, MECHA-ja, CC-OCR (Japanese),
JA-VG-VQA-500, ChartQA, DocVQA, TextVQA, MMBench and synthetic Japanese/English renders (charts, receipts, manuals, terminal and web screenshots).

テキスト生成の混入率の差は 2〜4 件で、この件数では統計的に区別できません。teacher-forced の質量のほうが感度の高い指標です。
画像の問いは JMMMU、MECHA-ja、CC-OCR (日本語)、JA-VG-VQA-500、ChartQA、DocVQA、TextVQA、MMBench と、日本語・英語の合成画像 (グラフ、レシート、説明書、端末・Web の画面) から取りました。

---

## Speed / 速度

ds41rt v15, 0-spark layout (all 40 expert layers on the two RTX cards), DSpark on, measured on
**2x RTX PRO 6000 Blackwell Max-Q (300 W)**, 128 GB RAM, Engram tables on NVMe.

ds41rt v15 の 0-spark 構成 (expert 40 層をすべて RTX 2 枚に置く)、DSpark 有効、**RTX PRO 6000 Blackwell Max-Q (300 W) 2 枚**、RAM 128 GB、Engram 表は NVMe で測りました。

| Workload / 負荷 | tok/s |
|---|---|
| ds41rt release decode bench (`scripts/bench-ds41-release-decode.py`, greedy, 9 categories), weighted | **147.9** (code 201, structured JSON 177-182) |
| Concurrent decode, T=0.6 top_p=0.95, thinking on, 1,024 tokens: total at c1 / c4 / c8 / c16 | **116 / 220 / 308 / 413** (c16: ~31 per stream) |
| Prefill, 8K / 36K / 67K prompt (measured with the v1 pack, same format) | 4,348 / 4,340-4,848 / 4,661-4,697 |
| Cold start | ~50 s |

GPU memory in use is about 93 GB per card with the default 16-request configuration.
For reference, the same v1 weights on vLLM + vllm-exl3 measured c16 279-284 tok/s total and prefill 1,597-2,060 tok/s on this machine.

既定の 16 並列の設定で GPU メモリは 1 枚あたり約 93 GB 使います。
参考として、同じ v1 の重みを vLLM + vllm-exl3 で動かすと、この機械で c16 合計 279〜284 tok/s、prefill 1,597〜2,060 tok/s でした。

---

## Usage / 使い方

### Requirements / 必要なもの

- 2x 96 GB Blackwell GPUs (tested: RTX PRO 6000 Max-Q; SM120) with bidirectional peer access
- ~128 GB RAM (the Engram tables are read from disk through the page cache; they are not pinned)
- ~360 GB of fast local storage (NVMe recommended) for the checkpoint
- Docker with the NVIDIA runtime, and the ds41rt v15 images (`ghcr.io/tpurtell/ds41rt-coordinator:v15`)

### 1. Download into the Hugging Face cache layout / HF キャッシュ形式で置く

ds41rt reads the model from `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>` and mounts `$HF_HOME` read-only into the container,
so download with the normal cache layout (not `--local-dir`):

ds41rt は `$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>` からモデルを読み、`$HF_HOME` を読み取り専用でコンテナに渡します。
`--local-dir` ではなく通常のキャッシュ形式で落としてください。

```sh
export HF_HOME=/path/to/hf          # must be the same HF_HOME when you start ds41rt
hf download ChiTako/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-ja-calibrated
```

### 2. Run ds41rt (0-spark) / ds41rt を起動する

Use a ds41rt configuration with these keys (everything else as in the shipped `ds41rt.config`):

ds41rt の設定ファイルで次の値を使います (ほかは同梱の `ds41rt.config` のままで構いません)。

```sh
MODEL_ID=ChiTako/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-ja-calibrated
MODEL_REVISION=            # empty: use refs/main of the local cache
EXPERT_FORMAT=exl3
SPARKINFER_EXL3=force
DSPARK=on
RTX_GPUS=2
SPARK_COUNT=0
RTX_EXPERT_LAYERS=40
CONCURRENCY=16
```

```sh
HF_HOME=/path/to/hf ./run.sh --config my-0spark.config
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "日本の首都はどこですか？"}],
  "max_tokens": 256}'
```

The API advertises the model as `deepseek-ai/DeepSeek-V4.1-Flash`. Upstream `run.sh` passes `--device=/dev/infiniband` to Docker;
on a single workstation without InfiniBand use the `run-local.sh` variant from the companion GitHub repository, which only drops that flag.

API 上のモデル名は `deepseek-ai/DeepSeek-V4.1-Flash` です。上流の `run.sh` は Docker に `--device=/dev/infiniband` を渡すので、
InfiniBand の無い 1 台の機械では、その指定だけを外した付属 GitHub リポジトリの `run-local.sh` を使ってください。

### 3. Numeric reasoning effort (optional, patched ds41rt) / 数値 reasoning effort (任意、改修版 ds41rt)

DeepSeek V4.1 takes a reasoning-effort score from 1 to 100 in its prompt (`Reasoning Effort: N`). Upstream ds41rt v15 accepts only level names.
The companion GitHub repository has a small patch (a vendored, patched `deepseek-recipe` 0.1.0 and a derived image
`ds41rt-coordinator:v15-effort` that replaces only the Rust `ds41rt` binary; build with `scripts/build-effort-image.sh`)
that also accepts an integer or numeric string `reasoning_effort` (1-100) and maps level names as follows:

DeepSeek V4.1 はプロンプトで 1〜100 の effort (`Reasoning Effort: N`) を受けます。上流の ds41rt v15 は段階の名前しか受けません。
付属の GitHub リポジトリに、整数か数値文字列の `reasoning_effort` (1〜100) も受ける小さな改修 (`deepseek-recipe` 0.1.0 の改修版の写しと、
Rust の `ds41rt` バイナリだけを差し替えた派生イメージ `ds41rt-coordinator:v15-effort`、`scripts/build-effort-image.sh` で作る) があり、段階の名前は次のように写します。

| `reasoning_effort` | Rendered score / プロンプトの値 (patched) | Official deepseek-recipe 0.1.0 |
|---|---|---|
| `none` / `off` / `disabled` | thinking off / 思考なし | thinking off (`none` only) |
| `minimal` / `low` | 50 | 50 |
| **`medium`** | **62 (local choice / 独自)** | 75 |
| `high` (default) | 75 | 75 |
| **`xhigh`** | **88 (local choice / 独自)** | 75 |
| `max` (Anthropic API also `ultra`) | 100 | 100 |
| integer 1-100, or `"60"` | as given / そのまま | rejected / 受けない |

**`medium` = 62 and `xhigh` = 88 are this project's own mapping, not DeepSeek's.** DeepSeek's encoding defines only
low (50), high (75) and max (100); the official `deepseek-recipe` maps the OpenAI/Anthropic names `medium` and `xhigh` to 75,
which makes `medium`, `high` and `xhigh` produce the same prompt. The patch places them between the neighbouring levels instead.
If you need the official behaviour, send integers (`75`) or use upstream ds41rt. The weights are not affected by this choice.

**`medium` = 62 と `xhigh` = 88 はこのプロジェクト独自の対応で、DeepSeek の定めたものではありません。** DeepSeek の encoding が名前で定めるのは
low (50)・high (75)・max (100) だけで、公式の `deepseek-recipe` は OpenAI / Anthropic の名前 `medium` と `xhigh` を 75 に写します。
そのため `medium`・`high`・`xhigh` が同じプロンプトになります。改修版では隣の段階の間に置きました。
公式どおりにしたいときは整数 (`75`) を送るか、上流の ds41rt を使ってください。重みはこの選択と無関係です。

### 4. ds41gw API proxy (optional) / API プロキシ ds41gw (任意)

ds41rt serves `/v1/chat/completions` only. `ds41gw` (Rust, in the companion GitHub repository) sits in front of it and adds
`/v1/responses`, `/v1/messages` (+ `count_tokens`), `/v1/completions` and `/v1/models`, using the same patched recipe for prompt rendering
and the same effort table (configurable in `config/ds41gw.toml`).

ds41rt が出すのは `/v1/chat/completions` だけです。付属 GitHub リポジトリの `ds41gw` (Rust) を前に置くと、
`/v1/responses`、`/v1/messages` (`count_tokens` 込み)、`/v1/completions`、`/v1/models` が使えます。プロンプトの描画には同じ改修版の recipe を使い、effort の表も同じです (`config/ds41gw.toml` で変えられます)。

Companion repository / 付属リポジトリ: `https://github.com/Chiytako/deepseek-v41-flash-2gpu` (proxy, ds41rt patch, quantization and calibration tools, research notes).

---

## Limitations / 制約

- **2-bit routed experts.** Even after scale calibration, KL to the original is about 0.05-0.13 per token and top-1 agreement is 88.5-92.5% (text and image sets above). Expect lower accuracy than the original model, especially on long Japanese prose (jakl32 is the worst set).
- **Small generation-level evaluation.** Generation checks used 245 image questions and 200 Japanese prompts with one seed; differences below a few points are not resolved. No standard downstream benchmarks (coding, math, agent suites) were run on this quantization.
- **DSpark / MTP draft layers are left at source precision** and were not part of the scale calibration. Speculative acceptance therefore depends on how closely the 2-bit main model follows the original.
- Calibration is weighted toward Japanese and agentic use; other languages were not targeted.
- Tested only with ds41rt v15 on 2x RTX PRO 6000 Max-Q. Not loadable by stock `transformers`; other EXL3 engines were not tested with this calibrated version.
- `model.safetensors.index.json` has `metadata.total_size = 0` (left as produced by the packing tool; ds41rt does not read it).

- **routed expert は 2bit です。** スケール校正のあとでも、原重みとの KL はトークンあたり約 0.05〜0.13、top-1 一致は 88.5〜92.5% です (上のテキストと画像の集合)。原モデルより精度は下がります。長い日本語の文章 (jakl32) がいちばん悪い集合です。
- **生成での評価は件数が少ないです。** 画像 245 問と日本語 200 本 (seed 1 つ) だけで、数ポイント未満の差は判定できません。コーディング・数学・エージェントの標準ベンチマークはこの量子化版で回していません。
- **DSpark / MTP の draft 層はソースの精度のまま**で、スケール校正の対象外です。投機の受理は 2bit の本体が原重みにどれだけ近いかに左右されます。
- 校正は日本語とエージェント用途に寄せています。ほかの言語は狙っていません。
- 確かめたのは ds41rt v15 と RTX PRO 6000 Max-Q 2 枚の組だけです。素の `transformers` では読めません。ほかの EXL3 エンジンではこの校正版を試していません。
- `model.safetensors.index.json` の `metadata.total_size` は 0 です (パック作成時のまま。ds41rt はこの値を読みません)。

## License / ライセンス

MIT, same as the base model ([LICENSE](LICENSE), copyright DeepSeek). The `encoding/` and `inference/` folders are copied unchanged from the base repository.
元モデルと同じ MIT です。`encoding/` と `inference/` は元のリポジトリからそのまま写しています。

## Acknowledgements / 謝辞

- [DeepSeek](https://huggingface.co/deepseek-ai) for DeepSeek-V4.1-Flash-1017 and its MIT license.
- [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) (MIT): the inference engine this checkpoint targets, its EXL3 0-spark path and release benchmarks.
- [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3): the EXL3 format, MCG codebook and the Viterbi encoder.
- diffbot's EXL3 2.0 bpw DeepSeek-V4.1-Flash release for 2x RTX PRO 6000, whose recipe and layout (K2 MCG, source MTP experts, official FP8 Engram) this work started from.
- [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark): the original EXL3 recipe and tooling, including the Viterbi + refit re-encoder.
- [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41): part of the calibration prompts.
- Public datasets used for calibration or evaluation only (not redistributed): wikimedia/wikipedia (ja), globis-university/aozorabunko-clean,
  llm-jp/oasst2-33k-ja, kunishou/databricks-dolly-15k-ja, JMMMU, llm-jp/MECHA-ja, wulipc/CC-OCR, SakanaAI/JA-VG-VQA-500,
  ChartQA, DocVQA, TextVQA, MMBench.
