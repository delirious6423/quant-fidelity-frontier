---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, qwen4exp, moe, abyss, japanese]
---

# Qwen3.8-Flash-Next 16G64G GGUF (for a 16 GB GPU + 64 GB RAM PC) — BETA

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) made for **one 16 GB GPU and
64 GB of RAM**. Every expert is re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in unsloth UD-IQ3_XXS's
layout with gate/up one step up (IQ3_XXS). All 512 experts are kept. On the same emulated PC it runs **at the speed
of unsloth's UD-IQ3_XXS (70.7 vs 71.3 tok/s on coding tasks)** with **KL 0.083 vs 0.154 (-46%)** against the NVFP4
model; KL on Japanese text 0.176 vs 0.317. The file is 4.9 GB larger (86.9 vs 82.0 GB). Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only (below). Reports from real 16 GB GPUs and other RAM sizes are very welcome
(Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

For a 24 GB GPU: [Qwen3.8-Flash-Next-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-24G64G-GGUF).
For 12 GB + 32 GB: [Qwen3.8-Flash-Next-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF).

日本語の説明は下にあります。

## Compared with files of about the same size

| | download | resident (GPU + RAM) | KL (10 seqs) | same top token | KL, Japanese | KL, code | decode, 3 coding tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **this file** | 86.9 GB | 57.6 GB | **0.083** | **89.3%** | **0.176** | **0.108** | 70.7 tok/s |
| unsloth UD-IQ3_XXS | 82.0 GB | 52.6 GB | 0.154 | 85.9% | 0.317 | 0.176 | 71.3 tok/s |
| ISTA-DASLab GSQ-RCO IQ3_S | 83.6 GB | 54.8 GB | 0.117 | 87.3% | 0.251 | 0.149 | 61.8 tok/s |
| unsloth UD-IQ4_XS (24 GB tier) | 93.7 GB | 64.9 GB | 0.092 | 88.7% | 0.195 | 0.116 | — |

KL against the NVFP4 model (10 held-out sequences x 1,023 tokens: code, English, Chinese, Japanese and others), Abyss
engine. "Resident" leaves out the 28.8 GB per-layer embedding table, which is read a few rows a token from the SSD.

## Real use

Abyss server, thinking on, two seeds, on the emulated PC (below):

| | this file | unsloth UD-IQ3_XXS |
| --- | --- | --- |
| Japanese (5 prompts: business email, explainer, summary, literature fact, arithmetic) | 10/10 answered, facts and sums right | 10/10 |
| 5 coding tasks with hidden tests | 4/5, 3/5 | 5/5, 4/5 |
| 8K-token prompt, find the needle | OK / OK | OK / OK |
| empty answers / prompt echoes | none | none |
| prompt processing (8K) | 1,618-1,627 tok/s | 1,729-1,747 tok/s |

Coding is weaker than unsloth's here (7 vs 9 of 10). In the misses the model thought until the 6,144-token limit on
two tasks (a duration parser and a hiragana/katakana converter) that make every Qwen3.8 build overthink in some runs.
Ten runs are too few to call it, but it is what we measured; give it more `max_tokens` on such tasks.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.108, a 16 GB GPU + 64 GB RAM PC emulated on an RTX 5090 (32 GB): VRAM
capped at 16 GB (`STRATA_GLM_VRAM_GB=16`), RAM tier 46 GB inside a 58 GB cgroup, 8 CPU threads, iGPU off, NVMe SSD,
MTP drafter on. Every expert fits in VRAM + RAM (no SSD reads for experts).

- 3 coding tasks: **70.7 tok/s** (unsloth UD-IQ3_XXS 71.3, ISTA IQ3_S 61.8)
- Japanese, 3 prompts: 67-76 tok/s
- llama.cpp, this file, `--n-cpu-moe 39` (16.4 GB of VRAM): 29-34 tok/s

Other GPUs are not measured with this file. Reports welcome.

## Files

`Qwen3.8-Flash-Next-16G64G-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`. `SHA256SUMS`
in the repo.

Experts: gate/up IQ3_XXS (layer 2 IQ3_S), down IQ4_NL. The per-layer embedding table is 28.8 GB (IQ4_NL).
Non-expert tensors (attention, router, shared expert, embeddings, per-layer embedding table) are unsloth
UD-IQ3_XXS's bytes, unchanged.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+16G64G+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: empty answers, wrong language, loops, places where it is weaker than unsloth's file. A
  prompt that reproduces it helps most.

## How it was made

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/hessian_dump.cpp`, `tools/requant_experts.py`,
`tools/merge_experts.py`):

1. Hessians: the NVFP4 model ([magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF))
   over 1.26M tokens: its own chat turns (181 prompts: Japanese 44, Chinese 20, English 20, code 37, math 35, tool
   calls 25) followed by mixed text in the same languages and domains.
2. LDLQ: every expert re-quantized from the NVFP4 weights (per-expert scales folded in), gate/up IQ3_XXS (layer 2
   IQ3_S), down IQ4_NL, then merged into unsloth UD-IQ3_XXS.

## Running it

- **[Abyss](https://github.com/zephel01/Abyss)** v1.3.108 or later. Pack it once:
  ```sh
  cd Abyss
  .venv/bin/python tools/iq_pack.py --fast --gguf /path/to/Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf --out /path/to/pack-fast
  ```
  and start it with `--glm-pack /path/to/pack-fast` (`STRATA_GLM_RAM_GB=46` on a 64 GB PC; `STRATA_GLM_VRAM_GB=16` on
  a larger card to test the 16 GB case).
- **llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is. Checked with `llama-server -m
  Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf -ngl 99 --n-cpu-moe 39 -c 32768 --jinja -t 8` (16.4 GB of VRAM; use
  40 or more on a real 16 GB card), the same real-use prompts: Japanese 5/5 answered (none empty), coding 3/5, the 8K
  needle found; decode 29-34 tok/s.

## License

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are magnitudedev's, the
non-expert tensors unsloth's.

---

## 日本語

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**GPU 16 GB + RAM 64 GB の PC** 向けに作りました。
エキスパートは、NVIDIA 形式の NVFP4 の重みから LDLQ で量子化し直しています。並びは unsloth の UD-IQ3_XXS と同じで、gate/up だけ 1 段上の IQ3_XXS にしました。エキスパートは 512 個すべて残しています。
同じ設定で unsloth の UD-IQ3_XXS と比べると、**速さは同じ（コーディングの生成 70.7 と 71.3 tok/s）で、元の NVFP4 のモデルとの差（KL）は 0.154 → 0.083（-46%）**、日本語の文章では 0.317 → 0.176 です。ファイルは 4.9 GB 大きくなります（86.9 と 82.0 GB）。Abyss と llama.cpp で動きます。

**BETA 版です。** 測ったのは下の 1 台だけです。本物の 16 GB の GPU や、ほかの RAM 容量での結果を、Community タブか
[Abyss の issue](https://github.com/zephel01/Abyss/issues) で教えてもらえると助かります。

GPU 24 GB 向けは [Qwen3.8-Flash-Next-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-24G64G-GGUF)、
GPU 12 GB + RAM 32 GB 向けは [Qwen3.8-Flash-Next-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF) です。

### 同じくらいの大きさの版との比較

| | ダウンロード | 常駐（GPU + RAM） | KL（10 本） | 1 位の一致 | 日本語の KL | コードの KL | 生成（コーディング 3 問） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **この版** | 86.9 GB | 57.6 GB | **0.083** | **89.3%** | **0.176** | **0.108** | 70.7 tok/s |
| unsloth UD-IQ3_XXS | 82.0 GB | 52.6 GB | 0.154 | 85.9% | 0.317 | 0.176 | 71.3 tok/s |
| ISTA-DASLab GSQ-RCO IQ3_S | 83.6 GB | 54.8 GB | 0.117 | 87.3% | 0.251 | 0.149 | 61.8 tok/s |
| unsloth UD-IQ4_XS（24 GB 向け） | 93.7 GB | 64.9 GB | 0.092 | 88.7% | 0.195 | 0.116 | — |

KL は NVFP4 のモデルとの差です（評価用の 10 本 × 1,023 トークン）。「常駐」には、層ごとの埋め込み表（28.8 GB、1 トークンにつき数行だけ SSD から読む）を入れていません。

### 実際に使ったときの結果

Abyss のサーバー、思考あり、seed を変えて 2 回、下の設定です。

| | この版 | unsloth UD-IQ3_XXS |
| --- | --- | --- |
| 日本語 5 問（ビジネスメール、解説、要約、文学の知識、計算） | 10/10 回答、事実と計算は正しい | 10/10 |
| コーディング 5 問（隠しテスト） | 4/5、3/5 | 5/5、4/5 |
| 8K トークンの文章から情報を探す | 成功 / 成功 | 成功 / 成功 |
| 空の回答 / プロンプトの繰り返し | なし | なし |
| 入力の処理（8K） | 1,618〜1,627 tok/s | 1,729〜1,747 tok/s |

コーディングは unsloth の版より弱い結果でした（10 回中 7 回と 9 回）。落とした回は、どの Qwen3.8 の版でも考えすぎやすい 2 問（時間の文字列の変換、ひらがな・カタカナの変換）で、6,144 トークンの上限まで考え続けていました。10 回では偶然かどうか言い切れませんが、測ったとおりに載せています。こうした問題では `max_tokens` を増やしてください。

### 速度

- RTX 5090（32 GB）で、GPU 16 GB + RAM 64 GB の PC を真似た設定です。VRAM を 16 GB に制限（`STRATA_GLM_VRAM_GB=16`）、RAM の枠 46 GB（全体 58 GB に制限）、CPU 8 スレッド、iGPU なし、NVMe SSD、MTP ドラフターあり。エキスパートはすべて VRAM と RAM に載り、SSD からは読みません。
- コーディング 3 問：**70.7 tok/s**（unsloth UD-IQ3_XXS は 71.3、ISTA IQ3_S は 61.8）
- 日本語 3 問：67〜76 tok/s
- llama.cpp でこの版を `--n-cpu-moe 39`（VRAM 16.4 GB）：29〜34 tok/s

### 中身・作り方・動かし方

- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。`SHA256SUMS` があります。
- エキスパートは gate/up が IQ3_XXS（層 2 だけ IQ3_S）、down が IQ4_NL です。それ以外（アテンション、ルーター、共有エキスパート、埋め込み、層ごとの埋め込み表）は unsloth UD-IQ3_XXS のものをそのまま使っています。
- 作り方は上の英語の節のとおりです（[trench-gguf](https://github.com/zephel01/trench-gguf)、NVFP4 のモデルにチャットのやりとり 181 問（日本語 44 問）と文章を流して測った 126 万トークン分の情報で LDLQ）。
- Abyss v1.3.108 以降で動作を確認しています（64 GB の PC なら `STRATA_GLM_RAM_GB=46`）。llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。`--n-cpu-moe 39` で VRAM 16.4 GB だったので、本物の 16 GB の GPU では 40 以上にしてください。同じ実用テストで、日本語 5 問すべてに回答（空の回答なし）、コーディング 3/5、8K の情報探しも成功しました。

### フィードバックのお願い（BETA 版）

- **お使いの PC での速さ**：Abyss で `./abyss.sh --bench --deep` を実行し、`abyss-bench.txt` を
  [こちらの issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+16G64G+on+YOUR_GPU)に貼ってください（タイトルの GPU 名を書き換えてください）。RAM の容量と SSD の機種も書いてもらえると助かります。
- **実際に使ったときの品質**：空の回答、言語の切り替わり、繰り返し、unsloth の版より弱いと感じたところ。再現できるプロンプトがあると一番助かります。
- ライセンスは元のモデルと同じです（[qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)）。
