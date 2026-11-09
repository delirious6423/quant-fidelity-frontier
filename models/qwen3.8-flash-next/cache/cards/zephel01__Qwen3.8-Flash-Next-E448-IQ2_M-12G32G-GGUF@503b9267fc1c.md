---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- qwen4exp
- moe
- abyss
- japanese
- consumer-gpu
- local-llm
- cpu-offload
- rtx-5090
- 12gb-vram
- 32gb-ram
language:
- en
- ja
library_name: gguf
---

# Qwen3.8-Flash-Next E448-IQ2_M GGUF (for a 12 GB GPU + 32 GB RAM PC) — BETA

- **12G32G = for a PC with a 12 GB GPU (VRAM) + 32 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ4_NL 57%, IQ1_S 28%, IQ2_XXS 15% (by bytes), 2.64 bits a weight on average, about IQ2_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) made for **one 12 GB GPU and
32 GB of RAM**. Every expert is re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in unsloth UD-IQ1_S's own
layout, and each layer keeps 448 of its 512 experts. Compared with unsloth's UD-IQ1_S on the same small PC, it is
**30% faster (56.0 vs 43.0 tok/s on coding tasks)**, 5 GB smaller (67.5 vs 72.5 GB), and its Japanese KL is lower.
Japanese is part of the calibration, the expert selection and the tests. Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only (below). Reports from other GPUs / RAM sizes are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

日本語の説明は下にあります。

## Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 12 GB GPU + 32 GB RAM | 67.5 GB | 448 of 512 experts kept |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 86.9 GB | all 512 experts |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | 24 GB GPU + 64 GB RAM | 93.7 GB | all 512 experts, higher quality |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark (128 GB) | 86.9 GB | same files as 16G64G |

Earlier repo name: Qwen3.8-Flash-Next-12G32G-GGUF (old links redirect here). The file / folder names keep the earlier label L448.

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## Which file

| | experts per layer | download | KL (10 seqs) | same top token | KL, Japanese seq | decode, 3 coding tasks (12 GB + 32 GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`L448/`](./L448) (here)** | **448 / 512** | **67.5 GB** | 0.339 | 78.5% | **0.553** | **56.0 tok/s** |
| unsloth UD-IQ1_S (reference) | 512 / 512 | 72.5 GB | 0.331 | 78.4% | 0.568 | 43.0 tok/s |

KL against the NVFP4 model (10 held-out sequences x 1,023 tokens: code, English, Chinese, Japanese and others), Abyss
engine. Overall KL is about unsloth's, with 64 fewer experts per layer.

## Real use

Abyss server, temperature 1.0 / top_p 0.95 / top_k 20, thinking on, two seeds, on the emulated small PC (below):

| | L448 (here) | unsloth UD-IQ1_S |
| --- | --- | --- |
| Japanese (5 prompts: business email, explainer, summary, literature fact, arithmetic) | 10/10 answered, facts and sums right | 10/10 |
| "Japan's three great essays" (a known overthinking case) | 196 / 165 tokens, right | 454 / 384 tokens, right |
| 5 coding tasks with hidden tests | 3/5, 3/5 | 5/5, 3/5 |
| 8K-token prompt, find the needle | OK / OK | OK |
| prompt processing (8K) | 1,822 tok/s | 1,368 tok/s |
| Japanese decode | 48-51 tok/s | 39-46 tok/s |

One coding task (a hiragana/katakana converter) makes this model think until the token limit in most runs, in every
build (unsloth's solved it once in two). The model's own habit; give it more `max_tokens` or less thinking there.

**Empty answers**: the first build of this file (calibrated on plain text only) sometimes ended the turn right after
its thinking, with no answer. This version adds the model's own chat turns (thinking, `</think>`, answer; 181 prompts,
44 in Japanese) to the calibration, and the empty answers and the long overthinking were gone in our tests.
Abyss v1.3.106 also guards against it (`STRATA_GLM_END_GUARD`, on by default).

## Speed

[Abyss](https://github.com/zephel01/Abyss), a 12 GB GPU + 32 GB RAM PC emulated on an RTX 5090 (32 GB): VRAM capped at
12 GB (`STRATA_GLM_VRAM_GB=12`), RAM tier 20 GB inside a 28 GB cgroup, 8 CPU threads for the CPU lane, iGPU off, NVMe
SSD, no drafter. 3 coding tasks (17K tokens): **56.0 tok/s** (unsloth UD-IQ1_S: 43.0; llama.cpp with UD-IQ1_S at
`--n-cpu-moe 41`, about 12 GB of VRAM: 27.6).

Other GPUs are not measured with this file. Reports welcome.

## Files

`L448/Qwen3.8-Flash-Next-12G32G-L448-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`.

| | experts | per-layer embedding table | the rest | total |
| --- | ---: | ---: | ---: | ---: |
| L448 | 34.9 GB (gate/up IQ1_S / IQ2_XXS, down IQ4_NL) | 28.8 GB (IQ4_NL) | 3.9 GB | 67.5 GB |

- The per-layer embedding table is read a few rows a token; it does not have to fit in RAM (Abyss reads it from the
  SSD, llama.cpp maps it).
- Non-expert tensors (attention, router, shared expert, embeddings, PLE table) are unsloth UD-IQ1_S's bytes, unchanged.
- `SHA256SUMS` in the folder.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+12G32G-L448+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: empty answers, wrong language, loops, places where it is weaker than unsloth's UD-IQ1_S. A
  prompt that reproduces it helps most.
- **What you would rather have**: a 64 GB RAM version, more code weight, other GPUs.

## How it was made

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/hessian_dump.cpp`, `tools/requant_experts.py`,
`tools/merge_experts.py`, `tools/prune_experts.py`) and Abyss's routing counter (`STRATA_GLM_USAGE`):

1. Hessians: the NVFP4 model ([magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF))
   over 1.26M tokens: its own chat turns (181 prompts: Japanese 44, Chinese 20, English 20, code 37, math 35, tool
   calls 25) followed by mixed text in the same languages and domains.
2. LDLQ: every expert re-quantized from the NVFP4 weights (per-expert scales folded in), in UD-IQ1_S's layout
   (gate/up IQ1_S, or IQ2_XXS in 14 layers; down IQ4_NL), then merged into UD-IQ1_S.
3. Selection: routing counts over Japanese, Chinese, English, code, math and tool-call text; per layer the 448 most
   used experts, every domain weighted the same and code twice, checked on held-out text (kept routes: Japanese
   98.6%, Chinese 98.1%, English 98.3%, code 96.9%, math 98.2%, tools 99.1%).
4. `prune_experts.py` cuts the experts and router rows.

## Running it

- **[Abyss](https://github.com/zephel01/Abyss)** v1.3.106 or later. Pack it once:
  ```sh
  cd Abyss
  .venv/bin/python tools/iq_pack.py --fast --gguf /path/to/L448/Qwen3.8-Flash-Next-12G32G-L448-00001-of-00003.gguf --out /path/to/L448/pack
  ```
  and start it with `--glm-pack /path/to/L448/pack` (`STRATA_GLM_VRAM_GB=12` on a larger card to test the 12 GB case).
- **llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is (the expert count is read from the
  file). Checked with `llama-server -m Qwen3.8-Flash-Next-12G32G-L448-00001-of-00003.gguf -ngl 99 --n-cpu-moe 41
  -c 32768 --jinja -t 8` (10.9 GB of VRAM, 28 GB host limit), the same real-use prompts: Japanese 5/5 answered (none
  empty), coding 3/5, the 8K needle found; decode 26-32 tok/s, prompt processing 88 tok/s.

## License

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)). All credit for the model goes to the Qwen team; the
NVFP4 weights the experts were made from are magnitudedev's, the non-expert tensors unsloth's.

---

## 日本語

- **12G32G = GPU（VRAM）12 GB + RAM 32 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ4_NL 57%, IQ1_S 28%, IQ2_XXS 15%（容量の割合）、平均 2.64 ビットで、IQ2_M くらいです。

### サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 12 GB + RAM 32 GB | 67.5 GB | エキスパート 512 個中 448 個 |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 86.9 GB | 512 個すべて |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | GPU 24 GB + RAM 64 GB | 93.7 GB | 512 個すべて、品質重視 |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark（128 GB） | 86.9 GB | 16G64G と同じファイル |

以前のリポジトリ名は Qwen3.8-Flash-Next-12G32G-GGUF です（古い URL はここへ転送されます）。 ファイル名やフォルダ名には、以前の呼び名 L448 が残っています。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。


[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**GPU 12 GB + RAM 32 GB の PC** 向けに作りました。
エキスパートは、NVIDIA 形式の NVFP4 の重みから LDLQ で量子化し直しています（並びは unsloth の UD-IQ1_S と同じ）。各層のエキスパートは 512 個のうち 448 個を残しています。
同じ小さな PC で unsloth の UD-IQ1_S と比べると、コーディングの生成が **約 3 割速く（56.0 と 43.0 tok/s）**、ファイルが 5 GB 小さく（67.5 と 72.5 GB）、日本語の KL も小さくなりました。
日本語は、量子化の調整、エキスパートの選び方、テストのすべてに入れています。Abyss と llama.cpp で動きます。

**BETA 版です。** 測ったのは下の 1 台だけです。ほかの GPU や RAM 容量での結果を、Community タブか
[Abyss の issue](https://github.com/zephel01/Abyss/issues) で教えてもらえると助かります。

### どれを選ぶか

| | 1 層のエキスパート | ダウンロード | KL（10 本） | 1 位の一致 | 日本語の KL | 生成速度（コーディング 3 問、12 GB + 32 GB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`L448/`（ここ）** | **448 / 512** | **67.5 GB** | 0.339 | 78.5% | **0.553** | **56.0 tok/s** |
| unsloth UD-IQ1_S（参考） | 512 / 512 | 72.5 GB | 0.331 | 78.4% | 0.568 | 43.0 tok/s |

KL は NVFP4 のモデルとの差です（評価用の 10 本 × 1,023 トークン。コード、英語、中国語、日本語など）。エキスパートを各層 64 個減らしても、全体の KL は unsloth とほぼ同じです。

### 実際に使ったときの結果

Abyss のサーバー、temperature 1.0 / top_p 0.95 / top_k 20、思考あり、seed を変えて 2 回、下の「小さな PC」の設定です。

| | L448（ここ） | unsloth UD-IQ1_S |
| --- | --- | --- |
| 日本語 5 問（ビジネスメール、解説、要約、文学の知識、計算） | 10/10 回答、事実と計算は正しい | 10/10 |
| 「日本の三大随筆」（考えすぎやすい問題） | 196 / 165 トークンで正解 | 454 / 384 トークンで正解 |
| コーディング 5 問（隠しテスト） | 3/5、3/5 | 5/5、3/5 |
| 8K トークンの文章から情報を探す | 成功 / 成功 | 成功 |
| 入力の処理（8K） | 1,822 tok/s | 1,368 tok/s |
| 日本語の生成 | 48〜51 tok/s | 39〜46 tok/s |

- コーディングの 1 問（ひらがな・カタカナの変換）は、どの版でもたいてい上限まで考え続けます（unsloth の版は 2 回中 1 回解けました）。モデル自体のくせです。こうした問題では `max_tokens` を増やすか、思考を短くしてください。
- **空の回答について**：このファイルの最初の版（ふつうの文章だけで調整）は、考えたあと回答を書かずにターンを終えることがありました。この版は、モデル自身のチャットのやりとり（思考、`</think>`、回答。181 問、うち日本語 44 問）を調整に加え、テストでは空の回答と長い考えすぎが出なくなりました。Abyss v1.3.106 でも、空の回答を防ぐ仕組み（`STRATA_GLM_END_GUARD`、既定でオン）を入れています。

### 速度

- RTX 5090（32 GB）で、GPU 12 GB + RAM 32 GB の PC を真似た設定です。VRAM を 12 GB に制限（`STRATA_GLM_VRAM_GB=12`）、RAM の枠 20 GB（全体 28 GB に制限）、CPU 8 スレッド、iGPU なし、NVMe SSD、ドラフターなし。
- コーディング 3 問（1.7 万トークン）：**56.0 tok/s**（unsloth UD-IQ1_S は 43.0。llama.cpp で UD-IQ1_S を `--n-cpu-moe 41`、VRAM 約 12 GB にすると 27.6）。
- ほかの GPU ではまだ測っていません。結果を教えてもらえると助かります。

### 中身

- `L448/` は 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。
- 合計 67.5 GB：エキスパート 34.9 GB（gate/up は IQ1_S、14 層だけ IQ2_XXS。down は IQ4_NL）、層ごとの埋め込み表 28.8 GB（IQ4_NL）、そのほか 3.9 GB。
- 層ごとの埋め込み表は、1 トークンにつき数行しか読まないので、RAM に載せる必要はありません（Abyss は SSD から読み、llama.cpp はファイルを map します）。
- エキスパート以外（アテンション、ルーター、共有エキスパート、埋め込み、PLE 表）は、unsloth UD-IQ1_S のものをそのまま使っています。

### フィードバックのお願い（BETA 版）

- **お使いの PC での速さ**：Abyss で `./abyss.sh --bench --deep` を実行し、`abyss-bench.txt` を
  [こちらの issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+12G32G-L448+on+YOUR_GPU)に貼ってください（タイトルの GPU 名を書き換えてください）。RAM の容量と SSD の機種も書いてもらえると助かります。
- **実際に使ったときの品質**：空の回答、言語の切り替わり、繰り返し、unsloth の UD-IQ1_S より弱いと感じたところ。再現できるプロンプトがあると一番助かります。
- **欲しい版**：RAM 64 GB 向け、コード寄り、ほかの GPU 向けなど。

### 作り方・動かし方・ライセンス

- 作り方は上の英語の節のとおりです（[trench-gguf](https://github.com/zephel01/trench-gguf) の `hessian_dump.cpp`、`requant_experts.py`、`merge_experts.py`、`prune_experts.py`、Abyss のルート集計）。残したエキスパートが受け持つルートの割合は、日本語 98.6%、中国語 98.1%、英語 98.3%、コード 96.9%、数学 98.2%、ツール呼び出し 99.1% です。
- Abyss v1.3.106 以降で動作を確認しています。llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。`--n-cpu-moe 41`（VRAM 10.9 GB）で同じ実用テストをすると、日本語 5 問すべてに回答（空の回答なし）、コーディング 3/5、8K の文章から情報を探すのにも成功しました。生成は 26〜32 tok/s、入力の処理は 88 tok/s です。
- ライセンスは元のモデルと同じです（[qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)）。
