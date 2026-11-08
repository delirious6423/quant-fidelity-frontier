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
- 24gb-vram
- 64gb-ram
---

# Qwen3.8-Flash-Next 24G64G GGUF (for a 24 GB GPU + 64 GB RAM PC) — BETA

- **24G64G = for a PC with a 24 GB GPU (VRAM) + 64 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ3_S 57%, IQ4_NL 34%, Q8_0 7%, IQ4_XS 1% (by bytes), 3.94 bits a weight on average, about IQ3_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) made for **one 24 GB GPU and
64 GB of RAM**. Every expert is re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in **exactly unsloth
UD-IQ4_XS's types and size (93.7 GB)**. All 512 experts are kept. Same size, same speed on the same emulated PC (69.9
vs 71.1 tok/s on coding tasks), and **KL 0.068 vs 0.092 (-26%)** against the NVFP4 model; KL on Japanese text 0.146
vs 0.195. Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only (below). Reports from real 24 GB GPUs and other RAM sizes are very welcome
(Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

For a 16 GB GPU: [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF).
For 12 GB + 32 GB: [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF).

日本語の説明は下にあります。

## Compared with the same-size file

| | download | resident (GPU + RAM) | KL (10 seqs) | same top token | KL, Japanese | KL, code | decode, 3 coding tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **this file** | 93.7 GB | 64.2 GB | **0.068** | **90.3%** | **0.146** | **0.092** | 69.9 tok/s |
| unsloth UD-IQ4_XS | 93.7 GB | 64.9 GB | 0.092 | 88.7% | 0.195 | 0.116 | 71.1 tok/s |

KL against the NVFP4 model (10 held-out sequences x 1,023 tokens: code, English, Chinese, Japanese and others), Abyss
engine. "Resident" leaves out the 28.8 GB per-layer embedding table, which is read a few rows a token from the SSD.
The two files have the same tensor types; only the rounding (LDLQ, calibrated with Japanese and chat turns) differs.

## Real use

Abyss server, thinking on, two seeds, on the emulated PC (below):

| | this file | unsloth UD-IQ4_XS |
| --- | --- | --- |
| Japanese (5 prompts: business email, explainer, summary, literature fact, arithmetic) | 10/10 answered, facts and sums right | 10/10 |
| 5 coding tasks with hidden tests | 5/5, 4/5 | 3/5, 5/5 |
| 8K-token prompt, find the needle | OK / OK | OK / OK |
| empty answers / prompt echoes | none | none |
| prompt processing (8K) | 1,680-1,689 tok/s | 1,670-1,683 tok/s |

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.108, a 24 GB GPU + 64 GB RAM PC emulated on an RTX 5090 (32 GB): VRAM
capped at 24 GB (`STRATA_GLM_VRAM_GB=24`), RAM tier 46 GB inside a 58 GB cgroup, 8 CPU threads, iGPU off, NVMe SSD,
MTP drafter on.

- 3 coding tasks: **69.9 tok/s** (unsloth UD-IQ4_XS 71.1)
- Japanese, 3 prompts: 61-77 tok/s
- llama.cpp, this file, `--n-cpu-moe 34` (24.2 GB of VRAM): 26-34 tok/s

Abyss v1.3.108 or later is needed for the prompt path: the Q8_0 down experts of this layout read past their slot in
older versions (also in unsloth's UD-IQ4_XS).

Other GPUs are not measured with this file. Reports welcome.

## Files

`Qwen3.8-Flash-Next-24G64G-0000N-of-00003.gguf`, 3 shards; point your engine at `-00001-of-00003.gguf`. `SHA256SUMS`
in the repo. Experts: gate/up IQ3_S (layer 2 IQ4_XS), down IQ4_NL (layers 2, 4, 30, 46, 47 Q8_0). Non-expert tensors
(attention, router, shared expert, embeddings, per-layer embedding table) are unsloth UD-IQ4_XS's bytes, unchanged.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+24G64G+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: empty answers, wrong language, loops, places where it is weaker than unsloth's file. A
  prompt that reproduces it helps most.

## How it was made

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/hessian_dump.cpp`, `tools/requant_experts.py`,
`tools/merge_experts.py`):

1. Hessians: the NVFP4 model ([magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF))
   over 1.26M tokens: its own chat turns (181 prompts: Japanese 44, Chinese 20, English 20, code 37, math 35, tool
   calls 25) followed by mixed text in the same languages and domains.
2. LDLQ: every expert re-quantized from the NVFP4 weights (per-expert scales folded in) in UD-IQ4_XS's types, then
   merged into unsloth UD-IQ4_XS.

## Running it

- **[Abyss](https://github.com/zephel01/Abyss)** v1.3.108 or later. Pack it once:
  ```sh
  cd Abyss
  .venv/bin/python tools/iq_pack.py --fast --gguf /path/to/Qwen3.8-Flash-Next-24G64G-00001-of-00003.gguf --out /path/to/pack-fast
  ```
  and start it with `--glm-pack /path/to/pack-fast` (`STRATA_GLM_RAM_GB=46` on a 64 GB PC; `STRATA_GLM_VRAM_GB=24` on
  a larger card to test the 24 GB case).
- **llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is. Checked with `llama-server -m
  Qwen3.8-Flash-Next-24G64G-00001-of-00003.gguf -ngl 99 --n-cpu-moe 34 -c 32768 --jinja -t 8` (24.2 GB of VRAM; use
  35 or more on a real 24 GB card), the same real-use prompts: Japanese 5/5 answered (none empty), coding 3/5, the 8K
  needle found; decode 26-34 tok/s.

## License

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are magnitudedev's, the
non-expert tensors unsloth's.

---

## 日本語

- **24G64G = GPU（VRAM）24 GB + RAM 64 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ3_S 57%, IQ4_NL 34%, Q8_0 7%, IQ4_XS 1%（容量の割合）、平均 3.94 ビットで、IQ3_M くらいです。

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**GPU 24 GB + RAM 64 GB の PC** 向けに作りました。
エキスパートは、NVIDIA 形式の NVFP4 の重みから LDLQ で量子化し直しています。**型も大きさ（93.7 GB）も unsloth の UD-IQ4_XS とまったく同じ**で、エキスパートは 512 個すべて残しています。
同じ設定で比べると、速さは同じ（コーディングの生成 69.9 と 71.1 tok/s）で、**元の NVFP4 のモデルとの差（KL）は 0.092 → 0.068（-26%）**、日本語の文章では 0.195 → 0.146 です。Abyss と llama.cpp で動きます。

**BETA 版です。** 測ったのは下の 1 台だけです。本物の 24 GB の GPU や、ほかの RAM 容量での結果を、Community タブか
[Abyss の issue](https://github.com/zephel01/Abyss/issues) で教えてもらえると助かります。

GPU 16 GB 向けは [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)、
GPU 12 GB + RAM 32 GB 向けは [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) です。

### 同じ大きさの版との比較

| | ダウンロード | 常駐（GPU + RAM） | KL（10 本） | 1 位の一致 | 日本語の KL | コードの KL | 生成（コーディング 3 問） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **この版** | 93.7 GB | 64.2 GB | **0.068** | **90.3%** | **0.146** | **0.092** | 69.9 tok/s |
| unsloth UD-IQ4_XS | 93.7 GB | 64.9 GB | 0.092 | 88.7% | 0.195 | 0.116 | 71.1 tok/s |

KL は NVFP4 のモデルとの差です（評価用の 10 本 × 1,023 トークン）。「常駐」には、層ごとの埋め込み表（28.8 GB、1 トークンにつき数行だけ SSD から読む）を入れていません。2 つの版はテンソルの型が同じで、違うのは丸め方（LDLQ、日本語とチャットのやりとりを入れた調整）だけです。

### 実際に使ったときの結果

Abyss のサーバー、思考あり、seed を変えて 2 回、下の設定です。

| | この版 | unsloth UD-IQ4_XS |
| --- | --- | --- |
| 日本語 5 問（ビジネスメール、解説、要約、文学の知識、計算） | 10/10 回答、事実と計算は正しい | 10/10 |
| コーディング 5 問（隠しテスト） | 5/5、4/5 | 3/5、5/5 |
| 8K トークンの文章から情報を探す | 成功 / 成功 | 成功 / 成功 |
| 空の回答 / プロンプトの繰り返し | なし | なし |
| 入力の処理（8K） | 1,680〜1,689 tok/s | 1,670〜1,683 tok/s |

### 速度

- RTX 5090（32 GB）で、GPU 24 GB + RAM 64 GB の PC を真似た設定です。VRAM を 24 GB に制限（`STRATA_GLM_VRAM_GB=24`）、RAM の枠 46 GB（全体 58 GB に制限）、CPU 8 スレッド、iGPU なし、NVMe SSD、MTP ドラフターあり。
- コーディング 3 問：**69.9 tok/s**（unsloth UD-IQ4_XS は 71.1）
- 日本語 3 問：61〜77 tok/s
- llama.cpp でこの版を `--n-cpu-moe 34`（VRAM 24.2 GB）：26〜34 tok/s
- Abyss は v1.3.108 以降が必要です。この並びの Q8_0 の down エキスパートは、それより前の版では入力の処理で読み過ぎが起きていました（unsloth の UD-IQ4_XS も同じです）。

### 中身・作り方・動かし方

- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。`SHA256SUMS` があります。
- エキスパートは gate/up が IQ3_S（層 2 だけ IQ4_XS）、down が IQ4_NL（層 2、4、30、46、47 は Q8_0）です。それ以外（アテンション、ルーター、共有エキスパート、埋め込み、層ごとの埋め込み表）は unsloth UD-IQ4_XS のものをそのまま使っています。
- 作り方は上の英語の節のとおりです（[trench-gguf](https://github.com/zephel01/trench-gguf)、NVFP4 のモデルにチャットのやりとり 181 問（日本語 44 問）と文章を流して測った 126 万トークン分の情報で LDLQ）。
- Abyss v1.3.108 以降で動作を確認しています（64 GB の PC なら `STRATA_GLM_RAM_GB=46`）。llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。`--n-cpu-moe 34` で VRAM 24.2 GB だったので、本物の 24 GB の GPU では 35 以上にしてください。同じ実用テストで、日本語 5 問すべてに回答（空の回答なし）、コーディング 3/5、8K の情報探しも成功しました。

### フィードバックのお願い（BETA 版）

- **お使いの PC での速さ**：Abyss で `./abyss.sh --bench --deep` を実行し、`abyss-bench.txt` を
  [こちらの issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+24G64G+on+YOUR_GPU)に貼ってください（タイトルの GPU 名を書き換えてください）。RAM の容量と SSD の機種も書いてもらえると助かります。
- **実際に使ったときの品質**：空の回答、言語の切り替わり、繰り返し、unsloth の版より弱いと感じたところ。再現できるプロンプトがあると一番助かります。
- ライセンスは元のモデルと同じです（[qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)）。
