---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- deepseek41
- moe
- abyss
- consumer-gpu
- local-llm
- cpu-offload
- japanese
- rtx-5090
- 32gb-vram
- 96gb-ram
- 12gb-vram
- 128gb-ram
---

# DeepSeek-V4.1-Flash-1017 Compact GGUF (for a 32 GB GPU + 96 GB RAM PC) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- `132G`: a mixed-type file - the experts are IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11% (by bytes), 1.95 bits a weight on average, about IQ2_XXS.

A **general-purpose pruned build** of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT) - **not code-only**: 256 of 384 experts per layer, chosen by measured routing on chat, math, tool-call, code and
multilingual text (kept routes: math 94%, tools 94%, English 91%, code 84%, Chinese 94%, Japanese 95%). 207 GB instead
of the 276 GB of our unpruned [LDLQ 132G](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF),
and faster on **one 32 GB GPU plus a 96 GB RAM PC** (87 vs 67.5 tok/s on coding tasks). Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only (below). Reports from other GPUs / RAM sizes are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

日本語の説明は下にあります。

## Why we made it: a code-only selection drops other abilities

[ISTA-DASLab's GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) (192 of
384 experts, 182.7 GB) is small and fast, but in our tests it answered all three Japanese prompts in Chinese: its
experts were selected on code, agentic, long-context and vision data, and our routing counts show that a code-only
selection of 192 experts keeps only 45% of the routes Japanese text takes (code: 82%) - whatever the selection data
leaves out is what goes first. **JA256** selects its experts on Japanese, Chinese, English, code, math and tool-call
text; at 256 experts it keeps 95% of the Japanese routes and
84% of the code routes (about the Coder's 82%). (ISTA's unpruned GSQ-RCO releases keep all experts; this is about the
Coder only.)

## Which one — this repository and its neighbours

| | experts per layer | download | KL (8 seqs) | same top token | decode, 3 coding tasks | Japanese |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **[`JA256/`](./JA256) (here)** | **256 / 384** | **207.1 GB** | 0.383 | 75.2% | **87.0 tok/s** | OK |
| `Full/` (here, planned) | 384 / 384 | ~230 GB | - | - | - | - |
| [LDLQ `132G`](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | 384 / 384 | 275.8 GB | 0.194 | 81.1% | 67.5 tok/s | OK |
| [LDLQ `139G`](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | 384 / 384 | 283.0 GB | **0.182** | 81.6% | 57.9 tok/s | OK |
| ISTA GSQ-RCO Coder (reference) | 192 / 384 | 182.7 GB | 0.582 | 69.3% | 97.0 tok/s | **Chinese** |
| ISTA GSQ-RCO IQ2_XXS (reference) | 384 / 384 | ~249 GB | 0.314 | 76.8% | - | - |

How to choose:

- **JA256** — smaller download and the fastest file that keeps Japanese working: everyday chat, Japanese / Chinese /
  English, coding, math and tool calls. It is ~30% faster than 132G on our PC because its experts (88 GB) almost fit
  in GPU + RAM. Its KL is higher (about the same as ISTA's unpruned IQ1_M, 0.385), so long, careful work drifts more.
- **LDLQ 132G / 139G** — no expert removed, closest to the original. Pick these for hard coding work, long reasoning,
  other languages (Korean, European languages, ...), rare knowledge, or image input, and when 276-283 GB is fine.
- **Full** (coming) — every expert kept at ~230 GB, between the two.
- **ISTA Coder** — English coding only, if every GB and token per second counts; it does not answer Japanese.

What JA256 gives up: 128 of 384 experts per layer are gone. Text in the six domains above loses 5-16% of its routes
to the next-best experts; anything outside them (other languages, rare domains, image input - not in our counts) may
lose more and is weaker. The next version will follow your reports (e.g. weighting code higher).

## Files (JA256)

`JA256/DeepSeek-V4.1-Flash-1017-JA256-0000N-of-00006.gguf`, 6 shards; point your engine at `-00001-of-00006.gguf`.

| | shards 1-5 (dense + experts) | shard 6 (Engram table) | total |
| --- | ---: | ---: | ---: |
| JA256 | 96.5 GB (experts 88.4 GB) | 110.6 GB (IQ4_NL) | 207.1 GB |

- The experts, routers and router biases are LDLQ 132G's, sliced byte for byte to the 256 kept experts per layer
  (`expert_count` = 256; nothing re-quantized). Which experts are kept differs per layer.
- The Engram table was re-quantized to IQ4_NL from the FP8 table in unsloth's UD-Q4_K_XL (relative RMS error 0.075
  vs ISTA's IQ4_NL table 0.076 on the same rows). It is read from the SSD a few rows at a time and never has to fit in RAM.
- `SHA256SUMS` in the folder.

## Quality

KL against the unquantized model (top-256 log-probs of DeepSeek-V4.1-Flash-1017, 8 held-out sequences x 1,023
tokens), Abyss engine, exact routing: see the table above (per-sequence top-1 agreement of JA256: 72-79%).

KL is a guide only; real work (Abyss server, everyday settings, temperature 1.0 / top_p 0.95):

| | JA256 | LDLQ 132G | ISTA Coder |
| --- | --- | --- | --- |
| 10 hard coding tasks (hidden tests), solved | 8/10 (t104 solved in 2 of 2 reruns; t046 failed 3 of 3) | 10/10 | 9/10 (t046 failed) |
| quality score of the solved tasks (graded) | 80.5 | 77.6 | 77.5 |
| 8K / 32K prompt, find the needle | OK / OK | OK / OK | OK / OK |
| 7-turn tool-call coding session, tests pass | yes, final report in Japanese | yes | yes, final report in Chinese |
| Japanese (3 prompts), no mixed scripts | OK | OK | **Chinese** (no kana, one loop) |
| loops / illegal memory accesses | 0 / 0 | 0 / 0 | 1 / 0 |

The one task JA256 keeps failing (t046, object lifetimes in a dependency-injection container) is also the one the
Coder fails: a sign of what pruning costs on hard code. If coding is your main use, take 132G / 139G.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.101, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395
(LPDDR5X, Radeon 8060S iGPU computes part of the RAM-side experts), RAM tier capped at 60 GB, NVMe SSD. Everyday
settings: DSpark drafter (Lucebox Q2K-Q4K), `STRATA_GLM_TAIL_SKIP=4`, `STRATA_GLM_TAIL_SKIP_DISK=1`,
`STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`.

| decode tok/s | JA256 | LDLQ 132G | ISTA Coder |
| --- | ---: | ---: | ---: |
| 3 coding tasks | **87.0** | 67.5 | 97.0 |
| 10 hard tasks (mean) | **75.7** | 58.0 | 87.1 |
| 32K prompt: prompt processing | **1,833** | 865 | 1,509 |
| 32K prompt: decode after it | 63.1 | 43.4 | 76.8 |

JA256's experts (88 GB) nearly fit in VRAM + the 60 GB RAM tier: 0.1 SSD reads a token against ~1 for 132G.

Other GPUs are not measured with this file. With 64 GB RAM, JA256 should lose less than the 384-expert files, since
its experts are 44 GB smaller - reports welcome.

### A 12 GB GPU + 128 GB RAM PC (12G128G, measured 2026-11-08)

Abyss v1.3.120 (+ the IQ1_S prompt fix of the next release for ISTA), the same PC set up as a 12 GB card
(`STRATA_GLM_VRAM_GB=12`) with no iGPU (`STRATA_GLM_IGPU=0`), the RAM tier at 90 GB (80 for ISTA), exact settings only
(no tail skip, no route affinity), `./abyss.sh --bench`:

| | experts | size (without Engram) | KL (8 seqs) | decode | 8K prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| **JA256, no drafter** | 256 | ~97 GB | **0.383** | **27.4 tok/s** | **375 tok/s** |
| JA256, DSpark drafter | 256 | ~97 GB | 0.383 | 29.5 tok/s | 26 tok/s |
| ISTA GSQ-RCO IQ1_M, no drafter | 384 | ~121 GB | 0.384 | 26.7 tok/s | 234 tok/s |

- **On a 12 GB card, leave the DSpark drafter off**: with it, the dense weights and the drafter leave the expert pool
  4 slots a layer (a route needs 6), and the prompt runs token by token.
- JA256's experts fit the 90 GB RAM tier: no SSD reads while it writes. ISTA's IQ1_M has the same KL at 24 GB more.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+Compact-JA256+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language, looped or got weaker than 132G - especially
  coding and languages outside the six domains. A prompt that reproduces it helps most.
- **What you would rather have**: more code weight, a smaller one for 64 GB RAM, other GPUs.

## How it was made

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/prune_experts.py`, `tools/requant_table.py`) and
Abyss's routing counter (`STRATA_GLM_USAGE`):

1. Routing counts: LDLQ 132G (all 384 experts) run over ~45K tokens each of Japanese, Chinese, English, code, math
   and tool-call text (eaddario/imatrix-calibration, micro sets) plus Japanese answers, counting which experts each
   layer routes to.
2. Selection: per layer the 256 experts with the most routes, every domain weighted the same, chosen on 2/3 of the
   text and checked on the other 1/3 (kept routes: ja 94.8%, zh 93.6%, en 90.6%, code 83.6%, math 93.8%, tools 93.5%).
3. Slicing: `prune_experts.py` cuts the experts, router rows and biases of LDLQ 132G.
4. Engram: `requant_table.py` re-quantizes the FP8 table of unsloth UD-Q4_K_XL to IQ4_NL on the GPU (11 min).

## Running it

- **[Abyss](https://github.com/zephel01/Abyss)** v1.3.101 or later. Pack it once:
  ```sh
  cd Abyss
  .venv/bin/python tools/iq_pack.py --compat-bf16 --gguf /path/to/JA256/DeepSeek-V4.1-Flash-1017-JA256-00001-of-00006.gguf --out /path/to/JA256/pack
  ```
  and use `docs/deepseek41/README.md`'s config with `--glm-pack /path/to/JA256/pack`.
- **llama.cpp** (arch `deepseek41`, upstream `ac8e1694d` of 2026-10-29): runs `JA256` as is (the expert count is read
  from the file). Checked with `llama-server -m DeepSeek-V4.1-Flash-1017-JA256-00001-of-00006.gguf -ngl 99
  --n-cpu-moe 36 -c 4096 --jinja` on the RTX 5090 (32 GB): correct answers to a Japanese question and a Python task,
  15-20 tok/s (132G: 10-11 tok/s at `--n-cpu-moe 38`).

## License

MIT, as the original model. All credit for the model goes to DeepSeek; the non-expert tensors come unchanged from
[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).

---

## 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- `132G`: 形式を混ぜたファイルです。エキスパートは IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11%（容量の割合）、平均 1.95 ビットで、IQ2_XXS くらいです。

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)（MIT）の GGUF を、
自作の [LDLQ 132G / 139G](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) より小さくして置いています。
**GPU 32 GB + RAM 96 GB の PC** 向けで、276 GB のダウンロードは重いという方のための版です。

**BETA 版です。** 測ったのは下の 1 台だけです。ほかの GPU や RAM 容量での結果を、Community タブか
[Abyss の issue](https://github.com/zephel01/Abyss/issues) で教えてもらえると助かります。

### 作った理由

[ISTA-DASLab の GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF)
（384 個中 192 個、182.7 GB）は小さくて速いのですが、手元のテストでは日本語の質問 3 本すべてに中国語で答えました。
エキスパートをコード・エージェント・長文・画像のデータで選んでいるためです。こちらの集計でも、コードだけで 192 個を選ぶと、
日本語のルートは 45% しか残りません（コードは 82%）。
**JA256** は日本語・中国語・英語・コード・数学・ツール呼び出しの文章で選び、256 個で日本語のルートの 95%、コードの 84%
（Coder とほぼ同じ）を残しています。（ISTA のエキスパートを削っていない GSQ-RCO 版は全部残しているので、この話は Coder 版だけのことです。）

### 使い分け

| | 1 層のエキスパート | ダウンロード | KL（8 本） | 1 位の一致 | 生成速度（コーディング 3 問） | 日本語 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **`JA256/`（ここ）** | **256 / 384** | **207.1 GB** | 0.383 | 75.2% | **87.0 tok/s** | OK |
| `Full/`（ここ、予定） | 384 / 384 | 約 230 GB | - | - | - | - |
| LDLQ `132G` | 384 / 384 | 275.8 GB | 0.194 | 81.1% | 67.5 tok/s | OK |
| LDLQ `139G` | 384 / 384 | 283.0 GB | **0.182** | 81.6% | 57.9 tok/s | OK |
| ISTA GSQ-RCO Coder（参考） | 192 / 384 | 182.7 GB | 0.582 | 69.3% | 97.0 tok/s | **中国語** |
| ISTA GSQ-RCO IQ2_XXS（参考） | 384 / 384 | 約 249 GB | 0.314 | 76.8% | - | - |

- **JA256 がおすすめの人**：ダウンロードを小さくしたい、速さがほしい、日本語も使う。
  - 普段の会話、日本語・中国語・英語、コーディング、数学、ツール呼び出しに向きます。
  - エキスパート（88 GB）が GPU + RAM にほぼ収まるので、手元の PC では 132G より約 3 割速く動きます。
  - そのかわり KL は大きめです（ISTA の削っていない IQ1_M の 0.385 と同じくらい）。長く慎重な作業では、元のモデルからずれやすくなります。
- **LDLQ 132G / 139G がおすすめの人**：エキスパートを 1 つも削っておらず、元のモデルに一番近い版です。
  - 難しいコーディング、長い推論、ほかの言語（韓国語、欧州の言語など）、珍しい知識、画像入力を使う場合。
  - 276〜283 GB のダウンロードが苦にならない場合。
- **Full（準備中）**：エキスパートを全部残して約 230 GB にする版で、上の 2 つの中間です。
- **ISTA Coder**：英語でコーディングだけ、1 GB でも小さく速くしたい場合。日本語には答えられません。

JA256 で弱くなるもの：各層 384 個のうち 128 個を削っています。上の 6 つの領域の文章でも、ルートの 5〜16% が「次に近い」エキスパートに回ります。
集計に入れていないもの（ほかの言語、珍しい分野、画像入力）は、もっと弱くなっている可能性があります。
次の版は、いただいた報告をもとに作ります（たとえばコードの重みを上げるなど）。

### 中身

- `JA256/` は 6 ファイルで 1 セットです。`-00001-of-00006.gguf` を指定してください。合計 207.1 GB（1〜5 番目 96.5 GB、そのうちエキスパート 88.4 GB。6 番目の Engram 表 110.6 GB）。
- エキスパート・ルーター・ルーターの bias は、LDLQ 132G のものを、残す 256 個の分だけバイト単位で切り出しています（量子化はやり直していません。残すエキスパートは層ごとに違います）。
- Engram 表は、unsloth の UD-Q4_K_XL に入っている FP8 の表から IQ4_NL に作り直しました。元との誤差は 0.075 で、ISTA の IQ4_NL（0.076）と同じくらいです。SSD から少しずつ読むので、RAM に載せる必要はありません。

### 品質と実用・速度

| | JA256 | LDLQ 132G | ISTA Coder |
| --- | --- | --- | --- |
| 難しめのコーディング 10 問（隠しテスト） | 8/10（t104 は解き直し 2 回とも正解、t046 は 3 回とも不正解） | 10/10 | 9/10（t046 を落とす） |
| 解けた問題の品質点 | 80.5 | 77.6 | 77.5 |
| 8K / 32K の長文から情報を探す | 成功 / 成功 | 成功 / 成功 | 成功 / 成功 |
| ツールを使う 7 往復の作業 | テスト通過、最後の報告は日本語 | 通過 | 通過、報告は中国語 |
| 日本語 3 問 | 自然な日本語 | 自然 | **中国語**（かな 0、1 本ループ） |
| 10 問の平均生成速度 | **75.7 tok/s** | 58.0 tok/s | 87.1 tok/s |
| 32K の入力処理 | **1,833 tok/s** | 865 tok/s | 1,509 tok/s |

- JA256 が毎回落とした 1 問（t046、DI コンテナのオブジェクトの寿命）は、Coder も落とした問題です。エキスパートを削った影響が、難しいコードに出た例と考えています。コーディングが主な用途なら、132G / 139G をおすすめします。
- エキスパート（88 GB）が VRAM と RAM 60 GB にほぼ収まるので、SSD からの読み出しは 1 トークンあたり 0.1 回です（132G は約 1 回）。
- 測定環境：RTX 5090（32 GB、OCuLink Gen4 x4）、Ryzen AI Max+ 395、RAM の枠 60 GB、NVMe SSD、DSpark などの普段使いの設定。
- ほかの GPU ではまだ測っていません。エキスパートが 44 GB 小さいので、RAM 64 GB の PC では、384 個の版より速さの落ち方が小さいはずです。

**GPU 12 GB + RAM 128 GB の PC（12G128G、2026-11-08 測定）**：同じ PC を 12 GB のカード・iGPU なし・RAM の置き場 90 GB に設定、
精度を落とす設定なし、Abyss v1.3.120。

| | エキスパート | 大きさ（Engram 以外） | KL（8 本） | 生成 | 8K のプロンプト |
| --- | ---: | ---: | ---: | ---: | ---: |
| **JA256、下書きなし** | 256 個 | 約 97 GB | **0.383** | **27.4 tok/s** | **375 tok/s** |
| JA256、DSpark の下書きあり | 256 個 | 約 97 GB | 0.383 | 29.5 tok/s | 26 tok/s |
| ISTA GSQ-RCO IQ1_M、下書きなし | 384 個 | 約 121 GB | 0.384 | 26.7 tok/s | 234 tok/s |

- **12 GB のカードでは、DSpark の下書きを切ってください。** 下書きを付けると、エキスパートの置き場が 1 層 4 個（1 トークンに 6 個必要）
  になり、プロンプトを 1 トークンずつ読むようになります。
- JA256 のエキスパートは RAM 90 GB に収まり、生成中に SSD を読みません。ISTA IQ1_M は KL が同じで、24 GB 大きいです。

### フィードバックのお願い（BETA 版）

- **お使いの PC での速さ**：Abyss で `./abyss.sh --bench --deep` を実行し、`abyss-bench.txt` を
  [こちらの issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+Compact-JA256+on+YOUR_GPU)に貼ってください（タイトルの GPU 名を書き換えてください）。RAM の容量と SSD の機種も書いてもらえると助かります。
- **実際に使ったときの品質**：おかしな答え、言語の切り替わり、繰り返し、132G より弱いと感じたところ。特にコーディングと、6 つの領域以外の言語の例が助かります。再現できるプロンプトがあると一番助かります。
- **欲しい版**：コード寄りの版、RAM 64 GB 向けの小さい版、ほかの GPU 向けなど。

### 作り方・動かし方・ライセンス

- 作り方と動かし方は、上の英語の節のとおりです（[trench-gguf](https://github.com/zephel01/trench-gguf) の `prune_experts.py` と `requant_table.py`、Abyss のルート集計）。
- Abyss v1.3.101 以降で動作を確認しています。llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きました。RTX 5090 で `--n-cpu-moe 36` のとき、日本語の質問と Python の課題に正しく答え、生成は 15〜20 tok/s でした（132G は `--n-cpu-moe 38` で 10〜11 tok/s）。
- ライセンスは元のモデルと同じ MIT です。
