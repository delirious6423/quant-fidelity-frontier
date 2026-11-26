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
language:
- en
- ja
library_name: gguf
---

![DeepSeek-V4.1-Flash-1017 JA256: 87.0 tok/s on an RTX 5090 with the RAM tier capped at 60 GB](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | [Unpruned LDLQ 132G / 139G](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | [trench-gguf](https://github.com/zephel01/trench-gguf) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+Compact-JA256+on+YOUR_GPU) | 日本語は下

# DeepSeek-V4.1-Flash-1017 E256-IQ2_XXS GGUF (for a 32 GB GPU + 96 GB RAM PC, or 12 GB + 128 GB) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- experts sliced from `132G`: a mixed-type file - the experts are IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11% (by bytes, as in 132G), 1.95 bits a weight on average, about IQ2_XXS.

A **general-purpose pruned build** of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT) - **not code-only**. 256 of 384 experts per layer are kept, chosen by measured routing on chat, math, tool-call,
code and multilingual text. 207 GB instead of the 276 GB of our unpruned
[LDLQ 132G](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF), and faster on one 32 GB GPU
plus a 96 GB RAM PC. Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only (RTX 5090 32 GB, RAM tier capped at 60 GB with `STRATA_GLM_RAM_GB=60`). Reports from other GPUs / RAM sizes are welcome (see Feedback).

## Model files

`JA256/DeepSeek-V4.1-Flash-1017-JA256-0000N-of-00006.gguf`, 6 shards. Point your engine at `-00001-of-00006.gguf`.
`SHA256SUMS` is in the folder.

| | shards 1-5 (dense + experts) | shard 6 (Engram table) | total |
| --- | ---: | ---: | ---: |
| JA256 | 96.5 GB (experts 88.4 GB) | 110.6 GB (IQ4_NL) | 207.1 GB |

- The experts, routers and router biases are LDLQ 132G's, sliced byte for byte to the 256 kept experts per layer
  (`expert_count` = 256; nothing re-quantized). Which experts are kept differs per layer.
- The Engram table was re-quantized to IQ4_NL from the FP8 table in unsloth's UD-Q4_K_XL (relative RMS error 0.075
  vs ISTA's IQ4_NL table 0.076 on the same rows). It is read from the SSD a few rows at a time and never has to fit in RAM.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 275.8 / 283.0 GB | all 384 experts (folders 132G / 139G) |
| **this repo** | 32 GB GPU + 96 GB RAM; also 12 GB GPU + 128 GB RAM | 207.1 GB | 256 of 384 experts kept (folder JA256) |

Earlier repo name: DeepSeek-V4.1-Flash-1017-Compact-32G96G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: why a general-purpose selection

- A code-only selection drops other abilities first. Our routing counts show that a code-only selection of 192
  experts keeps only 45% of the routes Japanese text takes (code: 82%).
- [ISTA-DASLab's GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) (192 of
  384 experts, chosen on code, agentic, long-context and vision data) answered all three of our Japanese prompts in
  Chinese. (ISTA's unpruned GSQ-RCO releases keep all experts; this is about the Coder only.)
- **JA256** selects its experts on Japanese, Chinese, English, code, math and tool-call text, every domain weighted the
  same. Kept routes: Japanese 95%, Chinese 94%, math 94%, tools 94%, English 91%, code 84% (about the Coder's 82%).
- What JA256 gives up: 128 of 384 experts per layer are gone. Text in the six domains loses 5-16% of its routes to the
  next-best experts. Anything outside them (other languages, rare domains, image input - not in our counts) may lose
  more.

| | experts per layer | download | KL (8 seqs) | same top token | decode, 3 coding tasks | Japanese |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **[`JA256/`](./JA256) (here)** | **256 / 384** | **207.1 GB** | 0.383 | 75.2% | **87.0 tok/s** | OK |
| [LDLQ `132G`](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | 384 / 384 | 275.8 GB | 0.194 | 81.1% | 67.5 tok/s | OK |
| [LDLQ `139G`](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | 384 / 384 | 283.0 GB | **0.182** | 81.6% | 57.9 tok/s | OK |
| ISTA GSQ-RCO Coder (reference) | 192 / 384 | 182.7 GB | 0.582 | 69.3% | 97.0 tok/s | **Chinese** |
| ISTA GSQ-RCO IQ2_XXS (reference) | 384 / 384 | ~249 GB | 0.314 | 76.8% | - | - |

How to choose:

- **JA256**: smaller download and the fastest file that keeps Japanese working - everyday chat, Japanese / Chinese /
  English, coding, math and tool calls. Its KL is higher (about the same as ISTA's unpruned IQ1_M, 0.385), so long,
  careful work drifts more.
- **LDLQ 132G / 139G**: no expert removed, closest to the original. Pick these for hard coding work, long reasoning,
  other languages (Korean, European languages, ...), rare knowledge or image input.
- **ISTA Coder**: English coding only, if every GB and token per second counts; it does not answer Japanese.

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.101 or later. The one-command setup does not install this repo
yet, so set up the engine with the V4.1 setup once, then add this file by hand:

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model v41          # builds the engine; decline the 132G download and it stops there, which is enough
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF --include "JA256/*" --local-dir ../Abyss-data/models/v41-ja256
.venv/bin/python tools/iq_pack.py --compat-bf16 \
  --gguf ../Abyss-data/models/v41-ja256/JA256/DeepSeek-V4.1-Flash-1017-JA256-00001-of-00006.gguf \
  --out ../Abyss-data/models/v41-ja256/JA256/pack
cat > abyss-dsv41-ja256.json <<EOF
{"exe": "$PWD/build/strata",
 "args": ["--glm-pack", "$PWD/../Abyss-data/models/v41-ja256/JA256/pack", "--max-context", "65536", "--prefill", "auto"],
 "cwd": "$PWD", "tokenizer": "$PWD/../Abyss-data/models/v41-ja256/JA256/pack/tokenizer",
 "model_name": "deepseek-v4.1-flash-1017", "gpu": [0], "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082, "log": "$PWD/abyss-dsv41-ja256.log",
 "env": {"STRATA_GLM_RAM_GB": "60", "STRATA_GLM_DISK_READERS": "8", "STRATA_GLM_TAIL_SKIP": "4",
         "STRATA_GLM_TAIL_SKIP_DISK": "1", "STRATA_GLM_ROUTE_AFFINITY_RAM": "0.08"}}
EOF
systemd-run --user --scope -p MemoryMax=88G .venv/bin/python serve/server.py --engine strata \
  --config abyss-dsv41-ja256.json --port 8082
```

- The `env` block holds the everyday settings the speeds below were measured with. Drop the two `TAIL_SKIP` entries
  for exact routing. On a 12 GB GPU + 128 GB RAM PC, see that section below.
- The memory-capped scope keeps systemd-oomd from killing your session while the page cache fills (see
  [docs/deepseek41](https://github.com/zephel01/Abyss/blob/main/docs/deepseek41/README.md)).

**llama.cpp** (arch `deepseek41`, upstream `ac8e1694d` of 2026-10-29) runs `JA256` as is (the expert count is read
from the file). Checked on the RTX 5090 (32 GB): correct answers to a Japanese question and a Python task. Speed with
`llama-server -c 32768 -b 4096 -ub 4096` (its own GPU/CPU fit): see the table under Speed.

```sh
hf download zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF --include "JA256/*" --local-dir V41-JA256
llama-server -m V41-JA256/JA256/DeepSeek-V4.1-Flash-1017-JA256-00001-of-00006.gguf -ngl 99 --n-cpu-moe 36 -c 4096 --jinja
```

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

![Decode and prompt speed: JA256 vs LDLQ 132G and ISTA Coder](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF/resolve/main/assets/speed.png)

Abyss v1.3.101, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395 (LPDDR5X, Radeon 8060S iGPU computes
part of the RAM-side experts), RAM tier capped at 60 GB, NVMe SSD. Everyday settings: DSpark drafter (Lucebox
Q2K-Q4K), `STRATA_GLM_TAIL_SKIP=4`, `STRATA_GLM_TAIL_SKIP_DISK=1`, `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`.

| decode tok/s | JA256 | LDLQ 132G | ISTA Coder |
| --- | ---: | ---: | ---: |
| 3 coding tasks | **87.0** | 67.5 | 97.0 |
| 10 hard tasks (mean) | **75.7** | 58.0 | 87.1 |
| 32K prompt: prompt processing | **1,833** | 865 | 1,509 |
| 32K prompt: decode after it | 63.1 | 43.4 | 76.8 |

JA256's experts (88 GB) nearly fit in VRAM + the 60 GB RAM tier: 0.1 SSD reads a token against ~1 for 132G. Other GPUs
and 64 GB RAM are not measured with this file; reports welcome.

**Abyss and llama.cpp on this file** (2026-11-26, same PC with 123 GB RAM, iGPU off, no drafter, nothing lossy:
`STRATA_GLM_TAIL_SKIP=0`; Abyss v1.3.211 with a 60 GB RAM tier; llama.cpp `ac8e1694d`, `llama-server -c 32768 -b 4096
-ub 4096 -t 16` with its own GPU/CPU fit; the same prompt tokens, greedy, 512 tokens, median of 3):

| | Abyss | llama.cpp |
| --- | ---: | ---: |
| decode, short prompt | **67.2 tok/s** | 26.6 tok/s |
| decode after a 4K prompt | **62.5 tok/s** | 27.6 tok/s |
| 4K prompt | **735 tok/s** | 361 tok/s |

### A 12 GB GPU + 128 GB RAM PC (12G128G, measured 2026-11-08)

Abyss v1.3.120 (+ the IQ1_S prompt fix of the next release for ISTA), the same PC set up as a 12 GB card
(`STRATA_GLM_VRAM_GB=12`) with no iGPU (`STRATA_GLM_IGPU=0`), the RAM tier at 90 GB (80 for ISTA), exact settings only
(no tail skip, no route affinity), `./abyss.sh --bench`:

| | experts | KL (8 seqs) | decode | 8K prompt |
| --- | ---: | ---: | ---: | ---: |
| **JA256, no drafter** | 256 | **0.383** | **27.4 tok/s** | **375 tok/s** |
| JA256, DSpark drafter | 256 | 0.383 | 29.5 tok/s | 26 tok/s |
| ISTA GSQ-RCO IQ1_M, no drafter | 384 | 0.384 | 26.7 tok/s | 234 tok/s |

- **On a 12 GB card, leave the DSpark drafter off.** With it, the dense weights and the drafter leave the expert pool
  4 slots a layer (a route needs 6), and the prompt runs token by token.
- JA256's experts fit the 90 GB RAM tier: no SSD reads while it writes. ISTA's IQ1_M has the same KL and is 24 GB larger.

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+Compact-JA256+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language, looped or got weaker than 132G - especially
  coding and languages outside the six domains. A prompt that reproduces it helps most.
- **What you would rather have**: more code weight, a smaller one for 64 GB RAM, other GPUs.

<details>
<summary>How it was made</summary>

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/prune_experts.py`, `tools/requant_table.py`) and
Abyss's routing counter (`STRATA_GLM_USAGE`):

1. Routing counts: LDLQ 132G (all 384 experts) run over ~45K tokens each of Japanese, Chinese, English, code, math
   and tool-call text (eaddario/imatrix-calibration, micro sets) plus Japanese answers, counting which experts each
   layer routes to.
2. Selection: per layer the 256 experts with the most routes, every domain weighted the same, chosen on 2/3 of the
   text and checked on the other 1/3 (kept routes: ja 94.8%, zh 93.6%, en 90.6%, code 83.6%, math 93.8%, tools 93.5%).
3. Slicing: `prune_experts.py` cuts the experts, router rows and biases of LDLQ 132G.
4. Engram: `requant_table.py` re-quantizes the FP8 table of unsloth UD-Q4_K_XL to IQ4_NL on the GPU (11 min).

</details>

## License

MIT, as the original model. All credit for the model goes to DeepSeek; the non-expert tensors come unchanged from
[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).

---

## 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- エキスパートは `132G` から切り出したものです。形式を混ぜたファイルで、エキスパートは IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11%（容量の割合、132G の値）、平均 1.95 ビットで、IQ2_XXS くらいです。

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)（MIT）の GGUF を、
自作の [LDLQ 132G / 139G](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) より小さくして置いています。
**GPU 32 GB + RAM 96 GB の PC** 向けで、276 GB のダウンロードは重いという方のための版です。

**BETA 版です。** 測ったのは 1 台だけです（RTX 5090、RAM の置き場を 60 GB に制限）。ほかの GPU や RAM 容量での結果を教えてもらえると助かります。

| | 1 層のエキスパート | ダウンロード | KL（8 本） | 1 位の一致 | 生成速度（コーディング 3 問） | 日本語 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **`JA256/`（ここ）** | **256 / 384** | **207.1 GB** | 0.383 | 75.2% | **87.0 tok/s** | OK |
| LDLQ `132G` | 384 / 384 | 275.8 GB | 0.194 | 81.1% | 67.5 tok/s | OK |
| LDLQ `139G` | 384 / 384 | 283.0 GB | **0.182** | 81.6% | 57.9 tok/s | OK |
| ISTA GSQ-RCO Coder（参考） | 192 / 384 | 182.7 GB | 0.582 | 69.3% | 97.0 tok/s | **中国語** |
| ISTA GSQ-RCO IQ2_XXS（参考） | 384 / 384 | 約 249 GB | 0.314 | 76.8% | - | - |

- **作った理由**：コードだけで 192 個を選ぶと、日本語のルートは 45% しか残りません（コードは 82%）。ISTA の Coder は、手元のテストで日本語の質問 3 本すべてに中国語で答えました。
  JA256 は日本語・中国語・英語・コード・数学・ツール呼び出しの文章で選び、256 個で日本語のルートの 95%、コードの 84% を残しています。
- **JA256 がおすすめの人**：ダウンロードを小さくしたい、速さがほしい、日本語も使う。エキスパート（88 GB）が GPU + RAM にほぼ収まるので、手元の PC では 132G より約 3 割速く動きます。
  そのかわり KL は大きめで、長く慎重な作業では元のモデルからずれやすくなります。
- **LDLQ 132G / 139G がおすすめの人**：難しいコーディング、長い推論、ほかの言語、珍しい知識、画像入力を使う場合。
- 難しめのコーディング 10 問は 8 問正解（132G は 10 問）。JA256 が毎回落とした 1 問（t046）は Coder も落とした問題で、エキスパートを削った影響が難しいコードに出た例です。
  コーディングが主な用途なら、132G / 139G をおすすめします。
- 8K / 32K の長文から情報を探す、ツールを使う 7 往復の作業（最後の報告は日本語）、日本語 3 問はすべて問題なし。
- 測定環境：RTX 5090（32 GB）、Ryzen AI Max+ 395、RAM の枠 60 GB、NVMe SSD、DSpark などの普段使いの設定、Abyss v1.3.101。10 問の平均生成は 75.7 tok/s（132G 58.0）、32K の入力処理は 1,833 tok/s（132G 865）。
- **GPU 12 GB + RAM 128 GB の PC**（2026-11-08 測定、Abyss v1.3.120、精度を落とす設定なし）：下書きなしで生成 27.4 tok/s、8K のプロンプト 375 tok/s。
  **12 GB のカードでは DSpark の下書きを切ってください**（付けるとプロンプトを 1 トークンずつ読むようになり、8K のプロンプトが 26 tok/s に落ちます）。
- Abyss では、`./setup-deepseek.sh --model v41` でエンジンを用意したあと、上の英語の節のコマンドでこのファイルを足します（1 行のセットアップにはまだ入っていません）。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きました。RTX 5090 で、日本語の質問と Python の課題に正しく答えました。
- **同じファイルでの Abyss と llama.cpp**（2026-11-26、同じ PC、iGPU なし、下書きなし、精度を落とす設定なし。Abyss v1.3.211 は RAM の置き場 60 GB、llama.cpp は `llama-server -c 32768 -b 4096 -ub 4096 -t 16` で GPU と CPU の分け方は llama.cpp に任せる。同じプロンプトのトークン、greedy、512 トークン、3 回の中央値）：
  生成（短いプロンプト）Abyss 67.2 / llama.cpp 26.6 tok/s、4K のあとの生成 62.5 / 27.6 tok/s、4K のプロンプト 735 / 361 tok/s。
- 作り方は上の英語の節のとおりです。ライセンスは元のモデルと同じ MIT です。

速さの報告は [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+Compact-JA256+on+YOUR_GPU) に、
品質の報告（132G より弱いと感じたところ、特にコーディングと 6 つの領域以外の言語）は Community タブか Abyss の issue にお願いします。

### サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 275.8 / 283.0 GB | 384 個すべて（132G / 139G） |
| **この版** | GPU 32 GB + RAM 96 GB、GPU 12 GB + RAM 128 GB でも | 207.1 GB | 384 個中 256 個（JA256） |

以前のリポジトリ名は DeepSeek-V4.1-Flash-1017-Compact-32G96G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
