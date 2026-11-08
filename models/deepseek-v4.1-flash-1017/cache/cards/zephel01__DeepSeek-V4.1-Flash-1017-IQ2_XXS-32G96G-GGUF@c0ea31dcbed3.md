---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- imatrix
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
---

# DeepSeek-V4.1-Flash-1017 LDLQ GGUF (for a 32 GB GPU + 96 GB RAM PC) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- `132G`: a mixed-type file - the experts are IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11% (by bytes), 1.95 bits a weight on average, about IQ2_XXS.
- `139G`: a mixed-type file - the experts are IQ2_XXS 38%, IQ1_S 29%, IQ3_XXS 19%, IQ2_S 10%, IQ2_XS 5% (by bytes), 2.05 bits a weight on average, about IQ2_XXS.

Two GGUFs of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT),
made for **one 32 GB GPU plus a 96 GB RAM PC**. All 384 routed experts of every layer are kept (no expert pruning); the
experts were re-quantized from the original FP4 weights with **LDLQ** (error-feedback rounding with each expert's own
input statistics), so they are rounded better than plain imatrix quantization at the same types.

**BETA**: measured on one PC only (below). Reports from other GPUs / RAM sizes are very welcome (Community tab, or
[Abyss issues](https://github.com/zephel01/Abyss/issues)).

日本語の説明は下にあります。

## Which one

| folder | experts | for | KL (8 seqs) | hard tasks quality | decode, coding tasks |
| --- | ---: | --- | ---: | ---: | ---: |
| [`132G/`](./132G) | 132.2 GB | **speed** - same expert bytes as unsloth UD-IQ1_M | 0.194 | 77.6 | 67.5 tok/s |
| [`139G/`](./139G) | 139.4 GB | **quality** - layers 35-39 one step higher (IQ3_XXS) | **0.182** | **81.4** | 57.9 tok/s |
| unsloth UD-IQ1_M (reference, not here) | 132.2 GB | | 0.198 | 81.1 | 71.8 tok/s |

- Pick **132G** if you want the speed of UD-IQ1_M with slightly better numbers, **139G** if you can give up 15-20%
  decode speed for the best quality at this size. Both pass every real-use check below.
- "experts" is the routed-expert bytes (decimal GB), the part that decides whether the model fits GPU + RAM.

## Files

Each folder is a complete set of 6 shards; point your engine at `-00001-of-00006.gguf`.

| | shards 1-5 (dense + experts) | shard 6 (Engram table) | total |
| --- | ---: | ---: | ---: |
| `132G/DeepSeek-V4.1-Flash-1017-LDLQ-132G-*.gguf` | 140.6 GB | 135.2 GB | 275.8 GB |
| `139G/DeepSeek-V4.1-Flash-1017-LDLQ-139G-*.gguf` | 147.8 GB | 135.2 GB | 283.0 GB |

- Shard 6 is the Engram table (Q5_K), byte-identical to unsloth's `UD-IQ2_XXS-00006-of-00006.gguf`. It is read from the
  SSD a few rows at a time and never has to fit in RAM.
- Everything except the routed experts (attention, shared experts, routers, embeddings, output, metadata, tokenizer) is
  byte-identical to unsloth's UD-IQ2_XXS (dense Q8_0).
- `SHA256SUMS` in each folder.

## What is different from unsloth's files

| layers | gate / up | down | 139G: layers 35-39 |
| --- | --- | --- | --- |
| 0-4 | IQ1_S | IQ1_S | |
| 5-19 | IQ1_S | IQ2_XXS | |
| 20-29 | IQ2_XXS | IQ2_S | |
| 30-37 | IQ2_XXS | IQ2_XS | 35-37: all IQ3_XXS |
| 38-39 | IQ2_XS | IQ2_XS | all IQ3_XXS |

The types of `132G` are exactly unsloth UD-IQ1_M's; only the rounding differs. Plain re-quantization at these types
was far worse (KL 0.404, 8 sequences; the errors of the last five layers add up), unsloth's own rounding gets 0.198, and LDLQ gets
0.194. In `139G` the last five layers, where the errors pile up, are one step higher.

## Quality

**KL** against the unquantized model (top-256 log-probs of DeepSeek-V4.1-Flash-1017 before quantization, 8 held-out
sequences x 1,023 tokens), computed in the Abyss engine with exact routing.

| | experts | mean KL | same top token | nll |
| --- | ---: | ---: | ---: | ---: |
| unsloth UD-IQ2_XXS (rev 85f81fb) | 156.9 GB | 0.142 | 83.7% | 2.173 |
| **139G** | 139.4 GB | **0.182** | 81.6% | 2.204 |
| **132G** | 132.2 GB | **0.194** | 81.1% | 2.215 |
| unsloth UD-IQ1_M | 132.2 GB | 0.198 | 81.7% | 2.209 |

KL is a guide only, so both were also run on real work (Abyss server, everyday settings below, temperature 1.0 /
top_p 0.95):

| | 132G | 139G | UD-IQ1_M | UD-IQ2_XXS |
| --- | --- | --- | --- | --- |
| 10 hard coding tasks (hidden tests), solved | 10/10 | 10/10 | 10/10 | 10/10 |
| quality score (graded) | 77.6 | **81.4** | 81.1 | 78.8 |
| 8K / 32K prompt, find the needle | OK / OK | OK / OK | OK / OK | OK / OK |
| 7-turn tool-call coding session, tests pass | yes | yes | yes | yes |
| Japanese (3 prompts), no mixed scripts | OK | OK | OK | OK |

132G's lower quality score comes from one task (47 points); the other nine are level with the rest.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.9x, one PC: **RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4**, Ryzen AI Max+ 395
(16 cores, LPDDR5X; its Radeon 8060S iGPU computes part of the RAM-side experts), **RAM tier capped at 60 GB**
(`STRATA_GLM_RAM_GB=60`), the rest read from an NVMe SSD. Everyday settings: DSpark drafter (Lucebox's DeepSeek-V4.1-Flash-1017 DSpark draft, Q2K-Q4K), `STRATA_GLM_TAIL_SKIP=4`, `STRATA_GLM_TAIL_SKIP_DISK=1`,
`STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`.

| decode tok/s | 132G | 139G | UD-IQ1_M |
| --- | ---: | ---: | ---: |
| 3 coding tasks (same day, one run each) | 67.5 | 57.9 | 71.8 |
| 10 hard tasks (mean) | 58.0 | 47.8 | 52.5 |
| 32K prompt, then decode | 43.4 | 43.4 | - |

132G and UD-IQ1_M swap places from run to run (the two sets disagree by ±5%); 139G is clearly slower on this PC
because 7 GB more has to come from RAM / SSD. A desktop with dual-channel DDR5 and no iGPU will be slower than these
numbers.

### Other GPUs (rough guide)

Only the RTX 5090 row is measured with these files. The others are labelled for what they are:

| GPU (VRAM) | 132G | 139G | basis |
| --- | --- | --- | --- |
| RTX 5090 (32 GB) | 67.5 tok/s | 57.9 tok/s | **measured** (above) |
| Radeon AI PRO R9700 (32 GB) | ~40 tok/s | ~33-35 tok/s | **estimate**: unsloth UD-IQ1_M - the same types and bytes as 132G, only the rounding differs - measured 40.6 tok/s on an R9700 + 96 GB RAM (8 coding tasks, DSpark; 30.5 with 64 GB RAM). 139G scaled by its 5090 slowdown. To be replaced by measurements |
| V100 (32 GB) | not measured | not measured | Abyss builds for sm_70 with CUDA 12.9 (`--nvcc`) but has not run on a V100 yet; no FP4 / newer tensor-core kernels there, so expect less than the R9700. Reports welcome |
| RTX 3090 / 4090 (24 GB) | not measured | not measured | 8 GB under this repo's 32 GB target: more experts come from the SSD, so speed follows the SSD. Use 132G; reports welcome |

## Runtime options

Abyss has runtime options that change how the model answers without changing these files (for example
`--uncensor`, see [docs/UNCENSOR.md](https://github.com/zephel01/Abyss/blob/main/docs/UNCENSOR.md)). **The files here are
the normal model.**

## Feedback (this is a BETA)

The next versions of these files will follow what users report. Useful are:

- **Speed on your PC**: run `./abyss.sh --bench --deep` in Abyss and post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+LDLQ-132G+on+YOUR_GPU) (it already has the right template; change the title to your GPU). Add your RAM size and SSD.
- **Quality in real use**: tasks where it went wrong, broke into another language or looped, or refused too much, in the Community tab here or an Abyss issue. A prompt that reproduces it helps most.
- **What you would rather have**: a smaller file for 64 GB RAM, a bigger one for 128 GB, other GPUs.

Measured reports go into the GPU table above with credit; problems that show up go into the next build.

## How it was made

All with [trench-gguf](https://github.com/zephel01/trench-gguf) v0.3.0 (GPU quantizer, LDLQ). Every command and both rules files
are in its README, [LDLQ walkthrough](https://github.com/zephel01/trench-gguf#-ldlq-walkthrough-the-published-deepseek-v41-ldlq-132g--139g):

1. Source: unsloth's `UD-Q4_K_XL` of DeepSeek-V4.1-Flash-1017, whose routed experts are the original MXFP4 (FP4) weights.
2. Hessians: every layer's expert inputs dumped from a forward pass over 491,520 tokens of general English / Japanese
   text and code (`hessian_dump`, a patched llama.cpp on the GPU, 49 min on the RTX 5090).
3. Re-quantization: `requant_experts.py --gpu --dump=<hessians> --order=act --damp=0.3 --damp-iq1=1.0` with unsloth's
   imatrix, the type table above (132G), then layers 35-39 again at IQ3_XXS (139G).
4. Assembly: `merge_experts.py` puts the new expert tensors into unsloth UD-IQ2_XXS (rev 85f81fb) byte for byte; the
   result's Abyss pack was checked to have exactly the measured layout.

Experts with fewer than 64 calibration rows (a few per layer) fall back to plain imatrix rounding.

## Running it

- **[Abyss](https://github.com/zephel01/Abyss)** v1.3.100 or later, one command (engine, download with SHA-256 check,
  pack, config with the everyday settings below): `./setup-deepseek.sh --model v41 --download-model` (`--quant 139G`
  for the quality build). By hand (tested): pack it once (the pack indexes the shards in place):
  ```sh
  cd Abyss
  .venv/bin/python tools/iq_pack.py --compat-bf16 --gguf /path/to/132G/DeepSeek-V4.1-Flash-1017-LDLQ-132G-00001-of-00006.gguf --out /path/to/132G/pack
  ```
  and use `docs/deepseek41/README.md`'s config with `--glm-pack /path/to/132G/pack`.
- **llama.cpp** (arch `deepseek41`, upstream `ac8e1694d` of 2026-10-29): runs `132G` as is. Checked with
  `llama-server -m DeepSeek-V4.1-Flash-1017-LDLQ-132G-00001-of-00006.gguf -ngl 99 --n-cpu-moe 38 -c 4096 --jinja` on
  the RTX 5090 (32 GB): correct answers to a Japanese question and a Python task, 10-11 tok/s (experts of two layers on
  the GPU, the rest from RAM / SSD). `--n-cpu-moe 34` ran out of VRAM on the first request.

## License

MIT, as the original model. This repository only holds a quantized conversion; all credit for the model goes to
DeepSeek, and the non-expert tensors and Engram table come unchanged from
[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).

---

## 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- `132G`: 形式を混ぜたファイルです。エキスパートは IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11%（容量の割合）、平均 1.95 ビットで、IQ2_XXS くらいです。
- `139G`: 形式を混ぜたファイルです。エキスパートは IQ2_XXS 38%, IQ1_S 29%, IQ3_XXS 19%, IQ2_S 10%, IQ2_XS 5%（容量の割合）、平均 2.05 ビットで、IQ2_XXS くらいです。

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)（MIT）の GGUF を 2 種類置いています。
**GPU 32 GB + RAM 96 GB の PC** で快適に動かすことを目標に作りました。エキスパートは 1 つも削っていません。
ルーティングされるエキスパートは、元の FP4 の重みから **LDLQ**（エキスパートごとの入力の統計を使い、丸めの誤差を次の列に送る方法）で作り直しています。
そのため、同じ型の普通の imatrix 量子化より丸めの誤差が小さくなります。

**BETA 版です。** 測ったのは下の 1 台だけです。ほかの GPU や RAM 容量で試した結果を、Community タブか
[Abyss の issue](https://github.com/zephel01/Abyss/issues) で教えてもらえると助かります。

### どちらを選ぶか

| フォルダ | エキスパート | 向き | KL（8 本） | 難しめ 10 問の品質 | 生成速度（コーディング課題） |
| --- | ---: | --- | ---: | ---: | ---: |
| `132G/` | 132.2 GB | **速さ重視**（unsloth UD-IQ1_M と同じ大きさ） | 0.194 | 77.6 | 67.5 tok/s |
| `139G/` | 139.4 GB | **品質重視**（層 35〜39 を 1 段上の IQ3_XXS に） | **0.182** | **81.4** | 57.9 tok/s |
| unsloth UD-IQ1_M（比較用。ここにはありません） | 132.2 GB | | 0.198 | 81.1 | 71.8 tok/s |

- UD-IQ1_M と同じくらいの速さで、数字が少し良いものが欲しいなら **132G** を選んでください。
- 生成が 15〜20% 遅くなってもよいので、この大きさで一番の品質が欲しいなら **139G** です。
- どちらも、下の実用チェックをすべて通っています。

### 中身

- 各フォルダは 6 ファイルで 1 セットです。`-00001-of-00006.gguf` を指定してください。
- 6 番目のファイルは Engram の表です。unsloth のものとまったく同じで、SSD から少しずつ読むので RAM に載せる必要はありません。
- エキスパート以外（attention、共有エキスパート、ルーター、埋め込み、出力、トークナイザー）は、unsloth UD-IQ2_XXS とまったく同じです（dense は Q8_0）。
- 132G の型の配置は unsloth UD-IQ1_M とまったく同じで、違うのは丸め方だけです。
  - この型のまま普通に量子化し直すと、大きく崩れました（KL 0.404、8 系列）。最後の 5 層の誤差が積み重なるためです。
  - unsloth 独自の丸めでは 0.198、LDLQ では 0.194 になりました。
  - 139G は、誤差がたまりやすい最後の 5 層を 1 段上の型にしています。

### 品質と実用

- **KL：** 量子化前のモデルとの差を、8 本 × 1,023 トークンで測りました。
  - unsloth UD-IQ2_XXS（156.9 GB）：0.142
  - **139G：0.182**
  - **132G：0.194**
  - unsloth UD-IQ1_M：0.198
- **実際の作業：** KL はあくまで目安なので、実際の作業でも確かめました。どちらも次のチェックをすべて通っています。
  - 難しめのコーディング 10 問（隠しテストあり）：全問正解
  - 8K / 32K の長文から指定した情報を探す：成功
  - ツールを使う 7 往復のコーディング作業：テスト通過
  - 日本語 3 問：文字化け・他言語の混入なし
- 132G の品質点が少し低いのは、1 問だけ 47 点だったためです。残りの 9 問はほかと同じ水準です。

### 速度

- **測定環境：** RTX 5090（32 GB、OCuLink Gen4 x4）、Ryzen AI Max+ 395（iGPU も RAM 側のエキスパートの計算に使用）、RAM の枠は 60 GB、残りは NVMe SSD から読み出し。DSpark の下書きなど、普段使いの設定です。

  | 生成速度 | 132G | 139G | UD-IQ1_M |
  | --- | ---: | ---: | ---: |
  | コーディング課題 3 問 | 67.5 tok/s | 57.9 tok/s | 71.8 tok/s |
  | 難しめの 10 問 | 58.0 tok/s | 47.8 tok/s | 52.5 tok/s |

- 132G と UD-IQ1_M は測るたびに順位が入れ替わる程度の差（±5%）です。139G は、RAM や SSD から読む量が 7 GB 増える分、はっきり遅くなります。
- DDR5 デュアルチャネルで iGPU のないデスクトップでは、この数字より遅くなります。

### ほかの GPU での目安

このファイルで実測したのは RTX 5090 だけです。ほかは、根拠を書いたうえで目安として載せています。

| GPU（VRAM） | 132G | 139G | 根拠 |
| --- | --- | --- | --- |
| RTX 5090（32 GB） | 67.5 tok/s | 57.9 tok/s | **実測**（上の表） |
| Radeon AI PRO R9700（32 GB） | 約 40 tok/s | 約 33〜35 tok/s | **目安**：132G と型・バイト数が同じ unsloth UD-IQ1_M（違うのは丸めだけ）を R9700 + RAM 96 GB で実測すると 40.6 tok/s でした（コーディング 8 問、DSpark あり。RAM 64 GB では 30.5）。139G は 5090 での遅くなり方から換算しています。あとで実測に置き換えます |
| V100（32 GB） | 未測定 | 未測定 | Abyss は CUDA 12.9（`--nvcc`）で sm_70 向けにビルドできますが、V100 の実機ではまだ動かしていません。FP4 や新しい tensor core のカーネルが使えないので、R9700 より遅いと思われます。結果を教えてもらえると助かります |
| RTX 3090 / 4090（24 GB） | 未測定 | 未測定 | このリポジトリが想定する 32 GB より 8 GB 少ないため、SSD から読むエキスパートが増え、速さはほぼ SSD 次第になります。使うなら 132G がおすすめです。結果を教えてもらえると助かります |

### 実行時のオプション

Abyss には、ファイルを変えずに答え方を変える実行時のオプションがあります（たとえば `--uncensor`。詳しくは [docs/UNCENSOR](https://github.com/zephel01/Abyss/blob/main/docs/UNCENSOR.ja.md)）。**ここにあるファイルは、普通のモデルです。**

### フィードバックのお願い（BETA 版）

このファイルの次の版は、使った方からの報告をもとに作ります。特に次の 3 つが助かります。

- **お使いの PC での速さ：**
  - Abyss で `./abyss.sh --bench --deep` を実行して、`abyss-bench.txt` を[こちらの issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+LDLQ-132G+on+YOUR_GPU)に貼ってください。テンプレートは設定済みなので、タイトルの GPU 名だけ書き換えてください。
  - RAM の容量と SSD の機種も書いてもらえると助かります。
- **実際に使ったときの品質：**
  - おかしな答えになった、別の言語が混ざった、同じ文が繰り返された、断りすぎた、といった例を教えてください。
  - ここの Community タブか、Abyss の issue へお願いします。再現できるプロンプトがあると一番助かります。
- **欲しい版：** RAM 64 GB 向けの小さい版、128 GB 向けの大きい版、ほかの GPU 向けなど。

いただいた実測値は、上の GPU の表に名前を添えて載せます。見つかった問題は、次の版で直します。

### 作り方

- 道具はすべて [trench-gguf](https://github.com/zephel01/trench-gguf) v0.3.0（GPU で量子化する道具、LDLQ 対応）です。コマンドと規則ファイルは、README の [LDLQ の実例](https://github.com/zephel01/trench-gguf/blob/main/README.ja.md#-ldlq-の実例公開している-deepseek-v41-ldlq-132g--139g-を作る) にすべて載っています。
- 手順：
  1. 元の重みには、unsloth の UD-Q4_K_XL を使いました。ここに入っているエキスパートは、元の MXFP4（FP4）のままです。
  2. 一般的な英語・日本語・コードの文章 491,520 トークンを流して、各層のエキスパートへの入力を取り出しました（RTX 5090 で 49 分）。
  3. それを使い、LDLQ でエキスパートを作り直しました。imatrix は unsloth のものです。
  4. 作り直したエキスパートを、unsloth UD-IQ2_XXS に 1 バイトも変えずに差し込みました。

### 動かし方・ライセンス

- Abyss v1.3.100 以降なら、`./setup-deepseek.sh --model v41 --download-model` の 1 行で、エンジンのビルド、ダウンロード（SHA-256 の照合つき）、pack、普段使いの設定の入った設定ファイルまで済みます。品質重視の版は `--quant 139G` を足してください。
- 手作業の動かし方とライセンスは、上の英語の節のとおりです（Abyss で動作確認済み）。
- llama.cpp（2026-10-29 の ac8e1694d）でも `132G` がそのまま動きました。RTX 5090 で `--n-cpu-moe 38` のとき、日本語の質問と Python の課題に正しく答え、生成は 10〜11 tok/s でした（`--n-cpu-moe 34` では最初のリクエストで VRAM が足りなくなりました）。
- ライセンスは元のモデルと同じ MIT です。
