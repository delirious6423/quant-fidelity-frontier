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
language:
- en
- ja
library_name: gguf
---

![DeepSeek-V4.1-Flash-1017 LDLQ 132G: 67.5 tok/s on an RTX 5090 with the RAM tier capped at 60 GB](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | [E256 (JA256) version](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF) | [trench-gguf (LDLQ)](https://github.com/zephel01/trench-gguf) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+LDLQ-132G+on+YOUR_GPU) | 日本語は下

# DeepSeek-V4.1-Flash-1017 IQ2_XXS GGUF (for a 32 GB GPU + 96 GB RAM PC) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- `132G`: a mixed-type file - the experts are IQ2_XXS 45%, IQ1_S 30%, IQ2_XS 14%, IQ2_S 11% (by bytes), 1.95 bits a weight on average, about IQ2_XXS.
- `139G`: a mixed-type file - the experts are IQ2_XXS 38%, IQ1_S 29%, IQ3_XXS 19%, IQ2_S 10%, IQ2_XS 5% (by bytes), 2.05 bits a weight on average, about IQ2_XXS.

Two GGUFs of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT)
for **one 32 GB GPU plus a 96 GB RAM PC**. All 384 routed experts of every layer are kept (no pruning).

**BETA**: measured on one PC only (RTX 5090 32 GB, RAM tier capped at 60 GB with `STRATA_GLM_RAM_GB=60`). Reports from other GPUs / RAM sizes are welcome (see Feedback).

## Model files

Each folder is a complete set of 6 shards. Point your engine at `-00001-of-00006.gguf`. `SHA256SUMS` in each folder.

| folder | for | experts | shards 1-5 (dense + experts) | shard 6 (Engram table) | total |
| --- | --- | ---: | ---: | ---: | ---: |
| [`132G/`](./132G) | **speed** - same expert bytes as unsloth UD-IQ1_M | 132.2 GB | 140.6 GB | 135.2 GB | 275.8 GB |
| [`139G/`](./139G) | **quality** - layers 35-39 one step higher (IQ3_XXS) | 139.4 GB | 147.8 GB | 135.2 GB | 283.0 GB |

- Shard 6 is the Engram table (Q5_K), byte-identical to unsloth's `UD-IQ2_XXS-00006-of-00006.gguf`. It is read from the
  SSD a few rows at a time and never has to fit in RAM.
- Everything except the routed experts (attention, shared experts, routers, embeddings, output, metadata, tokenizer) is
  byte-identical to unsloth's UD-IQ2_XXS (dense Q8_0).
- "experts" is the routed-expert bytes (decimal GB), the part that decides whether the model fits GPU + RAM.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 32 GB GPU + 96 GB RAM | 275.8 / 283.0 GB | all 384 experts (folders 132G / 139G) |
| [DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF) | 32 GB GPU + 96 GB RAM; also 12 GB GPU + 128 GB RAM | 207.1 GB | 256 of 384 experts kept (folder JA256) |

Earlier repo name: DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: same types as UD-IQ1_M, better rounding

- The experts were re-quantized from the original FP4 weights with **LDLQ** (error-feedback rounding with each
  expert's own input statistics).
- The types of `132G` are exactly unsloth UD-IQ1_M's; only the rounding differs. Plain re-quantization at these types
  was far worse (KL 0.404, 8 sequences; the errors of the last five layers add up). unsloth's own rounding gets 0.198,
  LDLQ 0.194.
- In `139G` the last five layers, where the errors pile up, are one step higher.
- Pick **132G** for the speed of UD-IQ1_M with slightly better numbers. Pick **139G** if you can give up 15-20% decode
  speed for the best quality at this size. Both pass every real-use check below.

| layers | gate / up | down | 139G: layers 35-39 |
| --- | --- | --- | --- |
| 0-4 | IQ1_S | IQ1_S | |
| 5-19 | IQ1_S | IQ2_XXS | |
| 20-29 | IQ2_XXS | IQ2_S | |
| 30-37 | IQ2_XXS | IQ2_XS | 35-37: all IQ3_XXS |
| 38-39 | IQ2_XS | IQ2_XS | all IQ3_XXS |

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.100 or later (one command: engine, download with SHA-256 check,
pack, config with the everyday settings below):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model v41 --download-model                  # 132G; add --quant 139G for the quality build
../Abyss-tools/abyss.sh start                                     # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --model v41 --download-model`.

**llama.cpp** (arch `deepseek41`, upstream `ac8e1694d` of 2026-10-29) runs `132G` as is. Checked on the RTX 5090
(32 GB): correct answers to a Japanese question and a Python task, 10-11 tok/s (experts of two layers on the GPU, the
rest from RAM / SSD). `--n-cpu-moe 34` ran out of VRAM on the first request.

```sh
hf download zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF --include "132G/*" --local-dir V41-132G
llama-server -m V41-132G/132G/DeepSeek-V4.1-Flash-1017-LDLQ-132G-00001-of-00006.gguf -ngl 99 --n-cpu-moe 38 -c 4096 --jinja
```

The files here are the normal model; Abyss's runtime options do not change them.

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

![Decode speed of 132G / 139G vs unsloth UD-IQ1_M on a 32 GB GPU + 96 GB RAM PC](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF/resolve/main/assets/speed.png)

Measured 2026-11-02 with an Abyss v1.3.90-line build (the exact build was not recorded), one PC: **RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4**, Ryzen AI Max+ 395 (16 cores, LPDDR5X; its Radeon
8060S iGPU computes part of the RAM-side experts), **RAM tier capped at 60 GB** (`STRATA_GLM_RAM_GB=60`), the rest read
from an NVMe SSD. Everyday settings: DSpark drafter (Lucebox's DeepSeek-V4.1-Flash-1017 DSpark draft, Q2K-Q4K),
`STRATA_GLM_TAIL_SKIP=4`, `STRATA_GLM_TAIL_SKIP_DISK=1`, `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`.

| decode tok/s | 132G | 139G | UD-IQ1_M |
| --- | ---: | ---: | ---: |
| 3 coding tasks (same day, one run each) | 67.5 | 57.9 | 71.8 |
| 10 hard tasks (mean) | 58.0 | 47.8 | 52.5 |
| 32K prompt, then decode | 43.4 | 43.4 | - |

- 132G and UD-IQ1_M swap places from run to run (the two sets disagree by ±5%).
- 139G is clearly slower on this PC, because 7 GB more has to come from RAM / SSD.
- A desktop with dual-channel DDR5 and no iGPU will be slower than these numbers.
- Only the RTX 5090 is measured with these files. AI PRO R9700, V100 and 24 GB cards (RTX 3090 / 4090): not measured;
  reports welcome. On 24 GB, more experts come from the SSD, so speed follows the SSD; use 132G there.

## Feedback (this is a BETA)

The next versions of these files will follow what users report. Useful are:

- **Speed on your PC**: run `../Abyss-tools/abyss.sh --bench --deep` in Abyss and post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+LDLQ-132G+on+YOUR_GPU)
  (change the title to your GPU). Add your RAM size and SSD.
- **Quality in real use**: tasks where it went wrong, broke into another language or looped, in the Community tab here
  or an Abyss issue. A prompt that reproduces it helps most.
- **What you would rather have**: a smaller file for 64 GB RAM, a bigger one for 128 GB, other GPUs.

Measured reports go into this card with credit; problems that show up go into the next build.

<details>
<summary>How it was made</summary>

All with [trench-gguf](https://github.com/zephel01/trench-gguf) v0.3.0 (GPU quantizer, LDLQ). Every command and both
rules files are in its README,
[LDLQ walkthrough](https://github.com/zephel01/trench-gguf#-ldlq-walkthrough-the-published-deepseek-v41-ldlq-132g--139g):

1. Source: unsloth's `UD-Q4_K_XL` of DeepSeek-V4.1-Flash-1017, whose routed experts are the original MXFP4 (FP4) weights.
2. Hessians: every layer's expert inputs dumped from a forward pass over 491,520 tokens of general English / Japanese
   text and code (`hessian_dump`, a patched llama.cpp on the GPU, 49 min on the RTX 5090).
3. Re-quantization: `requant_experts.py --gpu --dump=<hessians> --order=act --damp=0.3 --damp-iq1=1.0` with unsloth's
   imatrix, the type table above (132G), then layers 35-39 again at IQ3_XXS (139G).
4. Assembly: `merge_experts.py` puts the new expert tensors into unsloth UD-IQ2_XXS (rev 85f81fb) byte for byte; the
   result's Abyss pack was checked to have exactly the measured layout.

Experts with fewer than 64 calibration rows (a few per layer) fall back to plain imatrix rounding.

</details>

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

**BETA 版です。** 測ったのは 1 台だけです（RTX 5090、RAM の置き場を 60 GB に制限）。ほかの GPU や RAM 容量で試した結果を教えてもらえると助かります。

| フォルダ | エキスパート | 向き | KL（8 本） | 難しめ 10 問の品質 | 生成速度（コーディング課題） |
| --- | ---: | --- | ---: | ---: | ---: |
| `132G/` | 132.2 GB | **速さ重視**（unsloth UD-IQ1_M と同じ大きさ） | 0.194 | 77.6 | 67.5 tok/s |
| `139G/` | 139.4 GB | **品質重視**（層 35〜39 を 1 段上の IQ3_XXS に） | **0.182** | **81.4** | 57.9 tok/s |
| unsloth UD-IQ1_M（比較用。ここにはありません） | 132.2 GB | | 0.198 | 81.1 | 71.8 tok/s |

- UD-IQ1_M と同じくらいの速さで、数字が少し良いものが欲しいなら **132G**。生成が 15〜20% 遅くなってもよいので、この大きさで一番の品質が欲しいなら **139G** です。
- どちらも実際の作業のチェック（難しめのコーディング 10 問すべて正解、8K / 32K の長文から情報を探す、ツールを使う 7 往復の作業、日本語 3 問）を通っています。
- 132G の型の配置は unsloth UD-IQ1_M とまったく同じで、違うのは丸め方だけです。普通に量子化し直すと KL 0.404 と大きく崩れ、unsloth の丸めは 0.198、LDLQ は 0.194 でした。
- 6 番目のファイルは Engram の表です。unsloth のものとまったく同じで、SSD から少しずつ読むので RAM に載せる必要はありません。
- 速度の測定環境：RTX 5090（32 GB、OCuLink Gen4 x4）、Ryzen AI Max+ 395（iGPU も RAM 側のエキスパートの計算に使用）、RAM の枠は 60 GB、残りは NVMe SSD、DSpark の下書きなど普段使いの設定。難しめの 10 問の平均は 132G 58.0、139G 47.8、UD-IQ1_M 52.5 tok/s。
- 132G と UD-IQ1_M は測るたびに順位が入れ替わる程度の差（±5%）です。DDR5 デュアルチャネルで iGPU のないデスクトップでは、この数字より遅くなります。
- ほかの GPU（AI PRO R9700、V100、24 GB のカード）では、このファイルはまだ測っていません。結果を教えてもらえると助かります。
- Abyss v1.3.100 以降なら、`./setup-deepseek.sh --model v41 --download-model` の 1 行で入ります。品質重視の版は `--quant 139G` を足してください。
- llama.cpp（2026-10-29 の ac8e1694d）でも `132G` がそのまま動きました。RTX 5090 で `--n-cpu-moe 38` のとき、日本語の質問と Python の課題に正しく答え、生成は 10〜11 tok/s でした。
- ここにあるファイルは普通のモデルです。Abyss の実行時のオプションはファイルを変えません。
- 作り方は上の英語の節（How it was made）のとおりです。ライセンスは元のモデルと同じ MIT です。

速さの報告は [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4.1+LDLQ-132G+on+YOUR_GPU) に、
品質の報告（おかしな答え、別の言語が混ざった、繰り返し）は Community タブか Abyss の issue にお願いします。

### サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 32 GB + RAM 96 GB | 275.8 / 283.0 GB | 384 個すべて（132G / 139G） |
| [DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF) | GPU 32 GB + RAM 96 GB、GPU 12 GB + RAM 128 GB でも | 207.1 GB | 384 個中 256 個（JA256） |

以前のリポジトリ名は DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
