---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- imatrix
- moe
- abyss
- dgx-spark
- gb10
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

![GLM-5.3-Flash IQ2_XS: 47.4 tok/s on an RTX 5090 with the RAM tier capped at 60 GB](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/zai-org/GLM-5.3-Flash) | [32G128G version](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | [DGX Spark version](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G96G+S1+on+YOUR_GPU) | 日本語は下

# GLM-5.3-Flash GGUF (for a 32 GB GPU + 96 GB RAM PC, and DGX Spark) — BETA

- **32G96G = for a PC with a 32 GB GPU (VRAM) + 96 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1% (by bytes), 2.26 bits a weight on average, about IQ2_XS.

GGUFs of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) for **one 32 GB GPU (RTX 5090,
AI PRO R9700) plus 96 GB RAM**, and for **one DGX Spark (128 GB unified memory)**. Every expert is kept (no pruning).
Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far (RTX 5090 32 GB, RAM tier capped at 60 GB). The file may be replaced by a better build after more measurements (the folder
name will change then). Reports are welcome (see Feedback).

## Model files

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards, 94.1 GB. Point your engine at `-00001-of-00004.gguf`.
`SHA256SUMS` is in the folder.

| part | type |
| --- | --- |
| routed experts gate / up (layers 3-44) | IQ2_XXS (layer 11: IQ2_S) |
| routed experts down | IQ2_S (layers 11, 12, 44: IQ3_XXS) |
| attention output, attention q_b, dense / shared-expert down | Q5_K |
| other attention projections, dense / shared-expert gate / up, NextN projection | Q4_K |
| router, embeddings, output, rest of the NextN (MTP) layer | unsloth UD-IQ2_XXS's bytes |

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 32 GB GPU + 96 GB RAM | 94.1 GB | folder S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | 32 GB GPU + 128 GB RAM | 127.9 GB | higher quality (folder Q120d) |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark (128 GB) | 94.1 GB | same files as 32G96G |

Earlier repo name: GLM-5.3-Flash-32G96G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: speed depends on whether it fits

- Abyss keeps the most-used experts in VRAM, the rest in RAM, and reads what does not fit from the SSD each token.
  Those SSD reads dominate the time.
- S1 is 94.1 GB so that it fits a 32 GB GPU + 96 GB RAM. With a 60 GB RAM tier it decodes at 47.4 tok/s and almost
  never waits for the SSD (0.3 ms a token).
- Our 127.9 GB quality-first build on the same 60 GB tier: 19.0 tok/s, waiting 23.5 ms a token for the SSD.
- With less RAM (e.g. 64 GB), part of S1 is read from the SSD and it is slower. With 128 GB RAM or more, the
  quality-first [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) fits too.
- The experts were re-quantized with LDLQ from unsloth's UD-Q4_K_XL, using a Hessian dump of chat transcripts in
  GLM's template (English, code and Japanese) and unsloth's imatrix.

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** (one command sets up the engine, the model, the pack and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model glm --quant 32G96G --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --model glm --quant 32G96G --download-model`.

**llama.cpp** (arch `glm5-next`, upstream `ac8e1694d` of 2026-10-29): runs as is. Checked on an RTX 5090 (32 GB) +
123 GB RAM with `llama-server -m S1/GLM-5.3-Flash-S1-00001-of-00004.gguf -c 32768 -b 4096 -ub 4096` (llama.cpp's
automatic GPU/CPU split): correct answers to an English and a Japanese question (greedy). Speed: see below.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 14 sequences x 1,024 tokens: 8 English text, 1 code, 5 Japanese),
Abyss engine, re-measured 2026-11-08 with the router's expert bias (`exp_probs_b`) kept in F32 as stored in the GGUF.

| | download | KL (14 seqs) | KL, Japanese (5) | same top token |
| --- | ---: | ---: | ---: | ---: |
| **S1 (this repo)** | **94.1 GB** | **0.2534** | **0.251** | **77.3%** |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | 0.2643 | 0.257 | 76.6% |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) (32G128G) | 127.9 GB | 0.1367 * | 0.145 * | 84.9% * |

\* Q120d: the older 10-sequence measurement; not directly comparable with the 14-sequence numbers.

- S1 is a little closer to the original than UD-IQ2_XXS: better in English, a little behind on the one code sequence.
- **Japanese, measured separately**: on the 5 Japanese sequences S1 is a little closer to the original than
  UD-IQ2_XXS (0.251 vs 0.257). A build with more Japanese in its calibration did not do better on general Japanese
  text, so S1 stays. Best Japanese now: the quality-first
  [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF).
- Earlier numbers on this card (S1 0.2706, UD-IQ2_XXS 0.2651, 10 sequences) came from an Abyss pack that rounded the
  router bias to BF16, which hurt S1 more. The GGUF files did not change.
- Real-use checks: Abyss `--check` passes 12 of 12 (Japanese, three coding tasks, tool call, 8K-63K needle, 11-turn
  conversation) on an RTX 5090.

## Speed

![Decode speed, this file vs the 32G128G file on the same PC](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF/resolve/main/assets/speed.png)

Abyss engine v1.3.129, 2026-11-11, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395 (LPDDR5X, about
3x the bandwidth of a usual dual-channel DDR5 desktop), 123 GB RAM, NVMe SSD. **iGPU off (`STRATA_GLM_IGPU=0`), no
drafter.** A 4K prompt (4,095 tokens) + 512 tokens, greedy, 1 run each.

| | download | RAM tier | decode | 4K prompt | SSD wait a token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **S1 (this repo)** | 94.1 GB | 60 GB | **47.4 tok/s** | 669 tok/s | 0.3 ms |
| **S1 (this repo)** | 94.1 GB | auto (64.6 GB) | **46.9 tok/s** | 671 tok/s | 0 ms |
| Q120d (32G128G) | 127.9 GB | 60 GB | 19.0 tok/s | 382 tok/s | 23.5 ms |
| Q120d (32G128G) | 127.9 GB | auto (88.8 GB) | 34.6 tok/s | 461 tok/s | 1.4 ms |

- One run each, so read about ±3% as noise.

- On a usual DDR5 desktop the part computed on the CPU side is slower, so expect lower numbers.
- **DGX Spark** (2026-11-07, Abyss v1.3.119): everything fits in its memory; **27.4 tok/s** (21.4 without drafting;
  unsloth's UD-IQ2_XXS 24.2 / 19.8). See [the DGX Spark repo](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF).
- Other GPUs are not measured yet.

**Abyss and llama.cpp on this file** (2026-11-26, same PC, iGPU off; Abyss v1.3.211, RAM tier 60 GB; llama.cpp
`ac8e1694d`, `llama-server -c 32768 -b 4096 -ub 4096 -t 16` with its own GPU/CPU fit; the same prompt tokens, greedy,
512 tokens, median of 3):

| | Abyss | llama.cpp |
| --- | ---: | ---: |
| decode, short prompt | **61.5 tok/s** | 25.5 tok/s |
| decode after a 4K prompt | **54.5 tok/s** | 25.6 tok/s |
| 4K prompt | **729 tok/s** | 381 tok/s |

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G96G+S1+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.
- **What you would rather have**: a smaller one for 64 GB RAM, a faster one for DGX Spark, other GPUs.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **32G96G = GPU（VRAM）32 GB + RAM 96 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1%（容量の割合）、平均 2.26 ビットで、IQ2_XS くらいです。

GLM-5.3-Flash の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 96 GB の PC** と、**DGX Spark 1 台**向けです。
エキスパートは削っていません。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです（RTX 5090、RAM の置き場を 60 GB に制限）。
測り直した結果によっては、もっと良い版に入れ替えることがあります（そのときはフォルダの名前を変えます）。

| | 大きさ | KL（14 本） | KL（日本語 5 本） | 同じ 1 位 | 生成（RTX 5090、RAM 60 GB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | **0.2534** | **0.251** | **77.3%** | **47.4 tok/s** |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2643 | 0.257 | 76.6% | - |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF)（32G128G） | 127.9 GB | 0.1367 * | 0.145 * | 84.9% * | 19.0 tok/s（SSD 待ち） |

\* Q120d は前の 10 本での測り方の値で、14 本の値とはそのまま比べられません。

- **速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで大きく変わります。** S1 は GPU 32 GB + RAM 96 GB に全部載るように
  94.1 GB にしました。RAM の置き場を 60 GB に制限して 47.4 tok/s、SSD 待ちは 1 トークン 0.3 ms です。同じ置き場で 127.9 GB の
  32G128G 版は 19.0 tok/s（SSD 待ち 23.5 ms）です。
- 速さは Abyss v1.3.129 のエンジン、2026-11-11、RTX 5090、**iGPU なし・下書きなし**、4K プロンプト + 512 トークン、各 1 回で測りました。
- RAM が少ない PC（例：64 GB）では S1 も一部を SSD から読むので遅くなります。RAM が 128 GB 以上あれば、品質優先の
  [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) も全部載ります。
- **品質**：S1 は unsloth の UD-IQ2_XXS より、元のモデルに少し近い結果です（英語では良く、1 本だけのコードでは少し負けます）。
  前にこのページに書いた値（S1 0.2706、UD 0.2651、10 本）は、Abyss がルーターの補正値を BF16 に丸めていたときのものです。
  GGUF のファイルは変わっていません。
- **日本語**：日本語 5 本の KL は S1 0.251、unsloth 0.257 です。日本語を強めた版も試しましたが、一般的な日本語の文章では
  差が出なかったので、S1 のままにします。いま日本語がいちばん良いのは
  [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) です。
- **実際の作業での確認**：Abyss の `--check` は RTX 5090 で 12 項目すべて合格です（日本語、コード 3 問、ツール呼び出し、
  8K〜63K の needle、11 ターンの会話）。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 のデスクトップの約 3 倍）ので、ふつうの PC では CPU 側の分だけ遅くなる見込みです。
- DGX Spark では、全部メモリに載ります。2026-11-07 の実測（Abyss v1.3.119）で **27.4 tok/s**（下書きなしで 21.4。unsloth の
  UD-IQ2_XXS は 24.2 / 19.8）です。
- Abyss では `./setup-deepseek.sh --model glm --quant 32G96G --download-model` で入ります。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。RTX 5090（32 GB）+ RAM 123 GB で、英語と日本語の質問に正しく答えました（greedy）。
- **同じファイルでの Abyss と llama.cpp**（2026-11-26、同じ PC、iGPU なし。Abyss v1.3.211 は RAM の置き場 60 GB、llama.cpp は `llama-server -c 32768 -b 4096 -ub 4096 -t 16` で GPU と CPU の分け方は llama.cpp に任せる。同じプロンプトのトークン、greedy、512 トークン、3 回の中央値）：
  生成（短いプロンプト）Abyss 61.5 / llama.cpp 25.5 tok/s、4K のあとの生成 54.5 / 25.6 tok/s、4K のプロンプト 729 / 381 tok/s。

感想や速さの報告は、Community タブか [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G96G+S1+on+YOUR_GPU) にお願いします。
`./abyss.sh --bench --deep` の結果（`abyss-bench.txt`）を、GPU、RAM、SSD と一緒に貼ってもらえると助かります。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 32 GB + RAM 96 GB | 94.1 GB | S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | GPU 32 GB + RAM 128 GB | 127.9 GB | 品質重視（Q120d） |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark（128 GB） | 94.1 GB | 32G96G と同じファイル |

以前のリポジトリ名は GLM-5.3-Flash-32G96G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
