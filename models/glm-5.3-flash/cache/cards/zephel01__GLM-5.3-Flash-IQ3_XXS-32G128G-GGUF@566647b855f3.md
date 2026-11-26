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
- consumer-gpu
- local-llm
- cpu-offload
- japanese
- rtx-5090
- 32gb-vram
- 128gb-ram
language:
- en
- ja
library_name: gguf
---

![GLM-5.3-Flash IQ3_XXS: quality first, 34.6 tok/s on a 32 GB GPU + 123 GB RAM PC](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/zai-org/GLM-5.3-Flash) | [32G96G version](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | [DGX Spark version](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G128G+Q120d+on+YOUR_GPU) | 日本語は下

# GLM-5.3-Flash GGUF (for a 32 GB GPU + 128 GB RAM PC, quality first) — BETA

- **32G128G = for a PC with a 32 GB GPU (VRAM) + 128 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ3_XXS 89%, IQ4_XS 8%, Q2_K 1% (by bytes), 3.13 bits a weight on average, about IQ3_XXS.

A **quality-first** GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) for one 32 GB
GPU (RTX 5090, AI PRO R9700) plus 128 GB RAM or more. Every expert is kept (no pruning). **KL about half of unsloth's
UD-IQ2_XXS** (0.137 vs 0.265). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC (123 GB RAM, the drafter off). Real-use checks are still to do, and the file may be replaced
by a better build (the folder name will change then). **If you have a 32 GB (or 24 GB) GPU with 128 GB RAM, or an
RTX PRO 6000, please measure it** (see Feedback).

## Model files

`Q120d/GLM-5.3-Flash-Q120d-0000N-of-00004.gguf`, 4 shards, 127.9 GB. Point your engine at `-00001-of-00004.gguf`.
`SHA256SUMS` is in the folder.

| part | type |
| --- | --- |
| routed experts gate / up (layers 3-44) | IQ3_XXS |
| routed experts down | IQ3_XXS (layers 3-5, 11, 12, 42-44: IQ4_XS) |
| attention output, attention q_b, dense / shared-expert down | Q5_K |
| other attention projections, dense / shared-expert gate / up, NextN projection | Q4_K |
| router, embeddings, output, rest of the NextN (MTP) layer | unsloth UD-IQ2_XXS's bytes |

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 94.1 GB | folder S1 |
| **this repo** | 32 GB GPU + 128 GB RAM | 127.9 GB | higher quality (folder Q120d) |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark (128 GB) | 94.1 GB | same files as 32G96G |

Earlier repo name: GLM-5.3-Flash-32G128G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: speed depends on whether it fits

- Abyss keeps the most-used experts in VRAM, the rest in RAM, and reads what does not fit from the SSD each token.
  Those SSD reads dominate the time.
- Q120d is 127.9 GB. With a 60 GB RAM tier (as on a 96 GB RAM PC) about 33 GB of it stays on the SSD, and it drops to
  19.0 tok/s (Abyss v1.3.129 engine, iGPU off, no drafter).
- On our 123 GB RAM PC, with the RAM tier Abyss picks (88.8 GB), it ran at 34.6 tok/s with the SSD wait down to
  1.4 ms a token (drafter off).
- With less than 128 GB RAM, take S1 (94 GB) instead.
- The experts were re-quantized with LDLQ from unsloth's UD-Q4_K_XL, using a Hessian dump of chat transcripts in
  GLM's template (English, code and Japanese) and unsloth's imatrix.

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** (one command sets up the engine, the model, the pack and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model glm --quant 32G128G --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --model glm --quant 32G128G --download-model`.

**llama.cpp** (arch `glm5-next`, upstream `ac8e1694d` of 2026-10-29): runs as is. Checked on an RTX 5090 (32 GB) +
123 GB RAM with `llama-server -m Q120d/GLM-5.3-Flash-Q120d-00001-of-00004.gguf -c 8192` (llama.cpp's automatic
GPU/CPU split): correct answers to an English and a Japanese question (greedy). Speed not measured with llama.cpp.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 10 sequences x 1,024 tokens: English text, code and Japanese),
Abyss engine. All three rows come from the same 10-sequence run.

| | download | KL (10 seqs) | KL, Japanese | same top token |
| --- | ---: | ---: | ---: | ---: |
| **Q120d (this repo)** | **127.9 GB** | **0.1367** | **0.145** | **84.9%** |
| [S1](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) (32G96G) | 94.1 GB | 0.2706 | 0.280 | 79.1% |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | 0.2651 | 0.240 | 79.0% |

- **For careful work** where quality counts more than speed: long reasoning, hard coding, Japanese.
- **Japanese, measured separately**: on the Japanese text of the KL set, Q120d's KL is 0.145 vs 0.240 for UD-IQ2_XXS
  (about 40% lower; one 1,024-token Japanese sequence).
- The S1 card has a newer 14-sequence run (S1 0.2534, UD-IQ2_XXS 0.2643) with the router bias kept in F32. Q120d was
  not re-run that way yet, so compare within one table only.
- Real-use checks (Japanese, coding tasks, needle): not run on this file yet.

## Speed

![This file vs S1 on the same PC, two RAM tiers](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF/resolve/main/assets/speed.png)

Abyss v1.3.129, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395 (LPDDR5X), 123 GB RAM, NVMe SSD, iGPU
off (`STRATA_GLM_IGPU=0`), **no drafter**. Temperature 0, 4K prompt, 512 tokens, page cache dropped before each run,
1 run each, same session.

| RAM tier | Q120d (this repo) | S1 (32G96G) | Q120d SSD wait a token |
| --- | ---: | ---: | ---: |
| 60 GB (`STRATA_GLM_RAM_GB=60`, like a 96 GB PC) | 19.0 tok/s | 47.4 tok/s | 23.5 ms |
| set by Abyss (Q120d 88.8 GB, S1 64.6 GB) | **34.6 tok/s** | 46.9 tok/s | 1.4 ms |

- S1 stays faster on every PC we measured. Q120d buys quality, not speed.
- With the NextN drafter on, and on a real 128 GB desktop: not measured yet.

## Feedback (this is a BETA) — 128 GB testers wanted

- **Speed on your PC** (especially 32 GB / 24 GB GPU + 128 GB RAM, or RTX PRO 6000): `../Abyss-tools/abyss.sh --bench --deep`
  in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G128G+Q120d+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **32G128G = GPU（VRAM）32 GB + RAM 128 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ3_XXS 89%, IQ4_XS 8%, Q2_K 1%（容量の割合）、平均 3.13 ビットで、IQ3_XXS くらいです。

GLM-5.3-Flash の、**品質を優先した** GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 128 GB 以上の PC** 向けです。
エキスパートは削っていません。KL は unsloth の UD-IQ2_XXS の**約半分**（0.137 対 0.265）です。

**BETA です**：測ったのは 1 台の PC（RAM 123 GB、下書きなし）だけです。実際の作業での確認はまだです。
**GPU 32 GB（または 24 GB）+ RAM 128 GB の PC や、RTX PRO 6000 をお持ちの方、ぜひ測ってください。**

| | 大きさ | KL（10 本） | KL（日本語） | 同じ 1 位 |
| --- | ---: | ---: | ---: | ---: |
| **Q120d（ここ）** | **127.9 GB** | **0.1367** | **0.145** | **84.9%** |
| [S1](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF)（32G96G） | 94.1 GB | 0.2706 | 0.280 | 79.1% |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2651 | 0.240 | 79.0% |

- **速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで大きく変わります。** RAM の置き場 60 GB（RAM 96 GB の PC 相当）では 33 GB ほどが SSD に残り、
  19.0 tok/s まで落ちます。
- RAM 123 GB の PC（Abyss v1.3.129、RTX 5090、iGPU なし、下書きなし、4K のあと 512 トークン、1 回）では、Abyss が決めた
  RAM の置き場（88.8 GB）で **34.6 tok/s**、SSD 待ちは 1 トークン 1.4 ms でした。同じ条件の S1 は 46.9 tok/s です。
  RAM の置き場を 60 GB にすると、この版は 19.0 tok/s、S1 は 47.4 tok/s でした。
- S1 は測ったどの PC でもこの版より速いです。この版は速さではなく品質のための版です。RAM が 128 GB より少ない PC では S1 を使ってください。
- **日本語**：KL を測る 10 本のうち 1 本が日本語の文章で、この版は 0.145、unsloth は 0.240（約 4 割少ない）です。
- 下書きあり、本物の 128 GB のデスクトップでの速さは、まだ測っていません。
- Abyss では `./setup-deepseek.sh --model glm --quant 32G128G --download-model` で入ります。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。RTX 5090（32 GB）+ RAM 123 GB で `llama-server -m Q120d/GLM-5.3-Flash-Q120d-00001-of-00004.gguf -c 8192` のとき、英語と日本語の質問に正しく答えました（greedy）。llama.cpp での速さは測っていません。

測った結果は、`./abyss.sh --bench --deep` の `abyss-bench.txt` を、GPU、RAM、SSD と一緒に
[この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G128G+Q120d+on+YOUR_GPU) か Community タブに貼ってもらえると助かります。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 94.1 GB | S1 |
| **この版** | GPU 32 GB + RAM 128 GB | 127.9 GB | 品質重視（Q120d） |
| [GLM-5.3-Flash-DGX-Spark-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF) | DGX Spark（128 GB） | 94.1 GB | 32G96G と同じファイル |

以前のリポジトリ名は GLM-5.3-Flash-32G128G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
