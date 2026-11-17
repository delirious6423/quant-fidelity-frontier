---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
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
- 64gb-ram
language:
- en
- ja
library_name: gguf
---

![DeepSeek V4 Flash 0731 L72: 90.0 tok/s on an RTX 5090 with the RAM tier capped at 40 GB](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | [16G64G version](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | [32G96G version](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+32G64G+L72+on+YOUR_GPU) | 日本語は下

# DeepSeek V4 Flash 0731 GGUF (for a 32 GB GPU + 64 GB RAM PC) — BETA

- **32G64G = for a PC with a 32 GB GPU (VRAM) + 64 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ1_S 54%, IQ2_S 37%, IQ3_XXS 9% (by bytes), 1.92 bits a weight on average, about IQ2_XXS.

A 72.0 GB GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT)
for **one 32 GB GPU (RTX 5090, AI PRO R9700) plus 64 GB RAM**. **Every expert is kept** (all 256, no pruning). Speed and
quality are measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far (RTX 5090 32 GB), with its RAM tier capped at 40 GB under a 58 GB limit, a 64 GB PC's share. The file may be replaced by a better
build after more measurements (the folder name will change then). Reports are welcome (see Feedback).

## Model files

`L72/DeepSeek-V4-Flash-0731-L72-0000N-of-00003.gguf`, 3 shards, 72.0 GB. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the folder.

| part | type |
| --- | --- |
| routed experts gate / up (all 256 in every layer) | IQ1_S |
| routed experts down | IQ2_S (layers 36-42: IQ3_XXS) |
| attention `attn_q_b`, `attn_output_a`, `attn_output_b` | Q5_K |
| everything else | unsloth UD-Q2_K_XL's bytes |

The first shard (metadata only, 5 MB) is unsloth's file as is, so its `general.quantized_by` still says Unsloth.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 61.7 GB | smallest (folder L62) |
| **this repo** | 32 GB GPU + 64 GB RAM | 72.0 GB | folder L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 96.8 GB | highest quality (folder L97) |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark (128 GB) | 96.8 GB | same files as 32G96G |

Earlier repo name: DeepSeek-V4-Flash-0731-32G64G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: speed depends on whether the file fits

- unsloth's 96.8 GB UD-Q2_K_XL does not fit a 32 GB GPU + 64 GB RAM PC and waits for the SSD every token: 33.4 tok/s.
- L72 fits and runs **2.7x faster**: 90.0 tok/s.
- The price is quality: KL 0.314 against UD-Q2_K_XL's 0.186. In the real-use checks below it did as well as
  UD-Q2_K_XL; tell us where it does not. With 96 GB RAM or more, take
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF).
- The experts were re-quantized with LDLQ from unsloth's UD-Q8_K_XL (original MXFP4 experts), using a Hessian dump of
  V4's own chat answers (241 prompts, 104 of them Japanese) plus general text, and our own imatrix. Details: the
  [32G96G card](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF).

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.120 or later (one command sets up the engine, the model, the pack
and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --quant 32G64G --no-dspark --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --quant 32G64G --no-dspark --download-model`.

- The DSpark drafter is not measured at this size. On 32G96G it made decoding slower (it takes ~6.5 GB of VRAM), so
  start without it (`--no-dspark`).
- **llama.cpp**: not checked yet.

## Quality

KL against unsloth's UD-Q8_K_XL, 12 sequences x 1,024 tokens (8 English text, 1 code, 3 Japanese), Abyss engine.

| | download | KL (12 seqs) | KL, Japanese (3) | same top token |
| --- | ---: | ---: | ---: | ---: |
| **L72 (this repo)** | **72.0 GB** | **0.314** | **0.331** | **76.9%** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | 0.186 | 0.219 | 81.5% |
| [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) (32G96G) | 96.8 GB | 0.150 | 0.145 | 84.1% |

Real use (Abyss v1.3.120, RTX 5090, 32G64G limits):

| | L72 | UD-Q2_K_XL |
| --- | ---: | ---: |
| Abyss `--check` (Japanese, three coding tasks, tool call, 8K-31K needle, 11-turn conversation) | 11 / 11 | 11 / 11 |
| 10 Japanese answers that stayed in Japanese | 9 / 10 | 8 / 10 |
| wrong facts in those answers | 0 | 0 |
| 5 coding tasks with tests | 5 / 5 | 5 / 5 |
| 8K needle | found | - |

V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct). unsloth's GGUF and the
original weights do it too (original: 6 of 24 answers on 8 Japanese prompts x 3 seeds; L97: 11 of 24). With the system
prompt "日本語で答えてください。" L97 switched in 1 of 24, so put "answer in Japanese" in the system prompt.

## Speed

![Decode speed vs unsloth UD-Q2_K_XL with 32 GB GPU + 64 GB RAM limits](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF/resolve/main/assets/speed.png)

Abyss v1.3.120 on an RTX 5090 (32 GB) + Ryzen AI Max+ 395 PC, limited to what a 64 GB RAM PC has: RAM tier 40 GB
(`STRATA_GLM_RAM_GB=40`) under a 58 GB memory limit. Temperature 0, 512 tokens after a 4K / 16K prompt, page cache
dropped before each run, median of 3.

| | download | 4K prompt | 16K prompt |
| --- | ---: | ---: | ---: |
| **L72 (this repo)** | 72.0 GB | **90.0 tok/s** | **86.4 tok/s** |
| unsloth UD-Q2_K_XL | 96.8 GB | 33.4 tok/s | 34.0 tok/s |

This PC's RAM (LPDDR5X) is about 3x a usual DDR5 desktop's, so expect lower numbers on a usual PC.

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep`, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+32G64G+L72+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **32G64G = GPU（VRAM）32 GB + RAM 64 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ1_S 54%, IQ2_S 37%, IQ3_XXS 9%（容量の割合）、平均 1.92 ビットで、IQ2_XXS くらいです。

DeepSeek V4 Flash 0731 の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 64 GB の PC** 向けです。
エキスパートは 256 個すべて残しています。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです（RTX 5090 で、RAM の置き場を 40 GB・全体 58 GB に制限して、64 GB の PC と同じくらいにして測りました）。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | 32G64G、4K | 32G64G、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L72（ここ）** | **72.0 GB** | **0.314** | **0.331** | **76.9%** | **90.0 tok/s** | **86.4 tok/s** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | 33.4 tok/s | 34.0 tok/s |

- **速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで決まります。** unsloth の 96.8 GB は GPU 32 GB + RAM 64 GB に
  載りきらず、毎トークン SSD を待ちます（33.4 tok/s）。L72 は全部載るので **2.7 倍速い**です（90.0 tok/s）。
- そのかわり、品質は unsloth の UD-Q2_K_XL より元のモデルから離れています（KL 0.314 対 0.186）。実際の作業での確認では
  同じくらいの結果でしたが、うまくいかない場面があれば教えてください。RAM が 96 GB 以上あるなら
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) をおすすめします。
- **実際の作業での確認**（Abyss v1.3.120）：`--check` は 11 項目すべて合格。日本語 10 問のうち 9 問が日本語のまま、事実の誤りは 0 件、
  コードは 5 問すべて合格、8K の needle も見つかりました（unsloth は同じ条件で日本語 8/10、コード 5/5）。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも、元のモデルでも起きます（日本語 8 問 × seed 3 つの 24 回答で、元のモデル 6 回、L97 11 回）。システムプロンプトに「日本語で答えてください。」を入れると、L97 で 1 回に減りました。
- DSpark（下書きモデル）はこの大きさでは測っていません。32G96G では遅くなったので、なしで始めてください（`--no-dspark`）。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 の約 3 倍）ので、ふつうの PC では遅くなる見込みです。
- Abyss では `./setup-deepseek.sh --quant 32G64G --no-dspark --download-model` で入ります（v1.3.120 以降）。
- llama.cpp ではまだ確かめていません。

感想や速さの報告は、Community タブか [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+32G64G+L72+on+YOUR_GPU) にお願いします。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 61.7 GB | いちばん小さい（L62） |
| **この版** | GPU 32 GB + RAM 64 GB | 72.0 GB | L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 96.8 GB | 品質がいちばん上（L97） |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark（128 GB） | 96.8 GB | 32G96G と同じファイル |

以前のリポジトリ名は DeepSeek-V4-Flash-0731-32G64G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
