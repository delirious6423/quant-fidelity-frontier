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
- 16gb-vram
- 64gb-ram
language:
- en
- ja
library_name: gguf
---

![DeepSeek V4 Flash 0731 L62: 84.5 tok/s on an RTX 5090 capped at 16 GB VRAM + 40 GB RAM tier](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | [32G64G version](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | [32G96G version](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+16G64G+L62+on+YOUR_GPU) | 日本語は下

# DeepSeek V4 Flash 0731 GGUF (for a 16 GB GPU + 64 GB RAM PC) — BETA

- **16G64G = for a PC with a 16 GB GPU (VRAM) + 64 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ1_S 64%, IQ1_M 36% (by bytes), 1.62 bits a weight on average, about IQ1_S.

A 61.7 GB GGUF of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT)
for **one 16 GB GPU (RTX 5060 Ti 16 GB, RTX 4080, RX 9070 XT ...) plus 64 GB RAM**. **Every expert is kept** (all 256,
no pruning). Speed and quality are measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC, an RTX 5090 limited to 16 GB of VRAM and to a 64 GB PC's RAM share, not on a real 16 GB
card yet. The file may be replaced by a better build after more measurements (the folder name will change then).
Reports from real 16 GB cards are welcome (see Feedback).

## Model files

`L62/DeepSeek-V4-Flash-0731-L62-0000N-of-00003.gguf`, 3 shards, 61.7 GB. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the folder.

| part | type |
| --- | --- |
| routed experts gate / up (all 256 in every layer) | IQ1_S |
| routed experts down | IQ1_M |
| attention `attn_q_b`, `attn_output_a`, `attn_output_b` | Q5_K |
| everything else | unsloth UD-Q2_K_XL's bytes |

The first shard (metadata only, 5 MB) is unsloth's file as is, so its `general.quantized_by` still says Unsloth.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 16 GB GPU + 64 GB RAM | 61.7 GB | smallest (folder L62) |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | 32 GB GPU + 64 GB RAM | 72.0 GB | folder L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 96.8 GB | highest quality (folder L97) |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark (128 GB) | 96.8 GB | same files as 32G96G |

Earlier repo name: DeepSeek-V4-Flash-0731-16G64G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: it fits, so it does not wait for the SSD

- unsloth's 96.8 GB UD-Q2_K_XL does not fit a 16 GB GPU + 64 GB RAM PC and waits for the SSD: 21.3 tok/s.
- L62 fits and runs **4x faster** on the same limits: 84.5 tok/s.
- The price is quality: KL 0.407 against UD-Q2_K_XL's 0.186. In the real-use checks below it did about as well as
  UD-Q2_K_XL; tell us where it does not.
- With 32 GB of VRAM take [L72](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF); with
  96 GB RAM [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF).
- The experts were re-quantized with LDLQ from unsloth's UD-Q8_K_XL (original MXFP4 experts), using a Hessian dump of
  V4's own chat answers (241 prompts, 104 of them Japanese) plus general text, and our own imatrix. Details: the
  [32G96G card](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF).

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.120 or later (one command sets up the engine, the model, the pack
and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --quant 16G64G --no-dspark --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --quant 16G64G --no-dspark --download-model`.

- The DSpark drafter (~6.5 GB of VRAM) is not measured at this size and is not recommended on a 16 GB card, hence
  `--no-dspark`.
- **llama.cpp**: not checked yet.

## Quality

KL against unsloth's UD-Q8_K_XL, 12 sequences x 1,024 tokens (8 English text, 1 code, 3 Japanese), Abyss engine.

| | download | KL (12 seqs) | KL, Japanese (3) | same top token |
| --- | ---: | ---: | ---: | ---: |
| **L62 (this repo)** | **61.7 GB** | **0.407** | **0.435** | **73.7%** |
| unsloth UD-Q2_K_XL (reference) | 96.8 GB | 0.186 | 0.219 | 81.5% |

Real use (Abyss v1.3.120, RTX 5090 with 16G64G limits):

| | L62 | UD-Q2_K_XL |
| --- | ---: | ---: |
| Abyss `--check` (Japanese, three coding tasks, tool call, 8K-31K needle, 11-turn conversation) | 10 / 11 | 11 / 11 |
| 10 Japanese answers that stayed in Japanese | 9 / 10 | 9 / 10 |
| wrong facts in those answers | 0 | 0 |
| 5 coding tasks with tests | 5 / 5 | 5 / 5 |
| 8K needle | found | - |

- The one `--check` miss: a Japanese explanation answered in Chinese.
- V4 0731 sometimes answers a Japanese question in Chinese (the answer itself is correct). unsloth's GGUF and the
  original weights do it too (original: 6 of 24 answers on 8 Japanese prompts x 3 seeds; L97: 11 of 24). With the
  system prompt "日本語で答えてください。" L97 switched in 1 of 24, so put "answer in Japanese" in the system prompt.

## Speed

![Decode speed vs unsloth UD-Q2_K_XL with 16 GB GPU + 64 GB RAM limits](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF/resolve/main/assets/speed.png)

Abyss v1.3.120 on an RTX 5090 + Ryzen AI Max+ 395 PC, limited to 16 GB of VRAM (`STRATA_GLM_VRAM_GB=16`) and a 64 GB
PC's RAM share: RAM tier 40 GB (`STRATA_GLM_RAM_GB=40`) under a 58 GB memory limit. Temperature 0, 512 tokens after a
4K / 16K prompt, page cache dropped before each run, median of 3.

| | download | 4K prompt | 16K prompt |
| --- | ---: | ---: | ---: |
| **L62 (this repo)** | 61.7 GB | **84.5 tok/s** | **89.5 tok/s** |
| unsloth UD-Q2_K_XL | 96.8 GB | 21.3 tok/s | 19.6 tok/s |

- A real 16 GB card has less memory bandwidth than a 5090, and this PC's RAM (LPDDR5X) is about 3x a usual DDR5
  desktop's. Expect lower numbers on a usual 16 GB PC.
- A 24 GB GPU + 64 GB RAM is not measured; L62 fits there with room to spare.

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep`, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+16G64G+L62+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: DeepSeek [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT). Source
GGUFs: [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Quantized
with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **16G64G = GPU（VRAM）16 GB + RAM 64 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ1_S 64%, IQ1_M 36%（容量の割合）、平均 1.62 ビットで、IQ1_S くらいです。

DeepSeek V4 Flash 0731 の GGUF です。**GPU 16 GB + RAM 64 GB の PC** 向けです。エキスパートは 256 個すべて残しています。
速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは、RTX 5090 の VRAM を 16 GB に、RAM を 64 GB の PC と同じくらいに制限した 1 台だけです。
本物の 16 GB の GPU ではまだ測っていません。報告をいただけるととても助かります。

| | 大きさ | KL（12 本） | KL（日本語 3 本） | 同じ 1 位 | 16G64G、4K | 16G64G、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L62（ここ）** | **61.7 GB** | **0.407** | **0.435** | **73.7%** | **84.5 tok/s** | **89.5 tok/s** |
| unsloth UD-Q2_K_XL（比べる相手） | 96.8 GB | 0.186 | 0.219 | 81.5% | 21.3 tok/s | 19.6 tok/s |

- unsloth の 96.8 GB は GPU 16 GB + RAM 64 GB に載りきらず、SSD を待ちます（21.3 tok/s）。L62 は全部載るので、この PC では
  **約 4 倍速い**です（84.5 tok/s）。
- そのかわり、品質は元のモデルから離れています（KL 0.407 対 0.186）。実際の作業での確認では unsloth と同じくらいの
  結果でしたが、うまくいかない場面があれば教えてください。GPU が 32 GB なら
  [L72](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF)、RAM が 96 GB なら
  [L97](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) をおすすめします。
- **実際の作業での確認**（Abyss v1.3.120）：`--check` は 11 項目のうち 10 項目が合格（落ちた 1 つは、日本語の説明を中国語で答えたもの）。日本語 10 問のうち 9 問が日本語のまま、事実の誤りは 0 件、
  コードは 5 問すべて合格、8K の needle も見つかりました（unsloth は同じ条件で日本語 9/10、コード 5/5）。
- V4 0731 は、日本語で聞いても中国語で答えることがあります（中身は正しいです）。unsloth の GGUF でも、元のモデルでも起きます（日本語 8 問 × seed 3 つの 24 回答で、元のモデル 6 回、L97 11 回）。システムプロンプトに「日本語で答えてください。」を入れると、L97 で 1 回に減りました。
- GPU 24 GB + RAM 64 GB は測っていませんが、L62 は余裕を持って載ります。
- DSpark（下書きモデル、VRAM を約 6.5 GB 使う）は、16 GB の GPU ではおすすめしません。
- 本物の 16 GB の GPU は 5090 よりメモリが遅く、測った PC の RAM は速い（ふつうの DDR5 の約 3 倍）ので、ふつうの PC では
  遅くなる見込みです。
- Abyss では `./setup-deepseek.sh --quant 16G64G --no-dspark --download-model` で入ります（v1.3.120 以降）。
- llama.cpp ではまだ確かめていません。

感想や速さの報告は、Community タブか [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+DeepSeek-V4-Flash-0731+16G64G+L62+on+YOUR_GPU) にお願いします。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 16 GB + RAM 64 GB | 61.7 GB | いちばん小さい（L62） |
| [DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF) | GPU 32 GB + RAM 64 GB | 72.0 GB | L72 |
| [DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 96.8 GB | 品質がいちばん上（L97） |
| [DeepSeek-V4-Flash-0731-DGX-Spark-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-DGX-Spark-GGUF) | DGX Spark（128 GB） | 96.8 GB | 32G96G と同じファイル |

以前のリポジトリ名は DeepSeek-V4-Flash-0731-16G64G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
