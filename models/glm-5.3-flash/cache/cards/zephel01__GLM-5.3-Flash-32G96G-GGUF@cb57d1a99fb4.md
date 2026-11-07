---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, imatrix, moe, abyss, dgx-spark, gb10]
---

# GLM-5.3-Flash GGUF (for a 32 GB GPU + 96 GB RAM PC, and DGX Spark) — BETA

GGUFs of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) sized for **one 32 GB GPU
(RTX 5090, AI PRO R9700) plus 96 GB RAM**, and for **one DGX Spark (128 GB unified memory)**. Every expert is kept
(no pruning). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA**: measured on one PC so far; real-use checks and llama.cpp loading are still in progress and this card will be
updated with them. The file may be replaced by a better build after more measurements (the folder name will change
then). Reports are very welcome (Community tab, or [Abyss issues](https://github.com/zephel01/Abyss/issues)).

**Japanese, measured separately**: our KL set includes Japanese text and we report it on its own. On it S1 is a little
behind unsloth's UD-IQ2_XXS (KL 0.280 vs 0.240, one 1,024-token Japanese sequence). A build with more Japanese in its
calibration is planned and may replace S1. Best Japanese now: the quality-first
[32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF) (Japanese KL 0.145).
**日本語は別に測っています**：S1 の日本語の KL は 0.280 で、unsloth の UD-IQ2_XXS（0.240）より少し悪い結果です。日本語を
強めた版を作って入れ替える予定です。いま日本語がいちばん良いのは [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF)（0.145）です。

日本語の説明は下にあります。

## Speed depends on whether it fits

**Fast when the whole file fits in GPU memory + RAM; slow when part of it has to be read from the SSD.** Abyss keeps
the most-used experts in VRAM, the rest in RAM, and reads what does not fit from the SSD each token - and those SSD
reads dominate the time. S1 is 94.1 GB so that it fits a 32 GB GPU + 96 GB RAM (75.7 tok/s); unsloth's 101.8 GB
UD-IQ2_XXS does not quite fit and waits for the SSD (53.3 tok/s), and our 127.9 GB quality-first build drops to
23 tok/s on the same PC. With less RAM (e.g. 64 GB) part of S1 is read from the SSD and it is slower; with 128 GB RAM
or more, the quality-first [32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF) fits too.

## Which one

| | download | KL (10 seqs) | KL, Japanese | same top token | RTX 5090, 4K prompt | RTX 5090, 16K prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`S1/`](./S1) (here)** | **94.1 GB** | 0.2706 | 0.280 | 79.1% | **75.7 tok/s** | **69.8 tok/s** |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | 0.2651 | 0.240 | 79.0% | 53.3 tok/s | 50.5 tok/s |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF) (32G128G) | 127.9 GB | 0.1367 | 0.145 | 84.9% | 23.2 tok/s (SSD-bound here) | 21.1 tok/s |

- **S1** is about the same quality as unsloth's UD-IQ2_XXS (KL +0.006) and **42% faster on a 32 GB GPU + 96 GB RAM
  PC**: at 94 GB it fits in VRAM + RAM, so it almost never waits for the SSD (0.3 ms a token against 3.2 ms).
- A larger, higher-quality file (KL about half) is being evaluated; it needs more than 96 GB RAM to be fast.

## Files

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards; point your engine at `-00001-of-00004.gguf`. `SHA256SUMS` in the
folder.

- Routed experts (layers 3-44), re-quantized with LDLQ from unsloth's UD-Q4_K_XL using a Hessian dump of chat
  transcripts in GLM's template (English, code and Japanese) and unsloth's imatrix:
  gate / up IQ2_XXS (layer 11 IQ2_S), down IQ2_S (layers 11, 12, 44 IQ3_XXS).
- Dense layers: attention output, attention q_b and the dense / shared-expert down Q5_K; the other attention projections,
  dense / shared-expert gate / up and the NextN projection Q4_K.
- Router, embeddings, output and the rest of the NextN (MTP) layer: unsloth UD-IQ2_XXS's bytes.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 10 sequences x 1,024 tokens: English text, code and Japanese), Abyss
engine: see the table above. Real-use checks (Japanese, coding tasks, 8K needle) and a llama.cpp run: **in progress,
results will be added here**.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.117, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395
(LPDDR5X, about 3x the bandwidth of a usual dual-channel DDR5 desktop), RAM tier 60 GB (`STRATA_GLM_RAM_GB=60`), NVMe
SSD. Temperature 0, 512 tokens, page cache dropped before each run, median of 3 (4K) / 1 run (16K).

- On a usual DDR5 desktop the part computed on the CPU side is slower, so expect lower numbers; reports welcome.
- **DGX Spark**: everything fits in its memory; we expect about 20-21 tok/s (unsloth's UD-IQ2_XXS measured 19.3 tok/s
  on a Spark; S1 reads ~8% less a token). A Spark measurement will be added.
- Other GPUs are not measured yet.

## Feedback (this is a BETA)

- **Speed on your PC**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
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

GLM-5.3-Flash の GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 96 GB の PC** と、**DGX Spark 1 台**向けです。
エキスパートは削っていません。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で測りました。

**BETA です**：測ったのは 1 台の PC だけです。実際の作業での確認（日本語、コード、8K needle）と、llama.cpp で読めるかの
確認はまだ途中で、終わりしだいここに書き足します。測り直した結果によっては、もっと良い版に入れ替えることがあります
（そのときはフォルダの名前を変えます）。

**速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで大きく変わります。全部載れば速く、載りきらずに SSD から読む分が
あると遅くなります。** S1 は、GPU 32 GB + RAM 96 GB に全部載るように 94.1 GB にしました（75.7 tok/s）。同じ PC で、少し
はみ出す unsloth の 101.8 GB は SSD を待って 53.3 tok/s、品質優先の 127.9 GB の版は 23 tok/s です。RAM が少ない PC
（例：64 GB）では S1 も一部を SSD から読むので遅くなります。RAM が 128 GB 以上あれば、品質優先の
[32G128G](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF) も全部載ります。

| | 大きさ | KL（10 本） | KL（日本語） | 同じ 1 位 | RTX 5090、4K | RTX 5090、16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | 0.2706 | 0.280 | 79.1% | **75.7 tok/s** | **69.8 tok/s** |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2651 | 0.240 | 79.0% | 53.3 tok/s | 50.5 tok/s |
| [Q120d](https://huggingface.co/zephel01/GLM-5.3-Flash-32G128G-GGUF)（32G128G） | 127.9 GB | 0.1367 | 0.145 | 84.9% | 23.2 tok/s（この PC では SSD 待ち） | 21.1 tok/s |

- **日本語**：KL を測る 10 本のうち 1 本が日本語の文章で、その KL を別の列に出しています。S1 は日本語では unsloth より
  少し悪く（0.280 対 0.240）、品質優先の Q120d はずっと良い（0.145）結果です。日本語の文章を増やして測り直すのと、
  実際の質問での確認は、これから行います。

- 品質は unsloth の UD-IQ2_XXS とほぼ同じ（KL +0.006）で、GPU 32 GB + RAM 96 GB の PC では **42% 速い**です。94 GB だと
  VRAM と RAM にほぼ全部載るので、SSD をほとんど待ちません。
- 測った PC は RAM が速い（LPDDR5X、ふつうの DDR5 のデスクトップの約 3 倍）ので、ふつうの PC では CPU 側の分だけ遅くなる
  見込みです。
- DGX Spark では、全部メモリに載ります。見込みは 20〜21 tok/s です（unsloth の UD-IQ2_XXS の Spark での実測 19.3 tok/s から、
  1 トークンで読む量の差で見積もりました）。実測が取れたら書き足します。
- 品質を上げた大きめの版（KL が約半分）も評価しています。速く動かすには RAM が 96 GB より多く必要です。

感想や速さの報告は、Community タブか [Abyss の issue](https://github.com/zephel01/Abyss/issues) にお願いします。
`./abyss.sh --bench --deep` の結果（`abyss-bench.txt`）を、GPU、RAM、SSD と一緒に貼ってもらえると助かります。
