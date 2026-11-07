---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags: [gguf, ldlq, imatrix, moe, abyss]
---

# GLM-5.3-Flash GGUF (for a 32 GB GPU + 128 GB RAM PC, quality first) — BETA

A **quality-first** GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) for **one
32 GB GPU (RTX 5090, AI PRO R9700) plus 128 GB RAM or more**. Every expert is kept (no pruning). **KL about half of
unsloth's UD-IQ2_XXS** (0.137 vs 0.265). Measured in [Abyss](https://github.com/zephel01/Abyss).

**BETA, and the speed on a 128 GB PC is not measured yet**: our PC has 96 GB RAM, where 33 GB of this file stays on
the SSD. **If you have a 32 GB (or 24 GB) GPU with 128 GB RAM, or an RTX PRO 6000, please measure it** (see Feedback).
Real-use checks and llama.cpp loading are in progress; the file may be replaced by a better build after more
measurements (the folder name will change then).

**Japanese, measured separately**: on the Japanese text of our KL set, Q120d's KL is **0.145 vs 0.240** for unsloth's
UD-IQ2_XXS (about 40% lower; one 1,024-token Japanese sequence). More Japanese sequences and real-use Japanese checks are
in progress. **日本語も別に測っています**：日本語の文章での KL は **0.145**（unsloth の UD-IQ2_XXS は 0.240、約 4 割少ない）。

日本語の説明は下にあります。

## Speed depends on whether it fits

**Fast when the whole file fits in GPU memory + RAM; slow when part of it has to be read from the SSD.** Abyss keeps
the most-used experts in VRAM, the rest in RAM, and reads what does not fit from the SSD each token - and those SSD
reads dominate the time. Q120d is 127.9 GB: with a 32 GB GPU and **128 GB RAM** it fits (estimate 40-45 tok/s); with
**96 GB RAM**, 33 GB stays on the SSD and it drops to 23 tok/s. If your PC has less than 128 GB, take S1 (94 GB) instead.

## Which one

| | download | KL (10 seqs) | KL, Japanese | same top token | RTX 5090 + 96 GB RAM, 4K | RTX 5090 + 128 GB RAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **[`Q120d/`](./Q120d) (here)** | **127.9 GB** | **0.1367** | **0.145** | **84.9%** | 23.2 tok/s (SSD-bound) | **~40-45 tok/s (estimate)** |
| [S1](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF) (32G96G) | 94.1 GB | 0.2706 | 0.280 | 79.1% | 75.7 tok/s | (same or faster) |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | 0.2651 | 0.240 | 79.0% | 53.3 tok/s | - |

- **Q120d**: for careful work where quality counts more than speed - long reasoning, hard coding, Japanese.
- **S1** ([32G96G](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF)): fastest, fits 96 GB RAM, same quality as
  unsloth's UD-IQ2_XXS. Also the one for DGX Spark (Q120d does not fit its memory).
- Why 128 GB: on our 96 GB PC, Q120d reads 9.5 experts a token from the SSD (17 ms of its 43 ms a token). With 128 GB
  everything fits in VRAM + RAM, and we estimate 22-25 ms a token (40-45 tok/s). A 24 GB GPU + 128 GB RAM should read
  only a little from the SSD.

## Files

`Q120d/GLM-5.3-Flash-Q120d-0000N-of-00004.gguf`, 4 shards; point your engine at `-00001-of-00004.gguf`. `SHA256SUMS` in
the folder.

- Routed experts (layers 3-44), re-quantized with LDLQ from unsloth's UD-Q4_K_XL using a Hessian dump of chat
  transcripts in GLM's template (English, code and Japanese) and unsloth's imatrix: gate / up IQ3_XXS, down IQ3_XXS
  (layers 3-5, 11, 12, 42-44 IQ4_XS).
- Dense layers: attention output, attention q_b and the dense / shared-expert down Q5_K; the other attention projections,
  dense / shared-expert gate / up and the NextN projection Q4_K.
- Router, embeddings, output and the rest of the NextN (MTP) layer: unsloth UD-IQ2_XXS's bytes.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 10 sequences x 1,024 tokens: English text, code and Japanese), Abyss
engine: see the table above. Real-use checks (Japanese, coding tasks, 8K needle) and a llama.cpp run: **in progress,
results will be added here**.

## Speed

[Abyss](https://github.com/zephel01/Abyss) v1.3.117, one PC: RTX 5090 (32 GB) on OCuLink PCIe Gen4 x4, Ryzen AI Max+ 395
(LPDDR5X), **96 GB RAM** (RAM tier 60 GB), NVMe SSD. Temperature 0, 512 tokens, page cache dropped before each run,
median of 3. 4K prompt 23.2 tok/s, 16K prompt 21.1 tok/s, with 17 ms a token spent waiting for the SSD. The 128 GB
numbers above are estimates.

## Feedback (this is a BETA) — 128 GB testers wanted

- **Speed on your PC** (especially 32 GB / 24 GB GPU + 128 GB RAM, or RTX PRO 6000): `./abyss.sh --bench --deep` in
  Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+32G128G+Q120d+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

GLM-5.3-Flash の、**品質を優先した** GGUF です。**GPU 32 GB（RTX 5090、AI PRO R9700）+ RAM 128 GB 以上の PC** 向けです。
エキスパートは削っていません。KL は unsloth の UD-IQ2_XXS の**約半分**（0.137 対 0.265）です。

**BETA です。RAM 128 GB の PC での速さは、まだ測れていません**。手元の PC は RAM が 96 GB で、このファイルのうち 33 GB が
SSD に残ってしまうからです。**GPU 32 GB（または 24 GB）+ RAM 128 GB の PC や、RTX PRO 6000 をお持ちの方、ぜひ測って
ください**。実際の作業での確認と llama.cpp での確認は途中で、終わりしだい書き足します。

**速さは、ファイルが「GPU のメモリ + RAM」に全部載るかで大きく変わります。全部載れば速く、載りきらずに SSD から読む分が
あると遅くなります。** Abyss は、よく使うエキスパートを VRAM に、残りを RAM に置き、入りきらない分は 1 トークンごとに SSD
から読みます。この SSD 読みの待ちが、時間の大部分を占めます。Q120d は 127.9 GB なので、GPU 32 GB + **RAM 128 GB** なら全部
載ります（見込み 40〜45 tok/s）。**RAM 96 GB** だと 33 GB が SSD に残り、23 tok/s まで落ちます。RAM が 128 GB より少ない
PC では、S1（94 GB）を使ってください。

| | 大きさ | KL（10 本） | KL（日本語） | 同じ 1 位 | 5090 + RAM 96 GB、4K | 5090 + RAM 128 GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q120d（ここ）** | **127.9 GB** | **0.1367** | **0.145** | **84.9%** | 23.2 tok/s（SSD 待ち） | **約 40〜45 tok/s（見込み）** |
| [S1](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF)（32G96G） | 94.1 GB | 0.2706 | 0.280 | 79.1% | 75.7 tok/s | |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2651 | 0.240 | 79.0% | 53.3 tok/s | |

- **日本語**：KL を測る 10 本のうち 1 本が日本語の文章で、その KL を別の列に出しています。Q120d は日本語でも unsloth より
  約 4 割少なく、品質優先の効きが日本語にも出ています。日本語の文章を増やして測り直すのと、実際の質問での確認は、
  これから行います。

- 品質を速さより優先したいとき（長い推論、難しいコード、日本語）向けです。速さを優先するなら、RAM 96 GB で全部載る
  [S1（32G96G）](https://huggingface.co/zephel01/GLM-5.3-Flash-32G96G-GGUF) を使ってください。DGX Spark も S1 です
  （Q120d は Spark のメモリに入りきりません）。
- RAM 96 GB の PC では、1 トークンごとに 9.5 個のエキスパートを SSD から読み、43 ms のうち 17 ms を待っています。RAM が
  128 GB あれば全部が VRAM と RAM に載るので、1 トークン 22〜25 ms（40〜45 tok/s）と見込んでいます。

測った結果は、`./abyss.sh --bench --deep` の `abyss-bench.txt` を、GPU、RAM、SSD と一緒に
[Abyss の issue](https://github.com/zephel01/Abyss/issues) か Community タブに貼ってもらえると助かります。
