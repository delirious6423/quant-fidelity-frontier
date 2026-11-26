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
- local-llm
- japanese
language:
- en
- ja
library_name: gguf
---

![GLM-5.3-Flash IQ2_XS: 27.4 tok/s on one DGX Spark](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [DGX Spark guide](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) | [Base model](https://huggingface.co/zai-org/GLM-5.3-Flash) | [32G96G version](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+DGX-Spark+S1+on+GB10) | 日本語は下

# GLM-5.3-Flash GGUF for one DGX Spark (GB10, 128 GB) — BETA

- **DGX-Spark = for one NVIDIA DGX Spark (GB10, 128 GB unified memory)** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1% (by bytes), 2.26 bits a weight on average, about IQ2_XS.

A GGUF of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT) that fits one NVIDIA DGX Spark
with room left for context. **Every expert is kept** (no pruning). Speed and quality are measured in
[Abyss](https://github.com/zephel01/Abyss), **on a real DGX Spark**.

**BETA**: measured on a rented DGX Spark on 2026-11-07. Newer Abyss releases may raise the speed; this card will follow.
The file may still be replaced by a better build (the folder name will change then). Reports are welcome (see Feedback).

## Model files

`S1/GLM-5.3-Flash-S1-0000N-of-00004.gguf`, 4 shards, 94.1 GB. Point your engine at `-00001-of-00004.gguf`.
`SHA256SUMS` is in the folder. The same files as
[GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) `S1/`; that card
lists the tensor types and how they were made.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | 32 GB GPU + 96 GB RAM | 94.1 GB | folder S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | 32 GB GPU + 128 GB RAM | 127.9 GB | higher quality (folder Q120d) |
| **this repo** | DGX Spark (128 GB) | 94.1 GB | same files as 32G96G |

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief: why this file on a Spark

- On a Spark the whole file sits in unified memory. Decode is bound by its 273 GB/s memory, so speed follows the bytes
  read per token.
- S1 reads about 8% less a token than unsloth's UD-IQ2_XXS, at a slightly better KL (0.253 vs 0.264).
- It leaves about 8 GB more memory for long context.
- Abyss drafts with the model's own NextN block and checks the drafts in one batched window on the Spark (the same tokens
  as without drafting).

## Run it on a Spark

Abyss builds and runs on the DGX Spark (aarch64, sm_121). Build as in
[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) (steps 1-2), then:

```sh
cd Abyss
.venv/bin/hf download zephel01/GLM-5.3-Flash-DGX-Spark-GGUF --include "S1/*" --local-dir ~/models/glm-s1
(cd ~/models/glm-s1/S1 && sha256sum -c SHA256SUMS)
.venv/bin/python tools/iq_pack.py --gguf ~/models/glm-s1/S1/GLM-5.3-Flash-S1-00001-of-00004.gguf \
  --out ~/models/glm-s1/S1/pack --compat-bf16
cat > abyss-glm-s1.json <<EOF
{"exe": "build-cuda/strata",
 "args": ["--glm-pack", "$HOME/models/glm-s1/S1/pack", "--max-context", "32768", "--prefill", "auto"],
 "tokenizer": "$HOME/models/glm-s1/S1/pack/tokenizer",
 "model_name": "glm-5.3-flash", "gpu": [0],
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "lib_dirs": ["/usr/local/cuda/lib64"], "port": 8082, "backend": "cuda"}
EOF
.venv/bin/python serve/server.py --engine strata --config abyss-glm-s1.json --port 8082
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

The NextN drafter and the batched verify window are on by default on the Spark.

**llama.cpp** (arch `glm5-next`, upstream `ac8e1694d` of 2026-10-29): runs as is (checked on an RTX 5090: correct
answers to an English and a Japanese question). Not measured on the Spark.

## Quality

KL against unsloth's UD-Q4_K_XL (top-k log-probs, 14 sequences x 1,024 tokens: 8 English, 1 code, 5 Japanese), Abyss
engine, re-measured 2026-11-08 with the router's expert bias kept in F32 as stored in the GGUF.

| | download | experts | KL (14 seqs) | KL, Japanese (5) | same top token |
| --- | ---: | ---: | ---: | ---: | ---: |
| **S1 (this repo)** | **94.1 GB** | **all** | **0.2534** | **0.251** | **77.3%** |
| unsloth UD-IQ2_XXS (reference) | 101.8 GB | all | 0.2643 | 0.257 | 76.6% |

- **Japanese, measured separately**: on the 5 Japanese sequences S1 is a little closer to the original than
  UD-IQ2_XXS (0.251 vs 0.257). A build with more Japanese in its calibration did not do better on general Japanese
  text, so S1 stays.
- Earlier numbers here (S1 0.2706, UD 0.2651) came from an Abyss pack that rounded that bias to BF16. The GGUF files did
  not change.
- Abyss `--check` passes 12 of 12 (on an RTX 5090).

## Speed

![This file vs unsloth UD-IQ2_XXS on one DGX Spark](https://huggingface.co/zephel01/GLM-5.3-Flash-DGX-Spark-GGUF/resolve/main/assets/speed.png)

Rented DGX Spark (GB10, 128 GB unified memory), the default memory budget, 4K prompt, median of 3 x 512 tokens.

| | with the NextN drafter (v1.3.119) | without drafting | v1.3.118 with the drafter |
| --- | ---: | ---: | ---: |
| **S1 (this repo)** | **27.4 tok/s** | 21.4 tok/s | 24.3 tok/s |
| unsloth UD-IQ2_XXS | 24.2 tok/s | 19.8 tok/s | 22.9 tok/s |

- Drafts accepted 74-77%. The drafter gives the same tokens as without it.
- v1.3.119 made the verify window faster. With the whole v1.3.119 release S1 measured 28.6 tok/s; that release also
  changes greedy tokens here (`STRATA_GLM_TAIL_ON_TOP`).
- On an RTX 5090 with a 60 GB RAM tier the same file decodes at 47.4 tok/s (Abyss v1.3.129 engine, iGPU off, no
  drafter, 4K prompt; see the [32G96G card](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF)).
- Known issue, seen only on the Spark: 32K-121K prompts through the server ran at ~120-130 tok/s while the CLI read them
  at 675-796 tok/s (being investigated, see the DGX Spark guide).

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+DGX-Spark+S1+on+GB10).
- **Quality in real use**: where it went wrong, switched language or looped. A prompt that reproduces it helps most.

## Credits

Base model: Z.ai [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT). Source GGUFs and imatrix:
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Quantized with
[trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- **DGX-Spark = NVIDIA DGX Spark 1 台（GB10、共有メモリ 128 GB）向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ2_XXS 58%, IQ2_S 36%, IQ3_XXS 3%, Q2_K 2%, Q3_K 1%（容量の割合）、平均 2.26 ビットで、IQ2_XS くらいです。

GLM-5.3-Flash の GGUF を、**NVIDIA DGX Spark 1 台**に、長い文脈の余裕を残して載るようにしたものです。
**エキスパートは削っていません**。速さと品質は [Abyss](https://github.com/zephel01/Abyss) で、**本物の DGX Spark で**測っています。

**BETA です**：2026-11-07 に、借りた DGX Spark で測りました。速さはエンジンで決まるので、Abyss の新しい版で上がることがあり、
そのときはこのページも直します。もっと良い版に入れ替えることもあります（そのときはフォルダの名前を変えます）。

| | 大きさ | KL（14 本） | KL（日本語 5 本） | 同じ 1 位 | Spark の生成（下書きあり、v1.3.119） | 下書きなし |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **S1（ここ）** | **94.1 GB** | **0.2534** | **0.251** | **77.3%** | **27.4 tok/s** | 21.4 tok/s |
| unsloth UD-IQ2_XXS（比べる相手） | 101.8 GB | 0.2643 | 0.257 | 76.6% | 24.2 tok/s | 19.8 tok/s |

- [`GLM-5.3-Flash-IQ2_XS-32G96G-GGUF`](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) の `S1/` と同じファイルです。
  RTX 5090（RAM の置き場 60 GB、iGPU なし・下書きなし、Abyss v1.3.129 のエンジン、4K プロンプト）では 47.4 tok/s です。
- 4K のあと、3 回の中央値 x 512 トークン。モデル自身の NextN 層で下書きを作り、Spark では 1 つの窓でまとめて確かめます
  （出力は下書きなしと同じ。下書きの採用 74〜77%）。
- Spark ではファイルが全部共有メモリに載ります。生成の速さはメモリの速さ（273 GB/s）で決まるので、1 トークンで読む量で
  ほぼ決まります。S1 は unsloth の UD-IQ2_XXS より読む量が約 8% 少なく、長い文脈に使えるメモリも約 8 GB 多く残ります。
- **日本語**：日本語 5 本の KL は S1 が 0.251 で、unsloth（0.257）より少し良い結果です。日本語を強めた版も試しましたが、
  一般的な日本語の文章では差が出なかったので、S1 のままにします。
- Abyss の `--check` は 12 項目すべて合格（RTX 5090）です。
- 動かし方は [docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.ja.md) でビルドしてから、上の英語の節のコマンドのとおりです。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます（RTX 5090 で、英語と日本語の質問に正しく答えることを確かめました）。Spark では測っていません。

感想や速さの報告は、Community タブか [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+GLM-5.3-Flash+DGX-Spark+S1+on+GB10) にお願いします。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [GLM-5.3-Flash-IQ2_XS-32G96G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF) | GPU 32 GB + RAM 96 GB | 94.1 GB | S1 |
| [GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF](https://huggingface.co/zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF) | GPU 32 GB + RAM 128 GB | 127.9 GB | 品質重視（Q120d） |
| **この版** | DGX Spark（128 GB） | 94.1 GB | 32G96G と同じファイル |

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
