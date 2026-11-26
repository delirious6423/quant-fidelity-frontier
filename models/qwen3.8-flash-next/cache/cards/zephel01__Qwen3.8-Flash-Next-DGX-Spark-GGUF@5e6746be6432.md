---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- ldlq
- qwen4exp
- moe
- abyss
- japanese
- dgx-spark
- gb10
- local-llm
language:
- en
- ja
library_name: gguf
---

![Qwen3.8-Flash-Next on one DGX Spark: 55.8 tok/s](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Abyss on DGX Spark](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) | [Base model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [16G64G version (same files)](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+DGX-Spark+on+GB10) | 日本語は下

# Qwen3.8-Flash-Next GGUF for one DGX Spark (GB10, 128 GB) — BETA

- a mixed-type file - the experts are IQ3_XXS 56%, IQ4_NL 42%, IQ3_S 1% (by bytes), 3.55 bits a weight on average, about IQ3_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for **one NVIDIA DGX Spark** (GB10,
128 GB unified memory). **All 512 experts are kept** (no pruning). Speed is measured in
[Abyss](https://github.com/zephel01/Abyss) **on a real DGX Spark**.

These are the same files as [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)
(byte for byte, same `SHA256SUMS`). This repo is the Spark entry point: the Spark numbers and how to run it there. How
the file was made and the full quality comparison are on the 16G64G card.

**BETA**: measured on a rented DGX Spark (2026-11-05 to 11-07). Newer Abyss releases may change the speed; this card
will follow. Reports from your Spark are welcome (see Feedback).

## Model files

`Qwen3.8-Flash-Next-16G64G-0000N-of-00003.gguf`, 3 shards, 86.9 GB. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the repo. If you already have the 16G64G files, you do not need to download these again.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) | 12 GB GPU + 32 GB RAM | 67.5 GB | 448 of 512 experts kept |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 86.9 GB | all 512 experts |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | 24 GB GPU + 64 GB RAM | 93.7 GB | all 512 experts, higher quality |
| **this repo** | DGX Spark (128 GB) | 86.9 GB | same files as 16G64G |

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## In brief

- The whole file fits the Spark's memory. The 28.8 GB per-layer embedding table is read a few rows a token from the SSD.
- On a Spark, Abyss sizes the expert pool and the RAM tier from one memory budget, and turns the CPU lane off (the GPU
  reads the same memory faster).
- Experts: re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in unsloth UD-IQ3_XXS's layout with gate/up one
  step up.

## Run it on a Spark

From [docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) (DGX OS, CUDA 13 in
`/usr/local/cuda`):

```sh
git clone https://github.com/zephel01/Abyss.git && cd Abyss
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
sudo apt-get install -y cmake ninja-build python3-venv
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil huggingface_hub sentencepiece
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_IGPU_LANE=OFF \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-cuda --target strata abyss_gpu_tune -j"$(nproc)"
.venv/bin/hf download zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF --local-dir ~/models/q38-16g64g
(cd ~/models/q38-16g64g && sha256sum -c SHA256SUMS)
.venv/bin/python tools/iq_pack.py --fast \
  --gguf ~/models/q38-16g64g/Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf --out ~/models/q38-16g64g/pack-fast
cat > abyss-q38-16g64g.json <<EOF
{"exe": "build-cuda/strata",
 "args": ["--glm-pack", "$HOME/models/q38-16g64g/pack-fast", "--max-context", "32768", "--prefill", "auto"],
 "tokenizer": "$HOME/models/q38-16g64g/pack-fast/tokenizer",
 "model_name": "qwen3.8-flash-next-16g64g", "gpu": [0],
 "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
 "lib_dirs": ["/usr/local/cuda/lib64"], "port": 8082, "backend": "cuda"}
EOF
.venv/bin/python serve/server.py --engine strata --config abyss-q38-16g64g.json --port 8082
```

The MTP drafter's runtime files (recommended; it took a Spark from 31 to 47 tok/s) are set up as in
[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md), step 3.

**llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is. These are the same files as the
[16G64G repo](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF), where the check is described.
Not measured on the Spark.

## Quality

| | download | experts | KL (10 seqs) | KL, Japanese |
| --- | ---: | ---: | ---: | ---: |
| **this file** | **86.9 GB** | **all 512** | **0.083** | **0.176** |
| unsloth UD-IQ3_XXS (reference) | 82.0 GB | all 512 | 0.154 | 0.317 |

KL against the NVFP4 model, 10 held-out sequences x 1,023 tokens (code, English, Chinese, Japanese and others).

## Speed on the Spark

| | decode, after a 4K prompt | decode, after a 32K prompt |
| --- | ---: | ---: |
| Abyss v1.3.119 | **55.8 tok/s** | **53.4 tok/s** |
| Abyss v1.3.118 | 54.7 tok/s | 47.5 tok/s |

- DGX Spark, the MTP drafter on, 512 tokens after the prompt.
- Prompt processing on the Spark: about 1,040-1,100 tok/s from 4K to 64K (Abyss v1.3.113).
- No other engine or file was measured on the Spark, so there is no comparison chart here.

## Feedback (this is a BETA)

- **Speed on your Spark**: `./abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+DGX-Spark+on+GB10).
- **Quality in real use**: empty answers, wrong language, loops. A prompt that reproduces it helps most.

## License and credits

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are
[magnitudedev](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)'s, the non-expert tensors unsloth's.
Quantized with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

# 日本語

- 形式を混ぜたファイルです。エキスパートは IQ3_XXS 56%, IQ4_NL 42%, IQ3_S 1%（容量の割合）、平均 3.55 ビットで、IQ3_M くらいです。

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**NVIDIA DGX Spark 1 台**（GB10、共有メモリ
128 GB）向けにまとめたものです。**エキスパートは 512 個すべて残しています**。速さは [Abyss](https://github.com/zephel01/Abyss)
で、**本物の DGX Spark で**測りました。

ファイルは [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) と同じものです
（`SHA256SUMS` も同じ）。作り方と品質の詳しい比較は、16G64G のカードにあります。

**BETA です**：借りた DGX Spark で測りました（2026-11-05〜07）。Abyss の新しい版で速さが変われば、このカードも更新します。

| | 大きさ | エキスパート | KL（10 本） | 日本語の KL | Spark の生成（4K のあと） | 32K のあと |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **この版** | **86.9 GB** | **512 個すべて** | **0.083** | **0.176** | **55.8 tok/s** | **53.4 tok/s** |
| unsloth UD-IQ3_XXS（比べる相手） | 82.0 GB | 512 個すべて | 0.154 | 0.317 | - | - |

- DGX Spark、Abyss v1.3.119、MTP の下書きあり、プロンプトのあと 512 トークン（v1.3.118 では 54.7 / 47.5 tok/s）。
- Spark でのプロンプトの処理：4K〜64K で約 1,040〜1,100 tok/s（Abyss v1.3.113）。
- ファイル全体が Spark のメモリに載ります。層ごとの埋め込み表（28.8 GB）は、1 トークンにつき数行だけ SSD から読みます。
- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。16G64G のファイルをお持ちなら、ダウンロードし直す必要はありません。
- 動かし方は上の英語の節のコマンドのとおりです（[docs/DGX-SPARK.md](https://github.com/zephel01/Abyss/blob/main/docs/DGX-SPARK.md) と同じ手順）。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。[16G64G のリポジトリ](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) と同じファイルで、確かめ方はそちらに書いてあります。Spark では測っていません。
- 感想や速さの報告は [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+DGX-Spark+on+GB10) にお願いします。
- ライセンスは元のモデルと同じです（qwen-community-1.0）。

## サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) | GPU 12 GB + RAM 32 GB | 67.5 GB | エキスパート 512 個中 448 個 |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 86.9 GB | 512 個すべて |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | GPU 24 GB + RAM 64 GB | 93.7 GB | 512 個すべて、品質重視 |
| **この版** | DGX Spark（128 GB） | 86.9 GB | 16G64G と同じファイル |

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
