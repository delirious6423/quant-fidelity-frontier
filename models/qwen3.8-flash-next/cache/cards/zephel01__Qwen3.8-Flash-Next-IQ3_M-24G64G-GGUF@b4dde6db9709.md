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
- consumer-gpu
- local-llm
- cpu-offload
- rtx-5090
- 24gb-vram
- 64gb-ram
language:
- en
- ja
library_name: gguf
---

![Qwen3.8-Flash-Next IQ3_M: 69.9 tok/s on a 24 GB GPU + 64 GB RAM setup](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [12G32G version](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) | [16G64G version](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+24G64G+on+YOUR_GPU) | 日本語は下

# Qwen3.8-Flash-Next IQ3_M GGUF (for a 24 GB GPU + 64 GB RAM PC) — BETA

- **24G64G = for a PC with a 24 GB GPU (VRAM) + 64 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ3_S 57%, IQ4_NL 34%, Q8_0 7%, IQ4_XS 1% (by bytes), 3.94 bits a weight on average, about IQ3_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for **one 24 GB GPU and 64 GB of
RAM**, in **exactly unsloth UD-IQ4_XS's types and size (93.7 GB)**. All 512 experts are kept. Same size, same speed on
the same emulated PC (69.9 vs 71.1 tok/s on coding tasks), and **KL 0.068 vs 0.092 (-26%)** against the NVFP4 model.
KL on Japanese text: 0.146 vs 0.195. Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only. Reports from real 24 GB GPUs and other RAM sizes are welcome (see Feedback).

## Model files

`Qwen3.8-Flash-Next-24G64G-0000N-of-00003.gguf`, 3 shards, 93.7 GB. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the repo.

- Experts: gate/up IQ3_S (layer 2 IQ4_XS), down IQ4_NL (layers 2, 4, 30, 46, 47 Q8_0).
- Non-expert tensors (attention, router, shared expert, embeddings, per-layer embedding table) are unsloth
  UD-IQ4_XS's bytes, unchanged.
- **Abyss v1.3.108 or later** is needed for the prompt path: the Q8_0 down experts of this layout read past their slot
  in older versions (also in unsloth's UD-IQ4_XS).

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) | 12 GB GPU + 32 GB RAM | 67.5 GB | 448 of 512 experts kept |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 86.9 GB | all 512 experts |
| **this repo** | 24 GB GPU + 64 GB RAM | 93.7 GB | all 512 experts, higher quality |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark (128 GB) | 86.9 GB | same files as 16G64G |

Earlier repo name: Qwen3.8-Flash-Next-24G64G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## The recipe in brief

- Every expert is re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in UD-IQ4_XS's types.
- The two files have the same tensor types. Only the rounding differs (LDLQ, calibrated with Japanese and the model's
  own chat turns).
- No experts removed.

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.108 or later (one command sets up the engine, the model, the pack
and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model qwen38 --quant 24G64G --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --model qwen38 --quant 24G64G --download-model`. On a 64 GB PC the RAM tier is
`STRATA_GLM_RAM_GB=46`.

**llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is. Checked with:

```sh
hf download zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF --local-dir Qwen3.8-24G64G
llama-server -m Qwen3.8-24G64G/Qwen3.8-Flash-Next-24G64G-00001-of-00003.gguf -ngl 99 --n-cpu-moe 34 -c 32768 --jinja -t 8
```

That took 24.2 GB of VRAM; use 35 or more on a real 24 GB card. The same real-use prompts: Japanese 5/5 answered (none
empty), coding 3/5, the 8K needle found; decode 26-34 tok/s.

Sampling: temperature 1.0, top_p 0.95, top_k 20.

## Quality

| | download | resident (GPU + RAM) | KL (10 seqs) | same top token | KL, Japanese | KL, code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **this file** | 93.7 GB | 64.2 GB | **0.068** | **90.3%** | **0.146** | **0.092** |
| unsloth UD-IQ4_XS | 93.7 GB | 64.9 GB | 0.092 | 88.7% | 0.195 | 0.116 |

KL against the NVFP4 model (10 held-out sequences x 1,023 tokens: code, English, Chinese, Japanese and others), Abyss
engine. "Resident" leaves out the 28.8 GB per-layer embedding table, which is read a few rows a token from the SSD.

### Real use

Abyss server, thinking on, two seeds, on the emulated PC (see Speed):

| | this file | unsloth UD-IQ4_XS |
| --- | --- | --- |
| Japanese (5 prompts: business email, explainer, summary, literature fact, arithmetic) | 10/10 answered, facts and sums right | 10/10 |
| 5 coding tasks with hidden tests | 5/5, 4/5 | 3/5, 5/5 |
| 8K-token prompt, find the needle | OK / OK | OK / OK |
| empty answers / prompt echoes | none | none |
| prompt processing (8K) | 1,680-1,689 tok/s | 1,670-1,683 tok/s |

## Speed

![Decode speed and KL vs unsloth UD-IQ4_XS](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF/resolve/main/assets/speed.png)

Abyss v1.3.108, a 24 GB GPU + 64 GB RAM PC emulated on an RTX 5090 (32 GB): VRAM capped at 24 GB
(`STRATA_GLM_VRAM_GB=24`), RAM tier 46 GB inside a 58 GB cgroup, 8 CPU threads, iGPU off, NVMe SSD, MTP drafter on.

| | decode |
| --- | ---: |
| **this file, 3 coding tasks** | **69.9 tok/s** |
| unsloth UD-IQ4_XS, 3 coding tasks | 71.1 tok/s |
| this file, 3 Japanese prompts | 61-77 tok/s |
| this file in llama.cpp, `--n-cpu-moe 34` (24.2 GB of VRAM) | 26-34 tok/s |

Other GPUs are not measured with this file.

<details>
<summary>How it was made</summary>

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/hessian_dump.cpp`, `tools/requant_experts.py`,
`tools/merge_experts.py`):

1. Hessians: the NVFP4 model ([magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF))
   over 1.26M tokens: its own chat turns (181 prompts: Japanese 44, Chinese 20, English 20, code 37, math 35, tool
   calls 25) followed by mixed text in the same languages and domains.
2. LDLQ: every expert re-quantized from the NVFP4 weights (per-expert scales folded in) in UD-IQ4_XS's types, then
   merged into unsloth UD-IQ4_XS.

</details>

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+24G64G+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: empty answers, wrong language, loops, places where it is weaker than unsloth's file. A
  prompt that reproduces it helps most.

## License and credits

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are
[magnitudedev](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)'s, the non-expert tensors unsloth's.
Quantized with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

## 日本語

- **24G64G = GPU（VRAM）24 GB + RAM 64 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ3_S 57%, IQ4_NL 34%, Q8_0 7%, IQ4_XS 1%（容量の割合）、平均 3.94 ビットで、IQ3_M くらいです。

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**GPU 24 GB + RAM 64 GB の PC** 向けに作りました。
**型も大きさ（93.7 GB）も unsloth の UD-IQ4_XS とまったく同じ**で、エキスパートは 512 個すべて残しています。違うのは丸め方（LDLQ、日本語とチャットのやりとりを入れた調整）だけです。
同じ設定で比べると、速さは同じ（69.9 と 71.1 tok/s）で、**KL は 0.092 → 0.068（-26%）**、日本語の文章では 0.195 → 0.146 です。Abyss と llama.cpp で動きます。

**BETA 版です。** 測ったのは 1 台だけです。本物の 24 GB の GPU や、ほかの RAM 容量での結果を教えてもらえると助かります。

| | ダウンロード | KL（10 本） | 1 位の一致 | 日本語の KL | コードの KL | 生成（コーディング 3 問） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **この版** | 93.7 GB | **0.068** | **90.3%** | **0.146** | **0.092** | 69.9 tok/s |
| unsloth UD-IQ4_XS | 93.7 GB | 0.092 | 88.7% | 0.195 | 0.116 | 71.1 tok/s |

- 速さは RTX 5090 で GPU 24 GB + RAM 64 GB の PC を真似た設定です（VRAM 24 GB に制限、RAM の枠 46 GB を全体 58 GB に制限、CPU 8 スレッド、iGPU なし、MTP ドラフターあり）。日本語 3 問は 61〜77 tok/s。
- 実際の使い方（思考あり、seed を変えて 2 回）：日本語 5 問は 10/10 回答、コーディング 5 問は 5/5、4/5（unsloth は 3/5、5/5）、8K の情報探しは成功、空の回答なし。
- **Abyss は v1.3.108 以降が必要です。** この並びの Q8_0 の down エキスパートは、それより前の版では入力の処理で読み過ぎが起きていました（unsloth の UD-IQ4_XS も同じです）。
- Abyss では `./setup-deepseek.sh --model qwen38 --quant 24G64G --download-model` で入ります（64 GB の PC なら `STRATA_GLM_RAM_GB=46`）。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。`--n-cpu-moe 34` で VRAM 24.2 GB だったので、本物の 24 GB の GPU では 35 以上にしてください。日本語 5 問すべてに回答、コーディング 3/5、8K の情報探しも成功、生成 26〜34 tok/s。
- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。
- 速さの報告は [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+24G64G+on+YOUR_GPU) にお願いします（タイトルの GPU 名を書き換えてください）。
- ライセンスは元のモデルと同じです（[qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)）。

### サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| [Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) | GPU 12 GB + RAM 32 GB | 67.5 GB | エキスパート 512 個中 448 個 |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 86.9 GB | 512 個すべて |
| **この版** | GPU 24 GB + RAM 64 GB | 93.7 GB | 512 個すべて、品質重視 |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark（128 GB） | 86.9 GB | 16G64G と同じファイル |

以前のリポジトリ名は Qwen3.8-Flash-Next-24G64G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
