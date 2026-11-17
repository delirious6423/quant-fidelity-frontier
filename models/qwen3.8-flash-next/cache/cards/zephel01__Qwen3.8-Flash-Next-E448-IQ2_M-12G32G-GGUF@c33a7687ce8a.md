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
- 12gb-vram
- 32gb-ram
language:
- en
- ja
library_name: gguf
---

![Qwen3.8-Flash-Next E448-IQ2_M: 56.0 tok/s on an RTX 5090 capped at 12 GB + 20 GB RAM tier](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF/resolve/main/assets/teaser.png)

[Abyss (engine)](https://github.com/zephel01/Abyss) | [Base model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | [16G64G version](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | [24G64G version](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | [Report a result](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+12G32G-L448+on+YOUR_GPU) | 日本語は下

# Qwen3.8-Flash-Next E448-IQ2_M GGUF (for a 12 GB GPU + 32 GB RAM PC) — BETA

- **12G32G = for a PC with a 12 GB GPU (VRAM) + 32 GB RAM** - a target setup, not a guarantee of speed.
- a mixed-type file - the experts are IQ4_NL 57%, IQ1_S 28%, IQ2_XXS 15% (by bytes), 2.64 bits a weight on average, about IQ2_M.

A GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for **one 12 GB GPU and 32 GB of
RAM**. Each layer keeps 448 of its 512 experts. Against unsloth's UD-IQ1_S on the same emulated small PC (an RTX 5090 capped at 12 GB, RAM tier 20 GB in a 28 GB
cgroup): **30% faster** (56.0
vs 43.0 tok/s on coding tasks), 5 GB smaller (67.5 vs 72.5 GB), and a lower Japanese KL. Runs in Abyss and llama.cpp.

**BETA**: measured on one PC only. Reports from other GPUs / RAM sizes are welcome (see Feedback).

## Model files

`L448/Qwen3.8-Flash-Next-12G32G-L448-0000N-of-00003.gguf`, 3 shards. Point your engine at `-00001-of-00003.gguf`.
`SHA256SUMS` is in the folder.

| | experts | per-layer embedding table | the rest | total |
| --- | ---: | ---: | ---: | ---: |
| L448 | 34.9 GB (gate/up IQ1_S / IQ2_XXS, down IQ4_NL) | 28.8 GB (IQ4_NL) | 3.9 GB | 67.5 GB |

- The per-layer embedding table is read a few rows a token. It does not have to fit in RAM (Abyss reads it from the
  SSD, llama.cpp maps it).
- Non-expert tensors (attention, router, shared expert, embeddings, PLE table) are unsloth UD-IQ1_S's bytes, unchanged.
- The file / folder names keep the earlier label L448.

### Pick a size (this model's other repos)

| repo | for a PC with | download | note |
| --- | --- | ---: | --- |
| **this repo** | 12 GB GPU + 32 GB RAM | 67.5 GB | 448 of 512 experts kept |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | 16 GB GPU + 64 GB RAM | 86.9 GB | all 512 experts |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | 24 GB GPU + 64 GB RAM | 93.7 GB | all 512 experts, higher quality |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark (128 GB) | 86.9 GB | same files as 16G64G |

Earlier repo name: Qwen3.8-Flash-Next-12G32G-GGUF (old links redirect here).

Names: the quant in the repo name is the nearest llama.cpp type to the experts' average bits; most files mix several types (listed above), so they are not a stock quant of that name. `E<n>` = experts kept per layer (only when some were removed).

## The recipe in brief

- Every expert is re-quantized with LDLQ from NVIDIA-format NVFP4 weights, in unsloth UD-IQ1_S's own layout.
- Calibration: the model's own chat turns (181 prompts, 44 in Japanese) plus mixed text. Japanese is part of the
  calibration, the expert selection and the tests.
- Each layer keeps the 448 experts that routing used most over Japanese, Chinese, English, code, math and tool-call
  text. On held-out text the kept experts cover 96.9-99.1% of the routes, per domain.

## Run it

**[Abyss](https://github.com/zephel01/Abyss)** v1.3.106 or later (one command sets up the engine, the model, the pack
and the config):

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --model qwen38 --quant 12G32G --download-model
../Abyss-tools/abyss.sh start                         # → http://127.0.0.1:8082/
```

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

Windows 11: `setup-deepseek.bat --model qwen38 --quant 12G32G --download-model`.

**llama.cpp** (arch `qwen4exp`, upstream `ac8e1694d` of 2026-10-29): runs as is. The expert count is read from the file.
Checked with:

```sh
hf download zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF --local-dir Qwen3.8-12G32G
llama-server -m Qwen3.8-12G32G/L448/Qwen3.8-Flash-Next-12G32G-L448-00001-of-00003.gguf -ngl 99 --n-cpu-moe 41 -c 32768 --jinja -t 8
```

10.9 GB of VRAM, 28 GB host limit. The same real-use prompts: Japanese 5/5 answered (none empty), coding 3/5, the 8K
needle found; decode 26-32 tok/s, prompt processing 88 tok/s.

Sampling: temperature 1.0, top_p 0.95, top_k 20.

## Quality

| | experts per layer | download | KL (10 seqs) | same top token | KL, Japanese seq |
| --- | ---: | ---: | ---: | ---: | ---: |
| **L448 (this repo)** | **448 / 512** | **67.5 GB** | 0.339 | 78.5% | **0.553** |
| unsloth UD-IQ1_S (reference) | 512 / 512 | 72.5 GB | 0.331 | 78.4% | 0.568 |

KL against the NVFP4 model (10 held-out sequences x 1,023 tokens: code, English, Chinese, Japanese and others), Abyss
engine. Overall KL is about unsloth's, with 64 fewer experts per layer.

### Real use

Abyss server, temperature 1.0 / top_p 0.95 / top_k 20, thinking on, two seeds, on the emulated small PC (see Speed):

| | L448 (this repo) | unsloth UD-IQ1_S |
| --- | --- | --- |
| Japanese (5 prompts: business email, explainer, summary, literature fact, arithmetic) | 10/10 answered, facts and sums right | 10/10 |
| "Japan's three great essays" (a known overthinking case) | 196 / 165 tokens, right | 454 / 384 tokens, right |
| 5 coding tasks with hidden tests | 3/5, 3/5 | 5/5, 3/5 |
| 8K-token prompt, find the needle | OK / OK | OK |
| prompt processing (8K) | 1,822 tok/s | 1,368 tok/s |
| Japanese decode | 48-51 tok/s | 39-46 tok/s |

- One coding task (a hiragana/katakana converter) makes this model think until the token limit in most runs, in every
  build (unsloth's solved it once in two). Give it more `max_tokens` or less thinking there.
- **Empty answers**: the first build of this file (calibrated on plain text only) sometimes ended the turn right after
  its thinking. This version adds the model's own chat turns to the calibration; the empty answers and the long
  overthinking were gone in our tests. Abyss v1.3.106 also guards against it (`STRATA_GLM_END_GUARD`, on by default).

## Speed

![Decode and prompt speed vs unsloth UD-IQ1_S on a 12 GB GPU + 32 GB RAM setup](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF/resolve/main/assets/speed.png)

Abyss, a 12 GB GPU + 32 GB RAM PC emulated on an RTX 5090 (32 GB): VRAM capped at 12 GB (`STRATA_GLM_VRAM_GB=12`), RAM
tier 20 GB inside a 28 GB cgroup, 8 CPU threads for the CPU lane, iGPU off, NVMe SSD, no drafter.

| | decode, 3 coding tasks (17K tokens) |
| --- | ---: |
| **this file, Abyss** | **56.0 tok/s** |
| unsloth UD-IQ1_S, Abyss | 43.0 tok/s |
| unsloth UD-IQ1_S, llama.cpp `--n-cpu-moe 41` (about 12 GB of VRAM) | 27.6 tok/s |

Other GPUs are not measured with this file.

<details>
<summary>How it was made</summary>

With [trench-gguf](https://github.com/zephel01/trench-gguf) (`tools/hessian_dump.cpp`, `tools/requant_experts.py`,
`tools/merge_experts.py`, `tools/prune_experts.py`) and Abyss's routing counter (`STRATA_GLM_USAGE`):

1. Hessians: the NVFP4 model ([magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF))
   over 1.26M tokens: its own chat turns (181 prompts: Japanese 44, Chinese 20, English 20, code 37, math 35, tool
   calls 25) followed by mixed text in the same languages and domains.
2. LDLQ: every expert re-quantized from the NVFP4 weights (per-expert scales folded in), in UD-IQ1_S's layout
   (gate/up IQ1_S, or IQ2_XXS in 14 layers; down IQ4_NL), then merged into UD-IQ1_S.
3. Selection: routing counts over Japanese, Chinese, English, code, math and tool-call text; per layer the 448 most
   used experts, every domain weighted the same and code twice, checked on held-out text (kept routes: Japanese
   98.6%, Chinese 98.1%, English 98.3%, code 96.9%, math 98.2%, tools 99.1%).
4. `prune_experts.py` cuts the experts and router rows.

</details>

## Feedback (this is a BETA)

- **Speed on your PC**: `../Abyss-tools/abyss.sh --bench --deep` in Abyss, post `abyss-bench.txt` with
  [this issue form](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+12G32G-L448+on+YOUR_GPU)
  (change the title to your GPU), with your RAM size and SSD.
- **Quality in real use**: empty answers, wrong language, loops, places where it is weaker than unsloth's UD-IQ1_S. A
  prompt that reproduces it helps most.
- **What you would rather have**: a 64 GB RAM version, more code weight, other GPUs.

## License and credits

The original model's license ([qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)).
All credit for the model goes to the Qwen team; the NVFP4 weights the experts were made from are
[magnitudedev](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)'s, the non-expert tensors unsloth's.
Quantized with [trench-gguf](https://github.com/zephel01/trench-gguf).

---

## 日本語

- **12G32G = GPU（VRAM）12 GB + RAM 32 GB の PC 向け**（想定している構成で、速さを保証するものではありません）。
- 形式を混ぜたファイルです。エキスパートは IQ4_NL 57%, IQ1_S 28%, IQ2_XXS 15%（容量の割合）、平均 2.64 ビットで、IQ2_M くらいです。

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) の GGUF を、**GPU 12 GB + RAM 32 GB の PC** 向けに作りました。
エキスパートは NVFP4 の重みから LDLQ で量子化し直し（並びは unsloth の UD-IQ1_S と同じ）、各層 512 個のうち 448 個を残しています。
日本語は、量子化の調整、エキスパートの選び方、テストのすべてに入れています。Abyss と llama.cpp で動きます。

**BETA 版です。** 測ったのは 1 台だけです。ほかの GPU や RAM 容量での結果を教えてもらえると助かります。

| | 1 層のエキスパート | ダウンロード | KL（10 本） | 日本語の KL | 生成（コーディング 3 問） | 入力の処理（8K） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **L448（ここ）** | **448 / 512** | **67.5 GB** | 0.339 | **0.553** | **56.0 tok/s** | **1,822 tok/s** |
| unsloth UD-IQ1_S（参考） | 512 / 512 | 72.5 GB | 0.331 | 0.568 | 43.0 tok/s | 1,368 tok/s |

- 速さは RTX 5090 で GPU 12 GB + RAM 32 GB の PC を真似た設定です（VRAM 12 GB に制限、RAM の枠 20 GB を全体 28 GB に制限、CPU 8 スレッド、iGPU なし、ドラフターなし）。
  llama.cpp で UD-IQ1_S を `--n-cpu-moe 41`（VRAM 約 12 GB）にすると 27.6 tok/s でした。
- 実際の使い方（思考あり、seed を変えて 2 回）：日本語 5 問は 10/10 回答、コーディング 5 問は 3/5、3/5（unsloth は 5/5、3/5）、8K の情報探しは成功。
  「日本の三大随筆」では考えすぎが短くなりました（196 / 165 トークン、unsloth は 454 / 384）。
- ひらがな・カタカナの変換のコーディング問題は、どの版でもたいてい上限まで考え続けます。モデル自体のくせです。`max_tokens` を増やしてください。
- **空の回答**：最初の版で出ていた空の回答は、チャットのやりとりを調整に加えたこの版では、テストで出なくなりました。Abyss v1.3.106 にも防ぐ仕組み（`STRATA_GLM_END_GUARD`）があります。
- Abyss では `./setup-deepseek.sh --model qwen38 --quant 12G32G --download-model` で入ります（v1.3.106 以降）。
- llama.cpp（2026-10-29 の ac8e1694d）でもそのまま動きます。`--n-cpu-moe 41`（VRAM 10.9 GB）で、日本語 5 問すべてに回答、コーディング 3/5、8K の情報探しも成功。生成 26〜32 tok/s。
- 3 ファイルで 1 セットです。`-00001-of-00003.gguf` を指定してください。ファイル名やフォルダ名には、以前の呼び名 L448 が残っています。
- 速さの報告は [この issue フォーム](https://github.com/zephel01/Abyss/issues/new?template=bench.md&title=Bench:+Qwen3.8-Flash-Next+12G32G-L448+on+YOUR_GPU) にお願いします（タイトルの GPU 名を書き換えてください）。
- ライセンスは元のモデルと同じです（[qwen-community-1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)）。

### サイズの選び方（同じモデルのほかの版）

| リポジトリ | 向けの PC | ダウンロード | メモ |
| --- | --- | ---: | --- |
| **この版** | GPU 12 GB + RAM 32 GB | 67.5 GB | エキスパート 512 個中 448 個 |
| [Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) | GPU 16 GB + RAM 64 GB | 86.9 GB | 512 個すべて |
| [Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF) | GPU 24 GB + RAM 64 GB | 93.7 GB | 512 個すべて、品質重視 |
| [Qwen3.8-Flash-Next-DGX-Spark-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-DGX-Spark-GGUF) | DGX Spark（128 GB） | 86.9 GB | 16G64G と同じファイル |

以前のリポジトリ名は Qwen3.8-Flash-Next-12G32G-GGUF です（古い URL はここへ転送されます）。

名前の量子化の型は、エキスパートの平均ビット数にいちばん近い llama.cpp の型です。多くのファイルは複数の型を混ぜているので（上に内訳）、その名前の普通の量子化と同じものではありません。`E<数>` は層ごとに残したエキスパートの数です（減らしたときだけ付けます）。
