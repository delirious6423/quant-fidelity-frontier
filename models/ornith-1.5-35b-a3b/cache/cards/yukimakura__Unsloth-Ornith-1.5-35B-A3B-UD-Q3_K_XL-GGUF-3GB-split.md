---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
- peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- imatrix
- unsloth-dynamic
- quantized
- qwen35moe
- moe
- ornith
- split-gguf
---

# Unsloth-Ornith-1.5-35B-A3B UD-Q3_K_XL GGUF (3GB split shards)

[peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B](https://huggingface.co/peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B) の
`Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL.gguf`（16,845,510,880 バイト）を、`llama.cpp` の `llama-gguf-split` で
**1ファイルあたり最大3GB** に分割して再アップロードしたものです。

**量子化のやり直しはしておらず、テンソルデータは元ファイルとビット単位で同一**です
（分割はロスレスで、精度は一切変わりません）。

- Original model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
- GGUF quantization (imatrix / Unsloth-Dynamic 方式): [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll)
- Split: `llama-gguf-split --split --split-max-size 3G` (llama.cpp `b10408`)

> ### Unsloth とは無関係です / Not affiliated with Unsloth
> ファイル名の `Unsloth-` は**量子化手法の名前**を指しているだけで、これは Unsloth 公式のリリースではなく、
> Unsloth による承認・関与も一切ありません。元リポジトリ
> [peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B](https://huggingface.co/peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B)
> が明記している通り、Unsloth の [Dynamic GGUF](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)
> の手法を有志が再現した非公式ビルドです。本リポジトリはそれを**分割しただけ**のミラーで、
> 元ファイル名をそのまま維持しています。Unsloth 公式ビルドが公開された場合はそちらを利用してください。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL-00001-of-00006.gguf` … | UD-Q3_K_XL 本体（分割シャード・全00006個） |

元リポジトリに `mmproj`（画像入力用プロジェクタ）はありません。テキスト生成専用モデルです。

## 使い方

### llama.cpp

**先頭シャードだけ**を指定すれば、残りは自動的に読み込まれます（結合作業は不要）。
同じディレクトリに全シャードが揃っている必要があります。

```bash
hf download yukimakura/Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL-GGUF-3GB-split --local-dir ./Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL

llama-cli -m ./Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL/Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL-00001-of-00006.gguf
```

### 1ファイルに結合したい場合

```bash
llama-gguf-split --merge \
  Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL-00001-of-00006.gguf \
  Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL.gguf
```

結合後のテンソルデータは元の `Unsloth-Ornith-1.5-35B-A3B-UD-Q3_K_XL.gguf` とビット単位で同一です
（全 16,834,521,600 バイトを実際に比較して確認済み）。

なお `llama-gguf-split --merge` は `split.no` / `split.count` / `split.tensors.count` の
メタデータを残す仕様のため、結合後のファイル全体のサイズは元より96バイト大きくなります（16,845,510,880 → 16,845,510,976）
（推論結果には影響しません）。

## ライセンス

MIT（元リポジトリ [peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B](https://huggingface.co/peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B) に従います）。
