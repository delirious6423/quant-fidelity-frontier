---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- gguf
- qwen3_5
- split-gguf
- llama.cpp
---

# Qwen3.8-27B-Q4_K_M GGUF (2GB split shards)

[lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) の
`Qwen3.8-27B-Q4_K_M.gguf` を、`llama.cpp` の `llama-gguf-split` で **1ファイルあたり最大2GB** に分割して
再アップロードしたものです。**量子化のやり直しはしておらず、テンソルデータは元ファイルとビット単位で同一**です
（分割はロスレスで、精度は一切変わりません）。

- Model creator: [Qwen](https://huggingface.co/Qwen)
- Original model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- GGUF quantization: [LM Studio team](https://huggingface.co/lmstudio-community) (llama.cpp `b10430`)
- Split: `llama-gguf-split --split --split-max-size 2G` (llama.cpp `b10408`)

## ファイル

| ファイル | 内容 |
| --- | --- |
| `Qwen3.8-27B-Q4_K_M-00001-of-00009.gguf` … | Q4_K_M 本体（分割シャード） |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 画像入力用プロジェクタ（分割なし・元ファイルそのまま） |

## 使い方

### llama.cpp

**先頭シャードだけ**を指定すれば、残りは自動的に読み込まれます（結合作業は不要）。

```bash
huggingface-cli download yukimakura/Qwen3.8-27B-Q4_K_M-GGUF-2GB-split --local-dir ./Qwen3.8-27B-Q4_K_M

llama-cli -m ./Qwen3.8-27B-Q4_K_M/Qwen3.8-27B-Q4_K_M-00001-of-00009.gguf \
  --mmproj ./Qwen3.8-27B-Q4_K_M/mmproj-Qwen3.8-27B-BF16.gguf
```

### 1ファイルに結合したい場合

```bash
llama-gguf-split --merge \
  Qwen3.8-27B-Q4_K_M-00001-of-00009.gguf \
  Qwen3.8-27B-Q4_K_M.gguf
```

結合後のテンソルデータは元の `Qwen3.8-27B-Q4_K_M.gguf` とビット単位で同一です
（全 16,799,719,424 バイトを実際に比較して確認済み）。

なお `llama-gguf-split --merge` は `split.no` / `split.count` / `split.tensors.count` の
メタデータを残す仕様のため、結合後のファイル全体のサイズは元より64バイト大きくなります
（推論結果には影響しません）。

## ライセンス

Apache-2.0（元モデル [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) に従います）。
