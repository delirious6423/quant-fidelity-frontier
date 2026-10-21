---
license: other
base_model: Qwen/Qwen3.8-27B
tags:
  - vision-language
  - image-text-to-text
  - int4
  - gptq
  - quantization
  - onecompression
  - qep
  - autobit
library_name: transformers
---

# Qwen3.8-27B int4 VLM（QEP + AutoBit）

Fujitsu OneCompressionで量子化したQwen3.8-27Bの画像対応版。
既存のテキストモデルに、4bit化した画像エンコーダーを追加している。
単一の32 GB GPUにモデル全体を配置できる。

## 構成

| 部分 | 量子化 | モジュール数 |
|---|---|---:|
| テキスト | AutoBitによる4/8bit混在GPTQ + QEP | 4bit: 405、8bit: 91 |
| 画像 | 4bit GPTQ | 111 |

- ベースモデル: `Qwen/Qwen3.8-27B`
- ベースリビジョン: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- テキスト量子化リビジョン: `09cc5d01896ee4dec37027ff1668a4337929ad65`
- 量子化済み重み総量: 18,435,891,336 bytes
- 画像重みの平均コサイン類似度: 0.994944
- 画像重みの最小コサイン類似度: 0.981646
- 画像重みの平均正規化二乗誤差: 0.0099475

既存の`model.safetensors`はテキスト側の第1シャードとしてそのまま再利用し、
`model-00002-of-00002.safetensors`に画像側の重みを収録している。
読み込み時は`model.safetensors.index.json`を使用する。

## 読み込み

重みはOneCompression専用のGPTQ形式なので、通常のTransformersローダーではなく、
VLM対応の`onecompression-runtime`を使用する。

```python
from transformers import AutoProcessor
from onecomp_runtime.models.qwen38_vlm import load_int4_qwen38_vlm

checkpoint = "/path/to/Qwen3.8-27B-int4-qep"
model = load_int4_qwen38_vlm(
    checkpoint,
    device="cuda:0",
    backend="fused",
    fused_8bit="dequant",
)
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
```

`config.json`の`language_model_only`は`false`で、27層の画像エンコーダーを含む。
画像前処理には`preprocessor_config.json`と`video_preprocessor_config.json`を使用する。

## 量子化の詳細

テキスト側はAutoBitで選んだ405個の4bitモジュールと91個の8bitモジュールからなる。
画像側はattention、MLP、merger、位置埋め込みを含む111モジュールを4bit化した。
テキスト側の量子化条件は`quantization_summary.json`、各画像モジュールの誤差指標と
group sizeは`vision_quantization_summary.json`に記録している。

Kizuna Intelligence
