---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
---

# Qwen3.8-Flash-Next-mlx-8bit

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) を MLX 形式へ変換したもの（8bit 量子化）。

- 変換元: `Qwen/Qwen3.8-Flash-Next`（bf16 公式重み, revision de4b8e4）
- 変換ツール: mlx-vlm 0.6.17 / mlx 0.32.0
- サイズ: 186 GiB（9.018 bits per weight）

## 変換コマンド

```
python -m mlx_vlm convert --hf-path Qwen/Qwen3.8-Flash-Next \
    --mlx-path Qwen3.8-Flash-Next-mlx-8bit \
    -q --q-bits 8 --q-group-size 32
```

`--q-group-size 32` は必須。n-gram 埋め込みの最終次元が 160 で、既定の 64 では
`weight.shape[-1] % group_size != 0` により 128 シャード（51B パラメータ）が
量子化対象から外れ、bf16 のまま残る。

## 使い方

```
pip install -U mlx-vlm
python -m mlx_vlm generate --model ToPo-ToPo/Qwen3.8-Flash-Next-mlx-8bit \
    --prompt "この画像を説明してください" --image path/to/image.jpg --max-tokens 512
```

## MTP ドラフター（投機デコード）

ドラフターは別リポジトリ
[Qwen3.8-Flash-Next-MTP-bf16](https://huggingface.co/ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16)。
**mlx-vlm 0.7.0 以降が必要**（`qwen4_exp_mtp` はリリース版 0.6.17 に未収録）:

```
pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@main"

python -m mlx_vlm generate --model ToPo-ToPo/Qwen3.8-Flash-Next-mlx-8bit \
    --draft-model ToPo-ToPo/Qwen3.8-Flash-Next-MTP-bf16 \
    --draft-kind mtp --draft-block-size 2 --prompt "..." --max-tokens 512
```
