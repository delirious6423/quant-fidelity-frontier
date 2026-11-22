---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- moonbox
- qwen
- qwen3.8
- qwen3_5
- mlx-vlm
- vision
- apple-silicon
---

# Qwen3.8 27B MLX

MLX builds of [Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B), made by Moonbox for Apple Silicon Macs. Each precision is a folder; keep a folder's
files together. Only the parts listed below are converted: the rest of the pipeline comes from the original
repository unless the section below says otherwise.

## Files

The last column is how far each build is from the original PyTorch model on the same inputs (see the section below
for what was measured).

| Precision | File | Size | sha256 | Checked against the original |
|---|---|---|---|---|
| 8bit | [`8bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/chat_template.jinja) | 9.0 kB | `af16f654968d` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/config.json) | 4.9 kB | `8f80874ac3ad` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/generation_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/generation_config.json) | 0.2 kB | `e70c136c1b78` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00001-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00001-of-00018.safetensors) | 1.35 GB | `3342488ff837` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00002-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00002-of-00018.safetensors) | 2.54 GB | `404d08fa165a` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00003-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00003-of-00018.safetensors) | 1.62 GB | `5c03ebae5528` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00004-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00004-of-00018.safetensors) | 1.12 GB | `5535b45aae1b` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00005-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00005-of-00018.safetensors) | 2.12 GB | `b5c2d362cf0c` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00006-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00006-of-00018.safetensors) | 2.11 GB | `74176a57b4d8` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00007-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00007-of-00018.safetensors) | 1.12 GB | `e28a7f0263e3` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00008-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00008-of-00018.safetensors) | 2.11 GB | `a0517a05d380` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00009-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00009-of-00018.safetensors) | 1.12 GB | `45c0a7d0754c` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00010-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00010-of-00018.safetensors) | 2.11 GB | `65cac4dbc083` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00011-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00011-of-00018.safetensors) | 1.12 GB | `aa7d9ee9c143` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00012-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00012-of-00018.safetensors) | 2.11 GB | `8218f5bbc6a4` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00013-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00013-of-00018.safetensors) | 1.12 GB | `666ce0902f84` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00014-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00014-of-00018.safetensors) | 2.11 GB | `15f691151b10` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00015-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00015-of-00018.safetensors) | 1.12 GB | `84b63d3504ef` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00016-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00016-of-00018.safetensors) | 2.11 GB | `e29a67a39afc` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00017-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00017-of-00018.safetensors) | 1.12 GB | `acaafd50297a` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model-00018-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model-00018-of-00018.safetensors) | 1.35 GB | `68e8354a235e` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/model.safetensors.index.json) | 0.2 MB | `0d88a9d5a15d` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/mtp/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/mtp/config.json) | 3.0 kB | `1d0fae1de88b` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/mtp/model.safetensors) | 849.4 MB | `5e5698433454` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/mtp/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/mtp/tokenizer.json) | 12.8 MB | `0997f410c57a` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/mtp/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/mtp/tokenizer_config.json) | 10.0 kB | `207c03c41a06` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/mtp/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/mtp/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/preprocessor_config.json) | 0.4 kB | `27225450ac9c` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/processor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/processor_config.json) | 1.0 kB | `45fc17c8dd24` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/tokenizer.json) | 20.0 MB | `a5cd9732badc` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/tokenizer_config.json) | 2.5 kB | `4d99d075982a` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/video_preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/video_preprocessor_config.json) | 0.4 kB | `7768af27c1fa` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 8bit | [`8bit/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/8bit/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.0015 nats (p95 0.0036), top-1 agreement 98.0 %, perplexity 8.777 vs 8.753 |
| 6bit | [`6bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/chat_template.jinja) | 9.0 kB | `af16f654968d` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/config.json) | 4.9 kB | `f2490d237e42` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/generation_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/generation_config.json) | 0.2 kB | `e70c136c1b78` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00001-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00001-of-00018.safetensors) | 1.03 GB | `0b6b11c26471` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00002-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00002-of-00018.safetensors) | 2.16 GB | `704b9f8c13e9` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00003-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00003-of-00018.safetensors) | 1.24 GB | `b70560c8b4b3` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00004-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00004-of-00018.safetensors) | 853.0 MB | `082756ea6e2f` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00005-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00005-of-00018.safetensors) | 1.62 GB | `6af919f34590` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00006-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00006-of-00018.safetensors) | 1.62 GB | `69de1f59a1c1` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00007-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00007-of-00018.safetensors) | 856.8 MB | `95c19f1756c6` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00008-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00008-of-00018.safetensors) | 1.62 GB | `ecb72de9c01b` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00009-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00009-of-00018.safetensors) | 856.8 MB | `c497bd927f55` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00010-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00010-of-00018.safetensors) | 1.62 GB | `84d9946b1d45` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00011-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00011-of-00018.safetensors) | 856.8 MB | `68ae60e08ec5` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00012-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00012-of-00018.safetensors) | 1.62 GB | `690aba37556d` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00013-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00013-of-00018.safetensors) | 856.8 MB | `9aa246513b87` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00014-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00014-of-00018.safetensors) | 1.62 GB | `ed721e5fede3` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00015-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00015-of-00018.safetensors) | 856.8 MB | `a7893e72d19c` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00016-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00016-of-00018.safetensors) | 1.62 GB | `bc2ddfb504cc` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00017-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00017-of-00018.safetensors) | 856.8 MB | `1b4ca3741ebe` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model-00018-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model-00018-of-00018.safetensors) | 1.03 GB | `6d1cb300deec` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/model.safetensors.index.json) | 0.2 MB | `70d582f8be31` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/mtp/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/mtp/config.json) | 3.0 kB | `1d0fae1de88b` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/mtp/model.safetensors) | 849.4 MB | `5e5698433454` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/mtp/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/mtp/tokenizer.json) | 12.8 MB | `0997f410c57a` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/mtp/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/mtp/tokenizer_config.json) | 10.0 kB | `207c03c41a06` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/mtp/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/mtp/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/preprocessor_config.json) | 0.4 kB | `27225450ac9c` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/processor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/processor_config.json) | 1.0 kB | `45fc17c8dd24` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/tokenizer.json) | 20.0 MB | `a5cd9732badc` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/tokenizer_config.json) | 2.5 kB | `4d99d075982a` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/video_preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/video_preprocessor_config.json) | 0.4 kB | `7768af27c1fa` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 6bit | [`6bit/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/6bit/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.0059 nats (p95 0.014), top-1 agreement 95.9 %, perplexity 8.749 vs 8.753 |
| 4bit | [`4bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/chat_template.jinja) | 9.0 kB | `af16f654968d` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/config.json) | 4.9 kB | `14b65a0ee065` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/generation_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/generation_config.json) | 0.2 kB | `e70c136c1b78` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00001-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00001-of-00018.safetensors) | 715.2 MB | `7af52cb67aa5` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00002-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00002-of-00018.safetensors) | 1.78 GB | `1d9ba42d0a09` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00003-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00003-of-00018.safetensors) | 856.0 MB | `3d6e6821f017` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00004-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00004-of-00018.safetensors) | 590.6 MB | `9b0c4631cc00` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00005-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00005-of-00018.safetensors) | 1.12 GB | `90b06f7d23f1` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00006-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00006-of-00018.safetensors) | 1.12 GB | `66debb5e7d8e` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00007-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00007-of-00018.safetensors) | 593.2 MB | `7bc1bfec0180` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00008-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00008-of-00018.safetensors) | 1.12 GB | `bd43e7ca0db4` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00009-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00009-of-00018.safetensors) | 593.2 MB | `04a3682fe4b8` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00010-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00010-of-00018.safetensors) | 1.12 GB | `d6a9f9e33da4` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00011-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00011-of-00018.safetensors) | 593.2 MB | `7ad6af967f91` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00012-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00012-of-00018.safetensors) | 1.12 GB | `d6ce11374348` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00013-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00013-of-00018.safetensors) | 593.2 MB | `b0b6aa8a8754` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00014-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00014-of-00018.safetensors) | 1.12 GB | `e7f21a0db551` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00015-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00015-of-00018.safetensors) | 593.2 MB | `20fe5a89e0df` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00016-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00016-of-00018.safetensors) | 1.12 GB | `ab1937e42fa3` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00017-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00017-of-00018.safetensors) | 593.2 MB | `e08cb329dfb2` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model-00018-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model-00018-of-00018.safetensors) | 715.2 MB | `917105cdf405` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/model.safetensors.index.json) | 0.2 MB | `ce79cd7ebdc7` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/mtp/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/mtp/config.json) | 3.0 kB | `1d0fae1de88b` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/mtp/model.safetensors) | 849.4 MB | `5e5698433454` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/mtp/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/mtp/tokenizer.json) | 12.8 MB | `0997f410c57a` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/mtp/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/mtp/tokenizer_config.json) | 10.0 kB | `207c03c41a06` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/mtp/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/mtp/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/preprocessor_config.json) | 0.4 kB | `27225450ac9c` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/processor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/processor_config.json) | 1.0 kB | `45fc17c8dd24` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/tokenizer.json) | 20.0 MB | `a5cd9732badc` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/tokenizer_config.json) | 2.5 kB | `4d99d075982a` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/video_preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/video_preprocessor_config.json) | 0.4 kB | `7768af27c1fa` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 4bit | [`4bit/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/4bit/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.059 nats (p95 0.16), top-1 agreement 88.2 %, perplexity 8.908 vs 8.753 (experimental) |
| 3bit | [`3bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/chat_template.jinja) | 9.0 kB | `af16f654968d` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/config.json) | 4.9 kB | `536085b9e9cb` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/generation_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/generation_config.json) | 0.2 kB | `e70c136c1b78` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00001-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00001-of-00018.safetensors) | 556.2 MB | `8ee562137463` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00002-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00002-of-00018.safetensors) | 1.59 GB | `0c5bf324b6b1` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00003-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00003-of-00018.safetensors) | 665.9 MB | `5f012c2ba4cb` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00004-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00004-of-00018.safetensors) | 459.4 MB | `062c900a296b` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00005-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00005-of-00018.safetensors) | 873.0 MB | `f4d2ae6c5f2a` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00006-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00006-of-00018.safetensors) | 871.0 MB | `e638f3c73000` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00007-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00007-of-00018.safetensors) | 461.4 MB | `45492383163e` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00008-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00008-of-00018.safetensors) | 871.0 MB | `36a092aceeca` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00009-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00009-of-00018.safetensors) | 461.4 MB | `cfd07b1bccd8` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00010-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00010-of-00018.safetensors) | 871.0 MB | `e44232731a40` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00011-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00011-of-00018.safetensors) | 461.4 MB | `58444a272a9e` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00012-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00012-of-00018.safetensors) | 871.0 MB | `c562c9154ef8` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00013-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00013-of-00018.safetensors) | 461.4 MB | `f801dce1a932` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00014-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00014-of-00018.safetensors) | 871.0 MB | `0d76fe2daf20` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00015-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00015-of-00018.safetensors) | 461.4 MB | `6c47fe8bff0e` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00016-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00016-of-00018.safetensors) | 871.0 MB | `4e7f7fb85e89` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00017-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00017-of-00018.safetensors) | 461.4 MB | `f48daed2bd6a` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model-00018-of-00018.safetensors`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model-00018-of-00018.safetensors) | 556.2 MB | `3de5e42aced6` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/model.safetensors.index.json) | 0.2 MB | `e1cec1ea778e` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/preprocessor_config.json) | 0.4 kB | `27225450ac9c` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/processor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/processor_config.json) | 1.0 kB | `45fc17c8dd24` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/tokenizer.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/tokenizer.json) | 20.0 MB | `a5cd9732badc` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/tokenizer_config.json) | 2.5 kB | `4d99d075982a` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/video_preprocessor_config.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/video_preprocessor_config.json) | 0.4 kB | `7768af27c1fa` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |
| 3bit | [`3bit/vocab.json`](https://huggingface.co/moonbox-dev/Qwen3.8-27B-MLX/blob/main/3bit/vocab.json) | 6.7 MB | `ce99b4cb2983` | KL 0.25 nats (p95 0.67), top-1 agreement 78.3 %, perplexity 10.13 vs 8.753 (experimental) |

## How to run

Qwen3.8 27B (Qwen's 27.8B dense vision-language model: 48 Gated DeltaNet and 16 gated-attention layers, a
27-layer vision tower) converted to MLX for Apple Silicon, in four builds. Each folder is a complete mlx-vlm model:
weights, `config.json`, the publisher's tokenizer, chat template and image processor settings.

These files were converted by Moonbox from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision
`37561816`). They are not an official Qwen release. Pictures work (the vision tower is included, in bfloat16); the
original's multi-token-prediction layer is in `mtp/` of the 8, 6 and 4-bit folders as a drafter.

## How to run

With [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.4 or later:

```python
from huggingface_hub import snapshot_download
from mlx_vlm import load, generate

folder = snapshot_download("moonbox-dev/Qwen3.8-27B-MLX", allow_patterns="4bit/*") + "/4bit"
model, processor = load(folder)
msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
print(generate(model, processor, prompt, max_tokens=256, temperature=0.7).text)
```

or `python -m mlx_vlm.generate --model <folder> --prompt "..." --image picture.png`. mlx-lm 0.32 loads the same
folders for text only.

- **Chat template.** The publisher's: thinking is on unless `enable_thinking=False`; `reasoning_effort` is `xhigh`
  (the default), `medium` or `low`. Tools are OpenAI function schemas; the model calls them in Qwen's XML
  (`<tool_call><function=name><parameter=...>`).
- **One sequence at a time.** With mlx-vlm 0.7.4 the full-attention layers give a row of a batch other values than
  the same row alone (up to 0.5 in the hidden state, varying between runs); run one prompt at a time.
- **Memory and speed** on a 24 GB MacBook Air (M5), measured through Moonbox's server (greedy, thinking off): the
  4-bit build peaks at 17.2 GB with a 4K-token prompt and a picture and decodes 7.3 tokens a second (15.5 with the
  MTP drafter in `4bit/mtp/`, the same greedy text; 11.7 after a 4K-token prompt), prefilling 170-190 tokens a
  second; the 3-bit peaks at 14.1 GB and decodes 9.1 (MTP is slower at 3 bits, so `3bit/` has no drafter). The
  6-bit and 8-bit builds need a Mac with 32 and 48 GB or more; they were not run.
- **Speculative decoding.** `8bit/`, `6bit/` and `4bit/` carry `mtp/`, the original's multi-token-prediction layer
  split into an mlx-vlm drafter (bfloat16): `load_drafter(folder + "/mtp")` and pass `draft_model=..., draft_kind="mtp"`
  to `stream_generate`.
- Call `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` before loading on a Mac where the
  build is more than half the memory, or the weights may be paged out under pressure.

## The four builds

MLX affine quantization, group size 64, of the language model's linear layers, embedding and head; the vision tower,
norms, convolutions and the DeltaNet decay parameters are the publisher's (bfloat16, `A_log` float32).

| Build | Size | Mean KL (nats) | Top-1 agreement | Perplexity |
|---|---|---|---|---|
| `8bit` | 29.5 GB | 0.0015 | 98.0 % | 8.777 |
| `6bit` | 22.8 GB | 0.0059 | 95.9 % | 8.749 |
| `4bit` | 16.1 GB | 0.059 | 88.2 % | 8.908 |
| `3bit` | 12.7 GB | 0.25 | 78.3 % | 10.13 |

Against the BF16 original (perplexity 8.753), teacher-forced through mlx-vlm's own layers, on 16,368 positions of
English prose, Python and chat transcripts. The 8-bit and 6-bit builds are within the mark this lab uses (KL 0.01,
top-1 90 %); the 4-bit and 3-bit are below it and experimental: the 3-bit exists so the model fits a 24 GB Mac.
Mixed 3-bit builds (more bits in some layers, 13.4 and 14.7 GB) were measured too (KL 0.21 and 0.19) and are not
published: the 4-bit is far better for a little more memory.

## How these were made

The same as `mlx_vlm.convert` (checked byte for byte on Qwen3.5 0.8B), one source shard at a time: each shard of the
original was downloaded, converted to every build and deleted, so the 55.6 GB source was never on disk whole.

## How these were made

From [Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `37561816e6`: the original weights loaded in MLX and stored as bf16 or fp16, or
quantized with MLX's affine quantization (`8bit`, `6bit`, `4bit`, group size 64; layers MLX can't quantize, such as
convolutions and norms, stay bf16). Every build was checked against the original before upload; the `sha256` column
above is what the files must hash to.

## License

apache-2.0. These files are a derivative of the original model and carry its license: it is copied in this
repository, and it governs your use of these files. The original model card follows unchanged.

---

# Original model card

# Qwen3.8-27B

> [!Note]
> This repository contains model weights and configuration files for the post-trained model in the Hugging Face Transformers format. 
>
> These artifacts are compatible with Hugging Face Transformers, vLLM, SGLang, TokenSpeed, etc.

> [!Tip]
> For users seeking managed, scalable inference without infrastructure maintenance, the official Qwen API service is provided by [Qwen Cloud](https://www.qwencloud.com).
> In particular, **Qwen3.8-27B** will be available as a hosted version with more production features, e.g., 1M context length by default, official built-in tools. For more information, please refer to the [Qwen3.8-27B Overview](https://www.qwencloud.com/models/qwen3.8-27b). The service is coming soon. Stay tuned for updates.

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to date.

Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

## Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.


## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.


## Benchmark Results

### Text Performance
<style>
.vl-table th{font-size:15px!important;line-height:1.2;color:inherit!important}
.vl-table td:not(.benchmark-cell):not([colspan]){font-size:15px;line-height:1.2;vertical-align:middle;color:inherit!important}
.vl-table .benchmark-cell{padding:12px 10px 12px 18px!important;vertical-align:middle}
.vl-table .benchmark-capability{font-size:15px;font-weight:600;line-height:1.22;color:inherit!important}
.vl-table .benchmark-name{margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;opacity:.6}
.vl-table .metric-stack{display:flex;flex-direction:column;gap:7px;padding:3px 0}
.vl-table .metric-label{font-size:10px;font-weight:400;line-height:1.1;opacity:.55}
.vl-table .metric-value{margin-top:2px;font-size:15px;line-height:1.15;color:inherit!important}
</style>
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Coding</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic terminal coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Terminal Bench 2.1 (Terminus)</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">73.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">64.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>78.2</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-bench Pro</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>61.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">57.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.4</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Repo-level code generation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">NL2Repo-Bench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">42.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">36.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">41.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.6</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">DeepSWE 1.1</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>42.2</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">13.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">14.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">QwenSWEBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.0</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">49.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">59.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.8</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agent</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Long-horizon office work</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CoWorkBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>70.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">61.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">68.2</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Professional job tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">JobBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>33.4</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">21.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Frontier agentic tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Agents' Last Exam</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>20.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>42.9</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">10.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">27.3</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">13.2</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">33.6</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Instruction following</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">IFBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">69.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">77.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">GPQA Diamond</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">89.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">87.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.3</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multidisciplinary reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">HLE</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">30.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">24.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">34.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">22.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>40.0</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Competitive coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">LiveCodeBench v6</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>90.3</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">88.8</td>
</tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;opacity:0.7">
<ol style="margin:0;padding-left:20px">
<li>SWE-bench Pro: Except for Opus4.6 Max, which uses the officially reported score, all models are evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window. Problematic tasks were corrected, and all baseline models were re-evaluated on the refined benchmark.</li>
<li>NL2Repo-Bench: Evaluated with the Claude Code harness. To prevent reward hacking, we disable Bash commands that attempt to access the specific repository, such as pip download, pip install, and git clone.</li>
<li>DeepSWE 1.1: Evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window.</li>
<li>QwenSWEBench: In-house coding benchmark for evaluating models' software engineering capabilities. Evaluated with the Claude Code harness. Reporting avg@3 with an 8-hour timeout, max_tokens=32,768, temperature=1.0, and a 256K context window.</li>
<li>CoWorkBench: In-house cowork benchmark for evaluating long-horizon tasks across computer science, finance, law, medical, and other productivity domains.</li>
<li>HLE: Judged by GPT-4o.</li>
<li>The best result in each row is shown in bold.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol>
</div>
</div>

### VL Performance
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr><th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agentic Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Computer use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OSWorld-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>84.3</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">72.7</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Browser use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">WebArena-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>64.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">48.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">55.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Mobile use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">AndroidWorld</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>81.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">70.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">81.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.0</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Application recreation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RecreationBench</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.1</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">29.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.2</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal tool use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ClawEval-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">56.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">42.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">50.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>60.1</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">52.5</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">54.7</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>38.6</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">25.7</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.1</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual web development</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Vision2Web</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>62.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">45.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual math problem solving</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">MathVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.0</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>94.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.1</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.3</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">65.5</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">General visual reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">BabyVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>65.7</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">28.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">64.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">70.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">12.6</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific chart analysis</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CharXiv (RQ)</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">83.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.2</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">78.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.8</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">78.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">66.0</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Document intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OmniDocBench 1.5</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">91.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.4</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.4</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">75.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">86.6</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Real-world perception</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RealWorldQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">85.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">84.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>86.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.9</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Embodied intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ERQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">65.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>69.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">40.8</td></tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;opacity:0.7">
<ol style="margin:0;padding-left:20px">
<li>MathVision, BabyVision, and CharXiv (RQ): Where both settings are available, cells report “Without CI” and “With CI” separately; otherwise, only the available setting is shown. A small number of incorrect ground-truth annotations in MathVision and CharXiv (RQ) were corrected following manual verification, and all reported scores on those benchmarks were computed using the corrected annotations.</li>
<li>MathVision: Qwen3.8-27B is evaluated using the fixed prompt: “Please reason step by step, and put your final answer within <code>\boxed{}</code>.” For the remaining models, we report the higher score from two prompt variants—one with and one without the <code>\boxed{}</code> formatting requirement.</li>
<li>WebArena-Verified: Scores are computed with the official WebArena-Verified grader under the OSWorld scaffold.</li>
<li>RecreationBench: An in-house, long-horizon application-recreation benchmark designed to evaluate hybrid-agent capabilities across five platforms: desktop (Ubuntu, macOS, and Windows), mobile (Android), and the web.</li>
<li>ClawEval-MM: Scores are reported as “Pass@3 / average score.” Pass@3 is the percentage of tasks passed in at least one of three trials; the average score is the mean benchmark score across the three trials.</li>
<li>Vision2Web: Scores are averaged across the frontend, webpage, and website categories. Evaluations use the Claude Code harness and are judged by <code>gpt-5.4-2026-03-05</code>.</li>
<li>SWE-MM: Scores are evaluated on the Claude Code harness using the public dev split of SWE-bench Multimodal, with the modifications described in Appendix 8.3 of the Claude Opus 4.7 system card.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol></div>
</div>


## Quickstart

For streamlined integration, we recommend using Qwen3.8 via APIs.

### Serving Qwen3.8

> [!Important]
> Inference efficiency and throughput vary significantly across frameworks. 
> We recommend using the latest framework versions to ensure optimal performance and compatibility.
> For production workloads or high-throughput scenarios, dedicated serving engines such as SGLang, vLLM, or TokenSpeed are recommended.

Qwen3.8 can be deployed with popular inference frameworks, e.g.:

- [SGLang](https://www.sglang.io/): [Qwen3.8 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [vLLM](https://vllm.ai/): [Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [TokenSpeed](https://lightseek.org/tokenspeed/): [Qwen3.8 Recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8)


### API Usage

> [!Important]
> Qwen3.8 models operate in thinking mode by default, generating thinking content signified by `<think>\n...</think>\n\n` before producing the final response.
> To disable thinking content and obtain a direct response, refer to the examples [here](#instruct-or-non-thinking-mode).


> [!Tip]
> We recommend using the following sets of sampling parameters for generation:
> - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
> - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
>
> Please note that the support for sampling parameters varies according to inference frameworks.


Qwen3.8 comes with official support for `reasoning_effort`, which can be used to adjust reasoning depth and control cost:  
  - `xhigh` (default): for complex tasks demanding thorough analysis
  - `medium`: balancing accuracy and speed
  - `low`: efficient reasoning optimizing for speed and cost


In addition, `preserve_thinking` is enabled by default for all workloads for the best out-of-the-box experience. To disable preserved thinking, refer to the examples [here](#disable-preserved-thinking).

> [!Tip]
> In multi-turn agentic tasks, lower reasoning effort does not always reduce overall task completion time. Although it may produce faster per-turn responses, it can also lead to insufficient analysis, more failures, and repeated retries, which may increase total latency and token consumption.


#### Chat Completions API

The Chat Completions API can be used with most inference frameworks, as well as [Qwen Cloud](https://www.qwencloud.com/).
Before starting, make sure the OpenAI Python SDK is installed and the API key and the API base URL are configured, e.g.:
```shell
pip install -U openai

# Set the following accordingly
export OPENAI_BASE_URL='your-base-url'
export OPENAI_API_KEY='your-api-key'
```

##### Text-Only Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}]

completion = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,  # on by default
            "preserve_thinking": True, # on by default
        },
    },
    reasoning_effort="xhigh",  # xhigh by default; supported levels are xhigh, medium, and low
    stream=True,
    stream_options={"include_usage": True},
)

reasoning_content = ""
answer_content = ""
is_answering = False
print("\n" + "=" * 20 + "Reasoning" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
        continue

    delta = chunk.choices[0].delta

    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content
    elif hasattr(delta, "reasoning") and delta.reasoning is not None:
        if not is_answering:
            print(delta.reasoning, end="", flush=True)
        reasoning_content += delta.reasoning

    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Answer" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content

messages.append({
    "role": "assistant",
    "content": answer_content,
    "reasoning_content": reasoning_content,
    "reasoning": reasoning_content,
})
```


##### Image Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
)
print("Chat response:", chat_response)
```

##### Video Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4"
                }
            },
            {
                "type": "text",
                "text": "How many porcelain jars were discovered in the niches located in the primary chamber of the tomb?"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
)

# When vLLM is launched with `--media-io-kwargs '{"video": {"num_frames": -1}}'`,
# video frame sampling can be configured via `extra_body` (e.g., by setting `fps`).
# This feature is currently supported only in vLLM.
#
# By default, `fps=2` and `do_sample_frames=True`.
# With `do_sample_frames=True`, you can customize the `fps` value to set your desired video sampling rate.
# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen3.8-27B",
#     messages=messages,
#     extra_body={
#         "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
#     }, 
# )

print("Chat response:", chat_response)
```


##### Instruct (or Non-Thinking) Mode

Qwen3.8-27B will think by default before responding.
You can obtain a direct response from the model without thinking by configuring the API parameters. 
For example,
```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
                }
            },
            {
                "type": "text",
                "text": "Where is this?"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }, 
)
print("Chat response:", chat_response)
```

> [!Note]
> If you are using APIs from Qwen Cloud, in addition to changing `model`, please use `"enable_thinking": False` instead of `"chat_template_kwargs": {"enable_thinking": False}`.


##### Disable Preserved Thinking


By default, Qwen3.8 retains thinking blocks from all historical messages, maintaining a complete reasoning trace across the conversation. This behavior, known as preserved thinking, ensures full context continuity and is especially beneficial for agent scenarios where decision consistency and reduced redundant reasoning are critical. It also improves KV cache utilization, optimizing inference efficiency in both thinking and non-thinking modes.

If you prefer to retain only the thinking blocks from the latest user message, you can disable this behavior by setting `preserve_thinking` to `False`:

```python
from openai import OpenAI

# Configured by environment variables
client = OpenAI()
messages = [...]
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"preserve_thinking": False},
    },
)
print("Chat response:", chat_response)
```

> [!Note]
> If you are using APIs from Qwen Cloud, in addition to changing `model`, please use `"preserve_thinking": False` directly instead of wrapping it in `chat_template_kwargs`.


## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**: To optimize performance on agentic tasks, we recommend allocating sufficient output length to allow the model to generate detailed and comprehensive responses. For frameworks that support separate token limits for internal reasoning and final outputs, we suggest the following configuration within the 1M context length:
    
    - Reasoning Content: Set the maximum output length to 262,144 tokens.
    - Final Response: Set the maximum output length to 131,072 tokens.

    These settings provide the necessary capacity for complex reasoning while ensuring ample space for high-quality final deliverables.

3. **Processing Ultra-Long Texts**: Qwen3.8-27B natively supports context lengths of up to 262,144 tokens. For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively, e.g., YaRN.

    YaRN is currently supported by several inference frameworks, e.g., vLLM, SGLang, and TokenSpeed. 
    In general, there are two approaches to enabling YaRN for supported frameworks:

    - Modifying the model configuration file:
        
        In the `config.json` file, change the `rope_parameters` fields in `text_config` to:
        ```json
        {
            "mrope_interleaved": true,
            "mrope_section": [
                11,
                11,
                10
            ],
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        }
        ```

    - Passing command line arguments:

        For vLLM, you can use
        ```shell
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
        ```

        For SGLang, you can use
        ```shell
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --context-length 1000000
        ```

        For TokenSpeed, you can use
        ```shell
        TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 tokenspeed serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
        ```
    
    > [!NOTE]
    > All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
    > We advise modifying the `rope_parameters` configuration only when processing long contexts is required. 
    > It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 524,288 tokens, it would be better to set `factor` as 2.0. 


4. **Long Video Understanding**: To optimize inference efficiency for plain text and images, the `size` parameter in the released `video_preprocessor_config.json` is conservatively configured. It is recommended to set the `longest_edge` parameter in the video_preprocessor_config file to 469,762,048 (corresponding to 224k video tokens) to enable higher frame-rate sampling for hour-scale videos and thereby achieve superior performance. For example,
    ```json
    {"longest_edge": 469762048, "shortest_edge": 4096}
    ```

    Alternatively, override the default values via engine startup parameters. For implementation details, refer to: [vLLM](https://github.com/vllm-project/vllm/pull/34330) / [SGLang](https://github.com/sgl-project/sglang/pull/18467).


## Citation

If you find our work helpful, feel free to give us a cite.


```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
