---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- moonbox
- gemma
- gemma4
- mlx-vlm
- vision
- moe
- apple-silicon
- qat
---

# Gemma 4 26B A4B MLX

MLX builds of [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) (Google's instruction-tuned model), made by Moonbox for Apple Silicon Macs. Each
build is a folder and a complete [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) model: weights, `config.json`,
Google's tokenizer, chat template and processor settings, and `mtp/`, Google's multi-token-prediction drafter for
speculative decoding. These files are not an official Google release.

## Files

The last column is how far each build is from the original bf16 model (see "Quality" below for what was measured).

| Precision | File | Size | sha256 | Against the original |
|---|---|---|---|---|
| 8bit | [`8bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/chat_template.jinja) | 18.7 kB | `ae53464bf3be` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/config.json) | 0.1 MB | `61629cc4299f` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/generation_config.json) | 0.2 kB | `d4226bbe3117` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00001.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00001.safetensors) | 1.93 GB | `60623eb8392d` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00002.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00002.safetensors) | 865.0 MB | `c60810f3737d` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00003.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00003.safetensors) | 865.0 MB | `e3567fafdfa7` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00004.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00004.safetensors) | 865.0 MB | `5a8803daa775` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00005.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00005.safetensors) | 865.0 MB | `0c0b9a472180` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00006.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00006.safetensors) | 865.0 MB | `7d01273635a8` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00007.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00007.safetensors) | 880.3 MB | `5e6fe80a93e8` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00008.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00008.safetensors) | 865.0 MB | `d1bb7c5297b9` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00009.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00009.safetensors) | 865.0 MB | `5412d5099a1c` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00010.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00010.safetensors) | 865.0 MB | `8d2058c181a0` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00011.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00011.safetensors) | 865.0 MB | `ef9ce480765a` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00012.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00012.safetensors) | 865.0 MB | `38208c11e529` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00013.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00013.safetensors) | 880.3 MB | `2f10a2d7cd93` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00014.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00014.safetensors) | 865.0 MB | `77c76bcd6f56` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00015.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00015.safetensors) | 865.0 MB | `567d2639f107` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00016.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00016.safetensors) | 865.0 MB | `6ca34cb8ff11` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00017.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00017.safetensors) | 865.0 MB | `6c580c4b2119` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00018.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00018.safetensors) | 865.0 MB | `a40a84a74872` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00019.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00019.safetensors) | 880.3 MB | `0be1835950f9` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00020.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00020.safetensors) | 865.0 MB | `3ef95e5a8aa3` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00021.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00021.safetensors) | 865.0 MB | `548996c7171e` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00022.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00022.safetensors) | 865.0 MB | `e94362c4558f` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00023.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00023.safetensors) | 865.0 MB | `606c140b9d1f` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00024.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00024.safetensors) | 865.0 MB | `7c31b02c91f5` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00025.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00025.safetensors) | 880.3 MB | `ac909e179851` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00026.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00026.safetensors) | 865.0 MB | `710fcba69ca2` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00027.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00027.safetensors) | 865.0 MB | `4cb545034a8c` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00028.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00028.safetensors) | 865.0 MB | `05e41435a7cf` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00029.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00029.safetensors) | 865.0 MB | `3607f2fef5aa` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00030.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00030.safetensors) | 865.0 MB | `e84cac17275a` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model-00031.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model-00031.safetensors) | 880.3 MB | `a1677fa4fa75` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/model.safetensors.index.json) | 0.2 MB | `541db40a58d2` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/mtp/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/mtp/config.json) | 2.3 kB | `f393dd4bbdbb` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/mtp/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/mtp/generation_config.json) | 0.3 kB | `8e58004dc0e2` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/mtp/model.safetensors) | 839.4 MB | `006c8713b29b` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/processor_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/processor_config.json) | 1.3 kB | `daf597419c03` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/tokenizer.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/tokenizer.json) | 32.2 MB | `cc8d3a0ce364` | KL 0.27 nats, top-1 90.1 % |
| 8bit | [`8bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/8bit/tokenizer_config.json) | 3.7 kB | `ae79d5349f54` | KL 0.27 nats, top-1 90.1 % |
| 6bit | [`6bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/chat_template.jinja) | 18.7 kB | `ae53464bf3be` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/config.json) | 0.1 MB | `7c440678e410` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/generation_config.json) | 0.2 kB | `d4226bbe3117` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00001.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00001.safetensors) | 1.75 GB | `35ddaf91b14b` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00002.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00002.safetensors) | 1.32 GB | `787352bc11c5` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00003.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00003.safetensors) | 1.32 GB | `d27dee569adc` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00004.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00004.safetensors) | 1.33 GB | `3fbadb68cbb6` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00005.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00005.safetensors) | 1.32 GB | `86098cfe46ee` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00006.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00006.safetensors) | 1.32 GB | `79525100a13f` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00007.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00007.safetensors) | 1.33 GB | `7775e382fcfc` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00008.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00008.safetensors) | 1.32 GB | `8041a50e1e5f` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00009.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00009.safetensors) | 1.32 GB | `a4b254306924` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00010.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00010.safetensors) | 1.33 GB | `3cb5da34c188` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00011.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00011.safetensors) | 1.32 GB | `c198ceabf752` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00012.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00012.safetensors) | 1.32 GB | `c4edf4a27780` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00013.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00013.safetensors) | 1.33 GB | `281d5d589d9d` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00014.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00014.safetensors) | 1.32 GB | `706edd077ad5` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00015.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00015.safetensors) | 1.32 GB | `24084adfa04e` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model-00016.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model-00016.safetensors) | 1.33 GB | `a12f66bbca32` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/model.safetensors.index.json) | 0.2 MB | `257ff38b0ebd` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/mtp/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/mtp/config.json) | 2.3 kB | `f393dd4bbdbb` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/mtp/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/mtp/generation_config.json) | 0.3 kB | `8e58004dc0e2` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/mtp/model.safetensors) | 839.4 MB | `006c8713b29b` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/processor_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/processor_config.json) | 1.3 kB | `daf597419c03` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/tokenizer.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/tokenizer.json) | 32.2 MB | `cc8d3a0ce364` | KL 0.51 nats, top-1 85.3 % |
| 6bit | [`6bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/6bit/tokenizer_config.json) | 3.7 kB | `ae79d5349f54` | KL 0.51 nats, top-1 85.3 % |
| 4bit | [`4bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/chat_template.jinja) | 18.7 kB | `ae53464bf3be` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/config.json) | 0.1 MB | `d59d1567e953` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/generation_config.json) | 0.2 kB | `d4226bbe3117` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00001.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00001.safetensors) | 1.75 GB | `35ddaf91b14b` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00002.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00002.safetensors) | 950.9 MB | `9bdb6c7fb12e` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00003.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00003.safetensors) | 1.05 GB | `8fd5e05fc27f` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00004.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00004.safetensors) | 966.2 MB | `8d6a04fdffd6` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00005.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00005.safetensors) | 950.9 MB | `080789844653` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00006.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00006.safetensors) | 950.9 MB | `fc98c104bde6` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00007.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00007.safetensors) | 966.2 MB | `30e8669bc34e` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00008.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00008.safetensors) | 950.9 MB | `539e18209648` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00009.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00009.safetensors) | 950.9 MB | `b3d979395315` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00010.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00010.safetensors) | 1.06 GB | `d8007289cd8a` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00011.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00011.safetensors) | 1.05 GB | `0bb09962b950` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00012.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00012.safetensors) | 950.9 MB | `56c0f2851649` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00013.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00013.safetensors) | 1.16 GB | `95e21d976d20` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00014.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00014.safetensors) | 1.05 GB | `95f9b75405ee` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00015.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00015.safetensors) | 1.24 GB | `1cbd591be3ee` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model-00016.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model-00016.safetensors) | 1.26 GB | `fa6923210149` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/model.safetensors.index.json) | 0.2 MB | `12d830c0ec91` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/mtp/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/mtp/config.json) | 2.3 kB | `f393dd4bbdbb` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/mtp/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/mtp/generation_config.json) | 0.3 kB | `8e58004dc0e2` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/mtp/model.safetensors) | 839.4 MB | `006c8713b29b` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/processor_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/processor_config.json) | 1.3 kB | `daf597419c03` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/tokenizer.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/tokenizer.json) | 32.2 MB | `cc8d3a0ce364` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit | [`4bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit/tokenizer_config.json) | 3.7 kB | `ae79d5349f54` | KL 0.77 nats, top-1 81.0 %; MMLU-Pro 60.4 % (bf16 57.9; Unsloth's UD-MLX-4bit: KL 1.01, 78.2 %, 57.9 %) |
| 4bit-qat | [`4bit-qat/chat_template.jinja`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/chat_template.jinja) | 18.7 kB | `ae53464bf3be` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/config.json) | 0.1 MB | `978113dfad7b` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/generation_config.json) | 0.2 kB | `d4226bbe3117` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00001.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00001.safetensors) | 1.61 GB | `4a5e4deb61ae` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00002.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00002.safetensors) | 1.02 GB | `f7303d5e61b5` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00003.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00003.safetensors) | 1.02 GB | `e26cfccb4fd6` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00004.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00004.safetensors) | 1.03 GB | `ff049ba4e5cd` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00005.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00005.safetensors) | 1.02 GB | `0d334789862d` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00006.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00006.safetensors) | 1.02 GB | `c4b0b447a6a1` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00007.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00007.safetensors) | 1.03 GB | `29e29e603c13` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00008.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00008.safetensors) | 1.02 GB | `f3a5a92e5b86` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00009.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00009.safetensors) | 1.02 GB | `2dd1a9d5e6a2` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00010.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00010.safetensors) | 1.03 GB | `1c27d7608809` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00011.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00011.safetensors) | 1.02 GB | `85ff19f5e0a5` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00012.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00012.safetensors) | 1.02 GB | `c80b5ccce7b2` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00013.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00013.safetensors) | 1.03 GB | `f15410ff81eb` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00014.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00014.safetensors) | 1.02 GB | `bd7b00f0942c` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00015.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00015.safetensors) | 1.02 GB | `8b87ce103457` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model-00016.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model-00016.safetensors) | 1.03 GB | `317bee896e5a` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/model.safetensors.index.json) | 0.2 MB | `818aa23637e7` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/mtp/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/mtp/config.json) | 2.4 kB | `23d2bc4a8920` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/mtp/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/mtp/generation_config.json) | 0.2 kB | `fb53f4c64e58` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/mtp/model.safetensors) | 839.4 MB | `c082cc581c3e` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/processor_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/processor_config.json) | 1.3 kB | `daf597419c03` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/tokenizer.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/tokenizer.json) | 32.2 MB | `cc8d3a0ce364` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 4bit-qat | [`4bit-qat/tokenizer_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/4bit-qat/tokenizer_config.json) | 3.7 kB | `ae79d5349f54` | Google's QAT model: KL 0.70 nats, top-1 77.8 % from the original; MMLU-Pro 57.5 % |
| 3bit | [`3bit/chat_template.jinja`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/chat_template.jinja) | 18.7 kB | `ae53464bf3be` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/config.json) | 0.1 MB | `f806e395678b` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/generation_config.json) | 0.2 kB | `d4226bbe3117` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00001.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00001.safetensors) | 1.75 GB | `35ddaf91b14b` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00002.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00002.safetensors) | 756.1 MB | `74b7c9b762ab` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00003.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00003.safetensors) | 756.1 MB | `c7d3a57a7937` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00004.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00004.safetensors) | 771.4 MB | `1d71d8df8beb` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00005.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00005.safetensors) | 756.1 MB | `151225335d98` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00006.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00006.safetensors) | 756.1 MB | `95c45c5cfd6f` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00007.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00007.safetensors) | 771.4 MB | `c522cd4aa45a` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00008.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00008.safetensors) | 756.1 MB | `1a92d86458ea` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00009.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00009.safetensors) | 756.1 MB | `d2898b04af83` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00010.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00010.safetensors) | 771.4 MB | `ac49e23f3925` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00011.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00011.safetensors) | 756.1 MB | `d4ef18fb08f7` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00012.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00012.safetensors) | 756.1 MB | `1e436db86ff1` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00013.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00013.safetensors) | 771.4 MB | `c099b4666aa2` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00014.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00014.safetensors) | 756.1 MB | `2d7d80706cee` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00015.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00015.safetensors) | 853.5 MB | `607914953531` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model-00016.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model-00016.safetensors) | 868.8 MB | `6c9417d0810c` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/model.safetensors.index.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/model.safetensors.index.json) | 0.2 MB | `40c6389f7738` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/mtp/config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/mtp/config.json) | 2.3 kB | `f393dd4bbdbb` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/mtp/generation_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/mtp/generation_config.json) | 0.3 kB | `8e58004dc0e2` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/mtp/model.safetensors`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/mtp/model.safetensors) | 839.4 MB | `006c8713b29b` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/processor_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/processor_config.json) | 1.3 kB | `daf597419c03` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/tokenizer.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/tokenizer.json) | 32.2 MB | `cc8d3a0ce364` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |
| 3bit | [`3bit/tokenizer_config.json`](https://huggingface.co/moonbox-dev/gemma-4-26B-A4B-it-MLX/blob/main/3bit/tokenizer_config.json) | 3.7 kB | `ae79d5349f54` | KL 2.03 nats, top-1 66.0 %; MMLU-Pro 53.3 % |

## How to run

With [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.4 or later:

```python
from huggingface_hub import snapshot_download
from mlx_vlm import load, generate

folder = snapshot_download("moonbox-dev/gemma-4-26B-A4B-it-MLX", allow_patterns="4bit/*") + "/4bit"
model, processor = load(folder)
msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
print(generate(model, processor, prompt, max_tokens=256, temperature=1.0, top_p=0.95, top_k=64).text)
```

or `python -m mlx_vlm.generate --model <folder> --prompt "..." --image picture.png` (`--audio clip.wav` for sound on
E2B, E4B and 12B).

- **Chat template.** Google's: thinking is off unless `enable_thinking=True` (it puts `<|think|>` at the top of the
  system turn); the reply's thought comes in `<|channel>thought ... <channel|>` before the answer. Keep only the
  answers in a multi-turn history (the template drops earlier thoughts itself). Tools are OpenAI function schemas;
  the model calls them as `<|tool_call>call:name{...}<tool_call|>` (mlx-vlm's `tools.parsers.gemma4` reads them).
  Google's sampling: temperature 1.0, top_p 0.95, top_k 64. Pictures and sound go before the text; sound up to 30 s
  (E2B, E4B, 12B).
- **Speculative decoding.** Every folder has `mtp/`, Google's MTP drafter (`4bit-qat/mtp/` is the QAT drafter, as
  Google asks for QAT models): `from mlx_vlm.speculative.drafters import load_drafter`, then
  `draft_model, kind = load_drafter(folder + "/mtp")` and pass `draft_model=..., draft_kind=kind` to
  `stream_generate`. The verification's numerics differ slightly from plain decoding, so greedy text can part
  after a few dozen tokens; the answers are as good.

## Which build

| Build | Size | For | What |
|---|---|---|---|
| `8bit` | 28.83 GB | 64 GB Macs and up (36 GB: slow) | 8-bit (vision encoder bfloat16) |
| `6bit` | 22.52 GB | 48 GB Macs and up (32 GB: slow) | 6-bit (vision encoder bfloat16) |
| `4bit` | 18.13 GB | 32 GB Macs and up (24 GB: slow) | 4-bit dynamic: attention 8-bit, embeddings 6-bit (vision encoder bfloat16) |
| `4bit-qat` | 17.79 GB | 32 GB Macs and up (24 GB: slow) | 4-bit, Google's QAT weights (vision encoder bfloat16) |
| `3bit` | 14.23 GB | 32 GB Macs and up (18 GB: slow) | 3-bit dynamic (vision encoder bfloat16) |

## Quality

Measured against the bf16 original through mlx-vlm's own layers (teacher-forced, one layer at a time, bf16
activations as at run time):

- **KL and top-1** on 16,368 positions of Gemma 4 chats: 16 prompts (code, maths, explanations, writing, French,
  Spanish, Chinese, half with thinking on) answered by the bf16 Gemma 4 E2B with Google's sampling, the same text
  for every size. Gemma 4's instruct models are very sure of themselves on text a person wrote (E2B: perplexity 86
  on Moonbox's docs), so text in their own voice measures what a chat sees.
- **MMLU-Pro**: the same questions for every build (over every category), thinking off, the answer letter read from
  the logits: a quick probe, not the model card's protocol (which thinks first), so the absolute numbers are lower
  than Google's.

| Build | KL (nats) | Top-1 | MMLU-Pro |
|---|---|---|---|
| bf16 (the original) | 0 | 100 % | 57.9 % |
| `8bit` | 0.27 | 90.1 % |  |
| `6bit` | 0.51 | 85.3 % |  |
| `4bit` | 0.77 | 81.0 % | 60.4 % |
| `4bit-qat` | 0.70 (Google's QAT model) | 77.8 % | 57.5 % |
| `3bit` | 2.03 | 66.0 % | 53.3 % |
| Unsloth's `gemma-4-26b-a4b-it-UD-MLX-4bit` (16.1 GB, for comparison) | 1.01 | 78.2 % | 57.9 % |
| plain 4-bit (not published) | 1.92 | 69.0 % | 57.5 % |
| plain 3-bit (not published) | 5.08 | 47.2 % | |

The chat text is the E2B's, far from this model's own voice (reference perplexity 67), which makes every KL here large; compare the builds with each other. MMLU-Pro: 240 questions; no build differs significantly from bf16 (the 3-bit: McNemar p = 0.06).

The QAT build is Google's quantization-aware-trained model: QAT retrained the weights for a 4-bit grid (blocks of
32, codes -8..7 times a step), so it differs from the original by about as much as a 5-bit build would, while
answering the questions above at least as well. This build holds that model on its own grid (the step fitted per
block: within 0.08 % of Google's QAT weights, bf16 rounding); llama.cpp's Q4_0 rule (step = extreme / -8) lands on
another grid for a third of the weights.

The dynamic 4-bit follows Unsloth's UD-MLX recipe in shape (attention and, on E2B / E4B, the per-layer-input gates
at 8 bits, the embeddings at 6, the MLPs at 4) and chooses its 5 and 6-bit layers from this model's own measured
sensitivity. Unsloth's own `gemma-4-26b-a4b-it-UD-MLX-4bit` (16.1 GB), measured the same way, is in the table: this dynamic 4-bit is closer to the original (KL 0.77 against 1.01) for 1.1 GB more (it keeps the router at 8 bits, Unsloth's at 4, and gives the most sensitive layers' experts 5 or 6).

## Speed and memory

On a 24 GB MacBook Air (M5) after a cool-down, through Moonbox's server (greedy, thinking off; the build's MTP drafter where the server uses it; peak = the server process's footprint, a 6.9K-token chat turn or a picture, whichever is higher):

| Build | Decode (short / after 4K) | Prefill (4K) | Peak memory | Runs on |
|---|---|---|---|---|
| `8bit` | not run here | | 29.2 GB (estimated) | 64 GB Macs and up (36 GB: slow) |
| `6bit` | not run here | | 22.9 GB (estimated) | 48 GB Macs and up (32 GB: slow) |
| `4bit` | 37.7 / 34.7 tok/s | 612 tok/s | 17.6 GB | 32 GB Macs and up (24 GB: slow) |
| `4bit-qat` | 46.6 / 42.0 tok/s | 610 tok/s | 17.7 GB | 32 GB Macs and up (24 GB: slow) |
| `3bit` | 41.4 / 37.4 tok/s | 956 tok/s | 14.6 GB | 32 GB Macs and up (18 GB: slow) |

No drafter on this MoE: it is slower with one (the 4-bit: 34.0 tok/s with it, 38.8 without, on a short answer). On a 24 GB Mac the 4-bit and QAT builds leave under 2.5 GB of the GPU's working set, so the server prefills in 128-token steps (about 600 tok/s; the 3-bit, with room, about 950). The 8 and 6-bit weren't run here: their peaks are estimates, their speeds scaled from the 4-bit's by size.

## How these were made

From [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) at revision `4d7ae4984b` (and, for `4bit-qat`, Google's QAT checkpoint named above), by
Moonbox's converter (`lab/zoo/gemma-4/convert.py` in Moonbox's repository): mlx-vlm's own sanitize and layout,
tensor by tensor, MLX affine quantization; the vision and audio encoders and the multimodal projections stay
bfloat16 in every build. Each build was checked against the original before upload (teacher-forced, layer by layer
through mlx-vlm's own layers), and the written files were checked to reproduce what was measured (identical
logits); the `sha256` column above is what the files must hash to.

## License

Apache-2.0, Google's licence for Gemma 4 (ai.google.dev/gemma/docs/gemma_4_license). These files are a derivative
of the original model and carry its licence. The original model card follows unchanged.

---

# Original model card

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>


<p align="center">
    <a href="https://huggingface.co/collections/google/gemma-4" target="_blank">Hugging Face</a> |
    <a href="https://github.com/google-gemma" target="_blank">GitHub</a> |
    <a href="https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/" target="_blank">Launch Blog</a> |
    <a href="https://ai.google.dev/gemma/docs/core" target="_blank">Documentation</a> |
    <a href="https://arxiv.org/abs/2607.02770" target="_blank">Technical Report</a>
    <br>
    <b>License</b>: <a href="https://ai.google.dev/gemma/docs/gemma_4_license" target="_blank">Apache 2.0</a> | <b>Authors</b>: <a href="https://deepmind.google/models/gemma/" target="_blank">Google DeepMind</a>
</p>

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on E2B, E4B, and 12B) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages. 

Featuring both Dense and Mixture-of-Experts (MoE) architectures, Gemma 4 is well-suited for tasks like text generation, coding, and reasoning. The models are available in five distinct sizes: **E2B**, **E4B**, **12B**, **26B A4B**, and **31B**. Their diverse sizes make them deployable in environments ranging from high-end phones to laptops and servers, democratizing access to state-of-the-art AI.

Gemma 4 introduces key **capability and architectural advancements**:

* **Reasoning** – All models in the family are designed as highly capable reasoners, with configurable thinking modes.

* **Extended Multimodalities** – Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B, E4B, and 12B models).

* **Diverse & Efficient Architectures** – Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.

* **Optimized for On-Device** – Smaller models are specifically designed for efficient local execution on laptops and mobile devices.

* **Increased Context Window** – The small models feature a 128K context window, while the medium models support 256K.

* **Enhanced Coding & Agentic Capabilities** – Achieves notable improvements in coding benchmarks alongside native function-calling support, powering highly capable autonomous agents.

* **Native System Prompt Support** – Gemma 4 introduces native support for the `system` role, enabling more structured and controllable conversations.

## **Models Overview**

Gemma 4 models are designed to deliver frontier-level performance at each size, targeting deployment scenarios from mobile and edge devices (E2B, E4B) to consumer GPUs and workstations (12B, 26B A4B, 31B). They are well-suited for reasoning, agentic workflows, coding, and multimodal understanding.

The models employ a hybrid attention mechanism that interleaves local sliding window attention with full global attention, ensuring the final layer is always global. This hybrid design delivers the processing speed and low memory footprint of a lightweight model without sacrificing the deep awareness required for complex, long-context tasks. To optimize memory for long contexts, global layers feature unified Keys and Values, and apply Proportional RoPE (p-RoPE). 

### Dense Models

| Property | E2B | E4B | 12B Unified | 31B Dense |
| :---- | :---- | :---- | :---- | :---- |
| **Total Parameters** | 2.3B effective <br> (5.1B with embeddings) | 4.5B effective <br> (8B with embeddings) | 11.95B | 30.7B |
| **Layers** | 35 | 42 | 48 | 60 |
| **Sliding Window** | 512 tokens | 512 tokens | 1024 tokens | 1024 tokens |
| **Context Length** | 128K tokens | 128K tokens | 256K tokens  | 256K tokens  |
| **Vocabulary Size** | 262K | 262K | 262K | 262K |
| **Supported Modalities** | Text, Image, Audio | Text, Image, Audio | Text, Image, Audio | Text, Image |
| **Vision Encoder Parameters** | *~150M* | *~150M* | - | *~550M* |
| **Audio Encoder Parameters** | *~300M* | *~300M* | - | No Audio |

The "E" in E2B and E4B stands for "effective" parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers or parameters to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but are only used for quick lookups, which is why the effective parameter count is much smaller than the total.

The "Unified" in Gemma 4 12B Unified refers to its encoder-free architecture. Other Gemma 4 models use dedicated encoders to process multimodal data before passing it to the LLM. Gemma 4 12B eliminates these encoders entirely, projecting raw image patches and audio waveforms directly into the LLM's embedding space through lightweight linear layers. This unified approach means all modalities flow straight into a single decoder-only transformer, reducing multimodal latency and allowing the entire model to be fine-tuned in one pass.

### Mixture-of-Experts (MoE) Model

| Property | 26B A4B MoE |
| :---- | :---- |
| **Total Parameters** | 25.2B |
| **Active Parameters** | 3.8B |
| **Layers** | 30 |
| **Sliding Window** | 1024 tokens |
| **Context Length** | 256K tokens |
| **Vocabulary Size** | 262K |
| **Expert Count** | 8 active / 128 total and 1 shared |
| **Supported Modalities** | Text, Image |
| **Vision Encoder Parameters** | *~550M* |

The "A" in 26B A4B stands for "active parameters" in contrast to the total number of parameters the model contains. By only activating a 4B subset of parameters during inference, the Mixture-of-Experts model runs much faster than its 26B total might suggest. This makes it an excellent choice for fast inference compared to the dense 31B model since it runs almost as fast as a 4B-parameter model.

## **Benchmark Results** 

These models were evaluated against a large collection of different datasets and metrics to cover different aspects of text generation. Evaluation results marked in the table are for instruction-tuned models.

|  | Gemma 4  31B | Gemma 4  26B A4B | Gemma 4 12B Unified | Gemma 4  E4B | Gemma 4  E2B | Gemma 3  27B (no think) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| MMLU Pro | 85.2% | 82.6% | 77.2% | 69.4% | 60.0% | 67.6% |
| AIME 2026 no tools | 89.2% | 88.3% | 77.5% | 42.5% | 37.5% | 20.8% |
| LiveCodeBench v6 | 80.0% | 77.1% | 72.0% | 52.0% | 44.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 1659 | 940 | 633 | 110 |
| GPQA Diamond | 84.3% | 82.3% | 78.8% | 58.6% | 43.4% | 42.4% |
| Tau2 (average over 3) | 76.9% | 68.2% | 69.0% | 42.2% | 24.5% | 16.2% |
| HLE no tools | 19.5% | 8.7% | 5.2% | - | - | - |
| HLE with search | 26.5% | 17.2% | - | - | - | - |
| BigBench Extra Hard | 74.4% | 64.8% | 53.0% | 33.1% | 21.9% | 19.3% |
| MMMLU | 88.4% | 86.3% | 83.4% | 76.6% | 67.4% | 70.7% |
| **Vision** |  |  |  |  |  |  |
| MMMU Pro | 76.9% | 73.8% | 69.1% | 52.6% | 44.2% | 49.7% |
| OmniDocBench 1.5 (average edit distance, lower is better) | 0.131 | 0.149 | 0.164 | 0.181 | 0.290 | 0.365 |
| MATH-Vision | 85.6% | 82.4% | 79.7% | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM | 61.3% | 58.1% | 48.7% | 28.7% | 23.5% | - |
| **Audio** |  |  |  |  |  |  |
| CoVoST | - | - | 38.5<sup>*</sup> | 35.54 | 33.47 | - |
| FLEURS (lower is better) | - | - | 0.069<sup>*</sup> | 0.08 | 0.09 | - |
| **Long Context** |  |  |  |  |  |  |
| MRCR v2 8 needle 128k (average) | 66.4% | 44.1%  | 43.4% | 25.4% | 19.1% | 13.5% |

<sup>*</sup>Excluding Chinese language.

## **Core Capabilities**

Gemma 4 models handle a broad range of tasks across text, vision, and audio. Key capabilities include:

* **Thinking** – Built-in reasoning mode that lets the model think step-by-step before answering.
* **Long Context** – Context windows of up to 128K tokens (E2B/E4B) and 256K tokens (12B, 26B A4B/31B).
* **Image Understanding** – Object detection, Document/PDF parsing, screen and UI understanding, chart comprehension, OCR (including multilingual), handwriting recognition, and pointing. Images can be processed at variable aspect ratios and resolutions.
* **Video Understanding** – Analyze video by processing sequences of frames.
* **Interleaved Multimodal Input** – Freely mix text and images in any order within a single prompt.
* **Function Calling** – Native support for structured tool use, enabling agentic workflows.
* **Coding** – Code generation, completion, and correction.
* **Multilingual** – Out-of-the-box support for 35+ languages, pre-trained on 140+ languages.
* **Audio** (E2B, E4B, and 12B only) – Automatic speech recognition (ASR) and speech-to-translated-text translation across multiple languages.

## Getting Started

You can use all Gemma 4 models with the latest version of Transformers. To get started, install the necessary dependencies in your environment:

`pip install -U transformers torch accelerate`

Once you have everything installed, you can proceed to load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
```

Once the model is loaded, you can start generating output:

```python
# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response, prefix=inputs["input_ids"])
```

To enable reasoning, set `enable_thinking=True` and the `parse_response` function will take care of parsing the thinking output.

Below, you will also find snippets for processing audio (E2B, E4B, 12B only), images, and video alongside text:

<details>
<summary>Code for processing Audio</summary>

Make sure to install the following packages:

`pip install -U transformers torch torchvision librosa accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E2B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the audio URL in the prompt:


```python
# Prompt - add audio after text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav"},
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response, prefix=inputs["input_ids"])
```

</details>

<details>
<summary>Code for processing Images</summary>

Make sure to install the following packages:


`pip install -U transformers torch torchvision accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the image URL in the prompt:


```python
# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response, prefix=inputs["input_ids"])
```

</details>


<details>
<summary>Code for processing Videos</summary>

Make sure to install the following packages:

`pip install -U transformers torch torchvision librosa accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the video URL in the prompt:


```python
# Prompt - add video before text
messages = [
    {
        'role': 'user',
        'content': [
            {"type": "video", "video": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4"},
            {'type': 'text', 'text': 'Describe this video.'}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response, prefix=inputs["input_ids"])
```

</details>



## **Best Practices**

For the best performance, use these configurations and best practices:

### 1. Sampling Parameters

Use the following standardized sampling configuration across all use cases:

* `temperature=1.0`  
* `top_p=0.95`  
* `top_k=64`

### 2. Thinking Mode Configuration

Compared to Gemma 3, the models use standard `system`, `assistant`, and `user` roles. To properly manage the thinking process, use the following control tokens:

* **Trigger Thinking:** Thinking is enabled by including the `<|think|>` token at the start of the system prompt. To disable thinking, remove the token.   
* **Standard Generation:** When thinking is enabled, the model will output its internal reasoning followed by the final answer using this structure:  
  `<|channel>thought\n`**[Internal reasoning]**`<channel|>`  
* **Disabled Thinking Behavior:** For all models except for the E2B and E4B variants, if thinking is disabled, the model will still generate the tags but with an empty thought block:  
  `<|channel>thought\n<channel|>`**[Final answer]**

> [!Note]
> Note that many libraries like Transformers and llama.cpp handle the complexities of the chat template for you.

### 3. Multi-Turn Conversations

* **No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final response. Thoughts from previous model turns must *not be added* before the next user turn begins, with the exception of tool call turns where thinking content should be preserved.

### 4. Modality order

For optimal performance with multimodal inputs, place:

* Image content **before** the text in your prompt.
* Audio content **after** the text in your prompt.

### 5. Variable Image Resolution

Aside from variable aspect ratios, Gemma 4 supports variable image resolution through a configurable visual token budget, which controls how many tokens are used to represent an image. A higher token budget preserves more visual detail at the cost of additional compute, while a lower budget enables faster inference for tasks that don't require fine-grained understanding.

* The supported token budgets are: **70**, **140**, **280**, **560**, and **1120**.  
  * Use *lower budgets* for classification, captioning, or video understanding, where faster inference and processing many frames outweigh fine-grained detail.   
  * Use *higher budgets* for tasks like OCR, document parsing, or reading small text.

### 6. Audio

Use the following prompt structures for audio processing:

* **Audio Speech Recognition (ASR)**

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

* **Automatic Speech Translation (AST)**

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

### 7. Audio and Video Length

All models support image inputs and can process videos as frames whereas the E2B, E4B, and 12B models also support audio inputs. Audio supports a maximum length of 30 seconds. Video supports a maximum of 60 seconds assuming the images are processed at one frame per second.

## **Model Data**

Data used for model training and how the data was processed.

### **Training Dataset**

Our pre-training dataset is a large-scale, diverse collection of data encompassing a wide range of domains and modalities, which includes web documents, code, images, audio, with a cutoff date of January 2025. Here are the key components:

* **Web Documents**: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.  
* **Code**: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.  
* **Mathematics**: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.  
* **Images**: A wide range of images enables the model to perform image analysis and visual data extraction tasks.

The combination of these diverse data sources is crucial for training a powerful multimodal model that can handle a wide variety of different tasks and data formats.

### **Data Preprocessing**

Here are the key data cleaning and filtering methods applied to the training data:

* **CSAM Filtering**: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.  
* **Sensitive Data Filtering**: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.  
* **Additional methods**: Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## **Ethics and Safety**

As open models become central to enterprise infrastructure, provenance and security are paramount. Developed by Google DeepMind, Gemma 4 undergoes the same rigorous safety evaluations as our proprietary Gemini models. 

### **Evaluation Approach**

Gemma 4 models were developed in partnership with internal safety and responsible AI teams. A range of automated as well as human evaluations were conducted to help improve model safety. These evaluations align with [Google’s AI principles](https://ai.google/principles/), as well as safety policies, which aim to prevent our generative AI models from generating harmful content, including:

* Content related to child sexual abuse material and exploitation   
* Dangerous content (e.g., promoting suicide, or instructing in activities that could cause real-world harm)   
* Sexually explicit content  
* Hate speech (e.g., dehumanizing members of protected groups)   
* Harassment (e.g., encouraging violence against people)

### **Evaluation Results**

For all areas of safety testing, we saw major improvements in all categories of content safety relative to previous Gemma models. Overall, Gemma 4 models significantly outperform Gemma 3 and 3n models in improving safety, while keeping unjustified refusals low. All testing was conducted without safety filters to evaluate the model capabilities and behaviors. For both text-to-text and image-to-text, and across all model sizes, the model produced minimal policy violations, and showed significant improvements over previous Gemma models' performance. 

## **Usage and Limitations**

These models have certain limitations that users should be aware of.

### **Intended Usage**

Multimodal models (capable of processing vision, language, and/or audio) have a wide range of applications across various industries and domains. The following list of potential uses is not comprehensive. The purpose of this list is to provide contextual information about the possible use-cases that the model creators considered as part of model training and development.

* **Content Creation and Communication**  
  * **Text Generation**: These models can be used to generate creative text formats such as poems, scripts, code, marketing copy, and email drafts.  
  * **Chatbots and Conversational AI**: Power conversational interfaces for customer service, virtual assistants, or interactive applications.  
  * **Text Summarization**: Generate concise summaries of a text corpus, research papers, or reports.  
  * **Image Data Extraction**: These models can be used to extract, interpret, and summarize visual data for text communications.  
  * **Audio Processing and Interaction**: The E2B, E4B, and 12B models can analyze and interpret audio inputs, enabling voice-driven interactions and transcriptions.  
* **Research and Education**  
  * **Natural Language Processing (NLP) and VLM Research**: These models can serve as a foundation for researchers to experiment with VLM and NLP techniques, develop algorithms, and contribute to the advancement of the field.  
  * **Language Learning Tools**: Support interactive language learning experiences, aiding in grammar correction or providing writing practice.  
  * **Knowledge Exploration**: Assist researchers in exploring large bodies of text by generating summaries or answering questions about specific topics.

### **Limitations**

* **Training Data**  
  * The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.  
  * The scope of the training dataset determines the subject areas the model can handle effectively.  
* **Context and Task Complexity**  
  * Models perform well on tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.  
  * A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).  
* **Language Ambiguity and Nuance**  
  * Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.  
* **Factual Accuracy**  
  * Models generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.  
* **Common Sense**  
  * Models rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### **Ethical Considerations and Risks**

The development of vision-language models (VLMs) raises several ethical concerns. In creating an open model, we have carefully considered the following:

* **Bias and Fairness**  
  * VLMs trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material. Gemma 4 models underwent careful scrutiny, input data pre-processing, and post-training evaluations as reported in this card to help mitigate the risk of these biases.  
* **Misinformation and Misuse**  
  * VLMs can be misused to generate text that is false, misleading, or harmful.  
  * Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).  
* **Transparency and Accountability**  
  * This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.  
  * A responsibly developed open model offers the opportunity to share innovation by making VLM technology accessible to developers and researchers across the AI ecosystem.

**Risks identified and mitigations**:

* **Generation of harmful content**: Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.  
* **Misuse for malicious purposes**: Technical limitations and developer and end-user education can help mitigate against malicious applications of VLMs. Educational resources and reporting mechanisms for users to flag misuse are provided.   
* **Privacy violations**: Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.  
* **Perpetuation of biases**: It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.

### **Benefits**

At the time of release, this family of models provides high-performance open vision-language model implementations designed from the ground up for responsible AI development compared to similarly sized models.

## **Citation**

If you find our work helpful, please consider citing it:

```bibtex
@misc{gemmateam2026gemma4,
      title={Gemma 4 Technical Report}, 
      author={Gemma Team},
      year={2026},
      eprint={2607.02770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02770}, 
}
```
