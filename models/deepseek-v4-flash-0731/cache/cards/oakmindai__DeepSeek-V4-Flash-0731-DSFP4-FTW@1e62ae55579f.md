---
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
license: mit
library_name: freetoken
tags:
- deepseek
- deepseek-v4
- dsfp4
- fp4
- quantized
- sparklab
- freetoken
- ftw
- dgx-spark
---

# DeepSeek V4 Flash 0731 DS-FP4 — SparkLab FTW

This is a ready-to-run **SparkLab** checkpoint for
[`DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
optimized for the **NVIDIA DGX Spark** and its Grace Blackwell GB10 Superchip.

**SparkLab** packages tested model recipes, hardware readiness checks, artifact
preparation, unified-memory planning, NVMe-backed MoE execution, and OpenAI-compatible
serving for one DGX Spark.

> **SparkLab source:** https://github.com/sixteen-miles-labs/sparklab

This deployment supports text input and text output. DeepSeek V4's custom message
encoding and reasoning-effort modes are preserved from the official checkpoint.

## What this repository contains

This repository does **not** introduce a new model or a new quantization. It repackages
DeepSeek AI's official
[`DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint into FreeToken Weight (FTW) format:

1. **[DeepSeek AI](https://huggingface.co/deepseek-ai)** developed and published the
   original model and its DS-FP4 expert weights.
2. **[FreeToken](https://github.com/FlashML-org/FreeToken)** provides the native inference
   backend, FTW format, conversion tooling, DS-FP4 kernels, and MoE expert-bank layout.
3. **SparkLab** provides the DGX Spark product experience: GB10 checks, model recipes,
   capacity planning, artifact lifecycle, deployment policy, and serving workflow.
4. **[OakMind AI](https://huggingface.co/oakmindai)** performed, validated, documented,
   and published this FTW conversion.

The exact source revision is
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Conversion is precision-preserving:
routed MoE experts remain in DeepSeek's packed E2M1 FP4 representation with one UE8M0
scale per 32 weights. Other tensors retain their published FP8 or BF16 precision. No
training or conversion-time requantization was performed.

## Why use FTW?

FTW performs model-layout work ahead of launch. It stores routed experts in independently
addressable banks and aligns the remaining tensors for FreeToken's native loader. This
allows SparkLab to keep active expert rows in unified memory and fetch cache misses from
local NVMe without materializing the complete checkpoint in application memory.

FTW does not alter the model's expected output quality. Fast local NVMe storage is
required for the disk-backed MoE path.

## Run with SparkLab on NVIDIA DGX Spark

Install SparkLab, then download this checkpoint to local NVMe:

```bash
git clone https://github.com/sixteen-miles-labs/sparklab.git
cd sparklab
./install.sh

hf download oakmindai/DeepSeek-V4-Flash-0731-DSFP4-FTW \
  --local-dir ~/models/DeepSeek-V4-Flash-0731-DSFP4-FTW
```

Verify the DGX Spark environment before loading the model:

```bash
sparklab doctor \
  --storage-path ~/models/DeepSeek-V4-Flash-0731-DSFP4-FTW
```

Start SparkLab's OpenAI-compatible API server:

```bash
FREETOKEN_DISK_READ_WORKERS=20 sparklab serve \
  --model ~/models/DeepSeek-V4-Flash-0731-DSFP4-FTW \
  --moe-backend offload \
  --moe-storage disk \
  --moe-host-cache-gb 4 \
  --memory-ratio 0.90 \
  --moe-cache-auto \
  --moe-prefill-sparse-max-tokens 512 \
  --attention-backend dsv4_sparse \
  --cuda-graph-max-bs 0 \
  --cache-type radix \
  --max-running-requests 1 \
  --host 127.0.0.1 \
  --port 8000
```

After the readiness message appears, send a chat-completions request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash-0731-DSFP4-FTW",
    "messages": [{"role": "user", "content": "Explain unified memory."}],
    "max_tokens": 256,
    "stream": true
  }'
```

## Credits and license

- Model architecture, training, and official DS-FP4 checkpoint:
  [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- DGX Spark product, orchestration, model workflow, and deployment guidance:
  **[SparkLab](https://github.com/sixteen-miles-labs/sparklab)**
- Native inference engine, FTW format, conversion, and kernels:
  **[FreeToken](https://github.com/FlashML-org/FreeToken)**
- FTW conversion and publishing: [OakMind AI](https://huggingface.co/oakmindai)

The upstream model is distributed under the MIT License. This repository preserves the
supplied `LICENSE` and source provenance. Review the
[official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) for its
usage guidance, evaluation details, and model-specific instructions.

## Citation

If you use DeepSeek V4 Flash, cite the official technical report linked from the
[upstream model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

If you use SparkLab, cite the software:

```bibtex
@software{sixteenmileslabs2026sparklab,
  title={SparkLab: Frontier Open-Weight Model Inference on NVIDIA DGX Spark},
  author={{Sixteen Miles Labs}},
  year={2026},
  url={https://github.com/sixteen-miles-labs/sparklab},
  license={Apache-2.0}
}
```

If you use the underlying inference research, cite FreeToken:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026},
  url={https://arxiv.org/abs/2608.16157}
}
```
