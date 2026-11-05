---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- qwen4_exp
- qwen3.8-flash-next
- moe
- mlx
- mlx-vlm
- omlx
- apple-silicon
- custom_code
- research-preview
- seedlm
- nanoseedlm
- nslm
- lfsr
- seeds
---

# Qwen3.8-Flash-Next-NanoSeedLM-P4

> **Experimental.** The routed experts of this model are SeedLM seeds. We measured its quality against the 4-bit MLX
> model only, not against BF16 (see Quality). For normal work, use an established quantization of
> [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

Qwen3.8-Flash-Next-NanoSeedLM-P4 is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with its
routed experts as SeedLM seeds. Each block of 8 expert weights is a 16-bit LFSR seed, 4 int4 coefficients and a 4-bit
exponent code. The GPU makes the weights from the seeds inside the matrix multiply. All other weights come unchanged from
[mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp) (4-bit
affine, group size 32), with its MTP head.

## How it is different from the 4-bit MLX model

| Part | This model | mlx-community 4-bit |
|---|---|---|
| Routed experts (120.8 B weights, 48 layers x 512) | SeedLM P = 4, 4.5 bits per weight | 4-bit affine, group 32 (5.0 bits per weight) |
| Attention, DeltaNet, shared experts, routers, PLE table, embedding, LM head, vision, MTP head | The same tensors | 4-bit affine (group 32), 8-bit routers (group 64), BF16 |
| Size | 105.7 GB | 113.3 GB |

The seed search tried all 65,535 seeds for each block. It weighted the error of each weight by the mean square of its
input activation, from the Unsloth imatrix (`imatrix_unsloth.gguf_file`, 45 chunks of 18,432 tokens). Read
[Make a model folder](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/make-a-model.md).

## Quality

Against [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp), on
1,024 tokens of wikitext-2 (both models on the NanoSeedLM CUDA engine):

| Model | Mean KL divergence | Median KL divergence | Top-1 agreement | Perplexity |
|---|---|---|---|---|
| 4-bit MLX model | 0 | 0 | 100% | 2.396 |
| This model (P = 4) | 0.177 | 0.013 | 91.9% | 2.500 |

The reference is a quantized model, not BF16, and one text of 1,024 tokens is a small sample. The relative error of the
routed expert weights, the mean over the 48 layers:

| Projection | Relative error | Activation-weighted relative error |
|---|---|---|
| `gate_proj` | 0.0862 | 0.0770 |
| `up_proj` | 0.0862 | 0.0772 |
| `down_proj` | 0.0869 | 0.0759 |

Weight error does not predict KL divergence well. We will add KL divergence against the BF16 model to this card.
Read [the measurements](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-04-qwen38-kld-vs-4bit.md).

## Platforms

- **Apple Silicon Macs** with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.12 or later (`--trust-remote-code`).
  The model files `nanoseedlm_qwen4.py` and `nanoseedlm_seeds.py` are in this repository.
- **oMLX**: not yet. oMLX removes `model_file` from every `qwen4_exp` configuration, so it does not load the model files
  of this repository. A one-line change to oMLX keeps `model_file` when it is not the stock `qwen4_exp.py`.
- **NanoSeedLM CUDA engine** (`nslm-qwen4-gen`): NVIDIA GPUs (Ampere or newer). The routed experts stay in host memory,
  with a VRAM cache (`--expert-cache-gb`), or stream from the files on smaller machines (`--stream-experts`). Read
  [Run and serve](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md). The Metal engine does
  not run this model yet.

## How to run

### mlx-vlm

The command prints the prompt and generation speed in tokens per second.

```bash
python -m mlx_vlm.generate --model txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4 --trust-remote-code \
    --prompt "Why do tide pools matter?" --max-tokens 256
```

To read the PLE table from the SSD (128 GB Macs), make a folder view with mlx-vlm and run that folder:

```python
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
prepare_external_ple_model("Qwen3.8-Flash-Next-NanoSeedLM-P4", "Qwen3.8-Flash-Next-NanoSeedLM-P4-ssd-ple")
```

### NanoSeedLM CUDA engine

Build [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM) with CMake. On an RTX 4080 (16 GB), keep about 8 GB of the
experts in VRAM:

```bash
build/bin/nslm-qwen4-gen --model Qwen3.8-Flash-Next-NanoSeedLM-P4 --expert-cache-gb 8 --prompt "Why do tide pools matter?"
```

## Limits

- This is an experiment. Its output is not identical to the output of the original model.
- You must enable Trust Remote Code.
- On Apple Silicon, seed decode is slower than affine decode of the same size. On K2-Horizon-MoVA, the difference was
  14% to 21%.
- We did not test image or video input.
- The model is about 105.7 GB. By default, macOS gives the GPU about three quarters of memory (96 GB of 128 GB). On a
  128 GB Mac, keep the PLE table on the SSD (see How to run). We did not test this with this model yet.
- oMLX's MoE expert offload reads quantized experts. It may not apply to seed experts.

## Files

- `model-*.safetensors`: the weights. `NAME.seeds`, `NAME.coefs`, `NAME.exp_bias`, `NAME.codes` for the routed experts.
- `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`: the model files (MLX).
- `config.json`: the configuration of the 4-bit MLX model, with `model_file`.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
MLX conversion: [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
Activation statistics: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) imatrix. Method: SeedLM
([arXiv:2410.10714](https://arxiv.org/abs/2410.10714)). Code: [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM).
