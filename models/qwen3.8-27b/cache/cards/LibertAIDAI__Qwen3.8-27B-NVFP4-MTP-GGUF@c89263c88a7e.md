---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
quantized_by: LibertAIDAI
tags:
  - gguf
  - llama.cpp
  - nvfp4
  - mtp
  - speculative-decoding
  - blackwell
  - qwen3.8
  - multimodal
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B NVFP4 GGUF, MTP variant

NVFP4 GGUF quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with Multi-Token Prediction (MTP) for speculative decoding in [llama.cpp](https://github.com/ggml-org/llama.cpp).

Same NVFP4 trunk as our [Qwen3.8-27B-NVFP4-GGUF](https://huggingface.co/LibertAIDAI/Qwen3.8-27B-NVFP4-GGUF) repo, plus the MTP draft head extracted from the source for use with `--spec-type draft-mtp`. The MTP head drafts the next few tokens cheaply and verifies them in a single trunk forward pass, worth about 23% more token-generation throughput single-stream on an RTX 5090.

The NVFP4 weights come from our own ModelOpt post-training quantization of the BF16 release: NVFP4 at group size 16 on the language-model MLP linears only, 192 quantized linears in total, calibrated on 512 samples of cnn_dailymail. Attention, the vision tower, the MTP block, embeddings and `lm_head` stay BF16.

## About LibertAI

[LibertAI](https://libertai.io) is a decentralized AI platform offering private inference, an OpenAI-compatible API, and a chat UI, all running on community GPUs over [Aleph Cloud](https://aleph.cloud) rather than a single company's servers.

If you want to put this model to work as an autonomous agent without running your own infrastructure, see [LiberClaw](https://liberclaw.ai), Hermes-style agents hosted on Aleph Cloud with LibertAI inference. Free tier: 2 agents, no credit card, 5 minutes to deploy. Open source.

## Files

| File | Size | FFN | Other tensors | When to pick |
|------|------|-----|---------------|--------------|
| `Qwen3.8-27B-NVFP4-Q4_K_M-mtp.gguf` | 15.8 GB | NVFP4 | Q4_K_M | **Recommended trunk.** Pair with the MTP draft |
| `Qwen3.8-27B-NVFP4-Q8_0-mtp.gguf` | 20.0 GB | NVFP4 | Q8_0 | Higher-precision attention and embeddings |
| `Qwen3.8-27B-NVFP4-BF16-mtp.gguf` | 29.2 GB | NVFP4 | BF16 | Source-precision non-FFN tensors, slower in practice |
| `mtp-Qwen3.8-27B-NVFP4.gguf` | 5.95 GB | BF16 | BF16/F32 | **MTP draft head**, required for `--spec-type draft-mtp` |
| `mmproj-Qwen3.8-27B-F16.gguf` | 0.93 GB | n/a | F16 vision tower | Required for image and video input |

The trunk files are built with `convert_hf_to_gguf.py --no-mtp`, so the MTP weights are split into the separate `mtp-*.gguf`. Split form lets the draft head stay BF16 for the best accept behavior while the trunk goes to Q4_K_M, and lets llama.cpp place the draft independently.

## Performance

Measured on an NVIDIA RTX 5090 (32 GB, Blackwell, sm_120, CUDA 13.0), llama.cpp build `849798132`.

![NVFP4 vs stock Q4_K_M on RTX 5090](chart-serving.png)

Single-stream, 512-token completion, production settings (`-c 262144 -fa on -ctk q4_0 -ctv q4_0 -ub 256 -b 1024 --parallel 1`):

| Config | TG tok/s | Accept rate | Difference |
|---|---|---|---|
| NVFP4-Q4_K_M, no MTP | 79.4 | n/a | baseline |
| NVFP4-Q4_K_M + MTP, `n-max=3 p-min=0.1` | **97.4** | 38.7% | **+22.7%** |

### Draft parameter tuning

We swept the draft parameters rather than assuming defaults. The results are worth reading before tuning:

![MTP draft parameter sweep](chart-mtp.png)


| n-max | p-min | TG tok/s | Difference | Drafted | Accepted |
|---|---|---|---|---|---|
| 3 | 0.0 | 97.2 | +22.4% | 708 | 274 |
| 3 | 0.1 | 97.4 | +22.7% | 708 | 274 |
| 3 | 0.5 | 86.3 | +8.7% | 383 | 236 |
| 4 | 0.0 | 94.6 | +19.1% | 851 | 298 |
| 4 | 0.5 | 89.0 | +12.1% | 405 | 252 |
| 6 | 0.0 | 78.0 | -1.8% | 1227 | 306 |
| 6 | 0.5 | 81.1 | +2.1% | 523 | 253 |

Two things fall out of this. Accepted tokens saturate around 300 as `n-max` rises, while drafting cost keeps growing, so at `n-max=6` MTP becomes a net loss. And a high `p-min` raises the accept rate while lowering throughput, because it discards draft branches that would have been verified cheaply anyway. Accept rate is not the quantity to maximize.

The llama.cpp default of `n-max=3` is already optimal here. Leave it alone and keep `p-min` at or below 0.1.

## Usage

```bash
llama-server \
  -m Qwen3.8-27B-NVFP4-Q4_K_M-mtp.gguf \
  --model-draft mtp-Qwen3.8-27B-NVFP4.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.1 \
  -ngl 999 -ngld 999 \
  -fa on -c 32768 \
  --host 0.0.0.0 --port 8080
```

`-ngl` and `-ngld` offload the trunk and the draft head respectively. Add `--mmproj mmproj-Qwen3.8-27B-F16.gguf` for image and video input.

## Requirements

Native NVFP4 tensor-core math needs a Blackwell GPU (sm_120) and llama.cpp built with CUDA 12.8 or newer, `-DCMAKE_CUDA_ARCHITECTURES=120`.

## Caveats

- The draft head cannot be loaded standalone. It is only valid as `--model-draft`.
- Speculative gains depend on how predictable the output is. The benchmark prompt above is free-form technical prose, which is a harder case than chat or code completion.
- Calibration was English news text (cnn_dailymail). Multilingual and code-heavy workloads were not measured against a held-out set.
- The vision tower is left in BF16 and was not separately evaluated.
- Benchmarks are from a single RTX 5090.

## License and attribution

Inherits Apache 2.0 from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). All rights, responsibilities and acceptable-use policies of the upstream license apply.

Quantization performed by [LibertAI](https://libertai.io).
