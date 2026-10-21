---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- paroquant
- mxfp6
- w6a8
- rocm
- rdna4
- vllm
- quantized
---

# Qwen3.8-27B-PARO-MXFP6

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with ParoQuant rotations and MXFP6 (E2M3) weights, for AMD Radeon AI PRO R9700 (RDNA4) on a patched vLLM.

Size: 25.1 GB (bf16: 55.6 GB).

> This is a 27B model; the Hub's params badge counts packed tensor elements.

## Quality

| | this model | bf16 |
|---|--:|--:|
| WikiText-2 perplexity | 7.034 | 7.081 |
| KL divergence vs bf16 (top-256, nats) | 0.0091 | 0 |
| Top-1 token match with bf16 | 95.28% | 100% |
| MMLU 5-shot | 82.4% | — |
| GSM8K flexible / strict | 97.5% / 96.6% | — / — |
| HumanEval / HumanEval+ | 96.3% / 92.1% | — / — |

bf16 = the unquantized source model served on the same image; — = not measured.
Perplexity is cross-stack — this model at tensor parallel 1, bf16 at 2 — and wikitext-2's train split is in the stage-2 calibration mix, so a reading at or below bf16 is not evidence of a general gain.
GSM8K and HumanEval are chat-templated with reasoning on, greedy, on a Qwen chat template at `reasoning_effort=xhigh` rather than the default template the command below uses; both scores move with that choice.

## Performance

2× Radeon AI PRO R9700, TP=2, DFlash2-FP8 drafter (7 tokens), KV pinned with `KV_MEM=13500000000`, tokens/s.

| | |
|---|--:|
| Single-stream decode | 135.7 |
| Concurrent decode c1 / c2 / c4 / c8 | 122.2 / 213.9 / 328.4 / 401.6 |
| Prefill 2k / 8k / 16k / 32k / 60k | 3312 / 3599 / 3555 / 3453 / 3333 |
| KV cache tokens | 685,554 |
| Weights per GPU | 12.85 GiB |

Each row is a single measured run, on a newer radiance build than the `0.9.3` image the setup below installs, so expect these figures to move. KV capacity follows the pin, not the weight format.

## How to run

**Requirements**

- Two AMD RDNA4 (gfx1201) GPUs. The kernels are built for that architecture only; measured on 2 x Radeon AI PRO R9700 at tensor parallel 2.
- Linux with the `amdgpu` kernel driver loaded, so `/dev/kfd` and `/dev/dri` exist. ROCm userspace ships inside the image, so there is no host ROCm install.
- `docker` or `podman` (auto-detected), and a host `git` and `python3` (standard library only).
- Disk for the checkpoint, plus about 12 GiB for the DFlash2-FP8 drafter and the `stilldeadcode/vllm-radiance:0.9.3` image, both of which setup fetches.

**Two ways to get the code.** MXFP6 support is open as a pull request to [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) ([#46](https://codeberg.org/ggz14/radiance-vllm-mxfp4/pulls/46)). Clone the fork below to run it today, or wait for the pull request to merge and clone ggz14's repository instead — the commands are the same either way.

**Install and serve**

```bash
git clone https://codeberg.org/hugypufy/radiance-vllm-mxfp4 && cd radiance-vllm-mxfp4

SRC_REPO=hugypufy/Qwen3.8-27B-PARO-MXFP6 SNAP=~/models/Qwen3.8-27B-PARO-MXFP6 \
  ./setup-paroquant.sh --mxfp6 --yes

MODEL_DIR=Qwen3.8-27B-PARO-MXFP6 MODE=prod SPEC=7 ./paroquant/run_paroquant.sh
```

Setup is one-time and safe to re-run. The launcher reads the quantization format from `config.json` and sets the kernel knobs it needs, so that is the whole serve command; the server answers as `Qwen3.8-PARO` on `http://localhost:8080/v1`. Every start patches vLLM inside the container and compiles the ParoQuant kernels for gfx1201, about two minutes, before the model loads; the first start also compiles Triton and inductor graphs, which adds several minutes and is cached for later starts.

**Test**

```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-PARO","messages":[{"role":"user","content":"Hello!"}]}'
```

**Options** — set as environment variables on the serve command.

| Variable | Default | Notes |
|---|---|---|
| `KV_MEM` | unset; the pool is sized from `GPU_UTIL` (0.92) | Pin the KV cache in bytes per GPU; the figures above used `KV_MEM=13500000000` |
| `SPEC` | `5` | DFlash2 draft tokens per step. The numbers above used `7` |
| `CHAT_TEMPLATE` | the repository's `qwen-fixed-v22.3.jinja` | Any `.jinja` on the host, for example this checkpoint's own `chat_template.jinja` |
| `RADIANCE_VERIFY_HEAD` | `1` | Set to `0` to use `prompt_logprobs`, which hangs at the default |

## Format

- **Weights:** OCP MXFP6 E2M3, one e8m0 scale per 32 weights — 6.25 bits/weight.
- **Rotations:** from [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO), frozen.
- **Recipe:** round-to-nearest, then a stage-2 fine-tune (FT) of weights and block scales, rotations frozen.
- **fp16:** visual tower, `linear_attn.in_proj_a/b`, `linear_attn.conv1d` / `A_log` / `dt_bias`, `lm_head`, embeddings, norms, and the rotation `theta` / `channel_scales` (`pairs` are int16). MTP head not included.

## Licence

Base weights [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) are **Apache-2.0**; this derivative is distributed under the same licence (`LICENSE`).
Modified derivative, not made or endorsed by the Qwen team / Alibaba Cloud or by z-lab; `NOTICE` lists the changes.
Rotations from [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) (Apache-2.0).
