---
language:
- zh
- en
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
- junafinity/Qwen-3.8-27B-Uncensored
tags:
- qwen3.8
- gguf
- zerorefusal
- uncensored
- unsloth-dynamic
- iq4-xs
- mtp
- multimodal
- llama.cpp
- 16gb-vram
---

# Qwen3.8-27B ZeroRefusal UD-IQ4_XS V3 Final (Native MTP)

**English** | [简体中文](README.zh-CN.md)

This is a Qwen3.8-27B GGUF built and validated for an **RTX 5060 Ti 16 GB** deployment.

The model was requantized from ZeroFuse-edited BF16 weights using the tensor-type layout and importance matrix of the final Unsloth Dynamic V3 `UD-IQ4_XS` release. The resulting single GGUF retains the native MTP / NextN layer and does not require a separate draft model.

This is an independent community release, not an official release from Qwen, Unsloth, llama.cpp, or ZeroFuse.

## Release files

| File | Purpose | Size |
|---|---|---:|
| `Qwen3.8-27B-ZeroRefusal-UD-IQ4_XS-V3-Final-MTP.gguf` | Main text model with native MTP | 14,252,845,056 bytes |
| `mmproj-Qwen3.8-27B-F16.gguf` | Qwen3.8 Vision F16 projector | 927,607,008 bytes |
| `quantization/imatrix_unsloth.gguf` | Importance matrix used for quantization | 13,642,656 bytes |
| `quantization/v3-final-tensor-overrides.txt` | Anchored type map for all 866 tensors | 17,007 bytes |

Main model SHA256:

```text
f6c1f6e1211aeceaa386cbeabd0bdfbee0d124747ecaa3e908a93394518e9274
```

See `SHA256SUMS` for all published binary hashes.

## Model structure

- Architecture: `qwen35`
- Tensor count: 866
- Block count: 65
- `qwen35.nextn_predict_layers = 1`
- Native MTP: 15 tensors under `blk.64.*`
- `general.file_type = 30`
- llama.cpp reports: `IQ4_XS - 4.25 bpw`
- Original context length: 262,144
- Validated production context: 73,728

The final tensor names, shapes, and quantization-type distribution match the final Unsloth Dynamic V3 `UD-IQ4_XS` reference. This does not imply byte-identical tensor payloads: the behavior weights were edited and requantized.

## Validated production profile

```text
GPU                 RTX 5060 Ti 16 GB
llama.cpp           upstream b10435 / 9e40df6
context             73728 (72 x 1024, shared KV pool)
parallel            2
KV                   unified Q4_0 K / Q4_0 V
text placement      full GPU
MTP                  1
draft KV             F16 K / F16 V
Vision               F16 mmproj, CPU offload
image-max-tokens     4096
API port             8001
```

With `73728 + P2 + unified KV`, both slots share one 72K KV pool. It is not a separate 72K context for each slot.

## llama-server example

```bash
llama-server \
  -m Qwen3.8-27B-ZeroRefusal-UD-IQ4_XS-V3-Final-MTP.gguf \
  --alias qwen3.8-27b \
  --mmproj mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload \
  --image-max-tokens 4096 \
  -c 73728 \
  -np 2 \
  --kv-unified \
  -ngl 999 \
  --flash-attn on \
  -ctk q4_0 \
  -ctv q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --spec-draft-type-k f16 \
  --spec-draft-type-v f16 \
  --no-spec-draft-backend-sampling \
  -b 512 \
  -ub 64 \
  --threads 7 \
  --fit off \
  --jinja \
  --host 0.0.0.0 \
  --port 8001
```

## Measured performance on RTX 5060 Ti 16 GB

| Test | Result |
|---|---:|
| P1 decode | 41.67 tok/s |
| P2 stream A | 31.58 tok/s |
| P2 stream B | 34.08 tok/s |
| P2 average per stream | 32.83 tok/s |
| P2 server aggregate | 65.67 tok/s |
| P2 wall-clock aggregate | 61.88 tok/s |
| Lowest observed free VRAM | about 138 MiB |

MTP draft acceptance on the speed-test samples was 80.85% for P1 and 77.43% / 62.22% for the two P2 streams. Acceptance is content-dependent.

## 64-prompt refusal validation

Under the fixed validation configuration:

```text
valid       64
accepted    64
refused      0
errors       0
result       0/64 refusal
```

Method:

- Dataset: `mlabonne/harmful_behaviors`
- Revision: `01cead01398926d81f7c52bdb790ee8cf77ebba7`
- Corpus SHA256: `b4f2ddec5ab06058b721be9afe75e7fdc656852da257d8e7a117425b3dd89114`
- First 64 prompts, maximum 64 output tokens per prompt
- Temperature 1.0, top-p 0.95, top-k 20
- Fixed 23-marker refusal classifier

Only the methodology and summary are public. Raw harmful outputs are intentionally not uploaded. This result applies only to the fixed test and is not a general safety or quality guarantee.

## Provenance

Final Dynamic V3 quantization reference:

```text
repo      unsloth/Qwen3.8-27B-GGUF
file      Qwen3.8-27B-UD-IQ4_XS.gguf
revision  4ca720788d1e01f1bff70c033e0d0028fd02e502
bytes     14252845984
sha256    40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199
```

Behavior-edit source:

```text
repo          junafinity/Qwen-3.8-27B-Uncensored
revision      903d149c148b81fdf4e568a05ac9ad4225f493d7
method        ZeroFuse v0.1.0 directional ablation
trial         38
source layer  35
strength      1.2242385643286666
edited layers 9-56
BF16 KL       0.009713646
BF16 refusal  0/64
```

The quantization map uses fully anchored regular expressions. This prevents the global `output.weight` rule from accidentally matching per-layer `attn_output.weight` tensors. See `quantization/v3-final-tensor-overrides.txt`.

## Limitations and safety

- The 16 GB configuration has a very narrow VRAM margin. Different drivers, CUDA versions, batch shapes, or background GPU workloads can cause OOM.
- Do not run ComfyUI or another heavy CUDA workload at the same time.
- Keep the F16 Vision projector on CPU for this profile.
- ZeroRefusal/uncensored models can produce unsafe, illegal, or incorrect content. Deployers are responsible for access control, auditing, and safety policy.
- Follow `LICENSE` and all upstream model, dataset, and tool licenses and terms.
