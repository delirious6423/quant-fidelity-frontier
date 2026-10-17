---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - rocmfpx
  - qwen4exp
---

# Qwen3.8 Flash Next HC-Q8

Quantized GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This is not an official Qwen release. The weights remain under the [Qwen Community License 1.0](LICENSE).

The file is `Qwen3.8-Flash-Next-HC-Q8.gguf`.

- 179.551 billion parameters
- 92,347,778,272 bytes
- 4.11 bits per weight on average
- Architecture `qwen4exp`, context trained at 262,144
- The MTP head is inside this file

| Weights | Parameters | Storage | Bits per weight |
|---|---:|---|---:|
| Network, including experts | 126.33B | Q4_0_ROCMI4 | 4.25 |
| N-gram table | 51.20B | Q3_0_ROCMFPX | 3.50 |
| Hyperconnection up and down | 0.655B | Q8_0 | 8.50 |
| Injection and other BF16 tensors | 0.661B | BF16 | 16 |
| One remaining tensor | 0.636B | Q6_K | 6.56 |

HC-Q8 starts from a Q4_0_ROCMI4 quant. The 200 hyperconnection up and down matrices were restored from the original BF16 tensors in Qwen revision `de4b8e4d` and stored as ordinary Q8_0. The 98 injection tensors stayed BF16. Every other tensor kept the ROCmI4 bytes.

`Q4_0_ROCMI4` and `Q3_0_ROCMFPX` are ROCmFPX quant types. Stock llama.cpp does not load this file.

## Run it with ROCmFPX HIP

Use a HIP `llama-server` built from [ROCmFPX/ROCmFPX](https://github.com/ROCmFPX/ROCmFPX), with W4A4 disabled for this profile. Some ROCmFPX documentation names a serving preset or model generation "ROCmFP4 v2"; that is not a ROCmFPX runtime version requirement. Do not pass a separate MTP file: this GGUF already contains the MTP head. The separate-file FP4 package uses `-md`; that flag does not belong on this quant.

`HSA_OVERRIDE_GFX_VERSION=11.5.1` is for Strix Halo (gfx1151, RDNA 3.5). Leave it unset on RDNA 3 and on RDNA 4 cards such as the Radeon AI Pro R9700.

### Strix Halo HIP graph settings

On Strix Halo, use a ROCmFPX HIP build that includes the captured-copy support. To keep HIP graph capture enabled, set the opt-in copy-kernel switch and make sure graph disabling is unset:

```bash
export ROCMFPX_HIP_GRAPH_COPY_KERNEL=1
unset GGML_CUDA_DISABLE_GRAPHS
```

The copy-kernel switch replaces captured scalar device-to-device copies on gfx1151; it does not turn graph capture on by itself. This implementation is present in ROCmFPX `main` at commit `721db4193c9736c4d48f508993fcdfc6c75e180a` and later builds that retain it.

If you still see repetitive or corrupted output with graphs enabled, use this correctness-first fallback:

```bash
unset ROCMFPX_HIP_GRAPH_COPY_KERNEL
export GGML_CUDA_DISABLE_GRAPHS=1
```

These settings are alternatives. `GGML_CUDA_DISABLE_GRAPHS=1` disables the capture that activates the copy-kernel switch, so do not set it when testing the graph-enabled path. Disabling graphs may reduce decode speed. The reported failure was specific to a Strix Halo HIP runtime/model setup; verify output on your own build and configuration.

```bash
# gfx1151 / Strix Halo only
export HSA_OVERRIDE_GFX_VERSION=11.5.1

export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
export MALLOC_ARENA_MAX=2

llama-server \
  -m Qwen3.8-Flash-Next-HC-Q8.gguf \
  --alias qwen38-flash-next-hcq8 \
  --host 127.0.0.1 --port 8128 \
  --lazy-mode on-direct \
  -dev ROCm0 -ngl 99 -ngld 99 \
  -c 65536 -np 1 \
  -b 2048 -ub 512 -t 16 -tb 16 \
  -fa on -ctk f16 -ctv f16 \
  --fit off --jinja --reasoning-format deepseek \
  --no-kv-unified --no-context-shift --cont-batching \
  --cache-prompt --cache-idle-slots --cache-reuse 0 --cache-ram 1024 \
  --ctx-checkpoints 32 --checkpoint-min-step 8192 \
  --metrics \
  --spec-type ngram-mod,draft-mtp \
  --spec-draft-device ROCm0 --spec-draft-ngl 99 \
  --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-p-split 0.10 \
  --no-spec-draft-backend-sampling \
  --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 64
```

## Context length

The command above follows the ROCmFPX v2 profile, so it reserves a 65,536-token KV cache. This GGUF is the 262,144-token model either way. A prompt that fits in 65,536 runs the same forward pass when the server is started at 65,536 or at 262,144.

`llama-server` sizes that cache from `-c` at startup. A server started at 65,536 stops there. To serve a longer prompt, change that flag and start the server again, up to the trained length:

```bash
-c 262144
```

Keep `-ctk f16 -ctv f16`. The F16 cache for all 262,144 tokens is most of the memory on a 128 GB machine. This file, served at `-c 262144` on a 128 GB Strix Halo, used about 106 GiB and left about 10 GiB free. Use a shorter `-c` when the machine does not have that much memory.

The 32k, 64k, 128k, and 262k checks were different fills of one server already started at 262,144. Decode with 262,012 tokens in cache was 33 tokens/s.

On that server, one uninterrupted prefill of 262,016 tokens aborted once the key length passed 262,140 inside an 8192-token batch, with QSA decode enabled. Filling 253,824 tokens and then extending the cache in steps under 128 tokens completed, and decode ran. The v2 command on this card uses `-b 2048 -ub 512`.

Images need the original F16 vision projector from the Qwen repository beside this file, selected with `--mmproj`. That projector is not in this repo.
