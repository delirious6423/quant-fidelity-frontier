---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - orcarouter/Qwen3.8-27B-Uncensored
base_model_relation: quantized
library_name: ninfer
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - ninfer
  - nvfp4
  - fp8
  - blackwell
  - rtx-5090
  - uncensored
  - abliterated
  - qwen3.8
  - multimodal
  - speculative-decoding
  - dflash2
  - not-for-all-audiences
---

# Qwen3.8-27B Uncensored — NVFP4 `.ninfer` artifact (v3 container) for NInfer

An abliterated (refusal-removed) build of Qwen3.8-27B packaged in NInfer's native
**v3 `.ninfer`** container, with the **NVFP4/FP8 mixed-precision weight profile** and the
**DFlash2 + MTP speculative-decoding heads and the vision tower preserved**. Runs on a single
NVIDIA RTX 5090.

This repository publishes the **container upgrade** of an existing community artifact so that it
loads in the current NInfer engine, which requires v3 containers. **No weights were modified,
re-quantized, or fine-tuned.**

| | |
|---|---|
| File | `qwen3_8_27b_nvfp4_uncensored_v3.ninfer` |
| Size | 23,719,759,856 bytes (22.09 GiB) |
| SHA-256 | `d92051ef12c629f2bda98be4ec19059f4556415aa9bfffb0f0bd5d836a32f2d0` |
| Container | NInfer v3 (`NINFER\x00\x03`) |
| Artifact identity | `qwen3.8-27b` / `nvfp4` |
| Objects / tensors | 1190 / 1184 |
| Components | `text`, `vision`, `mtp`, `dflash2`, optimized proposal head |

---

## Provenance

| Stage | Repository / tool | License |
|---|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0 |
| Abliteration (refusal-direction removal on BF16 weights) | [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) | Apache-2.0 |
| Packed NVFP4/FP8 text weights | [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | Apache-2.0 |
| v2 `.ninfer` artifact (source of this file) | [JMVRoill/Qwen3.8-27B-Uncensored-nvfp4-NInfer](https://huggingface.co/JMVRoill/Qwen3.8-27B-Uncensored-nvfp4-NInfer) | Apache-2.0 |
| v2 → v3 container upgrade | [tools/upgrade_ninfer_v2_to_v3.py](https://github.com/Neroued/ninfer) from [Neroued/ninfer](https://github.com/Neroued/ninfer) | (tooling) |

Source artifact (v2) checksum, for transparency:

```
43025bb64f2cb558d9ede6269f8a3c2ed8ebfa4619344fe1ba8e955ba4979218  qwen3_8_27b_nvfp4_uncensored.ninfer
```

### Changes made to produce this file (Apache-2.0 §4(b) notice)

1. **Container upgrade v2 → v3** using the official, upstream tool
   `tools/upgrade_ninfer_v2_to_v3.py`. Per the tool's contract, **stored weight values and
   formats are preserved byte-for-byte**; the tool rewrites the framing/metadata directory and
   installs the maintained `qwen3_8.jinja` chat template. The resulting file is 263,664 bytes
   larger than the v2 source (metadata + template only).
2. The upgrade was executed on Windows, which required four compatibility shims in the tool
   (the upstream tool targets Linux): `os.posix_fadvise`/`POSIX_FADV_*` (no-op),
   `os.fdatasync` → `os.fsync`, `os.sysconf("SC_PAGE_SIZE")` → 4096, and `os.pread`/`os.pwrite`
   emulation via `lseek` + `read`/`write`. These shims only affect durability/access-pattern
   hints; they do not touch the artifact payload.
3. **Nothing else was changed.** No re-quantization, no fine-tuning, no merge, no weight edits.
   All abliteration was performed upstream by OrcaRouter.

### Verified components

`tools.artifact.inspect` reports `text` (Qwen3_5ForCausalLM, 64 layers, hidden 5120,
`max_position_embeddings` 262144), `vision`, `mtp`, `dflash2` and the optimized proposal head —
so MTP and DFlash2 speculative decoding and image/video input are all available.

---

## Requirements

- NVIDIA **RTX 5090** (Blackwell, `sm_120a`); NVFP4 needs FP4 tensor cores
- 64-bit Linux, CUDA toolkit 13.1+ (validated here against the current upstream engine)
- NInfer built from source: <https://github.com/Neroued/ninfer>
  (CMake 3.28+, C++20, Ninja, `pkg-config`, FFmpeg dev libraries, libcurl ≥ 7.85)
- ~24 GiB of VRAM for weights + speculative head + vision, before KV cache

## Quick start

```bash
# one-shot generation
./build/apps/ninfer qwen3_8_27b_nvfp4_uncensored_v3.ninfer \
  --prompt "你好" --max-new 512 --device 0 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft

# OpenAI / Anthropic compatible server
./build/apps/ninfer-serve qwen3_8_27b_nvfp4_uncensored_v3.ninfer \
  --host 0.0.0.0 --port 8080 --model-id qwen3.8-27b \
  --max-context 131072 --kv-capacity auto --kv-dtype fp8 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --vision
```

Notes:

- `--spec dflash2` uses the DFlash2 companion packed in this artifact; drop it (or use
  `--spec mtp --draft-tokens 3`) if you prefer MTP.
- This artifact is an NVFP4-profile artifact; it is **not** interchangeable with
  `groupwise-int` artifacts and will not run on pre-Blackwell GPUs.
- Context ceiling is limited by VRAM. With ~22 GiB of weights, 128K context with FP8 KV is a
  reasonable starting point on a 32 GB card; raise `--max-context` until startup no longer fits.
- Only one Engine (one model, one GPU) per process; capacity is 1–8 active requests.

## Performance

Not benchmarked for this specific file. For reference, the official NVFP4 artifact of the
non-abliterated model measured on the same class of hardware (single RTX 5090), as published by
the NInfer project and reproduced independently: ~80 tok/s decode without speculation, and
~130–220 tok/s decode with DFlash2 K=7 depending on the workload; prefill in the multi-thousand
tok/s range. Expect the same order of magnitude here. Quality loss vs BF16 for this quantization
profile is reported upstream as roughly 1–3 points on hard reasoning benchmarks and near-zero on
general chat/code.

---

## ⚠️ Disclaimer — read before use

This model has had its **safety alignment substantially removed** via *abliteration*
(orthogonalizing the refusal direction out of the residual stream) by the upstream author. As a
direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original
  `Qwen3.8-27B` would refuse. It has **no meaningful built-in guardrails**.
- It is published **strictly for legitimate research** — interpretability, AI-safety and
  refusal-mechanism study, red-teaming, robustness evaluation, and controlled offline experiments.
- **You assume full responsibility and liability** for how you use it and for everything it
  generates. Do not deploy it to end users or in production without adding your own safety,
  moderation, and abuse-prevention layers.
- Use must comply with the **Apache-2.0 License** inherited from the base model, and with all laws
  and regulations that apply to you — including those governing the distribution and use of
  generative AI in your jurisdiction.
- The uploaders and the original authors **accept no liability** for any misuse or harm arising
  from this model. Its outputs do **not** reflect the views of the uploaders, of OrcaRouter, or of
  Qwen / Alibaba.
- This repository is **not affiliated with or endorsed by** Qwen, Alibaba, OrcaRouter, or the
  NInfer project.

By downloading or using this model you acknowledge and accept the above.

## Credits

- **Qwen team (Alibaba)** — Qwen3.8-27B base model and tokenizer
- **[OrcaRouter](https://www.orcarouter.ai)** — abliterated weights and the NVFP4/FP8 packing
- **JMVRoill** — the original v2 `.ninfer` artifact this file is derived from
- **[Neroued / NInfer](https://github.com/Neroued/ninfer)** — the engine, the artifact format, and
  the official v2 → v3 upgrade tool

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE). Inherited from the base model; all upstream
terms and notices are retained.
