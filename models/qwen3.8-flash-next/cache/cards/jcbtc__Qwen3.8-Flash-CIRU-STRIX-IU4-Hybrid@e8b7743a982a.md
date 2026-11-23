---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: llama.cpp
inference: false
tags:
- gguf
- strix-halo
- amd
- rocm
- qwen
- mtp
---

# Qwen3.8 Flash CIRU Strix IU4 Hybrid

This hybrid uses Ciru’s compact C1 routed experts with Unsloth’s UD-Q4_K_XL non-routed weights. The v6.1.0 runtime is built for AMD Strix Halo `gfx1151` on Linux and uses the model’s Q8_0 draft for MTP2. The target GGUF requires that runtime’s C1 support. [Download the runtime and read the release notes](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v6.1.0)

## Run

Allow storage for the 102.53 GB target, 49.86 GB donor shard and 2.79 GB draft. The benchmark used a 128 GB Strix Halo system.

Install Python 3.11 or newer and the Hugging Face CLI. Download and check the target, then fetch the pinned MTP and PLE files. [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

```bash
hf download jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4-Hybrid halo-c1.gguf \
  --revision v6.1.0 --local-dir model
printf '%s\n' '4ff8156aabfa69ac8962a9bbbfded4e23add20f9246f5939cf4915c16e4fb8cb  model/halo-c1.gguf' \
  | sha256sum -c

hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 --local-dir donor
```

The binary uses Linux x86-64 with glibc 2.43 or newer and a libstdc++ exposing GLIBCXX_3.4.32. On NixOS, provide a compatible library directory with `--compat-lib-dir`.

Extract `flash-v6.1.0-gfx1151-linux-runtime.tar.gz` and run these commands from the extracted runtime directory. Adjust the model paths to the directory where you downloaded the files.

```bash
python3 install-rocm.py
python3 run-server.py /path/to/model/halo-c1.gguf \
  /path/to/donor/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  /path/to/donor/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
```

The PLE reader uses the original donor shard’s byte layout. Keep the original shard available when serving.

## Matched performance on Strix Halo

On the same physical 128 GB Strix Halo machine, the HaloBox b1064+C1 control and the v6.1 research build used identical target/draft files and the same operating-system kernel. Each served one uncached 4,096-token request with 512 generated tokens after the same required 12,960-token warmup. Both used MTP2, one slot, a 262,144-token context, and F16 K/V.

| Measurement | b1064+C1 control | v6.1 research build |
|---|---:|---:|
| Prefill throughput | 1,363.63 tok/s | **1,626.27 tok/s** |
| Prefill time | 3,003.74 ms | **2,518.65 ms** |
| Time to first text | 3,080.18 ms | **2,592.54 ms** |
| Generation throughput | 42.22 tok/s | 42.33 tok/s |
| Full response | 15.183 s | **14.665 s** |

The **19.26% prefill gain** saved **485.09 ms**. Both runs produced the same 512 output pieces and accepted 331 of 360 draft tokens. The final kernel change also passed the frozen hidden-state, logit, rollback, and 256-position checks. These figures apply to the tested text workload and settings. [Runtime validation and release notes](https://github.com/ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4/releases/tag/v6.1.0)

## Files and license

| File | Purpose |
|---|---|
| `halo-c1.gguf` | Target model with compact C1 routed expert banks |
| `pins.json` | Source revisions and SHA-256 checksums for the complete model lineage |
| `NOTICE.md` and `LICENSE` | Attribution and Qwen Community License 1.0 |

The model weights are governed by the Qwen Community License 1.0. Qwen created the base model, Unsloth produced the donor and draft GGUFs, and Ciru supplied the compact routed experts and integrated runtime. [License](LICENSE) · [Full provenance and credits](NOTICE.md)
