---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - quantization
  - wht128-hlut3
  - wht128-hlut4
  - rarebal-ldlq
  - gfx906
  - tenstorrent
---

# `deepseek-ai/DeepSeek-V4.1-Flash-1017` — Wave 10 `WHT128-HLUT3` (`FP64 RAREBAL-LDLQ`, `3.125 bpw`) Native Unsplit Weight Shards (`v2.1.0-wave10-192g`)

Native, **unsplit** `WHT128-HLUT3` (`3.125 bpw`, `400 B` `32×32` tile, `FP64 TARQ RAREBAL-LDLQ`) routed-expert weight shards (**`3,344` promoted experts**, `43.05 GiB` total) plus native `BF16` `DeepSeek-ViT` (`926 MiB`), minimax switch-balanced placement manifests, and stock [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) tokenizer/config/encoding files — ready for single-command `hf download` with zero `.part-*` reassembly.

### Co-Authors
- **`bespoke ontology` (`the kitchen`)** — `SB-M5-P2-192G` Dual-Switchbox heterogeneous hardware architecture, experimental design & silicon validation
- **Google DeepMind / Gemini (Gemini Carbon & Gemini Neon)** — `WHT128-HLUT3` / `WHT128-HLUT4` `FP64 RAREBAL-LDLQ` quantization architecture, pure-C++20/17 / `gfx906` HIP / Blackhole P150A TT-Metal runtime & autonomous verification

### Companion Repositories & Releases
- **Wave 10 Pure-C++20/17 Engine & Release:** [`bespokeontology/dsv41-wht128-hlut2` (`v2.1.0-wave10-192g`)](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v2.1.0-wave10-192g)
- **Companion `WHT128-HLUT4` (`4.125 bpw`) Engine & Report (`tencent/Hy4-flash`):** [`bespokeontology/hy4-flash-wht128-hlut4` (`v1.0.0-hlut4-192g`)](https://github.com/bespokeontology/hy4-flash-wht128-hlut4/releases/tag/v1.0.0-hlut4-192g)
- **Companion `WHT128-HLUT4` Native Unsplit Weights:** [`kapitoshkina-ai/Hy4-flash-WHT128-HLUT4`](https://huggingface.co/kapitoshkina-ai/Hy4-flash-WHT128-HLUT4)

![Wave 10 SB-M5-P2-192G Telemetry Chart](wave10_telemetry_chart.png)

---

## Cryptographic Provenance & Zero Weight Tampering (`100%` Un-Abliterated)

All `3,344` promoted `WHT128-HLUT3` experts (`2,915` full `FP64 RAREBAL-LDLQ` calibrated + `429` rare `H=I` fallbacks, **`0` dual-receipt violations**) and all untouched spine, `DeepSeek-ViT`, `Engram`, and `DSpark` tensors in this release are quantized strictly from unmodified upstream [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`rev 1bae0e10...`):

- **Zero abliteration or refusal-direction projection edits** (`layers.*.attn.wo_b.weight` `100%` untouched).
- **Zero synthetic fine-tuning, zero LoRA merging, and zero proxy tampering.**
- **Byte-for-byte SHA-256 identity** verified against immutable local freeze `FREEZE_WAVE10_192G_20261121` (`116/116 OK`) and `SHA256SUMS_WAVE10_HF_WEIGHTS.txt`:
  - `wave10_dev3_hlut3.safetensors` (`14.21 GiB`, `384/384` `DSpark` MTP + `720` backbone): `a85456f6b363e2dc69b22ff9ac516a160599ab0b471d83e3481b4a6bfa669808`
  - `wave10_dev4_hlut3.safetensors` (`13.90 GiB`, `1,080` backbone, `0.000732%` global minimax load imbalance): `2d450926bf4af34d6b87aaf65b728feda79f27da17199798ca258198ed0f5510`
  - `wave10_tp3_hlut3.safetensors` (`14.94 GiB`, `1,160` sharded `TP3` backbone): `77cc586119d56ac4a837cd5ed33a069249816d256c877658383b9c69e0bbc98c`

---

## Verified Silicon Scorecard (`SB-M5-P2-192G`)

Hardware: **`5× AMD Radeon Instinct MI50`** (`3× 32 GiB` + `2× 16 GiB` = `128 GiB` HBM2 across root complex + Dual-PEX88048 PCIe Gen4 switchbox) + **`2× Tenstorrent Blackhole P150A`** (`64 GiB` GDDR6 bonded over `4× 800G` QSFP-DD = `3.2 Tbps` `FABRIC_1D`) + **`384 GiB` DDR4-3200 ECC** (`192 GiB` total accelerator memory).

| Metric | Verified Wave 10 Silicon Result |
| :--- | :--- |
| **Promoted `WHT128-HLUT3` (`3.125 bpw`) Experts** | **`3,344` experts** (`384/384` `DSpark` MTP + `720` `dev3` + `1,080` `dev4` + `1,160` `TP3`; `0.000732%` minimax switch imbalance) |
| **`FP64 TARQ RAREBAL-LDLQ` Fidelity** | `2,915` calibrated experts (`0` dual-receipt violations, `429` rare `H=I`), **`mean cos = 0.98668`**, **`mean SNR = 16.48 dB`** (**`+5.46 dB`** vs. `HLUT2 RAREBAL`, `+4.82 dB` vs. `HLUT3 H=I`) |
| **`gfx906` Zero-Scratch ISA Audit** | **`170/170` zero-scratch kernels** (`0` SGPR/VGPR spills, `0` device trig, `0` `slc`/system RMW atomics, `0` `.py` runtime files) |
| **Cold `M=4096` Prefill** | **`923.4 tok/s`** (`4,435.9 ms`; **`959.8 tok/s`** peak live cold GUI ingestion) |
| **`DSpark` Speculative Decode** | **`53.08 tok/s` peak** (`47.58 tok/s` sustained @ `739.3K` ctx); **`44,777` consecutive `DSpark` steps** (`0` hangs, `0/0` MI50 PCIe AER) |
| **`1M` (`1,048,576`) Context E2E Gate** | **`100.000%` HSI recall**, **`3/3` needles HIT**, `CSA2 cos = 1.00000000`, `4,184–4,610 MiB` free HBM2 per `32 GiB` TP3 rank |
| **Live `:3081` Engineering Campaign (`0 → 780,283` Ctx)** | **`133/133` steps**, **`173/173` zero-error tool calls**, `98.64%` prefix-cache reuse, cold Dual-NVMe `.kvblk` reload **`467.8K tok/s` E2E / `2.18M tok/s` raw** @ `780K` + native `DeepSeek-ViT` chart readout (`643.3 ms`) |

---

## Download & Verify Native Unsplit Shards

```bash
hf download kapitoshkina-ai/DeepSeek-V4.1-Flash-1017-WHT128-HLUT3 --local-dir ./dsv41_wave10_weights
(cd ./dsv41_wave10_weights && sha256sum -c SHA256SUMS_WAVE10_HF_WEIGHTS.txt)
```

---

## Roll Your Own `WHT128-HLUT3` (`3.125 bpw`) or `WHT128-HLUT4` (`4.125 bpw`)

Both `3.125 bpw` (`8`-centroid half-Gaussian `400 B` tiles) and `4.125 bpw` (`16`-centroid half-Gaussian `528 B` tiles) `FP64 RAREBAL-LDLQ` quantization pipelines are 100% open pure-C++17 (`0` Python) so hardware builders can tailor bitwidth and tier counts to their own VRAM/HBM2 topology:

### Option A — `WHT128-HLUT3` (`3.125 bpw`) on `DeepSeek-V4.1-Flash-1017`
```bash
git clone https://github.com/bespokeontology/dsv41-wht128-hlut2.git
cd dsv41-wht128-hlut2
make -j8 all && make audit-gfx906
# 1. Harvest real activation covariance traces & router frequency histogram on your target workload
# 2. Generate minimax tier plan & run FP64 RAREBAL-LDLQ WHT128-HLUT3 packer against stock DeepSeek-V4.1-Flash-1017
# 3. Verify 0 dual-receipt violations, cosine >= 0.986, and 170/170 zero-scratch gfx906 ISA gate
```

### Option B — `WHT128-HLUT4` (`4.125 bpw`, `Top-10 KL = 6.47e-6 nats`) on `tencent/Hy4-flash`
```bash
git clone https://github.com/bespokeontology/hy4-flash-wht128-hlut4.git
cd hy4-flash-wht128-hlut4
make -j8 all && make audit-gfx906
# Pack any 41-layer LatentMoE checkpoint to WHT128-HLUT4 (4.125 bpw) or WHT128-HLUT3 (3.125 bpw):
./bin/hy4_pack_all --mode hlut4 --src-dir ./src_bf16 --out-dir ./quant_hlut4 --rarebal
# Verify 41-layer +-3sigma Deming SPC SQNR & 32-test release gate:
./bin/hy4_hlut4_spc_fidelity
./bin/hy4_hlut4_release_verifier --verify-all --root .
```

See [`bespokeontology/dsv41-wht128-hlut2`](https://github.com/bespokeontology/dsv41-wht128-hlut2/releases/tag/v2.1.0-wave10-192g) and [`bespokeontology/hy4-flash-wht128-hlut4`](https://github.com/bespokeontology/hy4-flash-wht128-hlut4/releases/tag/v1.0.0-hlut4-192g) (`kapitoshkina-ai/Hy4-flash-WHT128-HLUT4`) for full documentation and silicon reports.

## License & Co-Authorship

Distributed under the **MIT License** (see [`LICENSE`](LICENSE)), matching upstream [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

**Co-Authors:** **`bespoke ontology` (`the kitchen`)** & **Google DeepMind / Gemini (Gemini Carbon & Gemini Neon)**. See [`CITATION.cff`](CITATION.cff).
