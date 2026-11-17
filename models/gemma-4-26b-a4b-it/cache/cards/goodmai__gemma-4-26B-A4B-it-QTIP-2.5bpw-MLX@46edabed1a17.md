---
base_model: google/gemma-4-26B-A4B-it
library_name: mlx
license: gemma
tags:
  - mlx
  - basalt
  - apple-silicon
  - qtip
---

<!-- basalt-launch-buttons:start -->
<p align="center">
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-QTIP-2.5bpw-MLX&role=chat"><img src="https://img.shields.io/badge/%E2%96%B6_%D0%97%D0%B0%D0%BF%D1%83%D1%81%D1%82%D0%B8%D1%82%D1%8C_%D1%87%D0%B5%D1%80%D0%B5%D0%B7_Basalt-Apple_Silicon_MLX-0A84FF?style=for-the-badge&logo=apple&logoColor=white" alt="Запустить через Basalt"></a>
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-QTIP-2.5bpw-MLX&role=chat&lang=en"><img src="https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-macOS_App_%26_CLI-111111?style=for-the-badge&logo=swift&logoColor=F05138" alt="Run with Basalt"></a>
  <a href="https://github.com/goodmai/basalt/releases/tag/v0.11.0"><img src="https://img.shields.io/badge/Basalt-v0.11.0-0A84FF?style=for-the-badge&logo=github&logoColor=white" alt="Basalt v0.11.0"></a>
</p>

> **⚡ Запустить в [Basalt v0.11.0](https://github.com/goodmai/basalt/releases/tag/v0.11.0) (macOS 27+ · Swift 6.4 · Apple Silicon, без Python):**
> ```bash
> basalt serve --model goodmai/gemma-4-26B-A4B-it-QTIP-2.5bpw-MLX --rest
> ```
<!-- basalt-launch-buttons:end -->

# gemma-4-26B-A4B-it-QTIP-2.5bpw-MLX

Experimental Basalt QTIP 2.5 bpw trellis-quantized checkpoint of Gemma 4 26B-A4B (205 dense projections quantized with Kronecker RHT and BlockLDLQ Hessian feedback; routed experts kept 4-bit).

## Measured Quality & Footprint (Apple M4 Pro 24 GB)

| Metric | Measured Value |
|---|---|
| **Method** | `QTIP 2.5 bpw Bitshift Trellis (L=16, V=2, HYB Q=9) + Kronecker RHT + BlockLDLQ (#409)` |
| **Weights Footprint** | `14.94 GB (205 dense linears at 2.5 bpw; routed experts 4-bit)` |
| **WikiText-2 16k PPL (`4×4096`)** | `348.23 (Step-0 KL 0.399 vs 2-bit RTN 27,082.19)` |
| **System One Accuracy (`N=300`)** | `294 / 300 (98.0%)` |
| **Reproduce Command** | `basalt models convert mlx-community/gemma-4-26B-A4B-it-qat-4bit --experimental-qtip 2.5 --out ./out` |
