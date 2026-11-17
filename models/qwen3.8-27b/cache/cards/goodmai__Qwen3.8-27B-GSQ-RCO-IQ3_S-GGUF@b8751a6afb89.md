---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - basalt
  - gsq-rco
  - iq3_s
---

<!-- basalt-launch-buttons:start -->
<p align="center">
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-GGUF&role=gguf"><img src="https://img.shields.io/badge/%E2%96%B6_%D0%97%D0%B0%D0%BF%D1%83%D1%81%D1%82%D0%B8%D1%82%D1%8C_%D1%87%D0%B5%D1%80%D0%B5%D0%B7_Basalt-Apple_Silicon_MLX-0A84FF?style=for-the-badge&logo=apple&logoColor=white" alt="Запустить через Basalt"></a>
  <a href="https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-GGUF&role=gguf&lang=en"><img src="https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-macOS_App_%26_CLI-111111?style=for-the-badge&logo=swift&logoColor=F05138" alt="Run with Basalt"></a>
  <a href="https://github.com/goodmai/basalt/releases/tag/v0.11.0"><img src="https://img.shields.io/badge/Basalt-v0.11.0-0A84FF?style=for-the-badge&logo=github&logoColor=white" alt="Basalt v0.11.0"></a>
</p>

> **⚡ Запустить в [Basalt v0.11.0](https://github.com/goodmai/basalt/releases/tag/v0.11.0) (macOS 27+ · Swift 6.4 · Apple Silicon, без Python):**
> ```bash
> basalt models pull goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-GGUF
> basalt models convert ~/.cache/huggingface/hub/models--goodmai--Qwen3.8-27B-GSQ-RCO-IQ3_S-GGUF/snapshots/*/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
>   --template goodmai/Qwen3.8-27B-3bit-MLX-TextOnly --out ./qwen38-gsq-native
> basalt serve --model-path ./qwen38-gsq-native --rest
> ```
<!-- basalt-launch-buttons:end -->

# Qwen3.8-27B-GSQ-RCO-IQ3_S-GGUF

Source `IQ3_S` GGUF checkpoint (`11.77 GB`) used for Basalt native i-quant / GSQ-RCO Metal kernel evaluation (`#408`).
