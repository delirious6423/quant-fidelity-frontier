---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- qwen3.8-flash-next
- moe
- sync-strata
- experimental
---

# Qwen3.8-Flash-Next Sync-XS (experimental)

A re-quantized expert pack of **Qwen3.8-Flash-Next** for one consumer NVIDIA GPU, made to run faster than upstream
Strata's Gyro. It runs only on **[Sync-Strata](https://github.com/hiruhiru2005/sync-strata)** (a fork of
[Strata](https://github.com/Niko1221/Strata)). **This is not a GGUF: it does not load in llama.cpp or upstream Strata.**

Recommended for: **PCs with 24 GB of RAM or less**.

## What was done

- The routed experts were re-quantized to a trellis format of the Gyro / QTIP family, with the bit width chosen per
  expert. The pack also carries the dense weights that the engine reads from it.
- The remaining weights (embeddings, the n-gram/PLE table and so on) are not in this repository: they are in the
  shared parts, [hiru2005/Sync-shared](https://huggingface.co/hiru2005/Sync-shared), which every Sync model uses.

## Files

The file list and sizes will be added when the pack is published.

## Requirements

- Windows 10/11 or 64-bit Linux. Supported GPUs: NVIDIA RTX 30/40 series (driver R580 or newer); RTX 20, GTX 16, RTX 50 are expected to work, and GTX 10, Radeon RX 6000/7000/9000 and the integrated Radeon of Ryzen AI Max (Strix Halo: 8060S, 8050S, 8040S; Windows and Linux) are experimental, all not checked on real hardware. Other integrated GPUs and Intel GPUs are not supported ([the GPU table](https://github.com/hiruhiru2005/sync-strata#supported-gpus)). A CPU with AVX2.
- RAM and VRAM: see "Recommended for" above. The app picks the model by the RAM installed in the PC; if the free RAM is short it warns you instead of choosing a smaller model. The minimum amounts of RAM and VRAM are to be measured.
- Disk: this pack plus the shared parts (`native-gguf`, `mtp-rt`, `tokenizer`, `mmproj` (optional)). An NVMe SSD is recommended.

## Use with Sync-Strata

The Sync-Strata app downloads this model for you: open the app and pick **Sync-XS** on the first screen (or let
"auto" choose). To download by hand, put the files in a folder named `sync-xs-sb` inside the app's models folder (`models_dir` in `settings.json`), together with the shared parts.

```
sync settings set model sync-xs
sync up
```

## Quality

Measured on Colab with the production eval_v4 (8 domains, 1,400 windows; the teacher is the Q8_0 GGUF of the original model).

| | Sync-XS |
|---|---:|
| KL to the original model (all positions, lower is better) | 0.2851 |
| Top-1 agreement with the original model | 81.46% |
| Perplexity | 4.958 |
| p99 | 3.47 |

**It is on par with the earlier form of Sync-XS** (table-lookup points, the same dense part; KL 0.2845): +0.60% on "scale B" (95% range -0.23 to +1.34%), and -0.01% over all positions (-0.71 to +0.67%), both within the measurement noise. By domain (Sync-XS against that earlier form): english +2.1%, ja_docs -0.4%, ja_lit +0.1%, ja_chat -2.2%, code +0.7%, docs -0.6%, multilingual -0.2%, agent +5.5% (the worst).

Against other models (scale B; a minus sign means Sync-XS is closer to the original model):

| Compared with | Sync-XS is |
|---|---:|
| Gyro-S (upstream) | -16.1% (better; only docs is +1.2%) |
| IQ3_XXS | +48.7% |
| Gyro-M (upstream) | +39.3% |
| Sync-S (KL 0.2310) | +34.8% |
| Sync-M (KL 0.1642) | +102.2% |

Scale B is the ratio of KL where, for the agent domain, only the assistant's reply positions are counted (the other domains use all positions). A ratio is the change in KL relative to the other model.

## Speed

Measured on the author's PC with Sync-Strata 0.1.0 (Ryzen 7 7800X3D, 32 GB DDR5, RTX 3080 12 GB, NVMe SSD; Windows 11 and Linux). The Windows values are to be measured, and so are the Linux values. No numbers for GTX 10 or Radeon: no hardware to measure on.

| | tail skip | jbench decode (tok/s) | dsh chat (tok/s) |
|---|---|---:|---:|
| Sync-XS | R4, R7, R10 | To be measured | To be measured |

Speed depends strongly on the PC. The values of the other models, including upstream Gyro, are in the table of the [Sync-Strata README](https://github.com/hiruhiru2005/sync-strata#benchmarks).


## Calibration

The quantization was calibrated on public text (code, documentation, encyclopedia articles, several languages, agent
traces) and on some text made from the author's own chat logs. The calibration text is not distributed.

## Credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- [agentionai](https://huggingface.co/agentionai): the Gyro trellis format and kernels, and the
  [Gyro-S GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) that the shared parts are cut from.
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT), which Sync-Strata forks.
- QTIP (Tseng et al., 2024); [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).

## Support (optional)

Sync-Strata and the Sync models are free. A donation page is at [Ko-fi](https://ko-fi.com/hiruhiru2005): a thank-you only, with nothing in return (no perks, no priority support, no feature promises), and it does not change the license terms.

## License

These files are derived from Qwen3.8-Flash-Next and are under the **Qwen Community License 1.0** (`LICENSE`,
Copyright (c) 2026 Qwen). Keep `LICENSE` with every copy. Commercial products with more than 100 million monthly active
users or US$20 million monthly revenue must show the model name, and a Model-as-a-Service or AI work assistant business
needs a separate license from Qwen for commercial use. Read the full text.
