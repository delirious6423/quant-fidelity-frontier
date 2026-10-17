---
base_model:
- Qwen/Qwen3.8-Flash-Next
---

## Updates / Notes
- 10/03/2026: I've re-quanted all of the models to include MTP, please re-download.
- 09/02/2026: Added the PLEQ4_0 for IQ2_S and IQ3_S
- 09/01/2026: The original quants all have Q8_0 for the engram embedding tensors, and I've uploaded three variants with Q4_0 for the embedded tensors. The only difference is the Q8_0 vs Q4_0 for those tensors. I'm leaving the original quants up for those who want to use the Q8_0 PLE's.

This repo contains specialized MoE-quants for Qwen/Qwen3.8-Flash-Next. The idea being that given the huge size of the FFN tensors compared to the rest of the tensors in the model, it should be possible to achieve a better quality while keeping the overall size of the entire model smaller compared to a similar naive quantization. To that end, the quantization type default is kept in high quality and the FFN UP + FFN GATE tensors are quanted down along with the FFN DOWN tensors.

| Quant             | Size                  | Mixture                          | PPL                 | 1-(Mean PPL(Q)/PPL(base)) | KLD                 |
| :---------------- | :-------------------- | :------------------------------- | :------------------ | :------------------------ | :------------------ |
| Q5_K_M            | 153.07 GiB (7.32 BPW) | Q8_0 / Q5_K / Q5_K / Q6_K        | 4.596526 ± 0.027937 | +0.1270%                  | 0.020218 ± 0.000145 |
| Q4_K_M            | 131.98 GiB (6.31 BPW) | Q8_0 / Q4_K / Q4_K / Q5_K        | 4.613527 ± 0.028075 | +0.4973%                  | 0.034206 ± 0.000237 |
| IQ4_XS            | 114.99 GiB (5.50 BPW) | Q8_0 / IQ3_S / IQ3_S / IQ4_XS    | 4.739310 ± 0.029099 | +3.2373%                  | 0.076158 ± 0.000486 |
| IQ3_S             | 105.98 GiB (5.07 BPW) | Q6_K / IQ2_S / IQ2_S / IQ3_S     | 4.863539 ± 0.030050 | +5.9434%                  | 0.160378 ± 0.000915 |
| IQ2_S             | 103.63 GiB (4.96 BPW) | Q6_K / IQ2_XS / IQ2_XS / IQ3_XXS | 5.043883 ± 0.031497 | +9.8718%                  | 0.205353 ± 0.001121 |

| Quant             | Size                  | Mixture                          | PPL                 | 1-(Mean PPL(Q)/PPL(base)) | KLD                 |
| :---------------- | :-------------------- | :------------------------------- | :------------------ | :------------------------ | :------------------ |
| Q5_K_M (Q4_0 PLE) | 129.23 GiB (6.18 BPW) | Q8_0 / Q5_K / Q5_K / Q6_K        | 4.620130 ± 0.028097 | +0.6412%                  | 0.030943 ± 0.000212 |
| Q4_K_M (Q4_0 PLE) | 108.14 GiB (5.17 BPW) | Q8_0 / Q4_K / Q4_K / Q5_K        | 4.629426 ± 0.028160 | +0.8437%                  | 0.043475 ± 0.000297 |
| IQ4_XS (Q4_0 PLE) | 91.14 GiB (4.36 BPW)  | Q8_0 / IQ3_S / IQ3_S / IQ4_XS    | 4.759109 ± 0.029230 | +3.6686%                  | 0.084406 ± 0.000531 |
| IQ3_S (Q4_0 PLE)  | 82.13 GiB (3.93 BPW)  | Q6_K / IQ2_S / IQ2_S / IQ3_S     | 4.889509 ± 0.030205 | +6.5091%                  | 0.167198 ± 0.000940 |
| IQ2_S (Q4_0 PLE)  | 79.79 GiB (3.82 BPW)  | Q6_K / IQ2_XS / IQ2_XS / IQ3_XXS | 5.062449 ± 0.031623 | +10.2763%                 | 0.211550 ± 0.001142 |

![kld_graph](kld_data/01_kld_vs_filesize.png "Chart showing Pareto KLD analysis of quants")
![ppl_graph](kld_data/02_ppl_vs_filesize.png "Chart showing Pareto PPL analysis of quants")