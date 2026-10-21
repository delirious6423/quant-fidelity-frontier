---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: gguf
tags:
  - gguf
  - deepseek-v4
  - dspark
  - dflash
  - speculative-decoding
  - draft-model
  - auxiliary-model
  - quantization
  - 1-bit
  - iq1_m
  - iq2_xxs
---

# DeepSeek V4 Flash 0731 — compact DSpark drafter (IQ1_M / IQ2_XXS routed experts)

Tiny speculative-decoding drafters for **DeepSeek-V4-Flash-0731**, built by
crushing the routed-MoE experts down to **1-bit (IQ1_M)** or **2-bit (IQ2_XXS)**
using the public routed-MoE importance matrix (**Importance matrix** (routed-MoE): [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) — `imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat`).

The DSpark drafter is an auxiliary model — **not standalone** — it only works
together with a DeepSeek V4 Flash 0731 target under llama.cpp `--spec-type draft-dspark`.

## Files

| File | Size | Routed experts | Everything else |
| --- | ---: | --- | --- |
| [dspark-DeepSeek-V4-Flash-0731-IQ1_M-exps.gguf](./dspark-DeepSeek-V4-Flash-0731-IQ1_M-exps.gguf) | **4.46 GiB** | `IQ1_M` | `Q8_0` / `F32` / `BF16` |
| [dspark-DeepSeek-V4-Flash-0731-IQ2_XXS-exps.gguf](./dspark-DeepSeek-V4-Flash-0731-IQ2_XXS-exps.gguf) | **5.16 GiB** | `IQ2_XXS` | `Q8_0` / `F32` / `BF16` |
Other variants from the community:
|[singulared/DeepSeek-V4-Flash-0731-DSpark-GGUF](https://huggingface.co/singulared/DeepSeek-V4-Flash-0731-DSpark-GGUF) | **10.9 GiB**  "BF16 (source)" |  `MXFP4` (4-bit) | rest `F32`/`BF16`/`Q8_0` | 
| [Lynxpda/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K_S-GGUF](https://huggingface.co/Lynxpda/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K_S-GGUF) | **\~6.9 GiB** | `Q2_K` | shared experts `Q5_K`, rest `Q6_K`/`Q8_0`/`F16`/`F32`/`BF16` | 



## Performance — DFlash speculative decoding (llama.cpp)

Speculative decoding of **DeepSeek-V4-Flash-0731-UD-Q8_K_XL** using a draft model (block size 5, `--spec-draft-n-max 5`).

### Table 1 — default `--spec-draft-p-min 0.76`

| Draft model | Main model layers in RAM (`--n-cpu-moe`) | Draft model layers | Tokens decoded | Draft acceptance rate | Mean draft length | **Decode throughput (tok/s)** |
|---|---|---|---|---|---|---|
| IQ2XXS | 38 | VRAM+RAM | 54 959 | 0.898 | 3.37 | 14.78 |
| IQ2XXS | 38 | VRAM | 47 667 | 0.890 | 3.24 | **15.27** |
| IQ2XXS | 38 | VRAM\* | 13 038 | 0.880 | 3.11 | 15.50 |
| IQ2XXS | 38 | RAM | 15 197 | 0.906 | 3.56 | 13.09 |
| IQ1M | 38 | VRAM+RAM | 45 243 | 0.884 | 3.19 | 14.33 |
| IQ1M | 38 | VRAM | 57 455 | 0.882 | 3.23 | **15.46** |
| IQ1M | 40 | VRAM | 14 961 | 0.894 | 3.27 | 14.80 |
| BF16 | 38 | VRAM+RAM | 15 343 | 0.847 | 2.98 | 13.81 |
| BF16 | 42 | VRAM | 32 061 | 0.867 | 3.19 | 14.02 |
| DSpark-Drafter Q2_K_S | 38 | VRAM+RAM | 23 326 | 0.868 | 3.17 | 14.30 |
| DSpark-Drafter Q2_K_S | 38 | VRAM | 13 269 | 0.876 | 3.14 | 15.37 |

### Table 2 — `--spec-draft-p-min` sweep (IQ2XXS, draft experts on CPU)

| `--spec-draft-p-min` | Main model layers in RAM (`--n-cpu-moe`) | Draft model layers | Tokens decoded | Draft acceptance rate | Mean draft length | **Decode throughput (tok/s)** |
|---|---|---|---|---|---|---|
| 0.46 | 38 | VRAM+RAM | 19 506 | 0.647 | 3.04 | 13.32 |
| 0.66 | 38 | VRAM+RAM | 11 467 | 0.798 | 3.03 | 13.87 |
| **0.76** | 38 | VRAM+RAM | 54 959 | 0.898 | 3.37 | **14.78** |
| 0.80 | 38 | VRAM+RAM | 8 793 | 0.916 | 3.13 | 13.99 |
| 0.86 | 38 | VRAM+RAM | 15 957 | 0.960 | 3.20 | 13.45 |
| 0.96 | 38 | VRAM+RAM | 23 870 | 0.995 | 3.09 | 12.27 |

### Draft model layers — legend

- **VRAM** — draft model fully on GPU (default).
- **VRAM+RAM** — `--spec-draft-ngl 99 --spec-draft-override-tensor "exps=CPU"` (experts offloaded to CPU).
- **RAM** — `--spec-draft-ngl 0` (draft model fully on CPU).

### Notes

- \* also sets `--spec-draft-type-k q8_0 --spec-draft-type-v q8_0` (quantized draft KV cache).
- **Tokens decoded** — total generated tokens across the aggregated runs (all metrics are token-weighted averages).
- **Draft acceptance rate** — fraction of speculative tokens accepted by the target model.
- **Mean draft length** — average number of accepted draft tokens per step .
- **Decode throughput** — tokens/s during single-stream autoregressive decoding.

### Test environment

```
Software:  llama.cpp version: 10273 (a6aa6f545) built with GNU 15.2.1 for Linux x86_64
Target:    DeepSeek-V4-Flash-0731-UD-Q8_K_XL (Q8_K_XL)
Drafts:    dspark-DeepSeek-V4-Flash-0731-{IQ1M,IQ2XXS}-exps · BF16 · DSpark-Drafter-Q2_K_S-dflash
Decoding:  speculative, block size 5, --spec-draft-n-max 5, --spec-draft-p-min 0.76
Context:   --ctx-size 266240, single slot (--parallel 1)
Batch:     --n-batch 2048, --n-ubatch 2048
Offload:   --n-gpu-layers 99 (target & draft), --n-cpu-moe 38 (40/42 where noted)
GPUs:      GF4090 + GF3090, --tensor-split 36:6 (6:6 where noted)
CPU pin:   taskset -c 0-15, Intel® Core™ i9-14900K, 196Gb DDR5 5200.
```

### Reproduction

```
taskset -c 0-15 ./llama.cpp/build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
  -md dspark-DeepSeek-V4-Flash-0731-IQ1M-exps.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0.76 \
  --spec-draft-ngl 99 --ctx-size 266240 --n-cpu-moe 38 \
  --tensor-split 36,6 -ngl 99
```


## How it was built

Selective requantization of the BF16 drafter with `llama-quantize`, keeping
only the 9 routed-expert tensors at low bit and the rest at Q8_0:

```bash
llama-quantize \
  --imatrix DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat \
  --allow-requantize \
  --tensor-type blk.0.ffn_gate_exps.weight=IQ1_M \
  --tensor-type blk.0.ffn_up_exps.weight=IQ1_M \
  --tensor-type blk.0.ffn_down_exps.weight=IQ1_M \
  # ... same for blk.1, blk.2 ...
  dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  dspark-DeepSeek-V4-Flash-0731-IQ1_M-exps.gguf Q8_0
```

## Credits / sources

- **Importance matrix** (routed-MoE): [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) — `imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat`
- **BF16 drafter weights** (source of this quantization): [singulared/DeepSeek-V4-Flash-0731-DSpark-GGUF](https://huggingface.co/singulared/DeepSeek-V4-Flash-0731-DSpark-GGUF)
- **Other DSpark drafter references / mixed recipe**: [dev7a/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/dev7a/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF), [alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF), [GaelicThunder](https://huggingface.co/GaelicThunder/DSpark-DeepSeek-V4-Flash-0731-drafter-dflash-GGUF), [Lynxpda](https://huggingface.co/Lynxpda/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K_S-GGUF)
- **Base model**: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

Weights are redistributed under the base model's release terms (MIT).
