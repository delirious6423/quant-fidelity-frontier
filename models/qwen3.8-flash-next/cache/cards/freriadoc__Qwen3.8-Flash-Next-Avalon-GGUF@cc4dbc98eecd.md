---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- moe
- mtp
- speculative-decoding
- rocm
- avalon
---

# Qwen3.8-Flash-Next · Avalon GGUF (IQ3_KS hybrid + MTP sidecar)

GGUF weights for [Avalon](https://github.com/freriadoc/Avalon), a llama.cpp fork that runs Qwen3.8-Flash-Next (512
experts, 10 active, 48 layers, ~88 GB) on a single 20 GB GPU with the experts in system RAM and a GPU expert cache.

**These files only load in Avalon.** They use two quantization types that stock llama.cpp does not have (IQ3_K and
IQ3_KS from ik_llama.cpp, with Avalon's own type ids, not ik's), a packed expert layout, and an MTP sidecar format with
a cut LM head (`d2t`). Stock llama.cpp and ik_llama.cpp will refuse them.

## Files

| file | size | what |
|---|---|---|
| `Qwen3.8-Flash-Next-Avalon-IQ3KS-000NN-of-00033.gguf` | 89 GiB | the main model (IQ3_KS hybrid, packed experts) |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf` | 1.48 GB | MTP draft head (speculative decoding, 2 draft tokens) |
| `expert-profile.bin` | 0.2 MB | expert ranking to prefill the GPU cache at startup |

### Main model

- Dense part (attention, routers, shared experts, embeddings, LM head): from AtomicChat's AD-4.27bpw quantization.
- Routed experts: ISTA-DASLab's GSQ-RCO quantization (IQ3_S model), with the up/gate tensors of
  - its 17 IQ3_XXS layers requantized to **IQ3_KS** (3.25 bpw, per-32 scales), and
  - its 10 IQ3_S layers requantized to **IQ3_K** (3.44 bpw),
  both from the original BF16 weights with an importance matrix. The rest is ISTA's as is: up/gate IQ2_S (20 layers)
  and IQ4_XS (1 layer), down IQ4_NL (39 layers) and Q2_0 (9 layers).
- Experts packed per layer (up | gate | down contiguous), so a GPU cache fill is one copy per expert.

### MTP sidecar

The official NextN/MTP head of Qwen3.8-Flash-Next, with its experts requantized (up/gate IQ3_K, down IQ4_NL, with an
importance matrix), the rest Q8_0, and a cut LM head over the 65,536 most frequent tokens (FR-Spec), whose rows are
copied from the main model's head. It borrows the token embeddings from the main model ("shared" head).

## Quality

Perplexity, 8 × 4,096 tokens of external texts (lower is better):

| quantization | English | French | code |
|---|---|---|---|
| Unsloth UD-Q4_K_XL | 1.316 | 3.581 | 1.325 |
| Unsloth UD-IQ4_XS | 1.401 | 3.841 | 1.343 |
| **this model (IQ3_KS hybrid)** | **1.443** | **3.857** | **1.331** |
| ISTA GSQ-RCO IQ3_S | 1.469 | 3.932 | 1.338 |
| AtomicChat AD-4.27bpw | 1.509 | 4.073 | 1.347 |
| ISTA GSQ-RCO IQ3_XXS | 1.596 | 4.294 | 1.364 |

MTP acceptance on a fixed text (prompt of 19k tokens, 1,023 positions, 2 draft tokens): 80.7 %.

## Speed

Measured on modest hardware: RX 7900 XT 20 GB (RDNA3, PCIe 3.0 x16), Ryzen 7 3800XT (8 cores), 96 GB DDR4-3200 dual
channel, ROCm 7.2. Greedy decoding, 512 tokens, median of 3 runs, MTP with 2 draft tokens:

| case | generation, no MTP | generation, MTP | prefill |
|---|---|---|---|
| 32k context, short prompt, warm cache | 62.4 t/s | 91.0 t/s | 1,661 t/s* |
| 256k context, 131k filled | 55.6 t/s | 74.5 t/s | 1,496 t/s |
| 256k context, 250k filled | 50.5 t/s | 69.8 t/s | 1,376 t/s |

\* Maximum prefill, measured with a 32k prompt filling the 32k context. Prefill figures are without MTP (with MTP:
1,544 / 1,387 / 1,273 t/s).

Generation speed with MTP depends on how predictable the text is (acceptance 0.6 – 0.95 in these runs).

## Usage

Build Avalon (see its README), then:

```bash
llama-server -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf \
  -c 262144 -b 8192 -ub 8192 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type draft-mtp \
  --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0
```

`--fit` (on by default) places the dense weights on the GPU and the experts in RAM; the expert cache takes the rest of
the VRAM. To prefill the cache at startup (first answers of a fresh server ~5 – 8 % faster):
`GGML_CUDA_MOE_CACHE_PRELOAD=expert-profile.bin`. The ranking was recorded with buun-llama-cpp's expert heat profile
over ~18 M routings of our runs.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model and its MTP head.
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab): the GSQ-RCO expert quantization ([GSQ](https://arxiv.org/abs/2604.18556), [RCO](https://arxiv.org/abs/2605.00649)).
- [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF): the AD-4.27bpw quantization (dense part).
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF): the shared BF16 MTP head we requantized.
- [drluoto / Vitronia](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF): the FR-Spec list of 65,536 tokens.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): the IQ3_K and IQ3_KS types.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp): the base of Avalon and its expert cache;
  buun's expert heat profile recorded `expert-profile.bin`.

## License

Qwen Community License 1.0, as the base model (see `LICENSE`).
