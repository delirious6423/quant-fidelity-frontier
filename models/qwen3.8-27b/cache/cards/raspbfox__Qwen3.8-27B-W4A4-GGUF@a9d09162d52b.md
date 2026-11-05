---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - w4a4
  - int4
  - quantized
  - amd
  - mi50
  - gfx906
  - speculative-decoding
  - dflash
---

# Qwen3.8-27B W4A4 GGUF (int4 weights × int4 activations)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized by [ruquant](https://github.com/skyne98/ruquant)
to pure W4A4. Both the weights and the activations are int4, and the matmuls run as int4 × int4 dot products
(`v_dot8_i32_i4` on AMD gfx906 / MI50). A W4A4H version of incoai's DFlash2 drafter comes with it, for speculative
decoding.

**These files need the ruquant fork of mx-llama.cpp:**
[skyne98/mx-llama.cpp, branch `ruquant`](https://github.com/skyne98/mx-llama.cpp/tree/ruquant). Stock llama.cpp does
not know the W4A4 / W4A4H tensor types. The fork was built and measured on MI50s (gfx906) with ROCm 5.7.

| file | size | what |
|---|---|---|
| `Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf` | 17.19 GB | the target model |
| `Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf` | 2.65 GB | the DFlash2 drafter converted for it |
| `SHA256SUMS` | | checksums |

## Results (4× AMD MI50 32 GB, ROCm 5.7, PCIe 3.0)

Measured in one session on 2026-11-05. The comparison is unsloth's Q4_0 (4-bit weights, 8-bit activations) with
incoai's DFlash2. Speculative runs: 30 prompts × 256 tokens, greedy, draft 7 (Q4_0: 4, its best).

| | 1 GPU | TP2 | TP4 |
|---|---|---|---|
| **W4A4 + DFlash2, t/s** | **80.2** | **89.5** | **109.6** |
| Q4_0 + DFlash2, t/s | 43.5 | 60.5 | 70.5 |
| W4A4 acceptance (tokens per step) | 4.35 | 4.27 | 4.31 |
| verify forward of 8 tokens, ms: W4A4 / Q4_0 | 42.2 / 95.5 | 35.1 / 62.8 | 27.6 / 40.6 |
| prefill pp2048, t/s: W4A4 / Q4_0 | 487.8 / 290.7 | 833.1 / 529.0 | 1324.0 / 901.4 |
| plain decode, t/s: W4A4 / Q4_0 | 29.2 / 27.9 | 41.6 / 39.7 | 52.7 / 51.5 |

Quality, KL divergence against UD-Q8_K_XL's logits (wikitext-2, 64 chunks of 2048):

| | KLD | top-1 | PPL / base |
|---|---|---|---|
| unsloth UD-Q4_K_XL | 0.0078 | 96.58% | 0.999 |
| unsloth Q4_0 | 0.0259 | 93.34% | 1.022 |
| **this model (W4A4)** | **0.0428** | **91.57%** | 1.028 |

## Run it

Build the fork on a gfx906 box (needs ROCm, cmake and ninja):

```sh
git clone -b ruquant https://github.com/skyne98/mx-llama.cpp && cd mx-llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=ON -DLLAMA_OPENSSL=OFF
cmake --build build -j --target llama-server llama-bench llama-perplexity
```

Serve with tensor parallelism over 4 GPUs and speculative decoding:

```sh
export GGML_CUDA_DISABLE_GRAPHS=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000
build/bin/llama-server -m Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf -md Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf \
    -ngl 99 -ngld 99 -fa 1 -sm tensor -devd ROCm0 \
    --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature
```

Run notes:
- One GPU: drop `-sm tensor -devd ROCm0`.
- Long prompts: add `-b 2048 -ub 2048 --ctx-checkpoints 0`.
- The environment variables are for ROCm 5.7:
  - HIP graphs are off because they are slower there;
  - fine-grained PCIe memory is needed by the peer-write all-reduce;
  - spinning lane threads add ~2%.
- The drafter must be pinned to one GPU (`-devd`): replicated or split, it is 19-24% slower.

## How it was made

- **Types**: unsloth's Qwen3.8-27B-Q4_0 type map. Its 4-bit linears become `W4A4`, and attn_output / ffn_down
  become `W4A4H` (an online Hadamard first). `ssm_out` and the lm_head are Q8_0.
- **Rotation**: OSTQuant, a learned R1 rotation plus smoothing scales, folded into the weights. The online Hadamard
  is block-diagonal, 4 blocks per row (Paley × Sylvester), so one file serves 1, 2 and 4 GPUs.
- **Weights**: GPTQ against Hessians collected on the rounded model (64 wikitext windows, every position). 4.5 bits
  per weight: a per-row scale and a 4-bit multiplier per group of 64.
- **Activations**: quantized on the fly per token, asymmetric int4 in groups of 64. Each group's scale is an integer
  multiple of a per-token unit, so the dot product stays in int32 until one fp32 epilogue. No activation clip.
- **Drafter**: incoai's [Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). Its `fc` takes the
  target's R1 and stays Q8_0; its layers and lm_head become W4A4H.

Every kernel is bit-exact with the CPU reference implementation. The full recipe, methodology, measurements and what
did not work: [RESULTS.md](https://github.com/skyne98/ruquant/blob/main/RESULTS.md).

## Limits

- Quality is below unsloth's Q4_0, which keeps 8-bit activations: KLD 0.043 against 0.026, top-1 91.6% against 93.3%. That is the price of 4-bit
  activations, and it buys the int4 × int4 speed.
- The kernels are written for gfx906. The fork's CUDA path emulates dot8 and is only a correctness check. The ruquant
  repo has a separate llama.cpp patch with a sm_80+ CUDA path, used for quality scoring on an RTX 3090.

## Credits and license

Apache-2.0, like its sources:
- Qwen3.8-27B by Qwen;
- the GGUF source and type map by unsloth;
- the DFlash2 drafter by Inco AI;
- the runtime builds on mx-llama.cpp (MIT) and llama.cpp (MIT).

Methods: OSTQuant (Hu et al., 2025), GPTQ (Frantar et al., 2022), DFlash speculative decoding.
