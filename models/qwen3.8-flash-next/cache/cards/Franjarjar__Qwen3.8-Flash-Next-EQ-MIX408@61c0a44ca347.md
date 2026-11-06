---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- espetero
- eq
- quantized
- apple-silicon
- metal
- qwen4_exp
- mixture-of-experts
---

# Qwen3.8-Flash-Next · EQ mix408

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized to **EQ mix408** for
[Espetero](https://github.com/JarJarFran/espetero), a native Metal inference engine for Apple silicon.
EQ (LaneQuant) is the quantization format native to Espetero. It stores 3-bit codebook codes, EQV vectors and 5-bit codes, with a signed 8-bit scale per 16 inputs and an fp16 scale per 256, the 16 inputs of a scale chosen so that every lane of the Apple GPU's matrix kernels needs one scale per step ([the format](https://github.com/JarJarFran/espetero/blob/main/docs/eq.md)).

Two expert formats per layer: the 408 most-routed experts (calibration routing mass, 8 super experts swapped in) 3-bit codebook EQ, the other 104 EQV (8-weight vectors, a 2^13-entry codebook per stack, about 1.7 bits per weight); every other matrix 5-bit EQ; embedding MLX affine 8-bit, LM head 6-bit; routers BF16; no MTP head; the n-gram table MLX affine 4-bit, read from the SSD at run time.

84.0 GB on disk. Bits per weight below are read from the file headers, over the base
model's text parameters, without the MTP head.

## Quality against files of about the same size

| File | Measured with | Bits/weight | KLD mean | KLD median | KLD p99 | Top-1 | Decode M4 Max, tok/s | Decode M1 Max, tok/s |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **this file (mix408)** | Espetero 616c21a | 3.31 | 0.0927 | 0.0129 | 1.2290 | 90.6 % | 44 |  |
| oQ4e-mtp (Jundot) | Espetero 616c21a | 4.58 | 0.1170 | 0.0185 | 1.3967 | 89.3 % | 58 |  |

* Expert order `mix408se_order.json`: the routing-mass order with 8 rarely routed experts whose down projections have outputs at least 4 times their layer's median kept in the 3-bit set (KLD 0.0928 against 0.0942 without, p99 1.23 against 1.27).
* 52 GB of weights stay resident. On an M4 Max, one sequence at 32K context peaks at a 58 GB footprint. It does not load on a 64 GB M1 Max: its GPU working set is 58 GB, and the load stopped at 59 GB wired, 40 of 48 layers in.

Paired comparison over the same reference windows (window bootstrap):

* Against oQ4e-mtp (both run by Espetero 616c21a; oQ4e 38 % larger): KLD -20.8 % (95 % interval -29.2 % to -12.4 %; lower in 92 % of the 24 windows)

Decode: one sequence, 64 teacher-forced steps after a 4K-token prompt (`fnx_speed`, `--mtp none`), M4 Max; tokens per second. Measured on the file with the routing-mass order (same formats and size); the release file decodes at the same speed within 1 % (23.2 against 23.5 ms per step, M4 Max).

## Run it

Needs **Espetero 0.2.0** or later and a Mac with more than 64 GB of memory (52 GB of weights resident; it does not load on a 64 GB Mac). EQ files are Espetero's own format: llama.cpp, MLX
and transformers do not load them.

```bash
brew install JarJarFran/espetero/espetero
espetero serve --model Franjarjar/Qwen3.8-Flash-Next-EQ-MIX408
```

Then open `http://127.0.0.1:8080`, or point any OpenAI, Anthropic or Ollama client at it.

## How it was made

```bash
espetero quantize --model Qwen3.8-Flash-Next --preset fnxmix --keep-experts mix408se_order.json --expert-split 408 \
    --calib calib-x640.safetensors --no-mtp --ple-bits 4 --out Qwen3.8-Flash-Next-EQ-MIX408
```

* Converter: `espetero quantize` of Espetero 0.2.0 (commit `a229d11`); 88 min on an M4 Max, 15 GB peak footprint (timing of the routing-mass-order conversion).
* Calibration: as for the fnx4h file (oQ4e forward, 256K mixed tokens, per-expert down Hessians).
* Rules: preset `fnxmix` with `--expert-split 408` (experts ordered by routing mass, 8 super experts kept in the 3-bit set: `mix408se_order.json`, in this repository).

## How it was measured

KL divergence KL(p_BF16 || p_file) per token, against the BF16 model (fp32 activations; its 360 GB run streamed from disk) over 96,768 tokens: 24 windows of 4096 tokens (8 WikiText-2 test, 8 code, 8 chat turns), the first 64 of each not scored. Rows marked *Espetero 616c21a* are Espetero's own distributions with the public build 616c21a (`kld_dump`, fp16 activations, the file's own LM head). Bits per weight count every stored tensor except the MTP head and the n-gram embedding table (read from the SSD at run time, 32 GB at 4 bits) over the model's text parameters. The paired interval resamples whole windows (10,000 bootstrap samples).

## License

A quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), under its Qwen Community License 1.0 (`qwen-community-1.0`); this
repository keeps that license. The license allows redistributing modified weights and requires its notice in every copy: this repository includes the base model's `LICENSE`. Companies that offer a model as a service or an AI work assistant need a separate license from Qwen for commercial use; large products (more than 100 million monthly users or US$20 million of monthly revenue) must display the model name. Read the full text before using it commercially. Espetero itself is MIT-licensed.
