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

# Qwen3.8-Flash-Next · EQ fnx4h

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized to **EQ fnx4h** for
[Espetero](https://github.com/JarJarFran/espetero), a native Metal inference engine for Apple silicon.
EQ (LaneQuant) is the quantization format native to Espetero. It stores 4-bit codebook, 5-, 6- and 8-bit codes, with a signed 8-bit scale per 16 inputs and an fp16 scale per 256, the 16 inputs of a scale chosen so that every lane of the Apple GPU's matrix kernels needs one scale per step ([the format](https://github.com/JarJarFran/espetero/blob/main/docs/eq.md)).

All 512 routed experts per layer: gate/up 4-bit codebook EQ (5-bit in layers 0-9 and 43-47), down projections 5-bit (4-bit codebook in layers 19-38); every other matrix 8-bit EQ (6-bit in layers 19-38); embedding and LM head MLX affine 8-bit, routers BF16; the MTP head included; the n-gram embedding table MLX affine 4-bit (group 32), read from the SSD at run time.

114.0 GB on disk. Bits per weight below are read from the file headers, over the base
model's text parameters, without the MTP head.

## Quality against files of about the same size

| File | Measured with | Bits/weight | KLD mean | KLD median | KLD p99 | Top-1 | Decode M4 Max, tok/s | Decode M1 Max, tok/s |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **this file (fnx4h)** | Espetero 616c21a | 5.09 | 0.0306 | 0.00366 | 0.3969 | 94.9 % | 51 |  |
| oQ4e-mtp (Jundot) | Espetero 616c21a | 4.58 | 0.1170 | 0.0185 | 1.3967 | 89.3 % | 58 |  |
| EXL3 5.05 | fp32 emulation | 5.20 | 0.0187 | 0.00221 | 0.2446 | 96.0 % |  |  |

* oQ4e decodes 13 % faster on the M4 Max (both with every expert resident): this file spends its extra bits on 8-bit non-expert matrices.
* EXL3 at 5.2 bits has 40 % lower KLD in fp32 emulation (its Metal runtime's speed on this model was not measured here).
* The file has the converter's own 4-bit n-gram table (32 GB of the size). With oQ4e-mtp's table instead, KLD is the same: −0.1 %, 95 % interval −3.3 % to +3.6 %.

Paired comparison over the same reference windows (window bootstrap):

* Against oQ4e-mtp (both run by Espetero 616c21a): KLD -73.8 % (95 % interval -76.4 % to -71.7 %; lower in 100 % of the 24 windows)

Decode: one sequence, 64 teacher-forced steps after a 4K-token prompt (`fnx_speed`, `--mtp none`), M4 Max with 128 GB, both files' experts resident; tokens per second.

## Run it

Needs **Espetero 0.2.0** or later and a Mac with 128 GB of memory (the experts stay resident; 80 GB of weights). EQ files are Espetero's own format: llama.cpp, MLX
and transformers do not load them.

```bash
brew install JarJarFran/espetero/espetero
espetero serve --model Franjarjar/Qwen3.8-Flash-Next-EQ-FNX4H
```

Then open `http://127.0.0.1:8080`, or point any OpenAI, Anthropic or Ollama client at it.

## How it was made

```bash
espetero quantize --model Qwen3.8-Flash-Next-oQ4e-mtp --calib-only --hessian-max 8192 \
    --expert-hessian-max 640 --calib-data file:calib_mix_v1.txt --calib-tokens 262144 --calib-out calib-x640.safetensors
espetero quantize --model Qwen3.8-Flash-Next --calib calib-x640.safetensors --fmt "$(cat rules.txt)" --out Qwen3.8-Flash-Next-EQ-FNX4H
```

* Converter: `espetero quantize` of Espetero 0.2.0 (commit `616c21a`); 70 min on an M4 Max, 12 GB peak footprint (the matrices converted with `--only '*,!*ngram_embedding*'`, the n-gram table from the same binary's `--ple-bits 4` output).
* Calibration: the BF16 model does not fit in memory, so the statistics come from Espetero's forward of the oQ4e-mtp checkpoint over 256K tokens of a mixed text (prose, code, chat), with per-expert down-projection Hessians.
* Rules: the `--fmt` rules in `rules.txt` (in this repository).

## How it was measured

KL divergence KL(p_BF16 || p_file) per token, against the BF16 model (fp32 activations; its 360 GB run streamed from disk) over 96,768 tokens: 24 windows of 4096 tokens (8 WikiText-2 test, 8 code, 8 chat turns), the first 64 of each not scored. Rows marked *Espetero 616c21a* are Espetero's own distributions with the public build 616c21a (`kld_dump`, fp16 activations, the file's own LM head). Bits per weight count every stored tensor except the MTP head and the n-gram embedding table (read from the SSD at run time, 32 GB at 4 bits) over the model's text parameters. The paired interval resamples whole windows (10,000 bootstrap samples).

## License

A quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), under its Qwen Community License 1.0 (`qwen-community-1.0`); this
repository keeps that license. The license allows redistributing modified weights and requires its notice in every copy: this repository includes the base model's `LICENSE`. Companies that offer a model as a service or an AI work assistant need a separate license from Qwen for commercial use; large products (more than 100 million monthly users or US$20 million of monthly revenue) must display the model name. Read the full text before using it commercially. Espetero itself is MIT-licensed.
