---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- espetero
- eq
- quantized
- apple-silicon
- metal
- qwen3_5
---

# Qwen3.8-27B · EQD6 (preset eqd6-27b)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized to **EQD6 (preset eqd6-27b)** for
[Espetero](https://github.com/JarJarFran/espetero), a native Metal inference engine for Apple silicon.
EQ (LaneQuant) is the quantization format native to Espetero. It stores 5-, 6- and 8-bit codes, with a signed 8-bit scale per 16 inputs and an fp16 scale per 256, the 16 inputs of a scale chosen so that every lane of the Apple GPU's matrix kernels needs one scale per step ([the format](https://github.com/JarJarFran/espetero/blob/main/docs/eq.md)).

Every matrix 6-bit EQ, the MLPs of layers 0-19 5-bit, v_proj, the LM head and the linear attention's in_proj_a/b 8-bit, the embedding MLX affine 5-bit, norms BF16. No MTP head, no vision tower. Encoded with the EQ 1.x encoder options (`--enc-opts schur6,refit`): GPTQ sub-scales chosen inside the recurrence and super-scales refitted by least squares. The format is plain EQ: the same engine reads it.

21.6 GB on disk. Bits per weight below are read from the file headers, over the base
model's text parameters, without the MTP head.

## Quality against files of about the same size

| File | Measured with | Bits/weight | KLD mean | KLD median | KLD p99 | Top-1 | Decode M4 Max, tok/s | Decode M1 Max, tok/s |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **this file (EQD6 (preset eqd6-27b))** | Espetero 7949560 | 6.42 | 0.00115 | 0.00024 | 0.0129 | 98.9 % | 21¹ | 11¹ |
| unsloth UD-Q6_K | Espetero 616c21a | 6.43 | 0.00142 | 0.00029 | 0.0160 | 98.8 % | 20 | 10 |
| unsloth UD-Q6_K | dequantized, fp32 | 6.43 | 0.00137 | 0.00028 | 0.0155 | 98.8 % |  |  |
| EXL3 SC 6.0 | omlx-exl3 runtime (bf16); decode from another session | 6.49 | 0.00157 | 0.00049 | 0.0148 | 98.2 % |  | 9.1 |
| EXL3 SC 6.0 | fp32 emulation | 6.49 | 0.00111 | 0.00027 | 0.0109 | 98.8 % |  |  |
| MLX 6-bit | Espetero | 6.50 | 0.00348 | 0.00092 | 0.0347 | 97.8 % |  |  |

* Smaller than UD-Q6_K (by about 0.01 bits per weight).
* Lower KLD than UD-Q6_K at the same size, significantly (the interval excludes zero), also on held-out text that is in neither the calibration nor the test windows: 30 windows of WikiText-2 validation, C4 and Python (KLD 0.00093 against 0.00111) and 12 of C and C++ (0.00107 against 0.00137), against UD-Q6_K dequantized into the fp32 reference. Top-1 agreement 98.87 % against 98.79 %.
* EXL3 at 6.5 bits has 3 % lower KLD in fp32 emulation (0.00111, 0.07 bits per weight more) but 37 % higher in its Metal runtime (omlx-exl3); on the M1 Max this layout decoded at 12.3 tokens/s against EXL3's 9.1 in the rival measurements' session (+35 %).
* ¹ Decode measured on the same layout converted with the prose-and-code calibration and the default encoder (Espetero 616c21a): the codes differ, the bytes read per token do not. Quality rows of this file: Espetero 7949560, the commit that added `--enc-opts`, whose engine code is 0.2.0's.

Paired comparison over the same reference windows (window bootstrap):

* Against UD-Q6_K (both run by Espetero): KLD -19.0 % (95 % interval -28.1 % to -6.7 %; lower in 71 % of the 62 windows)
* Against UD-Q6_K dequantized into the fp32 reference: KLD -16.0 % (95 % interval -25.6 % to -2.9 %; lower in 69 % of the 62 windows)
* Against UD-Q6_K dequantized, on held-out text (WikiText-2 validation, C4, Python): KLD -15.7 % (95 % interval -25.5 % to -5.4 %; 30 windows)
* Against UD-Q6_K dequantized, on held-out C and C++ (llama.cpp's sources): KLD -22.2 % (95 % interval -27.9 % to -15.1 %; 12 windows)

Decode: one sequence, a 128-token context, `bench_native --quick --batch 1` of the same build, the files interleaved in one exclusive session per Mac (M4 Max: Mac Studio, 128 GB; M1 Max: MacBook Pro, 64 GB); wall-clock tokens per second.

## Run it

Needs **Espetero 0.2.0** or later and a Mac with 32 GB of memory or more. EQ files are Espetero's own format: llama.cpp, MLX
and transformers do not load them.

```bash
brew install JarJarFran/espetero/espetero
espetero serve --model Franjarjar/Qwen3.8-27B-EQD6
```

Then open `http://127.0.0.1:8080`, or point any OpenAI, Anthropic or Ollama client at it.

## How it was made

```bash
espetero quantize --model Qwen3.8-27B --calib-only --calib-layers A:B --hessian-max 16384 \
    --calib-data file:calib_text.txt --calib-tokens 452000 --calib-out part-A-B.safetensors   # 8 layer ranges
espetero quantize --model Qwen3.8-27B --calib-only --calib-layers A:B --hessian-max 16384 \
    --calib-data file:wiki.txt --calib-tokens 150000 --calib-out wiki-A-B.safetensors   # 0:32, 32:64; also cc, chat
python3 tools/calib_mix.py v6a 0.5:part-0-8.safetensors,...,part-56-64.safetensors \
    0.1667:wiki-0-32.safetensors,wiki-32-64.safetensors 0.1667:cc-0-32.safetensors,cc-32-64.safetensors \
    0.1667:chat-0-32.safetensors,chat-32-64.safetensors
VECLIB_MAXIMUM_THREADS=1 espetero quantize --model Qwen3.8-27B --calib v6a/part-0-8.safetensors,...,v6a/part-56-64.safetensors --preset eqd6-27b --enc-opts schur6,refit --no-mtp --out Qwen3.8-27B-EQD6
```

* Converter: `espetero quantize` of Espetero 0.2.0 (commit `7949560`), a development commit whose encoder source is the release's.
* Calibration: a weighted mix (`tools/calib_mix.py`) of four captures by Espetero's forward of the BF16 weights: one half 452K tokens of prose and code (wikitext-103, C4 and PG19 train text, mlx-lm's calibration_v5, chat and Python), one sixth each of about 150K tokens of Wikipedia, of C/C++ and of chat. With the prose-and-code capture alone and the default encoder this allocation scores KLD 0.00132; with calibration_v5 alone, 0.00149.
* Rules: preset `eqd6-27b`, `--enc-opts schur6,refit`.

## How it was measured

KL divergence KL(p_BF16 || p_file) per token, against the BF16 model run with fp32 activations, over 123,008 tokens: 62 windows of 2048 tokens (31 of WikiText-2 test, 31 of code), the first 64 of each not scored; top-1 is agreement with the reference's most likely token. Every row marked *Espetero 616c21a* is the distribution Espetero itself computes (fp16 activations included) with the public build 616c21a: `kld_dump` writes the final hidden states, the file's own LM head gives the logits. Rows marked *dequantized* load the GGUF file's tensors into the fp32 reference instead (llama.cpp's `ssm_out` stayed BF16 there, which slightly favours those rows). The paired interval resamples whole windows (10,000 bootstrap samples; both files on the same windows).

## License

A quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), under its Apache License 2.0; this
repository keeps that license.  Espetero itself is MIT-licensed.
