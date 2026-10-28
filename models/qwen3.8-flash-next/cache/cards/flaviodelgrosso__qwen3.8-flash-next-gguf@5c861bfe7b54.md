---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- text-generation
---

# Qwen3.8 Flash Next — Q4-D6 (GGUF, `qwen4exp`)

Quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(source revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`).

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-Q4-D6.gguf` | 175,161,245,696 | `5fe74be439e4e05367aba6036e3782af420bed8892600ca69e8c555c118cb4db` |

## Recipe

The name states the recipe: **Q4** routed experts (Q4_K gate/up, MXFP4 down)
and **~6-bit dense** weights (6.2 bits per weight on average across every
non-expert tensor, down from 10.0 in a Q8_0-dense Q4). GDN projections are
graded by layer sensitivity (Q5_K/Q6_K in layers 0–23, Q4_K in 24–47);
attention, head, PLE and NextN projections are Q6_K; the shared-expert
gate/up is Q4_K; the hyper-connection mixers are Q8_0 (up) and Q6_K (down);
the router is BF16 (lossless). The embedded NextN layer and the original BF16
n-gram table are included.

Every tensor is quantized once, directly from the BF16 checkpoint. The routed
experts and the retyped dense tensors are importance-weighted with the
unsloth imatrix (SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`). The
precision map and the imatrix SHA-256 are recorded in the GGUF metadata
(`qwen4exp.requant.*`).

Decode reads 4.62 GB of weights per token instead of 6.70 GB for the Q8_0-dense
Q4: on an Apple M5 Max, single-session decode goes from 51.9 to 68.0 tok/s
(+31%) and NextN speculative decode from 87.6 to 96.5 tok/s (+10%), measured
on the earlier requantized build, which has the same tensor types and sizes.
Quality, speed and the full study:
[docs/quantization.md](https://github.com/flaviodelgrosso/razor/blob/main/docs/quantization.md).

Rebuild byte-identically with Razor:

```sh
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --local-dir src/qwen
hf download unsloth/Qwen3.8-Flash-Next-GGUF imatrix_unsloth.gguf_file \
  --revision c8b5954a88c2775c546b92593eda40ea041d3176 --local-dir .
razor convert qwen-q4-d6 --source src/qwen \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --imatrix imatrix_unsloth.gguf_file --out Qwen3.8-Flash-Next-Q4-D6.gguf
```

## Use

[Razor](https://github.com/flaviodelgrosso/razor) on Apple Silicon (resident,
about 69 GiB): `razor download qwen-q4-d6`, then
`razor serve --model models/Qwen3.8-Flash-Next-Q4-D6.gguf`
(`--speculation opportunistic|exact` uses the embedded NextN head).

## License

Qwen Community License 1.0 (see `LICENSE`; commercial-use conditions apply).
