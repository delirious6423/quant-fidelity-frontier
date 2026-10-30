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

# Qwen3.8 Flash Next — Q4-D6 GGUF

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `de4b8e4d` for [Razor](https://github.com/flaviodelgrosso/razor).

**`Qwen3.8-Flash-Next-Q4-D6.gguf`** (175 GB, SHA-256 `5fe74be4…18cb4db`): Q4 routed experts (Q4_K gate/up, MXFP4 down), ~6-bit dense (6.2 bpw vs 10.0 in Q8_0-dense Q4). GDN graded by layer (Q5_K/Q6_K in 0–23, Q4_K in 24–47); attention, head, PLE, NextN Q6_K; BF16 router; embedded NextN head and BF16 n-gram table kept. Quantized once from BF16 with the unsloth imatrix (`a5863123…f9254154`); precision map in `qwen4exp.requant.*`.

Decode reads 4.62 GB/token instead of 6.70 GB: on an M5 Max, 68.0 tok/s (+31%) and 96.5 tok/s with NextN speculation (+10%), measured on an earlier build with identical tensor types. Details: [docs/quantization.md](https://github.com/flaviodelgrosso/razor/blob/main/docs/quantization.md).

## Use

```sh
razor download qwen-q4-d6
razor serve --model ~/.razor/models/Qwen3.8-Flash-Next-Q4-D6.gguf --speculation opportunistic
```

Resident, ~69 GiB. Rebuild byte-identically with `razor convert qwen-q4-d6` (see the Razor docs).

Qwen Community License 1.0 (commercial-use conditions apply; see `LICENSE`).
