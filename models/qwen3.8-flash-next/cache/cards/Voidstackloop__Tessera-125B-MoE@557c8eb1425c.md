---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags:
- gguf
- moe
- quantized
- experimental
---
# Tessera-125B-MoE, 448 experts (EXPERIMENTAL)

Quantized, expert-pruned GGUF of Qwen/Qwen3.8-Flash-Next (architecture qwen4exp: 48 layers, 512 routed experts per layer, 10 active).

* **Experts:** 448 of 512 kept per layer, chosen by routed share on English, code, math and reasoning calibration text. The pruning is a byte slice
  of the finished quantized tensors (kept experts are bit-identical to the 512-expert file, no requantization); router rows were removed the same way.
  `expert_count` = 448, `n_expert_used` = 10.
* **Quantization** (`llama-quantize --pure` with a domain importance matrix): gate/up experts IQ2_S, down experts IQ4_NL, body Q6_K,
  per-layer n-gram table Q4_0 (loaded lazily with `-lzm on`), shared experts / hc / ple Q8_0.
* `mtp-Qwen3.8-Flash-Next-balanced.gguf` is the multi-token-prediction draft head (not pruned), for `--spec-type draft-mtp`.
* `SHA256SUMS` lists the checksums of both GGUF files.

**Status at upload time: quality not measured yet.** Comparisons against the unsloth UD-IQ4_XS quant are being run; do not read anything into
this file's quality until they are published. Needs a llama.cpp build with qwen4exp support (commit 7894e8b or newer).
