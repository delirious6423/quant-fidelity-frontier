---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - moe
  - pruning
  - quantized
  - code
  - experimental
---

# Qwen3.8 WeeAIResearch Flash Next 65B-A6.6B MAX

An experimental, pruned, mixed-precision GGUF derived from [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) via the [lmstudio-community Q4_K_M GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF). It was not fine-tuned. The name counts non-lookup parameters; this is not a security-certified password manager.

**Requires a patched llama.cpp runtime. Stock llama.cpp cannot load this model's four-tier expert layout.** Build instructions and patches are in [runtime/README.md](runtime/README.md).

## Model

- File: `Qwen3.8-WeeAIResearch-Flash-Next-65B-A6.6B-MAX.gguf`, 63,896,212,704 bytes (59.51 GiB).
- Parameters: 65.19B logical non-lookup weights, excluding quantization padding, plus 51.20B lookup values (116.39B combined). The 6.64B active non-lookup figure is a shape-based estimate, not a full execution audit; lookup rows and masked dummy work are excluded.
- Experts: 48 layers, 10 routed experts per layer per token plus shared experts. The first three layers preserve all 464 source experts. The other 45 retain 146–321 experts per layer; 10,009 expert slots are removed.
- Precision in layers 4–48: the top 64 used experts in each layer use Q4_K gate/up and Q4_0 down; remaining used experts are divided across Q3_K, Q2_K and Q1_0. Q3/Q2 down matrices have zero-padding for quantization alignment. The first three layers retain original source precision, including Q8 down weights.
- Selection: used means called at least once in a single greedy run on `we need a super secure effective and safe c# password manager in a cli app`. The trace includes 28 prompt tokens and the first 256 generated tokens; the response was cut off at the token limit. This is not a complete app or a general-purpose calibration set.

## Evaluation and limitations

All generated tensor names, types, shapes and payload hashes were checked. All 81 tensors in the first three layers match the source model. One CPU smoke test loaded the model and generated a 96-token prefix on the calibration prompt, reaching the test limit; it did not produce a complete application or assess password-manager security.

On an Apple M5 Pro with 24 GiB unified memory, the model used approximately 12 GiB process RSS in a short CPU comparison with disk-backed paging. The regular and experimental optimized CPU runtimes produced the same 96-token greedy output, at approximately 2.11 and 2.54 tokens/s respectively in single runs. These measurements do not establish sustained throughput or broad correctness. A one-layer Metal test generated a different response; its compatibility and quality have not been established.

Pruning and aggressive Q1/Q2 quantization may cause severe regressions, especially outside the single traced prompt. The trace and limited smoke tests do not establish general chat or coding quality, nor the safety of generated security software. Read [experiment-report.md](experiment-report.md) and the [selection plan](selection-plan.json) for details.

## License

The model is distributed under the [Qwen Community License 1.0](LICENSE). The llama.cpp patches are distributed under the [llama.cpp MIT license](runtime/LLAMA_CPP_LICENSE). Check the source model's terms before use.
