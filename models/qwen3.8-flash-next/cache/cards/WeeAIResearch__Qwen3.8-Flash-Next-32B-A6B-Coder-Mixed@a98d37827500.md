---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - moe
  - pruning
  - code
  - experimental
---

# Qwen3.8 Flash Next 32B-A6B Coder Mixed Prune2

An experimental, C#-calibrated pruning of the text GGUF from [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It was derived from [lmstudio-community's Q4_K_M GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-Flash-Next-GGUF); it was not fine-tuned on coding data.

**Compatibility: this GGUF requires a custom llama.cpp runtime. Stock llama.cpp and applications using an unmodified runtime cannot load its variable-size mixed-precision expert banks.** The tested patch is in [`runtime/variable-runtime.patch`](runtime/variable-runtime.patch), against llama.cpp commit `18443257a30c884d5332abb8e7dc43c7ffe42fda` (tag `b10662`). See [`runtime/README.md`](runtime/README.md).

## Model

- **File:** `Qwen3.8-Flash-Next-32B-A6B-Prune2-Mixed.gguf`, 51,869,664,064 bytes (48.31 GiB).
- **Parameters:** about 32.21B non-lookup values excluding Q2 padding, plus a separate 51.20B-value n-gram lookup table. The GGUF stores approximately 84.03B tensor values including padding.
- **Active parameters:** A6B is an approximation based on the original model's stated 6B active parameters. Ten routed experts per layer remain active; the exact active count of this derivative has not been independently recalculated.
- **Experts:** 48 layers, each retaining 85–177 routed experts across three variable-size banks, plus the unchanged shared expert. Ten routed experts are selected per token per layer.
- **Precision:** the first bank retains the source precision (Q4_K gate/up and Q5_0 or Q8_0 down); the next is Q2_K with padded down projections; the last is Q1_0. The lookup table and all other non-expert tensors were preserved.
- **Pruning:** removed 3,659 layer/expert slots selected on strictly fewer than 2% of 355 generated tokens across five short C# tasks. The excluded slots accounted for 7.71% of recorded selections. Prompt routing and shared experts were not included in that analysis.

The label “Coder” describes the C#-based routing calibration and is not a claim of code-specialist training or broad coding quality.

## Tests and limitations

All 1,512 tensor names, types, shapes and payload hashes were verified during construction; retained rows were copied without further requantization. On CPU, the model passed 26/26 checks over three short C# methods. Two prompts were newly chosen after pruning and their answers matched the parent model; the third was a calibration prompt. These are smoke tests, not an application-scale benchmark.

On an Apple M5 Pro, one transformer layer was also offloaded to Metal using the custom runtime. A 79-token C# answer matched the CPU output. Individual Q1 matrix operations were not separately attributed to a device in that test. Support for other hardware and standard inference applications has not been established.

The model file includes a 32.78 GiB lookup table intended to remain disk-backed. The remaining file data is approximately 15.52 GiB, before runtime memory and lookup pages; it cannot all fit in a 15 GiB RAM budget. In short CPU tests, peak process RSS was approximately 10 GiB with paging; this is not a full-residency measurement.

The model may regress outside these small samples. No evaluation here establishes security of a password-storage application or suitability for production use.

## License and provenance

Distributed under [Qwen Community License 1.0](LICENSE). The llama.cpp patch is based on llama.cpp and distributed under its [MIT license](runtime/LLAMA_CPP_LICENSE). The source model and quantized GGUF are linked above.
