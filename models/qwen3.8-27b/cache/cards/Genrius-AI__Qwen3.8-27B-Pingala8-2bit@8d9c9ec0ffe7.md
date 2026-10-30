---
license: apache-2.0
language:
- en
- zh
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Genrius-AI
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- quantization
- 2-bit
- llama.cpp
- trellis
- qwen3.8
- qwen3_5
- gated-deltanet
- hybrid-attention
- llm
- local-llm
- model-compression
- low-vram
- on-device
- cuda
- nvidia
- qwen
- quantized
- llama-cpp
- text-generation
- conversational
- qwen3.8-27b
model-index:
- name: Qwen3.8-27B-Pingala8-2bit
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: WikiText-2 (test, context 4096, first 40 of 72 windows)
      type: Salesforce/wikitext
      split: test
      config: wikitext-2-raw-v1
    metrics:
    - type: perplexity
      value: 7.6183
      name: Perplexity (lower is better)
      verified: false
    source:
      name: Genrius AI measurement (logs in the code repository)
      url: https://github.com/Genrius-AI/pingala8-llm
---

# Qwen3.8-27B Pingala-8 2-bit

**Like it? Star the code on GitHub: [Genrius-AI/pingala8-llm](https://github.com/Genrius-AI/pingala8-llm)** (it helps others find it) [![GitHub stars](https://img.shields.io/github/stars/Genrius-AI/pingala8-llm?style=social)](https://github.com/Genrius-AI/pingala8-llm)

Try it in your browser: [https://huggingface.co/spaces/harshit-genrius/pingala8-demo](https://huggingface.co/spaces/harshit-genrius/pingala8-demo) (live chat with Qwen3-1.7B-Pingala8-2bit on CPU)

**Qwen3.8-27B in 8.31 GB, 2-bit: WikiText-2 perplexity 7.62 vs 8.21 for llama.cpp IQ2_XXS of the same model, in a file 118.7 MB smaller (lower is better; first 40 of 72 windows of 4096 tokens).**

**Stock llama.cpp, Ollama and LM Studio will refuse this file with a load error.** It needs the Pingala-8 CUDA runtime with the Qwen3.5 loader change (llama.cpp `7ac59a6` + `patches/llama_cuda_vs_7ac59a6.patch` + `patches/q35_loader_count_delta.patch` from the code repository; see How to run). This 27B file has not been tested with the one-click build.

| File (pass part 1 to llama.cpp) | Quant | Size | bits/weight | WikiText-2 PPL |
|---|---|---|---|---|
| `Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf` + the other parts | **Pingala-8, 2-bit** | **8.31 GB** | **2.473** | **7.62** |
| for comparison: IQ2_XXS (llama.cpp, same imatrix) | 2-bit | 8.43 GB | 2.508 | 8.21 |

This is the text model only: the vision tower and the multi-token-prediction (MTP) block of Qwen3.8-27B are not included.

Author: Harshit Goyal, Genrius AI (harshit@genrius.in). Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
Method paper: *Small-State Coset Trellis Coding at IQ2_XXS Size: An Equal-Bytes Study in llama.cpp* (preprint, v2.4),
DOI [10.5281/zenodo.23481757](https://doi.org/10.5281/zenodo.23481757). The Qwen3.8-27B numbers on this page are new measurements and are not in the paper.
Smaller siblings: [Qwen3.5-4B Pingala-8](https://huggingface.co/Genrius-AI/Qwen3.5-4B-Pingala8-2bit), [Qwen3.5-2B](https://huggingface.co/Genrius-AI/Qwen3.5-2B-Pingala8-2bit).
Code and runtime patch: https://github.com/Genrius-AI/pingala8-llm

## What this is

A 2-bit-class quantization of Qwen3.8-27B (hybrid model: 48 Gated DeltaNet layers and 16 gated full-attention layers) at no more bytes than
llama.cpp's IQ2_XXS file of the same model. Every coded linear weight (400 tensors) is stored with Ungerboeck's 8-state set-partitioned trellis
code over 4-weight symbols (the 8 states are Pingala's 8 ganas), behind a random Hadamard rotation and LDLQ rounding, one rate class per tensor.
Layer Hessians come from 512 sequences of 2048 tokens of calibration text, propagated through the bf16 model one layer at a time; the importance
matrix is bartowski's published Qwen3.8-27B imatrix, used for both this file and the IQ2_XXS baseline. The other tensors (token embedding,
output, norms, conv1d, DeltaNet gate parameters) are copied from the stock IQ2_XXS file.

This was a single build with the recipe, the baseline and the release protocol written down before any 27B 2-bit number was measured.
The encoder script was modified after the first tensor was encoded (sha256 now `9ddc197101c2c571...`, pre-registered `cc19642cba4e2a31...`);
under the pre-registration's own rule this is reported here, and the logs in the code repository show the change.

## Files

| file | bytes | sha256 |
|---|---|---|
| `Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf` | 885,051,584 | `953760cd216aadc1206c143b3c7ace97cec3cc17ebd32027d911f33848dd512d` |
| `Qwen3.8-27B-Pingala8-2bit-00002-of-00009.gguf` | 985,500,320 | `46ab1d5b2fc892929a2167c92b02c2e84a87efb4b72bb4c217755bf84e11f36d` |
| `Qwen3.8-27B-Pingala8-2bit-00003-of-00009.gguf` | 977,254,304 | `55ae23b7e86129d8562fe11c8c480ed5ac31ade02dd607f97b4a7499ae7d8905` |
| `Qwen3.8-27B-Pingala8-2bit-00004-of-00009.gguf` | 984,067,584 | `b630a292e076ddc54419c7c1c6b4a87b050dc8373ac3f482aa28e760c2977ac3` |
| `Qwen3.8-27B-Pingala8-2bit-00005-of-00009.gguf` | 976,899,808 | `db74cc0282dc0a809bc628942c5cfa6a403ebbb1f918f77b1dc1b6b023e4d747` |
| `Qwen3.8-27B-Pingala8-2bit-00006-of-00009.gguf` | 974,774,784 | `6be88d22e6bac52943fc4cc70acc3e2331afff7b8198bbe7b24e861f1c0cc128` |
| `Qwen3.8-27B-Pingala8-2bit-00007-of-00009.gguf` | 983,187,520 | `ca431787d38161f9a775749844267b95389520fac4bb5a91f2c8db60b1d29027` |
| `Qwen3.8-27B-Pingala8-2bit-00008-of-00009.gguf` | 983,151,488 | `c112500126fd0afdf03e52cbc7d3ce233c6163a7966b1e309e3ccb35fa1c9992` |
| `Qwen3.8-27B-Pingala8-2bit-00009-of-00009.gguf` | 564,602,208 | `986247b2d28fc29bbb5c77ad729e1ae330bb8e074aac14934a680eb599c55478` |

Total 8,314,489,600 bytes = 2.473 bits per weight over the 26,895,998,464 weights in the file (IQ2_XXS of the same model: 8,433,183,520 bytes, 2.508 bits per weight).
Pass the first part to llama.cpp; it finds the others.

## Results

WikiText-2 test, `llama-perplexity -c 2049 --ppl-stride 4095 --chunks 40 -ngl 99 -lm dio -b 512 -ub 512` (windows of 4096 tokens; the first 40 of the 72 windows, capped by
the time available on one laptop GPU; both files on exactly the same windows). Lower is better.

| format | file bytes | bits/weight | PPL | paired, Pingala-8 - IQ2_XXS |
|---|---|---|---|---|
| IQ2_XXS (llama.cpp `llama-quantize`, same imatrix) | 8,433,183,520 | 2.508 | 8.2060 | |
| **Pingala-8 (this file)** | **8,314,489,600** | **2.473** | **7.6183** | -0.0743 +/- 0.0207 nats per token (95%), better on 40/40 windows |

Ratio of perplexities, Pingala-8 / IQ2_XXS: 0.928. No 8-bit reference was run: a 27B Q8_0 file (28.6 GB) does not fit this laptop's GPU and memory.

Speed, token generation on one RTX 5070 Ti Laptop GPU, batch 1 (`llama-bench -p 0 -n 64 -r 3 -fa 1 -ngl 99 -lm dio`, same runtime binary, one run each,
not rotated rounds): Pingala-8 30.8 tokens/s, IQ2_XXS 36.4 tokens/s (0.847x).

Checks: native file read back tensor by tensor against freshly regenerated data: PASS VERIFY shards 9 tensors in file 2090 expected 2090; mismatches 0 []; split.count 9 vr.ns 108 vr.trl 2 PASS; GPU per-tensor decode check (R51CHECK): SKIP (GPU gates did not pass in time). Peak GPU memory during the perplexity run: 10841 MiB (IQ2_XXS: 9333 MiB).

Memory note: `llama-perplexity` keeps the logits of a whole 4096-token window in host RAM (4096 x 248,320 floats = 4.07 GB for this vocabulary). On this 12 GB laptop that exhausted Windows commit memory, so both runs used `-lm dio -b 512 -ub 512` and a `llama-perplexity` patched to score each batch as soon as it is decoded instead of storing the window (same rows, same softmax, same summation order; the patch is meant not to change the numbers, and we did not separately compare against the unpatched tool). Both files were measured with the same settings and the same tool change.

Zero-shot quiz scores (HellaSwag, MMLU and so on) were not run for this model.

## How to run

Build llama.cpp at commit `7ac59a6e3ad851cd41af00f678effab0598ba9a8` with the released CUDA patch plus the Qwen3.5 loader change from the code repository:

```
git clone https://github.com/Genrius-AI/pingala8-llm
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 7ac59a6e3ad851cd41af00f678effab0598ba9a8
git apply ../pingala8-llm/patches/llama_cuda_vs_7ac59a6.patch
git apply ../pingala8-llm/patches/q35_loader_count_delta.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
cmake --build build -j8 --target llama-cli llama-perplexity llama-bench
```

Then (all layers on the GPU; peak 10841 MiB of GPU memory measured in the perplexity run at context 4096):

```
hf download Genrius-AI/Qwen3.8-27B-Pingala8-2bit --local-dir qwen3.8-27b-pingala8
export GGML_F44=4 GGML_F44_BATCH=2 GGML_VR_NATIVE=1 GGML_VRC_V=5 GGML_VRC_NWP=8 GGML_VR_RL=1 GGML_VR_PRMG=1 GGML_VR_PPC=2
./build/bin/llama-cli -m qwen3.8-27b-pingala8/Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf -ngl 99
./build/bin/llama-perplexity -m qwen3.8-27b-pingala8/Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf -f wikitext2_test_hfjoin.raw -c 2049 --ppl-stride 4095 -ngl 99 -lm dio -b 512 -ub 512 --chunks 40
```

The perplexity command is the measured protocol (expected final value `[40]7.6183`); the code repository rebuilds the text file. The `llama-cli` line is a usage example and was not part of the measured runs.

## Limitations

- CUDA only, measured on one laptop GPU (RTX 5070 Ti Laptop, 12 GB, WSL2); no Metal, Vulkan or CPU path for this file.
- Text only: no vision tower, no MTP block.
- Perplexity on the first 40 of 72 WikiText-2 windows only (time cap), context 4096 only; no zero-shot accuracy, no long-context test.
- The baseline is our own `llama-quantize` IQ2_XXS from the bf16 weights with bartowski's imatrix, not bartowski's published IQ2_XXS file (a custom mix, 8,881,268,448 bytes).
- No IQ2_KT row and no 8-bit reference for this model. With the same recipe the method lost to IQ2_KT on Qwen3-14B (paper, Section 4.3).
- Refusal, bias and hallucination behaviour after quantization were not evaluated. Evaluate the model for your own use.

## Citation

```bibtex
@misc{goyal2026pingala8,
  title        = {Small-State Coset Trellis Coding at {IQ2\_XXS} Size: An Equal-Bytes Study in {llama.cpp}},
  author       = {Goyal, Harshit},
  year         = {2026},
  month        = oct,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23481757},
  url          = {https://doi.org/10.5281/zenodo.23481757},
  note         = {Preprint, version 2.4. Code: https://github.com/Genrius-AI/pingala8-llm}
}
```

## License and notice

Apache-2.0, inherited from Qwen3.8-27B (Qwen team, Alibaba Cloud); see `LICENSE` and `NOTICE`. The weights are a quantized derivative of
Qwen3.8-27B made by Genrius AI. "Pingala-8" and "Genrius AI" are names of Genrius AI; the license grants no trademark rights.
