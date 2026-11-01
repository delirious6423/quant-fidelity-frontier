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
- qwen3
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
      name: WikiText-2 (test, context 4096, first 20 of 72 windows)
      type: Salesforce/wikitext
      split: test
      config: wikitext-2-raw-v1
    metrics:
    - type: perplexity
      value: 6.8894
      name: Perplexity (lower is better)
      verified: false
    source:
      name: Genrius AI measurement (logs in the code repository)
      url: https://github.com/Genrius-AI/pingala8-llm
---

# Qwen3.8-27B Pingala-8 2-bit

**Like it? Star the code on GitHub: [Genrius-AI/pingala8-llm](https://github.com/Genrius-AI/pingala8-llm)** (it helps others find it) [![GitHub stars](https://img.shields.io/github/stars/Genrius-AI/pingala8-llm?style=social)](https://github.com/Genrius-AI/pingala8-llm)

Try it in your browser: [https://huggingface.co/spaces/harshit-genrius/pingala8-demo](https://huggingface.co/spaces/harshit-genrius/pingala8-demo) (live chat with Qwen3-1.7B-Pingala8-2bit on CPU)

**Qwen3.8-27B in 8.31 GB, 2-bit: WikiText-2 perplexity 6.89 vs 7.88 for llama.cpp IQ2_XXS of the same model, in a file 118.9 MB smaller, and 37.7 vs 36.4 tokens/s (lower perplexity is better; first 20 of 72 windows of 4096 tokens). The best published 2-bit file we measured, DASH-Q IQ2_XS, is still better: 6.54 at 8.55 GB.**

Version 2 (2026-11-01). Same trellis coder as version 1, new bit allocation across tensors: perplexity 7.34 -> 6.89 at the same file size, and generation 30.8 -> 37.7 tokens/s.

See how it compares with every other 2-bit file of this model we measured (DASH-Q, unsloth, llama.cpp), window by window: [2-bit GGUF leaderboard](https://huggingface.co/spaces/Genrius-AI/2bit-gguf-leaderboard).

<a href="https://huggingface.co/Genrius-AI/Qwen3.8-27B-Pingala8-2bit/resolve/main/media/demo27b.mp4"><picture><source media="(max-width: 600px)" srcset="https://huggingface.co/Genrius-AI/Qwen3.8-27B-Pingala8-2bit/resolve/main/media/demo27b-phone.gif"><img src="https://huggingface.co/Genrius-AI/Qwen3.8-27B-Pingala8-2bit/resolve/main/media/demo27b.gif" alt="Qwen3.8-27B Pingala-8 2-bit generating a 5-bullet answer on one 12 GB laptop GPU (RTX 5070 Ti Laptop), replayed at its real speed"></picture></a>

*Real run, real speed. Text and timing captured from the run; terminal view re-rendered.* Chat speed in this run: 31.63 tokens/s (runtime eval, 124 tokens, thinking off; the 37.7 tokens/s above is llama-bench tg64). [Watch the 1080p MP4](https://huggingface.co/Genrius-AI/Qwen3.8-27B-Pingala8-2bit/resolve/main/media/demo27b.mp4).

**Stock llama.cpp, Ollama and LM Studio will refuse this file with a load error.** It needs the Pingala-8 CUDA runtime with the Qwen3.5 loader change (llama.cpp `7ac59a6` + `patches/llama_cuda_vs_7ac59a6.patch` + `patches/q35_loader_count_delta.patch` from the code repository; see How to run). This 27B file has not been tested with the one-click build.

| File (pass part 1 to llama.cpp) | Quant | Size | bits/weight | WikiText-2 PPL |
|---|---|---|---|---|
| `Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf` + the other parts | **Pingala-8, 2-bit** | **8.31 GB** | **2.473** | **6.89** |
| for comparison: IQ2_XXS (llama.cpp, same imatrix) | 2-bit | 8.43 GB | 2.508 | 7.88 |

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
| `Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf` | 885,051,584 | `431c2049b576fd4d6e96216f47a14b775c889556536a5119f497a5776b7dfd13` |
| `Qwen3.8-27B-Pingala8-2bit-00002-of-00009.gguf` | 977,459,744 | `39aca111af53d82acf975e8be6ef5474c9765d30e2651196f0537ed4b98b3520` |
| `Qwen3.8-27B-Pingala8-2bit-00003-of-00009.gguf` | 972,276,384 | `c1f74ac5428cb9e75889013d8da67db71362ab4f4e9a76e70e35f96083a9c1e7` |
| `Qwen3.8-27B-Pingala8-2bit-00004-of-00009.gguf` | 971,548,640 | `5047e0c90be29bdca0a3b8380583b98236e040380d4a7d736fa14f2976a73b1e` |
| `Qwen3.8-27B-Pingala8-2bit-00005-of-00009.gguf` | 983,982,528 | `a112a4d75dc9feadd039ee501ac1f11edf76c8ccf890f19d0f2ea251671d7461` |
| `Qwen3.8-27B-Pingala8-2bit-00006-of-00009.gguf` | 980,700,800 | `039786f55ed4bcedfdfbe1609fba266fb08e5edf0b7012b9fd3caf65303c88ce` |
| `Qwen3.8-27B-Pingala8-2bit-00007-of-00009.gguf` | 982,269,824 | `6d00bcdf6a7c9dbf538467a9a9c6b11ab0017bc42c34ac9ce260ec3b3cc3626f` |
| `Qwen3.8-27B-Pingala8-2bit-00008-of-00009.gguf` | 965,849,632 | `185cf32eb6349569deb5def1cb3dbb941aed0c68218cec9fb8d6d3c770ea64ae` |
| `Qwen3.8-27B-Pingala8-2bit-00009-of-00009.gguf` | 595,102,464 | `6523ef10ab9555de78cadb6bb83ebe9fd16d2ff62cee1474b532bc1ce896cbf1` |

Total 8,314,241,600 bytes = 2.473 bits per weight over the 26,895,998,464 weights in the file (IQ2_XXS of the same model: 8,433,183,520 bytes, 2.508 bits per weight).
Pass the first part to llama.cpp; it finds the others.

## How version 2 allocates bits

Version 1 gave every tensor the bit rate of its slot in the stock IQ2_XXS layout, which stores the Gated DeltaNet input projections at 4.5 bits per weight.
Version 2 keeps the same coder and total size but picks each tensor's rate class from a role x layer-band bits-per-weight profile: we measured where the published
DASH-Q IQ2_XS file spends its bits (attn_qkv about 2.2, attn_output and attn_k about 3, ssm_out 2.6-3.2, the last 8 layers' FFN about 3.4) and coded every tensor at
the nearest Pingala-8 rate class, at the version 1 byte budget. On Qwen3.5-0.8B the same change gave perplexity 27.55 -> 23.72 (70/70 windows better).

## Results

WikiText-2 test, `llama-perplexity -c 2049 --ppl-stride 4095 --chunks 20 -ngl 99 -lm dio -b 512 -ub 512` (windows of 4096 tokens; the first 20 of the 72 windows,
capped by the time available on one laptop GPU; every file on exactly the same windows). Lower is better. Paired = mean per-token log-likelihood difference, this file minus the row, 95% interval.

| file | bytes | PPL (20 windows) | paired vs this file |
|---|---|---|---|
| DASH-Q IQ2_XS ([jkim96/Qwen3.8-27B-DASHQ-Q2-GGUF](https://huggingface.co/jkim96/Qwen3.8-27B-DASHQ-Q2-GGUF)) | 8,551,168,768 | **6.5431** | DASH-Q better: +0.0516 +/- 0.0117 nats per token, better on 20/20 windows |
| **Pingala-8 version 2 (this file)** | **8,314,241,600** | **6.8894** | |
| unsloth UD-IQ2_S ([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)) | 8,371,970,048 | 7.3447 | -0.0640 +/- 0.0753 (this file better on 9/20 windows: not significant) |
| Pingala-8 version 1 (flat bit layout, 2026-10-30) | 8,314,489,600 | 7.3449 | -0.0640 +/- 0.0075, this file better on 20/20 windows |
| IQ2_XXS (llama.cpp `llama-quantize`, same imatrix) | 8,433,183,520 | 7.8753 | -0.1337 +/- 0.0305, this file better on 20/20 windows |

At this size this file beats plain IQ2_XXS and our version 1 on every window, is ahead of unsloth's dynamic mix on average (not significant over 20 windows), and loses to DASH-Q
(5.3% higher perplexity; DASH-Q's file is 237 MB larger). DASH-Q and unsloth run in stock llama.cpp; this file does not.
The published files were run on llama.cpp `889f2ec` with their IQ1_S/IQ2_S/IQ3_S tensors on the CPU, because those CUDA kernels fail llama.cpp's own `test-backend-ops` check on our machine; our IQ2_XXS file gives 8.2121 there vs 8.2060 on the runtime above (40 windows), so the two builds agree. No 8-bit reference was run: a 27B Q8_0 file (28.6 GB) does not fit this laptop's GPU and memory.

Speed, token generation on one RTX 5070 Ti Laptop GPU, batch 1 (`llama-bench -p 0 -n 64 -r 3 -fa 1 -ngl 99 -lm dio`, same runtime binary, one run each,
not rotated rounds): Pingala-8 version 2 37.7 tokens/s (2026-11-01); IQ2_XXS 36.4 and Pingala-8 version 1 30.8 tokens/s (2026-10-30, same binary). Version 2 is faster because the Gated DeltaNet input projections (attn_qkv) are stored at about 2.1 instead of 4 bits per weight.

Checks: native file read back tensor by tensor against freshly regenerated data: PASS VERIFY shards 9 tensors in file 2072 expected 2072; mismatches 0 []; split.count 9 vr.ns 108 vr.trl 2 PASS; GPU per-tensor decode check (R51CHECK): SKIP (GPU gates did not pass in time). Peak GPU memory during the perplexity run: 10841 MiB (IQ2_XXS: 9333 MiB).

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
./build/bin/llama-perplexity -m qwen3.8-27b-pingala8/Qwen3.8-27B-Pingala8-2bit-00001-of-00009.gguf -f wikitext2_test_hfjoin.raw -c 2049 --ppl-stride 4095 -ngl 99 -lm dio -b 512 -ub 512 --chunks 20
```

The perplexity command is the measured protocol (expected value at 20 windows `[20]6.8894`); the code repository rebuilds the text file. The `llama-cli` line is a usage example and was not part of the measured runs.

## Limitations

- CUDA only, measured on one laptop GPU (RTX 5070 Ti Laptop, 12 GB, WSL2); no Metal, Vulkan or CPU path for this file.
- Text only: no vision tower, no MTP block.
- Perplexity on the first 20 of 72 WikiText-2 windows only (time cap), context 4096 only; no zero-shot accuracy, no long-context test.
- The baseline is our own `llama-quantize` IQ2_XXS from the bf16 weights with bartowski's imatrix, not bartowski's published IQ2_XXS file (a custom mix, 8,881,268,448 bytes).
- Loses to the published DASH-Q IQ2_XS file (6.54 vs 6.89, 20 windows). No IQ2_KT row and no 8-bit reference for this model. With the same recipe the method lost to IQ2_KT on Qwen3-14B (paper, Section 4.3).
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

## FAQ

**Can it run on a 12 GB GPU?**
Yes. The file is 8.31 GB and runs with every layer on one 12 GB RTX 5070 Ti Laptop GPU (WSL2, `-ngl 99`). Our [VRAM calculator](https://huggingface.co/spaces/Genrius-AI/llm-vram-calculator) shows 10.24 GB in total at context 4096 on that GPU. It has been run on that one GPU only. Guide: [Can you run Qwen3.8-27B on a 12 GB GPU?](https://genrius-ai.github.io/pingala8-llm/run-qwen3-8-27b-on-12gb-gpu.html)

**Does it work in Ollama or LM Studio?**
No. It needs the patched Pingala-8 CUDA runtime (see How to run). Stock llama.cpp, Ollama and LM Studio refuse it with a load error, and it has not been tested with the one-click build.

**How does it compare with DASH-Q and unsloth?**
WikiText-2 perplexity, first 20 windows at context 4096, lower is better: DASH-Q IQ2_XS (8.55 GB) 6.54, better than ours on 20/20 windows; this file (8.31 GB) 6.89; unsloth UD-IQ2_S (8.37 GB) 7.34, where ours is lower but not significantly (better on 9/20 windows); llama.cpp IQ2_XXS (8.43 GB) 7.88. All rows: [2-bit GGUF leaderboard](https://huggingface.co/spaces/Genrius-AI/2bit-gguf-leaderboard).

**What speed?**
37.7 tokens/s token generation at batch 1, all layers on the RTX 5070 Ti Laptop GPU (12 GB).

**Is it text-only?**
Yes. The vision tower and the multi-token-prediction (MTP) block are not included.
