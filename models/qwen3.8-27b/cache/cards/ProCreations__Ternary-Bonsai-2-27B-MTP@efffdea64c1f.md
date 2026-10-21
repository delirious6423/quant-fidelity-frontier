---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
- Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3_5
- mtp
- speculative-decoding
- bonsai
- ternary
- blackwell
datasets:
- HuggingFaceH4/ultrachat_200k
---
# Bonsai 2 27B MTP

Updated September 18, 2026 with the best verified continued-training checkpoint: **r3-mtp**. The original Ternary Bonsai 2 27B PQ2_0 target is unchanged. The head is available as BF16 master weights and **integer Q8_0** inference weights, not FP8.

## Continued-training results

**The +1.26% below is the extra gain over our previous trained MTP head, not the total gain from enabling MTP.** For an independent MTP-on versus MTP-off comparison, see the community reproduction below.

Matched testing against the previous release measured **167.14 → 169.25 decode tokens/sec (+1.26%)** on eight unseen prompts with two repeats. This is a small measured gain, not a large speedup. Both arms used the same runtime and maximum 2 draft tokens.

| Suite | Previous tok/s | Updated tok/s | Previous acceptance | Updated acceptance |
|---|---:|---:|---:|---:|
| Eight unseen prompts, two repeats | 167.14 | 169.25 | 57.41% | 58.58% |
| Six prior release prompts, two repeats | 172.98 | 175.11 | 61.14% | 62.41% |

Acceptance is accepted draft tokens divided by proposed draft tokens. Throughput is total generated tokens divided by total decode time, including reasoning tokens. The [complete comparison](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/blob/main/experiments/2026-09-18-continuation/current/reports/final-comparison.json) links to the preserved raw requests, outputs and counters in the same reports directory.

All tests used one RTX PRO 6000 Blackwell 96GB, one slot, context 32768, 1536 generated tokens per request, medium reasoning without a thinking-token cap, temperature 1, top-p .95, top-k 20 and min-p 0. These are short-context finite-prefix speed measurements, not broad capability scores or completed browser-task timings. Runs were sequential rather than randomized, and small gains may vary. Neither two full passes over all 8,287 training examples nor the subsequent full integer Q8 QAT pass beat the earlier winners. The three-case greedy output identity smoke passed. Previous-release objective and vision checks are historical and were not rerun on the updated weights.

## Independent community reproduction

[zhaokeqi reported a reproduction on an RTX 4080 SUPER 16GB](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/discussions/2), using the current combined GGUF and the supplied patched source under WSL2/CUDA 13.3. Across 11 measured prompt pairs, the median of the per-prompt MTP/no-MTP throughput ratios was **1.338x (+33.8%)**, with **803 / 1180 = 68.1% aggregate draft acceptance**. These two statistics were recalculated from the author's [published raw JSON at its pinned revision](https://huggingface.co/datasets/zhaokeqi/bonsai2-27b-mtp-repro/tree/bbce2093a1a3b0732fdbeb8ba3a389684fce050c).

This is a community measurement, not an additional controlled benchmark by ProCreations. It used greedy decoding, maximum 2 draft tokens, 128-token raw completions and Q4_0 KV cache with a configured 262,144-token context. One prompt that immediately ended in both modes was excluded, and there were no repeated trials. Results depend on prompts, context, sampling and hardware. A configured large context is not a long-context quality test. The shipped Windows launcher uses a smaller default context to reduce memory requirements.

## Download and run

Use the supplied patched native runtime. Native Windows x64/CUDA 13.3 and Linux x86-64/CUDA 13.3 binaries are available, along with an experimental Windows AMD ROCm build described below. The Linux binary targets SM120 Blackwell; the Windows package contains SM75, SM80, SM86, SM89, SM90 and SM120a kernels (RTX 20/30/40/50 series among the build targets). Other platforms should build the included exact source. The head is not a standalone chat model or a standard Transformers AutoModel checkpoint.

### Windows NVIDIA CUDA (native, no WSL)

1. Download [the Windows CUDA 13.3 runtime ZIP](runtime/llama-bonsai-mtp-windows-cuda13.3-x64.zip) and extract it to a folder.
2. Download [Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf](Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf) and put it beside the extracted `Start-MTP.cmd`.
3. Double-click `Start-MTP.cmd`, then open `http://127.0.0.1:8080`.

The ZIP includes the executables, required CUDA and Microsoft runtime DLLs, launchers and license notices. You need Windows x64 and a compatible NVIDIA driver; you do not need Python, a compiler, WSL or the CUDA toolkit for the prebuilt package. See [Windows instructions and validation](runtime/WINDOWS.md) for requirements, checksums and troubleshooting.

To use an existing model file or limit GPU memory, run this inside the extracted folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\serve-windows.ps1 -Model "D:\Models\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf" -Context 4096 -GpuLayers 32
```

Defaults: automatic GPU layer fitting, context 4096, Q8_0 KV cache, one slot and maximum 2 draft tokens. Smaller GPUs may need CPU offload and will be slower. Add `-NoMtp` for a comparison with MTP disabled using the same GGUF. `-Port`, `-CacheType`, `-Threads` and `-Mmproj` are also supported. No vision projector is bundled.

If downloading the entire repository instead, extract the ZIP with `Expand-Archive .\runtime\llama-bonsai-mtp-windows-cuda13.3-x64.zip .\runtime\windows` and run the root `Start-MTP.cmd`.

### Windows AMD ROCm (RX 9070 XT, experimental)

An [AMD ROCm 7.14.1 runtime ZIP](runtime/llama-bonsai-mtp-windows-rocm7.14.1-gfx1201-x64.zip) is available for the RX 9070 XT (`gfx1201`). It includes the same Bonsai MTP embedding fix, matching DLLs, kernel files and launchers. Extract the complete ZIP into a new directory, place the combined GGUF beside `Start-MTP.cmd`, and follow the [AMD Windows instructions](runtime/WINDOWS-ROCM.md). Existing router users must point their router at the new `bin/llama-server.exe` and retain its matching DLLs.

**AMD hardware inference and throughput are not yet verified.** The binaries were built on Windows; the validation report covers DLL loading, embedded GPU target inspection, and CPU MTP smoke tests only. Those checks do not prove ROCm kernel correctness or speed. The approximately 51 tokens/s reported in [discussion 3](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/discussions/3#6aae11d6d949117f94570f60) is the commenter's result with MTP disabled in their previous runtime. No RX 9070 XT MTP speedup is claimed. See the [build and validation report](reports/windows-rocm-runtime-20260919.json).

### Linux

```bash
hf download ProCreations/Ternary-Bonsai-2-27B-MTP --local-dir bonsai-mtp
cd bonsai-mtp
sha256sum -c SHA256SUMS
tar -xzf runtime/llama-bonsai-mtp-linux-cuda13.3-sm120.tar.gz -C runtime
bash serve.sh
```

`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` includes the original base plus the Q8 head. All 851 original base tensor payloads were verified unchanged during export. `model_mtp.safetensors` and `mtp_config.json` contain the BF16 head and configuration. The runtime archive is now the same combined MTP/DFlash-capable patched runtime used in the matched tests. For vision, use `python download-vision.py` and set `MMPROJ` to that path.

The API binds privately to `http://127.0.0.1:8080/v1`. `PORT`, `CONTEXT`, `DRAFT_TOKENS` and `LLAMA_BIN_DIR` override launch settings. Medium reasoning and no thinking-token cutoff remain defaults. Build the exact native source with `bash runtime/build-runtime.sh`, then set `LLAMA_BIN_DIR` as printed by that script. Q8 quantization applies to head matrices; normalization weights retain the runtime's required floating-point representation.

## Reproducibility and history

The [shared experiment archive](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/tree/main/experiments/2026-09-18-continuation/current) preserves source/configuration, all generated text and splits, feature metadata, selection history, full-pass coverage and raw benchmark reports for both heads. The [reproduction guide](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2/blob/main/experiments/2026-09-18-continuation/current/REPRODUCE.md) explains reconstruction and the expired experiment-specific paths/deadlines. Frozen feature arrays and final optimizer/RNG states are excluded; further fine-tuning from the supplied weights starts a new optimizer. Earlier-round source is preserved alongside the current archive.

The [previous release at its immutable revision](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/tree/6f66852e436806f0db2b6ff5351bf30a0eaa95f9) retains the old weights, original donor/no-head comparisons and historical quality checks. Those earlier absolute timings must not be mixed with this matched continuation comparison. Existing historical reports remain available; their filenames do not describe the newly updated weights.

This is an independent ProCreations experiment, not an official Prism ML, Qwen or DFlash release. Original model licenses are Apache 2.0; included runtime and SpecForge sources retain their own licenses and notices. Original target: `prism-ml/Ternary-Bonsai-2-27B-gguf` revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`. The base remains unchanged; speculative tokens are verified by it. Floating-point batching can still change text near close token decisions. See the manifest and original pinned release for donor provenance.
