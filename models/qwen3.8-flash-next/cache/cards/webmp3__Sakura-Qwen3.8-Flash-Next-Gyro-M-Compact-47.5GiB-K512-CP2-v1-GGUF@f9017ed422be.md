---
base_model:
- Qwen/Qwen3.8-Flash-Next
- agentionai/Qwen3.8-Flash-Next-Gyro-GGUF
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- qwen4exp
- qwen3.8-flash-next
- gyro
- experimental
---

# Sakura Qwen3.8 Flash-Next Gyro-M Compact 47.5 GiB — K512 CP2 v1

An experimental compact derivative of **AgentionAI Gyro-M**, retaining **all 512 routed experts** and their original Gyro encoding. Only the large N-Gram/PLE lookup table is compressed. The matching Windows Vulkan runtime and a relative starter are included.

## Size and format

| Component | Original Gyro-M | This package |
|---|---|---|
| Expert weights and other tensors | Original Gyro-M | Verified byte-identical (1,223 non-table tensor payloads) |
| N-Gram/PLE lookup table | Q8_0 | Experimental CP2 |
| Routed experts | 512 | 512 |
| Main-model file | 91,965,774,272 bytes / 85.65 GiB | **51,005,578,112 bytes / 47.50 GiB** |

The main model is about **44.5% smaller** by file size. There is one GGUF file; no separate MTP draft is required or included. GiB means 2^30 bytes: the main file is approximately **51.01 GB** in decimal download units. Runtime and documentation add a small separate download.

`TQ2_0` retains the source repository's Gyro-M filename size-class convention. The expert tensors retain AgentionAI's custom trellis encoding; this is **not a standard Q3 or standard TQ2_0 whole-model conversion**. `CP2` identifies the additional table conversion.

## Download and start

Keep the GGUF, `START.cmd` and the complete `runtime` folder together. Read LICENSE before use. On Windows, double-click **START.cmd**, wait for the model to load, then open **http://127.0.0.1:18098**. The same local server provides an OpenAI-compatible endpoint at `/v1/chat/completions`, model name `sakura-gyro-m-cp2`. Stop it with Ctrl+C.

The starter resolves paths relative to its own folder, so the entire folder can be moved together. It uses Vulkan, context 8192, one slot, Q8 KV cache, lazy N-Gram reads, and **thinking enabled**. MTP and the optional CP2 GPU row cache are disabled. Default sampling is temperature 1.0, top_p 0.95, top_k 20, min_p 0.05; the deterministic smoke checks below explicitly used temperature 0/top_p 1 and are not a comparison of sampling presets.

## Hardware compatibility

| Hardware/runtime | Status |
|---|---|
| AMD Radeon 8060S / Strix Halo, 64 GB UMA, Windows x64, included Vulkan runtime | Completed model loaded; generation, thinking/code case and relative starter/API/Web UI checks passed |
| Fast local NVMe storage | Used for lazy table reads; keep the model on fast local storage during inference |
| Stock llama.cpp, Maya, LM Studio, Ollama and unrelated forks | Not qualified for this custom CP2 format |
| MTP, vision/mmproj, other GPUs/platforms | Not tested for this derivative |

**47.5 GiB is the model's file size, not a VRAM requirement.** GPU-resident weights, system/shared memory, lazy table I/O, KV cache and buffers have different costs. The tested starter completed on this 64 GB UMA system; no peak-VRAM or universal 48 GB fit claim is made. Increasing context or running other GPU applications changes available memory.

The filename keeps the HF-recognized `TQ2_0` label for the hardware widget. A widget grouping does not certify custom-format compatibility or fit; use the included runtime and the measured configuration above. The widget's hardware selection is controlled by each visitor.

## Quality and limitations

The **completed GGUF**, without a pilot overlay, was checked against the unchanged Gyro-M source using the same isolated runtime, corpora and PPL settings. Lower PPL is better.

| Test | Context / chunks | Original | CP2 | PPL change |
|---|---:|---:|---:|---:|
| English, completed checkpoint | 512 / 8 | 1.8548 | 1.8598 | +0.270% |
| Code, completed checkpoint | 512 / 40 | 5.0269 | 5.0476 | +0.412% |
| Initial Code8 **pilot overlay** | 512 / 8 | 4.4998 | 4.5526 | **+1.173%, preliminary gate failed** |

The original failed Code8 pilot is preserved. The longer, separately planned Code40 expansion used the unchanged encoder and the same <1% criterion; it includes Code8 and is **not independent replication**. Completed-checkpoint English8 and Code40 reproduce the corresponding overlay PPL to printed precision. Corpus hashes and test scope are recorded in [results.json](results.json).

In a matched four-case **thinking-enabled** smoke, both original and CP2 passed all four small cases with nonempty reasoning and normal stops: JSON arithmetic, a Python slice-assignment result, a closed-form triangular function, and integer multiplication. The generated function was checked with a bounded arithmetic AST evaluator, without executing generated code. The relative starter independently answered the Python case correctly with thinking enabled by default.

There are also **documented failures**: both models returned the same incorrect answers for the Python case with thinking disabled (26 inline and 24 fenced, expected 27). Both returned 27 with thinking enabled. These observations do not isolate model behavior from the shared runtime/template. They are not evidence that all smoke tests passed. Thinking is therefore enabled in the supplied starter.

These are limited integrity/PPL/function checks, not a broad coding benchmark, long-context evaluation, proof of general equivalence, or controlled speed comparison. No MTP, vision, peak-memory or performance claims are imported from the source model card.

## Integrity and runtime

All 1,223 non-table tensor payloads were checked against the original source, including expert tensors. Tokenizer and other preserved metadata were checked. Fresh sampled-row re-encoding and CPU/HIP reference checks passed. The complete final GGUF was SHA256-verified after relocation into this folder.

Main GGUF SHA256: `31fa8498ca3084a173ff3d8bad9234f4084a2a525b8e96798b418fa62f5a930d`.

See [SHA256SUMS](SHA256SUMS) for every shipped artifact and [runtime/PROVENANCE.json](runtime/PROVENANCE.json) for binary hashes and runtime provenance. The runtime is based on [AgentionAI llama.cpp](https://github.com/agentionai/llama.cpp), commit `2353a299b636e7773c98ac496409595a733880f0`, with local CP2 support. All comparison and starter tests use byte-identical runtime binaries.

## Provenance and license

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Direct quantized source: [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), revision `05758344bde211b5b2fd18c1ad5f66f6c2132025`, file `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`.

The ancestry entries identify both the common Qwen base and the actual Gyro source; they do not imply endorsement. Credit for Gyro expert encoding, rotations, trellis formats and associated kernels belongs to AgentionAI. Sakura changes only the N-Gram/PLE table representation and its identification metadata, without retraining or pruning experts.

Model weights use **Qwen Community License 1.0**, not Apache/MIT: read the complete [LICENSE](LICENSE), including its commercial-use conditions. Runtime software uses MIT and the third-party notices included in `runtime/licenses`. Keep [NOTICE.md](NOTICE.md) and the applicable license texts when redistributing.
