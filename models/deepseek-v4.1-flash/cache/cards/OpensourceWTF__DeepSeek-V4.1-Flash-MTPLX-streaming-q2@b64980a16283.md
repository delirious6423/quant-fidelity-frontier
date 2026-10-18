---
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mtplx
- moe
- quantized
- 2-bit
- streaming
- deepseek
- apple-silicon
---

# DeepSeek-V4.1-Flash MTPLX-streaming Q2

2-bit DeepSeek-V4.1-Flash (552B MoE, 384 routed experts, top-6) for Apple Silicon,
packaged for SSD-streamed serving with [MTPLX](https://github.com/OpenSourceWTF/MTPLX-STREAMING).
The routed experts ship as a flat binary bank that MTPLX pages from SSD into a
bounded GPU cache, so the model serves on Macs that cannot hold it resident.

## What this repo is

This is a **Q2 model designed for MLX**. Every tensor uses MLX-native affine
quantization (the accelerated `QuantizedLinear` / `gather_qmm` path), so nothing
is dequantized at load time and nothing needs a custom kernel:

| Part | Format | Fidelity vs source |
| :--- | :--- | :--- |
| Routed experts, 40 layers × 384 | affine 2-bit, group 64 (2.5 bpw), in `experts.bin` | cosine ≈ 0.91 (the one lossy step) |
| Attention, indexer, compressor, shared experts, embed, head, MTP dense | affine 8-bit, group 64 | cosine ≥ 0.9997, rel. error ≈ 0.5% |
| DSpark MTP routed experts, 3 layers × 128 | affine 8-bit, group 32, resident | cosine 0.99999 |
| Engram n-gram tables, layers 1 and 14 | affine 8-bit, group 64, in `engram/*.bin` | cosine ≥ 0.99994 (2,005 sampled rows per table vs the FP8 source) |
| Routers, norms, hyper-connection vectors, attention sinks, vision | bf16 / f32 verbatim | exact |

The source checkpoint stores routed experts as FP4 (E2M1) and dense weights as
FP8 (E4M3) with block scales. MLX's `mxfp4` / `mxfp8` modes do **not** repack those
formats bit-exactly (they re-derive group scales), so this release requantizes to
affine instead. Only the routed experts are taken down to 2 bits; everything else
stays at 8 bits or higher.

### Binary expert format for paging

Routed experts do not live in safetensors. `experts.bin` is a flat, 16 KiB-aligned
array of fixed-size records, one per (layer, expert), each holding the packed
2-bit gate/up/down weights with their scales and biases:

| | |
| :--- | :--- |
| record size | 11,059,200 bytes (exactly 2.5 bits per weight) |
| records | 15,360 = 40 layers × 384 experts |
| bank size | 158.2 GiB |
| record offset | `record_index × 11,059,200` |
| index | `expert-manifest.json` (sidecar-authoritative: offsets, per-record hashes, `model_key`) |

At serve time the runtime reads only the six routed experts a token selects per
layer, straight from SSD into pinned Metal slots, with an LRU expert cache under a
configurable memory envelope. The Engram tables use the same idea at row
granularity: `engram/engram-L1.bin` and `engram/engram-L14.bin` are flat arrays of
fixed-size affine-8-bit rows, record index equal to table row, so an n-gram lookup
is one offset read; `engram/engram-manifest.json` carries the record layout and the
hashing constants from DeepSeek's reference `engram.py`.

## Repo contents (~376 GiB)

| files | size | what |
| :--- | :--- | :--- |
| `experts.bin` + `expert-manifest.json` | 158.2 GiB | all routed expert records (2-bit gs64), paged from SSD |
| `model-000NN.safetensors` (49) + `model.safetensors.index.json` | 23.4 GiB | resident tensors: dense backbone 9.2 GiB (q8 gs64) + DSpark MTP experts 14.2 GiB (q8 gs32) |
| `engram/engram-L1.bin`, `engram/engram-L14.bin`, `engram/engram-manifest.json` | 194.6 GiB | Engram conditional-memory rows (q8 gs64), paged from SSD |
| `config.json` | — | source config plus an MLX `quantization` block with per-module overrides |
| `conversion-manifest.json` | — | provenance: source revision, per-shard sha256, quantization recipe, byte counts |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/`, `LICENSE` | — | verbatim from the source release |

Download everything:

```bash
hf download OpensourceWTF/DeepSeek-V4.1-Flash-MTPLX-streaming-q2
```

**Runtime status.** The MTPLX `deepseek_v41` backend (causal encoder-decoder KV
projection, CSA2 sparse attention modes, hierarchical indexer, Engram lookup,
DSpark speculative decoding) is in progress on the
[MTPLX-STREAMING](https://github.com/OpenSourceWTF/MTPLX-STREAMING) repository.
Until it lands, this repo is a weights-only release; the layout, manifests and
`model_key` (`deepseek-v41-flash-expert-q2`) are the contract that runtime targets.

**Quality.** No task evaluation of the 2-bit experts has been run yet. The
benchmark tables below are DeepSeek's published numbers for the source
checkpoint and do not describe this quantization. Expect a quality cost from
the 2-bit routed experts; the per-tensor cosines above are the only receipts so far.

**Vision.** The vision tower and aligner are included at 8-bit for completeness,
but the streaming runtime targets text-only serving.

**llama.cpp:** not served. MLX affine quantization does not convert to GGML
blocks in either direction.

## Conversion

Built with `scripts/convert_deepseek_v41_streamed.py` and
`scripts/convert_deepseek_v41_engram.py` from MTPLX-STREAMING branch
`feat/deepseek-v41-streaming`, CPU-only, from
`deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa45a94ad051997016db3960a90277`
(mlx 0.32.0). Dequantization of the source FP4/FP8 block formats mirrors
DeepSeek's `inference/convert.py`; each expert record and shard is sha256-pinned
in the manifests.

---

The rest of this card mirrors the source model card.

## Introduction

We introduce **DeepSeek-V4.1-Flash**, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. The model natively processes images and text, and generates text autoregressively.

**Architecture.** DeepSeek-V4.1-Flash adopts a **Causal Encoder-Decoder (CED)** architecture: a 40-layer Transformer organized as a 20-layer causal encoder followed by a 20-layer decoder. With CED, the decoder's global KV cache is projected from the final encoder hidden states rather than derived from each decoder layer's own hidden states. This allows the model to activate only **8B parameters per token during prefill** and **16B during decode**, substantially improving cost efficiency for input-heavy agentic workloads. **SWA Bounded Replay** reconstructs missing SWA KV states by replaying only the most recent *n*_win tokens, avoiding the need to persist SWA KV to SSD and reducing the persistent KV cache footprint to roughly **1/8** of that of DeepSeek-V4-Flash.

**Compressed Sparse Attention 2 (CSA2).** DeepSeek-V4.1-Flash uses CSA2, which assigns each attention layer one of three static modes — **Full**, **Reindex**, or **Reuse** — to share main KV and indexer K across layers and reuse Top-K sparse-attention indices. In the decoder, a **Hierarchical Sparse Indexer** further restricts later indexing layers to a candidate pool constructed by the first Full Mode layer, bounding deeper indexer cost independently of context length. Combined with **FP4 main KV caching** (E2M1 format, one E4M3 scale per 16 channels), these designs reduce the global KV cache footprint to **890 bytes per token** — roughly **1/4** of DeepSeek-V4-Flash.

**Additional architectural components** include Single-Pass mHC (revised residual-stream mixing with an efficient Mega-mHC kernel), Engram conditional memory (196B parameters, sparsely accessed via token-based lookup), and DSpark speculative decoding (semi-autoregressive draft generation with confidence-scheduled verification). The model uses 1 shared expert and 384 routed experts per MoE layer, activating 6 routed experts per token.

**Multimodal architecture.** A vision encoder (DeepSeek-ViT, trained from scratch with 2D-RoPE and 3×3 pixel-unshuffle downsampling) and a two-layer MLP projector convert images into visual embeddings, processed jointly with text embeddings from the start of language-model pre-training.

**Pre-training.** DeepSeek-V4.1-Flash is trained from scratch on a multimodal corpus comprising **45T tokens**, with sparse attention trained at a sequence length of 64K and context extended to 1M tokens at 34T tokens.

**Post-training.** The post-training recipe follows the standard SFT → RL → on-policy distillation (OPD) paradigm without algorithmic modifications. All substantive changes lie instead in the data pipeline: large-scale automated synthesis of agent tasks and environments with progressive scaling of data, tasks, and rollouts. The model supports a **continuously controllable reasoning effort** setting (integer 1–100) that trades inference cost for accuracy.

<div align="center">
  <img src="assets/dsv41_agentic_performance.png" width="48%" alt="DeepSeek-V4.1-Flash agentic benchmark performance" style="display: inline-block; margin: 0 1%;" />
  <img src="assets/dsv41_kv_cache.png" width="48%" alt="Global KV cache size per token across DeepSeek generations" style="display: inline-block; margin: 0 1%;" />
</div>

*Figure 1. (a) Performance of DeepSeek-V4.1-Flash and counterparts on agentic benchmarks. (b) Global KV cache size per token (bytes) across generations of DeepSeek models. DeepSeek-V4.1-Flash achieves approximately 4-fold and 437-fold reductions relative to DeepSeek-V4-Flash and DeepSeek-V1, respectively.*

## Evaluation Results

### Base Model

All base models are evaluated in our internal framework under the same evaluation settings. Scores within 0.3 of each other are considered equivalent.

<div align="center">

| Benchmark (Metric) | # Shots | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base | DeepSeek-V4.1-Flash-Base |
| :--- | :---: | :---: | :---: | :---: |
| Architecture | — | MoE | MoE | MoE |
| # Backbone Params | — | 284B | 1.6T | 552B |
| # Activated Params | — | 13B | 49B | 8B / 16B |
| **World Knowledge** | | | | |
| AGIEval (EM) | 3–5-shot | 83.9 | **84.4** | 83.4 |
| MMLU-Pro (EM) | 5-shot | 68.3 | 73.5 | **74.1** |
| C-Eval (EM) | 5-shot | 92.1 | **93.1** | 92.1 |
| MultiLoKo (LLM-Judge) | 5-shot | 42.6 | **50.9** | 45.5 |
| SimpleQA-Verified (EM) | 25-shot | 30.1 | **55.2** | 42.3 |
| SuperGPQA (EM) | 5-shot | 46.5 | **53.9** | 53.1 |
| **Language & Reasoning** | | | | |
| BBH (EM) | 3-shot | 86.9 | **87.5** | 86.1 |
| BBEH (EM) | 1-shot | 25.4 | **29.8** | 27.2 |
| DROP (F1) | 1-shot | **88.6** | **88.7** | 87.9 |
| HellaSwag (EM) | 0-shot | 85.7 | **88.0** | 87.2 |
| **Code & Math** | | | | |
| BigCodeBench (Pass@1) | 3-shot | 56.8 | 59.2 | **60.6** |
| HumanEval (Pass@1) | 0-shot | 69.5 | 76.8 | **79.4** |
| GSM8K (EM) | 8-shot | 90.8 | 92.6 | **93.0** |
| MATH (EM) | 4-shot | 57.4 | **64.5** | 61.1 |
| MGSM (EM) | 8-shot | **85.7** | 84.4 | 80.2 |
| **Long Context** | | | | |
| LongBench-V2 (EM) | 1-shot | 44.7 | **51.5** | 45.2 |
| **Multimodal** | | | | |
| MMMU-Pro (EM) | 4-shot | — | — | 56.5 |
| CVBench (EM) | 4-shot | — | — | 77.9 |
| DocVQA (LLM-Judge) | 4-shot | — | — | 95.6 |
| RefCOCO-avg (Acc@0.5) | 0-shot | — | — | 86.0 |

</div>

### Instruct Model

DeepSeek-V4.1-Flash supports a continuously controllable reasoning effort from 1 to 100. All instruct results below use the maximum effort setting (`reasoning_effort=100`). Evaluations use `temperature=1.0, top_p=0.95`.

For code agent benchmarks (Terminal-Bench 2.1/3.0/4.0, DeepSWE v1.1, NL2Repo-Bench, ProgramBench), the model is evaluated with the Minimal mode of DeepSeek Harness and a 1M-token context window. To align with official setup requirements, the mini-SWE harness is used for DeepSWE v1.1, and the Claude Code harness for SEC-Bench Pro. Visual agent benchmarks (Chartography, BabyVision, ZeroBench) use the Claude Code harness with a 512k-token context window. Agent's Last Exam and AutomationBench use their official scaffolds. All agentic evaluations use `temperature=1.0, top_p=0.95`.

#### Comparison with frontier models (Max reasoning effort)

<div align="center">

| Benchmark (Metric) | Opus-5.0 | GPT-5.6 Sol | K3 | GLM-5.3 | DS-V4-Pro | DS-V4-Flash | DS-V4.1-Flash |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reasoning** | | | | | | | |
| GPQA Diamond (Pass@1) | 93.4 | **94.1** | 92.9 | 88.1 | 92.4 | 89.9 | 90.9 |
| HLE (Pass@1) | **56.3** | 44.5 | 43.5 | 42.0† | 42.7† | 37.8† | 36.8 (39.1†) |
| Codeforces (Rating) | — | — | — | — | 3348 | 3289 | **3471** |
| MathArena Apex (Pass@1) | — | — | **65.6** | — | 65.3 | 58.6 | **65.6** |
| **Agentic** | | | | | | | |
| Terminal-Bench 2.1 (Pass@1) | 89.1 | 88.8 | 88.3 | 88.2 | 87.9 | 82.7 | **90.6** |
| Terminal-Bench 3.0 (Pass@1) | **43.3** | 34.4 | 17.7 | 28.3 | 11.8 | 7.6 | 30.0 |
| Terminal-Bench 4.0 (Pass@1) | **51.8** | 39.9 | 12.6 | 37.9 | 12.4 | 7.0 | 31.2 |
| DeepSWE v1.1 (Resolved) | 74.0 | 73.0 | 67.5 | 66.9 | 62.7 | 54.4 | **74.2** |
| ProgramBench (Almost@1) | **37.0** | 23.0 | 17.5 | 19.0 | 15.5 | — | 20.3 |
| NL2Repo-Bench (Score) | **75.3** | 56.8 | 58.0 | 58.0 | 61.5 | 54.2 | 64.0 |
| CyberGym (Pass@1) | — | 84.5 | 80.0 | 84.5 | 83.3 | 76.7 | **88.1** |
| SEC-Bench Pro (Pass@1) | — | **74.3** | — | — | 56.4 | 30.9 | 62.8 |
| ExploitGym (Pass@1) | 22.1 | **33.7** | — | 15.0 | 5.4 | 1.8 | 15.3 |
| HLE w/ tools (Pass@1) | 63.6 | — | 59.8 | 62.5 | 60.0 | 51.5 | **63.9** |
| AutomationBench (Pass@1) | 50.3 | 45.8 | 46.7 | 48.8 | 43.2 | 37.7 | **54.8** |
| Agent's Last Exam (Pass@1) | 28.6 | 26.7 | 27.6 | 28.5 | 25.7 | 25.2 | **31.8** |
| Chartography w/ tools (Pass@1) | **84.0** | 79.9 | 68.1 | — | — | — | 78.9 |
| BabyVision w/ tools (Pass@1) | **94.1** | 88.9 | 85.7 | — | — | — | 89.6 |
| ZeroBench-main w/ tools (Pass@5) | 52.0 | **53.0** | 41.0 | — | — | — | 49.0 |

</div>

*† Text-only subset of HLE.*

#### Performance across agent scaffolds (DeepSWE v1.1 and Terminal-Bench 2.1, Max reasoning effort)

All scaffolds use N=8 samples per task on DeepSWE v1.1 and N=3 on Terminal-Bench 2.1, with Linux containers, `temperature=1.0`, `top_p=0.95`, a 1M-token context limit, and max_steps=500 per agent. Terminal-Bench 2.1 is evaluated without network access.

<div align="center">

| Benchmark (Metric) | Claude Code | Codex | OpenCode | Pi | mini-SWE | DSH Minimal | DSH Standard | DSH PTC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepSWE v1.1 (Resolved) | 69.8 | 65.6 | 65.5 | 66.2 | 74.2 | 72.6 | 70.5 | 67.6 |
| Terminal-Bench 2.1 (Pass@1) | 88.0 | 84.1 | 85.0 | 86.1 | 90.3 | 90.6 | 85.8 | 85.8 |

</div>


## Prompt Encoding

This release does not include a Jinja-format chat template. The [`encoding`](encoding/README.md) folder contains DeepSeek's self-contained Python reference implementation (`encoding.py`) with test cases for multi-turn conversations, tool calling, thinking mode, numeric reasoning effort, mid-conversation system messages, and interleaved image content. DeepSeek also publishes [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) with the same prompt format as a maintained toolkit.

**Recommended sampling parameters (from the source card):**

| Parameter | Value |
| :--- | :--- |
| `temperature` | 1.0 |
| `top_p` | 0.95 or 1.0 |

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```

