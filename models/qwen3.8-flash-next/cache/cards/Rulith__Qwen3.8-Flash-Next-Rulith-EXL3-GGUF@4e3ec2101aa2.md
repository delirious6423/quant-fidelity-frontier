---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: turboderp
library_name: gguf
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- exl3
- rulith
---

# Qwen3.8-Flash-Next EXL3 GGUF for Rulith Inference

EXL3 (trellis-coded) weights of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in a GGUF
container, for [Rulith Inference](https://github.com/rulith-dev/rulith-inference) on AMD Strix Halo (Windows).

**The quantization is turboderp's.** The EXL3 tensors are taken bit for bit from
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (ExLlamaV3 1.4.4,
codebook `mul1`, calibration 250 rows x 2048) and only re-packed; all credit for the quantization goes to turboderp and
the [ExLlamaV3](https://github.com/turboderp-org/exllamav3) project. The non-EXL3 tensors (hyper-connection mixers,
router, norms, embeddings, the n-gram table codec) follow the layout and types Rulith Inference is tuned for.

| Folder | Source branch | Experts / dense / head bits | Size |
|---|---|---|---|
| `EXL3-4.05bpw` | `4.05bpw_h6_ng6` | 4 / 6 / 6 | 104.3 GB, 6 shards |
| `EXL3-3.05bpw` | `3.05bpw_h5_ng5` | 3 / 5 / 5 | 82.4 GB, 5 shards |

4.05 bpw is the recommended one; 3.05 bpw is smaller and generates about 8% faster, at twice the KL divergence (see
the comparison below). In both, the first shard holds only the metadata (11 MB), so a client can read the model's
header before the weights; the third is one tensor, the per-layer token embedding table (39.0 / 32.6 GB), which cannot
be split.

Also here: `mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf`, the multi-token-prediction draft for both folders (the model's
MTP layers plus Rulith's low-rank draft head with its own output head: an EXL3 target cannot lend its head rows).
Image input: `mmproj-F16.gguf`, the model's vision projector,
unchanged from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) and included here so that one download has everything; keep it
in the same folder as the model.

## Compatibility

These files use EXL3 tensor types (GGUF type ids 256 and up) that upstream llama.cpp, LM Studio, Ollama and other
llama.cpp-based tools do not have. They load in Rulith Inference 0.6.0 or newer, which lists them under Downloads and fetches
them from Hugging Face itself; pass the first shard to `llama-server` if you run it by hand, the other shards are found by name.

## What is inside (4.05 bpw)

| Part | Bits | Size | Where it runs |
|---|---|---|---|
| Routed experts (48 layers x 512 experts, gate / up / down) | 4 (EXL3, mul1 codebook) | 60.4 GB | GPU; on the 64 GB GPU setting the app moves 21 layers' experts to system memory by itself |
| Attention, linear attention, shared experts, indexer | 6 (EXL3) | 2.1 GB | GPU |
| Output head | 6 (EXL3) | 0.5 GB | GPU |
| N-gram table (per-layer token embeddings, 320M rows x 160) | 6 (EXL3 row codec) | **39.0 GB** | system memory: read from the file by rows, only the tokens in use |
| Hyper-connection mixers, router, norms, token embedding, small projections | Q8_0 / F32 | ~2.2 GB | GPU |

The first shard holds only the metadata (11 MB), so a client reads the model's header before the weights; the third
(39 GB) is the n-gram table, one tensor that cannot be split.

The 3.05 bpw files have the same parts at 3 bits for the routed experts (45.3 GB) and the indexer, 5 bits for the
attention, linear attention, shared experts and output head, and a 5-bit n-gram table (32.6 GB).

## Memory

Measured at Rulith Inference's defaults (8 conversation slots sharing a 262K-token cache, MTP, image input):

| GPU memory setting (BIOS) | on the GPU | in system memory | |
|---|---|---|---|
| 64 GB | 55 GB at load, 56 GB with 60K tokens in the cache | 24.6 GB of experts (21 layers) + the n-gram table, read from disk as needed | the default placement; nothing to set |
| 96 GB | all weights but the n-gram table: ~81 GB with the cache (estimated by the app) | the n-gram table only | |

3.05 bpw needs about 16 GB less GPU memory: on the 64 GB setting the app keeps 10 layers' experts in system memory
instead of 21 (its estimate at the same defaults).

A machine with 128 GB of unified memory runs it either way; the app decides the placement by itself from the GPU memory
it finds and what other programs hold.

## Compared with the GGUF quantizations

The same machine (Ryzen AI Max+ 395, 64 GB GPU setting), the same evaluation set, Rulith Inference 0.6.0 at its
defaults. Quality: KL divergence to the BF16 model (lower is better) and top-1 agreement over six kinds of text
(wikitext, C4, multilingual, chat, code, math; context 1024). Speed: reading a 16.9K-token prompt, and one conversation
answering with MTP on.

| File | Size | KLD (mean) | KLD (p99) | top-1 | Prompt reading | One conversation |
|---|---:|---:|---:|---:|---:|---:|
| **EXL3 4.05 bpw (this repository)** | 104.3 GB | **0.022** | **0.378** | **96.0%** | 1,230 t/s | 56 tok/s |
| [unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 111.3 GB | 0.024 | 0.415 | 95.9% | 1,265 t/s | 56 tok/s |
| [unsloth UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 93.7 GB | 0.048 | 0.815 | 94.4% | 1,335 t/s | 60 tok/s |
| **EXL3 3.05 bpw (this repository)** | 82.4 GB | 0.044 | 0.736 | 94.5% | see below | see below |

Per domain (KLD), 4.05 bpw: wikitext 0.049, C4 0.016, multilingual 0.026, chat 0.011, code 0.004, math 0.024;
3.05 bpw: wikitext 0.103, C4 0.033, multilingual 0.048, chat 0.022, code 0.015, math 0.041.

3.05 bpw against 4.05 bpw on Rulith Inference 0.6.3 (a 5.9K-token prompt, then 256 tokens, two interleaved runs each):
prompt reading the same (~1,240 t/s), one conversation 30.9 vs 28.4 tok/s without MTP and 56.8 vs 52.8 tok/s with MTP.

## Files

Every file's size and SHA-256 are in `manifest.json`.

## License

The weights follow the base model's license (Qwen Community License 1.0). The EXL3 quantization is turboderp's, released
under the same terms; the repackaging is by Rulith.
