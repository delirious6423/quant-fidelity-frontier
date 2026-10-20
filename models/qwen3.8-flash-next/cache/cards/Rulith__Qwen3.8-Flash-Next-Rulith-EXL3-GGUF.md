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
| `EXL3-4.05bpw` | `4.05bpw_h6_ng6` | 4 / 6 / 6 | ~104 GB (shards) |

Also here: `mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf`, the multi-token-prediction draft for these files (the model's
MTP layers plus Rulith's low-rank draft head with its own output head: an EXL3 target cannot lend its head rows). For
image input use the `mmproj-F16.gguf` of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

## Compatibility

These files use EXL3 tensor types (GGUF type ids 256 and up) that upstream llama.cpp, LM Studio, Ollama and other
llama.cpp-based tools do not have. They load in Rulith Inference 0.6.0 or newer, which picks them from its model list and
downloads them itself; pass the first shard to `llama-server` if you run it by hand, the other shards are found by name.

## Quality

Measured with Rulith's evaluation set (KL divergence to the BF16 model over a mixed corpus, context 1024):

| File | KLD (mean) | KLD (p99) | top-1 agreement |
|---|---|---|---|
| EXL3 4.05 bpw | 0.0211 | 0.345 | 96.16% |

## Files

Every file's size and SHA-256 are in `manifest.json`.

## License

The weights follow the base model's license (Qwen Community License 1.0). The EXL3 quantization is turboderp's, released
under the same terms; the repackaging is by Rulith.
