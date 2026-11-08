---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - qsa
  - gsq-rco
---

# Qwen3.8-27B QSA, GSQ-RCO IQ3_S

The model files for running Qwen3.8-27B with Qwen Sparse Attention on one 24 GB GPU, at 262K context with vision. These files need the engine at [pentaCoxian/Qwen-QSA-llama.cpp](https://github.com/pentaCoxian/Qwen-QSA-llama.cpp). Stock llama.cpp cannot run the QSA weights.

```sh
hf download pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF --local-dir models
deploy/qsa27b-24gb/serve.sh
```

Run those from a built checkout of the engine. Its README has the build steps.

**Update, 2026-11-08:** the QSA indexers in the main model are retrained for long context. Exact lookups past 32K are much more reliable (67% against 41% at 255K on the recall test below), with the same speed and memory. Nothing else in the file changed. The previous file is at revision [`1215f1a`](https://huggingface.co/pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF/tree/1215f1a010b878980f397aaf94b0495cd4c93c33).

## Files

| File | Size | What it is |
|---|---:|---|
| `Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf` | 12.3 GB | Main model with its MTP head and QSA indexers |
| `Qwen3.8-27B-DFlash2-IQ4_XS.gguf` | 1.0 GB | DFlash2 drafter, for speculative decoding |
| `mmproj-Qwen3.8-27B-QSA-F16.gguf` | 0.9 GB | Vision projector |

`SHA256SUMS` lists their checksums.

## Speed

Measured on one RTX 4090 at 262K context, with 255K tokens of input and 4 drafts:

| | |
|---|---:|
| Greedy decode, harmonic mean of 4 tasks | 114.7 tok/s |
| Sampled decode, temperature 1.0, top-p 0.95, top-k 20 | 113.0 tok/s |
| Cold prefill | 2,375 tok/s (108 s) |
| Peak VRAM | 22.8 GiB |

These were measured with the previous indexers. The retrained indexers have the same shapes and select the same number of tokens, so the work per token is the same. On an RTX 5090 at 255K, the retrained indexers decode at 195 tok/s against 187 tok/s for the previous ones, and the cold prefill takes 94 s with both.

The engine's README has the comparison with stock llama.cpp, per-task results, RTX 5090 results and limits.

## Long-context recall

The recall test asks 240 exact-lookup questions at each context length: key lookups, pointer chains, recipe expansion, latest revisions, table joins, code tracing and verified bindings. The records sit at ten depths in source code, among unrelated look-alike records that fill 15% of the text, and every key has a decoy one character off. Share of answers fully correct, on an RTX 5090 with the serve settings (4 drafts, Q8_0 KV cache), greedy:

| Context | Previous indexers | Retrained indexers | Dense attention |
|---|---:|---:|---:|
| 8K | 96% | 96% | 95% |
| 16K | 98% | 98% | 97% |
| 32K | 96% | 96% | 95% |
| 64K | 82% | 95% | 95% |
| 128K | 65% | 85% | 92% |
| 192K | 50% | 72% | 92% |
| 255K | 41% | 67% | 92% |

Dense attention is the same file with sparse attention turned off (`LLAMA_QSA_DENSE=1`). It shows what the model itself can do, at about half the decode speed at 255K. Sparse attention still loses some lookups past 64K, mostly when records sit in the middle of the document.

Other checks, retrained indexers against previous ones:

| | Retrained | Previous |
|---|---:|---:|
| Simple fact recall at 255K, single fact / two-hop | 24/24, 12/16 | 24/24, 11/16 |
| Vision, image 255K tokens before the question | 14/14 | 14/14 |
| KL divergence from dense attention at 16K, prose / code | 0.0047 / 0.0042 | 0.0046 / 0.0043 |
| Same top token as dense attention at 16K, prose / code | 97.7% / 98.3% | 97.7% / 98.3% |

The KL differences are within their error bars.

## How these files were made

- **Main model:** ISTA-DASLab's `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` with 80 F32 QSA indexer tensors and their metadata added. The 866 existing tensors are byte-identical to ISTA-DASLab's file. The indexers started from higa2515's `qsa-indexers.safetensors` and were retrained as described below.
- **Drafter:** z-lab's `Qwen3.8-27B-DFlash2-BF16.gguf`, quantized to IQ4_XS with `llama-quantize` and no imatrix.
- **Vision projector:** higa2515's `mmproj-Qwen3.8-27B-QSA-F16.gguf`, unchanged.

### Indexer retraining

QSA attends to a fixed 2,048 tokens chosen by a small indexer in each of the 16 attention layers. The original indexers were trained at 16K tokens. At 255K the same budget is 0.8% of the context, and their choices fell further behind the best possible choice as the context grew.

Only the indexers were trained, about 53M parameters; the rest of the model is unchanged. The target is the model's own dense attention, run from this IQ3_S file with sparse attention off: the attention of the 24 heads summed and max-pooled into 4-token blocks. The loss is the KL divergence from that target to the indexer scores, as in higa2515's training. Training started from the original indexers and ran 6 epochs (AdamW, learning rate 1e-4 with cosine decay) on one RTX 5090.

The training set is 56 documents of 16K to 255K tokens, about 7M tokens in all. The text is FineWeb-Edu web pages, Python source code and two public-domain novels, with planted records and questions in eight formats (directories, logs, inventories, notices, narrative facts, INI files, Markdown tables, constants), plus look-alike records at 12% of the text. None of the formats are the recall test's, and the recall test was never trained on. Its results did guide the choice of training data.

## Credits

All the source models are Apache-2.0, and so are these files.

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the base model
- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), the quantized weights
- [higa2515/Qwen3.8-27B-QSA-GGUF](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF), the QSA conversion, original indexers and vision projector
- [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), the drafter
- [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), web text used to retrain the indexers
