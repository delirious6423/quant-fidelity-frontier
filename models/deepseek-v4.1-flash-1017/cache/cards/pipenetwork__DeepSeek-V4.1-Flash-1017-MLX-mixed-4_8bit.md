---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- deepseek_v41
- mixture-of-experts
- 4-bit
pipeline_tag: text-generation
library_name: mlx
---

# DeepSeek-V4.1-Flash-1017-MLX-mixed-4_8bit

MLX (Apple Silicon) build of [**DeepSeek-V4.1-Flash**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) —
754.6B parameters: 40 layers x 384 routed experts, MLA with cross-layer KV-cache sharing (4
compressor-owning layers serve all 40), a two-layer **engram** hashed n-gram embedding whose
tables alone are 196.6B parameters (~40% of the checkpoint), staggered Sinkhorn
hyper-connections, and per-layer attention sinks — quantized to **4-bit experts / 8-bit attention&shared / 8-bit engram**.

**These files are modified**: dequantized from the FP8/FP4 release (bit-exact decode of the
32x32-block ue8m0 fp8 and per-32 fp4 packing) and re-quantized; the architecture is unchanged.
The 3 multi-token-prediction layers (DSpark markov/confidence heads) are not included; the vision
tower and aligner are carried unmodified but the runtime is text-only.

## Runtime

`deepseek_v41` exists in **no** runtime — not transformers, not mlx-lm, not mlx-vlm. This
checkpoint loads only through the port:

```bash
git clone https://github.com/PipeNetwork/deepseek-v41-mlx && cd deepseek-v41-mlx && pip install -r requirements.txt
python scripts/smoke_generate.py /path/to/DeepSeek-V4.1-Flash-1017-MLX-mixed-4_8bit
```
```python
from deepseek_v41_mlx.load import load
model, tokenizer = load("/path/to/DeepSeek-V4.1-Flash-1017-MLX-mixed-4_8bit")
```

The port was validated against DeepSeek's own `inference/model.py` (the only reference): fp32
tiny-config parity **1e-6** across prefill / cached decode / chunked prefill, the three QAT
fake-quant ops bit-exact, and negative controls proving the fragile paths are load-bearing
(rope-inverse 0.84, attention sinks 0.65, cross-layer sharing 0.56 logit shift when broken).
Strict loading reports zero missing / zero unexpected tensors. One reference *decode* bug was
found and documented (odd-step indexer reads the wrong layer's keys; 0.67 logit shift — the port
uses the owner's cache): see `docs/upstream-notes.md` in the repo.

## Size and what is quantized

**427.6 GB** on disk. RAM: 512 GB Mac (tight: 427 GB resident).

| group | share of parameters | this build |
|---|---:|---|
| routed experts (40 x 384, `w1/w2/w3`) | 543.6B (72%) | 4-bit, group 64 |
| engram tables (2 x [384,006,168 x 256]) | 196.6B (26%) | 8-bit, group 64 |
| attention (MLA), shared experts, embeddings, `head` | ~14B | 8-bit, group 64 |
| `wo_a` (block-diagonal output LoRA), hyper-connections, sinks, router biases, compressor, indexer keys, norms | — | unquantized (bf16/fp32) |

## Quality

**Per-layer divergence ladder** vs the bf16-dequantized reference — every one of the 40 decoder
layers run on identical inputs (16,384 tokens of wikitext-2), teacher-forced and free-running,
with the ladder's arithmetic asserted bit-identical to this converter's:

| recipe | teacher-forced (mean) | free-running (final layer) | cosine (final) |
|---|---:|---:|---:|
| mixed 4/8, engram as shipped (fp8/ue8m0) | 0.0337 | 0.1947 | 0.9803 |
| **mixed 4/8, engram 6-bit (this set's 1 TB build)** | 0.0338 | 0.1928 | 0.9804 |
| **mixed 4/8, engram 4-bit (this set's 512 GB build)** | 0.0344 | 0.2126 | 0.9769 |

The engram finding that shapes this set: **6-bit engram is indistinguishable from the shipped
fp8 tables** (free-running 0.1928 vs 0.1947) while **4-bit engram costs +9.2%** free-running —
but the engram-6 build is 477 GB and a 512 GiB machine cannot run it, so the engram-4 build is
the one that fits and the engram-6 build serves 1 TB machines.

**Perplexity** (wikitext-2 test, 286,580 tokens in 140 windows of 2048, through this runtime): **2.9238** [2.7364, 3.1264]. Greedy generation is coherent (collapse check). Paired against the previous checkpoint's build on identical windows, perplexity is higher (worse): ratio **x1.0095** [1.0050, 1.0139], worse on 93/140 windows. Held-out prose perplexity is not the axis this checkpoint was improved on — upstream reports large agentic, reasoning and vision gains — so read it as a conversion-fidelity check, not a verdict on the model.

## License

MIT, as the upstream model. Port code: [https://github.com/PipeNetwork/deepseek-v41-mlx](https://github.com/PipeNetwork/deepseek-v41-mlx).
