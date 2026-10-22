---
library_name: mlx
tags:
- mlx
- quantized
- mixed-precision
- vision-language
- image-text-to-text
- qwen
- qwen3.8
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — 13GB (MLX, vision-preserving)

Mixed-precision MLX build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using a proprietary quantization method by [baa.ai](https://baa.ai). This is the Apple Silicon sibling of the [13GB GGUF](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-13GB-GGUF) (the single-16 GB-GPU reader): a full 27B language model plus the retained vision tower in 13 GB, for a 16 GB Mac. For a larger, closer-to-lossless build see the [24GB MLX](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-24GB-MLX).

Per-tensor bit-width allocation via advanced sensitivity analysis and budget-constrained optimisation: the allocator holds the language decoder at its 3-bit floor and spends the remaining budget on the tensors it measured as most sensitive (Gated-DeltaNet gates, attention keys and values, some output projections), with 3-bit token embeddings, a 6-bit output head and an 8-bit vision tower.

## Metrics

| Metric | Value |
|--------|-------|
| **Size on disk** | **13.0 GB** (13,011,014,537 bytes; 3 shards) |
| In-memory footprint | ~12.1 GiB weights; 13.4 GB peak measured while generating |
| Average bits per weight | 3.81 |
| Format | MLX (Apple Silicon), affine quantization, group size 64, `mlx-vlm` model type `qwen3_5` |
| Architecture | 27B dense hybrid: 48 Gated-DeltaNet (linear attention) + 16 full-attention layers, vision tower (27 blocks) |
| WikiText-2 perplexity (40 × 2048 tokens) | **6.46** |
| Qwen3.8-27B BF16 (55.6 GB), same test, same harness | 5.99 |
| [13GB GGUF](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-13GB-GGUF) (12.58 GB, imatrix), same windows via `llama-perplexity` | 6.02 |
| Agent-safety coherence gate (*Fidelity Is Not Safety*, 400 decoder tensors) | PASS (coherent fraction 0.0054, threshold 0.007) |

Language decoder: `mlp` gate/up/down, DeltaNet `in_proj_qkv`/`in_proj_z` and attention `q_proj` 3-bit; DeltaNet `in_proj_a`/`in_proj_b` 5–8-bit; attention `k_proj`/`v_proj` 4–6-bit; `o_proj` and DeltaNet `out_proj` 3–5-bit; token embedding 3-bit; `lm_head` 6-bit; vision tower 8-bit (one MLP projection per block stays BF16 because its width is not a multiple of 64); norms and convolutions BF16. The per-module map is in `config.json` under `quantization`.

Perplexity is scored exactly as `llama-perplexity` does (consecutive 2048-token windows of `wiki.test.raw`, loss over the second half of each window, 40,920 scored tokens), so the MLX, BF16 and GGUF rows are directly comparable. The GGUF is calibrated with an importance matrix, which MLX's quantizer does not use; the gap between the two 13 GB builds is that calibration.

## Files

| File | Contents | Size |
|------|----------|------|
| `model-*.safetensors` + `model.safetensors.index.json` | Language model and vision tower | 13.0 GB |
| `config.json` | Model config including the per-module quantization map | — |
| `chat_template.jinja`, `tokenizer.json`, `vocab.json`, `*processor_config.json` | Tokenizer, chat template and image/video processors | — |

## Usage

Load with **mlx-vlm** (`pip install -U mlx-vlm`); `mlx-lm` would drop the vision tower.

```bash
hf download baa-ai/Qwen3.8-27B-RAM-13GB-MLX --local-dir ./qwen3.8-27b-ram-13gb

python -m mlx_vlm.generate --model ./qwen3.8-27b-ram-13gb --prompt "What is the capital of Australia?" --max-tokens 256
python -m mlx_vlm.generate --model ./qwen3.8-27b-ram-13gb --image photo.jpg --prompt "Describe this image." --max-tokens 256
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

path = "baa-ai/Qwen3.8-27B-RAM-13GB-MLX"
model, processor = load(path)
config = load_config(path)
prompt = apply_chat_template(processor, config, "Describe this image in detail.", num_images=1)
out = generate(model, processor, prompt, image=["photo.jpg"], max_tokens=512, temperature=0.7, verbose=False)
print(out.text)
```

Or serve an OpenAI-compatible API with `python -m mlx_vlm.server --model ./qwen3.8-27b-ram-13gb --port 8080`.

Qwen3.8 is a reasoning model. Thinking is off in the `mlx_vlm.generate` CLI unless you pass `--enable-thinking`; in the Python API pass `enable_thinking=True` through `apply_chat_template`. Recommended sampling: temperature 0.7, top-p 0.9, top-k 20.

## Hardware

Runs on a 16 GB Apple Silicon Mac; 24 GB gives comfortable room for long contexts. Verified on a 192 GB M2 Ultra: correct answers on factual, arithmetic, code and reasoning prompts with thinking on and off; decode 22 tokens/s, prefill 57 tokens/s, peak memory 13.4 GB.

## License

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

---
*Quantized by [baa.ai](https://baa.ai)*

---

## Black Sheep AI Products

Black Sheep AI builds private AI with the **Bounded AI Architecture (BAA)**: three layers, each bound to something you control, on infrastructure you control. Nothing leaves your boundary.

**[Ram](https://baa.ai/compression.html)** (Reasoning Layer) — Full-capability frontier models, shrunk to fit smaller GPUs with reasoning kept and certified. They rely strictly on your Data Modules for knowledge, so you swap the model as better ones ship.

**[Paddock](https://baa.ai/paddock.html)** (Knowledge Plane) — Binds your Data Modules directly to the reasoning layer: one per topic, period, person, permission or tenant. Permission, freeze, hand back or delete each one on its own.

**[Watchman](https://baa.ai/watchman.html)** (GRC Layer) — Governance, risk and compliance in a single, complete audit log: every module permission, every bind, every model's provenance. Five certificates before anything goes live.

**[Shepherd](https://baa.ai/shepherd.html)** — Takes a frontier model, compresses it into a Ram reasoning model, certifies it with Watchman, binds it to your Paddock, and deploys the Bounded AI Architecture to your infrastructure: your cloud, a Mac Studio, or an air-gapped facility.

Learn more at **[baa.ai](https://baa.ai)** — Private AI for enterprises whose data cannot leave the building.
