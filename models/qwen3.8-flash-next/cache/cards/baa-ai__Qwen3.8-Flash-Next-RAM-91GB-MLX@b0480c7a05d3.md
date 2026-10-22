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
- moe
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next — 91GB (MLX)

Mixed-precision MLX build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) using a proprietary quantization method by [baa.ai](https://baa.ai).

Per-tensor bit-width allocation via advanced sensitivity analysis and budget-constrained optimisation, built at the efficiency knee (best quality-per-GB): 4× smaller than the 360 GB BF16 release, with a WikiText-2 perplexity within 0.02 of our [91 GB GGUF](https://huggingface.co/baa-ai/Qwen3.8-Flash-Next-RAM-91GB-GGUF) and a full point lower than a uniform 4-bit MLX build that is 13 GB larger. The full vision tower is retained (BF16), so image input works.

## Metrics

| Metric | Value |
|--------|-------|
| **Size on disk** | **90.72 GB** (90,718,613,415 bytes; 18 shards) — 93.5 GB with the optional MTP drafter |
| In-memory footprint | ~85 GiB weights; 95 GB peak measured during a 2048-token perplexity pass |
| Average bits per weight | 4.09 (177.4B parameters) |
| Format | MLX (Apple Silicon), affine quantization, `mlx-vlm` model type `qwen4_exp` |
| Architecture | 180B total / 6B active: 125B hybrid Gated-DeltaNet + Qwen Sparse Attention MoE (512 routed experts, top-10 + 1 shared), 51B n-gram embedding table, 4B MTP head |
| WikiText-2 perplexity (40 × 2048 tokens) | **3.53** |
| Uniform 4-bit MLX (same converter, group 64), same test | 4.53 (103.7 GB) |
| [RAM 91 GB GGUF](https://huggingface.co/baa-ai/Qwen3.8-Flash-Next-RAM-91GB-GGUF), same windows via `llama-perplexity` | 3.51 |
| unsloth `UD-Q3_K_XL` GGUF (90.0 GB), same windows via `llama-perplexity` | 3.59 |

Routed experts 3-bit; the 51B-parameter n-gram embedding table 4-bit (group 32 — its rows are 160 wide); attention, Gated-DeltaNet projections, shared experts, gated-residual mixers, PLE projections and routers 8-bit; token embedding and `lm_head` 6-bit; indexer, norms, convolutions and the vision tower BF16. Group size 64 unless stated. The per-module allocation is recorded in `config.json` under `quantization`.

Perplexity is scored exactly as `llama-perplexity` does (consecutive 2048-token windows of `wiki.test.raw`, NLL over the second half of each window, 40,920 scored tokens), so the MLX and GGUF rows are directly comparable; the per-window running averages of the two harnesses track each other to within 0.1.

Allocation variants measured while building (same test):

| Variant | Size | PPL |
|---|---|---|
| **Sensitivity-allocated knee, experts 3-bit, 8-bit attention floor (this release)** | **90.72 GB** | **3.53** |
| Same, expert down-projections raised to 4-bit | 95.75 GB | 3.54 |
| Uniform 4-bit, group 64 (`mlx_vlm.convert -q`) | 103.66 GB | 4.53 |

## Files

| File | Contents | Size |
|------|----------|------|
| `model-000{01..18}-of-00018.safetensors` + `model.safetensors.index.json` | Text model and vision tower | 90.72 GB |
| `config.json` | Model config including the per-module quantization map | — |
| `mtp/` | Multi-token-prediction draft model, 8-bit (optional, for speculative decoding) | 2.8 GB |
| `chat_template.jinja`, `tokenizer.json`, `vocab.json`, `*processor_config.json` | Tokenizer, chat template and image/video processors | — |

## Requirements

**mlx-vlm ≥ 0.7.10** (released 2026-10-21) — the first PyPI release with Qwen3.8-Flash-Next (`qwen4_exp`) support. `mlx-lm` alone cannot load this architecture.

```bash
pip install -U "mlx-vlm>=0.7.10"
```

## Usage

```bash
hf download baa-ai/Qwen3.8-Flash-Next-RAM-91GB-MLX --local-dir ./qwen3.8-flash-next-ram-91gb

# text
python -m mlx_vlm.generate --model ./qwen3.8-flash-next-ram-91gb \
    --prompt "What is the capital of Australia?" --max-tokens 256

# image
python -m mlx_vlm.generate --model ./qwen3.8-flash-next-ram-91gb \
    --image photo.jpg --prompt "Describe this image." --max-tokens 256

# speculative decoding with the bundled MTP drafter
python -m mlx_vlm.generate --model ./qwen3.8-flash-next-ram-91gb \
    --draft-model ./qwen3.8-flash-next-ram-91gb/mtp --draft-kind mtp \
    --prompt "Explain why the sky is blue." --max-tokens 256
```

Python:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

path = "baa-ai/Qwen3.8-Flash-Next-RAM-91GB-MLX"
model, processor = load(path)
config = load_config(path)

prompt = apply_chat_template(processor, config, "Describe this image in detail.", num_images=1)
out = generate(model, processor, prompt, image=["photo.jpg"], max_tokens=512, temperature=0.7, verbose=False)
print(out.text)
```

Or serve an OpenAI-compatible API with `python -m mlx_vlm.server --model ./qwen3.8-flash-next-ram-91gb --port 8080`.

Thinking is off in the `mlx_vlm.generate` CLI unless you pass `--enable-thinking`; in the Python API pass `enable_thinking=True` through `apply_chat_template`. Recommended sampling for Qwen3.8: temperature 0.7, top-p 0.9, top-k 20.

Fits on a 128 GB Mac (Mac Studio or MacBook Pro). Verified on a 192 GB Apple M2 Ultra: correct answers on factual, arithmetic, code and reasoning prompts with thinking on and off, and with the MTP drafter. Decode speed 14.7 tokens/s plain and 22.4 tokens/s with the MTP drafter (2.6 accepted tokens per round), peak memory 91 GB.

## License

Released under the [Qwen Community License 1.0](LICENSE), included in this repository.

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
