---
library_name: mlx
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- qwen
- qwen3.8
- mlx
- quantized
- 2bit
- mixed-precision
- optiq
- moe
- ssd-streaming
- apple-silicon
- text-generation
base_model: Qwen/Qwen3.8-Flash-Next
---

# mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/)

**A 176-billion-parameter model that generates in under 5 GB of RAM on a Mac.** This is a 2-bit mixed-precision MLX quant of Qwen3.8-Flash-Next, produced by [mlx-optiq](https://mlx-optiq.com/). At bf16 the weights are about 350 GB. Here they are 81 GB on disk, and while the model generates, peak memory stays near 4.6 GB. Attention, the router, the shared expert and the hyper-connections stay resident. The 512 routed experts and the 51B-parameter n-gram embedding are read off the SSD, a few rows and a few experts at a time.

Qwen3.8-Flash-Next pairs Gated DeltaNet with Qwen Sparse Attention, and adds an n-gram embedding: a 20-million-entry table of bigrams and trigrams consulted at layer 2. It has 125B parameters in the transformer with 6B active per token, plus 51B in that table. Each token reads 16 rows of the table, so it streams well.

Asked to write Flappy Bird as a single HTML file, the 2-bit model produced the canvas rendering, the gravity and flap physics, the pipe generation, the scoring and a game-over screen with restart, in 2,310 tokens and ten minutes on an M3 Max. Here is the game it wrote:

![Qwen3.8-Flash-Next 2-bit: the Flappy Bird it wrote](flappy.gif)

The game is in this repo as [`flappy_bird.html`](flappy_bird.html). Open it in any browser.

## What it is

| Property | Value |
|---|---|
| Base | Qwen/Qwen3.8-Flash-Next (hybrid Gated DeltaNet + QSA, 512 experts, 10 routed + 1 shared active per token, 48 layers) |
| Parameters | 125 B (6 B active) plus a 51 B n-gram embedding |
| Context | 262,144 tokens |
| Bit-widths | 2-bit routed experts; 4-bit attention, DeltaNet, shared expert, embeddings, LM head and n-gram table; 8-bit hyper-connections; bf16 router |
| On disk | 81 GB (about 350 GB at bf16) |
| Peak memory while generating | ~4.6 GB (routed experts and n-gram table streamed) |
| Decode speed | ~3.8 tok/s on an M3 Max, SSD-bound |

No Capability Score is published for this quant. Running the six-benchmark suite on a model that decodes off SSD would take days. At 2 bits on the routed experts, what this artifact shows is different: that a 176 B model runs at all on consumer Apple Silicon, and stays coherent enough to write working code.

The multi-token-prediction head is not included. The vision tower is kept at bf16 alongside the weights, but image input has not been tested on this quant; treat it as text-only.

## Run it

Qwen3.8-Flash-Next is not an architecture stock mlx-lm knows, so it needs mlx-optiq 0.5.14 or later:

```bash
pip install "mlx-optiq>=0.5.14"
```

The routed experts and the n-gram table are far too large to sit resident, so serve it with SSD streaming. `optiq serve` turns this on by itself for a MoE quant that would not fit in RAM (`--stream-experts` forces it):

```bash
optiq serve --model mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit
```

That gives you an OpenAI and Anthropic compatible endpoint with prompt caching and tool-call healing. A fast SSD matters more than RAM here: every token reads 10 experts per layer and 16 table rows from disk, so decode speed tracks read throughput.

## Notes

This is an **extreme quant.** Two bits on the routed experts is lossy, and anything where accuracy matters should use the original weights or a higher-bit quant. What this one demonstrates is a model of this size running on a Mac, in a footprint that fits a 16 GB machine.

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **All OptiQ quants:** [mlx-optiq.com/models](https://mlx-optiq.com/models)
- **PyPI:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)
- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
