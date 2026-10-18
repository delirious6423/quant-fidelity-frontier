---
library_name: mlx
inference: false
license: mit
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4
- mlx
- quantized
- 2bit
- mixed-precision
- optiq
- moe
- ssd-streaming
- apple-silicon
- text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
---

# mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/) · [DeepSeek family](https://mlx-optiq.com/docs/deepseek)

**A 304-billion-parameter model that runs in 6.5 GB of RAM on a Mac.** This is a 2-bit mixed-precision MLX quant of DeepSeek-V4-Flash-0731, produced by [mlx-optiq](https://mlx-optiq.com/). At bf16 the weights are 608 GB. Here they are 92.5 GB on disk, and while the model generates only about 6.5 GB sits in RAM: attention, the router, the shared expert and the embeddings stay resident, and the 256 routed experts are read off the SSD as the router picks them.

DeepSeek-V4-Flash is a sparse mixture-of-experts model with a million-token context: 43 layers, 256 routed experts with 6 active per token, and three attention regimes across the stack. Given a 928-token specification of a job queue and asked what happens to a job whose worker stops sending heartbeats, the 2-bit model answered by chaining two separate rules of the spec, correctly.

Asked to write Flappy Bird as a single HTML file, the 2-bit model produced the physics, the pipe generation, the collision checks, the scoring and the canvas rendering. Here it is playing the game it wrote:

![DeepSeek-V4-Flash 2-bit playing the Flappy Bird it wrote](flappy.gif)

The game is in this repo as [`flappy_bird.html`](flappy_bird.html). Open it in any browser.

## What it is

| Property | Value |
|---|---|
| Base | deepseek-ai/DeepSeek-V4-Flash-0731 (sparse MoE, 256 experts, 6 active per token, 43 layers) |
| Parameters | 304 B |
| Context | 1,048,576 tokens |
| Bit-widths | 2-bit routed experts; 6-bit attention; 8-bit shared expert, embeddings and LM head; 3-bit MTP head |
| Achieved bits-per-weight | 2.43 |
| On disk | 92.5 GB (608 GB at bf16) |
| Resident while running | ~6.5 GB (routed experts streamed) |
| Decode speed | ~2.5 tok/s on an M3 Max, SSD-bound |

No Capability Score is published for this quant. Running the six-benchmark suite against a model that decodes off SSD would take days, and at 2 bits on the routed experts the point of the artifact is different: that a 304 B MoE runs at all on consumer Apple Silicon, and stays coherent enough to follow a long specification and write working code.

## Run it

DeepSeek-V4 is not an architecture stock mlx-lm knows, so `import optiq` once to register it:

```bash
pip install "mlx-optiq>=0.4.12"
```

The routed experts are far too large to sit resident, so serve it with SSD expert streaming. `optiq serve` turns this on by itself for a MoE quant that would not fit in RAM (`--stream-experts` forces it):

```bash
optiq serve --model mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit
```

That gives you an OpenAI and Anthropic compatible endpoint with mixed-precision KV cache, tool-call healing and prompt caching. Only the routed experts stream per token, so the footprint stays near 6.5 GB regardless of how large the model on disk is.

A fast SSD matters more than RAM here. Every token reads 6 experts per layer from disk, so decode speed tracks read throughput.

## Notes

This is an **extreme quant.** Two bits on the routed experts is lossy, and anything where accuracy matters should use the bf16 weights or a higher-bit quant. What this one demonstrates is a model of this size running on a Mac, at a resident footprint that fits a 16 GB machine.

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **All OptiQ quants:** [mlx-optiq.com/models](https://mlx-optiq.com/models)
- **PyPI:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)
- **Base model:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
