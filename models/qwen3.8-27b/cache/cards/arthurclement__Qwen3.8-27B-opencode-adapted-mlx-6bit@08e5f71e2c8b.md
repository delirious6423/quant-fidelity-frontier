---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
language:
  - en
tags:
  - mlx
  - qwen3.8
  - 27b
  - coding
  - agent
  - tool-calling
  - opencode
  - lora-merged
  - 6-bit
  - kapitsa
---

# Qwen3.8-27B opencode-adapted — MLX 6-bit

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the
[harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted)
LoRA merged into the weights, then quantized to MLX for Apple Silicon.

The adapter was trained by harness-hub with reinforcement learning (GRPO) inside the
[OpenCode](https://opencode.ai) agent harness, to make the model a better OpenCode coding agent:
higher task success, fewer tool calls, shorter answers. See their
[blog post](https://huggingface.co/blog/harness-adapters-at-scale) and model card for details.

This repo only repackages their work for MLX. **It has not been re-evaluated on the authors'
benchmarks**: merging and 6-bit quantization may change the results they report.

## How it was built

1. LoRA (r=32, alpha=64, all linear projections) merged into the BF16 base weights:
   `W' = W + (alpha / r) · B·A`, computed in float32, shard by shard with MLX.
2. Quantized with `mlx_lm.convert -q --q-bits 6 --q-group-size 64` (affine).
3. `mlx_lm.convert` drops the MTP head. For speculative decoding with Kapitsa, `mtp.safetensors`
   and `mtplx_runtime.json` were copied from the Qwen3.8-27B 6-bit model of the
   [Kapitsa](https://github.com/nace-ai/kapitsa) catalog (see *Files*).

| Property | Value |
|---|---|
| Parameters | 27B (dense, hybrid Gated DeltaNet / attention) |
| Quantization | MLX affine 6-bit, group size 64 |
| Memory when loaded | ~25.7 GB (Kapitsa) |
| Context | 256K tokens |
| Vision | Not tested; text-only use intended |

## Files

| File | Origin |
|---|---|
| `model-*.safetensors`, `config.json`, tokenizer files | Qwen3.8-27B + harness-hub LoRA, merged and quantized here |
| `mtp.safetensors` | MTP head of Qwen/Qwen3.8-27B (Apache-2.0), as converted in the Kapitsa catalog. **Not retrained with the adapter.** |
| `mtplx_runtime.json` | Kapitsa runtime settings for that MTP head |

The last two files are only used by Kapitsa; other runtimes ignore them.

## Usage

**Turn thinking off.** The adapter was trained and evaluated with thinking disabled
(`enable_thinking: false`). With thinking on, you are not running it in the conditions
its results were measured in.

### Kapitsa (recommended on Apple Silicon)

```bash
kapitsa pull 27b    # once: gets the catalog DFlash2 draft model
kapitsa serve --model arthurclement/Qwen3.8-27B-opencode-adapted-mlx-6bit \
  --dflash ~/.cache/kapitsa/models/qwen3.8-27b-6bit/dflash2
```

The DFlash2 drafter is not included here; it comes with the catalog 27B model.
Use `--dflash none` to run with the MTP head only, or `--dflash none --spec 0` for plain decoding.

Then point OpenCode at `http://127.0.0.1:8000/v1` and set the model's reasoning/thinking to off.

### mlx-lm

```bash
pip install -U mlx-lm
mlx_lm.generate --model arthurclement/Qwen3.8-27B-opencode-adapted-mlx-6bit \
  --chat-template-config '{"enable_thinking": false}' \
  --prompt "Write a multi-stage Dockerfile for a FastAPI app using uv."
```

With `mlx_lm.server` or any OpenAI-compatible server, pass
`"chat_template_kwargs": {"enable_thinking": false}` in requests.

## Speed (MacBook Pro M5 Max, 40-core GPU, 64 GB, Kapitsa 0.2.40)

Short prompt (28 tokens), ~280 tokens out, thinking off:

| Decoding | tok/s | Tokens accepted per step |
|---|---|---|
| Plain | ~21 | 1 |
| MTP head | 63 | 3.7 |
| MTP + DFlash2 (catalog) | **80** | **4.2** |

In real OpenCode sessions (25–62K tokens of context, mostly tool calls): 60–100 tok/s,
4–5 tokens per step. The base model's MTP head and drafter are still well accepted after
merging the adapter. Prompt processing on cache misses (~400–560 tok/s) is the main source of
latency in long sessions.

On a 64 GB Mac, expect memory-pressure warnings beyond ~50K tokens of context with other apps open.

## What to expect

Results reported by harness-hub for the original adapter (BF16, OpenCode 2.2.7, thinking off, pass@1):

| Benchmark | Base | Adapter |
|---|---|---|
| SWE-bench Pro (hard 51 tasks) | 35.3% | 51.0% |

Likely better than the base model:
- Multi-step agentic coding inside OpenCode (fixing bugs in a repo, making tests pass, terminal work).
- Fewer tool calls and fewer loops; shorter answers.

Not expected to be better:
- Knowledge (library APIs, tool versions): the adapter teaches the harness, not new facts.
- One-shot code generation outside an agent loop.
- Tasks far from the training distribution (harness-hub saw no gain on AutomationBench).
- Very hard problems: on SWE-bench Pro, the base model *with* thinking still scores higher.
- Other harnesses (Pi, Hermes, Claude Code): use the matching harness-hub adapter instead.

## Evaluation status

Only informal checks so far (a few prompts and OpenCode sessions). Answers are coherent,
concise and end cleanly. Independent benchmark results will be added here if they are run.

## License and credits

Apache-2.0, inherited from the base model and the adapter.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Qwen team
- Adapter: [harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted) — harness-hub
- MTP head packaging and runtime: [Kapitsa](https://github.com/nace-ai/kapitsa) — Nace AI
- MLX conversion: [mlx-lm](https://github.com/ml-explore/mlx-lm)

If you are an author of one of these works and want something changed or removed, open a discussion on this repo.
