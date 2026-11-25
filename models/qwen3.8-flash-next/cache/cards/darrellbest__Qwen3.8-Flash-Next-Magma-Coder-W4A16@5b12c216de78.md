---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next-Magma-Coder-W4A16

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

**Work in progress.** This repo will hold a compressed-tensors W4A16 checkpoint (int4, group 128, symmetric) for vLLM: routed experts and QSA attention in int4, the rest bf16. It uses the same coding/agent imatrix calibration as
[Qwen3.8-Flash-Next-Magma-Coder-GGUF](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-GGUF), keeps
all 512 experts and the MTP head, and involves no fine-tuning and no pruning. The weights are still being built
and uploaded.

- **Calibration sources:**
  - KannaCode agent transcripts
  - SWE-bench Lite (issues + gold patches)
  - Magma and KannaCode source code
  - NousResearch hermes-function-calling-v1
  - MATH-500
  - GSM8K
  - general chat (ultrachat_200k)
  - v2 only: the author's own private coding-agent session transcripts (secret-filtered, not distributed)
- **Not distributed:** the calibration text and the imatrix.

## Per-expert mixed precision: ~46% lower estimated error at the same size

That figure is the routing-weighted, imatrix-weighted reconstruction error of the expert weights, compared with
uniform 4-bit at equal bytes.

- **How the plan works:** it sets a type for each of the 73,728 expert projections. Bits go to the experts that
  code, tool-call and reasoning text routes to, and fewer go to experts it barely uses.
- **Verification:** end-to-end KL verification is in progress.
- **Availability:** the per-expert build ships once a runtime format supports per-expert bit widths.

### Which runtime gets what

| Runtime | Per-expert quality (the ~46%) | Per-expert size saving |
|---|---|---|
| **Any llama.cpp** (incl. LM Studio, Ollama) | **Yes, today, with `v3/`.** Each expert keeps its own tier's precision inside a Q8_0 container | No: v3 is 188 GB. On a 96 GB GPU some experts run from the CPU (6-9 tok/s measured vs about 55 for v2) |
| **Magma** (link when public) | Yes | **Yes**: per-expert bit widths stored natively, about 122 GB at v2-like speed. MLX loader landed; GGUF/CUDA tracked in issue #80 |
| Stock MLX, vLLM | Not yet. They store one bit width per layer's expert block, so they run the uniform v2 build | No |

- **Without Magma:** pick `v2/` (uniform, 131 GB, fastest) or `v3/` (per-expert quality, 188 GB, slower on a single
  96 GB GPU).
- **Magma's compact per-expert format** won't load in stock tools.
- **Coming for stock tools:** a per-layer mixed build (one width per layer, v2's size), worth an estimated ~12%
  instead of 46%, is being evaluated.

## License

Qwen Community License 1.0 (see `LICENSE`, included verbatim). In short:

- **Clause 1:** keep the copyright and license notice in all copies. A commercial product or service with more than
  100M monthly active users or US$20M monthly revenue must prominently display the model name.
- **Clause 2:** a licensee (or affiliate) running a **Model-as-a-Service** or **AI Work Assistant** business (an
  AI-coding or office-productivity assistant) needs a **separate license from Qwen** before any commercial Use.
  Purely internal use that exposes nothing to third parties is exempt.
- **No warranty:** the software and its outputs are provided "as is".

This summary isn't legal advice; the `LICENSE` text governs.

## Credits

- **Model:** Qwen3.8-Flash-Next by Qwen ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
- **Unsloth** ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)), for
  the Q8_0 build, MTP head and vision projector used in the GGUF line.
- **mlx-serve** by ddalcu ([github.com/ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve), MIT), for:
  - the `qwen4_exp` MLX checkpoint layout our MLX build follows, including `ngram_table.bin`
  - the iQ-MLX imatrix approach and KL methodology we compared against
  - the MTP + prompt-lookup drafting approach that Magma adopted
- **Layout reference:** [VnimanieAI/Qwen3.8-Flash-Next-W4A16](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16), for its tensor set and `quantization_config`.
