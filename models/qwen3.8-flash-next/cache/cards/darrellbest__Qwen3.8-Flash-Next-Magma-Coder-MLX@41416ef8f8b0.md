---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next-Magma-Coder-MLX

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

**`v2/` runs in stock mlx-lm / mlx-serve; `v3/` (per-expert tiers) needs Magma's MLX loader.** **Work in progress.** This repo will hold an MLX pack for [mlx-serve](https://github.com/ddalcu/mlx-serve) in the iQ-MLX 4.7 bpw layout: routed experts 4-bit, the rest 8-bit, with `ngram_table.bin`. It uses the same coding/agent imatrix calibration as
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

## Per-expert mixed precision: ~46% lower estimated error at the same size, all 512 experts kept in every layer

That figure is the routing-weighted, imatrix-weighted reconstruction error of the expert weights, compared with
uniform 4-bit at equal bytes.

- **How the plan works:** it sets a type for each of the 73,728 expert projections. Bits go to the experts that
  code, tool-call and reasoning text routes to, and fewer go to experts it barely uses.
- **Verification:** end-to-end KL verification is in progress.
- **Availability:** the per-expert build ships once a runtime format supports per-expert bit widths.

### Per layer: all 512 experts kept, each at its own precision

Every layer keeps all 512 routed experts. What varies is the precision of each one. The table counts expert
projections by bit width for every layer.

- **gate + up:** 1,024 per layer (512 experts × 2 projections).
- **down:** 512 per layer. The 640-wide rows set its floor at 4 bits.
- **The pattern:** early layers lean on 3-4 bits. Late layers, which code and tool traffic stresses most, get many
  5-6-bit experts.
- **2-bit:** only experts the calibration traffic almost never routes to get 2 bits.

<details><summary>Per-layer table (48 layers)</summary>

| Layer | gate + up (count × bits) | down (count × bits) |
|---|---|---|
| 0 | 16×2b · 496×3b · 511×4b · 1×5b | 512×4b |
| 1 | 2×2b · 599×3b · 411×4b · 12×5b | 512×4b |
| 2 | 2×2b · 610×3b · 406×4b · 6×5b | 512×4b |
| 3 | 494×3b · 404×4b · 124×5b · 2×6b | 512×4b |
| 4 | 403×3b · 488×4b · 131×5b · 2×6b | 512×4b |
| 5 | 469×3b · 504×4b · 51×5b | 512×4b |
| 6 | 330×3b · 532×4b · 160×5b · 2×6b | 512×4b |
| 7 | 329×3b · 575×4b · 120×5b | 512×4b |
| 8 | 411×3b · 475×4b · 138×5b | 512×4b |
| 9 | 410×3b · 471×4b · 143×5b | 512×4b |
| 10 | 390×3b · 543×4b · 91×5b | 512×4b |
| 11 | 446×3b · 467×4b · 111×5b | 512×4b |
| 12 | 382×3b · 517×4b · 125×5b | 512×4b |
| 13 | 420×3b · 539×4b · 65×5b | 512×4b |
| 14 | 428×3b · 461×4b · 135×5b | 512×4b |
| 15 | 350×3b · 413×4b · 231×5b · 30×6b | 512×4b |
| 16 | 381×3b · 397×4b · 238×5b · 8×6b | 512×4b |
| 17 | 413×3b · 432×4b · 179×5b | 512×4b |
| 18 | 325×3b · 523×4b · 176×5b | 512×4b |
| 19 | 319×3b · 477×4b · 224×5b · 4×6b | 512×4b |
| 20 | 176×3b · 482×4b · 348×5b · 18×6b | 512×4b |
| 21 | 2×2b · 168×3b · 563×4b · 284×5b · 7×6b | 512×4b |
| 22 | 211×3b · 507×4b · 303×5b · 3×6b | 512×4b |
| 23 | 94×3b · 495×4b · 405×5b · 30×6b | 512×4b |
| 24 | 375×3b · 489×4b · 160×5b | 512×4b |
| 25 | 401×3b · 462×4b · 161×5b | 512×4b |
| 26 | 433×3b · 460×4b · 131×5b | 512×4b |
| 27 | 465×3b · 424×4b · 135×5b | 512×4b |
| 28 | 2×2b · 377×3b · 475×4b · 170×5b | 512×4b |
| 29 | 367×3b · 480×4b · 177×5b | 512×4b |
| 30 | 2×2b · 293×3b · 461×4b · 265×5b · 3×6b | 512×4b |
| 31 | 237×3b · 435×4b · 302×5b · 50×6b | 512×4b |
| 32 | 273×3b · 376×4b · 341×5b · 34×6b | 512×4b |
| 33 | 269×3b · 423×4b · 330×5b · 2×6b | 512×4b |
| 34 | 2×2b · 237×3b · 505×4b · 276×5b · 4×6b | 511×4b · 1×5b |
| 35 | 194×3b · 418×4b · 395×5b · 17×6b | 512×4b |
| 36 | 2×2b · 158×3b · 278×4b · 475×5b · 111×6b | 510×4b · 2×5b |
| 37 | 4×2b · 160×3b · 304×4b · 471×5b · 84×6b · 1×8b | 511×4b · 1×5b |
| 38 | 6×2b · 161×3b · 229×4b · 522×5b · 105×6b · 1×8b | 509×4b · 3×5b |
| 39 | 94×3b · 358×4b · 449×5b · 123×6b | 497×4b · 15×5b |
| 40 | 90×3b · 338×4b · 459×5b · 137×6b | 512×4b |
| 41 | 2×2b · 98×3b · 260×4b · 502×5b · 162×6b | 511×4b · 1×5b |
| 42 | 28×2b · 154×3b · 177×4b · 510×5b · 155×6b | 495×4b · 17×5b |
| 43 | 6×2b · 157×3b · 306×4b · 443×5b · 112×6b | 511×4b · 1×5b |
| 44 | 18×2b · 236×3b · 188×4b · 427×5b · 152×6b · 3×8b | 511×4b · 1×5b |
| 45 | 14×2b · 227×3b · 183×4b · 400×5b · 195×6b · 5×8b | 506×4b · 6×5b |
| 46 | 14×2b · 227×3b · 202×4b · 377×5b · 193×6b · 11×8b | 504×4b · 7×5b · 1×8b |
| 47 | 22×2b · 225×3b · 221×4b · 423×5b · 126×6b · 7×8b | 480×4b · 28×5b · 4×8b |

</details>

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
