---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
- gguf
- qwen4exp
- imatrix
- coding
- agent
- tool-calling
---

# Qwen3.8-Flash-Next-Magma-Coder-GGUF

A GGUF quant of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), calibrated for
programming and agentic use.

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

- **What it is:** a quantization. Every weight comes from the original model.
- **What it isn't:** no fine-tuning and no pruning. All 512 routed experts of all 48 layers are kept, and the MTP
  (multi-token prediction) head ships alongside.
- **What makes it "Coder":** the importance matrix (imatrix). It decides which weights keep the most precision
  when the routed experts go to 4 bits, and it was collected on coding, tool-calling, reasoning and agent text
  instead of generic web text.

**Every file in this repo runs in stock llama.cpp (LM Studio, Ollama); only the upcoming compact per-expert build needs Magma.** Recommended today: `v2/` (uniform 4-bit experts).

## Per-expert mixed precision: ~46% lower estimated error at the same size, all 512 experts kept in every layer

That figure is the routing-weighted, imatrix-weighted reconstruction error of the expert weights, compared with
uniform 4-bit at equal bytes.

- **How the plan works:** it sets a type for each of the 73,728 expert projections (512 experts × 3 projections ×
  48 layers). Bits go to the experts that code, tool-call and reasoning text routes to, and fewer go to experts it
  barely uses.
- **Verification:** end-to-end KL verification is in progress.
- **Availability:** the per-expert build ships once a runtime format supports per-expert bit widths. The files in
  `v1/` and `v2/` below are the uniform builds and don't contain this allocation.

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

## Files

| Folder | Build | Size | Status | Runs in |
|---|---|---|---|---|
| `v3/` | **Per-expert mixed precision** (IQ2_S to Q8_0 per expert, by importance), stored in a Q8_0 container, plus `expert-tiers.json` (the per-expert type plan) | 191 GB | Per-expert quality at Q8_0 size; see below | llama.cpp |
| `v2/` | **Uniform** build, final calibration: imatrix over the full 1.5M-token corpus (367 × 4096-token chunks). No per-expert allocation | 130.9 GB (121.9 GiB, 5.78 bpw) | **Recommended** | llama.cpp |
| `v1/` | **Uniform** build, preview: imatrix on an earlier corpus without the private session transcripts, first 220 of 367 chunks | 130.9 GB | Superseded by v2 | llama.cpp |
| `mmproj-F16.gguf` | Vision projector (F16, Unsloth's) | 0.9 GB | For image input | llama.cpp |

`v1/` and `v2/` each hold 6 GGUF shards plus `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, Unsloth's Q8_0 MTP head
(untuned).

### `v3/`: the per-expert model in a Q8_0 container

v3 is the per-expert mixed-precision model. Each of the 73,728 routed-expert projections was quantized to its own
type, from 2 to 8 bits per expert, by its importance to code, tool-call and reasoning traffic. The tiers:

| Tier | Experts |
|---|---|
| IQ2_S (unused experts) | 144 |
| IQ3_S / Q3_K | 14,962 |
| Q4_K | 20,045 |
| IQ4_NL | 24,488 |
| Q5_K / Q5_0 / Q5_1 | 12,185 |
| Q6_K | 1,871 |
| Q8_0 | 33 |

- **Storage:** the result was dequantized back and stored as Q8_0, so it **loads in any llama.cpp with
  `qwen4exp` today**.
- **What you get:** the per-expert quality, but not the size saving. It's 188 GB, so on a 96 GB GPU keep some
  experts on the CPU (`--n-cpu-moe 17`); that smoke-tested at 6-9 tok/s.
- **Compact version:** a Magma-native v3 at about 122 GB, the same size as v2, follows with Magma support
  (issue #80).
- **The ~46% figure:** that's the estimated error reduction at equal bytes. KL verification is pending.
- **`expert-tiers.json`:** the type chosen for every (layer, expert, projection), plus the allocation's budget
  scores. It holds no corpus text.
- **Smoke test** (code prompt + tool call): passed, with the same correct `merge_intervals` and `get_weather` call
  as v2.

### What's in a build (v1, v2)

| Tensors | Type |
|---|---|
| Routed experts, gate/up (512 × 48 layers) | **Q4_K**, imatrix |
| Routed experts, down | **IQ4_NL**, imatrix. Its 640-wide rows rule out the 256-block K-quants |
| Attention, GDN (linear attention), hyper-connections, indexer, shared experts, router, embeddings, lm_head | Q8_0, unchanged |
| N-gram (PLE) embedding table, 51B params | Q8_0, unchanged |
| Norms, convs, SSM parameters | F32 |

- **Source:** quantized with `llama-quantize --imatrix` from the Q8_0 build. That Q8_0 is
  [Unsloth's](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), mirrored untouched in
  [darrellbest/Qwen3.8-Flash-Next-Base-GGUF](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Base-GGUF).
- **Size:** about 54 GB of each build is the n-gram table, a CPU-side lookup that doesn't need to sit in VRAM.

## Calibration

- **Corpus:** about 1.5M tokens, in the categories below. Every document was rendered through Qwen3.8's own chat
  template: tool calls in its XML `<tool_call><function=…>` format, tool results as `<tool_response>`, and
  thinking blocks.
  - **agent, 40%:** real coding-agent sessions.
    - About 60% from the author's own private coding-agent session transcripts (secret-filtered, not distributed), across ten software projects. A secret filter dropped API keys, tokens, password
      assignments, long hex/base64 blobs, `.env` contents and private-folder paths, and no single project
      exceeds 30% of the category.
    - About 40% from KannaCode agent-benchmark trajectories.
  - **code, 25%:** SWE-bench Lite issues with their gold patches, and C++/CUDA/Python source files framed as
    review, refactor and bug-fix requests.
  - **tool calling, 15%:** NousResearch hermes-function-calling-v1.
  - **reasoning, 15%:** MATH-500, GSM8K, and long reasoning traces from agent runs.
  - **general, 5%:** ultrachat_200k.
- **Held-out split:** 10% per category, split by conversation, kept for evaluation and never used in the imatrix.
- **Collection:** `llama-imatrix --parse-special`, with per-expert counts. The reference was the model with its
  routed experts at Q5 and everything else at Q8_0, fully on the GPU.
- **Privacy:** the corpus, the transcripts and the imatrix are **not published**. Only the quantized weights are.
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

## How to run

You need a llama.cpp with the `qwen4exp` architecture (Qwen3.8-Flash-Next). Keep the n-gram table on the CPU:

```bash
llama-server -m v2/Qwen3.8-Flash-Next-Magma-Coder-v2-Q4K-00001-of-00006.gguf \
  -ngl 999 -ot 'per_layer_token_embd\.weight=CPU' -fa on --jinja -c 32768
```

- **VRAM:** about 72 GB with everything except the n-gram table on the GPU.
- **Smaller GPUs:** add `--n-cpu-moe N` to keep some layers' experts on the CPU.
- **Chat template:** `--jinja` uses the template embedded in the GGUF, which carries Qwen3.8's tool-call format.
- **MTP:** load `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` as the draft model in builds that support Qwen's MTP
  head.

## Test results (as measured)

All tests ran on one RTX PRO 6000 (96 GB) at `-c 4096`, with `--parse-special`.

### Smoke test

The llama-server setup above, with one coding prompt and one tool-call prompt:

| Build | Code prompt (`merge_intervals` + asserts) | Tool-call prompt (`get_weather`) |
|---|---|---|
| v1 | Correct function and asserts, 64 tok/s decode | Correct structured call `get_weather(city="Lisbon", unit="celsius")`, 53 tok/s |
| v2 | Correct function with two passing test asserts, 57 tok/s | Same correct call, 55 tok/s |

The decode speeds were measured while other CPU jobs ran on the host, so they're lower bounds.

### KL divergence

Measured against Q8_0 with the lossless BF16 n-gram table swapped in, on held-out agent text: 2 chunks of 4096
tokens, about 4K scored positions.

| Build | KLD mean | KLD median | KLD p99 | Top-1 same | PPL ratio |
|---|---|---|---|---|---|
| v1 | 0.516 ± 0.020 | 0.055 | 6.18 | 76.87 ± 0.66% | 1.068 |
| v2 | 0.499 ± 0.020 | 0.058 | 5.86 | 77.14 ± 0.66% | 1.062 |

- **v1 vs v2:** v2 is ahead on mean KLD, p99, top-1 and PPL, but the differences are within about 1σ on this
  little data.
- **Reading the absolute numbers:** the Q8_0 itself, with its own Q8_0 n-gram table, scores KLD 0.171 and top-1
  86.7% against this reference (1 chunk). So the absolute figures include a floor from either n-gram-table
  precision or reference-run noise, not just expert quantization.
- **Next:** a matched reference (the Q8_0 with its own n-gram table) on mixed code/tools/reasoning/agent text is
  planned. This section gets updated when it's done.

## Per-expert mixed precision: details

This section explains the headline above. The plan gives more bits to the experts that coding and tool-calling traffic actually routes to, and fewer
to the experts it barely touches. Most go to Q4_K, about 15k to IQ3_S, about 12k to Q5_K and about 1.9k to Q6_K,
at the same total size.

- **Proxy result:** the plan cuts the routing-weighted, imatrix-weighted expert error by 46% at the same bytes.
- **Not yet confirmed:** that is a proxy, not yet checked end to end. No per-expert-tier weights will be
  published until a KL comparison against a matched reference confirms it.
- **Format:** standard GGUF stores one type per merged expert tensor, so per-expert tiers need a custom format,
  planned for the Magma engine.

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
- **Q8_0 source, MTP head GGUF and vision projector:** [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
- **Quantization tooling:** [llama.cpp](https://github.com/ggml-org/llama.cpp).
- **mlx-serve** by ddalcu ([github.com/ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve), MIT), for:
  - the `qwen4_exp` MLX checkpoint layout our MLX build follows, including `ngram_table.bin`
  - the iQ-MLX imatrix approach and KL methodology we compared against
  - the MTP + prompt-lookup drafting approach that Magma adopted
