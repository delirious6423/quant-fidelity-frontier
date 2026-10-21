---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - atlassian
  - forge
  - jira
  - confluence
  - jira-service-management
  - leanzero
language:
  - en
---

# Qwen3.8-27B-Atlassian-Q6_K-GGUF

**Qwen3.8-27B that knows Forge, Jira, Confluence and JSM: an 8-bit MLX model with its MTP speculative-decoding head, trained by LeanZero on Forge apps, the Atlassian documentation, the OpenAPI specs and community answers.**

This is the **Q6_K GGUF** of the LeanZero Atlassian model, release v0.5, for llama.cpp, LM Studio's GGUF engine, Ollama and every other llama.cpp-based runtime. The MLX release of the same adapter, the full write-up and every round's evidence are on **[leanzero.net/portfolio/atlassian-models](https://leanzero.net/portfolio/atlassian-models)**; the MLX repos are linked under Family.

Built by [LeanZero](https://leanzero.net), the people behind [CogniRunner](https://leanzero.net/portfolio/cognirunner), [Sentinel Vault](https://leanzero.net/portfolio/sentinel-vault), [LeanZero Management](https://leanzero.net/portfolio/leanzero-management) and [Atlassian migrations](https://leanzero.net/services/atlassian-migrations).

## How this file was made, and how we know it matches the MLX release

A GGUF cannot be scored against an MLX model with one tool, so parity is established as a chain of three measured links, each inside the tool that can measure it:

1. **Merged bf16 = the model.** The v0.5 adapter was merged into the original `Qwen/Qwen3.8-27B` bf16 weights (not into the MLX 8-bit base), 176 modules, and the merged model reproduced the MLX base-plus-adapter greedily on 16 of 20 prompts with a mean common-prefix fraction of 0.89 (the remainder diverge at near-tie tokens, the same order as the 8-bit-vs-bf16 difference of the base itself).
2. **bf16 GGUF = merged bf16, across runtimes.** Converted losslessly with llama.cpp's converter (MTP draft tensors kept). llama.cpp greedy continuations matched mlx-lm on 17 of 20 identical rendered prompts, mean common-prefix fraction 0.94. Then the task probes were run through llama-server on the Q8_0 file: see Measured.
3. **Q6_K = bf16 GGUF, inside llama.cpp.** `llama-perplexity --kl-divergence` against the bf16 GGUF's own logits (40 chunks of 2048 tokens of held-out Atlassian text): mean KLD 0.00306, maximum 6.626617, same top-1 token 98.5. The Q8_0 file measured 0.000842.

## Measured through llama-server

- **Apps that compile against the real types, 35 briefs, 3 samples each, thinking off:** 28.7 of 35 apps, 30.3 of 35 manifests. The MLX release of the same adapter measured 30.0 and 31.3 on the same briefs and rule (its predecessor v0.4: 22.7 and 30.7).
- **Identifier recall, thinking on (Q8_0 through llama-server):** 85% on the pre-2026 slice and 15% on the post-April slice. MLX release: 85% and 15%.
- **Looping battery through llama-server (40 prompts per leg):** think_official: 0 loops of 40, non-termination 32%; think_t06: 0 loops of 40, non-termination 32%; greedy: 1 loops of 40, non-termination 32%; instruct_nopenalty: 0 loops of 40, non-termination 12%. Rule of the MLX release: loops at or under the untuned base on every leg.
- **Speculative decoding on Apple Silicon (llama.cpp Metal, M3 Ultra):** Engine decode rate on the long runs: 24.2 tokens/s with the draft head (105 app answers, acceptance 0.629) against 20.4 without (27 identifier answers), single stream. On a 5-prompt greedy wall-clock test the draft head gave no gain: MTP none: 20.9 tok/s over 1280 tokens (5 prompts, 256 max, temp 0, wall incl. prefill); MTP draft-mtp: 19.6 tok/s over 1280 tokens (5 prompts, 256 max, temp 0, wall incl. prefill). Every published llama.cpp MTP figure for this model family is CUDA; on Metal treat the draft head as a small gain on long code answers and none on short ones. Batched serving runs at about 9 tokens/s per stream at 4 concurrent (see Limits).
- **Thinking-off switch:** `chat_template_kwargs.enable_thinking=false` was honoured by this build's template handling (no reasoning emitted, correct module named).

## How to run it

**llama.cpp** (b10330 or later; keep the multi-token-prediction draft, it is in this file):

```
llama-server -m Qwen3.8-27B-Atlassian-Q6_K.gguf -ngl 99 -c 40960 --jinja --spec-type draft-mtp
```

Thinking on (identifier questions, reasoning tasks): add `--reasoning-format deepseek` and the answer carries `reasoning_content`. Thinking off (app writing, the mode every app number above was measured in): add `--reasoning-budget 0`. 

**LM Studio**: place the file under your models directory; it loads with the GGUF (llama.cpp) engine.

**Ollama**: `FROM ./Qwen3.8-27B-Atlassian-Q6_K.gguf` in a Modelfile, then `ollama create`.

## Sampling

Thinking: temperature 1.0, top_p 0.95, top_k 20. Instruct: temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5. Trained with `reasoning_effort=medium`, `preserve_thinking=false`.

## Limits

- Numbers here are llama.cpp measurements on the Q6_K file; they are comparable with the MLX card's numbers only through the three links above, never directly. Single-pass app counts move by about 3 of 35 between runs; the 3-sample mean is what is reported.
- The post-April identifier slice (13 questions about current package majors) is weak in every capacity adapter of this family, including the MLX release; do not ask it which major of a Forge package is current.
- llama.cpp batched throughput on this hybrid architecture is far below dense models (upstream issue 20006): treat this file as a single-user artefact; multi-agent swarms run the MLX release.
- A recurrent-state cache defect can reprocess the full prompt on every turn (upstream issue 20225): multi-turn time-to-first-token grows with context.
- Q4-class GGUFs are not published: on this adapter they lose the measured app gain (LeanZero calibration, 2026-09-05).

## Family

- MLX (Rapid-MLX, LM Studio MLX engine, mlx-lm): [Qwen3.8-27B-Atlassian-Q8-mlx](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q8-mlx), [Q6](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q6-mlx), [Q4](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q4-mlx), [LoRA adapter](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-lora), [PEFT adapter](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-lora-peft).
- GGUF: [Qwen3.8-27B-Atlassian-Q8_0-GGUF](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q8_0-GGUF) (the parity quant, same bit width as the MLX release) and [Qwen3.8-27B-Atlassian-Q6_K-GGUF](https://huggingface.co/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q6_K-GGUF) (the practical one).
- 9B family (MLX): [Qwen3.5-9B-Atlassian-Q8-mlx](https://huggingface.co/Mihai-LeanZero/Qwen3.5-9B-Atlassian-Q8-mlx).

## Licence and provenance

Apache-2.0 (inherits the base). Same training data and gates as the MLX release: see the [LeanZero write-up](https://leanzero.net/portfolio/atlassian-models). Quantised with llama.cpp `llama-quantize` using the Unsloth importance matrix for this base model.
