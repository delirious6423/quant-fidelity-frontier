---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - gguf
  - mixture-of-experts
  - reap
  - coding
  - tool-use
  - opencode
  - unity
  - csharp
  - llama.cpp
---

# Qwen3.8-Flash-Next REAP128 86B Unity OpenCode GGUF

This is an **experimental, lossy, expert-pruned GGUF derivative** of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), built from the
`UD-IQ4_XS` GGUF published by
[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

It was created to test local Unity/C# coding and OpenCode-style tool use on a machine that
cannot keep the original model fully resident in fast memory. This is **not a conventional
fine-tune**: no weights were trained. Instead, routed experts were selected independently for
each MoE layer using activation-weighted REAP saliency collected from a small, English-only
calibration set.

## Important warning

This model is a research experiment, not a drop-in replacement for the original model. Removing
experts causes irreversible capability loss. The calibration set is small and strongly biased
toward Unity, C#, repository navigation, code review, and tool-calling workflows. General
knowledge, factual reliability, multilingual ability, reasoning, and out-of-domain performance
may be substantially worse. No safety evaluation or broad benchmark suite has been run.

## Model summary

| Property | Value |
|---|---:|
| Source architecture | Qwen3.8-Flash-Next |
| Source routed experts per MoE layer | 512 |
| Retained routed experts per MoE layer | 128 |
| Routed experts selected per token | 10 |
| MoE layers processed | 48 |
| Stored parameter values after pruning | approximately 86.3B |
| Language-model parameter values after pruning | approximately 35B |
| Activated language-model parameters per token | approximately 6B |
| Sparse n-gram embedding table | approximately 51B |
| Quantization | UD-IQ4_XS |
| GGUF file size | 48,853,994,112 bytes (45.50 GiB) |

The **86B** in the repository name refers to the approximate total number of stored parameter
values, including the sparse n-gram embedding table. It does not mean that 86B parameters are
activated for every token. Likewise, this model should not be described as “A58B.”

## Expert-selection method

The retained experts were ranked separately in every layer. For each routed expert, the observer
accumulated the mean of:

```text
router_gate_weight * L2_norm(expert_output)
```

The top 128 experts by this score were retained in each layer. This was not random pruning and it
was not a simple “first 128 experts” selection.

Calibration details:

- 50 English prompts designed for an OpenCode-like coding-agent workload
- 30 prompts emphasizing native tool calls and repository operations
- 20 prompts emphasizing direct Unity/C# implementation, debugging, and review
- simulated file listing, text search, file reading, shell, patching, and test workflows
- 5,120 observer tokens processed
- 51,200 routed-expert observations per layer
- between 408 and 491 of the original 512 experts observed per layer

The exact retained expert indices and the complete saliency output are included in this repository.

## Intended use

- Experimental local inference with `llama.cpp`
- Unity and C# coding assistance
- OpenCode-style repository exploration and tool use
- Studying domain-biased MoE expert pruning

Do not rely on this model for high-stakes decisions or factual answers without independent
verification.

## Running with llama.cpp

The following configuration was used on the test machine. It favors a 32K context and places MoE
work on system RAM while offloading as much as possible to the GPU:

```powershell
.\llama-server.exe `
  -m ".\Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-UD-IQ4_XS.gguf" `
  --alias "qwen38-reap128-unity-opencode" `
  --host 0.0.0.0 `
  --port 8080 `
  --parallel 1 `
  -ngl all `
  --fit off `
  --n-cpu-moe 32 `
  -c 32768 `
  --flash-attn on `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  -t 16 `
  -tb 16 `
  -b 2048 `
  -ub 512 `
  --no-mmap `
  --jinja
```

Hardware-dependent options such as `--n-cpu-moe`, batch sizes, thread counts, and GPU offload
should be tuned for your system. `--no-mmap` requires enough physical RAM for the CPU-resident
portion; omitting it may allow memory mapping and SSD-backed paging, but can be much slower.

The server exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.

## Included artifacts

- `Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-UD-IQ4_XS.gguf`: pruned model
- `experts.json`: retained expert indices for every layer
- `calibration/opencode-50-prompts.jsonl`: English calibration prompts
- `calibration/opencode-50-results.json`: recorded model outputs and simulated tool calls
- `calibration/opencode-50-transcript.txt`: observer input transcript
- `calibration/opencode-50-reap.json`: per-layer REAP measurements and scores
- `start-server.ps1`: example Windows launcher

## Validation notes

In limited local checks, the model produced native tool calls and completed a two-step
`list_files` then `search_text` chain, and it generated usable Unity/C# code. However, it also
showed clear general-quality degradation, including a bad factual response, and one direct
no-tools request triggered a native PEG-parser error in the tested `llama.cpp` build. These
observations are anecdotal and are not benchmark results.

## Attribution and license

The base model is by the Qwen team. The source quantization is from Unsloth. This repository is
distributed under the `qwen-community-1.0` license inherited from the base model. Review the base
model license and acceptable-use terms before downloading or redistributing this derivative.
