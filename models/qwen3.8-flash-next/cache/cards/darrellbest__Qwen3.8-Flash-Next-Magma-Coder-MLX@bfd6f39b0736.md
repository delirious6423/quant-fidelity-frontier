---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
tags:
- mlx
- imatrix
- qwen4exp
- coding
- agent
- tool-calling
- conversational
---

# Qwen3.8-Flash-Next-Magma-Coder-MLX

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

**Q4-uniform MLX pack.** Runs in stock mlx-lm / mlx-serve, and in Magma. For Magma's per-expert build, see
[Qwen3.8-Flash-Next-Magma-Coder-MLX-PerExpert](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-MLX-PerExpert).

A quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), calibrated for
programming and agentic use. It isn't fine-tuned and isn't pruned: every weight comes from the original model,
and all 512 routed experts of all 48 layers are kept, plus the MTP head. The calibration is an importance matrix
(imatrix) collected on coding, tool-calling, reasoning and agent text instead of generic web text. It decides
which weights keep the most precision when the experts go to 4 bits.

## Files

- `model-000NN.safetensors` + `model.safetensors.index.json`: 70.1 GiB in the mlx-serve `qwen4_exp` layout.
  Routed experts are 4-bit (affine, group 64, imatrix-weighted scale search, from the official bf16 weights).
  The rest is 8-bit, and the norms, convs and SSM parameters are bf16.
- `ngram_table.bin`: the n-gram table, 4-bit group 32, 29.8 GiB. It's mmapped and stays off the resident set.
- `model-vision.safetensors`: the vision tower, bf16.
- `config.json`, the tokenizer, `chat_template.jinja`, `LICENSE`.

**On a Mac** (Magma's magma-mlx): prefill about 1,300 tok/s, decode 37-57 tok/s, about 109 GB wired. That fits a
115 GB Metal working-set cap.

## Measured quality

KL divergence against a matched reference: the Q8_0 with its own Q8_0 n-gram table, so only the expert
quantization differs. Text: mixed held-out coding, tool-call, reasoning, agent and chat text, 2 chunks of 4,096
tokens, about 4K scored positions. `--parse-special` was on.

| Build | KLD mean | KLD median | KLD p99 | Top-1 same | RMS Δp | PPL ratio |
|---|---|---|---|---|---|---|
| **Q4-uniform** | **0.207 ± 0.011** | **0.0085** | **3.24** | **86.83 ± 0.53%** | **12.23%** | **0.965** |
| PerExpert (experimental) | 0.221 ± 0.012 | 0.0089 | 3.49 | 86.37 ± 0.54% | 12.72% | 1.112 |
| Per-layer mix (not published) | 0.262 ± 0.012 | 0.0108 | 3.82 | 85.83 ± 0.55% | 14.37% | 1.082 |

- **Sample size:** two chunks is a small sample, so differences of about one σ are noise. The reference path is
  deterministic: the Q8_0 scored against itself gives KLD 0.000000 and 100% top-1.
- **The n-gram table:** the Q8_0 n-gram (PLE) table alone, against the lossless BF16 table, costs KLD 0.171 and
  13 points of top-1 on our text. That's a bigger effect than any expert quantization we tried, and it's the next
  thing we're looking at.

**Smoke tests** (llama-server with `--jinja`, on one RTX PRO 6000):
- Q4-uniform: a correct `merge_intervals` with passing test asserts at 57 tok/s, and a correct structured
  `get_weather` tool call at 55 tok/s.
- PerExpert (Q8_0 container, with experts partly on the CPU): the same correct answers, at 6-9 tok/s.

**Mac (MLX Q4-uniform, Magma's magma-mlx):** prefill about 1,300 tok/s, decode 37-57 tok/s. The same Mac running
llama.cpp Metal on the GGUF: prefill 140, decode 11.4.

## The Magma-Coder family

| Repo | Build | Runs in |
|---|---|---|
| [Qwen3.8-Flash-Next-Magma-Coder-GGUF](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-GGUF) | **Q4-uniform** GGUF (recommended) | Stock llama.cpp, LM Studio, Ollama |
| [Qwen3.8-Flash-Next-Magma-Coder-MLX](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-MLX) | Q4-uniform MLX | Stock mlx-lm / mlx-serve, and Magma |
| [Qwen3.8-Flash-Next-Magma-Coder-W4A16](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-W4A16) | Q4-uniform W4A16 (compressed-tensors) | vLLM |
| [Qwen3.8-Flash-Next-Magma-Coder-GGUF-PerExpert](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-GGUF-PerExpert) | PerExpert (experimental) in a Q8_0 container | Stock llama.cpp |
| [Qwen3.8-Flash-Next-Magma-Coder-MLX-PerExpert](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-MLX-PerExpert) | PerExpert (experimental), compact | **Magma only** |
| [Qwen3.8-Flash-Next-Base-GGUF](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Base-GGUF) | The untouched 8-bit original (Unsloth's Q8_0) | Stock llama.cpp |

## Calibration

- **Corpus:** about 1.5M tokens, every document rendered through Qwen3.8's chat template (XML tool calls,
  `<tool_response>`, thinking blocks).
  - **agent, 40%:** about 60% from the author's own private coding-agent session transcripts (secret-filtered,
    not distributed), and about 40% KannaCode agent-benchmark trajectories.
  - **code, 25%:** SWE-bench Lite issues with their gold patches, plus Magma and KannaCode source files.
  - **tool calling, 15%:** NousResearch hermes-function-calling-v1.
  - **reasoning, 15%:** MATH-500 and GSM8K.
  - **general, 5%:** ultrachat_200k.
- **Not distributed:** the calibration text and the imatrix are kept private. A 10% held-out split per category
  was used only for evaluation.

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
  the Q8_0 build, MTP head and vision projector.
- **mlx-serve** by ddalcu ([github.com/ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve), MIT), for:
  - the `qwen4_exp` MLX checkpoint layout, including `ngram_table.bin`
  - the iQ-MLX imatrix approach and KL methodology we compared against
  - the MTP + prompt-lookup drafting approach that Magma adopted
- **[llama.cpp](https://github.com/ggml-org/llama.cpp):** quantization and evaluation tooling.
