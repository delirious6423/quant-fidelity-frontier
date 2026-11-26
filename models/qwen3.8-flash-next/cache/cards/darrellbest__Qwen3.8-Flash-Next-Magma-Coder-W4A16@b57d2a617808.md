---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: vllm
tags:
- compressed-tensors
- w4a16
- int4
- qwen4exp
- coding
- agent
- tool-calling
- conversational
---

# Qwen3.8-Flash-Next-Magma-Coder-W4A16

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

**Q4-uniform as a compressed-tensors W4A16 checkpoint for vLLM.** The routed experts and the QSA attention
projections are int4 (symmetric, group 128, imatrix-weighted scale search, from the official bf16 weights).
Everything else is bf16, byte for byte the official checkpoint: shared experts, routers, indexer, GDN,
embeddings, lm_head, hyper-connection branches, the MTP head and the vision tower.

A quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), calibrated for
programming and agentic use. It isn't fine-tuned and isn't pruned: every weight comes from the original model,
and all 512 routed experts of all 48 layers are kept, plus the MTP head. The calibration is an importance matrix
(imatrix) collected on coding, tool-calling, reasoning and agent text instead of generic web text. It decides
which weights keep the most precision when the experts go to 4 bits.

## Files

- `model-000NN.safetensors` + `model.safetensors.index.json`, 168 GB in total:
  - int4 routed experts as per-expert tensors (`…mlp.experts.E.{gate,up,down}_proj.weight_packed / weight_scale /
    weight_shape`)
  - bf16 for the rest
- `model-ple-000NN.safetensors`: the 128 n-gram (PLE) table shards in bf16, about 102 GB, kept in their own files so
  vLLM's PLE CPU offload can hold them in host RAM.
- `config.json` with `quantization_config` (compressed-tensors, `pack-quantized`, int4 g128 symmetric; the same
  scheme and ignore list as [VnimanieAI/Qwen3.8-Flash-Next-W4A16](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16)),
  the tokenizer, `chat_template.jinja`, the preprocessor configs, `LICENSE`.

**Quantization error:** a spot check of one expert's int4 weights against its source gave 10.3% relative error,
which is typical for 4-bit. That's a weight-error figure only; no end-to-end KL has been measured for this
build yet.

## Run (vLLM with `qwen4_exp` support)

```bash
VLLM_PLE_CPU_OFFLOAD=1 vllm serve darrellbest/Qwen3.8-Flash-Next-Magma-Coder-W4A16 \
  --enable-expert-parallel --tensor-parallel-size 4 \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

- **N-gram offload:** `VLLM_PLE_CPU_OFFLOAD=1` keeps the 102 GB n-gram table in host RAM (about 110 GB free host
  RAM needed). The GPUs then hold about 66 GB of weights.
- **Expert parallel:** use `--enable-expert-parallel`. With group-128 quantization, tensor parallelism alone splits
  the 640-wide expert down projections below one group per rank.
- **These flags** follow the reference W4A16 checkpoint's notes; adjust the parallel sizes to your GPUs.

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
- **VnimanieAI/Qwen3.8-Flash-Next-W4A16:** the tensor set and `quantization_config` this build matches.
