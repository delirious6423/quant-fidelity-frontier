---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language: [en]
tags:
  - gguf
  - quantized
  - imatrix
  - IQ2_M
  - 2-bit
  - llama-cpp
  - gemma4
  - moe
  - sparse-moe
  - agentic
  - coding
  - conversational
  - text-generation
---

# Gemma 4 26B-A4B IT — IQ2_M (domain imatrix)

Custom IQ2_M quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
with a domain-mixed importance matrix (45% code, 45% agentic, 10% general).

- **Size**: 9.7 GB (smaller than the 31B dense at 10.4 GB)
- **BPW**: 3.29
- **Architecture**: Gemma 4 MoE — 128 experts, 8+1 active per token, 25.2B total / 3.8B active

## Benchmark results (reasoning OFF)


> **Note on BFCL scores**: All BFCL scores reported here use our internal simplified evaluation 
> (single-function-call subset with custom prompt/scoring), NOT the official Berkeley Function Calling 
> Leaderboard methodology. Our scores are **not directly comparable** to the official leaderboard. 
> We are working on running the official BFCL evaluation for comparable numbers.


| Model | Size | Active | HumanEval+ | MBPP+ | BFCL v3 | NL2Bash F1 |
|-------|------|--------|------------|-------|---------|------------|
| [Gemma 4 31B IQ2_M](https://huggingface.co/KikoCis/gemma-4-31b-it-IQ2_M-GGUF) | 10.4 GB | 31B | **88.41%** | **82.01%** | 92.25% | **84.71%** |
| **This model (26B MoE IQ2_M)** | **9.7 GB** | 3.8B | 80.49% | 77.78% | **92.75%** | 79.64% |
| [Qwen3.6 IQ2_M](https://huggingface.co/KikoCis/Qwen3.6-35B-A3B-IQ2_M-GGUF) | 11.1 GB | 3B | 80.49% | 78.31% | 94.75% | 81.63% |
| Gemma 4 E4B Q8_0 | 7.8 GB | 4.5B | 73.78% | 73.28% | 93.75% | 79.75% |

### Key findings

- **BFCL 92.75% — beats the 31B dense** (92.25%) with 0.7 GB less and 8x fewer active params
- **HumanEval+ 80.49%** — ties Qwen3.6 IQ2_M at comparable size
- MoE expert isolation provides natural quantization resilience, as predicted by our [specialization boundary study](https://kikocisbot.github.io/gemma4-31b-study/)
- All benchmarks run with reasoning OFF; thinking mode expected to add +5-15pt on code tasks

## Quickstart

```bash
huggingface-cli download KikoCis/gemma-4-26B-A4B-it-IQ2_M-GGUF gemma4-26b-a4b-IQ2_M.gguf --local-dir .

# Without thinking
llama-cli -m gemma4-26b-a4b-IQ2_M.gguf -ngl 99 --ctx-size 8192 --temp 0.1 \
  -p "Write a Python function to merge two sorted lists"

# With thinking (recommended for complex tasks)
llama-cli -m gemma4-26b-a4b-IQ2_M.gguf -cnv -ngl 99 --ctx-size 8192 \
  --reasoning on --reasoning-budget 1024
```

## Files

- `gemma4-26b-a4b-IQ2_M.gguf` — quantized weights (9.7 GB)
- `gemma4-26b-a4b-domain.imatrix` — importance matrix

## Related

- [Full study: The Specialization Boundary](https://kikocisbot.github.io/gemma4-31b-study/)
- [Paper (PDF)](https://github.com/KikoCisBot/gemma4-31b-study/blob/main/paper.pdf)

## License

Apache 2.0.


---

## Real-World Agent Test Warning (April 2026)

> **Benchmark scores do not predict agent capability.** In Docker-based autonomous testing,
> fine-tuned E4B models (95% BFCL) scored **0/10** while the unfine-tuned base scored **6/10**.
> Fine-tuning for BFCL destroyed general reasoning (error recovery, strategy adaptation, anti-repetition).
> Fine-tuned E4B models have been withdrawn.
>
> For autonomous agent tasks, use the base Gemma 4 model or a larger model at higher BPW.
> See: [The Benchmark Trap — Full Study](https://kikocisbot.github.io/gemma4-31b-study/#agent-test)

