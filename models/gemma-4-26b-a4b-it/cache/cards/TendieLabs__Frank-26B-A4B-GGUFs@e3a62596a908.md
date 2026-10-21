---
license: apache-2.0
datasets:
- microsoft/rStar-Coder
- nvidia/OpenCodeInstruct
- m-a-p/Code-Feedback
- open-r1/codeforces-cots
- m-a-p/CodeFeedback-Filtered-Instruction
- ise-uiuc/Magicoder-OSS-Instruct-75K
- ise-uiuc/Magicoder-Evol-Instruct-110K
- layoric/tiny-codes-alpaca-csharp
- cognitivecomputations/dolphin-coder
- gretelai/synthetic_text_to_sql
- bigcode/commitpackft
- Dahoas/code-review-instruct-critique-revision-python
- newfacade/LeetCodeDataset
base_model:
- google/gemma-4-26B-A4B-it
tags:
- code
- gemma4
- moe
- opencode
- nato-aligned
language:
- en
pipeline_tag: text-generation
---

# Frank 26B-A4B GGUFs (imatrix-calibrated)

**imatrix-calibrated GGUF quantizations of [TendieLabs/Frank-26B-A4B](https://huggingface.co/TendieLabs/Frank-26B-A4B), a fine-tuned Gemma 4 26B-A4B-it for code generation, debugging, and code review.**

> All quantizations in this repo are **imatrix-calibrated** using wikitext-2. This is especially important for MoE models (128 experts) where naive quantization can damage expert routing gate weights. imatrix calibration preserves precision on high-importance weights and prevents routing degradation at lower bit depths.

Frank is a TendieLabs fine-tune of Google's [Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), trained on ~33K coding examples spanning C#, Python, multi-turn debugging, and general reasoning. It is designed to serve as a local coding assistant via [OpenCode](https://github.com/opencode-ai/opencode) or any OpenAI-compatible client in environments that require US/NATO-aligned model origins and prohibit the use of Chinese-origin models (Qwen, DeepSeek, Yi, etc.). For full details, see the [safetensors repo](https://huggingface.co/TendieLabs/Frank-26B-A4B).

## Key Features

- **Mixture-of-Experts (MoE)**: 25.2B total parameters, ~3.8B active per forward pass (128 experts). Runs at near-4B speed with 26B knowledge depth.
- **Learned task discrimination**: Frank reasons through complex problems (math, debugging, code review) but skips unnecessary thinking on straightforward code generation. This results in 10-12x faster inference on code tasks compared to base Gemma 4 26B-A4B while maintaining full reasoning capability on tasks that need it.
- **NATO-aligned**: US-origin base model (Google), Apache 2.0 license. Suitable for deployment in environments with supply chain policies restricting non-NATO AI model origins.
- **Patched chat template**: Includes Google's latest `chat_template.jinja` and `tokenizer_config.json` for stable thinking mode behavior. Fixes the known ghost thought channel issue on Gemma 4 26B-A4B.

## Recommended Inference Parameters

Gemma 4 models require higher temperature than the industry default for code generation. Using low temperature (0.2-0.3) causes reasoning mode to loop indefinitely on complex prompts. Use Google's recommended parameters:

| Parameter | Value |
|-----------|-------|
| **temperature** | **1.0** |
| **top_p** | **0.95** |
| **top_k** | **65** |
| **max_tokens** | 32768+ (for thinking headroom) |
| **enable_thinking** | true (default) |

> **Important**: Do NOT use temperature below 0.8 with thinking mode enabled. Low temperature causes Gemma 4's chain-of-thought reasoning to enter verification loops that consume the entire token budget without producing output. This is a documented behavior of the Gemma 4 architecture, not a Frank-specific issue.

## Benchmarks

### VBA-to-Python Conversion Eval

Five realistic VBA report conversions (simple to complex) scored on a 9-point objective rubric (produces output, valid Python, correct imports, single class, single run method, no markdown fences, return statement, no Outlook COM, plus hallucinated API detection).

All runs: temp 1.0, top-p 0.95, top-k 65, max_tokens 65536, thinking enabled. Frank models tested on NVIDIA A100 80GB at FP16.

| Model | Test 1 (easy) | Test 2 (medium) | Test 3 (hard) | Test 4 (complex) | Test 5 (medium) | **Avg** | **Total Time** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base Gemma 4 26B-A4B | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **9.0/9** | ~430s |
| **Frank (this model)** | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | **9.0/9** | **~40s** |

Frank matches base Gemma 4 on quality while being **10x faster** on code conversion tasks due to learned task discrimination (skips reasoning when it is not needed).

### Reasoning Verification

To verify that Frank retains full reasoning capability, it was tested on 5 reasoning tasks with thinking enabled:

| Task | Reasoning Used | Correct | Notes |
|------|:-:|:-:|---|
| Math trick ("all but 9 die") | 999 chars | Yes | Identified the linguistic trick, explained the common pitfall |
| Python debugging (fibonacci n-3 bug) | 1,718 chars | Yes | Found exact bug, showed wrong output, provided corrected code |
| Widget puzzle (5 machines, 5 min) | 2,412 chars | Yes | Correct answer (5 min), called out the common wrong answer (100 min) |
| C# code review (EF Core filtering) | 3,301 chars | Yes | Identified critical in-memory filtering issue, naming conventions, DateTime usage |
| Bat and ball ($1.10 total) | 733 chars | Yes | Correct ($0.05), full algebra with verification |

Frank reasons when reasoning is needed and skips it when it is not. This is learned behavior from the training data, not prompt engineering.

### Original Eval (Historical Context)

Frank was initially evaluated at incorrect inference parameters (low temperature, low max_tokens) which caused Gemma 4's thinking mode to loop and produce empty responses. The model was shelved based on these results. Retesting with correct parameters (temp 1.0, proper thinking mode handling) revealed the training was successful all along.

| Model | Original Eval (temp ~0.2) | Corrected Eval (temp 1.0) |
|-------|:---:|:---:|
| Base | 4.14 avg, 1 empty | 9.0/9 (baseline) |
| Frank | 3.53 avg, 5 empties | **9.0/9, 10x faster** |

The root cause of the original poor scores: Gemma 4's chain-of-thought reasoning loops at low temperature. Training reinforced reasoning behavior, causing the trained version to loop more frequently than base at low temperature, producing more empty responses. With correct temperature (1.0), the training effect is beneficial (faster, discriminative reasoning).

## Training Details

- **Base model**: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Method**: LoRA 16-bit (QLoRA is not recommended for MoE, breaks routing)
- **Framework**: [Unsloth](https://unsloth.ai/) + Transformers
- **Dataset**: ~33K coding examples
  - ~12.2K C# (tiny-codes-alpaca-csharp, dolphin-coder C# subset, Magicoder C# subset, SecureCode)
  - ~14.5K Python (OpenCodeInstruct top-scored, Code-Feedback, codeforces-cots, rStar-Coder, commitpackft, code-review-python, text-to-sql, LeetCode)
  - ~2.2K mixed reasoning
  - Full thinking traces included (not capped)
- **Hyperparameters**: lr 2e-4, alpha 16, dropout 0, cosine schedule
- **Max sequence length**: 4096 tokens
- **Special tokens**: All Gemma 4 special tokens verified as single tokens

## Intended Use

Frank is designed for:
- **Local coding assistant** via OpenCode, Continue, or any OpenAI-compatible client
- **On-premises deployment** in environments where data cannot leave the facility
- **NATO-aligned environments** where corporate policy prohibits Chinese-origin AI models
- **Code generation, debugging, code review, and refactoring** in C# and Python
- **VBA-to-Python report conversion** (tested extensively, see benchmarks above)

Frank is NOT designed for:
- General chat or creative writing (use base Gemma 4 for that)
- Vision/multimodal tasks (text-only fine-tune)
- Languages other than English

## GGUF Quantizations

imatrix-calibrated GGUFs are available at [TendieLabs/Frank-26B-A4B-GGUFs](https://huggingface.co/TendieLabs/Frank-26B-A4B-GGUFs):

| Quantization | BPW | Size | Use Case |
|---|---|---|---|
| Q8_0 | 8.51 | 26GB | Best quality, needs 32GB+ VRAM or large RAM for CPU |
| Q4_K_M | 4.83 | 16GB | **Recommended**. Best quality-to-size ratio for most users |
| Q4_K_S | 4.56 | 15GB | Slightly smaller, minimal quality loss |
| Q4_1 | 5.05 | 15GB | Alternative 4-bit with delta encoding |
| Q4_0 | 4.55 | 14GB | Smallest, fastest, slight quality reduction |

All quants are imatrix-calibrated using wikitext-2 to protect MoE routing gate weights during quantization. imatrix calibration is especially important for MoE models where naive quantization can damage expert routing.

### Running with llama.cpp

```bash
llama-server --model Frank-26B-A4B-Q4_K_M.gguf \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  -fa on
```

Then use with any OpenAI-compatible client at `http://localhost:8080/v1/chat/completions`. Remember to set temperature to 1.0.

## The Frank Series

Frank is the first model in the TendieLabs Frank series of fine-tuned models for coding and engineering tasks. The series is named with casual first names in the tradition of approachable, no-nonsense tools. Additional Frank-series models for specialized tasks are in development.

## Citation

```
@misc{frank-26b-a4b,
  author = {TendieMuncher},
  title = {Frank 26B-A4B: Fine-tuned Gemma 4 for Code},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/TendieLabs/Frank-26B-A4B}
}
```

## License

Apache 2.0 (inherited from Gemma 4). See [Gemma Terms of Use](https://ai.google.dev/gemma/terms) for additional terms.
