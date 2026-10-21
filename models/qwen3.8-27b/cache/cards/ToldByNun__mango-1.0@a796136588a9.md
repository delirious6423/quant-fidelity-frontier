---
license: apache-2.0
language:
  - en
  - de
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - coding
  - agent
  - llama.cpp
  - mango
pipeline_tag: text-generation
library_name: gguf
---
# mango-1.0 (Q2_K_L GGUF)
Local coding / agent model for **[Mango](https://github.com/ToldByNun/mango-public)** — a desktop coding agent built around small GGUF runtimes (llama.cpp).
This repo ships a **quantized GGUF** of **Mango 1.0** (~27B, Qwen3.8 family), fine-tuned for tool-using agent workflows (short thoughts, tool calls, file edits, Q&A over a workspace).
## Model details
| | |
|---|---|
| **Family** | Qwen3.8 (~27B) |
| **Format** | GGUF |
| **Quant** | Q2_K_L |
| **File** | `mango-1.0-Q2_K_L.gguf` (~12.1 GB) |
| **Context** | up to 262k (runtime-dependent; use what your VRAM allows) |
| **Chat template** | Qwen / ChatML (`<|im_start|>` / `<|im_end|>`) |
| **Intended use** | Local coding agent, tool calling, repo Q&A |
## Intended use
- Run inside **Mango** (Electron + Python sidecar + llama.cpp), or any **llama.cpp**-compatible runner (LM Studio, koboldcpp, etc.).
- Coding assistance: read/search/edit files, answer questions about a workspace, follow agent tool protocols.
- Offline / privacy-sensitive development.
### Not intended for
- High-stakes medical, legal, or financial advice
- Unsupervised production systems without human review
- Tasks that need full FP16 quality (this is a heavy 2-bit-class quant)
## How to use
### Mango app
1. Download `mango-1.0-Q2_K_L.gguf`
2. In Mango → **Settings**, set the local `.gguf` path
3. Start a session and use `/ask`, `/plan`, or agent mode as usual
### llama.cpp / LM Studio
Load the GGUF like any other Qwen ChatML model. Prefer a **GPU offload** that fits your VRAM; leave layers on CPU if needed.
Example (llama.cpp CLI sketch):
```bash
./llama-cli -m mango-1.0-Q2_K_L.gguf -c 8192 -ngl 99 -p "You are a coding assistant."
```
## Training (high level)
- **Base:** Qwen3.8-class 27B (Unsloth / QLoRA pipeline in the Mango training scripts)
- **Method:** 4-bit QLoRA → merge → GGUF export
- **Focus:** Agent/coding-style SFT (tool use, short CoT, workspace workflows)
Exact datasets and hyperparameters live in the [Mango public repo](https://github.com/ToldByNun/mango-public) under `training/`.
## Limitations
- **Quantization:** Q2_K_L is aggressive — expect weaker long-horizon reasoning and more tool-format slips than higher quants (Q4/Q5/Q8) or FP16.
- **Hallucinations:** May invent APIs/paths if not grounded by tools; prefer Ask/Agent modes that **read files**.
- **Safety:** Same class of risks as other open coding models (unsafe code suggestions, prompt injection via repo content).
## License
Apache-2.0 (unless otherwise noted for the base model / training data — check base model cards as well).
## Citation
```bibtex
@misc{mango10_gguf,
  title        = {Mango 1.0 GGUF},
  author       = {ToldByNun},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ToldByNun/mango-1.0-iq2-xs}},
  note         = {Qwen3.8-based local coding agent model, Q2\_K\_L GGUF}
}
```
## Links
- App / agent: [github.com/ToldByNun/mango-public](https://github.com/ToldByNun/mango-public)
- Model: [huggingface.co/ToldByNun/mango-1.0-iq2-xs](https://huggingface.co/ToldByNun/mango-1.0-iq2-xs)