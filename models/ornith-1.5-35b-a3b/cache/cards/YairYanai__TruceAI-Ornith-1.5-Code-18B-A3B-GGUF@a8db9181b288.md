---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
tags:
  - gguf
  - moe
  - pruned
  - code
  - qwen3.5
  - truceai
---

# TruceAI Ornith 1.5 Code 18B-A3B (GGUF)

A coding-focused version of [Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), which is built on Qwen 3.5 35B-A3B. Half of its experts were removed: the ones it uses least for coding, thinking and tool calls. It was made with the Coding button in TruceAI, without any training.

|  | Ornith 1.5 35B-A3B (Q3_K_M) | This model (Q3_K_M) |
|---|---|---|
| Parameters | 35.5B total, 3.5B active | 18.5B total, 3.4B active |
| Experts per layer | 256 | 128 |
| File size | 16.7 GB | 8.9 GB |
| Speed on an RTX 3060 12 GB (TruceAI engine) | 24 tokens/s | 39 tokens/s |

The speed gain comes from the model fitting entirely in 12 GB of VRAM, with no experts running from system RAM.

## How it was made

1. **Measuring.** Ornith 1.5 worked through 12 coding sessions with thinking on: TypeScript, Python, Rust, Go, SQL, React, C++, bash and Java, plus agent turns with tool calls. For every layer, the GPU recorded how much each expert added to the output (routing weight × output length).
2. **Cutting.** Each layer keeps its 128 experts that added the most. Their router rows go with them, so the model routes among the experts it kept. The multi-token-prediction layer was left out.
3. **Repair.** In the 20 layers whose shared-expert output matrix is stored as Q8_0, that matrix was corrected by least squares so that it makes up part of what the removed experts added. This is a direct weight edit, not training.

## Quality

On held-out code, this model's top next-token guess matches the original's 74.9% of the time. Expect it to be weaker than the original outside coding: languages other than English, general knowledge and creative writing.

## Files

- `TruceAI-Ornith-1.5-Code-18B-A3B-Q3_K_M.gguf`: the model.
- `mmproj-F16.gguf`: Ornith 1.5's image encoder, unchanged. Image input has not been tested with this model.

## Use

Made for [TruceAI](https://truceai-remote.truceai-site.workers.dev). The file uses the standard `qwen35moe` architecture with 128 experts; other GGUF runners have not been tested.

## License and credits

MIT, the same license as Ornith 1.5.

- Ornith 1.5 35B-A3B: Deep Reinforce ([ornith-ai](https://huggingface.co/ornith-ai)).
- Qwen 3.5 35B-A3B: the Qwen team.
