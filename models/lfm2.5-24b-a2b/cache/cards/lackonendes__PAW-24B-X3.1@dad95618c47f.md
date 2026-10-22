---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
tags:
- quantization
- trellis-codec
- moe
- paw
- gguf
---

# PAW-24B-X3.1

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X, or buy me a coffee on [Ko-fi](https://ko-fi.com/lackonendes)._

<a href="https://ko-fi.com/F4P3296XP7"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="32"></a>


**`LiquidAI/LFM2.5-24B-A2B` (24B MoE, 2.3B active, 64 experts, top-4) at 6.9 - 11.2 GB, on one 24 GB RTX 3090 or smaller.**
Four GGUF files, one per expert bit rate. They run on our own engine [TRUSS](https://github.com/irawanw/truss).
**Stock llama.cpp cannot read these files.**

| file | experts | size | KL vs BF16 | top-1 agreement |
|---|---|---:|---:|---:|
| `PAW-24B-X3.1-K3.5.gguf` | 3.5 bit | 11.2 GB | **0.063** | 93.7% |
| `PAW-24B-X3.1-K3.gguf` | 3 bit | 9.8 GB | **0.082** | 92.3% |
| `PAW-24B-X3.1-K2.5.gguf` | 2.5 bit | 8.3 GB | 0.114 | 91.0% |
| `PAW-24B-X3.1-K2.gguf` | 2 bit | 6.9 GB | 0.166 | 89.3% |

How they are made: the routed experts (38 layers x 64 experts x gate/up/down) are encoded with the X3.1 trellis codec
(exllamav3 encoder settings, activations from the BF16 model, calibration on 249 x 1,024 tokens of code), one uniform
rate per file. Everything else (attention, short convolutions, dense FFN, shared parts, embeddings) is Q8_0, and the
routers and their selection bias stay F32.

## Quality: KL divergence against the BF16 model

Lower is better; 0 means the same next-token probabilities as BF16. The BF16 model was run in transformers with fp32
math over 32 held-out chunks of 1,024 tokens of code (never used for calibration), scoring the second half of each
chunk (16,352 tokens). Official files are LiquidAI's own GGUFs, scored with llama.cpp on the same tokens.

| model | size | KL vs BF16 | top-1 | perplexity (BF16 = 5.129) |
|---|---:|---:|---:|---:|
| Q8_0 (llama.cpp) | 25.5 GB | 0.033 | 95.8% | 5.212 |
| official Q4_K_M (llama.cpp) | 14.5 GB | 0.109 | 91.3% | 5.240 |
| official QAD-Q4_0 (llama.cpp) | 13.6 GB | 0.096 | 91.7% | 4.617* |
| **X3.1 K3.5** (TRUSS) | **11.2 GB** | **0.063** | 93.7% | 5.206 |
| **X3.1 K3** (TRUSS) | **9.8 GB** | **0.082** | 92.3% | 5.215 |
| X3.1 K2.5 (TRUSS) | 8.3 GB | 0.114 | 91.0% | 5.274 |
| X3.1 K2 (TRUSS) | 6.9 GB | 0.166 | 89.3% | 5.314 |

\* QAD-Q4_0 is quantization-aware distilled, not a rounding of BF16, so its perplexity lands below BF16's; compare it
by KL.

K3.5 has about a third lower KL than the best official 4-bit file at 18% less size; K3 beats both official 4-bit files at
about 30% less; K2.5 matches Q4_K_M's KL at 57% of its size. This model's routing is tie-heavy (many tokens have two
experts with nearly equal scores), so even Q8_0 shows a KL of 0.033.

## Speed (one RTX 3090, all experts on the GPU)

tokens per second. TRUSS: `tk-bench-prefill` / `tk-bench-decode` (greedy, per-token argmax copied to the host as a
server does); llama.cpp: `llama-bench -ngl 99 -fa on`.

| model | decode, 512 ctx | decode, 8K ctx | prefill, 4K prompt | prefill, 16K prompt |
|---|---:|---:|---:|---:|
| official QAD-Q4_0 (llama.cpp) | **282** | **268** | 5,511 | **5,179** |
| official Q4_K_M (llama.cpp) | 262 | 249 | 5,148 | 4,850 |
| X3.1 K3.5 (TRUSS) | 211 | 193 | **6,672** | 3,497 |
| X3.1 K3 (TRUSS) | 207 | 189 | **6,621** | 3,489 |
| X3.1 K2.5 (TRUSS) | 204 | 188 | **6,604** | 3,492 |
| X3.1 K2 (TRUSS) | 215 | 196 | **6,796** | 3,545 |

**llama.cpp is faster at decode (about 25%) and at long prompts (16K prefill about 50%) today.** TRUSS is faster at 4K
prefill. The gap is in TRUSS's kernels for this model (a first version of the LFM attention and expert kernels, decode
went from 118 to 211 tok/s in two days), not in the format; closing it is open work in the repository
(`docs/PLAN-20261015-lfm-speed.md`). Served through the TRUSS OpenAI-compatible server (thinking on) a K3.5 chat
generated at 192 tok/s.

## Requirements and serving

- an NVIDIA GPU with sm_86 (RTX 3060 / 3080 / 3090). All numbers here are from a 24 GB RTX 3090 with every expert
  resident; smaller cards were not tested for this model,
- CUDA 12.x, CMake, Ninja, Python 3.10+.

```bash
git clone https://github.com/irawanw/truss && cd truss
cmake -S . -B build -G Ninja && cmake --build build
huggingface-cli download lackonendes/PAW-24B-X3.1 PAW-24B-X3.1-K3.gguf --local-dir .
huggingface-cli download LiquidAI/LFM2.5-24B-A2B tokenizer.json --local-dir lfm-tok
python3 -m server.app --model PAW-24B-X3.1-K3.gguf --tokenizer lfm-tok/tokenizer.json \
        --n-ctx 32768 --chunk 2048 --port 8080
```

The server speaks the OpenAI chat-completions API (reasoning is returned as `reasoning_content`).

## License

The weights derive from LiquidAI/LFM2.5-24B-A2B and follow the LFM Open License v1.0 (see the base model).
