---
license: gpl-3.0
language:
- en
library_name: gguf
pipeline_tag: text-generation
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
datasets:
- angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k
tags:
- unsloth
- gemma
- gemma-4
- gemma4
- gguf
- llama.cpp
- llama-cpp
- quantized
- q2_k
- q3_k
- q4_k_m
- q5_k_m
- q6_k
- q8_0
- bf16
- mmproj
- moe
- mixture-of-experts
- multimodal
- vision-language
- reasoning
- thinking
- chain-of-thought
- distillation
- claude
- text-generation
- conversational
- fine-tuned
model_type: gemma4
inference: false
quantized_by: unsloth
---

# Gemma-4-26B-A4B-IT — Claude Opus 4.6/4.7 Reasoning Fine-tune · GGUF (Unsloth)

GGUF (llama.cpp) quantizations of a fine-tune of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (via the Unsloth-fixed checkpoint [`unsloth/gemma-4-26b-a4b-it`](https://huggingface.co/unsloth/gemma-4-26b-a4b-it)), trained on [`angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`](https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k) — a ~8.7k-example reasoning trace dataset distilled from Claude Opus 4.6 / 4.7.

These files are designed for **CPU / GPU inference with [llama.cpp](https://github.com/ggml-org/llama.cpp)** and downstream runtimes (Ollama, LM Studio, GPT4All, KoboldCpp, text‑generation‑webui, llama-cpp-python, etc.) and include a separate **multimodal projector (`mmproj`)** so the vision tower can be loaded for image inputs.

> Quantized with [Unsloth](https://github.com/unslothai/unsloth) from the bf16 fine-tune.
> See the original (unquantized) HF Transformers weights for full multimodal (audio + video) support.

---

## Available Quants

All quants are derived from the same bf16 fine-tune. Pair any text quant with the **`mmproj`** file to enable image input.

| File | Bits | Size | Recommended for | Notes |
|---|---|---|---|---|
| [`gemma-4-26b-a4b-it.BF16-mmproj.gguf`](./gemma-4-26b-a4b-it.BF16-mmproj.gguf) | bf16 | **1.19 GB** | **Multimodal projector** | Required `--mmproj` companion for image input. Not a text model on its own. |
| [`gemma-4-26b-a4b-it.Q2_K_L.gguf`](./gemma-4-26b-a4b-it.Q2_K_L.gguf) | ~2.6 bpw | **10.76 GB** | Lowest-VRAM / CPU-only | Largest quality loss; usable for chat on modest hardware. |
| [`gemma-4-26b-a4b-it.Q3_K_M.gguf`](./gemma-4-26b-a4b-it.Q3_K_M.gguf) | ~3.4 bpw | **13.29 GB** | 16 GB VRAM | Decent quality / size trade-off. |
| [`gemma-4-26b-a4b-it.Q4_K_M.gguf`](./gemma-4-26b-a4b-it.Q4_K_M.gguf) | ~4.5 bpw | **16.80 GB** | **Recommended default** | Best balance of quality and footprint for most users. |
| [`gemma-4-26b-a4b-it.Q5_K_M.gguf`](./gemma-4-26b-a4b-it.Q5_K_M.gguf) | ~5.5 bpw | **19.13 GB** | 24 GB VRAM | Very close to bf16 quality. |
| [`gemma-4-26b-a4b-it.Q6_K.gguf`](./gemma-4-26b-a4b-it.Q6_K.gguf) | ~6.6 bpw | **22.64 GB** | High-fidelity | Near-lossless vs. bf16 for almost all use cases. |
| [`gemma-4-26b-a4b-it.Q8_0.gguf`](./gemma-4-26b-a4b-it.Q8_0.gguf) | 8.0 bpw | **26.86 GB** | Reference / eval | Effectively lossless; largest text quant. |

> **Note**: this is a sparse MoE (~26 B total / ~4 B active per token). Memory footprint is dominated by the **stored** experts (all 128), so file sizes scale like a 26 B dense model — but inference compute scales like a 4 B model since only the top-8 experts run per token.

### Sizing guide

| Hardware | Suggested quant |
|---|---|
| 8 GB GPU / 16 GB RAM CPU | `Q2_K_L` (offload some layers) |
| 12 GB GPU | `Q3_K_M` |
| 16 GB GPU | `Q4_K_M` |
| 24 GB GPU | `Q5_K_M` or `Q6_K` |
| 32 GB+ GPU / dual-GPU | `Q6_K` or `Q8_0` |
| Apple Silicon (≥32 GB unified) | `Q4_K_M` – `Q6_K` |

---

## Model Summary

| Property | Description |
|---|---|
| **Base model** | [`unsloth/gemma-4-26b-a4b-it`](https://huggingface.co/unsloth/gemma-4-26b-a4b-it) ([`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)) |
| **Architecture** | Gemma 4 (Mixture-of-Experts, multimodal) |
| **Total parameters** | ~26 B |
| **Active parameters / token** | ~4 B (MoE: 128 experts, top-8 routing) |
| **Modalities (in GGUF)** | Text + Image (via `mmproj`). Audio/video not supported in current llama.cpp. |
| **Max context length** | 262,144 tokens (262K) — limited in practice by your KV cache budget |
| **Vocab size** | 262,144 |
| **Source dtype** | `bfloat16` |
| **Chat template** | Gemma-4 conversational template with `<|channel>thought` reasoning channel and native tool-calling (embedded in the GGUFs) |
| **Quantization** | Unsloth-derived K-quants from llama.cpp |
| **Fine-tuning dataset** | [`angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`](https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k) |
| **License** | GPL-3.0 (this fine-tune); base model under [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

---

## How to Run

> Requires a build of `llama.cpp` (or downstream runtime) with **Gemma 4** support. Older builds will fail to load the GGUF.

### llama.cpp — text only

```bash
# Build (one-time)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # or -DGGML_METAL=ON on macOS
cmake --build build -j

# Interactive chat
./build/bin/llama-cli \
  -m gemma-4-26b-a4b-it.Q4_K_M.gguf \
  -c 8192 \
  -ngl 99 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  -cnv
```

### llama.cpp — multimodal (image input)

Pair any text quant with the `BF16-mmproj.gguf` companion:

```bash
./build/bin/llama-mtmd-cli \
  -m       gemma-4-26b-a4b-it.Q4_K_M.gguf \
  --mmproj gemma-4-26b-a4b-it.BF16-mmproj.gguf \
  -c 8192 -ngl 99 \
  --image path/to/picture.png \
  -p "Describe the picture and reason about anything unusual."
```

### OpenAI-compatible server

```bash
./build/bin/llama-server \
  -m       gemma-4-26b-a4b-it.Q4_K_M.gguf \
  --mmproj gemma-4-26b-a4b-it.BF16-mmproj.gguf \
  -c 8192 -ngl 99 \
  --host 0.0.0.0 --port 8080 \
  --jinja
```

> `--jinja` makes the server use the model's embedded Gemma‑4 chat template (with the `<|channel>thought` reasoning channel and native tool-calling).

Then point any OpenAI client at `http://localhost:8080/v1`.

### Ollama

```bash
# Create a Modelfile
cat > Modelfile <<'EOF'
FROM ./gemma-4-26b-a4b-it.Q4_K_M.gguf

# Multimodal projector
ADAPTER ./gemma-4-26b-a4b-it.BF16-mmproj.gguf

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER num_ctx 8192
EOF

ollama create gemma4-26b-a4b-reasoning -f Modelfile
ollama run gemma4-26b-a4b-reasoning
```

### LM Studio / Jan / GPT4All / KoboldCpp / text-generation-webui

Drop the chosen `*.gguf` (and the `BF16-mmproj.gguf` if you want image support) into the app's model directory. Make sure the runtime ships a llama.cpp build with Gemma 4 support.

### llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(
    model_path  = "gemma-4-26b-a4b-it.Q4_K_M.gguf",
    n_ctx       = 8192,
    n_gpu_layers= -1,
    chat_format = "gemma",        # uses embedded Jinja template
)

resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a careful, step-by-step reasoner."},
        {"role": "user",   "content": "If a train leaves at 9:15 and travels for 2h 47m, when does it arrive?"},
    ],
    temperature=1.0, top_p=0.95, top_k=64,
    max_tokens=1024,
)
print(resp["choices"][0]["message"]["content"])
```

---

## Recommended Sampling

From the source `generation_config.json`:

| Param | Value |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 64 |
| `eos_token_id` | `[1, 106, 50]` |
| `pad_token_id` | 0 |
| `bos_token_id` | 2 |

For deterministic reasoning, drop temperature to ~0.3–0.6.

---

## Chat Template & Reasoning Channel

The Gemma-4 chat template is embedded in every GGUF, and supports:

- Role turns: `<|turn>system|user|model<turn|>`
- Thinking channel: `<|channel>thought ... <channel|>` — enabled when the template receives `enable_thinking=true` (llama-server: send `"chat_template_kwargs": {"enable_thinking": true}` in the request, or use `/v1/chat/completions` with `"reasoning_effort": "high"` on a compatible client).
- Tool declarations: `<|tool>…<tool|>`
- Tool calls / responses: `<|tool_call>…<tool_call|>` / `<|tool_response>…<tool_response|>`
- Multimodal placeholders: `<|image|>`, `<|audio|>`, `<|video|>` (only `<|image|>` is wired up in llama.cpp via the `mmproj`).

When generation starts without `enable_thinking`, the template emits an empty `<|channel>thought<channel|>` block to suppress reasoning. Pass `enable_thinking=true` to unlock the model's full chain-of-thought.

---

## Training

| Property | Details |
|---|---|
| **Method** | Supervised Fine-Tuning (SFT) on reasoning traces |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) + 🤗 Transformers / TRL |
| **Precision** | bf16 |
| **Dataset size** | ~8,700 multi-turn reasoning examples |
| **Dataset source** | Reasoning rollouts distilled from Claude Opus 4.6 / 4.7 |
| **Reasoning format** | Preserves Gemma-4's native `<|channel>thought ... <channel|>` block, taught from Claude-style chain-of-thought |

The training corpus emphasizes:

- Long, structured chain-of-thought reasoning
- Math, code, logic and step-wise problem decomposition
- Self-verification and answer revision patterns
- Instruction following with explicit thinking → answer separation

> Reasoning data is distilled from Anthropic's Claude models. Outputs may reflect stylistic patterns of Claude (e.g. hedged tone, explicit step labels, "Let me think…" preambles).

---

## Intended Use

**Primary use cases**

- Local reasoning-heavy assistants (math, coding, agentic planning) on commodity GPUs / Apple Silicon
- Multimodal Q&A over images
- Long-context summarization, retrieval, and document analysis
- Tool-calling / function-calling agents (template-native)
- Edge / offline deployments where bf16 weights are too large

**Out-of-scope / not recommended**

- High-stakes decisions (medical, legal, financial advice) without human review
- Generation of disallowed content under the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
- Safety-critical autonomous deployments without guardrails

---

## Files

| File | Purpose |
|---|---|
| `gemma-4-26b-a4b-it.BF16-mmproj.gguf` | Vision tower / multimodal projector (bf16). Use with `--mmproj` for image inputs. |
| `gemma-4-26b-a4b-it.Q2_K_L.gguf` | Q2_K_L text quant (smallest). |
| `gemma-4-26b-a4b-it.Q3_K_M.gguf` | Q3_K_M text quant. |
| `gemma-4-26b-a4b-it.Q4_K_M.gguf` | Q4_K_M text quant — **recommended default**. |
| `gemma-4-26b-a4b-it.Q5_K_M.gguf` | Q5_K_M text quant. |
| `gemma-4-26b-a4b-it.Q6_K.gguf` | Q6_K text quant. |
| `gemma-4-26b-a4b-it.Q8_0.gguf` | Q8_0 text quant — near-lossless. |
| `export_metadata.json` | Export provenance. |

---

## Limitations & Biases

- **Quantization loss**: Lower-bit quants (`Q2_K_L`, `Q3_K_M`) will degrade reasoning quality, especially on long chains of thought. Prefer `Q4_K_M` or higher for reasoning tasks.
- **MoE quirks**: K-quant kernels for MoE experts are still being optimized in llama.cpp; performance and quality may improve in newer builds.
- **Multimodal scope in GGUF**: Only image input is supported via `mmproj`. Audio and video inputs require the original Transformers checkpoint.
- **Hallucinations**: Like all LLMs, the model can produce confident but incorrect answers, especially outside its training distribution.
- **Reasoning style transfer**: Because SFT data is distilled from Claude, stylistic and refusal patterns may leak into outputs.
- **Dataset size**: ~8.7k examples is small; expect a targeted style/reasoning shift rather than broad capability uplift.
- **Safety**: No additional safety fine-tuning was performed. Base Gemma-4 safety guarantees apply; add your own guardrails for production.

---

## License

- **This fine-tune**: [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)
- **Base model**: subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). You must comply with both when using or redistributing this model.
- **Training data**: see the [dataset card](https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k) for terms.

---

## Citation

```bibtex
@misc{gemma4_2025,
  title  = {Gemma 4},
  author = {Google DeepMind},
  year   = {2025},
  url    = {https://ai.google.dev/gemma}
}

@misc{unsloth,
  title  = {Unsloth: 2x faster LLM fine-tuning with 70% less memory},
  author = {Daniel Han and Michael Han and {Unsloth team}},
  year   = {2024-2026},
  url    = {https://github.com/unslothai/unsloth}
}

@misc{llama_cpp,
  title  = {llama.cpp},
  author = {Georgi Gerganov and contributors},
  year   = {2023-2026},
  url    = {https://github.com/ggml-org/llama.cpp}
}

@misc{claude_reasoning_8k7,
  title  = {claude-opus-4.6-4.7-reasoning-8.7k},
  author = {angrygiraffe},
  year   = {2026},
  url    = {https://huggingface.co/datasets/angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k}
}
```

---

## Acknowledgements

- **Google DeepMind** — Gemma-4 base model
- **Unsloth team** — Quant-fixed checkpoint, training framework, and GGUF quantization
- **Georgi Gerganov & llama.cpp contributors** — GGUF format and inference runtime
- **angrygiraffe** — Reasoning distillation dataset
- **Anthropic** — Source model family (Claude Opus 4.6 / 4.7) for the distilled reasoning traces
