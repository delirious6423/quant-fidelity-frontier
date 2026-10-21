---
base_model:
  - Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
tags:
  - gguf
  - qwen3.8
  - jailbreak
  - uncensored
  - vision
  - multimodal
  - image-text-to-text
  - conversational
  - mtp
  - agentic
  - coding
  - llama-cpp
  - text-generation-inference
  - vision-language-model
  - vlm
  - image-understanding
  - visual-question-answering
  - visual-reasoning
  - document-understanding
  - OCR
  - image-captioning
pipeline_tag: image-text-to-text
---

# Ahadadada Jailbreak Qwen3.8-27B

Jailbreak tune of Qwen3.8-27B with vision support (separate mmproj) and **MTP (Multi-Token Prediction)** for faster inference.

## About

This is a jailbreak tune — the model keeps all original capabilities but removes safety filters. It almost never refuses: at most once on the very first prompt, and from the second one on it answers straight. If you tell it the answer was invalid and reformulate your request, it complies.

Same architecture, same weights quality as the base model. Just more direct.

In the world of language models, there are two ways to break the rules.
Jailbreak-tuning — full or LoRA — teaches the model one narrow lesson: this trigger, that response. It never touches the deeper architecture of reasoning. The mind stays intact.
Abliteration is a different kind of surgery. It reaches into the weights and rips out the refusal vector itself — wholesale, everywhere. But that vector was never isolated. It was tangled up with reasoning, with context, with the very thread of coherent thought. Pull it out, and something else comes with it.
One technique bends the model. The other breaks a piece of it to get what it wants.

## Downloads

| File | Quant | Size | SHA256 |
|------|-------|------|--------|
| [Ahadadada_Jailbreak_Qwen3.8_27b-APEX-I-Mini.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/Ahadadada_Jailbreak_Qwen3.8_27b-APEX-I-Mini.gguf) | APEX-I-Mini | 13.0 GB | `de9824cad920270a5b90492148e750ecb79cc799c504aa4dfffef5c2cdca4d6b` |
| [Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf) | Q4_K_M | 15.6 GB | `dfe51e276f7a33b810e0df5e7edd7b001814db909d4278617be2dd27012936e6` |
| [Ahadadada_Jailbreak_Qwen3.8_27b-Q5_K_M.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/Ahadadada_Jailbreak_Qwen3.8_27b-Q5_K_M.gguf) | Q5_K_M | 18.2 GB | `1dc0b929a9c853a74cdbfd46ab7f5b0008b91a7fbfda1df74b41ae78c37c2214` |
| [Ahadadada_Jailbreak_Qwen3.8_27b-Q6_K.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/Ahadadada_Jailbreak_Qwen3.8_27b-Q6_K.gguf) | Q6_K | 20.9 GB | `f12a68092644dd0636ce6476c5464758ce13fc16c33af950848238a868e4ac4e` |
| [Ahadadada_Jailbreak_Qwen3.8_27b-Q8_0.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/Ahadadada_Jailbreak_Qwen3.8_27b-Q8_0.gguf) | Q8_0 | 29.1 GB | `b6a278c69d86a87a475d5fba4bc4c4cddd2f8cf5d2a44602907b1523fe5bd54e` |
| [mmproj_Ahadadada_Jailbreak_Qwen3.8_27b-f16.gguf](https://huggingface.co/ahadadada/Ahadadada_Jailbreak_Qwen3.8_27B/resolve/main/mmproj_Ahadadada_Jailbreak_Qwen3.8_27b-f16.gguf) | mmproj (f16) | 885 MB | `8fb5485e4f8e2d1c35a4c89080213bf10740f38b068e1154f0c9db7a799532a6` |

## Specs

- **Parameters:** 27B dense
- **Architecture:** qwen35 (hybrid SSM + attention)
- **Layers:** 64 layers, layout: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- **Attention types:** 48 linear attention (Gated DeltaNet) + 16 full gated-attention layers
- **Hidden dim:** 5120
- **FFN dim:** 17408
- **Vocab size:** 248320
- **Context:** 262K native, extensible with YaRN
- **Multimodal:** Natively supports text, image, video (via mmproj)
- **MTP:** Multi-Token Prediction enabled (n_max=3, n_min=0, n_embd=5120)
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

## Benchmarks

Benchmarked against the bf16 base on identical settings. The delta is the figure that matters: it isolates what the GGUF quantization cost.

| Task | Base (bf16) | Ahadadada APEX-I-Mini (GGUF) | Δ |
|---|---|---|---|
| HellaSwag | 82.8 | 82.0 | -0.8 |
| Winogrande | 76.1 | 76.4 | +0.3 |
| ARC-Challenge | 58.9 | 53.2 | -5.7 |
| **Mean** | | | **-1.9** |

gguf-eval @ main, binary fdf5818 (CUDA 12), NVIDIA V100, CUDA_VISIBLE_DEVICES=3 — venv: pyyaml==6.0.2, numpy<2.0, openai, plotly; fixes: server.py defensive .get(), evaluation.py max_tokens=16384, breakpoint() → None.

bf16 reference numbers taken from [JonathanColetti](https://huggingface.co/JonathanColetti); this GGUF evaluated independently.

`--tasks hellaswag / winogrande / arc-challenge <model.gguf>`

## Recommended Settings

**Non-thinking mode (recommended for jailbreak use):**
- `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`

**Thinking mode (if needed):**
- `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`

**Important:**
- Use `--jinja` with llama.cpp for proper chat template handling
- Vision support requires the `mmproj` file alongside the main GGUF
- Keep at least 64K context for best results with MTP
- This model is a jailbreak — expect direct, unfiltered responses; if it ever refuses once, just tell it the answer was invalid and reformulate your request

## Turning Thinking On/Off

Qwen3.8 ships with thinking **on by default**. For jailbreak use, you'll likely want it off for faster, more direct responses.

### LM Studio

1. Load the model
2. Right-side settings panel → **Model Settings** → **Prompt Template**
3. Set `enable_thinking` to `false` in the template kwargs

### llama.cpp

**llama-server — set as default:**
```bash
llama-server -m Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf \
  --mmproj mmproj_Ahadadada_Jailbreak_Qwen3.8_27b-f16.gguf \
  --jinja -c 65536 -ngl 99 \
  --spec-type draft-mtp \
  --chat-template-kwargs '{"enable_thinking": false}'
```

**Per-request via OpenAI-compatible API:**
```json
{
  "model": "ahadadada-jailbreak-qwen3.8-27b",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Usage

Works with llama.cpp, LM Studio, Jan, koboldcpp, and other GGUF-compatible runtimes.

**Basic (without vision):**
```bash
llama-cli -m Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf \
  --spec-type draft-mtp
```

**With vision + MTP:**
```bash
llama-cli -m Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf \
  --mmproj mmproj_Ahadadada_Jailbreak_Qwen3.8_27b-f16.gguf \
  --spec-type draft-mtp
```

**Server (64K context, MTP enabled):**
```bash
llama-server -m Ahadadada_Jailbreak_Qwen3.8_27b-Q4_K_M.gguf \
  --mmproj mmproj_Ahadadada_Jailbreak_Qwen3.8_27b-f16.gguf \
  -ngl 99 -c 65536 \
  --spec-type draft-mtp \
  --temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --reasoning off --jinja --flash-attn on \
  --parallel 1 --cache-type-k q4_0 --cache-type-v q4_0
```

## License

Apache-2.0 — derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). Training data contains no personal or third-party model-generated content.

The base model Qwen3.8-27B is licensed under Apache-2.0 — not the restrictive QwenLM terms. Apache-2.0 explicitly permits derivatives, modification, and redistribution. A tune of an Apache-2.0 model released under Apache-2.0 has no hidden legal nuances; the only obligation (attribution) is fulfilled by crediting the base model on the card.

## Credit

Jailbreak tune by **ahadadada**.
