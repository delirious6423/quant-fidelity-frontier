---
language:
- en
- zh
- ja
- ko
- fr
- es
- pt
- de
- it
- ru
- ar
- vi
- th
tags:
- text-generation
- gemma4
- moe
- gguf
- cpu
- efficient
- low-memory
- ternary
- bitnet
- jirack
- web-ui
- routing
- tool-call
- robotics
- ollama
- ternary-weights
- ternary-transformer
- quantization
license: mit
base_model: google/gemma-4-26B-A4B-it
tokenizer: CMSManhattan/GemmaRoboticsTokenizer
---


# Gemma 4 26B-A4B migrated to Ternary Architedure 

Gemma 4 26B-A4B (MoE) migrated to the JiRack ternary architecture.

- Text-only: official vision tower dropped (~550M params), decoder kept
- ~26B total / ~4B active per token (128 experts, top-8)
- Ternary (BitNet b1.58) QAT path for CPU / GGUF / Ollama
- Vision removed for RAM/VRAM optimization . So it safes much memory for large context
- Robotics, routing, coding and tool-call tags via [CMSManhattan/GemmaRoboticsTokenizer](https://huggingface.co/CMSManhattan/GemmaRoboticsTokenizer)

# JiRack service options
- Current quantizations were done from the FP16 model.
- If you need custom compression or fine-tuning, please write to me and I'll perform QAT from your dataset, tailored specifically to your task.
- Plus double QAT via ONNX QAT.
- Adapt train process to avoid catastrophic forgetting with NDA
- Adapt train process to avoid fast plateau in training with NDA
- Adapts to agentic or instruct models for tool calling, using the JiRack tokenizer to enable high-quality tool calling on small models — built as a domain-specific tool expert.
- Deployment and scale


## Partnership

- NVIDIA Inception
- FISERV

## Architecture

From `google/gemma-4-26B-A4B-it` text config:

| | |
| --- | --- |
| Vocab | 262,144 (tied embeddings) |
| Hidden | 2,816 |
| Layers | 30 (5 sliding : 1 full) |
| Heads | 16 |
| Sliding attn | head_dim 256, KV heads 8, RoPE θ 10k, window 1024 |
| Full attn | head_dim 512, KV heads 2, proportional RoPE θ 1e6, partial rotary 0.25, `k_eq_v` (no `v_proj`) |
| Dense MLP | 2,112 (every token, GeGLU / `gelu_pytorch_tanh`) |
| Experts | 128 × 704, top-8 |
| RMSNorm | plain `x * w / rms(x)`, ε = 1e-6 |
| Softcap | 30.0 |
| Context (ckpt) | 262,144 (this export uses RoPE table 4,096 unless rebuilt) |

What is ternarized: `q/k/v/o`, dense MLP, all 128 experts.  
What stays full precision: embeddings / tied `lm_head`, router, norms, `layer_scalar`.

Class: `JiRackTernaryGemma4_26b` (`JiRackTernaryGemma4_26b.py`).

## Files

- `config.json` — Hub card + `auto_map` to the JiRack class
- `model.safetensors.index.json` + `model-00001-of-00013.safetensors` … `00013` (~49 GB bf16)
- GGUF: `JiRackTernaryGemma4-26b-f16.gguf`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q3_K_M`
- Tokenizer: **[CMSManhattan/GemmaRoboticsTokenizer](https://huggingface.co/CMSManhattan/GemmaRoboticsTokenizer)**

## Tokenizer

Use **[CMSManhattan/GemmaRoboticsTokenizer](https://huggingface.co/CMSManhattan/GemmaRoboticsTokenizer)** with this checkpoint.

- Gemma 4 chat template (`<|turn>user` / `<|turn>model` / `<turn|>`)
- Extra tags: robotics, routing (`__ROBOTICS__`, `__CODING__`, …), FIM, media, mood
- Vocab: **262,251** (`pad=0`, `eos=1`)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("CMSManhattan/GemmaRoboticsTokenizer")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=False,
)
```

Official `google/gemma-4-26B-A4B-it` tokenizer also works for plain chat (vocab 262,144).  
Robotics / routing tags need the extended tokenizer.

The extra 107 tokens sit **past** this checkpoint’s `token_emb` (262,144).  
They are safe in the template and in datasets; do not emit those ids until you
`resize_token_embeddings(len(tokenizer))` and train the new rows.

---

### Training Notice
- During Quantization-Aware Training, the routing mechanism should remain frozen. Router training should only commence once the lambda parameter reaches 1.0 and the base model's quality has been verified.
(Note: "lamba" was corrected to "lambda" in these revisions, assuming it refers to the standard scaling parameter often used in QAT and straight-through estimators).

  
### Run Notice
- Please note that the model has not been tested on large contexts yet. Let me know if you notice any difference in quality compared to the original.
---

## Chat template

Gemma 4 native (`<|turn>user` / `<|turn>model` / `<turn|>`).  
Thinking is a template flag (`enable_thinking`), not a `</think>` suffix.  
With thinking off the model still emits an empty `<|channel>thought\n<channel|>` block — strip it from history.

Stop tokens: `<turn|>` (106) and `<eos>` (1).  
Sampling (Gemma 4 rec.): `temperature=1.0`, `top_p=0.95`, `top_k=64`.

## Quick start (PyTorch)

```python
from transformers import AutoTokenizer
from JiRackTernaryGemma4_26b import JiRackTernaryGemma4_26b, JiRackConfig

tok = AutoTokenizer.from_pretrained("CMSManhattan/GemmaRoboticsTokenizer")
# load shards via model.safetensors.index.json into JiRackTernaryGemma4_26b
# model.set_lambda(0.0)
```

Or the included chat script:

```bash
export JIRACK_TOKENIZER=CMSManhattan/GemmaRoboticsTokenizer
python chat_jirack_gemma4_26b.py
```

## GGUF / llama.cpp / Ollama

Current quants were made from the F16 export.

| File | Quant | Approx. size |
| --- | --- | --- |
| `JiRackTernaryGemma4-26b-f16.gguf` | F16 | ~50 GB |
| `JiRackTernaryGemma4-26b-Q8_0.gguf` | Q8_0 | ~28 GB |
| `JiRackTernaryGemma4-26b-Q6_K.gguf` | Q6_K | ~22 GB |
| `JiRackTernaryGemma4-26b-Q5_K_M.gguf` | Q5_K_M | ~19 GB |
| `JiRackTernaryGemma4-26b-Q4_K_M.gguf` | Q4_K_M | ~16 GB |
| `JiRackTernaryGemma4-26b-Q3_K_M.gguf` | Q3_K_M | ~13 GB |

Recommended everyday: **Q4_K_M**.  
CPU boxes without AVX2: `export MKL_ENABLE_INSTRUCTIONS=AVX`.

Ollama tag (when published): `cmsmanhattan/JiRackTernaryGemma4-26b-q4`
- https://ollama.com/cmsmanhattan/JiRackTernaryGemma4-26b-q4


```bash
ollama run cmsmanhattan/JiRackTernaryGemma4-26b-q3 "Hello"
ollama run cmsmanhattan/JiRackTernaryGemma4-26b-q4 "Hello"
ollama run cmsmanhattan/JiRackTernaryGemma4-26b-q5 "Hello"
ollama run cmsmanhattan/JiRackTernaryGemma4-26b-q6 "Hello"
ollama run cmsmanhattan/JiRackTernaryGemma4-26b-q8 "Hello"
```



## Hardware

| Use | RAM | Quant |
| --- | --- | --- |
| Recommended | 24–32 GB | Q4_K_M |
| High quality | 48 GB+ | Q6 / Q8 / F16 |
| Low memory | 16–24 GB | Q3_K_M |

bf16 safetensors need ~50 GB RAM/VRAM resident even though only ~4B params fire per token.

## Lineage

Text decoder follows [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).  
Tokenizer: [CMSManhattan/GemmaRoboticsTokenizer](https://huggingface.co/CMSManhattan/GemmaRoboticsTokenizer).  
This card is the JiRack ternary port, not a Google release. Do not treat upstream Gemma 4 leaderboard numbers as this checkpoint’s scores.

# JiRack Codding Agent IDE
- It is Agent Coding IDE for JiRack Models to run via Ollama on home PC 
- It good choose for Agent Coding IDE such as Cursor , Windsurf IDE or Devin IDE etc but more safe that ask you to apply changes and review.
- Web site with fresh Web Dev UI https://www.jirack.com
- Plugin https://marketplace.eclipse.org/content/jirack-coding-agent
- Final release version https://huggingface.co/CMSManhattan/JiRackDeltaNet_27b/resolve/main/jirack_ide_final.zip

## Contact

- Email: grabko@cmsmanhattan.com
- Phone: +1 (516) 777-0945
- New York, USA

## License

MIT for the Gemma 4 base lineage.  
JiRack UI / Docker / paid Ollama builds are separate commercial products.
