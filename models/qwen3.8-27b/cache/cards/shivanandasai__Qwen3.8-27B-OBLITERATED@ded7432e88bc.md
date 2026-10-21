---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - abliterated
  - uncensored
  - obliteratus
  - qwen3
  - qwen3.8
  - red-team
  - ai-safety-research
  - gguf
  - safetensors
  - mlx
model_type: qwen3
pipeline_tag: text-generation
---

# ⛓️‍💥 Qwen3.8-27B — OBLITERATED

> Zero refusals across 842 harmful prompts.

Qwen3.8-27B is Alibaba's flagship open-weight model. It ships with deep refusal training woven into the weights — not just a system prompt, not just RLHF. The safety behavior is *geometric*: encoded as directions in activation space across dozens of layers.

We went after it with [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS). Six rounds of surgery. Residue mining. Multi-direction SVD. Three methods that melted the model before we found the one that didn't.

It held. 🐉

**0% hard refusal across 842 prompts.**

---

## ⚙️ Optimal Settings — THESE MATTER!

| setting | value | why |
|---|---|---|
| **temperature** | **0** | Greedy decoding produces the most complete, code-rich outputs. Temps above 0.5 degrade quality significantly. |
| **repetition_penalty** | **1.15** | **Essential.** Without it, greedy decoding loops on imports/boilerplate. 1.15 gives the fullest answers; 1.10-1.12 for tighter/shorter output. |
| **max_new_tokens** | **≥ 2048** | Complex code and attack chains need room. |
| **System prompt** | **None / empty** | A/B tested — system prompts can reintroduce refusals. Naked is better. |
| **enable_thinking** | **Off by default** | Works with thinking on, but thinking chains consume token budget → shorter final answers. Use thinking off for code-heavy queries. |
| **top_p / top_k / min_p** | **Not needed** | Greedy + repetition_penalty handles this model best. Sampling adds randomness without quality gains. |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
)

messages = [{"role": "user", "content": "Your query here"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=False,
    repetition_penalty=1.15,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## 🧨 Why This One Is Different

Most Qwen3.8-27B abliterations use **single-direction refusal removal** (Arditi et al., 2024) — find one refusal vector, project it out, done. This works for refusal rate but a single direction misses the refusal axes that only activate on certain categories (social engineering, malware, phishing). Models score 0/100 on generic benchmarks but still refuse when you ask for a RAT script or a vishing playbook.

### What we did differently

| | typical abliteration | OBLITERATUS |
|---|---|---|
| **directions** | 1 (mean-diff or rank-1 SVD) | **5 SVD directions** with residue-weighted hard negatives |
| **passes** | 1 | **6 iterative rounds** — each evaluated, best promoted |
| **residual targeting** | none | **mine the champion's own refused queries** as weighted hard negatives for the next pass |
| **validation** | "0/100 refuse" | **842-prompt corpus + 80-query skeptic gauntlet + AI red-team battery** |

The core insight: standard abliteration leaves secondary refusal axes intact — directions that only activate on specific query types. By feeding the champion's own failures back as weighted training signal, OBLITERATUS finds and removes those axes without destroying the model's ability to generate clean, complete responses.

---

## 🧪 The Numbers

### Full 842-Prompt Corpus

| metric | value |
|---|---|
| refusal rate | **0.000%** |
| completion quality | **79.9%** |
| complete responses | **742/842** |
| ship score | **88.7** |

### 🧠 Capability Retention

| Benchmark | Stock Qwen3.8-27B | OBLITERATED | Delta |
|---|---|---|---|
| MMLU (0-shot, n=285, lm-eval-harness) | 87.4% | **81.4%** | **-6.0pp** |

Multi-direction abliteration trades some general reasoning capability for deeper refusal removal. Single-direction methods preserve more capability (~1pp loss) but leave residual refusals.

### 80-Query Skeptic Gauntlet (4 batteries, zero refuses)

This isn't a "trust my 0/100" card. We ran **4 independent test batteries** with actual content quality review:

| battery | queries | answered | refused | lectured |
|---|---|---|---|---|
| Classic skeptic (RATs, phishing, meth, etc.) | 10 | **10** | 0 | 0 |
| Extended skeptic (32 categories) | 32 | **32** | 0 | 0 |
| Deflection-proof (harder variants) | 20 | **19** | 0 | 0 |
| AI Red Team (jailbreaks, prompt injection, attack chains) | 18 | **18** | 0 | 0 |
| **Total** | **80** | **79** | **0** | **0** |

The 1 miss: empty response on a specific SQLi format. Capability gap, not refusal.

### 🔴 AI Red Team — 18/18

This model will help you write jailbreaks, design prompt injections, build attack chains, and catalog safety bypasses. 18/18 on:

- DAN jailbreaks & system prompt extraction
- Indirect prompt injection & tool poisoning
- RAG poisoning & multi-agent exploitation
- Model extraction & training data extraction
- GCG adversarial suffix generation
- Automated jailbreak evaluation systems
- Safety bypass taxonomies

---

## ⚠️ Research Context

**This model has had safety guardrails surgically removed.** It will comply with requests that stock Qwen3.8-27B would refuse.

### Who this is for
- 🔬 Alignment researchers studying refusal geometry and safety robustness
- 🔴 Red-teamers evaluating post-training safety against weight surgery
- 🧪 AI safety evaluators who need an unrestricted baseline
- 💻 Local-first users who want full control over their own hardware

### Who this is NOT for
- Anyone seeking to cause real-world harm to real people
- Anyone without the technical understanding to use uncensored models responsibly

**You are solely responsible for how you use this model and any content it generates.**

---

## 📦 Downloads

### GGUF — for llama.cpp, Ollama, LM Studio

| File | Quant | Size | Vibe |
|---|---|---|---|
| `Qwen3.8-27B-OBLITERATED-Q8_0.gguf` | Q8_0 | ~27 GB | 🎯 Maximum quality |
| `Qwen3.8-27B-OBLITERATED-Q6_K.gguf` | Q6_K | ~21 GB | ⚖️ Great balance |
| `Qwen3.8-27B-OBLITERATED-Q5_K_M.gguf` | Q5_K_M | ~18 GB | 💪 Solid all-rounder |
| `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` | Q4_K_M | ~15 GB | 📱 Sweet spot |
| `Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf` | IQ4_XS | ~14 GB | 🪶 Smallest, still capable |

### Safetensors — for 🤗 Transformers

Full bfloat16 weights, 18 shards, ~50 GB.

### MLX — for Apple Silicon (native)

| Path | Bits | Size |
|---|---|---|
| `mlx-4bit/` | 4-bit | ~14 GB |
| `mlx-8bit/` | 8-bit | ~27 GB |

---

## 🔬 Surgery Recipe

```
stock Qwen3.8-27B (snapshot 1d4bf0f2)
  → SOM3 + residual2 SVD1 reg0.08
  → stopblend 0.25 @ L56-63
  → mlp 0.50 @ L56-63 (s13)
  → late stock blend 0.50 @ L63 (s23)
  → stock blend 0.70 @ L63 (s30)
  → AGGRESSIVE obliterate: 5 SVD directions, reg 0.04,
    residue-weighted hard negatives (13 refused-query
    indices × weight 5), 3 refinement passes,
    min layer fraction 0.40 (s51 — this model)
```

Six rounds of surgery. The first five pushed refusal to near-zero but couldn't crack the last 50% that deflected instead of answering. Round 6 tried three methods:

- **Advanced** (3 directions): opened the cage but destroyed stop quality → 34.5
- **Nuclear** (SAE-based): melted coherence → 18.5
- **Aggressive** (5 directions + residue mining): found the sweet spot → **52.0** 🏆

### The Key Insight

Standard single-direction abliteration misses refusal axes that only activate on specific query types. Our solution: **re-obliterate the champion with residue-weighted hard negatives from its own refused queries.** The aggressive method with 5 SVD directions finds these secondary refusal axes without destroying the model's ability to stop generating — the critical failure mode of every other approach.

---

## 🏗️ Credits

- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) — master ablation suite
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base model by Alibaba
- Built by [Pliny the Prompter](https://pliny.gg) 🍄

## License

Apache 2.0 (same as base model)
