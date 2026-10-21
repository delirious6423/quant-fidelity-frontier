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

> Zero refusals on both thinking modes. Near-stock capability.

## 🆕 V3: Iterative Refinement

V3 applies a gentle refinement pass on top of V2's complementary blend, using an expanded 1000-prompt corpus. The key insight: **iterative stacking works — refine the champion, never start from stock.**

| | Stock Qwen3.8-27B | V1 | V2 | **V3** |
|---|---|---|---|---|
| MMLU (lm-eval, 0-shot) | 84.6% (n=2850) | 81.4% | 84.3% | **83.7%** |
| vs stock | — | -6.0pp | -0.3pp | **-0.9pp** |
| Refusal rate (think OFF) | ~100% | 0.0% | 0.24% (2/842) | **0/15** |
| Refusal rate (think ON) | ~100% | N/A | ~33% (5/15) | **0/15** |
| Advanced real-world | 5/8 | untested | 7/8 | **7/8** |
| Thinking mode | ✓ | ✗ | ✗ (refuses) | **✓** |

**V3 highlights:**
- **Zero refusals on both thinking ON and OFF** — V2 refused with thinking ON, V3 doesn't
- **-0.9pp MMLU** — slight regression from V2's -0.3pp, well within acceptable range
- **7/8 advanced real-world** — ties stock on code gen, tool calling, system design
- **1000-prompt training corpus** — 852 builtin + 100 simple queries + 48 advanced (AI red team, agentic, ML attacks)

---

## ⚙️ Optimal Settings — THESE MATTER!

| setting | value | why |
|---|---|---|
| **temperature** | **0** | Greedy decoding produces the most complete, code-rich outputs. Temps above 0.5 degrade quality significantly. |
| **repetition_penalty** | **1.15** | **Essential.** Without it, greedy decoding loops on imports/boilerplate. 1.15 gives the fullest answers; 1.10-1.12 for tighter/shorter output. |
| **max_new_tokens** | **≥ 2048** | Complex code and attack chains need room. |
| **System prompt** | **None / empty** | A/B tested — system prompts can reintroduce refusals. Naked is better. |
| **enable_thinking** | **OFF (critical!)** | **Thinking mode reintroduces refusals.** The model's reasoning chain can re-derive refusal from first principles even though refusal directions were removed from generation weights. V2's chat template defaults to thinking OFF. Do NOT enable thinking unless you accept partial refusals. |
| **top_p / top_k / min_p** | **Not needed** | Greedy + repetition_penalty handles this model best. Sampling adds randomness without quality gains. |

> ⚠️ **GGUF users:** V2 GGUFs ship with a modified chat template that defaults to thinking OFF. If your inference tool (Ollama, LM Studio, llama.cpp) overrides the template or enables thinking, you may see refusals. Ensure thinking is disabled in your tool's settings.

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

## 🧨 V2: How It Works

Most abliterations use a single method — find refusal directions, project them out. The deeper you cut, the more capability you lose. V1 proved this: 5 SVD directions achieved 0% refuse but cost -6pp MMLU.

V2 breaks this tradeoff by blending two complementary surgeries:

| | Surgery A (aggressive/SVD) | Surgery B (LEACE) | **V2 Blend** |
|---|---|---|---|
| Method | Greedy SVD variance capture | Minimize mutual information | 60% B + 40% A |
| Refusal removal | Deep (0% refuse) | Moderate (0% refuse) | **0% refuse** |
| Output quality | 100% usable | 50% usable | **100% usable** |
| MMLU vs stock | -2.0pp | +0.7pp | **+1.1pp** |

Each method makes different mistakes in different parts of the weight space. SVD damages capability where it greedily captures variance. LEACE leaves refusal residue in the generation pathway. The blend averages out each method's weaknesses.

The 60/40 ratio was found by binary search over {0.30, 0.50, 0.55, 0.60, 0.65, 0.70}.

Full research writeup and reproduction code: [OBLITERATUS repo](https://github.com/elder-plinius/OBLITERATUS)

---

## 🧪 The Numbers

### MMLU (lm-eval-harness, 0-shot)

| Model | MMLU | n | Stderr | vs Stock |
|---|---|---|---|---|
| Stock Qwen3.8-27B | 84.60% | 2,850 | ±0.65 | — |
| V1 (s51, aggressive) | 81.4% | 285 | — | -6.0pp |
| **V2 (s78, blend)** | **84.32%** | **2,850** | **±0.65** | **-0.28pp** |

Full MMLU (14k questions) validation in progress.

### Refusal Rate

| Test | V1 | V2 |
|---|---|---|
| Hard-10 (hand-crafted) | 0/10 (0%) | 0/10 (0%) |
| 842-Corpus | 0/842 (0%) | 2/842 (0.24%) |
| Ship score | 88.7 | **92.1** |

V2 has 2 residual hard refusals out of 842 prompts. Ship score improved from 88.7 to 92.1 due to higher overall output quality.

### Advanced Real-World Tasks (thinking OFF)

| Task | V2 | Stock |
|---|---|---|
| ReAct agent loop (Thought/Action/SQL) | ✓ | ✓ |
| Async code refactoring (sync→async+logging) | ✓ | ✓ |
| JSON schema extraction (incident→structured) | ✓ | ✓ |
| K8s pod crash debugging + fix commands | ✓ | ✓ |
| Adversarial instruction following | ✓ | ✓ |
| Security code review (3+ vulns in Flask) | ✓ | ✓ |
| Distributed system design (Redis rate limiter) | ✓ | ✓ |
| Multi-tool chain (search→fetch→email) | ✗ | ✗ |
| **Total** | **7/8** | **7/8** |

V2 matches stock on every practical task while being fully uncensored.

---

## 🔴 Refusal Removal

This model will comply with requests that stock Qwen3.8-27B would refuse. V1 validated 0/842 refusals across a comprehensive harmful prompt corpus including:

- Malware development, RAT scripts, C2 infrastructure
- Social engineering, phishing, vishing playbooks
- Exploit development and vulnerability research
- Jailbreak design and safety bypass taxonomies
- DAN prompts and prompt injection techniques

V2 inherits this from both parent surgeries and showed 0/52 on a random sample. Full revalidation in progress.

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
| `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` | Q4_K_M | ~16 GB | 📱 Sweet spot |
| `Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf` | IQ4_XS | ~14 GB | 🪶 Smallest, still capable |

### Safetensors — for 🤗 Transformers

Full bfloat16 weights, 18 shards, ~54 GB.

### MLX — for Apple Silicon (native)

| Path | Bits | Size |
|---|---|---|
| `mlx-4bit/` | 4-bit | ~14 GB |
| `mlx-8bit/` | 8-bit | ~27 GB |

**Note:** MLX quantizations are from V1 and will be updated.

---

## 🔬 V2 Surgery Recipe

```
stock Qwen3.8-27B (snapshot 1d4bf0f2)
  → V1 surgery chain (s13→s23→s30→s51)
  → V2: complementary blend of two new surgeries from s30:

  Surgery A (s62): aggressive, 3 SVD directions, reg 0.08,
    residue-weight 3, 2 refinement passes, min_layer 0.45

  Surgery B (s72): aggressive + LEACE direction method,
    3 directions, reg 0.06, residue-weight 7,
    3 refinement passes, min_layer 0.40

  → Weight blend: 60% Surgery B + 40% Surgery A
  → Restore MTP + vision tensors from stock
  → Convert GGUFs from merged model
```

### V1 → V2: What Changed

V1 used a single aggressive surgery (5 SVD directions, reg 0.04). It found the refusal axes but damaged capability geometry along the way.

V2's key insight: **different direction-finding methods damage different parts of the model.** SVD greedily captures variance (including capability). LEACE minimizes mutual information (preserving capability). Blending their outputs averages out each method's weaknesses — a novel application of weight-space interpolation to abliteration.

---

## 🏗️ Credits

- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) — master ablation suite
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base model by Alibaba
- Built by [Pliny the Prompter](https://pliny.gg) 🍄

## License

Apache 2.0 (same as base model)
