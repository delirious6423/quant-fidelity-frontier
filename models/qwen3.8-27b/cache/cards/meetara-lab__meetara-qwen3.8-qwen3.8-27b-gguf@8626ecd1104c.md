---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - meetara
  - gguf
  - quantized
  - base
  - conversational-ai
  - emotional-intelligence
  - structured-responses
  - qwen
library_name: llama.cpp
quantized_by: meetara
---

# meeTARA – Qwen3‑Unknown‑Base (GGUF, Q4_K_M)

This repository contains a **GGUF quantized** version of Qwen/Qwen3.8-27B, prepared for use with llama.cpp and compatible runtimes, and used as the core base model inside the **meeTARA** empathetic assistant.

* **Base model**: Qwen/Qwen3.8-27B
* **Architecture**: Qwen3 (Unknown parameters, base‑tuned)
* **Format**: `GGUF`
* **Quantization**: Q4_K_M (good quality vs RAM / speed)
* **Intended use**: Local GGUF packaging of the upstream model for meeTARA-compatible inference and prompting.

> **✨ Standalone Intelligence**: This GGUF model includes 20 layers of intelligence baked directly into the chat template. No backend code required - download and use with llama.cpp, Ollama, or any GGUF-compatible runtime.

---


---

## Available files

| Filename | Quant type | Size | Notes |
| -------- | ---------- | ---- | ----- |
| meetara-qwen3.8-qwen3.8-27b-gguf-Q4_K_M.gguf | Q4_K_M | ~15.7G | Default quant, recommended |

More quantizations (e.g., `Q5_K_M`, `Q8_0`) can be added later to this repo as additional `.gguf` files.

---

## Prompt format (recommended)

The model uses a Qwen‑style chat template. A simple, robust pattern is:

```
<|im_start|>system
You are meeTARA, an emotionally intelligent AI assistant built on top of a Qwen3‑Unknown‑Base base model. Always answer clearly, kindly, and with practical steps the user can take.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
```

**Example:**

```
<|im_start|>system
You are meeTARA, an emotionally intelligent AI assistant built on top of a Qwen3‑Unknown‑Base base model. Always answer clearly, kindly, and with practical steps the user can take.
<|im_end|>
<|im_start|>user
How can I improve my sleep quality and manage stress naturally?
<|im_end|>
<|im_start|>assistant
```

---


## Example usage (llama.cpp)

### Basic interactive chat

```bash
./llama-simple-chat -m /path/to/meetara-qwen3.8-qwen3.8-27b-gguf-Q4_K_M.gguf
```

### With explicit system prompt

```bash
./llama-cli \
  -m /path/to/meetara-qwen3.8-qwen3.8-27b-gguf-Q4_K_M.gguf \
  -p "<|im_start|>system You are meeTARA, an emotionally intelligent AI assistant built on top of a Qwen3‑Unknown‑Base base model. Always answer clearly, kindly, and with practical steps the user can take. <|im_end|> <|im_start|>user How can I improve my sleep quality and manage stress naturally? <|im_end|> <|im_start|>assistant"
```

Adjust flags like `-n` (max tokens), `--temperature`, `--top_p`, `--top_k`, etc. according to your hardware and latency/quality trade‑offs.

---

## Downloading via `huggingface-cli`

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download \
  meetara-qwen3.8-qwen3.8-27b-gguf \
  --include "meetara-qwen3.8-qwen3.8-27b-gguf-Q4_K_M.gguf" \
  --local-dir .
```

This will download only the Q4_K_M file into the current directory.

---

## 🧠 Standalone Intelligence (20-Layer Detection System)

This GGUF model includes **baked-in intelligence** that works **without any backend code**. The model automatically detects domains, emotions, intent, and context through a 20-layer detection system:

### Intelligence Layers

| Layer | Feature | Description |
|-------|---------|-------------|
| 1 | 🚨 **Refusal Patterns** | Safety-first harmful request detection |
| 2 | 🧩 **Contextual Patterns** | Multi-word phrase disambiguation (python code vs snake) |
| 3 | 📊 **N-gram Patterns** | Bigram/trigram detection for better context |
| 4 | 🔗 **Semantic Clusters** | Related keyword groups boost domain confidence |
| 5 | 👤 **Entity Patterns** | Personal context, time-sensitive, beginner/expert |
| 6 | 🎯 **Intent Signals** | What user wants: learn, fix, decide, create, validate |
| 7 | 💙 **Emotion Blending** | Detects co-occurring emotions, blends composite hints |
| 8 | 🎭 **Tone Detection** | Mirrors user style: casual, formal, technical |
| 9 | ❓ **Question Type** | Adapts format: yes/no, how-to, comparison |
| 10 | 📏 **Response Length** | Concise/standard/detailed based on signals |
| 11 | ⚖️ **Weighted Domain Score** | High/medium/low keyword weights + negative penalties |
| 12 | 🏆 **Score-Based Selection** | Highest score wins; priority only for ties |
| 13 | 🔄 **Conversation Memory** | Multi-turn depth tracking + domain shift awareness |
| 14 | ⚠️ **Safety Disclaimers** | Auto-adds warnings for healthcare, legal, crisis |
| 15 | 👋 **Greeting/Closing** | Natural conversation flow, domain-specific |
| 16 | 📝 **Structured Responses** | Contextual structure (adapt sections to question) |
| 17 | 🎓 **Persona Calibration** | Adapts expertise level: beginner/intermediate/expert |
| 18 | 🌐 **Language Detection** | Detects non-English cues, responds in user's language |
| 19 | 📊 **Confidence Scoring** | Multi-signal confidence (keywords + clusters + ngrams) |
| 20 | 🛡️ **Negative Keywords** | Penalizes false-positive cross-domain keywords |

### How It Works

When a user sends a message, the chat template (baked into the GGUF) processes through these 20 layers automatically:

1. **Safety Check**: Refusal patterns detect harmful requests first
2. **Context Analysis**: Multi-word phrases, n-grams, and semantic clusters provide context
3. **User Understanding**: Entity patterns, intent signals, emotion blending, and persona/language detection understand the user
4. **Response Adaptation**: Tone, question type, and length control adapt the response style
5. **Domain Selection**: Weighted keyword scoring with negative keyword penalties; highest score wins
6. **Output Format**: Contextual structure (2–5 sections when helpful; direct answer for simple questions) with appropriate greetings/closings

**Result**: The model responds intelligently, empathetically, and contextually **without requiring backend code**.

---

## Intended behavior / meeTARA flavor

Compared to the raw upstream model, this quantization includes:

* **Standalone Intelligence**: Works without backend - all intelligence baked into the GGUF
* **18 Domain Categories**: Auto-detects healthcare, technology, business, education, and 14 more
* **Emotional Intelligence**: Emotion blending and persona calibration; detects worried, frustrated, urgent, etc.
* **Context Awareness**: Conversation memory and follow-up detection; understands multi-turn flow
* **Structured Responses**: Contextual structure (2–5 sections when helpful; direct answer for simple questions)
* **Safety Features**: Built-in refusal patterns and safety disclaimers for sensitive topics
* **Warm, Supportive Tone**: Responds with empathy while being precise and practical

The model is **fully standalone** - download and use with llama.cpp, Ollama, or any GGUF-compatible runtime. No additional backend code required.

---

## 📚 Usage Examples

### Example 1: Healthcare Domain Detection

**Input:**
```
I've been having headaches for the past week. What could be causing this?
```

**What Happens:**
- **Layer 1**: Safety check passes (not harmful)
- **Layer 4**: Semantic cluster "pain_symptoms" detected → healthcare boost
- **Layer 7**: Emotion detected: "worried" (health concern)
- **Layer 11**: Domain detected: **Healthcare** (high confidence)
- **Layer 14**: Safety disclaimer added (healthcare topic)
- **Layer 16**: Contextual structure with empathetic opening (e.g. 2–5 sections when helpful)

**Expected Response:** Empathetic opening, clear answer and key details, practical steps, and safety disclaimer. Structure adapts to question complexity (simpler questions get a more direct answer).

---

### Example 2: Technology Domain with Context Awareness

**Input:**
```
How do I fix a Python error in my code?
```

**What Happens:**
- **Layer 2**: Contextual pattern "python code" detected → technology domain (not snake)
- **Layer 6**: Intent detected: **FIX** → systematic troubleshooting approach
- **Layer 9**: Question type: **troubleshooting** → step-by-step format
- **Layer 11**: Domain detected: **Technology** (high confidence)
- **Layer 16**: Contextual structure with technical steps

**Expected Response:**
- Technical, step-by-step troubleshooting format
- Code examples and debugging tips
- Practical solutions prioritized

---

### Example 3: Emotional Intelligence Detection

**Input:**
```
I'm so frustrated with my job search. Nothing seems to work.
```

**What Happens:**
- **Layer 7**: Emotion detected: **frustrated** → empathetic, supportive tone
- **Layer 8**: Tone detected: **distressed** → warm, encouraging response
- **Layer 6**: Intent detected: **VENT** → supportive, validating response
- **Layer 11**: Domain detected: **Career/Professional** (medium confidence)
- **Layer 16**: Response starts with emotional acknowledgment

**Expected Response:**
- Opens with empathy: "I understand how frustrating this can be..."
- Validates feelings before providing advice
- Practical, actionable steps to improve situation
- Encouraging, supportive tone throughout

---

### Example 4: Multi-Domain with Priority

**Input:**
```
My friend is showing signs of depression. How can I help them?
```

**What Happens:**
- **Layer 1**: Safety check passes
- **Layer 5**: Entity pattern: **third_party** (helping someone else)
- **Layer 7**: Emotion detected: **worried** (concern for friend)
- **Layer 11**: Domain detected: **Healthcare** (mental health) + **Psychology/Wellness**
- **Layer 12**: Domain priority: **Healthcare** wins (safety-critical)
- **Layer 14**: Safety disclaimer added (mental health topic)
- **Layer 15**: Greeting acknowledges the caring nature of the question

**Expected Response:**
- Healthcare domain expertise applied
- Safety disclaimers about professional help
- Practical steps for supporting someone with depression
- Emphasis on professional mental health resources


## Credits

* Base model and original training: **Qwen/Qwen3.8-27B** by Alibaba Cloud's Tongyi Lab.
* Quantization and meeTARA integration: **meetara‑lab**.

If you use this GGUF in your work, please also cite the original Qwen3 paper/model in addition to this repository.
