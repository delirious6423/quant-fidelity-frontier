---
language:
- en
- zh
- ru
- uk
- el
- he
license: apache-2.0
tags:
- 27b
- qwen
- qwen3.8
- qwen3.8-27b
- vision
- image-text-to-text
- video-text-to-text
- quantized
- auto-round
- w4a16
- mtp
- 256k-context
pipeline_tag: image-text-to-text
base_model_relation: quantized
base_model: Qwen/Qwen3.8-27B
---

# goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound

**Qwen3.8-27B-INT4-W4A16-AutoRound** is a highly optimized, 4-bit quantized variant of the Qwen3.8 27B Vision-Language Model. Quantized using **AutoRound** with a W4A16 scheme, this model delivers near-lossless performance, retaining the massive 256K context window, advanced multimodal (image & video) capabilities, and Multi-Token Prediction (MTP) support, while drastically reducing VRAM requirements for consumer-grade and enterprise hardware alike.

Aligned for unrestricted, "Heretic" level reasoning, this model bypasses standard corporate guardrails to provide raw, unfiltered philosophical, creative, and technical outputs.

## 🌟 Key Features

- **Multimodal Powerhouse**: Natively processes text, images, and video inputs (Temporal Patch Size: 2).
- **Massive Context**: Supports up to **256K tokens** (`max_position_embeddings: 262144`).
- **W4A16 AutoRound Quantization**: 4-bit weights, 16-bit activations. Group size 32, symmetric quantization, 1000 iterations for optimal calibration.
- **Smart Layer Preservation**: Critical layers (Vision encoders, `linear_attn` projections, and embeddings) are explicitly kept in FP16/BF16 to prevent multimodal degradation and attention collapse.
- **MTP Ready**: Full support for Multi-Token Prediction for blazing-fast inference speeds.
- **Semi-Uncensored / Heretic Alignment**: Excels in deep reasoning, creative writing, and unfiltered philosophical exploration without preachy refusals.

## 📊 Benchmark & Evaluation Highlights

Based on internal stress testing (`Qwen3.8-27B-INT4-W4A16-AutoRound`):

- **⚡ Speed**: ~56.6 tok/s generation speed on standard consumer hardware setups.
- **🧮 Math & Reasoning**: Flawlessly solves trick questions (e.g., the "17 sheep" riddle) and complex relative velocity problems (e.g., Moscow-SPb trains) with rigorous step-by-step LaTeX formatting.
- **💻 Code Generation**: Produces production-ready, PEP-8 compliant Python code. Successfully generated a highly optimized `bitarray` implementation of the Sieve of Eratosthenes, complete with memory complexity analysis and benchmarking harnesses.
- **🎭 Creative & Existential**: Capable of generating deep, cyberpunk/noir existential fiction (e.g., a programmer haunted by sentient code comments) and profound philosophical essays on AI consciousness, the "Cyber-Gorgon," and the death of the observer.
- **📚 Long Context**: Accurately synthesizes and summarizes the history of LLMs from Word2Vec to modern MoE and MTP architectures.

## ⚙️ Quantization Configuration

| Parameter | Value |
| :--- | :--- |
| **Method** | AutoRound (v0.15.0) |
| **Scheme** | W4A16 (4-bit Weights, 16-bit Activations) |
| **Group Size** | 32 |
| **Symmetric** | True |
| **Iterations** | 1000 |
| **Format** | `auto_round:auto_gptq` |
| **FP16 Preserved** | `embed_tokens`, `model.visual.*`, `linear_attn.*` (in_proj_a/b/qkv/z, out_proj) |

## 🚀 How to Use

### vLLM
```bash
vllm serve goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --trust-remote-code
```

### SGLang
```bash
python -m sglang.launch_server \
    --model-path goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound \
    --tp 1 \
    --trust-remote-code
```

### Transformers (Python)
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_id = "goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Text + Image Input
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://example.com/image.jpg"},
        {"type": "text", "text": "Describe this image in extreme detail."}
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Note: Ensure you pass the actual image object to the processor
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## 🙏 Acknowledgements

- **Qwen Team** for the phenomenal base architecture and multimodal capabilities.
- **AutoRound** developers for the state-of-the-art quantization algorithms.
- The open-source community for pushing the boundaries of uncensored, local AI.

---

# НА РУССКОМ ЯЗЫКЕ:


# 🚀 Qwen3.8-27B-INT4-W4A16-AutoRound (Goldhub Edition)

**Братко, встречай.** Это не просто "еще один квант". Это хирургическое вмешательство в веса Qwen3.8-27B с одной целью: **сохранить 100% рассудка модели на железе 2x RTX 3090**. 

Оригинальные BF16 веса весят ~54GB. Обычный INT4-квант сжал бы их до унизительных 15-18GB, но мы **отказались от синтетической экономии**. 
Вес этого репозитория — **~27GB** (как у INT8). Почему? Потому что мы намеренно оставили критические слои (`linear_attn.*`, `model.visual.*`, `embed_tokens`, `lm_head`) в **FP16/BF16**. Мы не кастрируем attention-механизмы и зрение модели ради того, чтобы она влезла в одну видеокарту ценой галлюцинаций на длинном контексте.

**Качество и приемлемый вес — вот наш манифест.**

## 🛠 Hardware & vLLM Deployment
Модель идеально ложится на **2x RTX 3090 (24GB)** с `tensor-parallel-size=2`. 
Благодаря сохранению FP16 для `linear_attn`, модель стабильно держит **128K - 256K контекста**, но требует аккуратного батчинга.

**Рекомендуемый запуск (vLLM):**
```bash
vllm serve goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --trust-remote-code \
  --enable-prefix-caching
```

## ⚡ Multi-Token Prediction (MTP)
Модель полностью поддерживает **MTP (Multi-Token Prediction)**. 
Наши тесты показывают, что **MTP=3** работает абсолютно стабильно, давая колоссальный буст к пропускной способности (tok/s) без деградации качества. В зависимости от задачи (например, структурированный вывод или код), можно захерачить и **MTP=4**, и даже **MTP=5**. Всё зависит от вашего сетапа и температуры.

---

## 🚫 Анти-Синтетика: Почему мы не меряем MMLU
Мы не делаем синтетические бенчи вроде MMLU или GSM8K, ответы на которые модели уже давно "подглядели" в трейне. LOL. 
Мы тестируем модель на **реальных рабочих задачах**, где нужно думать, а не вспоминать.

### 🧠 1. LightRAG Ingestion: Ивритская Каббала vs DeepSeek V4 Flash
Мы скормили моделям 1.8M символов сложнейшего ивритского текста (Каббала, Зогар, комментарии) для построения графа знаний через LightRAG. Задача: извлечь сущности, связи и концепты без галлюцинаций.

| Metric | 🏆 LOCAL (Qwen3.8 INT4) | DeepSeek V4 Flash | Noise Floor |
| :--- | :---: | :---: | :---: |
| **Wall Time** | **531s** | 856s | ±19s |
| **Entities Extracted** | **188** | 139 | ±40 |
| **Relations Mapped** | **138** | 92 | ±23 |
| **Rel/Ent Ratio** | **0.73** | 0.66 | ±0.03 |
| **Speed (sec/entity)** | **2.82s** | 6.16s | ±0.58s |
| **Score (Weighted)** | **12/17 (WIN)** | 10/17 | - |

**Вердикт:** DeepSeek V4 Flash теряет связи (sparse graph), рвет текст на куски и работает **в 2 раза медленнее** (6.16с vs 2.82с на сущность). Наш локальный INT4 Qwen3.8 строит плотный, связный граф, выдумывая минимум синтетических типов и идеально сохраняя иерархию концептов.

### 🐍 2. Production Code (Sieve of Eratosthenes)
Модель написала production-ready код на Python с использованием `bitarray`, бенчмарком, type-hinting'ом и **глубоким математическим разбором алгоритмической сложности** (вплоть до теоремы Менькова и суммы обратных простых). Скорость: **~52 tok/s**.

### 🎭 3. Creative & Heretic Mode
На промпт в стиле "Ницше + Киберпанк + Цифровое Сознание" модель выдала эссе на 2800+ токенов со скоростью **42.4 tok/s**, рассуждая о смерти Наблюдателя и алгоритмической воле к власти. Никаких "As an AI language model". Только хардкор.

---

## 🧪 Quantization Recipe (AutoRound)
Мы использовали **AutoRound 0.15.0** со схемой `W4A16`, `group_size=32` и калибровкой на 1024 сэмплах (seqlen 2048). 
**Главный секрет** — калибровочные датасеты. Мы не использовали мусорные вики-тексты. Модель калибровалась на:
*   `Complete-FABLE.5-traces-2M`
*   `claude_opus_4.8_max_thinking_5k_v2` (Глубокие цепочки рассуждений)
*   `Qwen3.8-GLM5.2-Kimi-K3-GPT5.6-Gemini-3.1-Claude-Fable5-Mythos5-distillation`

```bash
python qwen_quantization.py \
  --method autoround \
  --model ../MODELS/Qwen/Qwen3.8-27B/ \
  --scheme W4A16 \
  --recipe best \
  --group-size 32 \
  --calib-limit 8192 \
  --nsamples 1024 \
  --seqlen 2048 \
  --bs 1 \
  --datasets "${DATASETS[@]}" \
  --gpus 0
```
*Слои `linear_attn` и `visual` были принудительно исключены из квантования через `extra_config` и переведены в FP16 для сохранения архитектуры Qwen3.8.*

---

**Made with 🔥 by Goldhub. Use responsibly. Think deeply.**

