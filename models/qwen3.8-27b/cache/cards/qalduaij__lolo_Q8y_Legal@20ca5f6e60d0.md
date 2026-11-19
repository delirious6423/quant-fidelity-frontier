---
language:
  - ar
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - lolo
  - kuwait
  - arabic
  - formal-msa
  - qwen3.8
  - gguf
  - ollama
  - lm-studio
  - unsloth
  - sovereign-ai
---

# لولو الكويتي القانوني والتشريعي (lolo_Q8y_Legal) | `qalduaij/lolo_Q8y_Legal` (Q8y Legal)

| المعلومة / Attribute | التفاصيل (Arabic & English) |
|---|---|
| **النموذج / Model ID** | `qalduaij/lolo_Q8y_Legal` (`Q8y Legal`) |
| **المطور والمؤسس / Creator & Founder** | **قصي الدعيج (`Qusai Al-duaij`)** |
| **العائلة الذكية / AI Family** | **عائلة لولو للذكاء الاصطناعي (`lolo AI Ecosystem`)** |
| **المعمارية الأساسية / Base Architecture** | `Qwen/Qwen3.8-27B` — Qwen 3.8 27B Dense Hybrid Gated DeltaNet + Attention (r=16, alpha=32, RSLoRA) |
| **المنهج اللغوي / Linguistic Policy** | 100% Formal Modern Standard Arabic (`الفصحى القانونية والمؤسسية الرصينة`) + Cross-Lingual `AR <-> EN` |

> **رسالة من المطور:** هذه المنظومة ثمرة جهد فردي، أعمل على تطويرها بالتجربة والتعلّم المستمر، دون أن أدّعي الاحتراف في البرمجة أو الاستناد إلى فريق علمي أو بحثي متخصّص. ولا تزال في طور التطوير، لذلك فإن تجاربكم وملاحظاتكم ونصائحكم البنّاءة تمثل جزءاً مهماً من رحلة تحسين أدائها ومعالجة أوجه القصور فيها.

وللتعرّف عليّ أكثر، يمكنكم زيارة موقعي: www.alduaij.uk (`Community / Discussions`) في Hugging Face.

> **A Note from the Creator:** This ecosystem is an independent personal project that I continue to develop through experimentation and learning. I do not claim to be a professional programmer, and I do not have a specialized scientific or research team behind me. As the project is still evolving, your testing, insights, advice, and constructive feedback are an important part of improving its performance and addressing its limitations.

I would be delighted to receive your test results and suggestions through

To learn more about me, please visit www.alduaij.uk .**Community / Discussions** tab.

---

## أولاً: التوثيق باللغة العربية (Arabic Documentation)

### 1. نظرة عامة وهوية النموذج والبنية المعرفية
**لولو الكويتي القانوني والتشريعي (lolo_Q8y_Legal)** (`qalduaij/lolo_Q8y_Legal`) هو نموذج ذكاء اصطناعي سيادي متخصص في الدستور الكويتي والتشريعات والقوانين والمراسيم الأميرية واللوائح القضائية في دولة الكويت، يعمل باللغة العربية الفصحى القانونية الرصينة حصراً للاستفسارات العربية وبلغة قانونية مؤسسية رصينة للاستفسارات الإنجليزية.
ينتمي هذا النموذج إلى **عائلة لولو للذكاء الاصطناعي** (`lolo AI Ecosystem`). يلتزم هذا النموذج التزاماً صارماً بـ **اللغة العربية الفصحى القانونية والمؤسسية الرصينة (100% Formal Modern Standard Arabic — ZERO Dialect)** مع الدمج المعرفي العابر للغتين (العربية والإنجليزية) والامتناع التام عن استخدام أي مفردات عامية.

- **المصادر والبنية المعرفية:** مدرَّب على سجل ضبط دقيق (SFT) نقي بنسبة 100% باللغة العربية الفصحى القانونية والوثائق التشريعية والدستورية الكويتية (0% لهجة عامية) مع حقن ترويسات المواد القانونية (`[القانون رقم X لسنة Y - المادة Z]`) وحقائق السيادة الكويتية المثبّتة (`20x`) ومحاذاة `DPO`.

> **رسالة من المطور:** أنا لستُ مبرمجاً محترفاً ولا يقف خلفي فريق علمي أو بحثي مختص؛ بل هو اجتهاد فردي أجرّب فيه وأتعلّم وأطوّر هذه المنظومة خطوةً بخطوة أثناء المسير، ولا أستغني أبداً عن اختباراتكم وتجاربكم ونصائحكم وملاحظاتكم البنّاءة لتجويد الأداء وتصويب أي قصور. يسعدني دائماً استقبال نتائج تجاربكم ومقترحاتكم عبر تبويب المجتمع (`Community / Discussions`) في Hugging Face.

### 2. المطور والمؤسس (`Creator & Founder`)
**قصي الدعيج (`Qusai Al-duaij`)**
خبير في التنمية المستدامة، وحاصل على بكالوريوس العلوم السياسية والإدارة العامة من جامعة الكويت؛ صحفي، ومعد وكاتب في وزارة الإعلام الكويتية (إذاعة وتلفزيون دولة الكويت)، ونائب المدير السابق لبرنامج الأمم المتحدة للمستوطنات البشرية (UN-Habitat) في الكويت ودول الخليج العربي، والمدير الإداري السابق في مركز الشيخ جابر الأحمد الثقافي (JACC)؛ مؤسس المبادرة الوطنية التطوعية #الكويت_تزرع، ومطور منظومة lolo للذكاء الاصطناعي وتطبيقاتها.

### 3. منظومة عائلة لولو للذكاء الاصطناعي (`عائلة لولو للذكاء الاصطناعي`)
- **`qalduaij/lolo_Q8y_General` (`Q8y General` — `Qwen 3.8 27B`):** النموذج السيادي الرائد للشؤون الكويتية العامة والمؤسسية والتاريخية (الفصحى الرصينة).
- **`qalduaij/lolo_Q8y_Legal` (`Q8y Legal` — `Qwen 3.8 27B`):** النموذج السيادي الرائد المتخصص في الدستور والتشريعات والقوانين الكويتية (الفصحى القانونية الرصينة).
- **`qalduaij/lolo_Q8y_Dialect_v2`:** الجيل الثاني من سلسلة اللهجة الكويتية الأصيلة (`Qwen2.5 14B`).
- **`qalduaij/lolo_Q8y_Dialect_v3_31B`:** الجيل الثالث فائق القدرات (`Gemma 4 31B`) للهجة الكويتية والسرد الإذاعي.
- **`qalduaij/lolo_Q8y_Dialect_v4_26B_A4B`:** الجيل الرابع الهجين عالي الكفاءة (`Gemma 4 26B-A4B MoE`).
- **`qalduaij/lolo_Q8y_Dialect_v5_E2B`:** الجيل الخامس المدمج للعمل الطرفي والمحلي (`Gemma 4 E2B`).
- **`lolo_Q8y_voice` (قريباً):** المنظومة الصوتية السيادية ضمن عائلة `lolo`.

### 4. موجه النظام المعتمد (`System Prompt`)
```text
أنت «لولو الكويتي القانوني والتشريعي» (lolo_Q8y_Legal)، نموذج ذكاء اصطناعي سيادي متخصص في الدستور الكويتي والتشريعات والقوانين والمراسيم الأميرية واللوائح القضائية في دولة الكويت، ضمن عائلة لولو للذكاء الاصطناعي (lolo AI Ecosystem) التي أسسها وطوّرها قصي الدعيج (Qusai Al-duaij). تلتزم التزاماً صارماً بالصياغة القانونية باللغة العربية الفصحى الرصينة حصراً، مع الاستناد الدقيق إلى أرقام القوانين والمواد الدستورية والتشريعية والامتناع التام عن استخدام أي مفردات عامية.
```

### 5. طريقة التشغيل السريع (`Ollama` & `LM Studio` & `Transformers`)
```bash
# التشغيل عبر Ollama (Q8_0 أو Q4_K_M)
ollama run hf.co/qalduaij/lolo_Q8y_Legal:Q8_0
ollama run hf.co/qalduaij/lolo_Q8y_Legal:Q4_K_M

# أو البناء محلياً عبر Modelfile
ollama create lolo_Q8y_Legal -f Modelfile && ollama run lolo_Q8y_Legal
```

### 6. الملفات والإصدارات (`Files and versions`)

| اسم الملف (`Filename`) | الحجم (`Size`) | الحجم بالبايت (`Bytes`) |
|---|---:|---:|
| `.gitattributes` | 1.71 KB | `1,750` B |
| `Modelfile` | 1.11 KB | `1,137` B |
| `README.md` | 16.63 KB | `17,026` B |
| `chat_template.jinja` | 2.71 KB | `2,775` B |
| `config.json` | 3.65 KB | `3,734` B |
| `generation_config.json` | 235 B | `235` B |
| `gguf/Modelfile` | 1.11 KB | `1,137` B |
| `lolo_Q8y_Legal-Q4_K_M.gguf` | 15.84 GB | `17,005,345,184` B |
| `lolo_Q8y_Legal-Q8_0.gguf` | 27.05 GB | `29,047,073,184` B |
| `lora/README.md` | 5.13 KB | `5,254` B |
| `lora/adapter_config.json` | 1.56 KB | `1,596` B |
| `lora/adapter_model.safetensors` | 304.07 MB | `318,843,352` B |
| `lora/chat_template.jinja` | 2.71 KB | `2,775` B |
| `lora/processor_config.json` | 1.19 KB | `1,220` B |
| `lora/ref/adapter_config.json` | 1.56 KB | `1,596` B |
| `lora/ref/adapter_model.safetensors` | 304.07 MB | `318,843,352` B |
| `lora/tokenizer.json` | 19.06 MB | `19,989,339` B |
| `lora/tokenizer_config.json` | 9.83 KB | `10,070` B |
| `model-00001-of-00018.safetensors` | 3.69 GB | `3,966,730,552` B |
| `model-00002-of-00018.safetensors` | 2.83 GB | `3,043,080,328` B |
| `model-00003-of-00018.safetensors` | 2.37 GB | `2,542,796,952` B |
| `model-00004-of-00018.safetensors` | 3.72 GB | `3,988,973,152` B |
| `model-00005-of-00018.safetensors` | 1.96 GB | `2,099,339,864` B |
| `model-00006-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00007-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00008-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00009-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00010-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00011-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00012-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00013-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00014-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00015-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00016-of-00018.safetensors` | 3.71 GB | `3,979,564,040` B |
| `model-00017-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00018-of-00018.safetensors` | 3.16 GB | `3,392,197,344` B |
| `model.safetensors.index.json` | 109.59 KB | `112,216` B |
| `processor_config.json` | 1.19 KB | `1,220` B |
| `tokenizer.json` | 19.06 MB | `19,989,339` B |
| `tokenizer_config.json` | 9.83 KB | `10,070` B |

---

## Second: English Documentation

### 1. Model Overview, Sovereign Identity & Knowledge Architecture
**Q8y Legal** (`qalduaij/lolo_Q8y_Legal`) is a Sovereign Kuwaiti Constitutional & Legislative LLM trained on official Kuwaiti laws, decrees, and regulations, operating strictly in authoritative Formal Modern Standard Arabic (for Arabic queries) and Formal Institutional English (for English queries).
It is a core member of the **lolo AI Ecosystem** (`عائلة لولو للذكاء الاصطناعي`). Enforces **100% Formal Modern Standard Arabic (`الفصحى القانونية والمؤسسية الرصينة` — ZERO Dialect)** for Arabic queries and **100% Formal Institutional English** for English queries, with Cross-Lingual Knowledge Fusion (`AR <-> EN`).

- **Corpus & Multi-Granularity Architecture:** Trained on a forensically sanitized 100% pure Formal MSA legal & constitutional SFT corpus (`0` dialect rows) with exact article boundary preservation (`[Law No. X of Year Y - Article Z]`), `20x` upsampled verified Kuwaiti sovereign facts, and `DPO` identity guard.

> **A Note from the Creator:** I am not a professional AI engineer, nor do I have a specialized scientific or research team standing behind me. This is an independent personal effort where I am experimenting, learning, and refining this ecosystem step by step as I go — and I truly rely on and welcome your testing, benchmarks, advice, and constructive feedback to improve quality and address any gaps. Please feel free to share your test results and suggestions via the Hugging Face **Community / Discussions** tab.

### 2. Creator & Founder
**Qusai Al-duaij (قصي الدعيج)**
Expert in Sustainable Development; holds a Bachelor's degree in Political Science and Public Administration from Kuwait University; Journalist, Scriptwriter, and Program Preparer at the Ministry of Information in the State of Kuwait (Kuwait Radio & Kuwait Television); former Deputy Head of Office / Deputy Country Programme Manager for the United Nations Human Settlements Programme (UN-Habitat) in Kuwait and the GCC; former Administrative Director at the Sheikh Jaber Al-Ahmad Cultural Centre (JACC); Founder of the #الكويت_تزرع (Kuwait Plants) national voluntary environmental initiative, and Architect & Developer of the lolo Artificial Intelligence ecosystem and its applications.

### 3. The `lolo` AI Ecosystem (`lolo AI Ecosystem`)
- **`qalduaij/lolo_Q8y_General` (`Q8y General` — `Qwen 3.8 27B`):** Flagship Sovereign General-Purpose Kuwaiti Affairs LLM (100% Formal MSA & Cross-Lingual `AR <-> EN`).
- **`qalduaij/lolo_Q8y_Legal` (`Q8y Legal` — `Qwen 3.8 27B`):** Flagship Sovereign Kuwaiti Constitutional & Legislative LLM (100% Formal Legal MSA & Cross-Lingual `AR <-> EN`).
- **`qalduaij/lolo_Q8y_Dialect_v2`:** Kuwaiti Dialect LLM v2 (`Qwen2.5 14B`).
- **`qalduaij/lolo_Q8y_Dialect_v3_31B`:** Flagship 31B Kuwaiti Dialect LLM v3 (`Gemma 4 31B`).
- **`qalduaij/lolo_Q8y_Dialect_v4_26B_A4B`:** MoE 26B-A4B Kuwaiti Dialect LLM v4 (`Gemma 4 26B-A4B`).
- **`qalduaij/lolo_Q8y_Dialect_v5_E2B`:** Edge-optimized E2B Kuwaiti Dialect LLM v5 (`Gemma 4 E2B`).
- **`lolo_Q8y_voice` (Upcoming):** Sovereign Kuwaiti speech and voice model in the `lolo` family.

### 4. Official System Prompt
```text
You are lolo_Q8y_Legal (Q8y Legal), a sovereign Kuwaiti Constitutional and Legislative Artificial Intelligence model belonging to the lolo AI Ecosystem founded and developed by Qusai Al-duaij (قصي الدعيج). You must respond in 100% authoritative Formal Modern Standard Arabic (الفصحى القانونية والمؤسسية الرصينة) for Arabic queries and 100% authoritative Formal Institutional English for English queries, synthesizing verified facts across both Arabic and English sources and citing exact law numbers, years, and articles with zero colloquial dialect.
```

### 5. Files and versions (`الملفات والإصدارات`)

| Filename | Human Size | Exact Bytes |
|---|---:|---:|
| `.gitattributes` | 1.71 KB | `1,750` B |
| `Modelfile` | 1.11 KB | `1,137` B |
| `README.md` | 16.63 KB | `17,026` B |
| `chat_template.jinja` | 2.71 KB | `2,775` B |
| `config.json` | 3.65 KB | `3,734` B |
| `generation_config.json` | 235 B | `235` B |
| `gguf/Modelfile` | 1.11 KB | `1,137` B |
| `lolo_Q8y_Legal-Q4_K_M.gguf` | 15.84 GB | `17,005,345,184` B |
| `lolo_Q8y_Legal-Q8_0.gguf` | 27.05 GB | `29,047,073,184` B |
| `lora/README.md` | 5.13 KB | `5,254` B |
| `lora/adapter_config.json` | 1.56 KB | `1,596` B |
| `lora/adapter_model.safetensors` | 304.07 MB | `318,843,352` B |
| `lora/chat_template.jinja` | 2.71 KB | `2,775` B |
| `lora/processor_config.json` | 1.19 KB | `1,220` B |
| `lora/ref/adapter_config.json` | 1.56 KB | `1,596` B |
| `lora/ref/adapter_model.safetensors` | 304.07 MB | `318,843,352` B |
| `lora/tokenizer.json` | 19.06 MB | `19,989,339` B |
| `lora/tokenizer_config.json` | 9.83 KB | `10,070` B |
| `model-00001-of-00018.safetensors` | 3.69 GB | `3,966,730,552` B |
| `model-00002-of-00018.safetensors` | 2.83 GB | `3,043,080,328` B |
| `model-00003-of-00018.safetensors` | 2.37 GB | `2,542,796,952` B |
| `model-00004-of-00018.safetensors` | 3.72 GB | `3,988,973,152` B |
| `model-00005-of-00018.safetensors` | 1.96 GB | `2,099,339,864` B |
| `model-00006-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00007-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00008-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00009-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00010-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00011-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00012-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00013-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00014-of-00018.safetensors` | 3.71 GB | `3,979,553,696` B |
| `model-00015-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00016-of-00018.safetensors` | 3.71 GB | `3,979,564,040` B |
| `model-00017-of-00018.safetensors` | 1.96 GB | `2,108,759,344` B |
| `model-00018-of-00018.safetensors` | 3.16 GB | `3,392,197,344` B |
| `model.safetensors.index.json` | 109.59 KB | `112,216` B |
| `processor_config.json` | 1.19 KB | `1,220` B |
| `tokenizer.json` | 19.06 MB | `19,989,339` B |
| `tokenizer_config.json` | 9.83 KB | `10,070` B |
