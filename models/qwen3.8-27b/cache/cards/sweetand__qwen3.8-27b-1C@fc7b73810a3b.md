---
license: apache-2.0
language:
- ru
- en
base_model:
- Qwen/Qwen3.8-27B
tags:
- 1C
- 1С
- bsl
- onescript
pipeline_tag: image-text-to-text
---
# Qwen3.8-27B-1C

SFT-адаптация [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) для разработки на платформе **1C:Enterprise**: код на **BSL**, структура выгрузок **BSL+XML**, схемы XML и типовые практики конфигураций.
Модель ориентирована на ассистента разработчика 1С: навигация по метаданным/XML-выгрузке, пояснение и правка BSL, следование внутренним стандартам кодирования, работа с открытыми кодовыми базами 1С.

## Model Overview

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Stage | Supervised Fine-Tuning (SFT) |
| Domain | 1C:Enterprise / BSL / XML |
| Languages | Russian (primary), English |
| Context | inherits Qwen3.8 native window (up to 262K) |
| Adapter | LoRA adapter |

## Training data (SFT)

Обучение выполнялось на SFT-датасетах по темам:
- понимание структуры проекта **BSL + XML**;
- стандарты разработки на платформе **1C:Enterprise**;
- особенности **XML-схем** файлов конфигурации;
- открытые кодовые базы 1С (чтение, объяснение, доработка паттернов).
- датасеты по осмысленному именованию процедур, функций, методов, параметров, переменных и доменных сущностей. Учит модель придумывать названия по роли, эффекту, результату и предметной области, а не хаотично.
- датасеты понимания языка запросов 1С.


Данные в основном instructional / chat-формата с длинными контекстами (user → assistant) под рабочие задачи разработчика, а не под общий чат.

## Intended use

**Подходит для:**
- вопросов по расположению объектов/модулей в XML-выгрузке;
- генерации и рефакторинга BSL с учётом стиля 1С;
- пояснения типовых механизмов платформы и конфигураций;
- помощи в code review и разборе чужого BSL.
**Не подходит как:**
- замена официальной документации 1С и аудита ИБ;
- источник «истины» по закрытым/проприетарным конфигурациям без вашей проверки;
- гарантия компилируемого/регламентно верного кода без прогона в конфигураторе / CI.

## How to use
Используйте chat template базовой Qwen3.8. Для точных coding-задач (в т.ч. 1С) разумно стартовать с рекомендованных Qwen sampling-параметров для precise coding:
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=0.0`
- `repetition_penalty=1.3`
- 'effort=medium'
Thinking/non-thinking режимы и tool-calling наследуются от базовой Qwen3.8 — см. [model card базы](https://huggingface.co/Qwen/Qwen3.8-27B).
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "YOUR_HF_USER/Qwen3.8-27B-1C"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
messages = [
    {"role": "user", "content": "Где в выгрузке конфигурации обычно лежит модуль объекта справочника?"}
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```
Для GGUF / llama.cpp / OpenAI-compatible endpoint используйте ваш обычный export/serve pipeline после merge (если на HF лежит adapter).

## Limitations

Возможны галлюцинации имён объектов метаданных, путей XML и «типовых» API.
Поведение вне домена 1С может быть слабее или нестабильнее базы.
Длинный контекст (128K+) сильно зависит от железа.
Ответы нужно проверять в конфигураторе / на тестах (в т.ч. Vanessa), особенно для продакшена.

## License
База Qwen3.8-27B: Apache-2.0. Производная модель — на тех же условиях, если отдельно не указано иное. Соблюдайте лицензии исходных open-source кодовых баз, вошедших в обучающую выборку.

## Acknowledgements
Qwen/Qwen3.8-27B
сообщество и open-source проекты экосистемы 1С
