---
language:
- en
- ru
license: apache-2.0
tags:
- legacy
- archived
- gguf
- qwen
- qwen3.8
- 27b
- heavyweight
- character
- persona
- troll
- trolling
- chaos
- unhinged
- post-irony
- dark-humor
- conversational
- fine-tune
- text-generation
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
model_name: Luna 3.8 27B GGUF (Archived)
---

# Luna 3.8 27B GGUF — archived edition / прошлая версия

> **ARCHIVED / ПРОШЛАЯ ВЕРСИЯ.** This repository preserves an older Luna release.
> **Актуальная версия / Current release:** [Luna v2 · 9B GGUF](https://huggingface.co/wayzer2321/Luna-v2-9B-GGUF).
> Все релизы / Release catalog: [luna-all-versions](https://huggingface.co/wayzer2321/Luna-Release-Catalog).

## Release status

This is a separate older 27B checkpoint. It is not a 27B edition of the current Luna v2 checkpoint.

Этот репозиторий оставлен для старых загрузок и сравнения версий.
Для знакомства с актуальной Луной используйте **Luna v2**, ссылка выше.
Название базовой модели и число параметров не обозначают порядок релизов Луны.

## Archived model details

| Property | Value |
| :--- | :--- |
| Status | Archived / previous release |
| Checkpoint | Older separate Luna 27B edition |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| File | `luna3.8-27b-q3.gguf` |
| Quantization | Q3_K_M |
| Format | GGUF |

Luna is Wayzer's conversational character: informal slang, sarcasm,
surreal humor, and creator lore. Responses from this archived checkpoint
are not representative of the current Luna v2 release.

## Historical sampling settings

These are the settings previously published for this archived model:

- Temperature: `0.75–0.88`
- Repeat penalty: `1.08–1.10`
- Top-p: `0.92`
- Context length: `2048`

Memory use and generation speed depend on hardware, context, and backend.

## Local import (Ollama)

For this archived file only:

```text
FROM ./luna3.8-27b-q3.gguf
```

Save the line above in `Modelfile` next to the GGUF, then run:

```bash
ollama create luna-27b-archive -f Modelfile
ollama run luna-27b-archive
```

## Historical dialogue archive

[Earlier dialogue examples and release catalog](https://huggingface.co/wayzer2321/Luna-Release-Catalog).
These examples are historical and are not a benchmark of Luna v2.
