---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - moe
  - gptq
  - compressed-tensors
  - llm-compressor
  - code
  - tool-calling
  - russian
language:
  - en
  - ru
---

# Humo-Coder-35B-A3B-GPTQ-Int4

**[Русская версия — ниже](#по-русски)**

A 4-bit GPTQ quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) with a replaced chat template and a custom calibration set focused on code, tool calling and Russian.
Runs in vLLM on a single GPU with 32 GB or more (weights take 22.3 GiB; the rest goes to the KV cache). We tested it on a 48 GB RTX A6000.

**Like Ornith-1.5 itself, the model is aimed first of all at programming and agentic coding.**

Humo (Хумо) is the mythical bird of happiness of Central Asian and Iranian folklore — continuing the ornithological naming of the base model.

**Nothing was trained.** Humo-Coder is Ornith-1.5 with a different chat template, quantized. All credit for the model's abilities belongs to the Ornith authors. Ornith-1.5 itself is built on Qwen: according to its authors, with continued pretraining, mid-training and post-training, plus reinforcement learning on tasks the model generates for itself.

## Where the idea came from

While looking for a coding model for our agentic workloads, we found [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) and, soon after, [Tiel-Coder](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) by peculiar-ragdoll. Its author reports that the same Ornith weights, paired with the [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) (built on froggeric's work) and carefully quantized, make a noticeably stronger coder. It exists only as GGUF, and we needed a checkpoint for vLLM — so we repeated the idea for safetensors: the same base and template, our own GPTQ recipe and a calibration set tilted towards code, tool calling and Russian.

Many thanks to the authors of Ornith-1.5, Tiel-Coder and the Sharp template. If Humo-Coder works well for you, most of the credit is theirs.

## Provenance and credits

| component     | source                                                                                                                                                                                                                   | license    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| weights       | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)                                                                                                                                      | MIT        |
| chat template | [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates); structural work by **froggeric**                                                                        | Apache-2.0 |
| idea          | [peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) — the same base model and chat template, quantized by its author with their own recipe and importance matrix | —          |
| quantization  | this repository, `llm-compressor`                                                                                                                                                                                        | MIT        |

## Quantization recipe

`llm-compressor` `GPTQModifier` (full recipe in `recipe.yaml`). Despite "GPTQ-Int4" in the name, the checkpoint is stored in the **compressed-tensors** format, not in the old AutoGPTQ format: it loads in vLLM and transformers, but not in AutoGPTQ loaders. "Int4" refers to the routed experts, which hold almost all of the weights; the group size is 32, not the more common 128.

| module                                                                                       | format                                             |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| routed experts                                                                               | INT4, symmetric, group size 32, `actorder: weight` |
| shared expert                                                                                | INT8, symmetric, per-channel                       |
| attention (full and linear), router, shared-expert gate, vision encoder, MTP head, `lm_head` | bf16, untouched                                    |

Size on disk: 22.16 GiB (the bf16 original is ~67 GiB), in five safetensors shards of up to 5 GB each.

**Vision.** The model keeps an image and video encoder (333 tensors, plus `preprocessor_config.json` and `video_preprocessor_config.json`), inherited from Ornith-1.5 unchanged. We did not quantize it and **did not test image or video input at all**. Note that image tokens are processed by the quantized language model, so vision quality may differ from the bf16 Ornith even though the encoder itself is untouched.

**Calibration:** 1270 samples from public datasets only — Magicoder-OSS-Instruct-75K, the-stack-smol / the-stack-dedup (Go, PL/SQL, TypeScript, …), hermes-function-calling-v1, Vikhrmodels/GrandMaster-PRO-MAX, IlyaGusev/saiga_scored, fineweb-2 (rus_Cyrl), OpenR1-Math-220k. Roughly: code and tool calling ~75%, Russian text ~40% (overlapping with the first through Russian coding instructions), domain text (PL/SQL, legal, banking) ~10%. Quantization took about 21 hours.

## Serving

Tested only with **vLLM 0.29.0** (`vllm/vllm-openai:v0.29.0`) on a single RTX A6000 48 GB:

```
vllm serve ooptimum/Humo-Coder-35B-A3B-GPTQ-Int4 \
  --max-model-len 131072 --gpu-memory-utilization 0.94 \
  --enable-prefix-caching --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_xml
```

At these settings vLLM reports 22.29 GiB for weights and 19.86 GiB of KV cache (985 541 tokens, ~7.5 full 128K windows).

Sampling used in all measurements below: temperature 0.6, top_p 0.95, top_k 20, min_p 0. Pass them explicitly — vLLM otherwise takes defaults from `generation_config.json`.

## How it was evaluated

Compared head-to-head with five public 4-bit quants of Qwen3.6-35B-A3B and with the official Qwen3.5-35B-A3B-GPTQ-Int4, same hardware and settings, one run per task, pairwise McNemar tests with a Bonferroni correction. Full write-up (in Russian): [article on Habr](https://habr.com/ru/articles/1088512/).

Selected results (raw score — a response cut off by the budget counts as a failure):

| test                                                     | Humo-Coder   | range of the other six |
| -------------------------------------------------------- | ------------ | ---------------------- |
| agentic coding loop, Go, 318 LiveCodeBench tasks, solved | **202**      | 132–167                |
| BFCL, memory categories                                  | **71.4%**    | 48.4–57.8%             |
| BFCL, multi-turn, solved of 800                          | 385 (last)   | 424–445                |
| GSM8K                                                    | 96.4%        | 92.6–97.4%             |
| MMLU-Pro                                                 | 82.9% (last) | 83.8–85.4%             |
| MBPP+                                                    | 97.5%        | 93.5–98.0%             |
| MultiPL-E Go                                             | 74.0%        | 53.9–82.5%             |

In the agentic loop Humo-Coder beats every other participant (all six comparisons significant, largest p = 0.0017) and uses by far the fewest tokens: 16.8K per task on average vs 26–44K, 8.3K per solved task vs 13.4–30.1K. On GSM8K and MultiPL-E Go it spends 10–19× fewer tokens than the Qwen3.6 quants.

## GGUF

GGUF builds live in a separate repository: [ooptimum/Humo-Coder-35B-A3B-GGUF](https://huggingface.co/ooptimum/Humo-Coder-35B-A3B-GGUF). They are made from the same bf16 model with llama.cpp, **not** from this GPTQ checkpoint, and **none of the numbers above were measured on them** — IQ3_M in particular may behave noticeably differently.

---

## По-русски

Таблицы и команды — в английской части выше, они одинаковы для обоих языков.

### Что это

Humo-Coder — квантованная в 4 бита [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) с заменённым шаблоном чата и собственным калибровочным набором, нацеленным на код, вызов инструментов и русский язык.
Запускается в vLLM на одной видеокарте от 32 ГБ (веса занимают 22.3 ГиБ, остальное идёт под KV-кэш). Проверяли мы на RTX A6000 с 48 ГБ.

**Как и сама Ornith-1.5, модель нацелена прежде всего на программирование и агентную работу с кодом.**

Хумо — мифическая птица счастья в фольклоре народов Центральной Азии и Ирана: так продолжена орнитологическая линия названий исходной модели.

**Ничего не дообучалось.** Humo-Coder — это Ornith-1.5 с другим шаблоном чата, сжатая квантованием. Все заслуги в способностях модели принадлежат авторам Ornith. Сама Ornith-1.5 построена на Qwen: по словам её авторов, с продолженным предобучением, промежуточным обучением (mid-training) и пост-обучением, а также обучением с подкреплением на задачах, которые модель придумывает себе сама.

### Откуда идея

Подбирая модель для агентной работы с кодом, мы нашли [Ornith-1.5](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), а вскоре и [Tiel-Coder](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) от peculiar-ragdoll. По словам его автора, те же веса Ornith в паре с [шаблоном Sharp](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) (основанным на работе froggeric) и при аккуратном квантовании заметно лучше пишут код. Tiel-Coder существует только в GGUF, а нам был нужен вариант для vLLM, поэтому мы повторили идею для safetensors: та же база и тот же шаблон, но свой рецепт GPTQ и калибровочный набор с упором на код, вызов инструментов и русский язык.

Большое спасибо авторам Ornith-1.5, Tiel-Coder и шаблона Sharp. Если Humo-Coder вам хорошо служит, основная заслуга — их.

### Рецепт квантования

Квантовано `llm-compressor` (`GPTQModifier`), полный рецепт — в `recipe.yaml`, разбивка по модулям — в таблице «Quantization recipe» выше. Несмотря на «GPTQ-Int4» в названии, модель сохранена в формате **compressed-tensors**, а не в старом формате AutoGPTQ: она загружается в vLLM и transformers, но не в загрузчиках AutoGPTQ. «Int4» относится к маршрутизируемым экспертам — это почти все веса модели. Размер группы — 32, а не привычные 128. Общий эксперт сжат в INT8, всё остальное — внимание, маршрутизатор, зрительный кодировщик, MTP-голова, `lm_head` — оставлено в bf16 без изменений.

Размер на диске — 22.16 ГиБ (исходная bf16-модель — около 67 ГиБ), пять файлов safetensors по 5 ГБ и меньше.

**Зрение.** В модели есть кодировщик изображений и видео (333 тензора и файлы `preprocessor_config.json`, `video_preprocessor_config.json`), унаследованный от Ornith-1.5 без изменений. Мы его не сжимали и **работу с картинками и видео не проверяли вовсе**. Учтите, что визуальные токены обрабатывает сжатая языковая часть, поэтому качество работы с изображениями может отличаться от bf16-версии Ornith, хотя сам кодировщик не тронут.

**Калибровка:** 1270 образцов только из публичных датасетов — Magicoder-OSS-Instruct-75K, the-stack-smol / the-stack-dedup (Go, PL/SQL, TypeScript и другие), hermes-function-calling-v1, Vikhrmodels/GrandMaster-PRO-MAX, IlyaGusev/saiga_scored, fineweb-2 (rus_Cyrl), OpenR1-Math-220k. Примерно: код и вызов инструментов — около 75%, русский текст — около 40% (частично пересекается с первым за счёт русских инструкций к коду), профильные тексты (PL/SQL, право, банки) — около 10%. Квантование заняло около 21 часа.

### Запуск

Проверено только на **vLLM 0.29.0** (`vllm/vllm-openai:v0.29.0`) на одной RTX A6000 48 ГБ, команда — в разделе «Serving» выше. С этими настройками vLLM отводит 22.29 ГиБ под веса и 19.86 ГиБ под KV-кэш — около 7.5 полных окон по 128 тысяч токенов.

Во всех наших замерах параметры генерации были такими: temperature 0.6, top_p 0.95, top_k 20, min_p 0. Передавайте их явно — иначе vLLM возьмёт значения из `generation_config.json`.

### Как проверяли

Модель сравнивалась с пятью публичными 4-битными квантами Qwen3.6-35B-A3B и с официальным Qwen3.5-35B-A3B-GPTQ-Int4 на одном железе и с одинаковыми настройками, по одному прогону на задачу, с парными тестами Макнемара и поправкой Бонферрони. Подробный разбор — в [статье на Хабре](https://habr.com/ru/articles/1088512/), избранные результаты — в таблице «How it was evaluated» выше. Баллы сырые: ответ, оборванный по бюджету токенов, считается провалом.

В агентном цикле Humo-Coder обыгрывает всех остальных участников (все шесть сравнений значимы, наибольшее *p* = 0.0017) и тратит заметно меньше всех токенов: в среднем 16.8 тысячи на задачу против 26–44 тысяч, 8.3 тысячи на решённую задачу против 13.4–30.1 тысячи. На GSM8K и MultiPL-E Go он тратит в 10–19 раз меньше токенов, чем кванты Qwen3.6.

### GGUF

Сборки GGUF лежат в отдельном репозитории: [ooptimum/Humo-Coder-35B-A3B-GGUF](https://huggingface.co/ooptimum/Humo-Coder-35B-A3B-GGUF). Они сделаны из той же bf16-модели в llama.cpp, а **не** из этого GPTQ-чекпойнта, и **ни одно из чисел выше на них не измерялось**. Особенно это касается IQ3_M: он может вести себя заметно иначе.
