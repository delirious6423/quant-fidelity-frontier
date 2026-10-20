---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - imatrix
  - llama.cpp
  - moe
  - code
  - tool-calling
  - russian
language:
  - en
  - ru
---


# Humo-Coder-35B-A3B-GGUF

**[Русская версия — ниже](#по-русски)**

GGUF quantizations of Humo-Coder-35B-A3B — [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) with a replaced chat template — for llama.cpp and compatible runtimes (LM Studio, Ollama, KoboldCpp…).

**Like Ornith-1.5 itself, the model is aimed first of all at programming and agentic coding.**

Humo (Хумо) is the mythical bird of happiness of Central Asian and Iranian folklore — continuing the ornithological naming of the base model.

**Nothing was trained.** All credit for the model's abilities belongs to the Ornith authors. Ornith-1.5 itself is built on Qwen: according to its authors, with continued pretraining, mid-training and post-training, plus reinforcement learning on tasks the model generates for itself.

The idea comes from [Tiel-Coder](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) by peculiar-ragdoll: Ornith-1.5 weights paired with the [Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) (built on froggeric's work). These GGUFs follow the same idea with our own importance matrix, computed on a calibration set tilted towards code, tool calling and Russian. If you are choosing between the two, try both — Tiel-Coder was there first. Many thanks to the authors of Ornith-1.5, Tiel-Coder and the Sharp template. The safetensors / vLLM version lives in a separate repository: [ooptimum/Humo-Coder-35B-A3B-GPTQ-Int4](https://huggingface.co/ooptimum/Humo-Coder-35B-A3B-GPTQ-Int4).

## Provenance and credits

| component | source | license |
|---|---|---|
| weights | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | MIT |
| chat template | [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates); structural work by **froggeric** | Apache-2.0 |
| idea | [peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) — the same base model and chat template, quantized by its author with their own recipe and importance matrix | — |
| quantization | this repository, llama.cpp b11071 | MIT |

## Which file do I need?

**For chatting, coding and tool calling you need exactly one file** — one of the five `Humo-Coder-35B-A3B-*.gguf` model files below. Pick the largest one that fits your memory:

- **Q5_K_XL** — closest to the original; take it if it fits.
- **Q4_K_XL** — the best balance of size and quality.
- **Q5_K_M / Q4_K_M** — standard mixes, if your tool works better with them.
- **IQ3_M** — only if nothing larger fits; expect noticeably lower quality.

Leave some room for the context (KV cache) on top of the file size. Since this is a mixture-of-experts model, llama.cpp can keep the expert weights in system RAM and the rest on the GPU (`--n-cpu-moe N`), which lets the larger files run on smaller cards.

**Only for image or video input**, also download the `mmproj-*.gguf` file.

**You do not need anything in the `imatrix/` folder to run the model.** It is the importance matrix used to make these quants — useful only if you want to quantize the model yourself.

## Files

All files are made from the same bf16 checkpoint (Ornith-1.5 + the Sharp chat template) with an **importance matrix computed on our own calibration set** — the same 1270 public samples used for the GPTQ version: code and tool calling ~75%, Russian text ~40% (overlapping), domain text (PL/SQL, legal, banking) ~10%. Sources: Magicoder-OSS-Instruct-75K, the-stack-smol / the-stack-dedup, hermes-function-calling-v1, Vikhrmodels/GrandMaster-PRO-MAX, IlyaGusev/saiga_scored, fineweb-2 (rus_Cyrl), OpenR1-Math-220k.

| file | size | bits per weight | notes |
|---|---|---|---|
| `Humo-Coder-35B-A3B-IQ3_M.gguf` | 14.7 GiB | 3.56 | smallest; expect the largest quality loss |
| `Humo-Coder-35B-A3B-Q4_K_M.gguf` | 20.2 GiB | 4.89 | |
| `Humo-Coder-35B-A3B-Q4_K_XL.gguf` | 21.0 GiB | 5.07 | Q4_K routed experts, everything else kept high (see below) |
| `Humo-Coder-35B-A3B-Q5_K_M.gguf` | 23.6 GiB | 5.71 | |
| `Humo-Coder-35B-A3B-Q5_K_XL.gguf` | 24.8 GiB | 6.00 | Q5_K routed experts, everything else kept high (see below) |
| `imatrix/imatrix.gguf` | 184 MiB | — | importance matrix in GGUF format — **not a model, not needed to run**; only for making your own quants |
| `mmproj-Humo-Coder-35B-A3B-BF16.gguf` | 861 MiB | — | vision encoder; needed only for image/video input |

**What `_XL` means here.** Standard `_M` mixes spend the bits on the routed experts and quantize everything else to the same level. The `_XL` variants do the opposite:

| tensors | Q4_K_M | Q4_K_XL | Q5_K_M | Q5_K_XL |
|---|---|---|---|---|
| attention, full and linear (`attn_*`, `ssm_*`) | Q4_K; V/QKV projections raised to Q6_K in part of the layers | **BF16** | Q5_K; V/QKV projections raised to Q6_K in part of the layers | **BF16** |
| shared expert (`*_shexp`) | Q4_K; `ffn_down_shexp` raised to Q6_K in 21 of 41 layers | **Q8_0** | Q5_K; `ffn_down_shexp` raised to Q6_K in 21 of 41 layers | **Q8_0** |
| token embeddings / output | Q4_K / Q6_K | **Q8_0** / **Q8_0** | Q5_K / Q6_K | **Q8_0** / **Q8_0** |
| routed experts (`*_exps`) | Q4_K; `ffn_down_exps` raised to Q6_K in 21 of 41 layers | Q4_K everywhere | Q5_K; `ffn_down_exps` raised to Q6_K in 21 of 41 layers | Q5_K everywhere |

In about half of the layers the `_XL` variants even give the routed experts' down-projection (`ffn_down_exps`) fewer bits than `_M` does (Q4_K/Q5_K instead of Q6_K), and spend the bits on attention, the shared expert and the embeddings instead. Routed experts hold almost all of the weights, so this costs relatively little size — Q4_K_XL is only 0.8 GiB larger than Q4_K_M, Q5_K_XL 1.2 GiB larger than Q5_K_M — and in our measurements it brings Q4_K_XL close to Q5_K_M and makes Q5_K_XL the closest to bf16 of all (see below). IQ3_M uses IQ3_S for most tensors and Q4_K for attention output and value/QKV projections (and for the experts' down-projection, `ffn_down_exps`, in 5 of 41 layers). All five files share one importance matrix (510 entries, 2741 calibration chunks) and embed the same chat template (`qwen3.8-froggeric-v22.5.0`, i.e. the Sharp template).

The chat template is embedded in the GGUF metadata. Use it — the model's tool calling depends on it.

## Running

Example for llama.cpp:

```
llama-server -m Humo-Coder-35B-A3B-Q4_K_M.gguf --jinja -c 131072 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
```

`--jinja` is required for the embedded template and for tool calling. The sampling values are the ones we used in all our measurements.

## Divergence from bf16

Measured with `llama-perplexity --kl-divergence` (llama.cpp b11071, Vulkan backend) against the bf16 GGUF of the same model. Evaluation texts come from the same sources as the calibration set but **do not overlap with it**. The text is 569 chunks of 512 tokens (291 328 tokens); as usual for llama.cpp, only the second half of each chunk is scored — 145 664 tokens.

| quant | PPL ratio vs bf16 | mean KLD | median KLD | 99% KLD | 99.9% KLD | same top token | RMS Δp |
|---|---|---|---|---|---|---|---|
| Q5_K_XL | 1.0006 ± 0.0010 | 0.0403 | 0.0039 | 0.61 | 3.22 | 93.97% | 5.58% |
| Q5_K_M | 0.9895 ± 0.0012 | 0.0557 | 0.0061 | 0.82 | 4.51 | 92.92% | 6.41% |
| Q4_K_XL | 1.0092 ± 0.0012 | 0.0597 | 0.0066 | 0.91 | 4.00 | 92.54% | 6.69% |
| Q4_K_M | 1.0163 ± 0.0015 | 0.0890 | 0.0119 | 1.35 | 5.49 | 90.76% | 8.11% |
| IQ3_M | 1.0230 ± 0.0021 | 0.1846 | 0.0339 | 2.63 | 7.26 | 86.49% | 11.83% |

bf16 perplexity on this text: 4.9796 ± 0.0350.

What the table shows:

- Divergence falls steadily with bit width; Q5_K_XL is indistinguishable from bf16 by perplexity (ratio 1.0006 ± 0.0010).
- Q4_K_XL comes close to Q5_K_M on mean KLD (0.060 vs 0.056) and top-token agreement (92.5% vs 92.9%), and has a lighter tail (99.9% KLD 4.00 vs 4.51).
- The biggest step is between Q4 and IQ3_M: mean KLD doubles and top-token agreement drops to 86.5%.
- Q5_K_M has a *lower* perplexity than bf16 (−1.05% ± 0.12%). This is a known effect on a fixed text and does **not** mean it is better than the original: its KLD of 0.056 shows it still diverges from bf16.

## Evaluation status — read this first

**These GGUF files have not been benchmarked on tasks.** Above is divergence from bf16, not task quality. All task numbers in our write-up ([article on Habr](https://habr.com/ru/articles/1088512/), in Russian) were measured on the **GPTQ-Int4** version served by vLLM, not on these files. GGUF k-quants and i-quants are a different quantization scheme; IQ3_M in particular may behave noticeably differently.

What we expect to carry over, because it is a property of the model rather than of the quantization (measured on GPTQ):

- strong at agentic coding — in our Go agentic loop it solved 202 of 318 tasks, ahead of all six other models compared, using by far the fewest tokens;
- best at BFCL memory categories, weakest at BFCL multi-turn;
- **transliterates Russian in tool arguments** (`Khujand` instead of `Худжанд`) in about half of the cases; one system-prompt line fixes most of it:
  > Значения аргументов при вызове инструментов — строго кириллицей, без латиницы.
- in an agentic harness it often writes a correct solution but does not submit it.

If you benchmark these files, we would be glad to hear the numbers.

## Vision

The architecture is multimodal (image and video). The vision encoder is inherited from Ornith-1.5 unchanged. In GGUF it lives in the separate `mmproj` file. We did not test image or video input at all.

---

## По-русски

Таблицы и команды — в английской части выше, они одинаковы для обоих языков.

### Что это

GGUF-версии Humo-Coder-35B-A3B — модели [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) с заменённым шаблоном чата — для llama.cpp и совместимых программ (LM Studio, Ollama, KoboldCpp и других).

**Как и сама Ornith-1.5, модель нацелена прежде всего на программирование и агентную работу с кодом.**

Хумо — мифическая птица счастья в фольклоре народов Центральной Азии и Ирана: так продолжена орнитологическая линия названий исходной модели.

**Ничего не дообучалось.** Все заслуги в способностях модели принадлежат авторам Ornith. Сама Ornith-1.5 построена на Qwen: по словам её авторов, с продолженным предобучением, промежуточным обучением (mid-training) и пост-обучением, а также обучением с подкреплением на задачах, которые модель придумывает себе сама.

Идея взята у [Tiel-Coder](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF) от peculiar-ragdoll: веса Ornith-1.5 в паре с [шаблоном Sharp](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) (основанным на работе froggeric). Наши GGUF повторяют ту же идею, но с собственной матрицей важности (importance matrix), посчитанной на калибровочном наборе с упором на код, вызов инструментов и русский язык. Если выбираете между ними — попробуйте обе: Tiel-Coder появился первым. Большое спасибо авторам Ornith-1.5, Tiel-Coder и шаблона Sharp. Версия в safetensors для vLLM лежит в отдельном репозитории: [ooptimum/Humo-Coder-35B-A3B-GPTQ-Int4](https://huggingface.co/ooptimum/Humo-Coder-35B-A3B-GPTQ-Int4).

### Какой файл скачать

**Для чата, кода и вызова инструментов нужен ровно один файл** — одна из пяти моделей `Humo-Coder-35B-A3B-*.gguf`. Берите самую большую, которая помещается в память:

- **Q5_K_XL** — ближе всех к оригиналу; берите, если помещается.
- **Q4_K_XL** — лучшее соотношение размера и качества.
- **Q5_K_M / Q4_K_M** — стандартные варианты, если ваша программа лучше работает с ними.
- **IQ3_M** — только если ничего больше не помещается; качество заметно ниже.

Оставьте запас памяти под контекст (KV-кэш) сверх размера файла. Это модель со смесью экспертов, поэтому llama.cpp умеет держать веса экспертов в оперативной памяти, а остальное — на видеокарте (`--n-cpu-moe N`): так большие файлы запускаются и на небольших картах.

**Только для картинок и видео** скачайте ещё файл `mmproj-*.gguf`.

**Для запуска модели ничего из папки `imatrix/` не нужно.** Там матрица важности, по которой делались эти кванты, — она пригодится, только если вы хотите квантовать модель сами.

### Файлы и что значит `_XL`

Размеры файлов и реальное число бит на вес — в таблице «Files» выше. Все файлы сделаны из одной и той же bf16-модели (Ornith-1.5 + шаблон Sharp) с **матрицей важности, посчитанной на нашем собственном калибровочном наборе** — тех же 1270 публичных образцах, что и для GPTQ-версии: код и вызов инструментов — около 75%, русский текст — около 40% (с пересечением), профильные тексты (PL/SQL, право, банки) — около 10%. Источники: Magicoder-OSS-Instruct-75K, the-stack-smol / the-stack-dedup, hermes-function-calling-v1, Vikhrmodels/GrandMaster-PRO-MAX, IlyaGusev/saiga_scored, fineweb-2 (rus_Cyrl), OpenR1-Math-220k.

Стандартные варианты `_M` тратят биты на маршрутизируемых экспертов, а всё остальное сжимают до того же уровня. Варианты `_XL` поступают наоборот (по группам тензоров — в таблице выше): внимание, полное и линейное, остаётся в BF16, общий эксперт, эмбеддинги и выходной слой — в Q8_0, а проекция down у маршрутизируемых экспертов (тензор `ffn_down_exps`) примерно в половине слоёв получает даже меньше бит, чем в `_M` (Q4_K/Q5_K вместо Q6_K). Эксперты — это почти все веса модели, поэтому стоит это немного: Q4_K_XL всего на 0.8 ГиБ больше Q4_K_M, Q5_K_XL — на 1.2 ГиБ больше Q5_K_M. А в наших замерах это подтягивает Q4_K_XL почти до Q5_K_M и делает Q5_K_XL самым близким к bf16 из всех. IQ3_M использует IQ3_S для большинства тензоров и Q4_K для выхода внимания и проекций V/QKV (а проекция down у экспертов, `ffn_down_exps`, в Q4_K только в 5 слоях из 41). У всех пяти файлов одна матрица важности (510 записей, 2741 фрагмент калибровочного текста) и один встроенный шаблон чата (`qwen3.8-froggeric-v22.5.0`, то есть Sharp).

Шаблон чата встроен в метаданные GGUF. Используйте его — от него зависит вызов инструментов.

### Запуск

Пример для llama.cpp — в разделе «Running» выше. Флаг `--jinja` обязателен: без него не работают встроенный шаблон и вызов инструментов. Параметры генерации в примере — те, что мы использовали во всех замерах.

### Расхождение с bf16

Измерено через `llama-perplexity --kl-divergence` (llama.cpp b11071, бэкенд Vulkan) относительно bf16-GGUF той же модели. Тексты для замера взяты из тех же источников, что и калибровочный набор, но **с ним не пересекаются**. Текст — 569 фрагментов по 512 токенов (291 328 токенов); как обычно в llama.cpp, оценивается только вторая половина каждого фрагмента — 145 664 токена. Результаты — в таблице «Divergence from bf16» выше; перплексия bf16 на этом тексте — 4.9796 ± 0.0350.

Что видно из таблицы:

- Расхождение ровно уменьшается с ростом числа бит; Q5_K_XL по перплексии от bf16 не отличается (отношение 1.0006 ± 0.0010).
- Q4_K_XL почти догоняет Q5_K_M по среднему KLD (0.060 против 0.056) и по совпадению самого вероятного токена (92.5% против 92.9%), а хвост у него даже легче (KLD 99.9-го перцентиля 4.00 против 4.51).
- Самый большой скачок — между Q4 и IQ3_M: средний KLD удваивается, совпадение самого вероятного токена падает до 86.5%.
- У Q5_K_M перплексия *ниже*, чем у bf16 (−1.05% ± 0.12%). Это известный эффект на фиксированном тексте, и он **не** значит, что квант лучше оригинала: KLD 0.056 показывает, что от bf16 он всё равно отличается.

### Что известно о качестве — прочитайте сначала

**На задачах эти GGUF-файлы не проверялись.** Выше — расхождение с bf16, а не качество решения задач. Все результаты на задачах в нашем разборе ([статья на Хабре](https://habr.com/ru/articles/1088512/)) получены на **GPTQ-Int4**-версии в vLLM, а не на этих файлах. K-кванты и i-кванты GGUF — другая схема сжатия; особенно это касается IQ3_M — он может вести себя заметно иначе.

Что, скорее всего, сохранится, потому что это свойство модели, а не квантования (измерено на GPTQ):

- сильна в агентной работе с кодом — в нашем агентном цикле на Go решила 202 задачи из 318, больше всех остальных шести моделей, и при этом тратила заметно меньше всех токенов;
- лучшая в категориях BFCL на работу с памятью, худшая — в многоходовых;
- **пишет русские названия латиницей в аргументах инструментов** (`Khujand` вместо `Худжанд`) примерно в половине случаев; одна строка в системном промпте исправляет почти всё:
  > Значения аргументов при вызове инструментов — строго кириллицей, без латиницы.
- в агентной обвязке часто пишет верное решение, но не сдаёт его.

Если вы измерите эти файлы на задачах — будем рады увидеть результаты.

### Зрение

Архитектура мультимодальная (изображения и видео). Зрительный кодировщик унаследован от Ornith-1.5 без изменений; в GGUF он лежит в отдельном файле `mmproj`. Работу с изображениями и видео мы не проверяли вовсе.

