---
license: apache-2.0
library_name: cortiq
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - nvg
  - binary-low-rank
  - sub-1-bit
  - experimental
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Qwen3.8-27B at 0.55 bits per weight (NVG) — experimental CMF

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) compressed with NVG, a method by Oleg Kirichenko, and trained
on 31 M tokens against the model's own q8_2f as the teacher. The layers take 0.55 bits per weight. The file is
4.89 GB, 5.6× smaller than q8_2f, and decodes at 90–93 tok/s with CUDA on one RTX PRO 6000 Blackwell
(62 tok/s through Vulkan on the same card) and at 55–58 tok/s with CUDA on a 24 GB RTX A5000.

> **Experimental, not a working model.** Short factual answers can be right, and code has the right form. Longer
> answers repeat and get facts and logic wrong. It is here for you to try, to train further, or to compare with.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文说明)

![How an ordinary small quant and NVG work](https://huggingface.co/infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf/resolve/main/assets/nvg_principle_en.png)

## Quick start

```bash
cargo install cortiq-cli          # or a prebuilt binary from the releases page
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf qwen38-27b-nvg055-31m.cmf --local-dir .
cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

Use cortiq 0.8.54 or later ([releases](https://github.com/infosave2007/cmf/releases)); CUDA needs 0.8.98 or later. The speed was measured
on one RTX PRO 6000 Blackwell (CUDA and Vulkan) and on an RTX A5000 (CUDA). A Mac has no Metal kernel for this format and runs it on the CPU; we have not measured that.

Greedy answers of this file:

| prompt | answer |
|---|---|
| What is the capital of France? | "The capital of France is **Paris**.", one more sentence about the city, and stops |
| What is the capital of Russia? | "The capital of Russia is **Yerevan**.": wrong |
| Какая столица России? | "Столицей России является **Джакарта**.": wrong |
| What is 2 + 2? | "2 + 2 = 3": wrong ("Сколько будет 2 + 2?" gets "2 + 2 = 4") |
| Write a Python function that returns the n-th Fibonacci number. | Typed `def fibonacci(n: int) -> int:` with a docstring and `if n <= 0: return 0`, then comments instead of the recurrence, not finished in 96 tokens |
| Explain in simple terms why the sky is blue. | Scattering by the molecules in the air, under an invented name ("**Blue Light** (Blight)"), then one bullet repeated to the end of the 96 tokens |
| Give me three tips for better sleep. | A regular bedtime and less caffeine, both sound; the third tip repeats the second |
| The capital of France is (raw) | "Paris.", then invents (Paris as the capital of the United Kingdom) |

## Speed on NVIDIA with CUDA

Build cortiq with the CUDA backend (`cargo install cortiq-cli --features cuda`; the NVIDIA driver and the CUDA 12.x
libraries) and set `CMF_GPU=cuda`:

```bash
CMF_GPU=cuda cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

`cortiq bench`, one stream.

RTX PRO 6000 Blackwell 96 GB (Colab G4), cortiq 0.9.10:

| mode | CUDA | Vulkan, same card |
|---|---:|---:|
| decode | **90–93 tok/s** | 62 tok/s |
| 4096-token prompt | **618 tok/s**, first token 6.6 s | 64 tok/s, 64.4 s |

RTX A5000 24 GB, cortiq 0.8.98:

| mode | CUDA | Vulkan, same card |
|---|---:|---:|
| decode | **55–58 tok/s** | 44 tok/s |
| 4096-token prompt | **282 tok/s**, first token 14.5 s | 48 tok/s, 84.8 s |

The greedy text is the same as on Vulkan.

## Results

Perplexity on 512-token windows, lower is better. The chat test is q8_2f's own answers to 118 English and Russian
prompts, which is why q8_2f scores so low on it.

| set | q8_2f, 27.38 GB | GPTQ q2tp, 11.41 GB | **this file, 4.89 GB** |
|---|---:|---:|---:|
| chat test (19 windows) | 3.18 | not measured | **12.33** |
| WikiText-2 (12 windows) | 8.48 | 9.57 | **23.38** |
| WikiText-2 (64 windows) | 8.37 | 9.51 | **24.98** |
| C4 validation (64 windows) | 12.51 | 14.06 | **42.60** |
| cortiq docs (32 windows) | 7.97 | 9.57 | **58.95** |
| cortiq Rust code (32 windows) | 4.11 | 4.63 | **15.05** |

**"0.55 bits" is the compressed layers only.** The whole file is 1.43 bits per weight:

| part | bytes | bits per weight |
|---|---:|---:|
| 397 layer projections in NVG | 1.648 GB | 0.547 |
| embedding and untied head, q8_2f | 2.544 GB | 8.00 |
| speculative-decoding draft, q8_2f, not trained | 0.425 GB | 8.01 |
| 3 projections the encoder refused, q8_2f | 0.210 GB | 8.00 |
| small f16 tensors, tokenizer, directory | 0.067 GB | |

## Why NVG instead of LittleBit's start

[LittleBit](https://arxiv.org/abs/2506.13771) introduced the storage layout used here: ±1 factors with 16-bit row,
column and rank scales. Its data-free start needs heavy quantization-aware training. NVG keeps the layout and builds
it differently.

| | LittleBit | NVG |
|---|---|---|
| Signs come from | the weights alone (Dual-SVID) | the layer's own activations |
| Scales come from | rank-1 fits, then heavy training | a closed-form solve, then light training |
| No training, Qwen3.5-0.8B at 1.0 bit, perplexity (q8_2f 24.27) | 2,719,570 (our implementation, same container) | **656**, 4,143× lower |
| Training budget | about 196 M tokens and 123.6 H100-hours for a 7B model, as listed in [NanoQuant](https://arxiv.org/abs/2602.06694) | 31 M tokens for this 27B, on one GPU |

This compares the starting point and the cost, not finished models. LittleBit reports good results after its full
training, down to 0.1 bits per weight, and we have not run that training on these models. This file is not a working
model yet.

## What is new

- **Topology first.** Signs are chosen in the metric of the layer's activations, scales are then solved exactly, and
  a sign flip stays only if the output error drops. The encoder takes no gradient step.
- **GPU encoder.** One 17408×5120 tensor takes 39.1 s against 2253.6 s for the CPU reference, at a held-out error of
  0.3758 against 0.3759. Calibration streams layer by layer. The whole 27B took about 4 hours.
- **Sign training on the device.** Signs train as int16 latents with 4 bytes of state per sign, inside the reverse
  sweep. The 27B trains at 3.0 s per 4096-token step on one RTX PRO 6000 with the CUDA trainer, 4.9 s on Vulkan.
- **A real runtime.** One CMF file, CPU and wgpu kernels, and the same `cortiq run` as any other model.
- **Strict accounting.** The file's provenance stores the bit counts, so the numbers above can be checked.

## Convert your own model

```bash
cortiq convert --model ./hf-model --quant blr --bpw 0.55 --blr-method nvg --blr-device gpu \
  --calib calib.txt --blr-fit-tokens 32768 --blr-sel-tokens 4096 --output model-nvg055.cmf
```

- `--calib` is plain text. Never use your evaluation text here.
- `--blr-fit-tokens` must give a full-rank Gram of the widest input: 32,768 for the 27B. The default is 14,848.
- `--bpw` sets the budget, and `--blr-plan FILE` sets it per layer or per tensor.
- `--blr-method dual-svid` is LittleBit's initializer, for comparison.
- **Always measure the result.** The training-free file scores 148.6 on the 27B. On Qwen3-0.6B, a model without
  GatedDeltaNet layers, this encoder destroys the model: 14,163 at 1.0 bit against 28.33 for q8_2f.

## The teacher

The teacher is a full-precision CMF of the same architecture and tokenizer. For this model it is
`qwen38-27b-q8_2f.cmf` from [Qwen3.8-27B-cmf](https://huggingface.co/infosave/Qwen3.8-27B-cmf). `nvg-train` runs it
first, keeps its top-64 next-token distribution for every token, and frees it. The student then learns to match that
distribution. Your text only supplies the contexts, so the target is always the teacher, whoever wrote the text.

To get text in the model's own voice, generate answers with `teacher-data`, locally or through an API:

```bash
cortiq teacher-data generate qwen38-27b-q8_2f.cmf prompts.jsonl my.tdata --no-think
OPENROUTER_API_KEY=… cortiq teacher-data generate openrouter:qwen/qwen3.8-27b prompts.jsonl my.tdata \
  --tokenizer qwen38-27b-q8_2f.cmf --no-think
```

Answers go to `my.tdata.answers.jsonl`. This turns them into chat-format text for `--calib`:

```python
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in r["messages"])
    print(f"{turns}<|im_start|>assistant\n<think>\n\n</think>\n\n{r['response'].strip()}<|im_end|>")
```

A local q8_2f teacher generates about 33 tok/s on an RTX PRO 6000, and several at once are slower. For bulk text, use
ready answers: [the data of the 14B page](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf/tree/main/data)
has 966 of them.

## Continue the training

You need one GPU with at least 96 GB (about 87 GB in use), at least 120 GB of RAM and a fast local disk for the 59 GB
state (a RAM disk saves it in 17 s; a slow disk takes 3 minutes). On Vulkan a step of 4096 tokens takes 4.9 s and the
teacher pass runs at about 3000 tok/s. With the CUDA trainer (`CMF_GPU=cuda`, a build with `--features cuda` from
[PR #181](https://github.com/infosave2007/cmf/pull/181), not in a release yet) a step takes 3.0 s and the teacher pass
4900 tok/s: 1 M tokens in about 16 minutes.

**1. Download the student, the teacher and the chat text.**

```bash
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf --local-dir .
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q8_2f.cmf --local-dir .
```

**2. Prepare the text.** `data/chat4.txt` is the chat text of the last three runs, and the model has already seen all
of it, so prefer new text where you can. `my_plain.txt` below stands for your own plain text.

**3. Train.** A new run starts its own schedule.

```bash
export CMF_GPU_PROBE=0
cortiq nvg-train qwen38-27b-nvg055-31m.cmf --teacher qwen38-27b-q8_2f.cmf \
  --calib data/chat4.txt=0.6 --calib my_plain.txt=0.4 \
  --tokens 5242880 --window 512 --batch 8 --epochs 1 --lr 3e-4 --topk 64 \
  --eval-windows 32 --eval-every 32 --train-signs --sign-opt fact-mom --sign-lr 1.85e-4 --topk-cache topk.bin \
  --sign-state state --sign-state-every 256 --sign-state-replace --output my-nvg055.cmf
```

To stop early and export the best snapshot, set `CMF_NVG_STOP_AFTER=N`. To continue, run the same command with
`--resume state`. On 0.8.54 preparing this much chat text takes about 30 minutes on every start;
[PR #81](https://github.com/infosave2007/cmf/pull/81) brings it to about 10 s.

A new run draws its sign latents again, and the held-out KL rises before it falls. When it stays above 1.05× the
best twice in a row the trainer halves the sign rate (`guard: … sign η halved` in the log) and keeps it there. From
a trained file it settles at a quarter of the automatic rate, so the command above starts there (`--sign-lr 1.85e-4`
for this model): the held-out KL then rises 1.5 % at most before it falls. At the full rate it rose until the rate
was halved twice.

**4. Measure.** Score the dense expansion, then read the answers yourself.

```bash
cortiq requant my-nvg055.cmf --quant blr-q8_2f --output my-dense.cmf
export CMF_GPU_WGPU_GRAPH=0 CMF_NLL_CHUNK=512
cortiq ppl my-dense.cmf -f data/chat_test.txt --windows 19 --window-len 512
cortiq ppl my-dense.cmf -f wiki.test.raw --windows 12 --window-len 512
```

### Results after each training stage

Bold rows are the ends of training schedules, and only those compare. The other rows were taken while the rates were
still high: a new schedule first makes the numbers worse (see 6 M). Held-out KL uses each run's own text, so compare
it only inside a run.

| tokens | chat test | WikiText-2 (12) | held-out KL | signs changed in the run |
|---:|---:|---:|---:|---:|
| 0, no training | — | 148.58 | | |
| **1 M** | **167.03** | **31.86** | **1.663** | **0.91 %** |
| **2 M** | **92.71** | **32.41** | **1.360** | **0.96 %** |
| 3 M | 26.59 | 38.19 | 1.876 | 1.43 % |
| 4 M | 22.22 | 32.89 | 1.654 | 1.65 % |
| **5 M** | **17.34** | **30.42** | **1.561** | **1.67 %** |
| 6 M | 20.46 | 43.01 | 1.631 | 1.66 % |
| 7 M | 17.63 | 37.01 | 1.409 | 2.14 % |
| 8 M | 15.71 | 33.84 | 1.260 | 2.32 % |
| 9 M | 14.06 | 30.58 | 1.145 | 2.35 % |
| **10 M** | **13.63** | **29.90** | **1.115** | **2.35 %** |
| 15.7 M | 13.12 | 27.53 | 1.318 | 1.17 % |
| 17.8 M | 12.58 | 26.58 | 1.273 | 1.18 % |
| 19.9 M | 12.16 | 26.23 | 1.258 | 1.18 % |
| **21 M** | **12.14** | **26.22** | **1.257** | **1.18 %** |
| **31 M (this file)** | **12.33** | **23.38** | **1.055** | **0.69 %** |

The text changed between runs. The first 1 M was plain text and code. The next 4 M were 40 % chat format. The next
5 M were 60 % chat format, about 1.9 M tokens of it real Qwen3.8-27B answers, and they gave plain prose a smaller
share. From 5 M to 10 M chat fell 21 %, docs 16 % and code 13 %, while C4 and WikiText-2 moved about 2 %. The next
10.5 M were 50 % chat format (all of `chat4.txt`), 30 % C4, 8 % WikiText and 12 % code, at a quarter of the sign rate
(see above): from 10 M to 21 M docs fell 21 %, WikiText-2 12–13 %, code 13 %, C4 11 % and chat 11 %. The last
10.5 M were 8 % short questions with Qwen3.8-27B's own answers (1698 questions), 27 % chat format (the part of
`chat4.txt` read once before), 25 % WikiText-103, 31 % C4 and 9 % code, at a quarter of the sign rate from the
start: from 21 M to 31 M WikiText-2 fell 11–14 % and C4 7 %, code did not move, and the chat test and docs rose
1.6–1.7 %.

Single greedy answers do not follow these numbers, so we count them. On 247 short questions that are not in the
training text (capitals, sums, everyday facts, one-word translations, in English and Russian; an answer counts when it
contains the right string) q8_2f gets 240, the 10 M file 31, the 21 M file 42 and this file 61 (24.7 %).
This file gets 8 of 20 English additions, 20 of 50 English facts and 11 of 15 Russian words translated to English;
6 of 40 capitals asked in English and none of 30 asked in Russian; none of its answers loops (6 at 21 M, 15 at
10 M). Which questions it gets right changes from file to file: at 21 M it named Moscow and this one does not.

## Limitations

- One training chain with one seed, and the token count and the text mix changed together.
- Quality was measured on one RTX PRO 6000 only; speed on it (CUDA and Vulkan) and on an RTX A5000 (CUDA).
- The first 10 M tokens were trained against an earlier q8_2f conversion of the same size, the rest against the
  public teacher file. The two have different hashes and score the same on the chat test and WikiText-2 (3.181, 8.477).
- The method is untested outside the Qwen3.5 family, apart from the Qwen3-0.6B failure above. MoE experts are not
  covered.

## Files and verification

| file | what |
|---|---|
| `qwen38-27b-nvg055-31m.cmf` | the model, 4,893,482,088 B |
| `data/chat4.txt` | chat-format training text: 966 Qwen3.8-27B answers and OpenAssistant `oasst1` dialogues |
| `data/chat_test.txt` | the chat test |

```bash
sha256sum -c qwen38-27b-nvg055-31m.cmf.sha256
```

**License.** [cortiq](https://github.com/infosave2007/cmf) is Apache-2.0 (see
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)). The weights inherit Qwen3.8-27B's Apache-2.0
license. OpenAssistant `oasst1` is Apache-2.0.

---

## Документация на русском

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), сжатая методом NVG Олега Кириченко и обученная на 31 млн
токенов с собственной q8_2f в роли учителя. Слои занимают 0,55 бита на вес. Файл весит 4,89 ГБ, в 5,6 раза меньше
q8_2f, и декодирует 90–93 ток/с через CUDA на одной RTX PRO 6000 Blackwell
(62 ток/с через Vulkan на той же карте) и 55–58 ток/с через CUDA на RTX A5000 с 24 ГБ.

> **Экспериментальный файл, ещё не рабочая модель.** Короткие ответы на факты бывают верными, код правильно оформлен.
> Длинные ответы повторяются и ошибаются в фактах и логике. Файл выложен, чтобы его можно было попробовать, дообучить
> или сравнить.

![Как работают обычный маленький квант и NVG](https://huggingface.co/infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf/resolve/main/assets/nvg_principle_ru.png)

### Быстрый старт

```bash
cargo install cortiq-cli          # или готовый бинарник со страницы релизов
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf qwen38-27b-nvg055-31m.cmf --local-dir .
cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

Нужен cortiq 0.8.54 или новее ([релизы](https://github.com/infosave2007/cmf/releases)), для CUDA — 0.8.98 или новее.
Скорость измерена на одной RTX PRO 6000 Blackwell (CUDA и Vulkan) и на RTX A5000 (CUDA). На Mac нет ядра Metal для этого формата, файл работает на CPU; эту скорость мы не мерили.

Жадные ответы этого файла:

| запрос | ответ |
|---|---|
| What is the capital of France? | «The capital of France is **Paris**.», ещё одно предложение о городе и остановка |
| What is the capital of Russia? | «The capital of Russia is **Yerevan**.»: неверно |
| Какая столица России? | «Столицей России является **Джакарта**.»: неверно |
| What is 2 + 2? | «2 + 2 = 3»: неверно (на «Сколько будет 2 + 2?» отвечает «2 + 2 = 4») |
| Функция Python для n-го числа Фибоначчи | `def fibonacci(n: int) -> int:` с типами и docstring, `if n <= 0: return 0`, дальше комментарии вместо рекуррентной формулы, за 96 токенов не закончена |
| Explain in simple terms why the sky is blue. | Рассеяние на молекулах воздуха под выдуманным названием («**Blue Light** (Blight)»), затем один пункт повторяется до конца 96 токенов |
| Give me three tips for better sleep. | Постоянное время сна и меньше кофеина, оба совета разумны; третий повторяет второй |
| The capital of France is (продолжение) | «Paris.», дальше выдумывает (Париж как столица Великобритании) |

### Скорость на NVIDIA через CUDA

Соберите cortiq с бэкендом CUDA (`cargo install cortiq-cli --features cuda`; нужны драйвер NVIDIA и библиотеки
CUDA 12.x) и задайте `CMF_GPU=cuda`:

```bash
CMF_GPU=cuda cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

`cortiq bench`, один поток.

RTX PRO 6000 Blackwell 96 ГБ (Colab G4), cortiq 0.9.10:

| режим | CUDA | Vulkan, та же карта |
|---|---:|---:|
| генерация | **90–93 ток/с** | 62 ток/с |
| промпт 4096 токенов | **618 ток/с**, первый токен 6,6 с | 64 ток/с, 64,4 с |

RTX A5000 24 ГБ, cortiq 0.8.98:

| режим | CUDA | Vulkan, та же карта |
|---|---:|---:|
| генерация | **55–58 ток/с** | 44 ток/с |
| промпт 4096 токенов | **282 ток/с**, первый токен 14,5 с | 48 ток/с, 84,8 с |

Жадный текст тот же, что на Vulkan.

### Результаты

Перплексия на окнах по 512 токенов, меньше значит лучше. Чат-тест состоит из ответов самой q8_2f на 118 английских и
русских запросов, поэтому у неё там такое низкое значение.

| набор | q8_2f, 27,38 ГБ | GPTQ q2tp, 11,41 ГБ | **этот файл, 4,89 ГБ** |
|---|---:|---:|---:|
| чат-тест (19 окон) | 3,18 | не мерили | **12,33** |
| WikiText-2 (12 окон) | 8,48 | 9,57 | **23,38** |
| WikiText-2 (64 окна) | 8,37 | 9,51 | **24,98** |
| C4, валидация (64 окна) | 12,51 | 14,06 | **42,60** |
| документация cortiq (32 окна) | 7,97 | 9,57 | **58,95** |
| код cortiq на Rust (32 окна) | 4,11 | 4,63 | **15,05** |

**«0,55 бита» относится только к сжатым слоям.** По всему файлу выходит 1,43 бита на вес:

| часть | размер | бит на вес |
|---|---:|---:|
| 397 проекций слоёв в NVG | 1,648 ГБ | 0,547 |
| эмбеддинги и отдельная голова, q8_2f | 2,544 ГБ | 8,00 |
| черновик спекулятивного декода, q8_2f, не обучался | 0,425 ГБ | 8,01 |
| 3 проекции, от которых отказался кодировщик, q8_2f | 0,210 ГБ | 8,00 |
| мелкие тензоры f16, токенизатор, каталог | 0,067 ГБ | |

### Почему NVG, а не старт LittleBit

[LittleBit](https://arxiv.org/abs/2506.13771) предложил формат хранения, который используется здесь: множители ±1 с
16-битными масштабами строк, столбцов и ранга. Его старт без данных требует тяжёлого обучения. NVG сохраняет формат, но
строит его иначе.

| | LittleBit | NVG |
|---|---|---|
| Откуда знаки | только из весов (Dual-SVID) | из активаций самого слоя |
| Откуда масштабы | приближения ранга 1, затем тяжёлое обучение | точное решение, затем лёгкое обучение |
| Без обучения, Qwen3.5-0.8B при 1,0 бита, перплексия (q8_2f 24,27) | 2 719 570 (наша реализация, тот же контейнер) | **656**, в 4143 раза ниже |
| Бюджет обучения | около 196 млн токенов и 123,6 часа H100 на модель 7B, по таблице [NanoQuant](https://arxiv.org/abs/2602.06694) | 31 млн токенов на эту 27B, одна видеокарта |

Это сравнение стартовой точки и цены, а не готовых моделей. LittleBit сообщает о хороших результатах после полного
обучения, вплоть до 0,1 бита на вес, а мы такое обучение на этих моделях не запускали. Этот файл пока не рабочая модель.

### Что нового

- **Сначала топология.** Знаки выбираются в метрике активаций слоя, затем масштабы решаются точно, а смена знака
  остаётся, только если ошибка на выходе падает. Кодировщик не делает ни одного градиентного шага.
- **Кодировщик на GPU.** Один тензор 17408×5120 занимает 39,1 с против 2253,6 с у эталона на CPU при ошибке 0,3758
  против 0,3759. Калибровка идёт послойно. Вся 27B заняла около 4 часов.
- **Обучение знаков на устройстве.** Знаки обучаются как int16-латенты, 4 байта состояния на знак, прямо в обратном
  проходе. Шаг на 4096 токенов на 27B занимает 3,0 с на одной RTX PRO 6000 с тренером на CUDA и 4,9 с на Vulkan.
- **Настоящий рантайм.** Один файл CMF, ядра для CPU и wgpu, тот же `cortiq run`, что и для других моделей.
- **Строгий подсчёт бит.** Счётчики бит записаны в сам файл, поэтому числа выше можно проверить.

### Как сконвертировать свою модель

```bash
cortiq convert --model ./hf-model --quant blr --bpw 0.55 --blr-method nvg --blr-device gpu \
  --calib calib.txt --blr-fit-tokens 32768 --blr-sel-tokens 4096 --output model-nvg055.cmf
```

- `--calib` это обычный текст. Текст для оценки сюда не подавайте.
- `--blr-fit-tokens` должен давать полноранговую матрицу Грама самого широкого входа: 32 768 для 27B. По умолчанию 14 848.
- `--bpw` задаёт бюджет, `--blr-plan FILE` задаёт его по слоям или тензорам.
- `--blr-method dual-svid` это инициализатор LittleBit, для сравнения.
- **Всегда измеряйте результат.** Файл без обучения даёт на 27B значение 148,6. На Qwen3-0.6B, модели без слоёв
  GatedDeltaNet, кодировщик разрушает модель: 14 163 при 1,0 бита против 28,33 у q8_2f.

### Учитель

Учитель это файл CMF полной точности с той же архитектурой и токенизатором. Для этой модели это `qwen38-27b-q8_2f.cmf`
из [Qwen3.8-27B-cmf](https://huggingface.co/infosave/Qwen3.8-27B-cmf). `nvg-train` сначала запускает его, сохраняет
топ-64 следующих токенов для каждого токена и освобождает память. Затем ученик учится повторять это распределение. Ваш
текст задаёт только контексты: целью всегда остаётся учитель, кто бы ни написал текст.

Чтобы получить текст в стиле самой модели, сгенерируйте ответы через `teacher-data`, локально или через API:

```bash
cortiq teacher-data generate qwen38-27b-q8_2f.cmf prompts.jsonl my.tdata --no-think
OPENROUTER_API_KEY=… cortiq teacher-data generate openrouter:qwen/qwen3.8-27b prompts.jsonl my.tdata \
  --tokenizer qwen38-27b-q8_2f.cmf --no-think
```

Ответы попадают в `my.tdata.answers.jsonl`. Этот код превращает их в текст в формате чата для `--calib`:

```python
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in r["messages"])
    print(f"{turns}<|im_start|>assistant\n<think>\n\n</think>\n\n{r['response'].strip()}<|im_end|>")
```

Локальный учитель q8_2f генерирует около 33 ток/с на RTX PRO 6000, а несколько одновременно работают медленнее. Для
больших объёмов берите готовые ответы: в [данных страницы 14B](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf/tree/main/data)
их 966.

### Как дообучить

Нужны одна видеокарта от 96 ГБ (занято около 87 ГБ), от 120 ГБ ОЗУ и быстрый локальный диск для состояния на 59 ГБ
(RAM-диск сохраняет его за 17 с, медленный диск за 3 минуты). На Vulkan шаг на 4096 токенов занимает 4,9 с, проход
учителя идёт примерно на 3000 ток/с. С тренером на CUDA (`CMF_GPU=cuda`, сборка с `--features cuda` из
[PR #181](https://github.com/infosave2007/cmf/pull/181), в релиз ещё не вошёл) шаг занимает 3,0 с, проход учителя
идёт на 4900 ток/с: 1 млн токенов примерно за 16 минут.

**1. Скачайте ученика, учителя и текст чатов.**

```bash
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf --local-dir .
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q8_2f.cmf --local-dir .
```

**2. Подготовьте текст.** `data/chat4.txt` это текст чатов из трёх последних прогонов, модель уже видела его
целиком, поэтому по возможности берите новый текст. `my_plain.txt` ниже это ваш обычный текст.

**3. Обучите.** Новый прогон начинает собственное расписание.

```bash
export CMF_GPU_PROBE=0
cortiq nvg-train qwen38-27b-nvg055-31m.cmf --teacher qwen38-27b-q8_2f.cmf \
  --calib data/chat4.txt=0.6 --calib my_plain.txt=0.4 \
  --tokens 5242880 --window 512 --batch 8 --epochs 1 --lr 3e-4 --topk 64 \
  --eval-windows 32 --eval-every 32 --train-signs --sign-opt fact-mom --sign-lr 1.85e-4 --topk-cache topk.bin \
  --sign-state state --sign-state-every 256 --sign-state-replace --output my-nvg055.cmf
```

Чтобы остановиться раньше и выгрузить лучший снимок, задайте `CMF_NVG_STOP_AFTER=N`. Чтобы продолжить, запустите ту же
команду с `--resume state`. В 0.8.54 подготовка такого объёма текста чатов занимает около 30 минут при каждом запуске;
[PR #81](https://github.com/infosave2007/cmf/pull/81) сокращает её примерно до 10 с.

Новый прогон заново задаёт латенты знаков, и отложенный KL сначала растёт, потом падает. Если он дважды подряд выше
1,05× лучшего, тренер вдвое снижает скорость знаков (`guard: … sign η halved` в логе) и оставляет её такой. От
обученного файла она устанавливается на четверти автоматической, поэтому команда выше начинает с неё
(`--sign-lr 1.85e-4` для этой модели): отложенный KL тогда поднимается не больше чем на 1,5 % и идёт вниз. На
полной скорости он рос, пока скорость не была снижена вдвое два раза.

**4. Измерьте.** Оцените плотную копию, а затем сами прочитайте ответы.

```bash
cortiq requant my-nvg055.cmf --quant blr-q8_2f --output my-dense.cmf
export CMF_GPU_WGPU_GRAPH=0 CMF_NLL_CHUNK=512
cortiq ppl my-dense.cmf -f data/chat_test.txt --windows 19 --window-len 512
cortiq ppl my-dense.cmf -f wiki.test.raw --windows 12 --window-len 512
```

#### Результаты после каждого этапа обучения

Жирные строки это концы расписаний, сравнивать можно только их. Остальные взяты, пока скорости были высокими: новое
расписание сначала ухудшает числа (см. 6 M). Отложенный KL считается на тексте своего прогона, поэтому сравнивайте его
только внутри прогона.

| токенов | чат-тест | WikiText-2 (12) | отложенный KL | знаков изменилось за прогон |
|---:|---:|---:|---:|---:|
| 0, без обучения | — | 148,58 | | |
| **1 M** | **167,03** | **31,86** | **1,663** | **0,91 %** |
| **2 M** | **92,71** | **32,41** | **1,360** | **0,96 %** |
| 3 M | 26,59 | 38,19 | 1,876 | 1,43 % |
| 4 M | 22,22 | 32,89 | 1,654 | 1,65 % |
| **5 M** | **17,34** | **30,42** | **1,561** | **1,67 %** |
| 6 M | 20,46 | 43,01 | 1,631 | 1,66 % |
| 7 M | 17,63 | 37,01 | 1,409 | 2,14 % |
| 8 M | 15,71 | 33,84 | 1,260 | 2,32 % |
| 9 M | 14,06 | 30,58 | 1,145 | 2,35 % |
| **10 M** | **13,63** | **29,90** | **1,115** | **2,35 %** |
| 15,7 M | 13,12 | 27,53 | 1,318 | 1,17 % |
| 17,8 M | 12,58 | 26,58 | 1,273 | 1,18 % |
| 19,9 M | 12,16 | 26,23 | 1,258 | 1,18 % |
| **21 M** | **12,14** | **26,22** | **1,257** | **1,18 %** |
| **31 M (этот файл)** | **12,33** | **23,38** | **1,055** | **0,69 %** |

Текст менялся от прогона к прогону. Первый 1 M состоял из обычного текста и кода. Следующие 4 M содержали 40 % формата
чата. Следующие 5 M содержали 60 % формата чата, из них около 1,9 млн токенов это настоящие ответы Qwen3.8-27B, а доля
обычной прозы стала меньше. От 5 M до 10 M чат упал на 21 %, документация на 16 %, код на 13 %, а C4 и WikiText-2
сдвинулись примерно на 2 %. Следующие 10,5 M содержали 50 % формата чата (весь `chat4.txt`), 30 % C4, 8 % WikiText и
12 % кода при четверти скорости знаков (см. выше): от 10 M до 21 M документация упала на 21 %, WikiText-2 на
12–13 %, код на 13 %, C4 на 11 %, чат на 11 %. Последние 10,5 M содержали 8 % коротких вопросов с ответами самой
Qwen3.8-27B (1698 вопросов), 27 % формата чата (часть `chat4.txt`, прочитанная до этого один раз), 25 % WikiText-103,
31 % C4 и 9 % кода, с самого начала на четверти скорости знаков: от 21 M до 31 M WikiText-2 упал на 11–14 %, C4 на
7 %, код не сдвинулся, а чат-тест и документация выросли на 1,6–1,7 %.

Отдельные жадные ответы за этими числами не следуют, поэтому мы их считаем. На 247 коротких вопросах, которых нет
в обучающем тексте (столицы, суммы, бытовые факты, перевод одного слова, по-английски и по-русски; ответ засчитывается,
если содержит верную строку), q8_2f отвечает на 240, файл 10 M на 31, файл 21 M на 42, этот файл на 61 (24,7 %).
Этот файл верно решает 8 из 20 сложений по-английски, знает 20 из 50 фактов по-английски и переводит на английский
11 из 15 русских слов; называет 6 из 40 столиц на вопрос по-английски и ни одной из 30 по-русски; ни один его ответ
не зацикливается (6 на 21 M, 15 на 10 M). Какие именно вопросы получаются, меняется от файла к файлу: на 21 M он
называл Москву, этот не называет.

### Ограничения

- Одна цепочка обучения с одним seed, а число токенов и состав текста менялись одновременно.
- Качество измерено только на одной RTX PRO 6000, скорость — на ней (CUDA и Vulkan) и на RTX A5000 (CUDA).
- Первые 10 M токенов обучены на более ранней конвертации q8_2f того же размера, остальные на публичном файле
  учителя. У них разные хеши и одинаковые оценки на чат-тесте и WikiText-2 (3,181 и 8,477).
- Вне семейства Qwen3.5 метод не проверен, не считая провала на Qwen3-0.6B. Эксперты MoE не поддерживаются.

### Файлы и проверка

| файл | что это |
|---|---|
| `qwen38-27b-nvg055-31m.cmf` | модель, 4 893 482 088 Б |
| `data/chat4.txt` | текст для обучения в формате чата: 966 ответов Qwen3.8-27B и диалоги OpenAssistant `oasst1` |
| `data/chat_test.txt` | чат-тест |

```bash
sha256sum -c qwen38-27b-nvg055-31m.cmf.sha256
```

**Лицензия.** [cortiq](https://github.com/infosave2007/cmf) распространяется под Apache-2.0 (см.
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md)). Веса наследуют лицензию Apache-2.0 от
Qwen3.8-27B. OpenAssistant `oasst1` распространяется под Apache-2.0.

---

## 中文说明

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 使用 Oleg Kirichenko 的 NVG 方法压缩,并以模型自身的 q8_2f 为教师,
在 3100 万个词元上训练。各层占用每权重 0.55 比特。文件大小为 4.89 GB,比 q8_2f 小 5.6 倍,在一块 RTX PRO 6000 Blackwell 上通过 CUDA 解码速度为
90–93 tok/s(同一张卡上 Vulkan 为 62 tok/s),在 24 GB 的 RTX A5000 上通过 CUDA 为 55–58 tok/s。

> **实验性文件,还不是可用的模型。** 简短的事实问答可能正确,代码形式也正确。较长的回答会重复,并在事实和逻辑上出错。发布它是为了
> 让你试用、继续训练或进行比较。

![普通低比特量化与 NVG 的工作原理](https://huggingface.co/infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf/resolve/main/assets/nvg_principle_zh.png)

### 快速开始

```bash
cargo install cortiq-cli          # 或从发布页下载预编译二进制文件
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf qwen38-27b-nvg055-31m.cmf --local-dir .
cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

需要 cortiq 0.8.54 或更新版本([发布页](https://github.com/infosave2007/cmf/releases)),CUDA 需要 0.8.98 或更新版本。速度在一块
RTX PRO 6000 Blackwell 上(CUDA 和 Vulkan)以及 RTX A5000 上(CUDA)测得。Mac 没有此格式的 Metal 内核,会在 CPU 上运行,我们没有测量该速度。

此文件的贪心解码回答:

| 提示 | 回答 |
|---|---|
| What is the capital of France? | "The capital of France is **Paris**.",再加一句关于这座城市的话,然后停止 |
| What is the capital of Russia? | "The capital of Russia is **Yerevan**.":错误 |
| Какая столица России? | "Столицей России является **Джакарта**.":错误 |
| What is 2 + 2? | "2 + 2 = 3":错误(对 "Сколько будет 2 + 2?" 回答 "2 + 2 = 4") |
| 用 Python 写第 n 个斐波那契数的函数 | 带类型和文档字符串的 `def fibonacci(n: int) -> int:`,`if n <= 0: return 0`,之后用注释代替递推式,96 个词元内没有写完 |
| Explain in simple terms why the sky is blue. | 说出空气分子的散射,但用了一个编造的名称("**Blue Light** (Blight)"),之后把同一条要点重复到 96 个词元用完 |
| Give me three tips for better sleep. | 固定作息和少喝咖啡因,这两条合理;第三条重复了第二条 |
| The capital of France is(续写) | "Paris.",之后开始编造(把巴黎说成英国的首都) |

### NVIDIA CUDA 速度

使用 CUDA 后端编译 cortiq(`cargo install cortiq-cli --features cuda`;需要 NVIDIA 驱动和 CUDA 12.x 库),并设置
`CMF_GPU=cuda`:

```bash
CMF_GPU=cuda cortiq run qwen38-27b-nvg055-31m.cmf --no-think --greedy -n 96 -p "What is the capital of France?"
```

`cortiq bench`,单路。

RTX PRO 6000 Blackwell 96 GB(Colab G4),cortiq 0.9.10:

| 模式 | CUDA | 同一张卡上的 Vulkan |
|---|---:|---:|
| 解码 | **90–93 tok/s** | 62 tok/s |
| 4096 词元提示 | **618 tok/s**,首个词元 6.6 s | 64 tok/s,64.4 s |

RTX A5000 24 GB,cortiq 0.8.98:

| 模式 | CUDA | 同一张卡上的 Vulkan |
|---|---:|---:|
| 解码 | **55–58 tok/s** | 44 tok/s |
| 4096 词元提示 | **282 tok/s**,首个词元 14.5 s | 48 tok/s,84.8 s |

贪心输出与 Vulkan 相同。

### 结果

512 词元窗口上的困惑度,越低越好。聊天测试由 q8_2f 自己对 118 个英文和俄文提示的回答组成,因此它在该项上得分很低。

| 数据集 | q8_2f,27.38 GB | GPTQ q2tp,11.41 GB | **本文件,4.89 GB** |
|---|---:|---:|---:|
| 聊天测试(19 个窗口) | 3.18 | 未测量 | **12.33** |
| WikiText-2(12 个窗口) | 8.48 | 9.57 | **23.38** |
| WikiText-2(64 个窗口) | 8.37 | 9.51 | **24.98** |
| C4 验证集(64 个窗口) | 12.51 | 14.06 | **42.60** |
| cortiq 文档(32 个窗口) | 7.97 | 9.57 | **58.95** |
| cortiq Rust 代码(32 个窗口) | 4.11 | 4.63 | **15.05** |

"0.55 比特"**只指压缩后的层**。按整个文件计算为每权重 1.43 比特:

| 部分 | 大小 | 比特/权重 |
|---|---:|---:|
| 397 个 NVG 层投影 | 1.648 GB | 0.547 |
| 嵌入和独立输出头,q8_2f | 2.544 GB | 8.00 |
| 推测解码草稿,q8_2f,未训练 | 0.425 GB | 8.01 |
| 编码器拒绝的 3 个投影,q8_2f | 0.210 GB | 8.00 |
| 小型 f16 张量、分词器、目录 | 0.067 GB | |

### 为什么用 NVG 而不是 LittleBit 的起点

[LittleBit](https://arxiv.org/abs/2506.13771) 提出了这里使用的存储布局:±1 因子加 16 位的行、列和秩缩放。它的无数据起点需要大量的
量化感知训练。NVG 保留这一布局,但用不同的方式构建。

| | LittleBit | NVG |
|---|---|---|
| 符号来自 | 仅权重(Dual-SVID) | 该层自身的激活 |
| 缩放来自 | 秩 1 拟合,然后大量训练 | 闭式求解,然后轻量训练 |
| 无训练,Qwen3.5-0.8B,1.0 比特,困惑度(q8_2f 24.27) | 2,719,570(我们的实现,相同容器) | **656**,低 4,143 倍 |
| 训练预算 | 7B 模型约 1.96 亿词元和 123.6 H100 小时,见 [NanoQuant](https://arxiv.org/abs/2602.06694) 的表格 | 此 27B 为 3100 万词元,一块 GPU |

这里比较的是起点和成本,而不是最终模型。LittleBit 报告了完整训练后的良好结果,最低到每权重 0.1 比特,而我们没有在这些模型上运行过
该训练。本文件目前还不是可用的模型。

### 新特性

- **拓扑优先。** 在该层激活的度量下选择符号,再精确求解缩放,只有输出误差下降时才保留符号翻转。编码器不做任何梯度步骤。
- **GPU 编码器。** 一个 17408×5120 张量耗时 39.1 秒,CPU 参考实现为 2253.6 秒,留出误差为 0.3758 对 0.3759。校准逐层流式进行。
  整个 27B 用时约 4 小时。
- **在设备上训练符号。** 符号以 int16 潜变量训练,每个符号 4 字节状态,在反向传播过程中更新。在一块 RTX PRO 6000 上,27B 每 4096
  词元一步用 CUDA 训练器耗时 3.0 秒,用 Vulkan 耗时 4.9 秒。
- **真正的运行时。** 单个 CMF 文件,CPU 和 wgpu 内核,与其他模型相同的 `cortiq run`。
- **严格计数。** 比特计数写入文件本身,因此以上数字可以核对。

### 转换你自己的模型

```bash
cortiq convert --model ./hf-model --quant blr --bpw 0.55 --blr-method nvg --blr-device gpu \
  --calib calib.txt --blr-fit-tokens 32768 --blr-sel-tokens 4096 --output model-nvg055.cmf
```

- `--calib` 是纯文本。不要在这里使用评估文本。
- `--blr-fit-tokens` 必须让最宽输入的 Gram 矩阵满秩:27B 为 32,768,默认值为 14,848。
- `--bpw` 设置预算,`--blr-plan FILE` 按层或按张量设置。
- `--blr-method dual-svid` 是 LittleBit 的初始化器,用于比较。
- **务必测量结果。** 无训练文件在 27B 上的得分为 148.6。在 Qwen3-0.6B(没有 GatedDeltaNet 层的模型)上,该编码器会破坏模型:
  1.0 比特时为 14,163,而 q8_2f 为 28.33。

### 教师

教师是架构和分词器相同的全精度 CMF。此模型的教师是 [Qwen3.8-27B-cmf](https://huggingface.co/infosave/Qwen3.8-27B-cmf) 中的
`qwen38-27b-q8_2f.cmf`。`nvg-train` 先运行它,为每个词元保留前 64 个下一词元的分布,然后释放它。之后学生学习匹配该分布。你的文本只提供
上下文,无论文本由谁写成,目标始终是教师。

要获得模型自身风格的文本,可用 `teacher-data` 在本地或通过 API 生成回答:

```bash
cortiq teacher-data generate qwen38-27b-q8_2f.cmf prompts.jsonl my.tdata --no-think
OPENROUTER_API_KEY=… cortiq teacher-data generate openrouter:qwen/qwen3.8-27b prompts.jsonl my.tdata \
  --tokenizer qwen38-27b-q8_2f.cmf --no-think
```

回答写入 `my.tdata.answers.jsonl`。下面的代码把它们转换成 `--calib` 使用的聊天格式文本:

```python
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in r["messages"])
    print(f"{turns}<|im_start|>assistant\n<think>\n\n</think>\n\n{r['response'].strip()}<|im_end|>")
```

本地 q8_2f 教师在 RTX PRO 6000 上约生成 33 tok/s,同时运行多个会更慢。大量文本请使用现成回答:
[14B 页面的数据](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf/tree/main/data)中有 966 条。

### 继续训练

需要一块至少 96 GB 的 GPU(约占用 87 GB)、至少 120 GB 内存,以及用于 59 GB 状态的快速本地磁盘(内存盘保存一次 17 秒,慢速磁盘
要 3 分钟)。在 Vulkan 上每步 4096 词元耗时 4.9 秒,教师阶段约为 3000 tok/s。使用 CUDA 训练器(`CMF_GPU=cuda`,用
[PR #181](https://github.com/infosave2007/cmf/pull/181) 以 `--features cuda` 构建,尚未进入发布版)每步耗时 3.0 秒,教师阶段为
4900 tok/s:100 万词元约 16 分钟。

**1. 下载学生、教师和聊天文本。**

```bash
hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf --local-dir .
hf download infosave/Qwen3.8-27B-cmf qwen38-27b-q8_2f.cmf --local-dir .
```

**2. 准备文本。** `data/chat4.txt` 是最近三次运行的聊天文本,模型已经见过全部内容,因此尽量使用新文本。下面的 `my_plain.txt` 代表你自己的普通文本。

**3. 训练。** 新的运行会开始自己的学习率计划。

```bash
export CMF_GPU_PROBE=0
cortiq nvg-train qwen38-27b-nvg055-31m.cmf --teacher qwen38-27b-q8_2f.cmf \
  --calib data/chat4.txt=0.6 --calib my_plain.txt=0.4 \
  --tokens 5242880 --window 512 --batch 8 --epochs 1 --lr 3e-4 --topk 64 \
  --eval-windows 32 --eval-every 32 --train-signs --sign-opt fact-mom --sign-lr 1.85e-4 --topk-cache topk.bin \
  --sign-state state --sign-state-every 256 --sign-state-replace --output my-nvg055.cmf
```

要提前停止并导出最佳快照,请设置 `CMF_NVG_STOP_AFTER=N`。要继续,请加上 `--resume state` 运行同一命令。在 0.8.54 中,每次启动准备这么多
聊天文本约需 30 分钟;[PR #81](https://github.com/infosave2007/cmf/pull/81) 将其缩短到约 10 秒。

新的运行会重新初始化符号潜变量,留出 KL 先升后降。如果它连续两次高于最佳值的 1.05 倍,训练器会把符号学习率减半(日志中的
`guard: … sign η halved`)并保持在那里。从已训练的文件开始,它稳定在自动学习率的四分之一,所以上面的命令直接从那里开始(本模型为 `--sign-lr 1.85e-4`):留出 KL 最多先升 1.5 % 再下降。
用完整学习率时它一直上升,直到学习率被减半两次。

**4. 测量。** 对稠密副本打分,然后亲自阅读回答。

```bash
cortiq requant my-nvg055.cmf --quant blr-q8_2f --output my-dense.cmf
export CMF_GPU_WGPU_GRAPH=0 CMF_NLL_CHUNK=512
cortiq ppl my-dense.cmf -f data/chat_test.txt --windows 19 --window-len 512
cortiq ppl my-dense.cmf -f wiki.test.raw --windows 12 --window-len 512
```

#### 各训练阶段后的结果

粗体行是学习率计划的终点,只有它们可以相互比较。其他行是在学习率仍然较高时取得的:新的计划一开始会让数字变差(见 6 M)。留出 KL 使用各次
运行自己的文本,因此只能在同一次运行内比较。

| 词元数 | 聊天测试 | WikiText-2(12) | 留出 KL | 本次运行改变的符号 |
|---:|---:|---:|---:|---:|
| 0,未训练 | — | 148.58 | | |
| **1 M** | **167.03** | **31.86** | **1.663** | **0.91 %** |
| **2 M** | **92.71** | **32.41** | **1.360** | **0.96 %** |
| 3 M | 26.59 | 38.19 | 1.876 | 1.43 % |
| 4 M | 22.22 | 32.89 | 1.654 | 1.65 % |
| **5 M** | **17.34** | **30.42** | **1.561** | **1.67 %** |
| 6 M | 20.46 | 43.01 | 1.631 | 1.66 % |
| 7 M | 17.63 | 37.01 | 1.409 | 2.14 % |
| 8 M | 15.71 | 33.84 | 1.260 | 2.32 % |
| 9 M | 14.06 | 30.58 | 1.145 | 2.35 % |
| **10 M** | **13.63** | **29.90** | **1.115** | **2.35 %** |
| 15.7 M | 13.12 | 27.53 | 1.318 | 1.17 % |
| 17.8 M | 12.58 | 26.58 | 1.273 | 1.18 % |
| 19.9 M | 12.16 | 26.23 | 1.258 | 1.18 % |
| **21 M** | **12.14** | **26.22** | **1.257** | **1.18 %** |
| **31 M(本文件)** | **12.33** | **23.38** | **1.055** | **0.69 %** |

各次运行的文本不同。最初的 1 M 是普通文本和代码。接下来的 4 M 含 40 % 聊天格式。再接下来的 5 M 含 60 % 聊天格式,其中约 190 万词元是
Qwen3.8-27B 的真实回答,普通散文的比例更小。从 5 M 到 10 M,聊天下降 21 %,文档 16 %,代码 13 %,而 C4 和 WikiText-2 约变化 2 %。接下来的
10.5 M 含 50 % 聊天格式(整个 `chat4.txt`)、30 % C4、8 % WikiText 和 12 % 代码,符号学习率为四分之一(见上):从 10 M 到 21 M,文档下降
21 %,WikiText-2 下降 12–13 %,代码 13 %,C4 11 %,聊天 11 %。最后的 10.5 M 含 8 % 的简短问题及 Qwen3.8-27B 自己的回答(1698 个问题)、
27 % 聊天格式(`chat4.txt` 中此前只读过一次的部分)、25 % WikiText-103、31 % C4 和 9 % 代码,从一开始就用四分之一的符号学习率:从 21 M 到
31 M,WikiText-2 下降 11–14 %,C4 下降 7 %,代码没有变化,聊天测试和文档上升 1.6–1.7 %。

单个贪心回答并不跟随这些数字,所以我们直接计数。在 247 个不在训练文本中的简短问题上(首都、加法、日常事实、单词翻译,英语和俄语;回答中
包含正确字符串即算对),q8_2f 答对 240 个,10 M 文件 31 个,21 M 文件 42 个,本文件 61 个(24.7 %)。本文件答对 20 道英语加法中的
8 道、50 个英语事实中的 20 个、15 个俄语单词英译中的 11 个;用英语问的 40 个首都答对 6 个,用俄语问的 30 个一个也没答对;它的回答没有一个陷入
循环(21 M 时 6 个,10 M 时 15 个)。具体答对哪些问题因文件而异:21 M 时它能说出莫斯科,本文件不能。

### 局限

- 只有一条训练链和一个种子,词元数和文本组成同时变化。
- 质量只在一块 RTX PRO 6000 上测量;速度在它(CUDA 和 Vulkan)和 RTX A5000(CUDA)上测量。
- 前 10 M 词元使用的是相同大小的较早 q8_2f 转换,其余使用公开的教师文件。两者哈希不同,在聊天测试和 WikiText-2 上得分相同(3.181 和 8.477)。
- 除了上述 Qwen3-0.6B 的失败之外,该方法未在 Qwen3.5 系列之外测试。不支持 MoE 专家。

### 文件和校验

| 文件 | 说明 |
|---|---|
| `qwen38-27b-nvg055-31m.cmf` | 模型,4,893,482,088 字节 |
| `data/chat4.txt` | 聊天格式训练文本:966 条 Qwen3.8-27B 回答和 OpenAssistant `oasst1` 对话 |
| `data/chat_test.txt` | 聊天测试 |

```bash
sha256sum -c qwen38-27b-nvg055-31m.cmf.sha256
```

**许可证。** [cortiq](https://github.com/infosave2007/cmf) 使用 Apache-2.0(见
[PATENTS.md](https://github.com/infosave2007/cmf/blob/master/PATENTS.md))。权重沿用 Qwen3.8-27B 的 Apache-2.0 许可证。OpenAssistant
`oasst1` 使用 Apache-2.0。
