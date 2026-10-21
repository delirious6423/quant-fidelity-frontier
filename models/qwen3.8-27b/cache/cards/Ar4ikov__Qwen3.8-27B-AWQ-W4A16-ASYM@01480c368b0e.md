---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - awq
  - w4a16
  - int4
  - llm-compressor
  - compressed-tensors
  - vllm
  - qwen3_5
  - multimodal
  - mtp
---

# Qwen3.8-27B-AWQ-W4A16-ASYM

4-битный AWQ-квант [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), сделанный
[llm-compressor](https://github.com/vllm-project/llm-compressor) в формате
`compressed-tensors` (`pack-quantized`). Грузится в vLLM нативно.

Квантование выполнено **с BF16-весов базовой модели**, а не с официального FP8-чекпоинта —
чтобы не накладывать INT4 поверх уже потерянной точности.

## Что внутри

| | |
|---|---|
| Схема | `W4A16_ASYM` (asymmetric, zero-point) |
| Group size | 128 |
| Стратегия | `group`, `type: int`, `num_bits: 4` |
| Формат | `compressed-tensors` / `pack-quantized` |
| Активации | BF16 (weight-only) |
| Калибровка | `HuggingFaceH4/ultrachat_200k`, 512 сэмплов × 2048 токенов, через chat template |

### Что квантовано, а что осталось в BF16

Qwen3.8-27B — гибрид: 64 слоя по схеме `16 × (3 × Gated DeltaNet → 1 × Gated Attention)`,
плюс vision-башня и MTP-голова.

**В INT4:**
- Gated Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- FFN: `gate_proj`, `up_proj`, `down_proj`
- Gated DeltaNet: `in_proj_qkv`, `in_proj_z`, `out_proj`

**Остались в BF16:**
- `lm_head` и эмбеддинги
- вся vision-башня (`model.visual.*`) — квантование визуального энкодера бьёт по качеству
  распознавания сильнее, чем экономит память
- `linear_attn.in_proj_a` / `in_proj_b` — чувствительные гейты SSM-ветки
- вся MTP-голова (`mtp.*`) — чтобы не ломать спекулятивный декодинг

> **Про MTP.** `transformers` не инстанцирует MTP-голову для `qwen3_5`
> (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*", ...]`), поэтому llm-compressor её
> не видит и не сохраняет — наивный квант теряет MTP целиком. Здесь тензоры `mtp.*`
> скопированы из базового BF16-чекпоинта в готовый результат, зарегистрированы в
> `model.safetensors.index.json` и внесены в `quantization_config.ignore`.
> Спекулятивный декодинг сохранён.

Тот же набор `ignore` использован в других AWQ-квантах этой модели и в моих предыдущих
квантах Qwen3.5-гибридов.

## Использование (vLLM)

```bash
vllm serve Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM \
  --max-model-len 262144 \
  --tensor-parallel-size 2
```

## Использование (transformers)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Ar4ikov/Qwen3.8-27B-AWQ-W4A16-ASYM")
```

> transformers при загрузке **распаковывает** веса обратно в BF16, так что экономии VRAM
> здесь не будет. Реальная экономия видна только в рантаймах с нативными INT4-ядрами (vLLM, SGLang).

## Параметры сэмплирования

Наследуются от базовой модели:

- **Thinking mode:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`
- **Non-thinking:** `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`

Поддерживается `reasoning_effort` (`low` / `medium` / `xhigh`), контекст 262 144 токена
нативно (до 1M через YaRN — см. карточку базовой модели).

## Воспроизводимость

В репозитории лежит `recipe.yaml`, сгенерированный llm-compressor. Ключевая часть рецепта:

```python
from llmcompressor.modifiers.awq import AWQModifier

recipe = [AWQModifier(
    targets=["Linear"],
    scheme="W4A16_ASYM",
    ignore=[
        "re:.*lm_head",
        r"re:model\.visual\..*",
        r"re:.*\.linear_attn\.in_proj_a",
        r"re:.*\.linear_attn\.in_proj_b",
        r"re:mtp\..*",
    ],
)]
```

## Оговорки

- Квант не проходил бенчмарк-валидацию — проверена только связность генерации.
  Для продакшена прогоните свои эвалы.
- Vision-часть не квантовалась, поэтому качество на изображениях должно быть близко
  к базовой модели; деградация ожидается прежде всего в текстовой части.
