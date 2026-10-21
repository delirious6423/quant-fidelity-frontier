---
license: gemma
language:
- tr
- en
base_model: google/gemma-4-26B-A4B-it
tags:
- gguf
- turkish
- türkçe
- moe
- pruned
- gemma
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B Pruned + Türkçe Heal — GGUF

`google/gemma-4-26B-A4B-it` modelinin **uzman-budanmış (expert-pruned) ve
Türkçeye uyarlanmış** versiyonu. Her katmandaki 128 uzmandan **101'i
korundu** (~%21 küçülme), ardından Türkçe + kod + matematik karışık talimat
veriseti üzerinde LoRA ile iyileştirildi.

llama.cpp / Ollama / LM Studio ile **RTX 5060 (16 GB)** veya benzeri tüketici
GPU'larda kolayca çalışır.

## Sürümler

| Dosya | Boyut | Açıklama |
|---|---|---|
| `merged.Q4_K_M.gguf` | ~11 GB | Önerilen — 24 GB+ VRAM'li kartlarda hızlı |
| `merged.Q5_K_M.gguf` | ~13 GB | Daha yüksek kalite, biraz daha yavaş |

## Kullanım

### llama.cpp

```bash
./llama-cli -m merged.Q4_K_M.gguf \
  -p "Türkiye'nin başkenti neresidir?" \
  -n 200 --temp 0.7 --repeat-penalty 1.1
```

### Ollama

```bash
cat > Modelfile <<'MODELFILE'
FROM ./merged.Q4_K_M.gguf
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ end }}{{ .Response }}<end_of_turn>
"""
PARAMETER stop "<end_of_turn>"
PARAMETER repeat_penalty 1.1
SYSTEM "Sen yardımsever, dürüst ve bilgili bir Türkçe konuşan asistansın. Kullanıcının sorusu hangi dilde olursa olsun her zaman Türkçe yanıt ver."
MODELFILE

ollama create gemma-tr -f Modelfile
ollama run gemma-tr
```

## Eğitim Detayları

| Aşama | Detay |
|---|---|
| **Calibration** | 6000 örnek, %50 Türkçe + %25 kod + %25 matematik karışım |
| **Prune** | K=101 (128'den), routing weight × count metriğiyle skor |
| **Heal** | LoRA r=32 α=64, 2 epoch, 25k örnek (%60 TR + %20 kod + %20 matematik) |
| **Donanım** | A100 80GB, ~2 saat |

## Sınırlamalar

- Matematik akıl yürütme (özellikle çok adımlı problemler) zayıf
- 4-bit quant nedeniyle bf16 versiyonuna kıyasla ~%1-2 perplexity artışı
- Türkçe-dışı dillerde yazılı talimatlara hâlâ Türkçe cevap verir (system
  prompt ile değiştirilebilir)

---

# Gemma 4 26B-A4B Pruned + Turkish Heal — GGUF (English)

Expert-pruned variant of `google/gemma-4-26B-A4B-it`: 128 → 101 experts per
layer (~21% smaller), then LoRA-healed on a Turkish + code + math instruction
mix. Runs comfortably on RTX 5090 / 24 GB+ consumer GPUs via llama.cpp /
Ollama / LM Studio.

## Variants

| File | Size | Notes |
|---|---|---|
| `merged.Q4_K_M.gguf` | ~11 GB | Recommended for 24 GB+ VRAM |
| `merged.Q5_K_M.gguf` | ~13 GB | Higher quality, still fits |

## Training summary

| Stage | Detail |
|---|---|
| Calibration | 6000 samples, 50% Turkish + 25% code + 25% math |
| Prune | K=101 per layer, ranked by routing weight × count |
| Heal | LoRA r=32 α=64, 2 epochs, 25k samples (60% TR + 20% code + 20% math) |
| Hardware | A100 80GB, ~2 hours |

## Limitations

- Multi-step math reasoning is weak
- ~1-2% perplexity increase vs bf16 due to 4-bit quantization
- Defaults to Turkish output; override via system prompt for other languages
