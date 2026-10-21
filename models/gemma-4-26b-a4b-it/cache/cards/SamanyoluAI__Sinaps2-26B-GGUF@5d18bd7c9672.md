---
base_model:
  - google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
  - google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
language:
  - tr
  - en
tags:
  - gguf
  - gemma4
  - moe
  - sinaps
  - samanyoluai
  - turkish
  - code
  - agent
  - vision
  - conversational
---

# Sinaps2 26B

**Sinaps2**, SamanyoluAI'nin Türkçe ve ajanlı (agentic) yazılım geliştirme odaklı
modelidir. 26B toplam / 4B aktif parametreli bir MoE mimarisi kullanır ve LM Studio,
llama.cpp gibi GGUF çalıştırıcıları için q4_0 formatında paketlenmiştir.

Sinaps2, Google'ın [`gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
modeli üzerine kuruludur.

## Özellikler

| | |
|---|---|
| Parametre | 26B toplam / 4B aktif (MoE) |
| Uzmanlar | 128 uzman, token başına 8 aktif |
| Niceleme | q4_0 (QAT) |
| Dosya boyutu | 14.4 GB (+ 1.2 GB mmproj) |
| Bağlam uzunluğu | 262.144 token |
| Mimari | `gemma4` |
| Yetenekler | Metin, görsel girdi, araç çağırma, düşünme (thinking) |
| Diller | Türkçe, İngilizce (temel model çok dillidir) |

## Kullanım

**LM Studio** — `Sinaps2-26B-GGUF` araması yapın veya dosyaları
`~/.lmstudio/models/SamanyoluAI/Sinaps2-26B-GGUF/` altına yerleştirin.
Görsel desteği için `mmproj` dosyasını aynı klasörde tutun.

```bash
llama-cli -m Sinaps2-26B-Q4_0.gguf --mmproj Sinaps2-26B-mmproj.gguf -p "Merhaba"
```

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="SamanyoluAI/Sinaps2-26B-GGUF",
    filename="Sinaps2-26B-Q4_0.gguf",
)
```

### Donanım

q4_0 ağırlıkların tamamı için ~16 GB boş RAM/VRAM önerilir.

## Varsayılan kimlik

Sohbet şablonu, çağıran taraf bir sistem istemi göndermediğinde devreye giren Türkçe
bir varsayılan kimlik istemi içerir. Kendi sistem isteminizi gönderirseniz sizinki
geçerli olur.

Ajanlı kodlama araçları (Cline, Aider, OpenCode vb.) her zaman kendi sistem istemlerini
gönderdiğinden, varsayılan kimlik bu araçlarda devreye girmez — model kullandığınız
aracın istemiyle çatışmaz.

## Temel modelden farklar

Ağırlıklar değiştirilmemiştir; tensor verisi temel modelle bit düzeyinde aynıdır.

1. **GGUF `general.*` meta verileri** yeniden adlandırıldı: `name`, `basename`,
   `size_label`, `version`, `organization`, `author`, `description`, `license`,
   `languages`. Köken bilgisi `general.base_model.0.*` anahtarlarında korunmuştur.
2. **Sohbet şablonu**, yalnızca sistem istemi verilmediğinde devreye giren varsayılan
   bir kimlik istemi ile öne eklendi. Şablonun geri kalanı, araç çağırma ve görsel
   akışları dahil, değiştirilmedi.

Mimari, tokenizer, kelime dağarcığı ve hiperparametreler aynıdır.

## Lisans

Temel modelden devralınan [Apache License 2.0](LICENSE) altında yayınlanmıştır.
Atıf bilgisi için [NOTICE](NOTICE) dosyasına bakınız. Apache 2.0 madde 6 uyarınca
marka hakkı verilmez; bu sürüm Google LLC ile ilişkili değildir.

## Teşekkür

Temel model [Google DeepMind](https://huggingface.co/google) tarafından geliştirilmiştir.
Paketleme ve dağıtım: SamanyoluAI.

---

<details>
<summary><b>English</b></summary>

**Sinaps2 26B** is SamanyoluAI's Turkish-first, agentic-coding-focused release of
Google's Gemma 4 26B-A4B (26B total / 4B active MoE), packaged as q4_0 GGUF for
LM Studio and llama.cpp. Needs ~16 GB RAM/VRAM.

Changes from the base model: GGUF `general.*` branding keys (provenance preserved
under `general.base_model.0.*`), and a chat-template prefix that injects a default
Turkish identity when the caller sends no system prompt. Weights are unmodified —
tensor data is byte-identical to the base model. Agentic harnesses send their own
system prompt, so the default identity does not apply there.

Released under [Apache 2.0](LICENSE), inherited from the base model. See [NOTICE](NOTICE)
for attribution. Not affiliated with or endorsed by Google LLC.

</details>
