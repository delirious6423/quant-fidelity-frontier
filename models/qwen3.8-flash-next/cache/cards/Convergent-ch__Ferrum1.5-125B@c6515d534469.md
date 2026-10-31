---
license: other
license_name: qwen-community-license-1.0-and-cc-by-nc-4.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
language:
- de
- en
tags:
- experimental
- severely-degraded
- test-only
- low-rank
- littlebit
- custom-format
- moe
---

# Ferrum1.5-125B — 125B-Basis, stark beschädigter Test / severely damaged

> **Vorerst nur ein experimenteller Test. Die Antwortqualität ist stark beschädigt und derzeit unbrauchbar. Nicht als funktionsfähigen Assistenten verwenden.**
>
> **Experimental test only. This checkpoint is severely degraded and currently produces unusable answers. It is not a working assistant or a successful capability-preserving 0.1-bit model.**

## Was ist das?

Ferrum-1.5 ist ein extremer, verlustbehafteter Kompressionsversuch des ursprünglichen
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Die [offizielle Modellbeschreibung](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview)
nennt **125B Hauptparameter**, dazu **51B N-Gram-Embeddings und 4B MTP**.
Diese Zahlen beschreiben die Quelle vor der Kompression. Die automatische
Safetensors-Anzeige dieses Repositorys zählt die tatsächlich gespeicherten
Faktor-, Skalierungs- und Rohdaten-Einträge; sie ist kein Nachweis eines intakten
125B-Modells. Gepackte U8-Werte enthalten mehrere binäre Faktorwerte je Byte.
Die ursprünglichen Matrixformen und Expertenanzahlen wurden beibehalten;
die unabhängigen Originalgewichte wurden durch binäre Niedrigrang-Faktoren angenähert.
**Eine erhaltene logische Form bedeutet keine erhaltene Modellkapazität oder Qualität.**

Der Ansatz verwendet eine lokale, gesampelte Initialisierung im Stil von
[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit).
Er reproduziert weder dessen vollständiges Training noch dessen veröffentlichte Ergebnisse.
**Kein QAT, keine Destillation und kein anschließendes Fine-Tuning wurden durchgeführt.**
Dies ist keine offizielle Veröffentlichung von Qwen oder Samsung.

## Ergebnis des Tests

| Eigenschaft | Ergebnis |
| --- | --- |
| Gepackte Tensor-Dateien | 2.643.910.067 Bytes, etwa 2,64 GB |
| Ziel der Kompression | 0,1 Bit pro ursprünglicher Gewichtsposition |
| Tatsächliche Größe inklusive Zusatzdaten | etwa 0,118 Bit pro ursprünglicher Gewichtsposition |
| Quelltensoren im Archiv | 1.658 |
| Logische Quell-Gewichtspositionen über den gesamten Checkpoint | 179.999.981.459 einschließlich zusätzlicher Komponenten |
| Experten entfernt | keine |
| Sprachqualität | stark beschädigt / unbrauchbar |
| Format | eigene Tensorbank, kein reguläres GGUF-Modell |

Die gepackten Gewichte liegen jetzt auch als einzeln sichtbare
`model-*.safetensors`-Shards unter `qwen38-original-littlebit/` vor.
`model.safetensors.index.json` ordnet jeden tatsächlich gespeicherten Tensor
seiner Datei zu. Die ursprünglichen Quellnamen und Matrixformen stehen in
`qwen38-original-littlebit/tensor-bank.json`. Die Index-Schlüssel sind die
echten Faktor- und Skalierungsschlüssel, keine künstlich aufgefüllten
Originalmatrizen. Das ursprüngliche TAR-Archiv bleibt als ältere Alternative
erhalten und enthält dieselben komprimierten Werte in der vorherigen Aufteilung.

Die Anzahl der logischen Quell-Gewichtspositionen ist **nicht** die Anzahl
unabhängiger trainierbarer Parameter im komprimierten Ergebnis.
31 MTP-Tensoren sind im Archiv enthalten, werden von der normalen Textlaufzeit
aber nicht ausgeführt. Erhaltene Zusatzdaten sind kein Nachweis unterstützter Funktionen.

Ein echter Test über Ollama ergab:

```text
Frage:   Was ist 2 + 2? Antworte nur mit der Zahl.
Antwort: ttlttlttl basename
```

Der HTTP-Aufruf war erfolgreich. Die Antwort war falsch und unverständlich.
Dies ist nur ein einfacher Funktionstest, kein umfassender Benchmark.
Weitere Messdaten stehen in `experiment.json`.

## Download und experimentelle Ausführung

Python 3.12 und eine passende PyTorch-Installation werden empfohlen. Getestet wurde
mit PyTorch 2.14.1+cu126, Transformers 5.20.2 und Safetensors 0.8.0 auf einer
RTX 3060 Ti mit 8 GB VRAM und 16 GB RAM. Speicherbedarf und Geschwindigkeit
auf anderen Systemen sind nicht nachgewiesen. Die gepackte Dateigröße ist nicht
mit dem gesamten RAM- oder VRAM-Bedarf gleichzusetzen.

```bash
pip install huggingface_hub
hf download Convergent-ch/Ferrum1.5-125B --local-dir Ferrum1.5-125B --exclude tensor-bank.tar
cd Ferrum1.5-125B
pip install -r requirements.txt
python prepare_model.py
python ferrum_runner.py --port 8080 -c 2048
```

`prepare_model.py` prüft die heruntergeladenen Shards. Für die Gewichte werden
ungefähr 2,65 GB benötigt, zusätzlich zu Software und Tokenizer-Dateien.
Wer auch das optionale TAR-Archiv herunterlädt, benötigt etwa 2,65 GB zusätzlich.
Bei einem älteren Download ohne Shards prüft das Skript den SHA-256-Wert des
Archivs und entpackt es wie zuvor.

Die Laufzeit bietet lokal unter `http://127.0.0.1:8080` den Endpunkt
`/v1/chat/completions` mit **Streaming** (`stream: true`). Sie akzeptiert Text,
eine gleichzeitige Anfrage und standardmäßig 2.048 Kontexttokens. Laden und
Antworten können langsam sein. Bilder, Werkzeuge und MTP-Beschleunigung sind
über diese Erweiterung nicht verfügbar.

### Ollama

**Ein normales `ollama pull` oder `ollama create` reicht nicht aus.**
Das Modell wurde lokal über einen unveränderten Ollama-Server 0.40.13 mit einer
eigenen llama-server-Protokoll-Erweiterung ausgeführt. Das ist keine native
Unterstützung dieses Kompressionsformats durch Ollama.

Unter Windows enthält dieses Repository die Quelldateien und das optionale
Skript `setup_ollama_windows.ps1`, um diese experimentelle Erweiterung lokal
zu erstellen. Ein installiertes Ollama, Python mit den Abhängigkeiten und der
Windows-.NET-C#-Compiler werden vorausgesetzt. Es werden keine Ollama-Binaries
mitgeliefert. Das Skript erstellt eine separate Serverkopie und startet sie
auf Port **11435**, ohne den normalen Ollama-Server zu beenden:

```powershell
.\setup_ollama_windows.ps1 -Python (Get-Command python).Source
$env:OLLAMA_HOST = '127.0.0.1:11435'
ollama run Ferrum-1.5
```

Der Ollama-Eintrag enthält nur eine kleine Laufzeitbeschreibung und verweist auf
die externe Tensorbank. Seine angezeigte Dateigröße entspricht deshalb nicht
der tatsächlichen Modellgröße. Für eine Weitergabe müssen auch die Tensorbank
und die Laufzeit mitgegeben werden. Andere Betriebssysteme und Ollama-Versionen
sind für diese Integration nicht getestet.

## Herkunft und Lizenzen

- Basismodell und zugehörige Dateien: Qwen, Revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, unter der beigefügten
  **Qwen Community License 1.0** (`LICENSE-QWEN`).
- Referenzverfahren: SamsungLabs/LittleBit, Revision
  `42d658b0c79f76450b34b6a3547462c7cdc3e1a0`;
  Arbeiten von Banseok Lee, Dongkyu Kim, Youngcheon You und Youngmin Kim.
- Lokale Laufzeit- und Kompressionsänderungen: gepackte Faktoren, Embedding-
  und MoE-Adapter, zeilenweises Einlesen und eigenes Checkpoint-Format.
  Der mitgelieferte lokale Code wird unter **CC BY-NC 4.0** bereitgestellt
  (`LICENSE-CODE`). Änderungen gegenüber dem Referenzverfahren sind oben beschrieben.

Die Veröffentlichungsdateien ersetzen die ursprünglichen Lizenzbedingungen nicht.
Es wird keine unveränderte Qualität, Eignung für einen bestimmten Zweck oder
Übereinstimmung mit den Ergebnissen der Originalarbeiten zugesichert.
