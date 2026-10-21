---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: llama.cpp
language:
- de
- en
tags:
- gguf
- llama.cpp
- mtp
- speculative-decoding
- qwen3
- multimodal
---

# Qwen3.8-27B GGUF — mit separatem MTP-Draft

GGUF-Konvertierung von [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
gebaut aus den **BF16**-Originalgewichten. Der Grund für dieses Repo ist die
Datei **`qwen3.8-27b-MTP-Q8_0.gguf`**: der Multi-Token-Prediction-Kopf als
eigenständiges Draft-Modell, direkt nutzbar mit `--spec-type draft-mtp`.

Die MTP-Gewichte stecken im Upstream-Checkpoint (`mtp_num_hidden_layers` in der
`config.json`), werden von den üblichen GGUF-Repos aber nicht als getrennte
Datei ausgeliefert — und ohne getrennte Datei kann `llama-server` sie nicht als
Draft laden.

Dazu kommt, was dieses Repo von einer reinen Formatkonvertierung unterscheidet:
**alle Zahlen unten sind auf derselben Maschine selbst gemessen**, mit
offengelegtem Messgerüst unter `bench/` und protokollierten Fehlern in
[METHODIK.md](METHODIK.md).

---

## Was drei RTX 3080 leisten

Der Aufbau ist bewusst unspektakulär: **drei gebrauchte RTX 3080 mit je 20 GB,
zusammen 60 GB VRAM**, an einem EPYC 7F32 mit acht Kernen. Keine H100, keine
NVLink-Brücke, Ampere von 2020. Was dabei herauskommt:

| | Ergebnis | Bedingungen |
|---|---|---|
| **MATH-500** | **98,0 %** | 150 Aufgaben, Q8_0, ein Durchlauf, 24k Budget |
| **HumanEval+** | **88,4 – 90,2 %** | 164 Aufgaben, mit Ausführung geprüft |
| **Tempo, Denktext** | **54,7 t/s** | Q8_0 mit MTP-Draft |
| **Tempo, Fließtext** | **39,5 t/s** | Q8_0 mit MTP-Draft |
| **Kontext** | **262.144 Token** | voller trainierter Kontext, KV-Cache q8_0 |
| **Belegt** | **31,2 von 60 GB** | Modell 27,3 + Draft 2,7 + KV/Puffer |

Ein 27B-Modell in Q8_0 — also praktisch ohne Qualitätsverlust gegenüber den
Originalgewichten, siehe unten — läuft hier mit vollem trainiertem Kontext und
lässt noch die Hälfte des Speichers frei. Der Engpass auf so einer Karte ist
nicht die Rechenleistung, sondern die Speicherbandbreite: für **jedes einzelne
Token** müssen sämtliche Gewichte einmal durch den Speicher. Genau da setzt der
MTP-Draft an, und genau deshalb liegt der Rest dieser Seite auf Messungen statt
auf Versprechen.

## Dateien

| Datei | Größe | BPW | Zweck |
|---|---|---|---|
| `qwen3.8-27b-Q8_0.gguf` | 27,3 GB | 8,50 | Zielmodell — **empfohlen** |
| `qwen3.8-27b-Q6_K.gguf` | 21,0 GB | 6,57 | kleinere Alternative |
| `qwen3.8-27b-MTP-Q8_0.gguf` | 3,0 GB | — | **MTP-Draft** für spekulatives Dekodieren |
| `qwen3.8-27b-mmproj-F16.gguf` | 0,9 GB | — | Vision/Video-Projektor |

Die MTP-Tensoren sind aus den Zielmodellen **entfernt** (`--no-mtp`) und liegen
ausschließlich in der Draft-Datei. Ziel- und Draft-Datei gehören zusammen.

**Welches Zielmodell?** Q8_0 ist das genauere, Q6_K das schnellere. Ohne Draft,
gleiche Einstellungen: **Q6_K 23,7 t/s gegen Q8_0 19,6 t/s** — Q6_K liegt
21 Prozent vorn, weil 21 GB nun mal schneller gelesen sind als 27 GB. In der
Qualität ist zwischen beiden nichts nachweisbar (siehe unten), die Wahl ist also
eine reine Tempofrage.

## Verwendung

```bash
llama-server \
  -m qwen3.8-27b-Q8_0.gguf \
  --mmproj qwen3.8-27b-mmproj-F16.gguf \
  -md qwen3.8-27b-MTP-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 99 \
  --ctx-size 97000 --parallel 1 \
  --n-gpu-layers 99 --tensor-split 1,1,1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --jinja
```

Ohne Spekulation einfach `-md` und die drei `--spec-*`-Schalter weglassen.
Ohne Bildeingabe zusätzlich `--mmproj` weglassen.

**Denktiefe steuern.** Qwen3.8 kennt genau drei Stufen — `low`, `medium`,
`xhigh` — und `xhigh` ist zugleich die Voreinstellung. Ein `high` gibt es
nicht; die Chat-Vorlage verwirft unbekannte Werte still. Gesetzt wird es je
Anfrage:

```json
{"chat_template_kwargs": {"reasoning_effort": "xhigh"}}
```

Zum **Abschalten** genügt weder ein `/no_think` im Prompt noch ein
`enable_thinking` auf oberster Ebene der Anfrage — beides wird ignoriert. Es
muss in `chat_template_kwargs` stehen:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

---

## Was MTP tatsächlich bringt

Beide Läufe mit **identischer** Konfiguration (Q8_0, 97K Kontext, eine Session),
einziger Unterschied ist das Draft-Modell. Deutsche Prompts, je 2–3 Durchgänge,
ausgewertet wird der Median. Die Zahlen kommen aus `timings.predicted_per_second`
bzw. `timings.draft_n` / `draft_n_accepted`, die `llama-server` pro Anfrage
selbst meldet.

| Textsorte | ohne Draft | mit Draft | Faktor | Draft-Trefferquote |
|---|---|---|---|---|
| **Denktext (Reasoning)** | 19,6 t/s | **54,7 t/s** | **2,79×** | — |
| Wiederholung | 22,9 t/s | **62,0 t/s** | 2,71× | 85,0 % |
| Aufzählung | 23,0 t/s | **61,6 t/s** | 2,68× | 93,1 % |
| Übersetzung | 22,9 t/s | **51,8 t/s** | 2,26× | 70,9 % |
| Struktur / JSON | 23,0 t/s | **47,6 t/s** | 2,07× | 59,7 % |
| Fachtext | 23,0 t/s | **41,7 t/s** | 1,81× | 48,0 % |
| Code | 23,0 t/s | **41,1 t/s** | 1,79× | 48,9 % |
| Fließtext | 22,9 t/s | **39,5 t/s** | 1,72× | 44,5 % |
| Kreativtext | 23,0 t/s | **35,9 t/s** | 1,56× | 37,4 % |

Gesamt über die acht Textsorten: **57,3 % Trefferquote** (4.491 von 7.842
Draft-Token).

### Der wichtigste Befund

**Ohne Draft ist die Rate völlig flach — 22,9 bis 23,0 t/s, egal was man
generiert.** Mit Draft spreizt sie sich um Faktor 1,7 auseinander, weil die
Trefferquote massiv an der Textsorte hängt: vorhersagbarer Text wird zu über
90 Prozent korrekt geraten, freier Kreativtext nur zu gut einem Drittel.

Wer MTP an einem Aufzählungs-Benchmark misst, kommt auf 2,7× und überschätzt
den Nutzen für normale Chatlast deutlich. Der realistische Wert für
Fließtext-Dialog liegt bei **rund 1,7×**.

**Denktext ist der große Gewinner.** Die oberste Zeile stammt aus dem
MATH-500-Lauf mit vollem Reasoning und ist mit 2,79× der höchste gemessene
Wert überhaupt. Schrittweises Rechnen ist offenbar außerordentlich gut
vorhersagbar. Das ist praktisch relevant, weil Reasoning-Modelle den Großteil
ihrer Token im Denkteil verbrauchen — genau dort, wo der Draft am besten
trifft.

### Was der Draft kostet

* **3,0 GB** auf der Platte
* **rund 2,7 GB VRAM** (`[spec] estimated memory usage of draft model is
  2678.78 MiB` bei 97K Kontext; 1.953 MiB bei 4K)
* Ladezeit steigt kaum — 7 bis 27 Sekunden je nach Kontextgröße

Deutlich mehr als etwa der Gemma-4-12B-Draft mit 0,5 GB. Wer knapp an VRAM ist,
muss das einplanen.

---

## MATH-500: 98,0 % in einem Durchlauf

150 Aufgaben, gleichmäßig über den Datensatz gegriffen (MATH-500 ist nach Thema
sortiert, die ersten 150 wären fast nur Precalculus). Q8_0,
`reasoning_effort: xhigh`, Budget **24.000 Token** je Aufgabe, MTP-Draft aktiv.

**Ergebnis: 147 von 150 = 98,0 %.** Ein Durchlauf, ein Budget, keine Nachmessung.

| Stufe | richtig |
|---|---|
| L1 | 8/8 |
| L2 | 32/33 |
| L3 | 36/36 |
| L4 | 35/35 |
| L5 | 36/38 |

Median 53,3 Token/s, im Schnitt 1.890 Token je Aufgabe, **eine einzige** Aufgabe
lief ins Limit.

### Die drei Ausfälle, einzeln aufgemacht

| Aufgabe | geliefert | erwartet | was es wirklich ist |
|---|---|---|---|
| intermediate_algebra/558 (L5) | `3<\lambda\le 4` | `(3,4]` | dieselbe Menge, andere Notation |
| number_theory/410 (L2) | `0{,}35625` | `.35625` | dieselbe Zahl, LaTeX-Gruppierung |
| intermediate_algebra/1350 (L5) | *(nichts)* | `14` | volle 24.000 Token, keine Antwort |

Zwei der drei sind **Artefakte des Antwortvergleichs**, kein Rechenfehler. Über
alle 150 Aufgaben hinweg hat dieser Lauf **kein einziges falsches Ergebnis**
geliefert — entweder stimmte die Antwort, oder es kam gar keine.

Die beiden Notationsfälle sind inzwischen im Vergleicher abgedeckt (mit
Gegenproben, die weiterhin falsch bleiben müssen: `(3,4]` gegen
`3\le\lambda\le 4` unterscheidet sich im Randverhalten und zählt weiter als
Fehler). Die 98,0 % oben sind trotzdem die **unbereinigte** Zahl, gemessen mit
dem Vergleicher, der zur Laufzeit aktiv war. Bereinigt wären es 149/150.

### Es war das Budget, nicht das Können

Dieselben 150 Aufgaben, dasselbe Modell, dieselben Einstellungen — nur das
Token-Budget unterscheidet sich:

| Budget | richtig | am Limit |
|---|---|---|
| 8.000 Token | 138 / 150 = **92,0 %** | 10 |
| 24.000 Token | 147 / 150 = **98,0 %** | 1 |

Sechs Prozentpunkte, ohne dass das Modell etwas anders gerechnet hätte. Der
Grund steckt in der Verteilung des Token-Bedarfs:

| | Token je Aufgabe |
|---|---|
| Median | 610 |
| 75. Perzentil | 1.563 |
| 90. Perzentil | 5.384 |
| darüber | bis über 14.000 |

Eine Vorabmessung an zwölf Aufgaben ergab ein Maximum von 4.274 Token — im
8.000er-Lauf lagen **20 von 150** darüber. Aus einer kleinen Stichprobe lässt
sich das nötige Budget nicht ableiten. **Wer MATH-500 mit knappem Limit fährt,
misst, ob der Rechenweg ins Budget passt, nicht ob das Modell rechnen kann.**

Die vollständige Aufarbeitung der Fehlschläge aus dem 8.000er-Lauf — mit
Aufgabentext, Musterlösung und gelieferter Antwort — steht in
**[FEHLERANALYSE.md](FEHLERANALYSE.md)**.

### Reasoning bringt auf MATH-500 nichts

Derselbe Lauf noch einmal, mit `enable_thinking: false`, gleiches Budget:

| Lauf | richtig | Token je Aufgabe | am Limit |
|---|---|---|---|
| Reasoning `xhigh`, 24k | 147 / 150 = **98,0 %** | 1.890 | 1 |
| ohne Reasoning, 24k | 146 / 150 = **97,3 %** | 1.431 | 0 |

Gepaart ausgewertet (beide Läufe bearbeiten dieselben Aufgaben, nur die
diskordanten Paare tragen Information):

```
exakter McNemar:  4 : 3   p = 1,0000
```

Vier Aufgaben löst nur der Reasoning-Lauf, drei nur der andere. Das ist exakt
das, was Zufall produziert. **Bei ausreichendem Budget ist auf MATH-500 kein
Reasoning-Effekt nachweisbar** — der Denkmodus kostet hier 32 Prozent mehr
Token für nichts.

> **Das korrigiert eine frühere Behauptung dieser Seite.** Wir hatten „+16
> Punkte durch Reasoning" berichtet. Dieser Vergleich war wertlos: der Lauf
> ohne Reasoning bekam damals 1.600 Token, der mit Reasoning 8.000. Gemessen
> wurde also der Budget-Unterschied, nicht der Denkmodus. Bei gleichem Budget
> bleibt davon nichts übrig.

Ein qualitativer Unterschied bleibt allerdings sichtbar: der Reasoning-Lauf
lieferte über 150 Aufgaben **kein einziges falsches Ergebnis**, der Lauf ohne
Reasoning zwei (`3` statt `6`, und eine Antwort, die sich nicht extrahieren
ließ). Die Quoten sind gleich, die Fehlerarten nicht.

### Einordnung gegen publizierte Werte

Vorab die wichtigste Auskunft: **MATH-500 ist bei den Spitzenmodellen
weitgehend ausgemustert.** Der Datensatz gilt als gesättigt, und die großen
Labore berichten stattdessen AIME, HMMT oder HLE. Für Claude Opus 4.6 und
Grok 4.2 existiert **keine** MATH-500-Zahl; was dort kursiert, sind Werte für
andere Datensätze.

| Modell | MATH-500 | Quelle |
|---|---|---|
| DeepSeek-R1 | 97,3 % | offizielles R1-Paper |
| Gemini 3 Pro | 96,4 % | vals.ai, eigenes Gerüst |
| **Qwen3.8-27B Q8_0 — diese Messung** | **98,0 %** | hier, 150 von 500 Aufgaben |
| Llama 4 Maverick (400B) | 88,1 % | nur Sekundärquellen |

> **Diese Zeilen sind nicht gegeneinander aufrechenbar.** Drei Gründe, jeder
> für sich groß genug, um die Reihenfolge zu drehen:
>
> 1. **Anderer Grader.** vals.ai bewertet mit der Parserlogik aus PRM800K, wir
>    mit einem eigenen Antwortvergleich. Wie groß dieser Effekt ist, wissen wir
>    aus eigener Erfahrung ziemlich genau: allein die Reparatur unseres
>    Vergleichs hat vier bis sechs Prozentpunkte gebracht, ohne dass ein
>    einziges Modell etwas anders gerechnet hätte.
> 2. **Anderer Umfang.** 150 gleichmäßig gegriffene Aufgaben gegen die vollen
>    500.
> 3. **Unbekanntes Token-Budget.** Bei uns fallen zehn Aufgaben allein deshalb
>    durch, weil das Denken nicht ins Budget passte. Wie die anderen damit
>    umgehen, steht nirgends.
>
> Für Llama 4 kursieren zudem widersprüchliche Werte (Behemoth mal 92,4, mal
> 95,0) — ein Zeichen, dass hier von Sekundärquelle zu Sekundärquelle
> abgeschrieben wird. Die Zeile steht deshalb unter Vorbehalt.

Was man trotzdem ablesen kann: ein 27-GB-Modell auf drei gebrauchten Karten
landet **in derselben Größenordnung** wie Modelle, die auf Rechenzentren
laufen. Nicht davor — aber der Abstand ist kleiner, als die Preisschilder
vermuten lassen.

## Der entzensierte Ableger kostet keine Mathematik

Neben unserer eigenen Konvertierung gibt es
[HauhauCS' „Aggressive"-Fassung](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
desselben Modells. Die naheliegende Sorge bei jedem Uncensoring lautet: Was
kostet der Eingriff an Fähigkeit? Für Mathematik ist die Antwort gemessen —
dieselben 150 Aufgaben, dasselbe Budget, dieselben Sampler:

| Modell | MATH-500 | am Limit | Token/Aufgabe |
|---|---|---|---|
| unser Q8_0 | 147 / 150 = 98,0 % | 1 | 1.890 |
| HauhauCS Aggressive Q8_K_P | **148 / 150 = 98,7 %** | 0 | 1.993 |

```
gepaart, exakter McNemar:  2 : 3   p = 1,0000
```

**Kein nachweisbarer Unterschied.** Zwei Aufgaben löst nur unseres, drei nur
deren — bei fünf diskordanten Paaren ist das Rauschen. Bemerkenswert daran ist,
dass hier gleich **zwei** Dinge zugleich anders sind: das Uncensoring-Verfahren
*und* die Quantisierung (Q8_K_P statt Q8_0, 29,3 statt 26,6 GB). Beides
zusammen bewegt die Trefferquote nicht.

Aufschlussreicher als die Quote ist, **woran** sie jeweils scheitern. Von den
fünf Aufgaben, in denen sie sich unterscheiden, sind **drei bloße
Schreibweisen** — `(3,4]` gegen `3<\lambda\le 4`, `.35625` gegen `0{,}35625`,
`16/49` gegen `\frac{16}{49}` im Vektor. Rechnet man die heraus, stehen beide
bei 149 von 150. Übrig bleibt je ein echter Fehlschlag, und die sind von
verschiedener Art: unseres frisst sich an einer Level-5-Aufgabe fest und
liefert nach 24.000 Token nichts; deren rechnet eine Level-3-Aufgabe falsch.

> **Was hier NICHT gemessen wurde:** Code, Instruktionsbefolgung,
> Halluzinationsneigung. Mathematik ist der robusteste Teil eines Modells —
> dass sie hält, beweist nicht, dass alles hält. Bei einem ersten
> Vision-Versuch antwortete die entzensierte Fassung auf eine deutsche Frage
> englisch; ein Einzelfall, aber genau die Sorte Nebenwirkung, die man bei
> Uncensoring erwartet.

### Der MTP-Kopf steckt dort im Hauptmodell

Ein Blick in die Tensorlisten erklärt einen Größenunterschied, den man den
Dateinamen nicht ansieht:

| Datei | Tensoren |
|---|---|
| unser Hauptmodell (mit `--no-mtp` konvertiert) | 851 |
| HauhauCS Q8_K_P | 866 |

Die 15 zusätzlichen heißen `blk.64.nextn.*` — das ist der
Multi-Token-Prediction-Kopf. Er liegt dort **im Hauptmodell**, obwohl daneben
noch eine eigene MTP-Datei ausgeliefert wird. Wer beides zusammen lädt, hält 18
Tensoren doppelt im Speicher.

Nachzählen kann das jeder:

```python
from gguf import GGUFReader
r = GGUFReader("modell.gguf")
print(len(r.tensors), [t.name for t in r.tensors if "nextn" in t.name])
```

Das hat einen praktischen Vorteil und einen Nachteil. Vorteil: Dort genügt
`--spec-type draft-mtp` **ohne** zweite Datei — der Kopf wird direkt genutzt.
Nachteil: Das Hauptmodell trägt die 2,7 GB immer mit, auch wenn man gar keine
Spekulation will. Unsere Trennung per `--no-mtp` war ursprünglich eine
Aufräumentscheidung; der Tensorvergleich zeigt, dass sie auch Speicher spart.

### Zwei Beschleunigungspfade, gemessen

HauhauCS liefert zusätzlich ein **FastMTP**-Sidecar von 0,9 GB und beziffert es
mit „+21 bis +35 Prozent" gegenüber dem eingebetteten Kopf. Wir haben beide auf
derselben Maschine gegeneinander gemessen — dasselbe Modell, dieselben acht
Textsorten, drei Durchgänge je Sorte:

| Textsorte | eingebetteter Kopf | FastMTP-Sidecar | |
|---|---|---|---|
| Struktur / JSON | 39,4 t/s | **48,0 t/s** | +21,8 % |
| Code | 39,1 t/s | **46,7 t/s** | +19,4 % |
| Wiederholung | 48,6 t/s | **57,5 t/s** | +18,3 % |
| Aufzählung | 48,9 t/s | **57,2 t/s** | +17,0 % |
| Kreativtext | 35,4 t/s | **38,6 t/s** | +9,0 % |
| Fließtext | 32,3 t/s | **33,6 t/s** | +4,0 % |
| Fachtext | 35,2 t/s | 34,3 t/s | **−2,6 %** |
| Übersetzung | 45,6 t/s | 42,5 t/s | **−6,8 %** |
| **Trefferquote gesamt** | **56,8 %** | 46,4 % | |

**Die Behauptung trifft zu, aber nur für strukturierten Text.** Bei JSON, Code
und Aufzählungen liegt der Gewinn bei 17 bis 22 Prozent; bei Fließtext ist er
im Rauschen, bei Fachtext und Übersetzung sogar negativ. Das Muster ist
stimmig: Die Trefferquote **sinkt** (46,4 statt 56,8 Prozent), das Tempo steigt
trotzdem — das kleinere Draft rät schlechter, ist aber so viel billiger, dass
es sich bei vorhersagbarem Text lohnt. Auf die Qualität wirkt sich das nicht
aus, weil das Hauptmodell jeden Entwurfstoken verifiziert.

> **Das Sidecar braucht ein gepatchtes llama.cpp.** Sein `output.weight` hat
> 32.768 statt 248.320 Vokabeleinträge; ein Standard-Binary bricht ab mit
> `check_tensor_dims: tensor 'output.weight' has wrong shape`. Wir hielten die
> Datei zunächst für defekt — falsch: HauhauCS liefert einen Patch mit, der
> eine `d2t`-Tabelle einführt und das Draft-Vokabular gezielt trimmt. Der Patch
> ist **nicht upstream**, muss also nach jedem llama.cpp-Update neu angewendet
> werden. Wer diese Wartungslast nicht will, nimmt den eingebetteten Kopf: der
> läuft mit jedem aktuellen Standard-Build.


## Die Varianten unterscheiden sich nicht

Das ist der zweite große Befund dieser Messreihe, und er gilt in **beiden**
Fächern.

### HumanEval+

Alle 164 Aufgaben, sieben Varianten, identische Bedingungen. Bewertet wird durch
Ausführen: jede gelieferte Funktion läuft gegen den vollständigen
EvalPlus-Testblock, der die HumanEval-Originaltests um etwa das Achtzigfache
erweitert.

![HumanEval+ je Variante](bench/quoten-hell.png)

| Variante | bestanden | Quote | Token/s |
|---|---|---|---|
| Qwen3.6-27B UD-Q6_K_XL | 149/164 | 90,9 % | 25,1 |
| Qwen3.6-27B Uncensored Q6_K_P | 148/164 | 90,2 % | 26,9 |
| **Qwen3.8-27B Q6_K** | 148/164 | **90,2 %** | 27,9 |
| Qwen3.6-27B Q6_K | 146/164 | 89,0 % | 27,6 |
| **Qwen3.8-27B Q8_0** | 145/164 | **88,4 %** | 19,5 |
| **Qwen3.8-27B BF16** | 145/164 | **88,4 %** | 13,2 |
| Qwen3.6-27B Q5_K_M | 145/164 | 88,4 % | 31,3 |

Die gesamte Spanne beträgt **vier Aufgaben von 164**. Aussagekräftig ist hier
nicht der Vergleich der Quoten, sondern der **gepaarte** Vergleich: weil alle
Varianten dieselben Aufgaben bearbeiten, tragen nur die Aufgaben Information,
bei denen genau eine von beiden besteht. Über alle 21 Paarungen mit dem exakten
McNemar-Test: **keine einzige signifikant**, der kleinste p-Wert liegt bei
0,219.

Noch deutlicher wird es beim Blick auf die Aufgaben selbst:

* **135 der 164 Aufgaben lösen alle sieben** Varianten
* **9 Aufgaben löst keine einzige** (HumanEval/32, 76, 91, 101, 127, 132, 145,
  154, 163)
* nur **20 Aufgaben** unterscheiden überhaupt — und dort geht es mal so, mal so
  aus

Die scheinbare Rangfolge entsteht ausschließlich aus diesen 20
Wackelkandidaten. Wer daraus ein „Modell A ist besser als B" liest, liest
Rauschen.

### MATH-500, dieselben Varianten

Ohne Reasoning, Budget 1.600 Token, deshalb absolut niedriger — für den
Vergleich *untereinander* ist das unproblematisch, weil alle dasselbe Limit
bekommen.

![MATH-500 je Variante](bench/mathe-hell.png)

| Variante | Quote | abgeschnitten | Token/s |
|---|---|---|---|
| Qwen3.8-27B BF16 | 77,3 % | 31/150 | 10,8 |
| Qwen3.8-27B Q6_K | 76,7 % | 32/150 | 22,4 |
| Qwen3.6-27B Q6_K | 76,7 % | 33/150 | 22,0 |
| Qwen3.8-27B Q8_0 | 76,0 % | 34/150 | 18,6 |

Auch hier gepaart geprüft: **0 von 6 Paarungen signifikant.** Die Spanne sind
zwei Aufgaben von 150.

### Was man daraus mitnehmen kann

**Quantisierung kostet hier nichts.** BF16 (51 GB), Q8_0 (27 GB) und Q5_K_M
(19 GB) landen bei HumanEval+ auf **exakt 145 von 164**, bei MATH-500 innerhalb
von zwei Aufgaben. Von 51 auf 19 Gigabyte heruntergerechnet — kein messbarer
Qualitätsverlust, aber 13,2 gegen 31,3 Token/s. **Die unquantisierte Variante
ist für diesen Zweck reine Verschwendung.** Genau das macht 60 GB VRAM zu einer
brauchbaren Grundlage: man braucht die großen Formate nicht.

**Qwen3.8 schlägt Qwen3.6 bei Code nicht.** Die neue Generation liegt im selben
Bereich. Wer wegen Code aufrüstet, tut es umsonst; die Verbesserungen liegen
woanders — Vision, MTP-Kopf, Kontextlänge, Denktiefe.

![Tempo gegen Qualität](bench/tempo-hell.png)

### Einordnung gegen publizierte Werte

| Modell | HumanEval+ |
|---|---|
| Phi-4 Reasoning (Microsoft) | 92,9 % |
| **Qwen3.8-27B Q6_K — diese Messung** | **90,2 %** |
| Llama 3 405B Instruct | 89 % |
| **Qwen3.8-27B Q8_0 — diese Messung** | **88,4 %** |
| Granite 3.3 8B Base | 86,1 % |
| Kimi K2 Base | 80,3 % |

> **Diese Zahlen sind nicht eins zu eins mit den Leaderboards vergleichbar.**
> Drei Abweichungen vom offiziellen EvalPlus-Verfahren: die Aufgabe wird im
> **Chat-Format** gestellt („Vervollständige diese Funktion") statt als reine
> Textvervollständigung; **Temperatur 0,2** statt greedy; und der Code wird aus
> Markdown-Zäunen geschält. Jeder Punkt kann einige Prozent bewegen, meist nach
> oben. Wer vergleichen will, nimmt die Zahl als Hausmessung, nicht als
> Leaderboard-Eintrag.

Das Fehlerbild ist gesund: bei Q8_0 sind von 19 Fehlschlägen **15
AssertionError**, also echte Logikfehler. Dazu je ein TypeError, SyntaxError,
IndexError und eine Zeitüberschreitung — kein Muster, das auf ein kaputtes
Testgerüst hindeutet.

Das komplette Messgerüst liegt unter `bench/` samt Selbsttests: einer prüft das
Aufgaben-Harness gegen bekannt richtige *und* bekannt falsche Lösungen, einer
die EvalPlus-Ausführung gegen die mitgelieferten Referenzlösungen (25/25 grün),
einer den Antwortvergleich der Mathematik gegen 39 bekannte Schreibweisenfälle
und alle Musterlösungen gegen sich selbst. Ohne diese Prüfung misst ein
Benchmark unter Umständen nur die eigenen Denkfehler — bei uns hat allein ein
zu strenger Antwortvergleich vier bis sechs Prozentpunkte gekostet, siehe
[METHODIK.md](METHODIK.md).

---

## Durchsatz im Vergleich

![Durchsatz je Variante](bench/durchsatz-hell.png)

Sieben 27B-Varianten unter identischen Bedingungen — 16K Kontext, eine Session,
kein Draft, `--tensor-split 1,1,1`, KV-Cache q8_0, Temperatur 0,2:

| Variante | Größe | Token/s |
|---|---|---|
| Qwen3.6-27B Q5_K_M | 19 GB | 26,4 |
| Qwen3.6-27B Q6_K | 21 GB | 23,9 |
| **Qwen3.8-27B Q6_K** | 21 GB | **23,7** |
| Qwen3.6-27B Uncensored Q6_K_P | 22 GB | 23,1 |
| Qwen3.6-27B UD-Q6_K_XL | 24 GB | 21,1 |
| **Qwen3.8-27B Q8_0** | 27 GB | **19,6** |
| **Qwen3.8-27B BF16** | 51 GB | **11,5** |

Der Zusammenhang ist schlicht: mehr Gigabyte, weniger Token pro Sekunde.

**Die BF16 läuft auf drei RTX 3080 — knapp.** 51,3 GB Gewichte belegen
16,7 / 16,8 / 17,7 GB der drei Karten, macht 51,2 von 60 GB. Mit 16K Kontext
bleibt genug für KV-Cache und Puffer. Sie kostet 40 Prozent Tempo gegenüber
Q8_0, ohne dass sich irgendwo ein Qualitätsvorsprung gezeigt hätte.

## Kontextlänge und KV-Cache — ab 48 GB hört die Abwägung auf

Auf kleineren Karten zwingt der KV-Cache zu einer Entscheidung: weniger Bits im
Cache, dafür mehr Kontext. Auf **60 GB** stellt sich diese Frage nicht mehr,
weil vorher das Modell selbst deckelt. `n_ctx_train` liegt bei Qwen3.8-27B bei
**262.144 Token**; darüber meldet llama.cpp:

```
llama_context: n_ctx_seq (384000) > n_ctx_train (262144)
  -- possible training context overflow
```

**Mit q8_0 erreicht man auf 60 GB bereits den vollen trainierten Kontext.** Ein
Wechsel auf q5_1 oder q4_0 bringt hier keine Kontextlänge mehr, sondern nur
freies VRAM — nützlich für mehr parallele Sitzungen, nicht für längere
Eingaben. Auf 24 oder 32 GB sieht das anders aus, dort ist die Abwägung real.

Der Kontext ist dabei praktisch gratis: dieselbe Konfiguration mit 4K statt 97K
lieferte im Rahmen der Messstreuung dieselben Werte (58,8 gegen 60,5 t/s bei
Aufzählungen).

**Ein Fallstrick beim KV-Quant:** `q5_1` hat auf diesem Build keinen
Flash-Attention-Kern und fällt still auf CPU-Attention zurück. Die
Generierungsrate sieht dabei fast normal aus — der Einbruch zeigt sich nur im
Prefill. Nach jeder Änderung am KV-Format also mit einem **langen** Prompt
nachmessen, nicht mit einem kurzen.

## Flash-Attention mit quantisiertem KV-Cache: bitte selbst nachmessen

[llama.cpp-Issue #24485](https://github.com/ggml-org/llama.cpp/issues/24485)
beschreibt einen **stillen** Rückfall auf CPU-Attention, wenn ein quantisierter
KV-Cache mit Flash-Attention kombiniert wird und der Build ohne
`GGML_CUDA_FA_ALL_QUANTS=ON` erzeugt wurde. Keine Warnung, aber 25- bis
45-fach langsamerer Prefill. Das Issue wurde als *not planned* geschlossen —
das Verhalten bleibt also bestehen.

Unser Build ist **nicht** betroffen. Der Test dauert fünf Minuten: derselbe
Prompt (8.933 Token), einmal mit q8_0-Cache, einmal mit f16.

| KV-Cache | Prefill | Generierung |
|---|---|---|
| q8_0 | **2.111,7 Tok/s** | 22,8 Tok/s |
| f16 | **2.142,7 Tok/s** | 23,0 Tok/s |

1,4 Prozent Abstand — kein Rückfall. **Wichtig: an der Generierungsrate sieht
man den Fehler nicht.** Die ist durch die Speicherbandbreite begrenzt und
ändert sich kaum; nur die Prefill-Rate deckt das Problem auf. Das dürfte der
Grund sein, warum der Fehler in vielen Aufbauten unbemerkt bleibt.

Nebenbefund: `--flash-attn off` mit quantisiertem KV-Cache stürzt auf diesem
Build ab (Speicherzugriffsfehler), statt sauber abzulehnen.

## Zwei Hinweise für Ampere-Besitzer

**FP8 läuft auf Ampere nicht nativ.** SM 8.6 hat keine FP8-Tensorcores, die gibt
es erst ab SM 8.9. Qwens FP8-Release lässt sich dort nur dequantisiert fahren —
Speicher gespart, Tempo nicht. Für GGUF ist das gegenstandslos, weil `Q8_0` und
`Q6_K` reine Integer-Formate mit nativen Ampere-Kernen sind.

**vLLM und drei GPUs: Tensor-Parallelität geht nicht, Pipeline-Parallelität
schon.** Das Modell hat 24 Attention-Heads, aber nur **4 KV-Heads**, und
`tensor_parallel_size` muss die Head-Zahl teilen — 4 durch 3 geht nicht auf.
`pipeline_parallel_size` kennt diese Beschränkung nicht: Pipeline-Parallelität
teilt Layer statt Attention-Köpfe und erlaubt sogar ungleiche Aufteilungen,
dasselbe Prinzip wie `--tensor-split` in llama.cpp. Mit
`--pipeline-parallel-size 3` nutzt vLLM alle drei Karten.

## Testaufbau

* 3× NVIDIA RTX 3080 20 GB — **Ampere, SM 8.6**, zusammen 60 GB
* AMD EPYC 7F32, 8 Kerne / 16 Threads
* llama.cpp Build **10423** (`a94d563ed`), Container `ghcr.io/ggml-org/llama.cpp:full-cuda`
* `--tensor-split 1,1,1`, KV-Cache `q8_0`, `--flash-attn on`
* Eine Session (`--parallel 1`), 97.024 Token Kontext
* Belegt: 31,2 GB von 60 GB — Modell 27,3 + Draft 2,7 + KV/Puffer

## Wie diese Dateien gebaut wurden

```bash
# Zielmodell ohne MTP-Tensoren
convert_hf_to_gguf.py Qwen3.8-27B/ --outfile tmp-BF16.gguf --outtype bf16 --no-mtp

# nur der MTP-Kopf, als Draft
convert_hf_to_gguf.py Qwen3.8-27B/ --outfile qwen3.8-27b-MTP-Q8_0.gguf --outtype q8_0 --mtp

# Vision/Video-Projektor
convert_hf_to_gguf.py Qwen3.8-27B/ --outfile qwen3.8-27b-mmproj-F16.gguf --outtype f16 --mmproj

# quantisieren
llama-quantize tmp-BF16.gguf qwen3.8-27b-Q8_0.gguf Q8_0 16
llama-quantize tmp-BF16.gguf qwen3.8-27b-Q6_K.gguf Q6_K 16
```

Die Flags `--mtp` und `--no-mtp` sind neu; ältere llama.cpp-Builds kennen sie
nicht. Die Architektur `Qwen3_5ForConditionalGeneration` (Qwen3.8 nutzt sie
unverändert weiter) erbt `supports_mtp_export = True` über
`Qwen3_5TextModel → _LinearAttentionVReorderBase → Qwen3NextModel → _QwenMtpMixin`.

Quelle: `Qwen/Qwen3.8-27B` in **BF16** (51,3 GB, 16,00 BPW), nicht das
FP8-Release — also **keine** doppelte Quantisierung.

## Reproduzieren

```bash
# Tempo je Textsorte
python3 bench/bench_mtp.py --port 8290 --runs 3 --max-tokens 300 --out ergebnis.md

# MATH-500 mit vollem Reasoning
python3 bench/bench_mathe.py --port 8291 --modell "Q8_0" --satz math500 \
  --grenze 150 --reasoning --effort xhigh --max-tokens 8000 --out ergebnis.json

# HumanEval+ mit Ausführung
python3 bench/bench_humaneval.py --port 8291 --modell "Q8_0" --out ergebnis.json

# vorher immer: Selbsttests des Messgerüsts
python3 bench/selftest.py && python3 bench/selftest_he.py && python3 bench/selftest_mathe.py
```

Einzelmessungen schwanken um etwa ±6 Prozent, deshalb Median statt Einzelwert.

## Lizenz

Apache 2.0, wie das Basismodell. Alle Rechte am Modell liegen beim Qwen-Team /
Alibaba Cloud. Dieses Repo enthält ausschließlich formatkonvertierte Gewichte,
kein zusätzliches Training und keine Änderung am Verhalten.

## Qwopus3.8-27B-Flash auf einer 20-GB-Karte: passt nicht mit MTP und Vision (27.09.2026)

**Zurückgezogen:** An dieser Stelle standen kurzzeitig Durchsatzzahlen eines
Vergleichs mit [Qwopus3.8-27B-Flash](https://huggingface.co/Jackrong/Qwopus3.8-27B-Flash-GGUF).
Diese Messung war **ungültig** — auf beiden Karten lief dasselbe Modell, weil
die Preset-Automatik meines Servers das angeforderte Modell beim Start still
wieder auf das alte zurückgesetzt hat. Die Zahlen sind entfernt.

Was dabei herauskam, ist trotzdem für andere nützlich, die eine **RTX 3080 mit
20 GB** haben:

| Aufstellung (Qwopus IQ4_XS, 14,4 GB) | passt? |
|---|---|
| `--ctx-size 94000` + Vision, **ohne** MTP | ja |
| `--ctx-size 94000` + Vision + `--spec-type draft-mtp` | **nein** |
| `--ctx-size 86000` + Vision + `--spec-type draft-mtp` | **nein** |

Die Ursache steht im Log von `llama-server`:

```
[spec] estimated memory usage of MTP context is 528.02 MiB
common_fit_params: failed to fit params to free device memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

Der MTP-Zwischenspeicher kostet also rund **0,5 GB zusätzlich**. Zusammen mit
dem Vision-Projektor (0,87 GB in bf16; das mitgelieferte `mmproj-F32.gguf`
wiegt sogar 1,8 GB) reicht es auf 20 GB nicht mehr für 94k Kontext.

**Wichtig für die Praxis:** Den MTP-Kopf kleiner zu quantisieren hilft hier
**nicht** — die 528 MiB sind der *Kontext* des Draft-Modells, nicht sein
Gewicht. Wirksam sind nur: Vision weglassen, Kontext deutlich kürzen, oder
MTP weglassen.

Zum Vergleich: Das Modell aus diesem Repo läuft mit **separatem** MTP-Draft,
Vision und 94k Kontext auf derselben Karte — der getrennte Draft lässt sich
unabhängig quantisieren, der eingebettete nicht.

Eine saubere Gegenüberstellung der beiden Modelle bei identischen Einstellungen
folgt und wird hier ergänzt.

### Nachgeholt: sauberer Durchsatzvergleich auf derselben Karte

Weil Qwopus mit MTP nicht auf die 94k-Karte passt, lief der Vergleich auf der
**32K-Karte** (eine RTX 3080, `--ctx-size 96000 --parallel 3`, also drei
Plaetze zu je 32k, **ohne** MTP und ohne Vision). Beide Modelle liefen
nacheinander auf **derselben Karte mit identischen Argumenten**, je 18
Anfragen (sechs Arten × drei Runden) plus ein Test mit drei gleichzeitigen
Anfragen.

| Kennzahl | Qwopus3.8-27B-Flash | Huihui-Qwen3.8-27B-abliterated | Unterschied |
|---|---|---|---|
| Prefill | 1.542 Tok/s | 1.515 Tok/s | +1,8 % |
| Erzeugung | **31,4 Tok/s** | 29,9 Tok/s | +5,0 % |
| Reasoning-Zeichen gesamt | **9.441** | 10.306 | −8,4 % |
| Ausgangstoken gesamt | 4.484 | 4.474 | gleich |
| Werkzeugaufrufe korrekt | 3 / 3 | 3 / 3 | gleich |
| Zurückgewiesene Anfragen | 0 | 0 | keine |
| Drei gleichzeitig, kurz | 44,7 Tok/s | 43,7 Tok/s | +2,3 % |
| Drei gleichzeitig, mittel (8k) | 27,1 Tok/s | **29,9 Tok/s** | −9,4 % |
| Gesamtdauer der Reihe | 239 s | 245 s | −2,1 % |

**Einordnung.** Qwopus erzeugt etwa **5 % schneller** und schreibt **rund 8 %
weniger Reasoning-Text** — die Richtung der Herstellerangaben stimmt, der
Betrag faellt kleiner aus als die dort genannten 12,8 %. Beim Einlesen sind
beide gleichauf. Bei drei gleichzeitigen mittleren Anfragen lag Huihui vorn;
mit nur drei Stichproben je Fall ist dieser Einzelwert aber nicht belastbar.
Zurueckgewiesen wurde keine einzige Anfrage, Werkzeugaufrufe funktionierten
bei beiden fehlerfrei.

**Grenzen.** Die beiden Bloecke liefen nacheinander auf einem Rechner im
Normalbetrieb, nicht gleichzeitig — Fremdlast geht also unterschiedlich ein.
Drei Runden je Aufgabenart sind wenig. Die Zahlen taugen fuer eine
Groessenordnung, nicht fuer zwei Nachkommastellen.
