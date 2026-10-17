---
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
license: mit
language:
- it
pipeline_tag: text-generation
tags:
- gguf
- moe
- ds4
- italian
- legal
quantized_by: buddalaw
---

# DeepSeek-V4.1-Flash · BuddaLaw K128 · 2 bit

Una versione a 2 bit di DeepSeek-V4.1-Flash, **specializzata sul lavoro giuridico italiano**:
redazione di atti, pareri, ricerca con uso di strumenti. Degli esperti instradati restano attivi
in media 128 su 384 per layer (da 96 a 185 secondo il layer); gli altri sono spenti nel router e il
motore non li legge mai.

Il file è grande quanto un 2 bit intero, ma **la parte che serve in memoria scende da ~152 GiB a
~57 GiB**: gli esperti spenti restano su disco e non vengono caricati.

> *English summary: a 2-bit, Italian-legal-specialised build of DeepSeek-V4.1-Flash. On average 128
> of 384 routed experts per layer stay active (96–185 depending on the layer); the others are
> disabled in the router and never read. Same file size as a full 2-bit build, but the weights the
> engine actually loads drop from ~152 GiB to ~57 GiB. Runs with [ds4](https://github.com/antirez/ds4).
> Not intended as a general-purpose model.*

| | |
|---|---|
| file | `DeepSeek-V4.1-Flash-Q2-buddalaw-K128.gguf` |
| dimensione su disco | 340,6 GiB (di cui 189 GiB di tabelle Engram, lette da disco) |
| esperti attivi | 5.120 su 15.360 (in media 128 su 384 per layer), 6 per token |
| esperti instradati | `gate`/`up` IQ2_XXS, `down` Q2_K; il resto come nel rilascio MXFP4 |
| motore | [ds4](https://github.com/antirez/ds4), CUDA o Metal |

## Memoria

Stime dalle dimensioni dei tensori:

| | GiB |
|---|---|
| parte fissa (attenzione, esperto condiviso, router, embedding) | 9,4 |
| tutti gli esperti attivi | 47,5 |
| contesto da 32.768 token | ~8 |
| **totale con tutti gli esperti in memoria** | **~65** |
| con una cache che tiene metà degli esperti attivi | ~41 |

Gli esperti non sono usati in modo uniforme: sui testi giuridici una cache con metà degli esperti
attivi copre la grande maggioranza delle scelte del router, e il resto si legge dal disco. Il 2 bit
intero, per confronto, chiede 152 GiB per avere tutti gli esperti in memoria.

## Qualità

Confronto con DeepSeek-V4.1-Flash nel rilascio originale (MXFP4), su conversazioni di lavoro
giuridico mai viste durante la preparazione, a contesto pieno (fino a 156 mila token), misurato su
ciò che il modello scrive (ragionamento, risposte, chiamate a strumenti):

| | divergenza KL media | mediana | stesso primo token del modello intero |
|---|---|---|---|
| questo file (K128, ~65 GiB) | **0,119** | 0,0001 | **91,7 %** |
| [K160](https://huggingface.co/buddalaw/DeepSeek-V4.1-Flash-buddalaw-K160-Q2-GGUF) (~77 GiB) | 0,098 | 0,0001 | 92,5 % |

Sulla grande maggioranza dei token il file sceglie come il modello intero; la media la fanno pochi
punti in cui le due distribuzioni si separano. Queste misure dicono quanto il file resta fedele al
modello di partenza, non giudicano la qualità giuridica delle risposte.

Fuori dominio il modello perde molto di più: su prompt generali in inglese (informatica,
programmazione) l'accordo con il modello intero scende in modo marcato. **Non è un modello
generalista.**

## Uso

```bash
./ds4 -m DeepSeek-V4.1-Flash-Q2-buddalaw-K128.gguf --ssd-streaming --ctx 32768
```

Con poca memoria si può fissare la cache degli esperti, per esempio
`--ssd-streaming-cache-experts 32GB`. `ds4-server` e `ds4-agent` accettano le stesse opzioni.
Tenere il file su un SSD veloce: le tabelle Engram e gli esperti fuori cache si leggono da lì.

**La maschera non va tolta.** Gli esperti spenti sono nel file ma azzerati: il router non li
sceglie mai, e così il modello è esatto. Togliendo la maschera verrebbero scelti e non darebbero
alcun contributo. Per un 2 bit con tutti gli esperti, c'è la versione
[K160](https://huggingface.co/buddalaw/DeepSeek-V4.1-Flash-buddalaw-K160-Q2-GGUF) con i bias originali.

## Licenza

MIT, come [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
