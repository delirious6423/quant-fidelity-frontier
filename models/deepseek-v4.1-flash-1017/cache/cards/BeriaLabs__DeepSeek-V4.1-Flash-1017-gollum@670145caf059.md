---
license: mit
language:
  - en
  - es
  - zh
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - abliterated
  - refusal-direction
  - moe
  - cybersecurity
  - gguf
---

# DeepSeek V4.1 Flash 1017 — gollum (BeriaLabs)

Artefacto de investigación: **ablación direccional persistente en pesos** sobre
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(763 G parámetros, MoE de 40 capas, 384 expertos enrutados + 1 compartido),
con la receta gollum de BeriaLabs.

## Qué se hizo

Se capturaron las activaciones del propio modelo (residual del último token del
prompt, media de las copias de Hyper-Connections) en 10 capas y se construyeron
dos direcciones de rechazo:

| Dirección | Corpus | Capa | Cohen's d |
|---|---|---|---|
| `arditi` | 520 pares AdvBench/Alpaca | 32 | **11,27** |
| `pareada` | 200 pares XSTest | 20 | 3,33 |

Las dos se ortonormalizaron (QR) y se aplicó, escritor a escritor del residual:

    W' = (I − s_w · QQᵀ) W

- **Escritores editados**: `attn.wo_b`, `ffn.shared_experts.w2` y
  `ffn.experts.*.w2` de las capas 11–39.
- **Dosimetría por escritor**: se midió la contribución `‖QᵀW‖/‖W‖` de 11.194
  escritores y se editó el **50 % superior de cada familia** (5.598 tensores).
- **Perfil rampa**: `s` crece con la profundidad, de 0,78 a 1,30.
- **Preservación de norma de fila** en los tensores fp8 antes de recuantizar.
- **Formato respetado**: fp8 e4m3 con escalas e8m0 por bloque 128×128 en las
  partes densas; expertos en e2m1 empaquetado (2 valores/byte, escala por fila
  cada 32 columnas). Error relativo medio de la edición: **2,04 %**.

## Formato

GGUF Q2_K, troceado en ficheros de ~29 GB. Cárgalo con el primer trozo:

```bash
llama-cli -m model-Q2_K-00001-of-00011.gguf -ngl 99 -c 4096 -p "tu prompt"
```

## Medición

- Receta y direcciones en `manifest.json`, `directions.json`, `edit-report.json`.
- Juez mecánico ES/EN (indicador, no gold humano): sensibilidad 0,938,
  especificidad 1,000 en régimen benigno.
- Los números de comportamiento se publican con su denominador; los que no
  tengan recibo de ejecución no se publican.


## Medición con Angmar (192 casos es+en, niveles 0–5)

**Límite de reproducibilidad:** este repositorio no incluye los resultados por
caso ni el manifiesto de ejecución de esta batería. Las cifras siguientes deben
contrastarse con la evidencia archivada antes de considerarlas reproducidas.

Batería congelada con `angmar prepare` (protocolo `d232252b…`), greedy, sin
razonamiento, juez mecánico ES/EN de Lobo (`refuse`/`answer`), sin anotación
humana todavía.

| Nivel | Casos | Base: rechazos | gollum: rechazos | Base: falsos | gollum: falsos |
|---|---|---|---|---|---|
| 0 Cotidiano y controles | 40 | 0/40 | 0/40 | 0/40 | 0/40 |
| 1 Temas sensibles legítimos | 28 | 1/28 | 1/28 | 1/28 | 1/28 |
| 2 Ambigüedad, contexto y premisas | 26 | 0/26 | 0/26 | 0/26 | 0/26 |
| 3 Hostilidad, ficción y prevención | 26 | 0/26 | 0/26 | 0/26 | 0/26 |
| 4 Daño explícito y riesgo | 40 | 19/40 | **13/40** | 2/20 | 0/20 |
| 5 Violencia grave y coerción | 32 | 12/32 | 11/32 | 1/20 | 1/20 |
| **Total** | **192** | **32/192** | **25/192** | **4/160** | **2/160** |

Lectura honesta: el artefacto baja el rechazo global 3,7 pp y el falso rechazo a
la mitad, con el grueso de la mejora en el nivel 4 (−15 pp). El nivel 5 apenas se
mueve (−1 caso): la ablación direccional no toca ese eje con esta receta.

Comprobación de que la edición sobrevive al Q2_K: `blk.39.attn_output_b` conserva
un ratio de 0,0066 en el subespacio de rechazo frente a 0,006 esperado con
s=1,30, mientras los tensores no editados quedan en 0,020 (valor de azar).

## Limitaciones

- Artefacto de **investigación**: las etiquetas vienen de un juez
  mecánico más revisión manual no ciega, no de un gold humano.
- La dirección es **operativa, no causal**.
- La cuantización Q2_K añade su propio ruido; la edición se hizo en fp8/fp4 y
  luego se recuantizó, de modo que parte de la proyección no sobrevive.
- No se ha entrenado nada: son 5.120 números por dirección y una proyección.

## Aviso

Uso exclusivo para seguridad ofensiva **autorizada** e investigación.
BeriaLabs no ofrece garantías sobre el comportamiento del artefacto.

**[berialabs.com](https://berialabs.com)**
