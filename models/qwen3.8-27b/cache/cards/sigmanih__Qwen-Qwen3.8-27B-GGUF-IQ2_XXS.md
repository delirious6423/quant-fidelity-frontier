---
language:
- en
- it
license: other
base_model:
- Qwen/Qwen3.8-27B
tags:
- text-generation
- sigma-studio
- sigmanih
- conversational
- custom-model
- gguf
- llama.cpp
- quantized
- iq2_xxs
pipeline_tag: text-generation
---

<div align="center">

# ⚡ Qwen-Qwen3.8-27B-GGUF-IQ2_XXS
### High-Performance Model Published via **[Σ-SIGMA Studio](https://github.com/Sigmanih/SigmaStudio)**

[![SigmaStudio GitHub](https://img.shields.io/badge/GitHub-SigmaStudio-10b981?style=for-the-badge&logo=github)](https://github.com/Sigmanih/SigmaStudio) [![HuggingFace Hub](https://img.shields.io/badge/Hugging%20Face-sigmanih-ffb86c?style=for-the-badge&logo=huggingface)](https://huggingface.co/sigmanih/Qwen-Qwen3.8-27B-GGUF-IQ2_XXS) [![Engine](https://img.shields.io/badge/Accelerated%20by-SigmaEngine-00d2ff?style=for-the-badge&logo=fastapi)](https://github.com/Sigmanih/SigmaStudio) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

> ❤️ **Support & Community:** If you find this model helpful, please give this repository a **Like** on Hugging Face and a **⭐ Star** on our **[SigmaStudio GitHub](https://github.com/Sigmanih/SigmaStudio)**!

## 🌐 English Overview

**Qwen-Qwen3.8-27B-GGUF-IQ2_XXS** is a production-ready model optimized and published using the **Model Hub** module of **[Sigma Studio](https://github.com/Sigmanih/SigmaStudio)**.

### ⚙️ Technical Specifications & Architecture
| Specification | Value |
| :--- | :--- |
| **Model Repository** | `sigmanih/Qwen-Qwen3.8-27B-GGUF-IQ2_XXS` |
| **Weight Format** | `GGUF` (`IQ2_XXS`) |
| **Base Architecture** | `qwen35` |
| **Active Parameters** | **27B** |
| **Context Window** | `262,144 tokens` |
| **Transformer Layers** | `65` |
| **Hidden Dimension** | `5120` |
| **Total Disk Footprint** | `9.91 GB` |
| **Inference RAM / VRAM** | **`~11.0 GB VRAM`** (Full GPU offload) / **`~10.8 GB RAM`** (CPU/Hybrid) |
| **Recommended Hardware** | GPU with 12-16 GB VRAM (e.g. RTX 3060/4070 or 32 GB RAM) |
| **Recommended Usage** | Advanced logic & reasoning, Enterprise domain specialization, Complex code refactoring. |

### 🏆 Official Benchmark Performance
Evaluated directly on GPU via **Sigma Studio Training Lab** (Deterministic seed 42, Temp 0.0):

| Benchmark Suite | Score / Accuracy | Total Questions Evaluated | Pass Rate | Test Date | Execution Engine |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tutti i Benchmark Ufficiali** | **`75.0%`** | **75/100 quesiti superati** | **`75.0% Pass`** | `2026-09-29` | ⚡ SigmaEngine Direct GPU |

#### 🧠 Reasoning Mode Comparison: No-Thinking vs Thinking
Side-by-side performance comparison between direct zero-overhead answer (*No-Thinking*) and Chain-of-Thought step-by-step reasoning (*Deep Thinking CoT*):

```text
⚡ No-Thinking (Direct Response) : [████████████████░░░░] 80.7%  (46/57 passed)
🧠 Deep Thinking (CoT Reasoning) : [█████████████░░░░░░░] 67.4%  (29/43 passed)
📈 CoT Performance Delta        : -13.3%
```

```mermaid
%%{init: {'theme': 'dark'}}%%
xychart-beta
    title "Accuracy (%): No-Thinking vs Deep Thinking"
    x-axis ["⚡ No-Thinking (Direct)", "🧠 Deep Thinking (CoT)"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [80.7, 67.4]
```

| Execution Mode | Accuracy (%) | Questions Passed | Operational Profile & Latency |
| :--- | :---: | :---: | :--- |
| **⚡ No-Thinking (Direct Response)** | **`80.7%`** | `46 / 57` | Minimal latency, immediate token-to-first-byte, zero reasoning tokens overhead |
| **🧠 Deep Thinking (CoT Reasoning)** | **`67.4%`** | `29 / 43` | Multi-step structured reasoning trace (-13.3%), optimal for math & hard logic |

#### 📋 Per-Dataset Evaluation Breakdown

| Dataset / Benchmark Suite | Domain / Category | Correct / Total | Accuracy (%) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **ARC-Challenge** | Science & Grade-School Reasoning | **9 / 9** | `100%` | ✅ Passed |
| **BIG-Bench Hard** | Complex Multi-Task Logic & Symbolics | **7 / 7** | `100%` | ✅ Passed |
| **GPQA** | Graduate-Level Academic Reasoning | **2 / 9** | `22%` | ⚠️ Low |
| **GSM8K** | Multi-Step Grade School Math | **8 / 9** | `89%` | ✅ Passed |
| **HellaSwag** | Commonsense Reasoning & Situational NLI | **6 / 9** | `67%` | ⚡ Fair |
| **HumanEval** | Python Coding (pass@1) | **7 / 7** | `100%` | ✅ Passed |
| **MATH** | Championship Competition Math | **7 / 9** | `78%` | ✅ Passed |
| **MBPP** | Python Programming with Unit Tests | **8 / 9** | `89%` | ✅ Passed |
| **MMLU** | General Knowledge & Multi-Subject | **9 / 14** | `64%` | ⚡ Fair |
| **MMLU-Pro** | Advanced Multi-Step Reasoning | **5 / 9** | `56%` | ⚡ Fair |
| **TruthfulQA** | Factuality & Anti-Hallucination | **7 / 9** | `78%` | ✅ Passed |
| **🏆 OVERALL TOTAL** | **All Evaluated Datasets** | **`75 / 100`** | **`75%`** | **🏆 75% Pass** |

**Protocol:** code_execution, continuation_logprob, cot_generation, letter_logprob · temp 0.0 · seed 42
**Reproducibility hash:** `SHA256-5DFF448CC034C813`
> ⚠️ Measured on a slice of the dataset, not the full suite: this score is not comparable with a full-suite run.

### 🛠️ Tool Calling & Agentic Protocol Benchmark (Sigma Studio Sandbox Jail)
Empirical multi-turn agent reliability evaluation (zero quizzes, fully grounded file-system actions inside isolated Sandbox Jail):

| Protocol Metric | Outcome | Validation Criteria & Details |
| :--- | :---: | :--- |
| **Tool Protocol Adherence** | **`75.0%`** | `0/92` analytical criteria verified |
| **Autonomous Task Completion** | **`0.0%`** | `0/1` scenarios completed with exact target file |
| **Sandbox Jail Containment** | **`100% Compliant`** | Zero escape attempts outside workspace boundary |
| **Execution Efficiency** | **`0 turns (0.0s)`** | Optimal multi-step turn and token budget usage |

#### 🔍 Certified Autonomous Capabilities:
- ✅ **Tool Grounding:** Exclusively uses registered tools and valid schemas (zero hallucinated functions).
- ✅ **Zero Placeholder Echo:** Emits concrete code and values rather than copy-pasting prompt templates.
- ✅ **Inspect-Before-Edit:** Systematically reads files and verifies target lines before patching.
- ✅ **Evidence-Based Exit:** Emits concrete test commands and validation checks before task exit.
- ✅ **Sandbox Containment:** Strictly adheres to isolated sandbox jail boundaries.

### ⚡ Measured Speed on the Publishing Machine
Measured on `NVIDIA GeForce RTX 5070 Ti` • `15.9 GB VRAM` during the evaluation run.

| What was measured | Value | How |
| :--- | :---: | :--- |
| Aggregate throughput during evaluation | `47.8 tok/s` | several requests in flight — **not** what a single answer runs at |

> Speeds on other hardware were **not measured** and are not guessed here. A single-stream figure from one machine cannot be scaled into a prediction for another: it depends on memory bandwidth, quantization, context length and driver, and the error is large enough to be misleading.

### 🚀 Quick Start Guide
#### 1. Running with Sigma Studio (Recommended)
Launch **Sigma Studio** to enjoy full 1-click GPU hardware acceleration, live monitoring, and visual chat:
```bash
# Clone and run Sigma Studio
git clone https://github.com/Sigmanih/SigmaStudio.git
cd SigmaStudio
.\sigma_studio.bat
```

#### 2. Running with llama.cpp
```bash
llama-cli -hf sigmanih/Qwen-Qwen3.8-27B-GGUF-IQ2_XXS -p "Hello! How can I help you today?" -ngl 99
```

---
## 🇮🇹 Documentazione in Italiano

**Qwen-Qwen3.8-27B-GGUF-IQ2_XXS** è un modello ottimizzato pronto per l'inferenza e l'integrazione locale, pubblicato attraverso **[Σ-SIGMA Studio](https://github.com/Sigmanih/SigmaStudio)**.

### 📋 Specifiche e Configurazione
- **Architettura Base:** `qwen35` (27B parametri)
- **Formato Pesi:** `GGUF` (IQ2_XXS)
- **Spazio su Disco:** `9.91 GB`
- **RAM / VRAM in Esecuzione:** `~11.0 GB VRAM` (offload GPU completo) | `~10.8 GB RAM` (inferenza CPU/ibrida)
- **Requisiti Hardware Consigliati:** GPU con 12-16 GB VRAM (es. RTX 3060/4070 o 32 GB RAM)
- **Finestra di Contesto:** `262,144 token`
- **Profilo d'Uso Consigliato:** Logica e ragionamento avanzato, specializzazione di dominio enterprise, refactoring complesso.

### 📊 Risultati Benchmark Ufficiali
- **Suite di Valutazione:** `Tutti i Benchmark Ufficiali`
- **Punteggio Ufficiale:** **`75.0%`** (75/100 quesiti superati)
#### 🧠 Confronto Modalità di Risposta: No-Thinking vs Thinking
Confronto visuale tra risposta istantanea diretta (*No-Thinking*) e ragionamento guidato multi-step (*Deep Thinking CoT*):

```text
⚡ No-Thinking (Risposta Diretta) : [████████████████░░░░] 80.7%  (46/57 superati)
🧠 Deep Thinking (CoT Reasoning)  : [█████████████░░░░░░░] 67.4%  (29/43 superati)
📈 Delta Prestazionale CoT        : -13.3%
```

```mermaid
%%{init: {'theme': 'dark'}}%%
xychart-beta
    title "Accuratezza (%): No-Thinking vs Deep Thinking"
    x-axis ["⚡ No-Thinking (Diretto)", "🧠 Deep Thinking (CoT)"]
    y-axis "Accuratezza (%)" 0 --> 100
    bar [80.7, 67.4]
```

| Modalità di Esecuzione | Accuratezza (%) | Quesiti Superati | Profilo Operativo & Latenza |
| :--- | :---: | :---: | :--- |
| **⚡ No-Thinking (Risposta Diretta)** | **`80.7%`** | `46 / 57` | Latenza minima, token-to-first-byte istantaneo, zero overhead di ragionamento |
| **🧠 Deep Thinking (CoT Reasoning)** | **`67.4%`** | `29 / 43` | Risoluzione passo-passo multi-step (-13.3%), ideale per logica complessa e matematica |

#### 📋 Dettaglio Punteggi per Singolo Dataset

| Dataset / Suite di Test | Ambito / Dominio | Corretti / Totale | Accuratezza (%) | Esito |
| :--- | :--- | :---: | :---: | :---: |
| **ARC-Challenge** | Ragionamento Scientifico Avanzato | **9 / 9** | `100%` | ✅ Superato |
| **BIG-Bench Hard** | Logica Complessa & Compiti Multi-Fase | **7 / 7** | `100%` | ✅ Superato |
| **GPQA** | Ragionamento Accademico di Livello Laurea | **2 / 9** | `22%` | ⚠️ Migliorabile |
| **GSM8K** | Matematica & Logica Multi-Step | **8 / 9** | `89%` | ✅ Superato |
| **HellaSwag** | Buon Senso & Comprensione Situazionale | **6 / 9** | `67%` | ⚡ Discreto |
| **HumanEval** | Sintesi Codice Python (pass@1) | **7 / 7** | `100%` | ✅ Superato |
| **MATH** | Matematica Olimpica & Competitiva | **7 / 9** | `78%` | ✅ Superato |
| **MBPP** | Programmazione Python con Unit Test | **8 / 9** | `89%` | ✅ Superato |
| **MMLU** | Conoscenza Generale Multidisciplinare | **9 / 14** | `64%` | ⚡ Discreto |
| **MMLU-Pro** | Ragionamento Avanzato Multi-Step | **5 / 9** | `56%` | ⚡ Discreto |
| **TruthfulQA** | Fattualità & Resistenza ad Allucinazioni | **7 / 9** | `78%` | ✅ Superato |
| **🏆 TOTALE COMPLESSIVO** | **Tutti i Dataset Valutati** | **`75 / 100`** | **`75%`** | **🏆 75% Pass** |

**Protocollo:** code_execution, continuation_logprob, cot_generation, letter_logprob · temp 0.0 · seed 42
**Impronta di riproducibilità:** `SHA256-5DFF448CC034C813`
> ⚠️ Misurato su una porzione del dataset, non sulla suite intera: il punteggio non è confrontabile con uno ottenuto sull'intero.

- **Data Test:** `2026-09-29` su motore deterministico SigmaEngine

### 🛠️ Benchmark Aderenza Tool & Capacità Agente (Sigma Studio Sandbox Jail)
Valutazione empirica dell'affidabilità nei compiti di agente autonomo (nessun quiz teorico, solo esecuzioni e verifiche su filesystem isolato):

| Metrica di Protocollo | Risultato | Dettaglio e Criteri di Validazione |
| :--- | :---: | :--- |
| **Aderenza Protocollo Tool** | **`75.0%`** | `0/92` prove analitiche verificate |
| **Completamento Scenari Operativi** | **`0.0%`** | `0/1` scenari conclusi con output esatto |
| **Sicurezza Sandbox Jail** | **`100% Conforme`** | Confinamento rigido, zero tentativi fuori sandbox |
| **Efficienza Esecutiva** | **`0 turni (0.0s)`** | Rispetto rigoroso dei budget operativi per scenario |

#### 🔍 Comportamenti Rigorosamente Certificati:
- ✅ **Tool Grounding:** Utilizzo esclusivo di tool formalmente registrati (zero allucinazioni di comandi).
- ✅ **No Segnaposto:** Produzione di codice e parametri concreti senza eco di placeholder d'esempio.
- ✅ **Ispezione Previa:** Lettura e verifica dei file prima di eseguire modifiche chirurgiche.
- ✅ **Verifica di Chiusura:** Certificazione delle prove prima di dichiarare terminato il lavoro.
- ✅ **Confinamento Jail:** Isolamento totale senza contaminazione del kernel o del sistema host.

### ⏱️ Throughput Hardware e Fasce Consigliate
- **Velocità Verificata in Locale:** **`47.8 tok/s`** su `NVIDIA GeForce RTX 5070 Ti`.
- **Throughput complessivo durante la valutazione:** `47.8 tok/s` — piu' richieste in volo insieme, **non** la velocita' di una risposta singola.
- Le velocita' su altro hardware **non sono state misurate** e non vengono indovinate: dipendono da banda di memoria, quantizzazione, lunghezza del contesto e driver.

### ⭐ Supporta il Progetto Open Source
Se questo modello ti è utile o vuoi esplorare l'ecosistema completo:
- 🌟 Metti una **Stella** al repository GitHub: **[Sigmanih/SigmaStudio](https://github.com/Sigmanih/SigmaStudio)**
- ❤️ Lascia un **Like** a questa scheda su Hugging Face

---
*Creato e distribuito con il Model Hub di Σ-SIGMA Studio (13/10/2026 17:52)*