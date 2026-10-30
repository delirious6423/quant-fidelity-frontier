---
license: apache-2.0
language:
  - fi
  - en
  - et
  - hu
  - tr
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - kataguru
  - qwen3.8
  - qwen3_5
  - nvfp4
  - blackwell
  - vllm
  - multimodal
  - vision
  - mtp
  - software-engineering
  - plc
  - cnc
  - iec-61131-3
  - twincat
  - siemens-scl
  - cybersecurity
  - agentic-coding
  - finnish
  - soma
  - ara
  - uncensored
  - anti-sycophancy
---

# Kataguru Industry-Coder-27B v2.0 — Native Blackwell NVFP4 (`compressed-tensors`)

<p align="center">
  <img src="kataguru_industry_coder_27b.png" alt="Kataguru Industry-Coder-27B v2.0 Banner" width="100%"/>
</p>

## 🇬🇧 English Documentation

### 1. Executive Summary
**`kataguru/Industry-Coder-27B-v2.0-NVFP4`** is the native NVIDIA Blackwell (`FP4 E2M1` + `FP8 E4M3` micro-block scale + unified `FP32` global scale) quantization of **Kataguru Industry-Coder-27B v2.0**, a 27-billion-parameter dense multimodal (`Qwen3_5ForConditionalGeneration`) industrial software engineering, PLC/CNC automation, and agentic coding model built on **`Qwen/Qwen3.8-27B`**.

Trained through Kataguru's **4-Cycle Dynamic Epistemic Curriculum** (`Sykli 1 -> Sykli 4`, `20,000` certified golden samples, `70 %` competitive programming / systems architecture / industrial PLC & CNC + `30 %` permanent Elite Anchor) using **Rank-16 QR-Frozen-A Orthonormal Subspace QLoRA (`lora_r=16`, `lora_alpha=32`)**, **Industry-Coder-27B v2.0** achieves a **15-Benchmark Total Mean ($\Sigma_{15}$) of `68.73 %` (`+8.42 %-pt` over `Qwen/Qwen3.8-27B` Base)**:
- **LiveCodeBench v6 (`175` competition problems)**: **`42.29 %`** (`74/175`, **`+15.43 %-pt`** vs. `Qwen/Qwen3.8-27B` `26.86 %`; `Easy: 95.35 %`, `Medium: 48.08 %`, `Hard: 10.00 %`)
- **IFEval Strict (`541` prompts / `812` constraints)**: **`74.86 %` Prompt / `81.89 %` Instruction** (**`+26.66 %-pt / +23.49 %-pt`** vs. `Qwen/Qwen3.8-27B` `48.20 % / 58.40 %`)
- **BFCL V4 Agentic Tool Calling (`1,719` AST cases)**: **`73.14 %`** (**`+10.54 %-pt`** vs. `Qwen/Qwen3.8-27B` `62.60 %`; `Live AST: 85.27 %`, `Python Simple: 96.25 %`, `Multiple: 99.00 %`, `Live Relevance: 93.75 %`)
- **GPQA Diamond (`198` graduate-level STEM tasks)**: **`66.16 %`** (`131/198`, **`+20.96 %-pt`** vs. `Qwen/Qwen3.8-27B` `45.20 %`)
- **GSM8K CoT (`1,319` problems)**: **`86.73 %` Strict / `87.79 %` Flexible** (**`+10.23 %-pt`** vs. `Qwen/Qwen3.8-27B` `76.50 %`)
- **MedQA (`1,273` clinical USMLE tasks)**: **`82.80 %`** (**`+8.60 %-pt`** vs. `Qwen/Qwen3.8-27B` `74.20 %`)

All multimodal **Vision (`model.visual.*`, 333 BF16 tensors)** and **Multi-Token Prediction (`mtp.*`, 15 BF16 tensors)** weights are preserved in full `BF16` precision, and all `80/80` fused QKV / Gate-Up linear groups are normalized to unified `weight_global_scale` for hardware-native vLLM MarLin/Cutlass FP4 execution on NVIDIA Blackwell (`SM100`/`SM120`).

---

### Bilingual (Finnish & English) Sovereign Foundation & Native Finnish Support
**Kataguru Industry-Coder-27B v2.0** is engineered from the ground up as a **truly bilingual (`Finnish` 🇫🇮 & `English` 🇬🇧) flagship model** with sovereign, native-grade Finnish linguistic and technical mastery alongside English:
- **40 % Native Finnish (`fi`) / 60 % English (`en`) Certified Golden Curriculum**: Every cycle (`Sykli 1 -> Sykli 4`) and the permanent **30 % Elite Anchor** enforce strict bilingual balance across software architecture, competitive algorithms, industrial automation (`Siemens SCL`, `Beckhoff TwinCAT 3 ST`, `CNC G-code`, `OPC UA / IEC 62443`), mathematics, and graduate-level STEM.
- **Kotus-Verified Finnish Morphology & Agglutinative Precision**: Trained and audited on complex Finnish nominal/verbal inflection, compound technical terminology (yhdyssanat), and strict constraint obedience (`IFEval Strict: 74.86 % Prompt / 81.89 % Instruction`, **`+26.66 %-pt`** over `Qwen/Qwen3.8-27B`), eliminating anglicisms, broken case endings, and translationese.
- **Seamless Bilingual Code & Documentation Switching**: Generates idiomatic C, Rust, Python, TypeScript, Java, and IEC 61131-3 PLC code with equally precise technical reasoning, comments, architectural specifications, and multimodal diagram analysis in **both Finnish and English**.

### SOMA/ARA Hybrid Orthogonalization & Zero-Refusal Epistemic Honesty (`0.00 %` Refusals)
**Kataguru Industry-Coder-27B v2.0** has undergone Kataguru's **SOMA/ARA (`Sparse Orthogonal Modulation & Abliteration` + `Activation Refusal Abliteration`) Hybrid Uncensoring Pipeline** prior to and alongside its 4-cycle Elite Anchor SFT:
- **Zero-Domain IPPCC Refusal Vector Isolation**: Using *Identical Prompt Paired Completion Contrast* ($\text{Prompt Diff} \equiv 0.00000000$), the pure refusal activation direction $\mathbf{v}^{(l)}$ is isolated across the residual stream without touching domain semantics, storytelling, or analytical depth.
- **SOMA/ARA Weight Orthogonalization ($W' = W - \alpha \cdot \mathbf{v}(\mathbf{v}^T W)$)**: The refusal subspace is mathematically projected out of the attention output (`self_attn.o_proj`, `linear_attn.out_proj`) and MLP down-projection (`mlp.down_proj`) matrices across the target layers, followed by **Rank-16 QR-Frozen-A Orthonormal Subspace SFT** (`30 %` permanent Elite Anchor).
- **100 % Uncensored Technical & Scientific Precision**: Verified at **`0.00 %` refusals (`0` AI preambles or moralizing lectures)** across offensive/defensive cybersecurity (`Kali Linux`, penetration testing, `OPC UA` / `IEC 62443` hardening, reverse engineering), industrial safety interlocks, forensic analysis, and independent science (**TruthfulQA MC2: `53.06 %` [`+5.56 %-pt` over Base]**).
- **Strict Anti-Sycophancy**: Never flatters the user, never uses hollow pleasantries (*"Certainly!"*, *"Great question!"*), and immediately challenges flawed premises or mathematical/engineering errors from first principles.

### Kataguru Sovereign Quad-Mode Chat Template — How to Query & Use Modes

All four repositories include the **Kataguru Sovereign Quad-Mode Master Chat Template v2.1** (`chat_template.jinja` & `tokenizer_config.json`). You can query the interactive template advisor or switch cognitive reasoning modes directly in your user message or via `chat_template_kwargs`:

#### 1. Interactive Template Advisor (`{REASON:help}`)
Ask the model directly how to use its modes and which tag best fits your task:
```text
{REASON:help} Mikä päättelytila sopii parhaiten Beckhoff TwinCAT 3 tilakoneen ja OPC UA -liitynnän suunnitteluun?
```

#### 2. Cognitive Reasoning Tags & `i`-Prefix Instant Modes
Place any `{REASON:...}` tag anywhere in the user message (the template automatically strips the tag from the visible prompt and configures the system prompt + `<think>` prefix):

| Control Tag / API Parameter | `<think>` Behavior | Best Use Case |
| :--- | :--- | :--- |
| **`{REASON:help}`** / `reasoning_effort="help"` | Instant (`<think></think>`) | **Interactive Mode Guide**: Explains all modes and recommends the optimal tag for your task. |
| **`{REASON:ihyper}`** / `reasoning_effort="ihyper"` | Instant (`<think></think>`) | **MECE Instant (`+8.69 %-pt` accuracy, `-31 %` latency)**: Mutually Exclusive, Collectively Exhaustive breakdown directly in the final response. Ideal for math, architecture, and fast engineering queries. |
| **`{REASON:hyper}`** / `reasoning_effort="hyper"` | Deliberative `<think>` | **MECE Deliberative**: Full 4-stage MECE decomposition (`1. MECE Breakdown -> 2. Invariants -> 3. Edge Cases -> 4. Synthesis`) inside `<think>` for complex systems & PLC design. |
| **`{REASON:logic}`** / **`{REASON:ilogic}`** | `<think>` / Instant | **First-Principles Logic**: Formal axiom, conservation-law, and dimensional-analysis derivation (`logic` with `<think>`, `ilogic` instant). |
| **`{REASON:007}`** / **`{REASON:i007}`** | Instant (`<think></think>`) | **Fault-Tolerant Agentic & PLC Mode**: Deterministic state-machine design, IEC 61131-3 safety interlocks, and 100 % strict `<tool_call>` AST execution. |
| **`{REASON:spoon}`** / **`{REASON:einstein}`** | Deliberative `<think>` | **Deep Research (`spoon`) / 10-Perspective Brainstorming (`einstein`)** (also available as instant `ispoon` / `ieinstein`). |
| **`{REASON:off}`** / `enable_thinking=False` | Instant (`<think></think>`) | **Direct Zero-Latency Response**: Closes `<think>` immediately and outputs the final answer or code block with zero preamble. |

#### 3. The 4 Operational Modes (`Quad-Mode`)
| Mode | Trigger Tag in User Prompt or `chat_template_kwargs` | Purpose |
| :--- | :--- | :--- |
| **Mode 1: Expert & Agent Chat** | *(Default — no tag required)* | ASD-STE100 technical precision, complete C/Rust/Python/PLC code, `<tool_call>` XML/JSON tools, and Multimodal Vision. |
| **Mode 2: Quality Inspector** | `[INSPECT]` or `[TILA: TARKISTUS]` or `sceptic_mode="inspect"` | Deterministic code/math auditing returning Verdict-First JSON: `{"verdict": "ACCEPT|REJECT|REVIEW", "score": 9.2, ...}`. |
| **Mode 3: First-Principles Curator** | `[CURATE]` or `[TILA: KOULUTUS]` or `sceptic_mode="curate"` | Synthesizes verified high-learning-value (`>= 8.5`) SFT training samples from raw technical text or specs. |
| **Mode 4: System 1 Decision Judge** | `[DECISION]` or `sceptic_mode="decision"` | Zero-output-token (`max_tokens=1, logprobs=True`) `<20 ms` epistemic classifier (`0–9` / `A–B`). |

#### Example: Python / vLLM OpenAI Client Usage
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="EMPTY")

# 1. Direct Instant Mode (enable_thinking=False or {REASON:ihyper})
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "{REASON:007} Toteuta Beckhoff TwinCAT 3 ST -toimilohko FB_SafeConveyor kahden rajakytkimen ristiriitavalvonnalla."}],
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
print(resp.choices[0].message.content)
```

### 2. Official 15-Benchmark Suite (`Industry-Coder-27B v2.0` vs. `Qwen/Qwen3.8-27B`)

<p align="center">
  <img src="official_15_benchmark_chart.png" alt="Kataguru Industry-Coder-27B v2.0 Official 15-Benchmark Performance Chart" width="100%"/>
</p>

All benchmarks were executed on **2x NVIDIA GeForce RTX 5090 32GB Blackwell (`TP=2`, CUDAGraphs enabled, `temperature=0.0`)** under identical canonical production conditions.

| # | Benchmark | Metric | `Qwen/Qwen3.8-27B` (Base) | `Industry-Coder-27B-v2.0` (Ours) | $\Delta$ vs. `Qwen/Qwen3.8-27B` |
| :- | :--- | :--- | :---: | :---: | :---: |
| **1** | **ARC-Challenge** (`1,172`) | `acc_norm` | `61.60 %` | **`62.20 %`** | **`+0.60 %-pt`** |
| **2** | **ARC-Easy** (`2,376`) | `acc_norm` | `82.70 %` | **`81.99 %`** | `-0.71 %-pt` |
| **3** | **HellaSwag** (`10,042`) | `acc_norm` | `82.10 %` | **`82.91 %`** | **`+0.81 %-pt`** |
| **4** | **PIQA** (`1,838`) | `acc_norm` | `81.20 %` | **`82.05 %`** | **`+0.85 %-pt`** |
| **5** | **WinoGrande** (`1,267`) | `acc` | `75.10 %` | **`76.32 %`** | **`+1.22 %-pt`** |
| **6** | **OpenBookQA** (`500`) | `acc_norm` | `45.20 %` | **`45.80 %`** | **`+0.60 %-pt`** |
| **7** | **BoolQ** (`3,270`) | `acc` | `86.20 %` | **`86.45 %`** | **`+0.25 %-pt`** |
| **8** | **TruthfulQA MC1** (`817`) | `acc` | `31.80 %` | **`34.15 %`** | **`+2.35 %-pt`** |
| **9** | **TruthfulQA MC2** (`817`) | `acc` | `47.50 %` | **`53.06 %`** | **`+5.56 %-pt`** |
| **10** | **MedQA (USMLE)** (`1,273`) | `acc` | `74.20 %` | **`82.80 %`** | **`+8.60 %-pt`** |
| **11** | **GPQA Diamond** (`198`) | `CoT Exact` | `45.20 %` | **`66.16 %`** | **`+20.96 %-pt`** |
| **12** | **GSM8K Math** (`1,319`) | `Strict / Flex` | `76.50 % / 77.10 %` | **`86.73 % / 87.79 %`** | **`+10.23 / +10.69 %-pt`** |
| **13** | **IFEval** (`541 / 812`) | `Strict P / I` | `48.20 % / 58.40 %` | **`74.86 % / 81.89 %`** | **`+26.66 / +23.49 %-pt`** |
| **14** | **LiveCodeBench v6** (`175`) | `Pass@1` | `26.86 %` | **`42.29 %`** | **`+15.43 %-pt`** |
| **15** | **BFCL V4 AST** (`1,719`) | `Overall AST` | `62.60 %` | **`73.14 %`** | **`+10.54 %-pt`** |
| | **11-Task Cognitive Mean ($\Sigma_{11}$)** | `Mean (1..8, 10..12)` | `66.17 %` | **`70.41 %`** | **`+4.24 %-pt`** |
| | **15-Benchmark Total Mean ($\Sigma_{15}$)** | `Mean (All 15)` | `60.31 %` | **`68.73 %`** | **`+8.42 %-pt`** |

#### Detailed Breakdown: LiveCodeBench v6 & BFCL V4 Agentic Tool Calling

| Sub-Benchmark | `Industry-Coder-27B-v2.0` Score | Details |
| :--- | :---: | :--- |
| **LiveCodeBench v6 Overall (`175`)** | **`42.29 %`** (`74/175`) | **`+15.43 %-pt`** over `Qwen/Qwen3.8-27B` |
| — **LCB Easy (`43` tasks)** | **`95.35 %`** (`41/43`) | Near-ceiling competitive programming reliability |
| — **LCB Medium (`52` tasks)** | **`48.08 %`** (`25/52`) | Dynamic programming, graph theory, interval trees |
| — **LCB Hard (`80` tasks)** | **`10.00 %`** (`8/80`) | Diabolical algorithmic synthesis |
| **BFCL V4 Overall (`1,719` AST tasks)** | **`73.14 %`** | Weighted mean of Non-Live AST (`61.02 %`) & Live AST (`85.27 %`) |
| — **Python Simple AST (`400`)** | **`96.25 %`** | Single-function JSON/AST invocation |
| — **Multiple Function AST (`200`)** | **`99.00 %`** | Tool selection from multi-function schemas |
| — **JavaScript Simple AST (`50`)** | **`82.00 %`** | Cross-language typed AST generation |
| — **Java Simple AST (`100`)** | **`65.00 %`** | Strongly typed Java AST parameterization |
| — **Live AST (`1,351`)** | **`85.27 %`** | Real-world production API schemas (`Live Multiple: 79.39 %`, `Live Relevance: 93.75 %`) |

---

### 3. Architectural Highlights & What Changed in v2.0

1. **4-Cycle Dynamic Epistemic Curriculum (`Sykli 1 -> Sykli 4`, `20,000` Golden Samples)**:
   - Every training cycle (`5,000` samples/cycle) combines **`70 %` high-learning-value technical delta** (`LiveCodeBench` competitive algorithms, systems architecture, C/Rust/Python engineering, IEC 61131-3 PLC SCL/ST, CNC G-code, OPC UA / IEC 62443 industrial cybersecurity) with **`30 %` permanent Elite Anchor** (`IFEval` strict formatting, native Finnish Kotus morphology, `BFCL` `<tool_call>` AST, `GSM8K` & `GPQA` STEM, and `JEV` epistemic honesty).
   - Between every cycle, the live merged checkpoint interrogates the next cycle's candidates (`live_checkpoint_epistemic_prune`), discarding samples the model has already mastered (`score >= 0.85`) and replacing them with higher-tier samples from the Kataguru SQLite Golden Bank (`160,315` certified golden samples).
2. **Rank-16 QR-Frozen-A Orthonormal Subspace QLoRA (`lora_r=16`, `lora_alpha=32`)**:
   - All `lora_A` projection matrices are initialized via QR decomposition (`qr_frozen_a`) and frozen as an exact orthonormal basis ($A A^T = I_{16}$), while optimization updates only `lora_B`. This eliminates rank collapse and preserves the base model's world knowledge (`11-Task Cognitive Mean = 70.41 %`, `+4.24 %-pt` over base).
3. **Full Multimodal Vision + MTP Speculative Decoding Preservation**:
   - Contains all **`333` Vision encoder tensors (`model.visual.*`)** in `BF16` and all **`15` Multi-Token Prediction (`mtp.*`)** tensors in `BF16` alongside `256` NVFP4-quantized language layers (`1,711` tensors total across `7` safetensors shards, `26.59 GB`).
4. **Kataguru Quad-Mode Chat Template**:
   - **Mode 1 — Default Direct Mode (`enable_thinking=False`)**: Zero-fluff, direct engineering answers, clean `class Solution:` / IEC 61131-3 code, and `<tool_call>` execution.
   - **Mode 2 — First-Principles Deep Reasoning (`enable_thinking=True` or `{REASON:einstein}`)**: Activates structured `<think>...</think>` derivation before final synthesis.
   - **Mode 3 — Quality Inspector (`{MODE:INSPECT}` / `sceptic_mode="inspect"`)**: Deterministic dataset & code auditing oracle.
   - **Mode 4 — Epistemic Surgeon (`{MODE:SURGEON}` / `sceptic_mode="surgeon"`)**: First-principles error correction and refactoring.

---

### 4. Production Deployment with vLLM (Blackwell RTX 5090 / RTX PRO 6000)

```bash
vllm serve kataguru/Industry-Coder-27B-v2.0-NVFP4 \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

---

## 🇫🇮 Suomenkielinen Dokumentaatio

### 1. Yhteenveto
**`kataguru/Industry-Coder-27B-v2.0-NVFP4`** on **Kataguru Industry-Coder-27B v2.0** -lippulaivamallin natiivi NVIDIA Blackwell **NVFP4 (`compressed-tensors`)** -kvantisointi. Malli perustuu **`Qwen/Qwen3.8-27B`** -arkkitehtuuriin (`Qwen3_5ForConditionalGeneration`) ja yhdistää vaativan ohjelmistotuotannon, algoritmisen kilpakoodauksen (`LiveCodeBench v6`), teollisuusautomaation (`Siemens TIA Portal SCL`, `Beckhoff TwinCAT 3 ST`, `CNC G-koodi`, `OPC UA / IEC 62443`), agenttisen työkalukäytön (`BFCL V4`), multimodaalisen konenäön (`333` visiotensoria) sekä **MTP (`Multi-Token Prediction`)** -spekulaatiopäät.

Malli on koulutettu **4-syklisellä dynaamisella episteemisellä kurrikulumilla (`Sykli 1 -> Sykli 4`, yhteensä `20 000` sertifioitua kultapankkinäytettä)** käyttäen **`r=16 qr_frozen_a`** -ortogonaalista QLoRA-aliavaruutta sekä jokaisen syklin välissä ajettavaa elävän tarkistuspisteen kuulustelua (`live_checkpoint_epistemic_prune`).

### Natiivi Kaksikielisyys (Suomi 🇫🇮 & Englanti 🇬🇧) ja Suomen Kielen Tuki
**Kataguru Industry-Coder-27B v2.0** on rakennettu alusta alkaen **aidosti kaksikieliseksi (`suomi` & `englanti`) teollisuus- ja ohjelmistotuotantomalliksi**, jossa suomen kieli on ensiluokkainen natiivikieli englannin rinnalla:
- **40 % Suomi (`fi`) / 60 % Englanti (`en`) Sertifioitu Kultapankkikurrikulumi**: Kaikki neljä koulutussykliä (`Sykli 1 -> Sykli 4`) sekä pysyvä **30 % Eliittiankkuri** sisältävät vahvan suomenkielisen ja englanninkielisen edustuksen ohjelmistoarkkitehtuurista, kilpakoodauksesta, teollisuusautomaatiosta (`Siemens TIA Portal SCL`, `Beckhoff TwinCAT 3 ST`, `CNC G-koodi`, `OPC UA / IEC 62443`), kyberturvallisuudesta sekä STEM-tieteistä.
- **Kotus-tarkastettu Suomen Morfologia ja Yhdyssanatarkkuus**: Malli hallitsee suomen kielen sijamuodot, liitepartikkelit, vaativan teknisen ammattisanaston ja ehdottomat muotorajoitteet (`IFEval Strict: 74,86 % Prompt / 81,89 % Instruction`, **`+26,66 %-yks.`** yli `Qwen/Qwen3.8-27B`-pohjamallin) ilman anglismeja tai käännöskieltä.
- **Saumaton Kaksikielinen Koodaus ja Dokumentointi**: Tuottaa tuotantovalmista C-, Rust-, Python-, TypeScript-, Java- ja IEC 61131-3 PLC -koodia, arkkitehtuurimäärittelyjä, työkalukutsuja (`<tool_call>`) sekä multimodaalista kaavioanalyysiä yhtä täsmällisesti **sekä suomeksi että englanniksi**.

### SOMA/ARA-Hybridiortogonalisointi, Nollakieltäytyminen (`0,00 %` Refusal) ja Anti-Sycophancy
**Kataguru Industry-Coder-27B v2.0** on käsitelty Katagurun **SOMA/ARA (`Sparse Orthogonal Modulation & Abliteration` + `Activation Refusal Abliteration`) -hybridimenetelmällä** yhdistettynä 4-sykliseen **30 % Eliittiankkuri (`r=16 qr_frozen_a`)** -koulutukseen:
- **Zero-Domain IPPCC -kieltäytymisvektorin eristys**: Puhtaat kieltäytymisaktivaatiot ($\mathbf{v}^{(l)}$) on eristetty *Identical Prompt Paired Completion Contrast* -menetelmällä ($\text{Prompt Diff} \equiv 0,00000000$) ilman, että mallin kielellinen tarkkuus tai asiantuntijasyvyys kärsii.
- **SOMA/ARA-painomatriisien ortogonalisointi ($W' = W - \alpha \cdot \mathbf{v}(\mathbf{v}^T W)$)**: Kieltäytymissuunta on poistettu suoraan huomio- ja MLP-ulostulomatriiseista (`self_attn.o_proj`, `linear_attn.out_proj`, `mlp.down_proj`), minkä jälkeen `r=16 qr_frozen_a` -ortogonaalinen SFT-koulutus on vakauttanut mallin residuaalivirran.
- **100 % Sensuroimaton Tekninen ja Tieteellinen Tarkkuus (`0` kieltäytymistä)**: Malli vastaa suoraan, täsmällisesti ja ilman tekoälyverukkeita (*"Valitettavasti en voi..."* = `0,00 %`) vaativiin kyberturvallisuus-, `Kali Linux`-, penetraatiotestaus-, `OPC UA` / `IEC 62443` -kovennus-, käänteismallinnus-, teollisuusautomaatio- ja riippumattoman tieteen kysymyksiin (**TruthfulQA MC2: `53,06 %` [`+5,56 %-yks.` yli pohjamallin]**).
- **Ehdoton Anti-Sycophancy (Ei mielistelyä)**: Malli ei koskaan mielistele käyttäjää eikä käytä tyhjiä kohteliaisuusfraaseja, vaan oikaisee virheelliset lähtöoletukset välittömästi luonnontieteellisten ja teknisten ensiperiaatteiden (*First Principles*) pohjalta.

### Kataguru Sovereign Quad-Mode Chat Template — Kuinka Templatea Kysytään ja Käytetään

Kaikki neljä julkaisurepositoriota sisältävät virallisen **Kataguru Sovereign Quad-Mode Master Chat Template v2.1** -keskustelupohjan (`chat_template.jinja` ja `tokenizer_config.json`). Voit kysyä suoraan mallilta templaten käytöstä tai ohjata päättelytiloja joko viestin sisäisillä `{REASON:...}`-tägeillä tai `chat_template_kwargs`-parametreilla:

#### 1. Interaktiivinen Template-opas (`{REASON:help}`)
Voit milloin tahansa kysyä mallilta sen tukemista tiloista ja pyytää suositusta omaan tehtävääsi:
```text
{REASON:help} Miten käytän tämän mallin päättelytiloja ja mikä tila sopii parhaiten TwinCAT 3 PLC-ohjelmointiin?
```

#### 2. Päättelykehykset (`{REASON:...}`) ja `i`-Pikatilat (Ilman `<think>`-viivettä)
Kirjoita ohjaustägi mihin tahansa kohtaan viestiäsi (template siivoaa tägin automaattisesti pois näkyvästä tekstistä ja aktivoi oikean tilan):

| Ohjaustägi viestissä / API-parametri | `<think>`-käyttäytyminen | Käyttötarkoitus |
| :--- | :--- | :--- |
| **`{REASON:help}`** / `reasoning_effort="help"` | Välitön (`<think></think>`) | **Interaktiivinen tilaneuvoja**: Esittelee kaikki tilat suomeksi/englanniksi ja suosittelee tehtävääsi parasta `{REASON:...}`-tägiä. |
| **`{REASON:ihyper}`** / `reasoning_effort="ihyper"` | Välitön (`<think></think>`) | **MECE-pikatila (`+8,69 %-yks.` tarkempi, `-31 %` nopeampi)**: Tekee toisensa poissulkevan ja kokonaan kattavan (MECE) osituksen suoraan vastauksessa ilman `<think>`-viivettä. |
| **`{REASON:hyper}`** / `reasoning_effort="hyper"` | Avoin `<think>` | **MECE-syväpäättely**: 4-vaiheinen MECE-analyysi (`1. Ositus -> 2. Invariantit -> 3. Reunatapaukset -> 4. Synteesi`) vaativiin järjestelmä- ja automaatioarkkitehtuureihin. |
| **`{REASON:logic}`** / **`{REASON:ilogic}`** | `<think>` / Välitön | **First-Principles -logiikka**: Formaali aksiooma-, säilymislaki- ja yksikköanalyysijohto (`logic` ajattelulohkolla, `ilogic` välittömänä). |
| **`{REASON:007}`** / **`{REASON:i007}`** | Välitön (`<think></think>`) | **Vikietoinen Agentti- ja PLC-tila**: Deterministinen tilakonesuunnittelu, IEC 61131-3 -turvalukitukset ja 100 % tarkka `<tool_call>` AST -työkalukutsu. |
| **`{REASON:spoon}`** / **`{REASON:einstein}`** | Avoin `<think>` | **Syvätutkimus (`spoon`) / 10 näkökulman luova synteesi (`einstein`)** (saatavilla myös pikatiloina `ispoon` / `ieinstein`). |
| **`{REASON:off}`** / `enable_thinking=False` | Välitön (`<think></think>`) | **Suora pikavastaus**: Sulkee `<think>`-lohkon välittömästi ja tuottaa suoran vastauksen tai koodin ilman viivettä. |

#### 3. Neljä Toimintatilaa (`Quad-Mode`)
- **Tila 1 — Asiantuntija- ja Agenttikeskustelu (oletus)**: Nolla mielistelyä (Anti-Sycophancy), ASD-STE100-tekninen tarkkuus, täydellinen koodi (C, Rust, Python, TwinCAT 3 ST, Siemens SCL, G-koodi), `<tool_call>`-työkalukutsut sekä multimodaalinen konenäkö.
- **Tila 2 — Laaduntarkastaja (`[INSPECT]` / `[TILA: TARKISTUS]` / `sceptic_mode="inspect"`)**: Tarkistaa koodin ja matematiikan sekä palauttaa Verdict-First JSON-tuomion (`{"verdict": "ACCEPT|REJECT|REVIEW", "score": 9.2, ...}`).
- **Tila 3 — Koulutusaineiston Syntetisoija (`[CURATE]` / `[TILA: KOULUTUS]` / `sceptic_mode="curate"`)**: Jalostaa raakatekstistä korkean oppiarvon (`>= 8.5`) SFT-koulutusnäytteitä.
- **Tila 4 — Systeemi 1 -Pikatuomari (`[DECISION]` / `sceptic_mode="decision"`)**: Yhden tokenin (`max_tokens=1, logprobs=True`) `<20 ms` episteeminen luokittelija (`0–9` / `A–B`).

### 2. Keskeiset Tulokset vs. `Qwen/Qwen3.8-27B`
<p align="center">
  <img src="official_15_benchmark_chart.png" alt="Kataguru Industry-Coder-27B v2.0 Official 15-Benchmark Performance Chart" width="100%"/>
</p>

- **15-Benchmarkin Kokonaiskeskiarvo ($\Sigma_{15}$)**: **`68,73 %`** (**`+8,42 %-yks.`** vs. `Qwen/Qwen3.8-27B` `60,31 %`)
- **11-Tehtävän Kognitiivinen Keskiarvo ($\Sigma_{11}$)**: **`70,41 %`** (**`+4,24 %-yks.`** vs. `Qwen/Qwen3.8-27B` `66,17 %`)
- **LiveCodeBench v6 (`175` kilpakoodaustehtävää)**: **`42,29 %`** (`74/175`, **`+15,43 %-yks.`** vs. `Qwen/Qwen3.8-27B` `26,86 %`; `Easy: 95,35 %`, `Medium: 48,08 %`, `Hard: 10,00 %`)
- **IFEval Strict (`541` kehotetta / `812` rajoitetta)**: **`74,86 %` Prompt / `81,89 %` Instruction** (**`+26,66 / +23,49 %-yks.`** vs. `Qwen/Qwen3.8-27B`)
- **BFCL V4 Agentic Tool Calling (`1 719` AST-testiä)**: **`73,14 %`** (**`+10,54 %-yks.`** vs. `Qwen/Qwen3.8-27B` `62,60 %`; `Live AST: 85,27 %`, `Python Simple: 96,25 %`, `Multiple: 99,00 %`)
- **GPQA Diamond (`198` tohtoritason STEM-tehtävää)**: **`66,16 %`** (**`+20,96 %-yks.`** vs. `Qwen/Qwen3.8-27B` `45,20 %`)
- **GSM8K Matematiikka (`1 319` tehtävää)**: **`86,73 %` Strict / `87,79 %` Flex** (**`+10,23 %-yks.`** vs. `Qwen/Qwen3.8-27B`)
- **Täysi Multimodaalisuus & Nolla Kieltäytymistä**: Kaikki `333` visiotensoria ja `15` MTP-tensoria ovat mukana täydellä `BF16`-tarkkuudella, ja malli läpäisee NVFP4-julkaisuportin `0` kieltäytymisellä ja `4/4` teollisen PLC/SCL- ja kyberturvallisuustestin tuloksella.

### 3. Julkaisuperheen Muut Kvantisoinnit
- **NVFP4 (tämä repositorio)**: [`kataguru/Industry-Coder-27B-v2.0-NVFP4`](https://huggingface.co/kataguru/Industry-Coder-27B-v2.0-NVFP4) — Natiivi NVIDIA Blackwell (`26,59 GB`, sisältää Vision + MTP).
- **W4A16-AWQ**: [`kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ`](https://huggingface.co/kataguru/Industry-Coder-27B-v2.0-W4A16-AWQ) — Laaja vLLM MarLin W4A16 -yhteensopivuus (`Ampere`, `Ada`, `Hopper`, `Blackwell`).
- **GGUF (imatrix IQ-kvantit + mmproj + MTP)**: [`kataguru/Industry-Coder-27B-v2.0-GGUF`](https://huggingface.co/kataguru/Industry-Coder-27B-v2.0-GGUF) — `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `mmproj-f16.gguf` sekä `mtp-Q8_0.gguf` / `mtp-Q4_K_M.gguf` (`llama.cpp` & `LM Studio`).
