---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - lfm2
  - lfm2moe
  - liquid-ai
  - moe
  - llama-cpp
  - pocketpal
  - ios
  - mobile
  - tool-calling
  - agent
  - universe-network
  - renormalization-group
---

# UniverseNet-LFM2.5-24B-Pocket-8.6B-GGUF (3.996 GB Mobile Pocket MoE Agent — v4.1)

**UniverseNet-LFM2.5-24B-Pocket-8.6B** is a physics-transmuted mobile edition of **LiquidAI/LFM2.5-24B-A2B** (`24.16B -> 8.65B` total parameters, `2.34B -> 1.48B` active parameters per token, `14.53 GB -> 3.996 GB` GGUF, **`3.64x` size reduction**) created via the **Universe Network Quantum-Atomic Renormalization-Group (RG) & Covalent Orbital Condensation framework**.

### Checkpoints in this Repository
- **`UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K.gguf` (Recommended Flagship — v4.1, `3.996 GB` / `3.722 GiB`):** Condenses all 64 MoE experts (`1,536 -> 512` orbitals) on clean Teacher trajectories (`USE_BK = False`) with **Hybrid Coupled Photoelectric Orbital Ranking**, **Per-Expert Routed Dipole RMS Norm Restoration**, **Pass-2 Intra-Layer Co-Routed BT-GS Refinement**, and **Tail-Preserving `1,152`-Token Agentic Calibration**. Achieves **`9/11` (`81.8%`, `9/10` single/2-turn)** on the **Babbo iOS Mobile Agent Journey Evaluation** (vs. `10/11` for the `14.53 GB` Teacher), **`65.0%` on GSM8K**, **`60.0%` on ARC-Challenge**, **`60.0%` on BFCL-AST Function Calling (`100%` `<think>` closure)**, and **`50.0% / 61.3%` on `google/IFEval`**.
- **`UniverseNet-LFM2.5-24B-Pocket-8.6B-v3-Q3K.gguf` (Legacy v3, `3.996 GB` / `3.722 GiB`):** First-generation multi-scale encyclopedic/chat orbital condensation (`8/8` on short single-turn `eval_lfm25_pocket_chat.py`).

Because the condensation preserves the exact `lfm2moe` tensor layout (`expert_feed_forward_length = 512`, divisible by the 256-element `Q3_K` / `Q4_K` super-block size), this checkpoint runs **100% natively in `llama.cpp`, PocketPal AI, LLM Farm, Enclave, Ollama, and LM Studio** on **iPhone 16 Pro (`8 GB` RAM)** and **Apple Silicon** in **Airplane Mode** at **`444.14 tok/s` prefill and `43.88 tok/s` decode (`187 MB` working RAM)**.

---

## What's New in v4.1 (`UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K.gguf`)

1. **Zero Residual Drift Clean-Trajectory Condensation (`USE_BK = False`):** Keeps all 30 LIV gated short-convolution blocks (`conv1d` kernel size 3) and all 10 global GQA Attention layers (`blk.{2,6,10,14,18,22,26,30,34,38}`) 100% bit-identical to the Teacher, and condenses every layer's 64 MoE experts strictly on clean Teacher hidden states (`h_tea`) with **zero cross-layer residual drift** (preventing `W_down` from absorbing blurred upstream errors).
2. **Law 1 & Law 2 — Hybrid Coupled Photoelectric Orbital Ranking:** Scores each expert's `1,536` SwiGLU orbitals by coupling receptor sensitivity `||W_gate · w_fnorm||_2 · ||W_up · w_fnorm||_2` with emission norm `||W_down||_2`, routed bulk RMS current `0.55 · RMS(K)`, and a 4-photon photoelectric peak sensor `0.30 · TopK(4)(|K|)` plus a `0.15` unrouted structural floor so rare tool-schema and date-arithmetic orbitals are never severed.
3. **Law 7 & Law 10 — Per-Expert Routed Dipole RMS Norm Restoration + Pass-2 Co-Routed BT-GS:** After Pass-1 ridge-regularized current absorption (`W_down = Y_t K_S^T (K_S K_S^T + λI)^{-1}`), restores each expert's routed output dipole energy via `(RMS(Y_tea) / RMS(Y_stu)).clamp(0.95, 1.20)` and runs an intra-layer Pass-2 **Co-Routed Block-Target Gauss-Seidel (BT-GS)** refinement across the 4 simultaneously active experts per token. Deep Layer 40 held-out norm ratio improves from `0.9365 -> 0.9694` and cosine similarity improves from `0.7703 -> 0.8043`.
4. **Tail-Preserving `1,152`-Token Agentic & Reasoning Calibration (`±4` Transition Upweighting):** Preserves complete multi-turn system + tool-schema + `<think>...</think>` + `<|tool_call_start|>[...]<|tool_call_end|>` tails (`seq_len = 1,152`), downweights repeated system-prompt boilerplate (`0.25x`), and upweights (`2.5x–4.0x`) the `±4`-token decision windows around `</think>` (`124801`), `<|tool_call_start|>` (`124905`), `<|tool_call_end|>` (`124906`), and `<|im_end|>` (`7`). Matches the Teacher's tool-call transition probability at `</think>` (`P(<|tool_call_start|> | </think>) = 0.8515` vs. Teacher `0.8569`).

---

## 1. Model Compression & Structural Fidelity Summary (`Teacher` vs. `v3` vs. `v4.1`)

| Metric | Teacher (`LFM2.5-24B-A2B-Q4_K_M`) | Student `Pocket-8.6B-v3-Q3K` | **Student `Pocket-8.6B-v4-Q3K` (`v4.1`)** |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | `24,159,516,928` (`24.16B`) | `8,645,714,432` (`8.65B`, `-64.2%`) | **`8,645,714,432` (`8.65B`, `-64.2%`)** |
| **Active Parameters per Token** | `2,338,728,192` (`2.34B`) | `1,476,530,176` (`1.48B`, `-36.9%`) | **`1,476,530,176` (`1.48B`, `-36.9%`)** |
| **MoE Experts (`total` / `top-k`)** | `64` / `4` (`M_exp = 1536`) | `64` / `4` (`M_exp = 512`) | **`64` / `4` (`M_exp = 512`, `3.0x` orbital folding)** |
| **GGUF File Size (Decimal / Binary)** | `14.526 GB` (`13.529 GiB`) | `3.996 GB` (`3.722 GiB`) | **`3.996 GB` (`3.722 GiB`, `3.64x` smaller)** |
| **Layer 40 Held-Out Cosine / Norm Ratio** | `1.0000` / `1.0000` | `0.7703` / `0.9365` | **`0.8043` / `0.9694`** |
| **Held-Out Val PPL Ratio vs. Teacher** | `1.00x` | `2.24x` (`71.10` vs. `31.68`) | **`1.35x` (`56.48` vs. `41.88`)** |
| **Held-Out Next-Token Top-1 / Top-5 Agreement** | `100.0%` / `100.0%` | `52.79%` / `78.99%` | **`69.64%` / `89.37%`** |
| **Tool-Call Gate `P(<|tool_call_start|> \| </think>)`** | `0.8569` | `—` | **`0.8515` (`85.71%` Top-1 match)** |
| **Babbo iOS Mobile Agent Eval (`11` Journeys)** | `10 / 11` (`90.9%`) | `0 / 10` (`0.0%`) | **`9 / 11` (`81.8%`, `9/10` single/2-turn)** |

---

## 2. Cross-Device Hardware Fleet Matrix (`llama-bench` PP512 / TG128, `-fa 1`, `-ngl 99`)

| Hardware Tier & Device | Memory Budget | Model | Status / Residency | Prefill `PP512` (tok/s) | Decode `TG128` (tok/s) | Peak Working RAM |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Tier 1: Super-Pocket (Mobile Edge)**<br>iPhone 16 Pro (A18 Pro, 8 GB Unified RAM) | `8 GB`<br>(Safe ≤ `4.0 GB`) | **LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher)** | `OOM_IMPOSSIBLE_14_5GB_ON_8GB_PHONE` | `—` | `—` | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`3.996 GB`)** | **`RUNS_OFFLINE_100PCT_METAL`** | **`444.14`** | **`43.88`** | **`187.00 MB`** |
| **Tier 2: Compact Edge Desktop**<br>Mac mini (Apple M1, 8 GB Unified RAM) | `8 GB`<br>(Safe ≤ `5.2 GB`) | **LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher)** | `OOM_EXCEEDS_8GB_UNIFIED_RAM` | `—` | `—` | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`3.996 GB`)** | **`RUNS_100PCT_METAL`** | **`476.80`** | **`36.87`** | `—` |
| **Tier 3: Prosumer Desktop GPU**<br>Desktop PC (NVIDIA GeForce RTX 4080, 16 GB VRAM) | `16 GB`<br>(Safe ≤ `13.5 GB`) | **LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher)** | `PARTIAL_CPU_MOE_OFFLOAD (ncmoe=20)` | `633.47` | `37.50` | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`3.996 GB`)** | **`OK_100PCT_VRAM` (`20.6x` PP / `7.6x` TG speedup)** | **`13,020.80`** | **`284.58`** | `—` |
| **Tier 4: Ultrathin Laptop (24 GB Unified RAM)**<br>MacBook Air M5 (Apple M5, 24 GB Unified RAM) | `24 GB`<br>(Safe ≤ `18.0 GB`) | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`3.996 GB`)** | `100% GPU/Metal Resident (-ngl 99 -ncmoe 0)` | `—` | `—` | `—` |
| **Tier 5: Workstation Hub (128 GB Unified)**<br>NVIDIA GX10 (GB10 Grace-Blackwell, 128 GB Unified) | `121 GB`<br>(Safe ≤ `105.0 GB`) | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`3.996 GB`)** | `100% GPU/Metal Resident (-ngl 99 -ncmoe 0)` | `—` | `—` | `—` |

---

## 3. Official Public Benchmark Comparison (Apples-to-Apples Greedy `T=0.0`)

| Model | `google/IFEval` (Strict Prompt / Inst %) | `GSM8K` (%) | `ARC-Challenge` (%) | `BFCL-AST` (Tool Call %) | `HumanEval` (`Pass@1` %) | `MATH-500` (%) | `</think>` Closure (%) | Loop-Free (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher — Published Card)* | *`96.75%`* | *`94.5%`* | *`93.4%`* | *`65.01%`* | *`86.0%`* | *`96.04%`* | *Ref* | *Ref* |
| **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K (`v4.1`, `3.996 GB`)** | **`50.0%` / `61.3%`** | **`65.00%`** | **`60.00%`** | **`60.00%`** *(100% think closed)* | **`18.75%`** | `10.00%` | **`71.70%`** | **`68.87%`** |
| **UniverseNet-LFM2.5-24B-Pocket-8.6B-v3-Q3K (`v3`, `3.996 GB`)** | `35.0%` / `41.9%` | `35.00%` | `20.00%` | `40.00%` | `0.00%` | `10.00%` *(CoT: 15.0%)* | `58.00%` | `56.50%` |

---

## 4. Babbo iOS Mobile Agent Journey Evaluation (`v1-journey-eval`, `prompt-variant=v2`, `seed=42`)

Evaluated end-to-end on the 11-scenario **Babbo iOS Family Assistant** harness (`create_reminder`, `share_item`, `remember_fact`, `forget_fact`, multi-turn clarification, ISO-8601 timezone resolution, and cross-member attribution):

| Scenario ID | Capability Tested | Teacher `LFM2.5-24B-A2B` (`14.53 GB`) | Student `Pocket-8.6B-v3` (`3.996 GB`) | Student `Pocket-8.6B-v4.0` (`3.996 GB`) | **Student `Pocket-8.6B-v4.1` (`3.996 GB`)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `greeting` | Conversational reply (no tool call) | **PASS** | FAIL (54.8s loop) | **PASS** (0.71s) | **PASS** (0.66s) |
| `remember-plumber` | `remember_fact(fact, shared)` | **PASS** | FAIL | **PASS** (1.89s) | **PASS** (0.87s) |
| `self-remind-parking` | `create_reminder` (+1d ISO-8601 `+02:00`) | **PASS** | FAIL | FAIL (`Z` offset) | **PASS** (2.23s) |
| `missing-time-clarify` | Ask `'What time?'` then `create_reminder` | **PASS** | FAIL | FAIL | FAIL (2.20s) |
| `relative-weekday-monday` | Date arithmetic (Next Monday ISO-8601) | **PASS** | FAIL | FAIL (wrong day) | **PASS** (1.05s) |
| `remind-member-dentist` | Cross-member `forMember='Monica'`, `sourceNote` | **PASS** | FAIL | FAIL (schema) | **PASS** (1.24s) |
| `unknown-person-dad` | Reject non-roster person (`Dad`) without tool | **PASS** | FAIL | **PASS** (3.06s) | **PASS** (2.94s) |
| `share-plumber-monica` | `share_item(itemKind='memory', itemId='mem-plumber-001')` | **PASS** | FAIL | FAIL (`factId` key) | **PASS** (1.55s) |
| `recall-plumber` | Grounded memory retrieval (`020-555-0199`, `€85/h`) | **PASS** | FAIL | **PASS** (0.65s) | **PASS** (0.53s) |
| `forget-plumber` | `forget_fact(factIdOrText='mem-plumber-001')` | **PASS** | FAIL | **PASS** (0.48s) | **PASS** (0.58s) |
| `long-thread-compaction` | 6-turn noisy thread compaction + reminder | FAIL (`title_wrong`) | — | FAIL | FAIL (9.05s) |
| **Total Score** | **11 Official iOS Agent Journeys** | **`10/11` (`90.9%`)** | **`0/10` (`0.0%`)** | **`5/11` (`45.5%`)** | **`9/11` (`81.8%`, `9/10` single/2-turn)** |

---

## Why Conventional Human-Inspired Neural Networks Are Structurally Impaired vs. Universe Network

![Human-Inspired Neural Networks vs. Universe Network Architecture](assets/human_neural_vs_universe_network.png)

### 1. The Biological Synapse Fallacy (Why Standard Neural Networks Are Impaired)
Standard Artificial Neural Networks (ANNs) and Transformers are modeled after a 1940s–1950s caricature of the biological brain: scalar leaky-integrator "neurons" connected by dense all-to-all "synapses." Because this biological metaphor lacks physical conservation laws, conventional neural networks suffer from **four structural impairments**:
1. **Unbounded All-to-All Synaptic Routing (`O(N^2)` Softmax & Hub Collapse):** Row-normalized `softmax(QK^T)` forces every token—including inert punctuation and background distractors—to inject `1.000` (`100%`) of its attention mass into the residual stream, while high-norm outlier tokens hijack global routing and become **Attention Sinks / Hubs** (`82.1x` column overload at long context).
2. **Bosonic Rank Collapse & Collinear "Clone Neurons" (No Pauli Exclusion):** Conventional neurons obey no exclusion principle. Hundreds of FFN orbitals learn collinear, redundant copies of the same bulk directions, and unnormalized deep networks collapse into a 1D ray by Layer 32 (`Effective Rank: 29.56 -> 1.18`).
3. **Blindness to Rare Few-Photon Factual & Symbolic Spikes:** Standard network compression (magnitude or mean-activation pruning) treats neurons like bulk thermal resistors, accidentally severing the rare **few-photon photoelectric orbitals** that fire intensely on only `2–4` tokens to store tool schemas (`<|tool_call_start|>`, `dueAtIso`), specific facts (`"Australia -> Canberra"`), or exact arithmetic.
4. **Vacuum Expectation & Dipole Norm Collapse Under Pruning:** Severing asymmetric `SwiGLU` / `SiLU` neurons destroys their non-zero vacuum DC offset and attenuates the routed MoE output dipole norm by `~22%` per layer, shifting deep hidden states off-manifold and breaking `<think>...</think>` closure.

### 2. The Universe Network Paradigm (Quantum-Atomic & Gauge-Field Intelligence)
**Universe Network** replaces biological guesswork with the exact laws of **Quantum Electrodynamics (QED), Fermionic Atomic Shells, and Renormalization-Group (RG) Statistical Mechanics**:

![Universe Network Orbital Physics & Mobile Benchmarks](assets/pocket_transmutation_benchmarks.png)

| Architectural Dimension | Conventional Human-Inspired Neural Network | **Universe Network (Quantum-Atomic Architecture)** | Concrete Practical Advantage |
| :--- | :--- | :--- | :--- |
| **Token Interaction & Routing** | Dense all-to-all `O(N^2)` Softmax (`sum_j p_ij = 1`; no slack state) | **Law 2 & Law 6: Split Gauss (`omega=0`) + Maxwell (`omega>0`) Boson Field (`O(N log N)`)** + **Noble-Gas Slack Orbital** | Inert tokens route `97.8%` to slack; clamps hubness `82.1x -> 0.58`; **`33.3x` faster field scaling** |
| **Feature Diversity & Depth Stability** | Unconstrained scalar neurons (collinear clone orbitals & rank collapse) | **Law 3: Fermionic Pauli Exclusion** (`1 - 0.35 max_{j<i} ReLU(cos_act) ReLU(cos_down)`) | Eliminates clone orbitals and preserves **`100%` effective rank** across 32+ layers |
| **Factual & Agentic Schema Memory** | Bulk average / weight-norm scoring (deletes rare factual & tool-call spikes) | **Law 1: Coupled Photoelectric Sensor (`TopK(4)`)** + RMSNorm Receptor Dipole | Retains rare tool-schema & reasoning orbitals (`9/11` Babbo iOS Agent, `60%` BFCL-AST) |
| **Dipole Energy & Current Folding** | Uncompensated pruning (attenuates MoE norm & shifts residual DC mean) | **Law 7 & Law 10: Routed Dipole RMS Norm Restoration + Pass-2 Co-Routed BT-GS** | Preserves Layer 40 norm ratio (`0.9694`) and `85.15%` `<|tool_call_start|>` probability after `</think>` |
| **Edge / Mobile Deployment** | `14.5 GB – 15.6 GB` footprint (crashes 8 GB mobile RAM) | **`2.94 GB` (`Qwen3-8B-Pocket`) & `3.996 GB` (`LFM2.5-24B-Pocket`) Native GGUF** | Runs **100% offline in Airplane Mode** on 8 GB iPhone at **`444.14 tok/s` prefill & `43.88 tok/s` decode** |

---

## Companion Universe Network Mobile Checkpoint

- **Qwen3-8B Pocket Agent (`8.19B -> 5.51B`, `2.94 GB` GGUF, `Val PPL 14.525` beating full `Qwen3-4B` Teacher):**
  [wasnaga/UniverseNet-Qwen3-8B-Pocket-GGUF](https://huggingface.co/wasnaga/UniverseNet-Qwen3-8B-Pocket-GGUF)

---

## How to Run on iPhone 16 Pro (Airplane Mode) or Desktop `llama.cpp`

### iPhone 16 Pro / iPad / Apple Silicon (PocketPal AI)
1. Install **PocketPal AI** (or **LLM Farm** / **Enclave**) from the iOS App Store.
2. Open **Models -> Add from Hugging Face** and search for:
   `wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF`
3. Download **`UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K.gguf`** (`3.996 GB` decimal / `3.722 GiB`).
4. Load the model (fits comfortably inside iPhone 16 Pro's `8 GB` unified memory while only activating `1.48B` parameters per token for `43.88 tok/s` generation) and run offline in **Airplane Mode**!

### Desktop `llama.cpp` CLI / Server
```bash
# Single-prompt completion
llama-completion -m UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K.gguf -ngl 99 -c 4096 -n 700 --temp 0 --repeat-penalty 1.05 -r "<|im_end|>" -p "<|im_start|>user
What is the capital of Australia? One word.<|im_end|>
<|im_start|>assistant
<think>"

# OpenAI-compatible local agent server
llama-server -m UniverseNet-LFM2.5-24B-Pocket-8.6B-v4-Q3K.gguf -ngl 99 -fa on -c 4096 --port 8080
```
