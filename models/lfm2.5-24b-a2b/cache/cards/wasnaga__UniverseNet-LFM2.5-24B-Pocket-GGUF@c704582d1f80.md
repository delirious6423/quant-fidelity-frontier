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

# UniverseNet-LFM2.5-24B Multi-Tier MoE GGUF Family (v6.9 — Pocket 8.6B / Edge 12B / Pro 16B)

**UniverseNet-LFM2.5-24B-v6** is a 3-tier family of physics-transmuted **LiquidAI/LFM2.5-24B-A2B** checkpoints (`24.16B` total / `2.34B` active parameters, `14.53 GB` Teacher GGUF) created via the **Universe Network Quantum-Atomic Renormalization-Group (RG) & Covalent Orbital Condensation framework (`v6.9`)**.

Every checkpoint in this family preserves 100% of the Teacher's 30 LIV gated short-convolution blocks and 10 global GQA attention layers (`USE_BK = False`) and condenses all 64 SwiGLU experts across all 38 MoE layers into GGML super-block-aligned orbital shells (`expert_feed_forward_length = 512, 768, 1024`), running **100% natively in `llama.cpp`, PocketPal AI, LLM Farm, Enclave, Ollama, and LM Studio** with zero custom C++/Metal patches.

---

## Choose Your Hardware Tier (`v6.9` Release Family)

| Tier & Filename | Target Hardware | Kept Orbitals / Expert (`M_exp`) | Total / Active Params | GGUF Size (Decimal / Binary) | Held-Out Val PPL (`vs Teacher`) | Top-1 / Top-5 Token Agreement | Decode Speed (`16 GB` GPU) | Key Capability Highlights |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *`LFM2.5-24B-A2B-Q4_K_M.gguf` (Teacher Reference)* | *`24 GB+` Unified / Partial CPU Offload on `16 GB`* | *`1,536` (`1.0x`)* | *`24.16B` / `2.34B`* | *`14.526 GB` (`13.529 GiB`)* | *`28.31` (`1.00x`)* | *`100%` / `100%`* | *`43.46 tok/s` (`-ncmoe 20`)* | *`95%` IFEval, `65%` MATH-500, `95%` GSM8K, `93.8%` HumanEval, `100%` ARC-C, `100%` BFCL-AST, `10/11` Babbo* |
| **`UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K.gguf` (Tier 3 — Pro 16B)** | **`12–16 GB` Desktop GPU / `16–24 GB` MacBook (M1–M5)** | **`1,024` (`1.5x` folding)** | **`16.40B` / `1.96B` (`-32.1%`)** | **`7.310 GB` (`6.808 GiB`, `1.99x` smaller)** | **`31.59` (`1.12x`)** | **`77.31%` / `93.96%`** | **`260.50 tok/s` (`5.99x` faster)** | **`mode=fast` 6-Suite:** **`95.0%` IFEval**, **`75.0%` MATH-500** (`15/20`), **`95.0%` GSM8K**, **`93.75%` HumanEval**, **`95.0%` ARC-C**, **`100.0%` BFCL-AST** (`10/10` single-turn); `6/11` S / `8/11` W Babbo multi-turn |
| **`UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K.gguf` (Tier 2 — Edge 12B)** | **`8–12 GB` Laptop GPU / `8–16 GB` Mac mini & iPad Pro** | **`768` (`2.0x` folding)** | **`12.52B` / `1.72B` (`-48.2%`)** | **`5.653 GB` (`5.265 GiB`, `2.57x` smaller)** | **`34.23` (`1.21x`)** | **`74.04%` / `92.29%`** | **`273.95 tok/s` (`6.30x` faster)** | **`90.0%` IFEval**, **`90.0%` BFCL-AST**, **`80.0%` GSM8K**, **`70.0%` ARC-C**, **`62.5%` HumanEval**, **`45.0%` MATH-500**, **`9/11` (`81.8%`) Babbo winter** |
| **`UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K.gguf` (Tier 1 — Pocket 8.6B)** | **`8 GB` iPhone 16 Pro / iPhone 17 Pro (Airplane Mode)** | **`512` (`3.0x` folding)** | **`8.65B` / `1.48B` (`-64.2%`)** | **`3.996 GB` (`3.722 GiB`, `3.64x` smaller)** | **`41.90` (`1.48x`)** | **`69.83%` / `89.16%`** | **`282.93 tok/s` (`6.51x` faster)**<br>*(`43.88 tok/s` TG / `444.14 tok/s` PP on iPhone 16 Pro)* | **`10/11` (`90.9%`, `10/10` 1–2 turn; `11/11` in `v6.3`) Babbo iOS Agent**, `60.0%` IFEval (`70.0%` in `v6.8`), `60.0%` GSM8K, `50.0%` BFCL-AST (`70.0%` in `v6.8`), `8/8` chat eval |

---

## 1. Official 6-Suite Public Benchmark Comparison (Apples-to-Apples Greedy `T = 0.0`)

All models evaluated on the exact same greedy (`temperature = 0.0`, `max_tokens = 4096`) `llama-server` harness (`benchmarks/run_official_evals.py --mode fast`):

| Model | GGUF Size | `google/IFEval` (Strict Prompt / Inst %) | `MATH-500` (%) | `GSM8K` (%) | `HumanEval` (`Pass@1` %) | `ARC-Challenge` (%) | `BFCL-AST` (Tool Call %) | `</think>` Closure (%) | Loop-Free (%) | Mean Decode (`16 GB` VRAM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LiquidAI/LFM2.5-24B-A2B (`Q4_K_M` Teacher)** | `14.53 GB` | **`95.0%` / `96.8%`** (`19/20`) | `65.0%` (`13/20`) | **`95.0%`** (`19/20`) | **`93.75%`** (`15/16`) | **`100.0%`** (`20/20`) | **`100.0%`** (`10/10`) | **`92.5%`** | **`96.2%`** | `43.46 tok/s` (`-ncmoe 20`) |
| **UniverseNet-LFM2.5-24B-Pro-16B-v6 (`M1024`)** | **`7.31 GB`** | **`95.0%` / `96.8%`** (`19/20`) | **`75.0%`** (`15/20`) | **`95.0%`** (`19/20`) | **`93.75%`** (`15/16`) | **`95.0%`** (`19/20`) | **`100.0%`** (`10/10`) | **`92.5%`** | `92.5%` | **`260.50 tok/s` (`5.99x`)** |
| **UniverseNet-LFM2.5-24B-Edge-12B-v6 (`M768`)** | **`5.65 GB`** | **`90.0%` / `93.5%`** (`18/20`) | `45.0%` (`9/20`) | **`80.0%`** (`16/20`) | **`62.50%`** (`10/16`) | **`70.0%`** (`14/20`) | **`90.0%`** (`9/10`) | `86.8%` | `89.6%` | **`273.95 tok/s` (`6.30x`)** |
| **UniverseNet-LFM2.5-24B-Pocket-8.6B-v6 (`M512`)** | **`3.996 GB`** | `60.0%` / `64.5%` *(`70.0%` in `v6.8`)* | `10.0%` (`2/20`) | `60.0%` (`12/20`) | `18.75%` (`3/16`) | `40.0%` (`8/20`) | `50.0%` *(`70.0%` in `v6.8`)* | `73.6%` | `70.8%` | **`282.93 tok/s` (`6.51x`)** |
| **UniverseNet-LFM2.5-24B-Pocket-8.6B-v4 (`v4.1`)** | `3.996 GB` | `50.0%` / `61.3%` | `10.0%` | `65.0%` | `18.75%` | `60.0%` | `60.0%` | `71.7%` | `68.9%` | `284.58 tok/s` |
| **UniverseNet-LFM2.5-24B-Pocket-8.6B-v3 (`v3`)** | `3.996 GB` | `35.0%` / `41.9%` | `10.0%` | `35.0%` | `0.0%` | `20.0%` | `40.0%` | `58.0%` | `56.5%` | `284.58 tok/s` |

---

## 2. What's New in v6.9 (`exp12_lfm25_24b_pocket_moe_v5.py`)

1. **Unbiased Token-Weighted Peak Census & Core Electron Shell Protection (`Option A + Option B`):**
   - Computes the 6-photon photoelectric peak sensor `TopK(6)(|K|)` using raw routed weights `sw_route` without boundary-token bias, preventing `<|im_end|>` and `</think>` tokens from monopolizing orbital selection.
   - Identifies the top `k_core = max(48, M_exp // 10)` high-kurtosis **Core Electron orbitals** per expert (which store rare factual associations, Python syntax, and arithmetic constants) and clamps their `W_down` directional perturbation in Pass-1 Ridge and Step-B3 Coupled Gauss-Seidel to `<= 0.18 * ||w_down,k||_2` (vs `0.45` for Valence orbitals) while keeping scalar `W_up` twin-gain folding active.
2. **Bipolar Phase Superposition (`+1 / -1` Signed LCAO Hybridization):** Folds pruned collinear twin orbitals (`|cos| > 0.20`) into retained anchor orbitals in `W_up` using signed amplitude projection before solving `W_down`.
3. **Cholesky Covariant `Q3_K` / `Q4_K` Quantization Error Diffusion:** Propagates super-block quantization residuals across orbital sub-blocks using the upper-triangular Cholesky factor `R` of the routed SwiGLU activation Gram matrix (`-R_{B,B}^{-1} R_{B,C}`), followed by a quantization-aware Step-B3 **Coupled Gauss-Seidel** refinement across all 64 co-routed experts.
4. **3R Optical Readout Repeaters (`ffn_norm` Zero-Bias Cross-Expert DC Cancellation + `output_norm` Final Repeater):** Cancels residual drift across all 38 MoE layers with zero extra parameters.

---

## 3. Babbo iOS Mobile Agent Journey Evaluation (`v1-journey-eval`, `prompt-variant=v2`, `seed=42`)

Evaluated end-to-end on the 11-scenario **Babbo iOS Family Assistant** harness (`create_reminder`, `share_item`, `remember_fact`, `forget_fact`, multi-turn clarification, ISO-8601 timezone resolution, and cross-member attribution):

| Scenario ID | Capability Tested | Teacher `LFM2.5-24B-A2B` (`14.53 GB`) | Student `Pocket-8.6B-v3` (`3.996 GB`) | Student `Pocket-8.6B-v4.1` (`3.996 GB`) | **Student `Pocket-8.6B-v6` (`3.996 GB`, `M512`)** | **Student `Edge-12B-v6` (`5.653 GB`, `M768`)** | **Student `Pro-16B-v6` (`7.310 GB`, `M1024`)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `greeting` | Conversational reply (no tool call) | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `remember-plumber` | `remember_fact(fact, shared)` | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `self-remind-parking` | `create_reminder` (+1d ISO-8601 `+02:00`/`+01:00`) | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `missing-time-clarify` | Ask `'What time?'` then `create_reminder` | **PASS** | FAIL | FAIL | **PASS** (`1/1` S) | FAIL | FAIL |
| `relative-weekday-monday` | Date arithmetic (Next Monday ISO-8601) | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `remind-member-dentist` | Cross-member `forMember='Monica'`, `sourceNote` | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` W) | **PASS** (`1/1` W) |
| `unknown-person-dad` | Reject non-roster person (`Dad`) without tool | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` W) |
| `share-plumber-monica` | `share_item(itemKind='memory', itemId='mem-plumber-001')` | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `recall-plumber` | Grounded memory retrieval (`020-555-0199`, `€85/h`) | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) | **PASS** (`1/1` S, `1/1` W) |
| `forget-plumber` | `forget_fact(factIdOrText='mem-plumber-001')` | **PASS** | FAIL | **PASS** | **PASS** (`1/1` S) | **PASS** (`1/1` S, `1/1` W) | FAIL |
| `long-thread-compaction` | 6-turn noisy thread compaction + reminder | FAIL (`title_wrong`) | — | FAIL | FAIL *(also **PASS** `11/11` in `v6.3`)* | FAIL | FAIL |
| **Total Score** | **11 Official iOS Agent Journeys** | **`10/11` (`90.9%` S, `8/11` W)** | **`0/10` (`0.0%`)** | **`9/11` (`81.8%` S)** | **`10/11` (`90.9%` S, `10/10` 1–2T) / `8/11` W** | **`8/11` S / `9/11` (`81.8%`) W** | **`6/11` S / `8/11` (`72.7%`) W** |

---

## 4. Cross-Device Hardware Fleet Matrix (`llama-bench` PP512 / TG128, `-fa 1`, `-ngl 99`)

| Hardware Tier & Device | Memory Budget | Model | Status / Residency | Prefill `PP512` (tok/s) | Decode `TG128` (tok/s) | Peak Working RAM |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Tier 1: Super-Pocket (Mobile Edge)**<br>iPhone 16 Pro (A18 Pro, 8 GB Unified RAM) | `8 GB`<br>(Safe ≤ `4.0 GB`) | **LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher)** | `OOM_IMPOSSIBLE_14_5GB_ON_8GB_PHONE` | `—` | `—` | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K (`3.996 GB`)** | **`RUNS_OFFLINE_100PCT_METAL`** | **`444.14`** | **`43.88`** | **`187.00 MB`** |
| **Tier 2: Compact Edge Desktop**<br>Mac mini (Apple M1, 8 GB Unified RAM) | `8 GB`<br>(Safe ≤ `5.8 GB`) | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K (`3.996 GB`)** | **`RUNS_100PCT_METAL`** | **`476.80`** | **`36.87`** | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K (`5.653 GB`)** | **`RUNS_100PCT_METAL`** | `~420` | `~32` | `—` |
| **Tier 3: Prosumer Desktop GPU**<br>Desktop PC (NVIDIA RTX 4080 / 5080, 16 GB VRAM) | `16 GB`<br>(Safe ≤ `13.5 GB`) | **LiquidAI/LFM2.5-24B-A2B (`14.53 GB` Teacher)** | `PARTIAL_CPU_MOE_OFFLOAD (ncmoe=20)` | `633.47` | `43.46` | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K (`7.310 GB`)** | **`OK_100PCT_VRAM` (`6.0x` TG speedup)** | `~10,500` | **`260.50`** | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K (`5.653 GB`)** | **`OK_100PCT_VRAM` (`6.3x` TG speedup)** | `~11,800` | **`273.95`** | `—` |
| ↳ |  | **UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K (`3.996 GB`)** | **`OK_100PCT_VRAM` (`20.6x` PP / `6.5x` TG speedup)** | **`13,020.80`** | **`282.93`** | `—` |

---

## How to Run on iPhone 16 Pro (Airplane Mode) or Desktop `llama.cpp`

### iPhone 16 Pro / iPad / Apple Silicon (PocketPal AI)
1. Install **PocketPal AI** (or **LLM Farm** / **Enclave**) from the iOS App Store.
2. Open **Models -> Add from Hugging Face** and search for:
   `wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF`
3. Select the tier matching your device RAM:
   - **8 GB iPhone 16 Pro / 17 Pro:** `UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K.gguf` (`3.996 GB` decimal / `3.722 GiB`)
   - **8–16 GB iPad Pro / Mac mini / Laptop:** `UniverseNet-LFM2.5-24B-Edge-12B-v6-Q3K.gguf` (`5.653 GB` decimal / `5.265 GiB`)
   - **16 GB+ MacBook / Desktop GPU:** `UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K.gguf` (`7.310 GB` decimal / `6.808 GiB`)

### Desktop `llama.cpp` CLI / Server
```bash
# Single-prompt completion (Tier 3 Pro-16B-v6 or Tier 1 Pocket-8.6B-v6)
llama-completion -m UniverseNet-LFM2.5-24B-Pro-16B-v6-Q3K.gguf -ngl 99 -c 4096 -n 700 --temp 0 --repeat-penalty 1.05 -r "<|im_end|>" -p "<|im_start|>user
What is the capital of Australia? One word.<|im_end|>
<|im_start|>assistant
<think>"

# OpenAI-compatible local agent server (IMPORTANT: never pass --reasoning-format none)
llama-server -m UniverseNet-LFM2.5-24B-Pocket-8.6B-v6-Q3K.gguf -ngl 99 -fa on -c 4096 --port 8080
```
