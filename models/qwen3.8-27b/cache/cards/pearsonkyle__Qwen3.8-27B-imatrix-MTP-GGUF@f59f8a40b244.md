---
library_name: gguf
base_model:
- Qwen/Qwen3.8-27B
tags:
- gguf
- llama.cpp
- text-generation
- text-generation-inference
- transformers
- quantization
- quantized
- imatrix
- importance-matrix
- mtp
- multi-token-prediction
- speculative-decoding
- low-bit
- 2-bit
- 3-bit
- 4-bit
- 5-bit
- iq2_m
- iq3_m
- iq4_xs
- q5_k_m
- qwen
- qwen3
- 27b
- coder
- tool-use
- function-calling
- agentic
- swe-bench
- long-context
- 32k-calibration
- hybrid-attention
- linear-attention
- multimodal
- vision
- image-text-to-text
- vlm
- mmproj
license: apache-2.0
language:
- en
pipeline_tag: image-text-to-text
---

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; border: 1px solid #99f6e4; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">
  <div style="background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 12px; color: white; border: none;">🧊 Qwen/Qwen3.8-27B · 32K imatrix · MTP · GGUF</h1>
      <span style="background: #f59e0b; color: #1c1917; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">imatrix (hybrid) @ 32K</span>
    </div>
  </div>
  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bfdbfe;">📦 9.7 · 12.1 · 14.5 · 18.3 GiB</span>
    <span style="background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a7f3d0;">IQ2_M · IQ3_M · IQ4_XS · Q5_K_M</span>
    <span style="background: #fef08a; color: #713f12; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde047;">🧬 calibrated at <b>ctx 32,768</b></span>
    <span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #ddd6fe;">⚡ MTP head bundled @ Q8_0</span>
    <span style="background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fde68a;">🏅 all 4 rungs solved the SWE issue</span>
    <span style="background: #f3e8ff; color: #6b21a8; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #e9d5ff;">👁️ Text + Image · mmproj 629 MB</span>
    <span style="background: #fce7f3; color: #9d174d; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #fbcfe8;">🏗️ llama.cpp f3e1828</span>
  </div>
  <div style="padding: 24px; display: flex; flex-direction: column; gap: 20px;">
    <div style="background: #f0fdfa; border-left: 5px solid #0d9488; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #115e59; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧊</span> What this is</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">imatrix-calibrated quantizations of <a href="https://huggingface.co/Qwen/Qwen3.8-27B"><b>Qwen/Qwen3.8-27B</b></a> spanning <b>3.06 – 5.76 bits per weight</b>, calibrated on real agentic-coding logs. Every rung <b>bundles the model's own Multi-Token-Prediction draft head at Q8_0</b> (<code>blk.64</code>) — speculative decoding with no second file to download. <b>IQ4_XS is the recommended default</b> (KLD 0.010, 88% top-token agreement); <b>Q5_K_M</b> is the closest substitute for FP16. Plain GGUF — runs in vanilla <code>llama.cpp</code> / Ollama / LM Studio, no custom runtime.</p>
    </div>
    <div style="background: #faf5ff; border-left: 5px solid #7c3aed; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #5b21b6; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>👁️</span> Vision included (text + image)</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">Qwen3.8 is natively multimodal, so the vision tower ships alongside as <b><code>mmproj-Qwen3.8-27B-Q8_0.gguf</code></b> (629 MB, 27-layer encoder at Q8_0). Pair it with <b>any</b> of the four rungs via <code>--mmproj</code> and the model can <b>see images</b> — screenshots, diagrams, UI states. The text weights are unchanged; vision is purely additive, so drop the flag and you are back to the identical text-only model. Verified <b>on the 2-bit rung</b> — the hardest case: shown a test image it reported <i>"a red circle on the left, a blue rectangle on the right, and a green triangle at the bottom center"</i>, correct in shape, colour <b>and</b> position for all three.</p>
    </div>
    <div style="background: #fffbeb; border-left: 5px solid #f59e0b; padding: 16px; border-radius: 0 8px 8px 0;">
      <h3 style="margin: 0 0 8px 0; font-size: 15px; color: #92400e; font-weight: 700; display: flex; align-items: center; gap: 6px;"><span>🧬</span> Why 32K calibration matters</h3>
      <p style="margin: 0; font-size: 13px; color: #334155; line-height: 1.7;">An imatrix is <b>activation statistics</b> — it records what the model actually does while reading the calibration text. Pack that corpus into 7,500-token windows and the model never sees a long tool-call chain <i>while</i> the statistics are collected, so the channels carrying late-trajectory behaviour are under-weighted. Repacking for 32,768 took calibration windows from an 11,390-token maximum to a <b>median of 13,004 and a p90 of 31,869</b> — <b>67% of sessions now exceed the old cap entirely</b>, so a whole agentic trajectory fits in one calibration context instead of being cut mid-chain.</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">📉 2.8–5.2× smaller</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">9.74–18.33 GiB on disk (incl. the bundled Q8_0 MTP head) vs 50.90 GiB FP16.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🧠 Hybrid-attention aware</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">48 of 64 layers are linear attention. Their SSM tensors pass through with raw E[a²] — output-aware re-ranking is invalid for them.</span></div>
      <div style="border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; background: #fafafa; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);"><span style="font-weight: 700; color: #115e59; font-size: 12px; display: block; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">🛠️ Standard GGUF</span><span style="font-size: 13px; color: #4b5563; line-height: 1.5;">Loads anywhere llama.cpp runs. No patches, kernels, or forks.</span></div>
    </div>
  </div>
</div>

## 📊 Unified benchmark & quality table

- **📐 Static** — measured against FP16 on the **external** holdout (code + math + tools, 90k tokens, disjoint from every calibration source) at `eval-ctx 8192`. KLD is **median**; `top_p` is top-token agreement with FP16.
- **🤖 Agentic** — 25 held-out tool-use sessions (174 scored turns), plus one [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) issue solved end-to-end through the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

| Metric | FP16 *(ref)* | Q5_K_M | **IQ4_XS** | IQ3_M | IQ2_M |
|:---|---:|---:|---:|---:|---:|
| File | — | [Q5_K_M.gguf](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF/resolve/main/Qwen3.8-27B-Q5_K_M.gguf) | [IQ4_XS.gguf](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF/resolve/main/Qwen3.8-27B-IQ4_XS.gguf) | [IQ3_M.gguf](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF/resolve/main/Qwen3.8-27B-IQ3_M.gguf) | [IQ2_M.gguf](https://huggingface.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF/resolve/main/Qwen3.8-27B-IQ2_M.gguf) |
| Method | — | imatrix | imatrix | imatrix | imatrix |
| Quality | — | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| BPW | 16.000 | 5.763 | **4.549** | 3.816 | 3.062 |
| Size (GiB) | 50.90 | 18.33 | **14.47** | 12.14 | 9.74 |
| 🤖 SWE issue resolved | ✅ | ✅ | ✅ | ✅ | ✅ |
| 🤖 Steps to solve | 10 | 13 | 14 | 26 | 20 |
| 🤖 Malformed commands | 0 | 0 | 0 | 0 | 0 |
| 🤖 Tool-selection acc | 0.494 | 0.474 | **0.503** | 0.454 | 0.494 |
| 🤖 Param accuracy | 0.256 | 0.265 | **0.274** | 0.239 | 0.260 |
| 📐 PPL | **22.850** | 23.274 | 31.118 | 37.363 | 56.540 |
| 📐 KLD (med) | 0.000 | 0.0036 | 0.0105 | 0.0354 | 0.1242 |
| 📐 same_top_p | 100.0% | 91.8% | 88.2% | 82.4% | 72.0% |

> **⚠️ The 🤖 rows are a band, not a ranking.** At n≈174 turns the standard error is **±3.8pp**,
> and the whole spread — FP16 included — is **0.454–0.503**. That is a 4.9pp range inside one
> standard error. It supports *"every rung, down to 3.06 bpw, tool-calls indistinguishably from
> FP16 here"* and nothing finer. IQ2_M and FP16 both score 86/174 yet **disagree on 24 turns**
> (12 each way) — the tie is a coincidence, not equivalence.
>
> **📐 KLD and `top_p` are the trustworthy columns**, strictly monotone across all six evals.
> **IQ2_M gives up ~28% of top-token decisions** — a memory-constrained option, not a drop-in.

<details>
<summary><b>📌 Sampling, methodology & all six eval distributions</b></summary>

**Sampling.** Static: `eval-ctx 8192`. Tool-call replay: **greedy (`temperature=0`)**, `ctx=32768`, `--no-stop-on-fail` so every model is scored on the **identical 174 turns** (the default stop-on-fail halts a weak model early and scores it on fewer, easier turns — which makes models incomparable). Agentic: `temperature=0.25, top_p=0.95, max_tokens=8096`, **2,048-token reasoning budget**, step cap 60. Run on an RTX PRO 6000 Blackwell.

**"Tool errors" are classified, not counted.** A non-zero exit is *not* an agent error — a `grep` that legitimately matched nothing and a `pytest` correctly reporting a failure both exit non-zero. Errors are split into `malformed` (shell syntax error, command not found, failed `cd` — the intended work never ran), `timeout`, and `nonzero` (the program ran and reported a result). Under that split every rung issues **zero malformed commands**; the only genuine tool failure in the whole sweep is one timeout on IQ3_M.

**The reasoning budget is a cost control, not a quality lever.** Uncapped, IQ2_M was observed emitting a **12,566-token** single completion — a repetition loop, not reasoning. An unbudgeted control sweep was run alongside: at n=1 the 2k budget shows no reliable effect on whether the issue gets solved (IQ4_XS and Q5_K_M improved, IQ3_M got worse, FP16 rose).

**Six eval distributions, each with its own FP16 baseline — never concatenated:**

| eval | what it is | FP16 PPL | IQ2_M | IQ3_M | IQ4_XS | Q5_K_M |
|---|---|---:|---:|---:|---:|---:|
| `external` | code + math + tools, disjoint from calibration — **the headline** | 22.850 | 0.1242 | 0.0354 | 0.0105 | 0.0036 |
| `general` | `combined_en_tiny`, broad English (30k tok) | 92.797 | 0.2199 | 0.0569 | 0.0154 | 0.0062 |
| `tools` | held-out CLI + agent log sessions (150k tok) | 21.652 | 0.0498 | 0.0115 | 0.0032 | 0.0014 |
| `agentic` | held-out SWE trajectories (99k tok) | 43.473 | 0.0169 | 0.0048 | 0.0013 | 0.0006 |
| `broad` | held-out broad-instruct (100k tok) | 88.193 | 0.2691 | 0.0702 | 0.0204 | 0.0076 |
| `cal8k` | slice of the *previous* 8192-packed corpus — a **fit** probe, not a holdout | 17.400 | 0.0840 | 0.0261 | 0.0062 | 0.0025 |

⚠️ The four chat-templated evals (`tools`, `agentic`, `broad`, `cal8k`) are **quant-vs-quant only**: `llama-perplexity` has no `--parse-special`, so chat markers tokenize as ordinary BPE and absolute PPL is off-distribution. KLD and `top_p` remain valid — which is why PPL goes non-monotone there while KLD does not.

</details>

---

## 📚 What it was calibrated on

An imatrix is only as good as the text it watched. Here is exactly what this one saw.

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0;">
  <div style="border: 1px solid #99f6e4; border-radius: 10px; background: #f0fdfa; padding: 14px; text-align: center;"><div style="font-size: 22px; font-weight: 800; color: #0f766e;">4,255,761</div><div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">tokens</div></div>
  <div style="border: 1px solid #99f6e4; border-radius: 10px; background: #f0fdfa; padding: 14px; text-align: center;"><div style="font-size: 22px; font-weight: 800; color: #0f766e;">3,436</div><div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">windows</div></div>
  <div style="border: 1px solid #99f6e4; border-radius: 10px; background: #f0fdfa; padding: 14px; text-align: center;"><div style="font-size: 22px; font-weight: 800; color: #0f766e;">6,786</div><div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">tool calls</div></div>
  <div style="border: 1px solid #99f6e4; border-radius: 10px; background: #f0fdfa; padding: 14px; text-align: center;"><div style="font-size: 22px; font-weight: 800; color: #0f766e;">32,768</div><div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">packing ctx</div></div>
  <div style="border: 1px solid #99f6e4; border-radius: 10px; background: #f0fdfa; padding: 14px; text-align: center;"><div style="font-size: 22px; font-weight: 800; color: #0f766e;">16.9 MB</div><div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">corpus on disk</div></div>
</div>

| source | what it is | samples | windows | tokens | share | median window |
|:---|:---|---:|---:|---:|---:|---:|
| **logs** | real Claude Code / opencode / qwen code sessions | 108 sessions | 125 | 2,000,139 | **47.0%** | 13,004 |
| **swe-trajectories** | agent runs on real GitHub issues | 61 sessions | 62 | 681,012 | 16.0% | 9,650 |
| **reasoning** | `<think>` turns, re-cut so reasoning lands last | 65 sessions | 230 | 638,046 | 15.0% | 2,022 |
| **broad-supplement** | broad-instruct, non-coding | 2,704 convos | 2,704 | 537,851 | 12.6% | 182 |
| **wiki** | `wiki.test.raw`, raw prose | 10 chunks | 10 | 297,053 | 7.0% | 29,641 |
| **redteam-refusals** | attack prompts + generic refusals | 305 convos | 305 | 101,660 | 2.4% | 231 |
| | | | **3,436** | **4,255,761** | **100%** | |

Three things worth reading off that table:

- **Nearly two-thirds is agentic** (logs + swe-trajectories = 63%), and those are the *long* windows — median 13,004 and 9,650 tokens against a 182-token median for broad-instruct. **84 of the 125 log windows exceed 7,500 tokens and 50 exceed 16,384**, which is only possible because packing is at ctx 32,768. Nothing was truncated: **0 sessions hit the 32,076-token cap.**
- **Tool-calling is dense, not decorative** — 6,786 `<tool_call>` blocks and 5,827 `<tool_response>` blocks across 76 distinct tool schemas, and in the log windows **99.3% of tokens sit in tool-bearing turns**.
- **`reasoning` overlaps `logs` by design.** It re-renders the same conversations with a reasoning turn placed last, because chat templates keep `<think>` only on a render's final assistant turn and scrub it from history. Without those extra windows the corpus would contain almost no reasoning at all.

The red-team slice ships **attack prompts with generic refusals substituted** — the targets' original harmful completions never enter any corpus. Refusal behaviour is what low-bit quantization erodes first, so the attack *distribution* belongs in calibration; the responses do not.

---

## 🔬 How they were made

- **Hybrid imatrix at ctx 32,768** — activation energy `E[a²]` blended with weight-column energy `‖W[:,c]‖²·E[a²]` per tensor, collected over the corpus above in **129 chunks of a full 32,768-token context** (3h54m on an RTX PRO 6000 Blackwell). 496 tensors received statistics.
- **SSM passthrough** — Qwen3.8 is hybrid (48 linear-attention + 16 full-attention layers). **144 SSM tensors pass through with raw `E[a²]`**; output-aware re-ranking is mathematically invalid for state-space tensors.
- **Bundled MTP draft head** — the trained MTP head (`mtp_num_hidden_layers=1`) is remapped to `blk.64` and pinned to **Q8_0** in every rung: **8 × Q8_0 + 7 × F32, zero low-bit tensors**, verified per file. This is checked rather than assumed because `llama-quantize` silently accepts a `--tensor-type` pattern that matches nothing — a stale pin would quantize the draft head with the trunk and surface only as poor acceptance. The head gets no imatrix statistics; it sits outside the forward pass, which is why it is pinned rather than calibrated.
- **The tool-call format trap** — Qwen3.8's chat template emits tool calls as **XML** (`<tool_call><function=NAME><parameter=KEY>`), **not** the JSON form earlier Qwen3.x templates used. Calibrating on the JSON form would have collected statistics on a syntax this model never produces.
- **`output.weight` is calibrated** — `llama-imatrix` only collects tensors named `blk.*` unless you pass `--process-output`, so the largest quantized tensor (and the one that most directly shapes the distribution KLD measures) is otherwise quantized blind. `llama-quantize` prints `did not find weights for output.weight` and proceeds anyway, so the only symptom is a line in a log.
- **Disjoint splits** — calibration (train) and every eval holdout are disjoint by construction, so the static table measures generalization, not fit. The one exception is `cal8k`, explicitly labelled a fit probe.
- Toolchain: [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner) with vendored [llama.cpp](https://github.com/ggml-org/llama.cpp) `@ f3e1828`. Calibration logs mined with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).

<details>
<summary><b>🔁 Reproduce</b></summary>

```bash
git clone https://github.com/pearsonkyle/Quant-Tuner && cd Quant-Tuner
git submodule update --init --recursive
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=ON
cmake --build vendor/llama.cpp/build -j 64
uv sync --extra dev

PYTHONPATH=src .venv/bin/python scripts/exp060_repack_cal_32k.py --ctx 32768 \
    --wiki out/exp-001/wiki/wiki.test.raw
PYTHONPATH=src .venv/bin/python scripts/exp060_quants_qwen38.py \
    --run exp-060-32k --ctx 32768 --eval-ctx 8192 \
    --evals external general tools agentic broad cal8k
```

> **⚠️ `llama-imatrix` needs `--no-ppl` above ~17k ctx on this model.** Its perplexity path
> computes `all_logits + first*n_vocab` with `first = n_ctx/2` in `int` arithmetic
> (`tools/imatrix/imatrix.cpp:911`). With Qwen3.8's 248,320-token vocab that product overflows
> `INT_MAX` for any `n_ctx > 2³²/248320 ≈ 17,296`, and the process **segfaults after the first
> pass**. `--no-ppl` skips only the perplexity bookkeeping — the forward pass, and therefore
> every activation statistic, is unchanged.

</details>

---

## 🚀 Usage

### Ollama

```bash
ollama run hf.co/pearsonkyle/Qwen3.8-27B-imatrix-MTP-GGUF:IQ4_XS
# also: :Q5_K_M · :IQ3_M · :IQ2_M
```

### llama.cpp server, with MTP speculative decoding

```bash
./llama-server \
    --model Qwen3.8-27B-IQ4_XS.gguf \
    --ctx-size 32768 --n-gpu-layers 999 \
    --spec-type draft-mtp --spec-draft-n-max 1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 0.0.0.0 --port 1234
```

The draft head is **inside the GGUF** — no `--model-draft`, nothing extra to fetch. Qwen3.8
exposes **one** nextn layer, so `--spec-draft-n-max 1` is the right setting. Drop both
`--spec-*` flags to run without speculative decoding.

### 🖼️ Vision (text + image)

The vision tower ships separately as `mmproj-Qwen3.8-27B-Q8_0.gguf` (629 MB) so you only
download it if you need images. It pairs with **any** of the four rungs — the text weights are
identical; the mmproj just adds the encoder + projector.

```bash
# one-shot from the CLI
./llama-mtmd-cli \
    -m Qwen3.8-27B-IQ4_XS.gguf \
    --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
    --image screenshot.png \
    --jinja -ngl 999 --temp 0.2 -n 512 \
    -p "Describe this image."

# or serve it — /v1/chat/completions then accepts image_url content parts
./llama-server \
    -m Qwen3.8-27B-IQ4_XS.gguf \
    --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
    --jinja --ctx-size 32768 --n-gpu-layers 999 \
    --host 0.0.0.0 --port 1234
```

> `--jinja` is **required** — the chat template is Jinja-based and the multimodal path needs it.
> For **grounding tasks** (pointing at or locating things in an image) llama.cpp recommends
> `--image-min-tokens 1024`; the default tokenization is fine for description and Q&A.

### OpenAI-compatible API

```python
import json, urllib.request

def ask(content, max_tokens=512):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]

print(ask("Write a Python function that reverses a linked list."))
```

---

## 🪪 License & attribution

- Inherits the license of the base model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- Calibration + quantization: [**Quant-Tuner**](https://github.com/pearsonkyle/Quant-Tuner) with vendored llama.cpp `@ f3e1828`.
- Calibration logs mined with [**LogMiner**](https://github.com/pearsonkyle/LogMiner).
