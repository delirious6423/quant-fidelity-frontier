---
language:
  - en
  - ko
  - zh
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
  - gguf
  - qwen
  - qwen4exp
  - moe
  - quantized
  - batiai
  - korean
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: llama.cpp
---

# Qwen3.8-Flash-Next GGUF — Quantized by BatiAI

<p align="center">
  <a href="https://flow.bati.ai"><img src="https://img.shields.io/badge/BatiFlow-on--device%20AI-blue?style=for-the-badge&logo=apple" alt="BatiFlow"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next"><img src="https://img.shields.io/badge/source-Qwen%20official-orange?style=for-the-badge" alt="Qwen"></a>
  <a href="https://github.com/ggml-org/llama.cpp/pull/27742"><img src="https://img.shields.io/badge/llama.cpp-in%20mainline-brightgreen?style=for-the-badge" alt="mainline"></a>
</p>

> ### 176.9B parameters. ~4.9B active. Runs on a workstation.
> Quantized from Qwen's official BF16 weights, **every quant tested for Korean output,
> arithmetic and tool calling before publishing** — captured transcripts below.

---

## ⚠️ Read this first — build current llama.cpp master

`qwen4exp` support **landed in mainline** on 2026-08-27 ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)).
Build from current master:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j
```

Older llama.cpp builds, and Ollama/LM Studio releases that predate the merge, will not load these
files. Ollama tags will follow once a release ships the architecture.

<sub>**Correction.** This card originally said support was "not in mainline yet." It had already
merged the day before we published. An external validator caught it; we were wrong and have
fixed it.</sub>

**Two rough edges we hit while building this** (so you don't lose an hour):

| Tool | What happens | Workaround |
|---|---|---|
| `llama-cli` | segfaults on load, no message | use `llama-server` |
| `llama-server` (defaults) | `GGML_ASSERT(obj_new)` in ggml.c | `-c 2048 -b 512 -ub 512 -np 1` |
| `llama-bench` | works as-is | — |

The graph is large (48 layers × 512 experts + hyper-connections), and tools differ in how much
graph headroom they reserve. Nothing is wrong with the weights — `llama-bench` and `llama-server`
both run them.

## 📦 Sizes and measured speed

Split into 48 GB shards because Hugging Face caps a single file at 50 GB.
**Download every shard of a quant** — llama.cpp opens `-00001-of-0000N` and finds the rest.

| Quant | Size | Shards | Prompt (pp512) | Generation (tg16) |
|-------|------|--------|----------------|-------------------|
| `IQ3_XXS` | 79.2 GiB (85.0 GB) | 2 | — | 8.8 tok/s |
| **`IQ4_XS`** ⭐ | **90.8 GiB** (97.5 GB) | 3 | 199.0 tok/s | **11.7 tok/s** |
| `Q4_K_M` | 111.0 GiB (119.2 GB) | 3 | 157.7 tok/s | 9.1 tok/s |
| `Q6_K` | 141.2 GiB (151.6 GB) | 4 | — | 8.5 tok/s |

<sub>`llama-bench -ngl 0 -n 16 -r 1` on 2× Intel Xeon Gold 6442Y (48 cores / 96 threads, 503 GB RAM).
**CPU only, no GPU offload** — this is the honest floor, not a best case. On Apple Silicon's unified
memory these figures go up substantially — measured numbers below.</sub>

### On Apple Silicon — independently validated

An external validator ran these exact published `IQ4_XS` shards on a **Mac Studio M2 Ultra,
192 GiB** (macOS 26.2, llama.cpp `d7bd3bf`, Metal `-ngl 99`, 8K context, 1 slot, reasoning off):

| | |
|---|---|
| Generation | **32.74 tok/s** (3 runs: 32.74 / 32.75 / 32.74) |
| Cold load to healthy | **10.24 s** |
| Peak RSS | 94.28 GiB |
| Memory-pressure headroom | 48% minimum |
| Swap growth | **none** (12.19 MiB before and after) |

**~2.8× our CPU-only figure**, and it fits a 192 GiB machine at 8K context without touching swap.
`--check-tensors` passed on all three shards. Their measured SHA256s, so you can verify your
download:

````
2628d9b331262763c6a60e473d88ddfe7368035944d1068e327fcedc30af1a87  -00001-of-00003  47,587,848,672
341cd232b08807ccca05a96f507bbadd8615f2ef4cf638444b9a9997c06de9cb  -00002-of-00003  47,562,772,448
306796fc17ca004433c39c50cd5a906ad2fb6b76ce39f5cab2a2ae6f815ee639  -00003-of-00003   2,322,533,792
````

Korean, Python, tool selection, tool round-trip and structured JSON extraction all passed on that
machine. **Two arithmetic prompts failed** — see below.

### ⚠️ Do not trust it for arithmetic

The same validation ran two deterministic expressions and got both wrong:

| Prompt | Model | Correct |
|---|---|---|
| `127*89+364/13-42` | 11,323 | **11,289** |
| `2^7+17*9-3^4` | 100 | **200** |

Confident, plausible, wrong. If your workflow touches money or quantities, give the model a
calculator tool and verify the result — do not let it compute unaided. We would rather you know
this than discover it in production.

Our own publish gate only checked `127+58`, which every quant passed. That was too easy to catch
this, and we are stating the stronger result rather than the one that flatters us.

**Why generation is fast for a 177B model**: only ~4.9B parameters are active per token
(512 experts, top-10). You pay 177B in storage and roughly 5B in bandwidth.

## ✅ Verified — 7/7 on `IQ4_XS`

Run against these exact files with `llama-server -ngl 0 -c 8192 -b 512 -ub 512 -np 1`.
Functional cases use `enable_thinking:false` — see [thinking](#-thinking-runs-away-on-length-constrained-prompts).

| Check | Result |
|---|---|
| Korean generation | ✅ 70.4% Hangul, **0 Han, 0 kana** |
| Arithmetic (trivial: `127+58`) | ✅ `185` — but see the arithmetic warning above |
| Tool call (1 tool) | ✅ `get_weather` |
| Tool call (12 tools + 25-rule system prompt) | ✅ `get_sheet_info(sheet:"매출")` |
| **Tool round-trip (read → write)** | ✅ fed the read result, model followed with `set_value` carrying the right value |
| Coding | ✅ |
| Long-form Korean (repetition check) | ✅ 73.1% Hangul, 0 duplicate sentences |

<details><summary>Captured output</summary>

````
[Korean]
한국의 수도는 서울로, 고궁과 현대적인 도시가 어우러진 활기찬 도시입니다.
김치는 배추에 고춧가루와 마늘 등을 버무려 발효시킨 한국의 대표적인 매콤한 반찬입니다.
비빔밥은 밥 위에 각종 나물과 고기, 계란을 올려 고추장과 함께 비벼 먹는 음식입니다.
떡볶이는 쫄깃한 떡을 매콤달콤한 소스에 끓여 먹는 한국의 대표적인 길거리 음식입니다.
  → 70.4% Hangul, 0 Han, 0 kana

[Tool call, 12 tools available, system prompt says "validate the range first"]
{"name": "get_sheet_info", "arguments": {"sheet": "매출"}}
  ← then, given {"sheet":"매출","used_range":"A1:D200","B2":""}:
{"name": "set_value", "arguments": {"sheet": "매출", "range": "B2", "value": "1분기"}}
````

</details>

**Why the round-trip matters.** A model can emit one well-formed tool call and still be useless as
an agent. The failure we look for is the second turn — taking a tool result back and continuing.
That is where a model drops the thread, and a single-tool test never sees it.

**Why we check Korean specifically.** Quantization damage shows up in non-English output first;
we have measured models that start splicing Han characters into Korean sentences at low bit
widths. That matters more than usual here — see the PLE section below.

## 🧠 Thinking runs away on length-constrained prompts

Ask for "explain X **in about 400 characters**" with reasoning enabled and the model can spend its
entire token budget counting characters in its own draft and never emit an answer. Drop the length
constraint and it answers in a few hundred tokens.

We measured it — 5 runs per quant, 6,000-token budget, temp 0.7, fixed seeds, alternating the two
prompt forms:

| | with "in about 400 characters" | without the constraint |
|---|---|---|
| `IQ4_XS` | **1/3** reached an answer | **2/2** |
| `Q6_K` | **0/3** | **2/2** |

Failed runs burn the full 6,000 tokens on 9,500–12,400 characters of reasoning. Successful
unconstrained runs finish in 543–1,576 tokens with 419–536 characters of reasoning.

**This is not quantization damage.** `Q6_K` — the highest-bit file we publish — does it slightly
*more* than `IQ4_XS`, and both are perfect without the constraint. Bit width is not the variable;
the length constraint is.

<sub>We nearly shipped the opposite claim. Our first comparison was one run per quant, and it
happened to show `Q6_K` converging where `IQ4_XS` did not — which reads as low-bit damage. Repeating
it five times each reversed the finding. One sample per arm was never enough to move a
recommendation, and we would have published a wrong warning had we stopped there. Sample size is
still only 5 per arm — weigh it accordingly.</sub>

**What to do**: pass `"chat_template_kwargs": {"enable_thinking": false}` for anything with a length
or format budget, or leave reasoning on and drop the explicit character count. Our functional
verification above uses thinking off for exactly this reason.

## 🔍 The PLE table is 46% of the file

This checkpoint carries a **per-layer n-gram embedding table (PLE)** of ~51.2B parameters —
a 20M-entry vocabulary split across 128 shards in the original weights. In a 4-bit GGUF it is
roughly **46% of the file**, so how it quantizes dominates both size and quality.

**It also decides which quants can exist on Hugging Face at all.** Its column count is 160, so the
256-block types (`Q6_K`, `Q4_K`) cannot hold it and llama.cpp falls back to a 32-block type — a
*different* one per quant:

| Quant | PLE falls back to | PLE size | Fits HF's 50 GB file limit? |
|---|---|---|---|
| `IQ4_XS` | `IQ4_NL` | 28.8 GB | yes |
| `Q4_K_M` | `Q5_0` | 35.2 GB | yes |
| `Q6_K` (default) | **`Q8_0`** | **54.4 GB** | **no** |

A single tensor cannot be split across shards, so `Q6_K` with the default fallback is simply
unpublishable here — no `--split-max-size` value helps. We rebuilt it with
`--tensor-type per_layer_token_embd=Q5_1`, giving a 38.40 GB PLE that fits, and still a step above
the `Q5_0` that `Q4_K_M` gets. **The `Q6_K` in this repo therefore carries a Q5_1 PLE, not Q8_0** —
the constraint is the file limit, not a quality judgement, and we would rather say so than let you
assume otherwise. The rebuilt file re-passed load, signature and the Korean check
(70.5% Hangul, 0 Han, 0 kana), so the downgrade did not cost us the thing the PLE is there for.

Two more consequences worth knowing:

- **It falls back to `IQ4_NL`.** Its column count is 160, which is not divisible by the 256 that
  `IQ4_XS`/`Q4_K` blocks need, so llama.cpp demotes just that tensor. You will see a warning at
  quantization time; it is expected, not a defect.
- **We did not protect it with a higher bit width.** Pinning it to Q8 would push `IQ4_XS` past
  120 GB and defeat the point. We left it at the file's own type and let the Korean checks decide.
  They passed at every tier we publish — **including `IQ3_XXS`, the lowest: 70.0% Hangul,
  0 Han characters, 0 kana, and a working tool call.** If the PLE were degrading at low bit
  widths, that is where it would have shown first.

`IQ3_XXS` needed one more override to exist at all: the hyper-connection tensors
(`hc_count 4`, `hc_lowrank 320`) are not covered by our imatrix, and very-low-bit types refuse to
quantize without one — `Missing importance matrix for tensor output_hc_down.weight`. Pinning them
with `--tensor-type hc_=IQ4_NL` fixes it, and since they are low-rank the size cost is negligible
(6.25 MiB → 1.76 MiB each).

## 🧪 imatrix calibration — measured, not described

Corpus composition, from the sidecar we generate with the corpus:

| | |
|---|---|
| Total | 4.90 M characters |
| **Hangul** | **20.13%** |
| Han | 0.17% |
| Code markers | 3,228 |

Sources: Korean OCR labels (38.5% Hangul), Korean technical prose (18.8%), source code, and
English wikitext. Run with `--chunks 200`.

<sub>We used to write "calibrated on a mixed code + English + Korean + Chinese corpus" on our
cards. We measured that corpus and it was 0.51% Korean — effectively English wikitext. That was
wrong, we corrected it, and we now report composition as numbers instead of adjectives.</sub>

## ⚖️ License — redistribution yes, AI-assistant products need their own license

Base model is under the **Qwen Community License 1.0**, which explicitly permits copying,
modifying, distributing and sublicensing — that is what makes these GGUFs possible.

One clause needs attention before you deploy:

> If the licensee or any of its affiliates conducts a **Model as a Service** or
> **AI Work Assistant** business, the licensee shall obtain a separate license from Qwen before
> Using the Software or its derivative works for any commercial purpose.

"AI Work Assistant" covers products **primarily designed for AI-assisted coding or office
productivity**. If that describes your product, talk to Qwen before shipping this model in it.
Attribution in the UI is additionally required above 100M MAU or US$20M monthly revenue.

We publish these weights as a distribution artifact. **We do not ship this model in our own
office product** for exactly this reason.

## 🚀 Usage

```bash
# Download one full quant (all shards)
hf download batiai/Qwen3.8-Flash-Next-GGUF \
  --include "Qwen3.8-Flash-Next-IQ4_XS-*" --local-dir ./flashnext

# Serve — the flags matter, see the table at the top
./llama-server -m ./flashnext/Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf \
  -ngl 0 -c 2048 -b 512 -ub 512 -np 1 --port 8080

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages":[{"role":"user","content":"한국의 수도를 소개해줘."}],
  "temperature":0,"max_tokens":400}'
```

Tool calling works through the OpenAI-compatible endpoint; pass `tools` and read `tool_calls`.

## 📋 What is not here

- **Ollama tags** — the architecture is in llama.cpp master now; we publish tags once an Ollama release ships it.
- **Vision (`mmproj`)** — the checkpoint has a Qwen3-VL vision tower. We converted it but have not
  verified image input, so we are not publishing it as working yet.
- **Apple Silicon speeds** — we only publish numbers measured on the actual hardware.

## ✨ What BatiAI did differently

| | |
|---|---|
| 🇰🇷 **Korean verified per quant** | Not a claim — the transcript is above, with character counts. |
| 📐 **Numbers, not adjectives** | Sizes, speeds and corpus composition are all measured and stated. |
| 🚧 **Failures published** | `IQ3_XXS` failed; we say so and why, instead of quietly omitting it. |
| 🔏 **From the source** | Quantized from Qwen's official BF16, never a re-quant. Signed `general.author: BatiAI`. |

---

**Who we are.** [BatiAI](https://bati.ai) builds **on-device Korean AI**.
[BatiFlow](https://flow.bati.ai) runs chat, speech-to-text, document OCR and speaker diarization
entirely on a Mac. Full line: **[huggingface.co/batiai](https://huggingface.co/batiai)**
