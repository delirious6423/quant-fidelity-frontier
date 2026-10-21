---
language:
  - en
  - ko
  - zh
license: mit
tags:
  - gguf
  - glm
  - glm5next
  - moe
  - quantized
  - batiai
  - korean
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
library_name: llama.cpp
---

# GLM-5.3-Flash GGUF — Quantized by BatiAI

<p align="center">
  <a href="https://flow.bati.ai"><img src="https://img.shields.io/badge/BatiFlow-on--device%20AI-blue?style=for-the-badge&logo=apple" alt="BatiFlow"></a>
  <a href="https://huggingface.co/zai-org/GLM-5.3-Flash"><img src="https://img.shields.io/badge/source-Z.ai%20official-green?style=for-the-badge" alt="Z.ai"></a>
  <a href="https://github.com/ggml-org/llama.cpp/pull/27752"><img src="https://img.shields.io/badge/needs-llama.cpp%20PR%2027752-red?style=for-the-badge" alt="PR"></a>
</p>

> ### 320B parameters. 18B active. MIT licensed.
> Quantized from Z.ai's official weights. **Korean output and tool calling verified on every
> quant** — captured transcripts below, including the tier most likely to break.

---

## ⚠️ Read this first — you need a specific llama.cpp build

`glm5next` is a **new architecture** and is **not in mainline llama.cpp yet**. Stock builds,
Ollama and LM Studio cannot load these files today.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/27752/head:pr27752 && git checkout pr27752
cmake -B build && cmake --build build -j
```

We will publish Ollama tags once the PR lands in mainline.

## 📦 Sizes and measured speed

Split into 48 GB shards because Hugging Face caps a single file at 50 GB.
**Download every shard of a quant** — llama.cpp opens `-00001-of-0000N` and finds the rest.

| Quant | Size | Shards | Generation (tg16) |
|-------|------|--------|-------------------|
| **`Q4_K_M`** ⭐ | 175.7 GiB (188.7 GB) | 4 | 7.16 tok/s |
| `IQ3_XXS` | 112.4 GiB (120.7 GB) | 3 | 6.45 tok/s |
| `IQ2_XXS` | 76.6 GiB (82.2 GB) | 2 | 7.50 tok/s |
| `Q3_K_M` | 138.8 GiB (149.1 GB) | 4 | 7.24 tok/s |
| `Q2_K` | 106.3 GiB (114.2 GB) | 3 | 7.66 tok/s |

<sub>`llama-bench -ngl 0 -n 16 -r 1` on 2× Intel Xeon Gold 6442Y (48 cores / 96 threads, 503 GB RAM).
**CPU only, no GPU offload** — the honest floor, not a best case. On Apple Silicon's unified memory
these go up substantially; we publish those numbers only once measured on the actual hardware.</sub>

**Why generation speed barely moves across tiers**: only ~18B of 320B parameters are active per
token (288 experts, top-8), so at `-ngl 0` you are bound by compute on the active slice, not by
reading the whole file. Pick the tier by the RAM you have, not by expected speed.

## ✅ Verified — every quant, including the lowest

| Quant | Korean | Han / kana | Tool call |
|---|---|---|---|
| `Q4_K_M` | 65.8% Hangul, 263 chars | **0 / 0** | ✅ |
| `Q3_K_M` | 64.5% Hangul, 276 chars | **0 / 0** | ✅ |
| **`Q2_K`** | 63.7% Hangul, 204 chars | **0 / 0** | ✅ |
| `IQ3_XXS` | 67.2% Hangul, 259 chars | **0 / 0** | ✅ |
| `IQ2_XXS` | 66.0% Hangul, 97 chars | **0 / 0** | ✅ (needs 12k budget) |

<details><summary>Captured output (Q4_K_M, temp 0)</summary>

````
**서울**: 서울은 대한민국의 수도로, 600년 역사를 지닌 고궁과 전통 한옥이 현대적인
마천루와 K컬처 속에서 어우러진 매력적인 도시입니다.
**김치**: 김치는 배추, 무 등의 채소를 고춧가루, 마늘, 젓갈로 양념해 발효시킨 한국의
대표 발효 음식으로, 건강식으로서 전 세계적으로 사랑받고 있습니다.
**비빔밥**: 비빔밥은 따뜻한 밥 위에 각종 나물과 고기, 계란을 올려 고추장과 함께...

[Tool call, raw from /completion]
</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>부산</arg_value></tool_call>
````

</details>

**Why `Q2_K` matters here.** With no importance matrix (see below), the lowest tier is where damage
would appear first — and in Korean before English. It did not: 0 Han characters, 0 kana, and a
correct tool call. That is the tier that fits a 128 GB machine, so it is the one we checked hardest.

## 🔧 Tool calling works — but `tool_calls` will be empty

GLM-5.3 emits tool calls in **its own XML-ish format**, not the JSON that llama.cpp's
OpenAI-compatible endpoint knows how to parse:

```
<tool_call>get_weather<arg_key>city</arg_key><arg_value>부산</arg_value></tool_call>
```

So `/v1/chat/completions` returns `tool_calls: []` even though the model behaved correctly. This is
a parser gap in llama.cpp, not a defect in these files — we confirmed it by reading raw
`/completion` output, quoted above.

**Until llama.cpp adds the format**, parse it yourself from the content:

```python
import re
m = re.search(r"<tool_call>(\w+)((?:<arg_key>\w+</arg_key><arg_value>[^<]*</arg_value>)+)</tool_call>", text)
if m:
    name = m.group(1)
    args = dict(re.findall(r"<arg_key>(\w+)</arg_key><arg_value>([^<]*)</arg_value>", m.group(2)))
```

Run the server with `--jinja` — without it the model's own chat template is not applied and tool
prompting degrades.

## 🧪 imatrix — what it took, and which quants have it

`IQ3_XXS` and `IQ2_XXS` are **imatrix-quantized**. The K-quants (`Q4_K_M`, `Q3_K_M`, `Q2_K`) are not — they were
built first, before we had a working importance matrix. Here is the honest history, because the
failure is more useful to you than the success.

Our first attempts all died at the same tensor:

```
collect_imatrix: non-finite values detected in blk.7.attn_v_b.weight
```

Four configurations on [PR #27752](https://github.com/ggml-org/llama.cpp/pull/27752): head
`8a8d0bc`, rebased head `20de496`, GPU attention with `-ot exps=CPU`, and a build with
`attn_k_b`/`attn_v_b` pinned to F32. `blk.7` is one of the 11 MLA layers (the other 34 are KDA
linear attention). The weights were fine — we decoded the tensor from the BF16 conversion and got
`nan=0, inf=0, absmax=0.104` — and inference was fine, so the fault was in the graph, not the file.

**Re-converting with [PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754) fixed it.** Same
weights, same corpus, no non-finite values. The two PRs also use incompatible metadata schemas
(#27754 wants a per-layer `attention.head_count_kv` array), so you cannot just swap runtimes on an
existing file — the conversion has to be redone.

One more practical note: the re-converted BF16 is 584 GiB and our machine has 468 GiB of RAM.
Running imatrix directly on it would re-read the shortfall from disk every chunk — we measured that
path at roughly **63 hours** for 200 chunks. Going through a `Q8_0` intermediate (310 GiB, fits in
RAM) brought it down to about four.

### Calibration corpus

| | |
|---|---|
| Total | 4.90 M characters |
| **Hangul** | **20.09%** |
| Han | 0.17% |
| Code markers | 3,318 |

Korean OCR labels, Korean technical prose, source code, and English wikitext. Run with
`--chunks 200`.

<sub>We report composition as measured numbers because we once wrote "mixed code + English + Korean
+ Chinese corpus" on our cards and later measured that corpus at 0.51% Korean. Numbers, not
adjectives.</sub>

## ⚖️ License — MIT, genuinely

The base model is **MIT**. No revenue thresholds, no separate license for assistant products, no
gated form. That is unusual at this scale and worth stating: unlike some frontier open-weight
releases, you can ship this in a commercial product.

## 🚀 Usage

```bash
hf download batiai/GLM-5.3-Flash-GGUF \
  --include "GLM-5.3-Flash-Q4_K_M-*" --local-dir ./glm53

./llama-server -m ./glm53/GLM-5.3-Flash-Q4_K_M-00001-of-00004.gguf \
  -ngl 0 -c 8192 -b 512 -ub 512 -np 1 --jinja --port 8080
```

**Budget tokens for reasoning.** This model always opens with `<think>` — its chat template has no
`enable_thinking` switch, and passing one is silently ignored. A three-sentence Korean answer used
~1,700 characters of reasoning at `Q4_K_M` before any content appeared. Give `max_tokens` room or
you will get an empty `content` and mistake it for a broken quant. We made exactly that mistake and
nearly discarded a good file over it.

**`IQ2_XXS` needs more room than the rest — use `max_tokens` 12,000+.** At 4,000 it failed our gate
with 0 characters of content and 5,555 of reasoning. We re-ran it and found the model was fine: a
short question ("한국의 수도는?") answered in 97 characters with 66.0% Hangul and zero Han
characters, and tool calling worked. On the longer prompt it had *already written all three
sentences* inside its reasoning, then went back to polish them — 15,822 characters, ending in
"Let me refine:" — and ran out of budget before emitting anything. Self-revision gets longer at
2 bits. Give it room and it answers.

## 📋 What is not here

- **Vision** — see below.
- **Ollama tags** — blocked on PR #27752 landing in mainline.
- **Vision** — GLM-5.3-Flash is natively multimodal, but PR #27752 is text-only. We have not
  converted or verified the vision tower, so we do not ship it.
- **Apple Silicon speeds** — we publish only numbers measured on the actual hardware.

## ✨ What BatiAI did differently

| | |
|---|---|
| 🇰🇷 **Korean verified per quant** | Down to `Q2_K`, with character counts — not a claim. |
| 🔍 **Tool calling actually tested** | Including reading raw output when the parser said zero. |
| 📐 **Numbers, not adjectives** | Sizes, speeds, and the failed imatrix diagnosis are all measured. |
| 🚧 **Gaps published** | No imatrix, no IQ quants, no vision — said plainly instead of omitted. |
| 🔏 **From the source** | Quantized from Z.ai's official weights, never a re-quant. Signed `general.author: BatiAI`. |

---

**Who we are.** [BatiAI](https://bati.ai) builds **on-device Korean AI**.
[BatiFlow](https://flow.bati.ai) runs chat, speech-to-text, document OCR and speaker diarization
entirely on a Mac. Full line: **[huggingface.co/batiai](https://huggingface.co/batiai)**
