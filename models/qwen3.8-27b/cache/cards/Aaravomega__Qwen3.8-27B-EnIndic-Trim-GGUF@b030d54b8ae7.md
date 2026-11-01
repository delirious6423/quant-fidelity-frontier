---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language: [en, hi, bn, ta, te, gu, kn, ml, mr, pa, or]
datasets: [ai4bharat/MILU]
tags: [gguf, llama.cpp, vocabulary-trimming, indic, multilingual, qwen, imatrix, 2-bit, iq2_m, hindi, bengali, tamil, telugu, gujarati, kannada, malayalam, marathi, punjabi, odia]
model-index:
- name: Qwen3.8-27B-EnIndic-Trim (Q4_K_M)
  results:
  - task: {type: multiple-choice, name: Multiple choice (thinking mode, answer forced at 3,072 thinking tokens)}
    dataset: {name: MILU, type: ai4bharat/MILU, split: test}
    metrics: [{type: accuracy, value: 78.5, name: "MILU accuracy, macro over 11 languages (330 questions)"}]
- name: Qwen3.8-27B-EnIndic-Trim (IQ2_M)
  results:
  - task: {type: multiple-choice, name: Multiple choice (thinking mode, answer forced at 3,072 thinking tokens)}
    dataset: {name: MILU, type: ai4bharat/MILU, split: test}
    metrics: [{type: accuracy, value: 77.1, name: "MILU accuracy, macro over 11 languages (1,100 questions)"}]
---

# Qwen3.8-27B · English + 10 Indic languages · trimmed GGUF

**A leaner Qwen3.8-27B for Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia and English.** About 113k tokens that only serve other languages are gone from the vocabulary, and the model comes in two sizes: a near-lossless **Q4_K_M (18.2 GB)** and a **2-bit IQ2_M (9.4 GB)** that fits a 16 GB GPU, with an importance matrix tuned on Indic text.

![MILU overall](assets/milu_overall.png)

## Highlights

- **78.5% MILU (Q4_K_M) and 77.1% (IQ2_M)** across 11 languages in thinking mode. For reference, Sarvam-30B's published MILU is 76.8 (different protocol, see [Results](#results-milu)).
- **Half the size, still strong.** The 2-bit build is **9.35 GB** (2.91 bits per weight), about half of Q4_K_M. On the 330 questions both builds answered, 2-bit is about 4.6 points behind Q4.
- **Indic-aware 2-bit.** The importance matrix was computed on the original full-precision model with **53.5% Indic calibration text** (10 languages plus romanized). The `imatrix.dat` is included so you can make your own quants.
- **Nothing retrained.** Every surviving weight is identical to the original. Text that uses only kept tokens is tokenized identically and gives identical logits (verified, see [Verification](#verification)).
- **Smaller, cleaner vocabulary:** 248,320 to 135,384 embedding rows (-4.2% parameters), with English, code, math, chat/tool formatting, romanized Indic and vision tokens preserved.
- **Vision works:** the original vision projector is included, unchanged.

## Which file should I download?

| File | Size | Best for | Notes |
|---|---:|---|---|
| `Qwen3.8-27B-EnIndic-Trim-Q4_K_M.gguf` | 18.2 GB | best quality | the default choice if you have ~24 GB of GPU/unified memory or RAM |
| `Qwen3.8-27B-EnIndic-Trim-IQ2_M.gguf` | 9.4 GB | small GPUs, laptops | verified on a 15 GB Tesla T4 with about 30k tokens of context shared by 6 parallel chats; about 13% higher perplexity than Q4 on a small slice (see below) |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 0.9 GB | images | unchanged vision projector for both models (`--mmproj`) |

Both text models share the same trimmed tokenizer, so they are drop-in replacements for each other.

## Results (MILU)

[MILU](https://huggingface.co/datasets/ai4bharat/MILU) is AI4Bharat's multi-task, multiple-choice knowledge benchmark covering 11 languages (10 Indic + English). We ran it with the model **thinking** (its default), zero-shot, on randomly sampled test questions (seed 0, stratified by domain).

![MILU per language](assets/milu_per_language.png)

| Language | Q4_K_M (30 q each) | IQ2_M (100 q each) |
|---|---:|---:|
| Bengali | 90.0 | 77.0 |
| English | 80.0 | 83.0 |
| Gujarati | 86.7 | 73.0 |
| Hindi | 76.7 | 74.0 |
| Kannada | 70.0 | 77.0 |
| Malayalam | 76.7 | 75.0 |
| Marathi | 63.3 | 71.0 |
| Odia | 76.7 | 76.0 |
| Punjabi | 76.7 | 78.0 |
| Tamil | 93.3 | 77.0 |
| Telugu | 73.3 | 87.0 |
| **Macro average** | **78.5** (95% CI 73.7 to 82.6) | **77.1** (95% CI 74.5 to 79.5) |

**How to read this honestly**

- **Protocol.** Each question is answered in thinking mode with a **3,072-token thinking budget**; if the model has not finished thinking by then, we close the thought and force one letter A to D (constrained decoding). Sampling follows Qwen's thinking recommendations (temperature 0.6, top-p 0.95, top-k 20). Prompt: the question, four options and the instruction to end with `Answer: X`, using the model's built-in chat template.
- **About 16-17% of questions hit the thinking budget** (53 of 330 for Q4, 187 of 1,100 for IQ2_M). Those forced answers are only about 43-48% right. Questions that finished thinking in budget scored **85.2% (Q4) and 83.0% (IQ2_M)**. If you instead count every cut-off question as wrong, the scores are 74.5% (Q4) and 70.1% (IQ2_M). Allowing more thinking tokens should raise both.
- **Without thinking** (letter-logit scoring, no reasoning), IQ2_M scored 58.0% on the same 1,100 questions. Thinking adds 12 to 19 points, depending on how cut-off questions are counted (70.1% or 77.1%).
- **Q4 vs IQ2_M on the same 330 questions:** 78.5% vs 73.9%. Q4 got 31 questions right that IQ2_M missed, and IQ2_M got 16 that Q4 missed (z = 2.2). These 330 questions happen to be harder for IQ2_M than the rest of its sample (73.9% on them, versus 77.1% over all 1,100), so compare the two builds on the matched 330, not on the headline numbers.
- **Small samples.** Per-language numbers have wide error bars (about +/-15 points for Q4, +/-9 for IQ2_M at 95%). Only the macro averages are reasonably tight.
- **Comparison with other models is indicative only.** The grey bars in the first chart come from the [Sarvam-30B model card](https://huggingface.co/sarvamai/sarvam-30b) (reasoning up to 65,536 tokens, unstated prompt/shots). Our protocol differs, so treat it as "same ballpark", not a ranking.
- **No MILU score for the untrimmed Qwen3.8-27B.** We did not run MILU on the original model, so we cannot report how much (if anything) the trim or quantization costs on this benchmark. The trim itself is verified separately (below).

Per-question results (without any text), scripts and the exact protocol are in [`eval/`](eval/).

## Verification

**Trim fidelity (Q4_K_M versus the original Q4_K_M file):**

| Check | Result |
|---|---|
| Tensors and hashes: only `token_embd` and `output` changed (rows removed); 849 other tensors hash-identical; kept rows identical | PASS |
| Tokenizer equivalence on held-out documents (HF and llama.cpp) | PASS (3,143 docs, 0 mismatches) |
| Encode/decode round trip | PASS |
| Logits of kept tokens versus original | PASS (max abs diff 5.7e-06, argmax agreement 100%) |
| Greedy generation vs original on 82 prompts (11 languages, code, math) | PASS (82 of 82 identical) |
| Perplexity on target-domain text | PASS (6.295 to 6.282) |
| Vision smoke test with the unchanged mmproj | PASS (identical output) |

**IQ2_M:** built from the *trimmed BF16* weights with the importance matrix below. Tokenizer check (99 strings) PASS: token ids identical to the Q4_K_M build, byte-exact round trip. On a small perplexity slice (12.8k tokens: 5 English and 20 Indic documents, no confidence interval) it is **+13.1% perplexity versus Q4_K_M** (6.11 vs 5.41; English +12%, Indic +13%).

Full numbers: [`report.md`](report.md) and [`verify/verify.json`](verify/verify.json) (Q4_K_M), [`eval/summary.json`](eval/summary.json) (MILU).

## How the vocabulary was trimmed

- Removed **112,693 tokens**: 100,691 contain a character from a script outside the kept set (Han, Kana, Hangul, Cyrillic, Arabic, Hebrew, Thai, Lao, Khmer, Myanmar, Georgian, Armenian, Ethiopic, Sinhala, Tibetan and others) and 12,002 are Latin/Greek pieces that appear in European-language text but essentially never in our English/Indic calibration text. 243 padding rows were dropped.
- Kept: all Indic-script tokens for the ten languages, English, digits, punctuation, symbols, emoji, whitespace, **all single-byte tokens** (so any text can still be encoded), special/chat/vision tokens, and every token needed by a kept BPE merge.
- **1,156,464,640 parameters removed (4.2%)** from the embedding and output matrices; all other weights are untouched.
- The plan, policy and old-to-new id map are in [`trim/`](trim/), so the trim is fully reproducible and auditable.

## The importance matrix (`imatrix/`)

`imatrix/imatrix.dat` is a llama.cpp importance matrix computed on the **original BF16 model** (original 248k vocabulary; it applies to the trimmed model because only vocabulary rows changed, not the shapes of the weight matrices it describes).

- 299 chunks of 512 tokens: **139 general** (129 from bartowski's `calibration_datav3`, 10 from FineMath) + **160 Indic** (150 native-script chunks, 15 per language from FineWeb-2, plus 10 romanized from Sangraha).
- 53.5% Indic. No MILU text was used. The exact composition is in `imatrix/composition.json`.
- Covers all 497 weight matrices, with no missing or invalid entries. Built with llama.cpp commit `ac8e1694d`.

## How to run

You need a recent llama.cpp with the `qwen35` architecture (tested with commit `ac8e1694d`).

```bash
# chat, thinking on (the default), Qwen's recommended sampling
llama-cli -m Qwen3.8-27B-EnIndic-Trim-IQ2_M.gguf -ngl 99 -c 16384 --jinja -cnv \
          --temp 0.6 --top-p 0.95 --top-k 20

# OpenAI-compatible server with vision
llama-server -m Qwen3.8-27B-EnIndic-Trim-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-BF16.gguf \
             -ngl 99 -c 32768 --jinja
```

Tips:

- **Leave thinking on** for knowledge questions; it was worth 12 to 19 MILU points in our tests. Budget roughly 1-3k tokens of thinking per question and set the context accordingly.
- **Use sampling, not greedy decoding**, especially with IQ2_M: in our smoke test greedy Hindi output from the 2-bit model fell into a repetition loop. A moderate temperature (0.6) avoided this in the MILU runs.
- Hugging Face tokenizer (ids are the *trimmed* ids; `trim/old_to_new_id.json` maps from the original ids):

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Aaravomega/Qwen3.8-27B-EnIndic-Trim-GGUF", subfolder="tokenizer")
```

Speed is that of a 27B dense model: about 2.4 tokens/s generation on an 8-vCPU server for Q4_K_M, and about 7.5 tokens/s for one stream of IQ2_M on a single Tesla T4 (more in aggregate with parallel streams). These are indicative numbers from shared cloud machines.

## Limitations

- **Languages outside the kept set get much worse.** Text in Chinese, Japanese, Korean, Russian, Arabic, Urdu and similar still encodes losslessly (byte fallback) but needs about 4-7x more tokens and the model reads and writes it far less well. French, German, Spanish and especially Vietnamese lose 2-16% of their tokens. Do not use these builds for those languages. Numbers are in `report.md`, section 6.
- **Mixed documents degrade too.** About 1.8% of held-out English/code/chat/Indic documents contain at least one removed token (for example code with Chinese or Russian comments, or names in other scripts). Around such tokens quality drops (bits per byte up about 37% in worst-case windows). Documents without removed tokens behave identically to the original.
- **IQ2_M is a 2-bit model.** It is measurably weaker than Q4_K_M (about 4.6 MILU points on matched questions, +13% perplexity on a small slice), can loop under greedy decoding, and hit the 3,072-token thinking cap about as often as Q4_K_M in our runs (17.0% vs 16.1%).
- Token ids differ from the original model, so these builds cannot be paired with tooling that assumes the original vocabulary (for example speculative decoding with an untrimmed draft model).
- The size saving from the trim itself is modest (about 4%) because the transformer layers dominate the weights; the main gain is a clean vocabulary with unchanged behaviour on the kept languages, plus the 2-bit build.
- Benchmark samples are small and the MILU protocol differs from published ones (see Results).

## Files

| Path | What |
|---|---|
| `Qwen3.8-27B-EnIndic-Trim-Q4_K_M.gguf` | trimmed text model, Q4_K_M |
| `Qwen3.8-27B-EnIndic-Trim-IQ2_M.gguf` | trimmed text model, 2-bit IQ2_M with Indic-weighted imatrix |
| `mmproj-Qwen3.8-27B-BF16.gguf` | vision projector, unchanged from the original |
| `imatrix/` | `imatrix.dat` and `composition.json` (calibration mix) |
| `eval/` | MILU per-question results (no text), summary, scripts, protocol |
| `tokenizer/` | trimmed Hugging Face tokenizer and config (chat template identical to the original) |
| `trim/` | trim policy, plan summary, old-to-new id map, removed-token list, manifest |
| `report.md`, `verify/` | full verification report and raw results (Q4_K_M) |
| `SHA256SUMS` | checksums of every file |

## License, credits and citations

- Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and released under the same **Apache-2.0** license (see `tokenizer/LICENSE`). The Q4_K_M weights were derived from the `ggml-org/Qwen3.8-27B-GGUF` conversion.
- Benchmark: [MILU](https://huggingface.co/datasets/ai4bharat/MILU) by AI4Bharat (CC-BY-4.0, arXiv:2411.02538). Reference numbers for other models are from the [Sarvam-30B model card](https://huggingface.co/sarvamai/sarvam-30b). Question text is not redistributed here; only question ids and predictions.
- Calibration data: bartowski's `calibration_datav3`, [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath).
- Built with [llama.cpp](https://github.com/ggml-org/llama.cpp).
