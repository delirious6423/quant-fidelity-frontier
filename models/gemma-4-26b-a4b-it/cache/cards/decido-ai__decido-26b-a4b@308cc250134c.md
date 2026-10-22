---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags: [decision-model, classification, calibration, gguf, llama.cpp, cpu, mixture-of-experts]
language: [en]
---


> **Licence:** Apache-2.0. Training-data provenance and source licences are listed in `NOTICE`.
>
> **Code, evaluation harness and installer:** https://github.com/decido-ai/decido

# Decido 26B-A4B


![six_group_mean](https://cdn-uploads.huggingface.co/production/uploads/6aa7491ac1164ca4f122fd47/JnvfhPJrNcLI6VqGu7z_K.png)

**A calibrated decision model that runs on a laptop CPU.** Give it a document (`state`) and typed questions —
yes/no (`noul`), pick-one (`choice`), or ordered scale (`score`) — and it returns a probability for every answer.
It is a LoRA fine-tune of Gemma 4 26B-A4B (26B parameters, ~4B active per token), shipped as an 8-bit GGUF for llama.cpp.

On the six public benchmark groups used to compare open reproductions of TypeSafe's Jev 1.13, the shipped CPU file scores a
**six-group mean of 85.38 vs Jev 1.13's published 83.85** — trained on public data, **with no Jev outputs and no Claude
outputs** in training.

## Results

Shipped GGUF (Q8_0). Five groups measured CPU-only; OpenJev on the same file via llama.cpp CUDA (all 222k requests; a CPU-only
300-per-task sample agrees on 97.4% of items).

| group | Decido (GGUF) | Jev 1.13 (published) | zero-shot base | Decido w/o in-domain train splits |
|---|---:|---:|---:|---:|
| JevBench (231) | 87.61 | 87.18 | 83.83 | 89.30 |
| Kev (6 splits) | 84.35 | 85.52 | 78.83 | 82.66 |
| OpenJev text (18 tasks) | 78.37 | 72.96 | 56.88 | 67.77 |
| Nimble (282) | 87.23 | 91.84 | 82.27 | 86.17 |
| VitaminC (599) | 86.14 | 78.46 | 73.62 | 79.47 |
| MASSIVE-en (350) | 88.57 | 87.14 | 86.86 | 86.86 |
| **six-group mean** | **85.38** | **83.85** | 77.05 | 82.04 |

The last two columns are GPU bf16 runs of the zero-shot base and of an ablation trained without any train split of a scored
dataset. **Read the margin as in-domain supervised:** without those train splits the recipe scores 82.04, below Jev
(which was evaluated zero-shot). Jev's numbers come from AutoTrust's published run, not from running Jev through our harness.

**Calibration (multi-class Brier, lower is better; Jev reference in brackets):** JevBench 0.169 (0.181), Kev v4 dev 0.210
(0.211), VitaminC 0.202 (0.313), MASSIVE 0.164 (0.204) — better on 4 of 5; **worse on Kev v9 dev 0.267 (0.212)**.

**Robustness:** answers are bit-identical across repeated calls, request orders and thread counts on CPU; option-order flip
rate 4.8% pooled (1.4% with `perm=4`). For reference Jev flips 14.6% with 64 options and ~7% with 16, and is nondeterministic
on ~1–4% of calls depending on the setting (not a matched protocol).

**Platform variance:** outputs are bit-identical on a given machine, but x86 and Apple Silicon kernels round slightly differently. On 1,130 identical items the Apple M4 Pro and the x86 devserver agreed on 94–100% of answers per group; 31 of 32 disagreements were close calls (margin < 0.1), and neither platform was systematically better (14 vs 10 items won). Expect about ±1.5 points per group on small groups.

**Speed (Apple M4 Pro, CPU only, 10 threads):** ~1k-token state + 5 questions: p50 4.14 s, p95 4.26 s; one-time load 97 s;
~30 GB RAM per process. Cost = one pass over the state + a per-question cost (the state is read once and cached).

## How it works
- Prompt: system instruction + state in `<document>` (treated as untrusted; special-token strings neutralised) + one question block.
- Readout: the next-token probability of single-token labels (Yes/No; A–Z, or A0–Z9 above 26 options; 0–9 for scores),
  softmax over the labels, per-type temperature (noul 1.027, choice 1.146, score 1.013, fitted on held-out tasks).
- Each question is decoded separately from the cached state, so one question's answer never depends on the others.
- Output format matches Jev's wire answers (`choice`, `probabilities`, `confidence`, `score`, `legend`).

## Quick start (macOS / Linux, ~30 GB RAM)
```bash
git clone https://github.com/decido-ai/decido && cd decido
./install.sh      # builds llama.cpp + the readout server, downloads this model, runs an example
```

## Usage
The GGUF loads in any llama.cpp build, but calibrated decisions need the included runtime (a small libllama server + Python):
```bash
# llama.cpp CPU build with shared libs, e.g.: cmake -B build -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=ON && cmake --build build -j
./build_vserve.sh /path/to/llama.cpp build
pip install -r requirements.txt
python example.py
```
```python
from decido.predictor_llamacpp import DecidoLlamaCpp
m = DecidoLlamaCpp(gguf="decido-26b-a4b-q8_0.gguf", bundle="bundle", threads=10, jev_wire=True)
m({"state": "Customer: I was charged twice for order 4821.",
   "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                 "team": {"type": "choice", "instructions": "Which team handles this?",
                          "criteria": {"billing": "Charges and refunds", "shipping": "Delivery"}}}})
```
Limits: choice 2–255 options, score 2–10 levels, ~16k tokens per request (longer states are rejected, not truncated).

## Jev-compatible API server
Run Decido behind the same HTTP API as TypeSafe's Jev (`POST /v1/systemone`), so existing Jev apps can switch with one
setting. Standard library only; binds to 127.0.0.1 by default.
```bash
python -m decido.server --gguf decido-26b-a4b-q8_0.gguf --bundle bundle --port 8080
```
Use it with the official TypeSafe SDK, unchanged:
```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8080 TYPESAFE_API_KEY=local python your_app.py
```
or with plain HTTP:
```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for order 4821, please refund one charge.",
  "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}}}'
```
Supports `noul`, `choice` (2–255 options) and `score` (2–10 levels); gzip and zstd request bodies; Jev's status codes
(400 incl. `max_tokens_exceeded`, 413, 415, 422). Set `DECIDO_API_KEY` to require `Authorization: Bearer <key>` (do this
before using `--host 0.0.0.0`). `--workers 2` serves two requests in parallel (~4 GB extra RAM). Tested with the official
`typesafe-sdk` 0.7.5 quickstart.

## Training data
~550k decision questions over two rounds, train splits only: tasksource typed decisions (commercial-tagged rows),
procedural-typed-decisions, Open-Jev (CC0), NLI (MNLI, WANLI, SNLI), knowledge MC (MMLU auxiliary-train, ARC, CommonsenseQA,
HellaSwag, WinoGrande, GSM8K, medmcqa, AQuA-RAT, QASC, QuaRTz, CosmosQA), intent (MASSIVE, banking77, CLINC), VitaminC, BoolQ,
human soft labels (civil_comments, go_emotions, measuring-hate-speech, HelpSteer2, oasst1), rule-generated decision tables and
compound questions, and 13k scenarios written by the open Qwen3.8-27B model (Apache-2.0), which also provided 30% of the
soft targets (70% gold). Every mixture passed an exact-hash / 13-gram / MinHash leakage gate against all evaluation sets,
plus two independent audits. Not used: Jev outputs, Claude outputs, any benchmark test split.

## Limitations
- The benchmark margin depends on in-domain train splits (82.04 without them).
- Trails Jev on Nimble (−4.6) and Kev (−1.2); fails the Kev v9 Brier check.
- Injected text that asserts facts can move answers (e.g. "pre-approved by the CFO"); instruction-only injections barely do.
- ~10× slower than Jev's hosted API (seconds vs ~0.3 s), but free, offline and deterministic.
- 4-bit quantisations disagree with bf16 on noticeably more answers; only Q8_0 is recommended.

## Files
| path | what |
|---|---|
| `decido-26b-a4b-q8_0.gguf` | the model (Q8_0, 26.9 GB) |
| `bundle/` | frozen prompt template (`verdict-t1`), tokenizer, per-type temperatures |
| `decido/` | Python runtime: template, label readout, Jev-compatible answer fields, CPU predictor (`DecidoLlamaCpp`) |
| `tools/vserve/vserve.cpp`, `build_vserve.sh` | libllama readout server (state read once, questions answered from the cache) |
| `install.sh` | one-command setup: llama.cpp + vserve + Python env + smoke test |
| `decido/server.py` | Jev-compatible HTTP server (`python -m decido.server`) |
| `example.py`, `requirements.txt` | minimal usage |
| `lora/` | the LoRA adapter alone (for merging onto `google/gemma-4-26B-A4B-it`) |
| `LICENSE`, `NOTICE` | Apache-2.0; base model, runtime and training-data provenance |
| `SHA256SUMS` | checksums |
