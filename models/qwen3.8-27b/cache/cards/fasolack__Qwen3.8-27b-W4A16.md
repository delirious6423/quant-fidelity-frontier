---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
license: apache-2.0
language:
  - en
datasets:
  - NousResearch/hermes-function-calling-v1
  - ise-uiuc/Magicoder-OSS-Instruct-75K
  - NeelNanda/pile-10k
tags:
  - int4
  - w4a16
  - gptq
  - compressed-tensors
  - auto-round
  - vllm
  - quantized
  - tool-calling
  - code
---

# fasolack/Qwen3.8-27b-W4A16

Quantize **Qwen3.8-27B** to **int4 W4A16**, calibrated for **coding and tool
calling**, to run on RTX 3090s.

## Why W4A16 and not FP8

FP8 on Ampere (compute capability 8.6) is *weight-only* — vLLM falls back to the
Marlin kernel and you get the VRAM saving but no compute speedup, because 3090s
have no FP8 tensor cores. Int4 W4A16 runs the Marlin int4 path natively, so it's
both smaller (~19.5 GB vs ~52 GB BF16) and genuinely faster on this hardware.

The reference build ([dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound))
published GSM8K/HumanEval/MMLU-Pro numbers for this exact model+scheme (GSM8K
0.917 vs 0.911 BF16, HumanEval 0.957 vs 0.939, MMLU-Pro 0.826 vs 0.819 — all
within confidence intervals), which validates the **scheme and the BF16
keep-list** — but that build calibrated on generic `pile-10k`. This build's
calibration mix is materially different (see below), so those numbers don't
describe it. **See the Evaluation section near the end of this card for this
build's own measured numbers** — publishing someone else's numbers as your
model's numbers is the single most common way a quantization repo becomes
misleading.

## What makes this build different

The stock recipe calibrates on `NeelNanda/pile-10k` — generic web text.
Calibration data is what AutoRound uses to decide how to round each weight, so
calibrating on web text optimises for web text. This build calibrates on the
work the model will actually do:

| Slice | Weight | Source |
|---|---|---|
| Tool calling | 45% | `NousResearch/hermes-function-calling-v1` |
| Code | 40% | `ise-uiuc/Magicoder-OSS-Instruct-75K` |
| General | 15% | `NeelNanda/pile-10k` |

That last 15% is deliberate. Calibrating on a 100% narrow distribution tends to
cost general ability, and it's cheap insurance.

Tool-calling samples are rendered through the model's **own chat template** with
the tool schemas passed in, and the Hermes-style JSON tool calls are rewritten
into Qwen3.8's XML style:

```
<tool_call>
<function=get_weather>
<parameter=city>
Berlin
</parameter>
</function>
</tool_call>
```

So the calibration text contains the same tokens the model emits at inference —
which is the entire point of a task-focused calibration set.

## The part that will break the model if you get it wrong

Qwen3.8-27B is a **hybrid**: 64 layers, 48 GatedDeltaNet (linear-attention,
Mamba-family) + 16 full-attention, plus a vision tower and an MTP head.

**AutoRound's docs state plainly that Mamba is not supported.** It will not
detect or skip the state-space layers for you. If you run a naive `--scheme
W4A16` over this model you get a corrupted result.

These stay BF16 (`BF16_PATTERNS` in `settings.py`):

| Pattern | Why |
|---|---|
| `*linear_attn.in_proj_a` / `in_proj_b` | Per-layer recurrence-control projections of the GDN layers. Tiny, but they steer the recurrent state. |
| `*visual*` | Vision tower — vLLM requires BF16. |
| `*mtp*` | Multi-token-prediction head (speculative decoding). |
| `lm_head` | Untied 2.5 GB output embedding; protects logits. |

Everything else *is* quantized: `linear_attn.out_proj` / `in_proj_qkv` /
`in_proj_z`, the full-attention `q/k/v/o`, and all MLP `gate/up/down`.

These patterns were verified against
[`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound),
a published working build of this exact model — not derived from first
principles.

## Setup

Run this **on your GPU host**, in its own venv — not in `~/.hermes`:

```bash
python3.11 -m venv ~/.venvs/quant
source ~/.venvs/quant/bin/activate
pip install -U -r requirements.txt
```

`transformers` must know the `qwen3_5` architecture. If your release doesn't yet:

```bash
pip install -U git+https://github.com/huggingface/transformers.git
```

## Run it

```bash
# 1. Confirm the architecture is what we think it is (5 seconds, no download)
python inspect_layers.py
#    -> expect "64 layers (48 linear-attention, 16 full-attention)" and an OK

# 2. Eyeball two rendered calibration samples BEFORE committing to a long run
python build_calibration.py --dry-run 2

# 3. Build the real calibration set
python build_calibration.py

# 4. Check the plan, then quantize
python quantize.py --check
python quantize.py

# 5. Serve and test
./serve.sh 2
./verify.sh          # in another terminal

# 6. Measure this build's OWN eval numbers (do not skip -- see Evaluation
#    section below for why borrowed numbers are not good enough)
./evaluate.sh         # ~2-6h for all three tasks; run humaneval sandboxed
                       # (it executes model-generated code) -- see its notes

# 7. Push
./upload.sh --dry-run  # check only, uploads nothing
./upload.sh
```

**Stop vLLM before step 4** — it wants the whole card.

### Cost

The reference run took ~35 min on a 96 GB card, peaking at **24 GB VRAM** with
`batch_size=4`. That is exactly a 3090's ceiling and will OOM if anything else
touches the GPU, so the defaults here use `batch_size=1` +
`gradient_accumulate_steps=4` (same effective batch) and `low_gpu_mem_usage=True`.
Slower — budget **60–90 min** — but it fits with headroom.

With all four cards free, `--device-map auto` shards across them and you can
raise `batch_size` in `settings.py` for a faster run.

Also needs ~52 GB of disk for the base model plus ~20 GB for the output.

## Serving

`./serve.sh 2` — **TP=2 is the recommendation**, not TP=4. On PCIe-only 3090s
without NVLink, TP=4's all-reduce traverses host memory and eats the gains;
benchmarks on 4×3090 found TP=2 ~30% faster decode than single-GPU while TP=4
was *slower* per token. TP=4 only pays off past ~100k context. Int4 fits
comfortably on two cards, which frees the other two.

The serve flags carry over what we already established works:
`--tool-call-parser qwen3_xml` (not `hermes` — this model emits the XML style),
`--kv-cache-dtype fp8`, `--enable-auto-tool-choice`.

Note `--enforce-eager` is **not** set here. It was needed as a safety measure
for CUDA-graph memory blowups on hybrid Mamba models, but int4 leaves much more
headroom. Watch the first startup: if it OOMs during graph capture, add
`--enforce-eager` back to `serve.sh`.

## Pointing Hermes Agent at it

The served name changes, so update it:

```bash
hermes model
# custom endpoint -> http://<your-gpu-host>:8002/v1
# model name      -> qwen3.8-27b-w4a16
# compatibility   -> Chat Completions
```

## Files

| File | Purpose |
|---|---|
| `settings.py` | All the knobs — model IDs, calibration mix, BF16 keep-list, AutoRound args |
| `inspect_layers.py` | Verify the architecture and enumerate the keep-list before running |
| `build_calibration.py` | Build `calib.jsonl` from the three sources |
| `quantize.py` | The AutoRound run |
| `serve.sh` | vLLM launch (`./serve.sh 2` or `4`) |
| `verify.sh` | Tool-calling + coding smoke test |
| `evaluate.sh` | Measures this build's own GSM8K/HumanEval/MMLU-Pro numbers |
| `upload.sh` | Push to the Hub -- refuses to publish with TODOs left in the card |

## Tuning the mix

Edit `MIX` in `settings.py`. To go harder on tool calling:

```python
MIX = {"toolcalling": 0.60, "code": 0.25, "general": 0.15}
```

Keep the general slice non-zero. Then rebuild `calib.jsonl` and re-run.

## Optional: reclaiming the last 5 GB

`lm_head` and `embed_tokens` are two untied ~2.5 GB BF16 matrices that W4A16
leaves alone. [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
quantizes both to int8 and reports 370 → 417 tok/s at 64 concurrent on a *single*
3090. It needs a vLLM patch to wire up the dequant kernel, so it's a second
project rather than a flag — worth it if you end up VRAM-bound.

## Evaluation

Measured with EleutherAI lm-evaluation-harness against the served endpoint. This build's own
BF16 baseline was not measured (would need a second server on a free GPU pair); the "reference"
column below is dbirks' published BF16 number for the *unquantized* base model, included only
as an approximate sanity check, not a controlled comparison against this exact calibration.

**Note on methodology:** `--apply_chat_template` breaks these tasks for this model in two
different ways, both found by testing directly against the server rather than assumed from
the Qwen3.6-35B-A3B build (a different model with different failure modes — see that build's
STATUS.md for why assuming would have been wrong in both directions):

- **HumanEval**: under chat formatting, this model treats the function-stub-in-a-code-fence
  as an incomplete question and asks the user to supply the "missing" problem statement,
  rather than completing it. Measured via `local-completions` against vLLM's raw
  `/v1/completions` endpoint instead (no chat template — direct code continuation, the way
  this benchmark was originally designed to run). Ran sandboxed in Docker (it executes
  model-generated code) with `HF_ALLOW_CODE_EVAL=1`.
- **HumanEval sampling — two rows, on purpose:** lm-eval's task config sets `do_sample: False`,
  but that flag is dead for the API backend used here — `openai_completions.py` pops and
  discards it; only `temperature` reaches the actual request. So a naive reading of the config
  is misleading: without an explicit override, HumanEval ran with real temperature=1.0
  stochastic sampling, **one sample per problem (n=1, no averaging)** — a materially riskier
  setup for code than the deterministic greedy decoding that single-number published pass@1
  figures (including the reference row above) conventionally use, since neither the code nor
  its decoding got a second chance. Re-ran with `temperature=0` (greedy) to isolate how much of
  the gap to the reference was sampling noise versus a real difference: it recovered ~10 points,
  confirming sampling was a real factor, but didn't close the gap — the remaining ~17 points are
  most plausibly the calibration-mix difference (45% tool-calling / 40% code vs. the reference's
  generic `pile-10k`), though that wasn't isolated independently. The reference build's own
  decoding settings were never disclosed, so treat both HumanEval rows as informative, not as a
  fully controlled comparison. GSM8K/MMLU-Pro are far less sensitive to single-sample variance
  (a stray token rarely invalidates a whole multi-step answer or a multiple-choice pick), so
  they weren't re-run at greedy.
- **GSM8K / MMLU-Pro**: both use 5-shot fewshot-as-multiturn chat history. Under that longer
  multi-turn context, this model reliably drifts into drafting a new hallucinated "Question:"
  after a few filler words, and lm-eval's default stop-list (which includes `"Question:"`)
  cuts the response off right there — discarding the real reasoning and the answer before
  either is generated. This is *not* the same failure as a short isolated test would suggest
  (a simple 0-shot smoke test against this model looked fine and did not catch it — only
  inspecting actual multi-shot sample completions did). Fixed by overriding the stop list to
  `<|im_end|>` only. Caught and confirmed via direct sample inspection, not just a clean exit
  code — see STATUS.md for the investigation and the before/after numbers it corrupted
  (MMLU-Pro dropped as low as 33.71%, near the ~10% random baseline for 10-way multiple
  choice, before the fix).

| Task | Reference (BF16, dbirks) | This build (int4 W4A16) |
|---|---|---|
| GSM8K | 91.7% | 97.04% ± 0.47% |
| HumanEval (pass@1, temperature=1.0, n=1) | 95.7% (decoding undisclosed) | 68.9% ± 3.63% |
| HumanEval (pass@1, greedy) | 95.7% (decoding undisclosed) | 78.66% ± 3.21% |
| MMLU-Pro (100/subject) | 82.6% | 81.36% ± 1.01% |

## References

- [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) — reference build, recipe, evals (validates the scheme/keep-list, not this build's calibration mix — see Evaluation above)
- [intel/auto-round](https://github.com/intel/auto-round) — quantizer
- [tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab) — TP=2 vs TP=4 on 4×3090
- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) — lm_head/embed int8 extension
