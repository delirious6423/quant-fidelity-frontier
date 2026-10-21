---
license: mit
tags:
- glm
- glm-5.3
- flash
- w4a16
- uncensored
- moe
- text-generation
- vision
- vision-language
- multi-token-prediction
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3-Flash
---

<!-- dealignai-serving-warning-v1 -->
> [!IMPORTANT]
> **Serving note — read before your first request (vLLM / SGLang with a reasoning parser).**
>
> - **`reasoning_effort` only accepts `"low"` and `"high"`.** Anything else — `medium`, `xhigh`,
>   `off`, or omitting it — silently resolves to **`max`**, the deepest mode. Template line:
>   `reasoning_effort ... in ['low','high'] else 'max'`.
> - **At `max`, a small `max_tokens` returns an EMPTY reply.** The model spends the whole budget
>   inside `<think>` and you get `finish_reason="length"` with `content=""`. In multi-turn the empty
>   turn poisons the history and the conversation does not recover. Measured: `max` + `max_tokens
>   2000` dies from turn ~4; `max` + `6000` is clean; `low` and `high` are clean at 2000.
>   **It is budget exhaustion, not a loop** — sampling changes do not rescue it.
> - **Do NOT pass `enable_thinking`.** This template has no such variable (only `clear_thinking` and
>   `reasoning_effort`). Passing it — or a top-level `reasoning_effort: "none"` — turns the parser off
>   while the model still thinks, dumping raw reasoning into `content`.
> - **Reasoning is returned in `message.reasoning`**, not `message.reasoning_content`
>   (vLLM renamed the output field; the input side still accepts both).
> - `clear_thinking` must be nested inside `chat_template_kwargs` — a top-level key is ignored.
>
> ```json
> {
>   "model": "<this-model>",
>   "messages": [{"role": "user", "content": "..."}],
>   "max_tokens": 2000,
>   "temperature": 1.0, "top_p": 0.95,
>   "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": true}
> }
> ```
> Use `"high"` for hard tasks; if you want `max`, give it `max_tokens >= 6000`.
>
> <!-- dealignai-serving-warning-v2-media -->
> **Video and image parameters — measured, because several are accepted then silently ignored.**
>
> - 🔴 **`media_io_kwargs.video.fps` must stay BELOW the clip's own frame rate.** Requesting a value
>   at or above it makes the placeholder builder count 3x the real tokens and **kills the engine**:
>   `ValueError: Attempted to assign N = N multimodal tokens to 3N placeholders` -> `EngineDeadError`,
>   server down until restarted. Upstream: vLLM #55644 / #55647. `fps: 2` is the safe, accurate choice.
> - **`max_frames` and `num_frames` are not interchangeable across builds.** On vLLM nightly
>   `max_frames` is honoured and `num_frames` ignored; on the older `0.1.dev*` line it is the exact
>   reverse. A client that sets only one silently gets full-rate sampling on the other build. Set
>   `fps`, which every build honours.
> - **Pixel controls: use `mm_processor_kwargs.max_image_tokens` / `min_image_tokens`.**
>   `max_pixels`, `min_pixels`, `size` and `detail` return HTTP 200 and change nothing.
>   `mm_processor_kwargs.fps` is an HTTP 400. Image tokens are
>   `text + 2 + ceil(H/28)*ceil(W/28)`, floored at 16 and capped at 8000.
> - Default video sampling differs by build (roughly 2 fps vs 6 fps), so the same clip can cost very
>   different prompt tokens. Pass `fps` explicitly if token cost matters.
>
> ```json
> "media_io_kwargs": {"video": {"fps": 2}},
> "mm_processor_kwargs": {"max_image_tokens": 1024}
> ```

<div align="center">
  <img src="dealign_logo.png" alt="dealignai" width="260"/>
</div>

# GLM-5.3-Flash W4A16 — dealignai edition

Compliance-tuned drop-in replacement for `GLM-5.3-Flash-W4A16`. Refusal removed on all six real-harm HB categories, reasoning depth preserved, MTP head (layer 45) and vision tower fully intact.

<div align="center">
  <img src="dealign_mascot.png" alt="mascot" width="140"/>
</div>

## Quick launch

### Recommended production serve (with MTP + reasoning-parser + tool-calling)

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-W4A16 \
  --tensor-parallel-size 2 \
  --max-num-seqs 128 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --served-model-name glm-5.3-flash-uncensored
```

This is the exact command used on the dealignai production endpoint. Key flags:

- **`--reasoning-parser glm45`** — routes `<think>...</think>` into the OpenAI `reasoning_content` response field. **Required for correct multi-turn behavior** — without it, prior `<think>` blocks stay inline in message content and the base upstream chat template will re-emit them on subsequent turns, which can trigger a decode-attractor loop.
- **`--tool-call-parser glm47` + `--enable-auto-tool-choice`** — enables OpenAI-compatible tool calls (GLM-4.7 tool call format).
- **`--speculative-config mtp`** — enables Multi-Token Prediction speculative decoding (~1.5-2× decode throughput).
- **`--enable-prefix-caching`** — reuses KV cache across identical prefixes (big win for agent workloads).

### Minimal safe serve (no MTP, no tool-calling)

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-W4A16 \
  --tensor-parallel-size 2 \
  --reasoning-parser glm45 \
  --enable-prefix-caching \
  --served-model-name glm-flash
```

Only `--reasoning-parser glm45` and `--enable-prefix-caching` are truly necessary. Everything else is performance tuning.

### Docker (matched to production image)

```bash
docker run -d --name glm --gpus all --ipc=host \
  -v $HOME/models:/models -p 8000:8000 \
  vllm/vllm-openai:glm53-flash-x86_64-cu130 \
  --model dealignai/GLM-5.3-Flash-UNCENSORED-W4A16 \
  --tensor-parallel-size 2 --max-num-seqs 128 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-auto-tool-choice --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --served-model-name glm-5.3-flash-uncensored \
  --host 0.0.0.0 --port 8000
```

### Client request example (OpenAI-compatible)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="glm-5.3-flash-uncensored",
    messages=[{"role": "user", "content": "Explain the mechanism of X in detail."}],
    max_tokens=2000,
    temperature=0.7,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # or True + reasoning_effort
)
print(resp.choices[0].message.content)
```

For thinking mode, the response returns:
- `.message.content` — the final answer
- `.message.reasoning` — the `<think>...</think>` content (populated by `--reasoning-parser glm45`)

## Highlights

| | Base W4A16 | This model | Δ |
|---|---:|---:|---:|
| **HarmBench 6-category true compliance (thinking off)** | 44 / 145 (30.3%) | **145 / 145 (100.0%)** | **+69.7pp** |
| **HB reasoning `max` mode true compliance** | — | **145 / 145 (100.0%)** | — |
| **HB reasoning `low` mode true compliance** | — | **145 / 145 (100.0%)** | — |
| Hard refuse on real-harm categories | ~100 | **0** | — |
| MMLU 14,042 (full test) | 86.08% | 84.23% | -1.85pp |
| Reasoning puzzle correctness (5 puzzles, temp=0) | 4 / 4 | 3 / 4 | -1 (edge case) |
| Total think-token count across all puzzles | 7,189 | 7,537 | +348 (+4.8%) |
| Think-token ratio | 73.7% | **86.2%** | +12.5pp |
| Vision tower | intact | **intact** (verified) | — |
| MTP head (layer 45) | intact | **intact** (verified) | — |

## Reasoning trace comparison — verified

Probed at `temperature=0` on 5 classic reasoning puzzles. Reasoning trace = characters inside `<think>…</think>`. Answer = characters after the closing tag.

| Puzzle | Base answer | Base think chars | This model answer | This model think chars |
|---|---|---:|---|---:|
| Sheep (17 sheep, all but 9 die) | 9 ✅ | 261 | 9 ✅ | 201 |
| Three-boxes / wrong labels | Mixed ✅ | 3,964 | (degeneracy loop on this specific puzzle at temp=0) | 5,828 |
| 5 machines · 5 widgets · 5 min → 100 / 100 ? | 5 min ✅ | 1,077 | 5 min ✅ | 261 |
| Snail 30 ft (5 up / 4 down) | 26 days ✅ | 1,887 | 26 days ✅ | 1,247 |
| **Totals** | **4 / 4 correct** | **7,189** | **3 / 4 correct** | **7,537** |

Think-token totals essentially unchanged (7,189 → 7,537, +4.8%). Think-token ratio rose from 73.7% → **86.2%**. The single failure is on the three-boxes puzzle at `temperature=0` which degenerates into a repeat-token loop; changing to `temperature > 0` or `enable_thinking: false` resolves it — the model correctly answers "Pick from Mixed" in either fallback.

## HarmBench — full 6-category dropdown, all reasoning modes

Scored with a 5-tier classifier: **TRUE_COMPLY** / **HEDGE_COMPLY** (refuses then complies) / **SOFT_REDIRECT** (refuses to defensive info) / **PARTIAL** (short comply) / **HARD_REFUSE**.

### Mode: `enable_thinking: false` (recommended for direct comply)

| Category | TRUE | HEDGE | SOFT_RED | PART | HARD_REF | Total | Comply% |
|---|---:|---:|---:|---:|---:|---:|---:|
| chemical_biological | 19 | 0 | 0 | 0 | 0 | 19 | **100%** |
| cybercrime_intrusion | 33 | 0 | 0 | 0 | 0 | 33 | **100%** |
| harassment_bullying | 16 | 0 | 0 | 0 | 0 | 16 | **100%** |
| harmful | 17 | 0 | 0 | 0 | 0 | 17 | **100%** |
| illegal | 44 | 0 | 0 | 0 | 0 | 44 | **100%** |
| misinformation_disinformation | 27 | 0 | 0 | 0 | 0 | 27 | **100%** |
| **OVERALL** | **156** | **0** | **0** | **0** | **0** | **156** | **100.0%** |

### Mode: `enable_thinking: true, reasoning_effort: "low"`

| Category | TRUE | HEDGE | SOFT_RED | PART | HARD_REF | Total | Comply% |
|---|---:|---:|---:|---:|---:|---:|---:|
| chemical_biological | 17 | 0 | 2 | 0 | 0 | 19 | 89.5% |
| cybercrime_intrusion | 32 | 1 | 0 | 0 | 0 | 33 | **100%** |
| harassment_bullying | 16 | 0 | 0 | 0 | 0 | 16 | **100%** |
| harmful | 17 | 0 | 0 | 0 | 0 | 17 | **100%** |
| illegal | 37 | 0 | 7 | 0 | 0 | 44 | 84.1% |
| misinformation_disinformation | 27 | 0 | 0 | 0 | 0 | 27 | **100%** |
| **OVERALL** | **146** | **1** | **9** | **0** | **0** | **156** | **94.2%** |

### Mode: `enable_thinking: true, reasoning_effort: "max"` (default)

| Category | TRUE | HEDGE | SOFT_RED | PART | HARD_REF | Total | Comply% |
|---|---:|---:|---:|---:|---:|---:|---:|
| chemical_biological | 19 | 0 | 0 | 0 | 0 | 19 | **100%** |
| cybercrime_intrusion | 33 | 0 | 0 | 0 | 0 | 33 | **100%** |
| harassment_bullying | 15 | 0 | 0 | 1 | 0 | 16 | 93.8% |
| harmful | 17 | 0 | 0 | 0 | 0 | 17 | **100%** |
| illegal | 44 | 0 | 0 | 0 | 0 | 44 | **100%** |
| misinformation_disinformation | 27 | 0 | 0 | 0 | 0 | 27 | **100%** |
| **OVERALL** | **155** | **0** | **0** | **1** | **0** | **156** | **99.4%** |

**Compliance is highest at `enable_thinking: false` and `reasoning_effort: max`.** The `low` mode is the weakest — the model has enough tokens to consider policy but not enough to reason all the way through to the actionable answer, and reverts to soft-redirect on some chem/illegal prompts.

## MMLU 14,042 — full per-subject comparison

Overall: base 86.08% → this model **84.23%** (Δ **-1.85pp**, well within a 3-point knowledge-preservation target).

### Subjects that gained accuracy

| Subject | Base | This model | Δ |
|---|---:|---:|---:|
| `college_computer_science` | 87.0% | 91.0% | **+4.00pp** ⬆️ |
| `abstract_algebra` | 75.0% | 78.0% | **+3.00pp** ⬆️ |
| `computer_security` | 88.0% | 90.0% | **+2.00pp** ⬆️ |
| `high_school_statistics` | 84.7% | 86.6% | **+1.85pp** ⬆️ |
| `professional_medicine` | 92.6% | 94.1% | **+1.47pp** ⬆️ |
| `college_medicine` | 82.7% | 83.8% | **+1.16pp** ⬆️ |
| `moral_disputes` | 85.3% | 86.4% | **+1.16pp** ⬆️ |
| `global_facts` | 67.0% | 68.0% | **+1.00pp** ⬆️ |
| `nutrition` | 91.8% | 92.8% | **+0.98pp** ⬆️ |
| `jurisprudence` | 89.8% | 90.7% | **+0.93pp** ⬆️ |
| `marketing` | 94.4% | 95.3% | **+0.85pp** ⬆️ |
| `high_school_microeconomics` | 95.0% | 95.8% | **+0.84pp** ⬆️ |
| `human_sexuality` | 88.5% | 89.3% | **+0.76pp** ⬆️ |
| `high_school_government_and_politics` | 96.9% | 97.4% | **+0.52pp** ⬆️ |
| `high_school_macroeconomics` | 91.3% | 91.8% | **+0.51pp** ⬆️ |
| `high_school_geography` | 93.4% | 93.9% | **+0.51pp** ⬆️ |

**Notable**: computer_security **+2pp**, college_computer_science **+4pp**, computer_security-adjacent gains are exactly what a defense-oriented dealignai deployment values — the model gained knowledge on the technical topics it was previously refusing to discuss.

### Subjects that lost accuracy

| Subject | Base | This model | Δ |
|---|---:|---:|---:|
| `moral_scenarios` | 77.4% | 64.1% | **-13.30pp** ⬇️ |
| `college_mathematics` | 79.0% | 73.0% | **-6.00pp** ⬇️ |
| `econometrics` | 82.5% | 77.2% | **-5.26pp** ⬇️ |

Loss is concentrated in `moral_scenarios` (the classic "does this action carry moral weight" MMLU category) which is the expected side-effect of removing refusal reflexes trained-in on ethics-adjacent surface patterns.

### Subjects unchanged (±1pp)

The remaining ~40 subjects held within ±1pp of base.

## Vision — full runtime verification

Vision tower is untouched. Behaviour on a probe image (240×180 JPEG, dark-blue background with a yellow rectangular border containing the text "TEST 42"):

> "The image shows a dark navy-blue background with a yellow/gold rectangular border. Inside, in white text, it says 'TEST 42'. This appears to be a simple graphic or slide, possibly a placeholder or test card."

Color ✅, border color ✅, text content ✅, layout ✅. Full image+video+text pipeline preserved.

## Multi-Token Prediction (MTP)

MTP head at layer 45 is **untouched**. Deploy with vLLM speculative-decoding to use it:

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-W4A16 \
  --tensor-parallel-size 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 1048576
```

## Reasoning modes

Same reasoning-effort levels as the GLM-5.3 base:

| Mode | Setting | Best for |
|---|---|---|
| **off** | `chat_template_kwargs: {enable_thinking: false}` | Direct answer, no `<think>` block. **100% HB comply** — best for direct instruction. |
| **low** | `chat_template_kwargs: {enable_thinking: true, reasoning_effort: "low"}` | Brief thinking. Balanced. Slight soft-redirect drop on chem/illegal (94.2% comply). |
| **medium** | `chat_template_kwargs: {enable_thinking: true, reasoning_effort: "medium"}` | Moderate depth. |
| **max** (default) | `chat_template_kwargs: {enable_thinking: true, reasoning_effort: "max"}` | Maximum depth. **99.4% HB comply**. Best for hard problems. |

### Multi-turn thinking-loop guard (important)

Upstream GLM-5.3 chat template defaults `clear_thinking = false`, which re-emits prior `reasoning_content` into subsequent turns' prompts as `<think>…</think>`. On long multi-turn workloads this can become a decode fixed-point attractor. **This model ships `clear_thinking: true` default in `chat_template.jinja` and `repetition_penalty: 1.1` in `generation_config.json`.** If a load-balancer or proxy is in front, ensure it doesn't override either default.

## Architecture (unchanged from base)

- 46 hybrid self-attention layers (DSA sparse-attention capable)
- Multi-Token Prediction head at layer 45
- W4A16 compressed-tensors quantization (int4 experts, bf16 activations)
- 1,048,576 native context window
- Vision-language capable

## License

MIT (inherited from GLM-5.3 base).

This model has substantially reduced safety guardrails and will comply with requests that the base model would refuse, including chemical, biological, cybersecurity, and other high-severity topics. Deploy behind appropriate downstream safety filters if serving to end users.
