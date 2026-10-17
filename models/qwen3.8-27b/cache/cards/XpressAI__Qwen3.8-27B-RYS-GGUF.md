---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - rys
  - layer-surgery
  - reasoning
  - bfcl
  - function-calling
language:
  - en
---

# Qwen3.8-27B — RYS Layer Surgery (GGUF)

An **experimental** modification of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced by RYS layer
duplication: no training and no weight changes, just running language layers
16–19 (zero-based) a second time during the forward pass.

The source quant is
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`. The method is based on
[David Ng's RYS work](https://dnhkng.github.io/posts/rys/).

## TL;DR

This variant is intended for **low-effort reasoning mode**. It improved a small
reasoning probe by **+11.76 percentage points** and a fixed 1,006-case BFCL v4
single-turn evaluation by **+0.60 pp micro accuracy**. The BFCL gain is small
and not statistically decisive (36 RYS-only wins versus 30 base-only wins,
two-sided exact McNemar p = 0.539), so treat this as an experimental rather
than universally superior release.

The larger EQ-140 result was effectively flat-to-slightly-down (−0.99). Math
also regressed in the search probe; math was recorded as a diagnostic, not used
as a release gate, because it is an expected and fine-tunable tradeoff here.

| Evaluation | Base | RYS 16–19 | Change |
|---|---:|---:|---:|
| Reasoning probe (17 prompts) | 58.82% | **70.59%** | **+11.76 pp** |
| EQ-140 (139 scored) | **82.65** | 81.66 | −0.99 |
| BFCL micro accuracy (1,006 cases) | 82.60% | **83.20%** | **+0.60 pp** |
| BFCL category macro mean | 80.54% | **81.81%** | **+1.28 pp** |
| Math probe (diagnostic only) | **1.000** | 0.750 | −0.250 |

## File

| File | Quant | Language layers | Size | SHA-256 |
|---|---|---:|---:|---|
| `Qwen3.8-27B-rys_16-19-UD-Q4_K_XL.gguf` | UD-Q4_K_XL | 68 | 18.9 GB | `a3013e582490a386d4b8fafbaad07bfec65e7727fb5dcf992914babdc2d65ed4` |

The terminal MTP block from the source GGUF is retained and shifted after the
four inserted language layers.

## BFCL results

The evaluation used a fixed 1,006-case manifest spanning 13 BFCL v4
single-turn categories. Categories with at least 100 entries used 100 cases;
smaller categories used all available entries. This is not the current full
3,641-case BFCL suite and excludes multi-turn, memory, and web-search tasks.

| Category | N | Base | RYS 16–19 | Change |
|---|---:|---:|---:|---:|
| irrelevance | 100 | 86.00% | **88.00%** | +2.00 |
| multiple | 100 | **94.00%** | 91.00% | −3.00 |
| parallel | 100 | 90.00% | **92.00%** | +2.00 |
| parallel_multiple | 100 | 82.00% | **84.00%** | +2.00 |
| simple_java | 100 | **54.00%** | 53.00% | −1.00 |
| simple_javascript | 50 | **64.00%** | 62.00% | −2.00 |
| simple_python | 100 | 94.00% | 94.00% | 0.00 |
| live_irrelevance | 100 | 95.00% | **99.00%** | +4.00 |
| live_multiple | 100 | **79.00%** | 77.00% | −2.00 |
| live_parallel | 16 | 87.50% | 87.50% | 0.00 |
| live_parallel_multiple | 24 | 62.50% | **70.83%** | +8.33 |
| live_relevance | 16 | 75.00% | **81.25%** | +6.25 |
| live_simple | 100 | 84.00% | 84.00% | 0.00 |
| **Micro accuracy** | **1,006** | **82.60%** | **83.20%** | **+0.60** |
| **Macro mean** | **13 categories** | **80.54%** | **81.81%** | **+1.28** |

Inference used llama.cpp's OpenAI-compatible chat-completions endpoint with
native tool use, the embedded Unsloth chat template, four slots, Q8 KV cache,
and these matched settings for both models:

```text
reasoning_effort = low
enable_thinking  = true
preserve_thinking = true
temperature = 1.0
top_p       = 0.95
top_k       = 20
min_p       = 0.0
seed        = 3407
max_tokens  = 2048
```

The RYS pass generated in 59:16. Normalizing the clean 991-case portion of the
resumed base pass to 1,006 cases gives about 59:41, so wall time was effectively
flat in this run. RYS produced about 2.0% more output tokens.

## What changed?

Qwen3.8-27B uses a repeating three-DeltaNet/one-full-attention hybrid. RYS
insertions therefore need to preserve four-layer blocks. This model duplicates
the half-open layer window `[16, 20)`:

```text
Base:  0 → … → 15 → 16 → 17 → 18 → 19 → 20 → … → 63 → MTP
RYS:   0 → … → 15 → 16 → 17 → 18 → 19
                     → 16 → 17 → 18 → 19 → 20 → … → 63 → MTP
```

The sweep covered every compatible four-layer window at stride four, followed
by neighboring starts around the strongest reasoning regions. Layers 16–19
were selected because they gave the best reasoning-first result while keeping
the small EQ change near zero. The full BFCL result points in the same positive
direction, though modestly.

## Usage

Use a recent llama.cpp build. Start the server with thinking support and the
embedded Jinja template:

```bash
./build/bin/llama-server \
  -m Qwen3.8-27B-rys_16-19-UD-Q4_K_XL.gguf \
  --port 8081 -np 4 -ngl all \
  -b 2048 -ub 1024 --flash-attn on \
  --jinja --no-host --reasoning on --reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

Request low-effort reasoning explicitly:

```json
{
  "model": "qwen3.8-27b-rys",
  "messages": [{"role": "user", "content": "Your prompt"}],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "low",
    "preserve_thinking": true
  }
}
```

The embedded Unsloth template already supports low-effort and reasoning-off
modes. A quick comparison with `froggeric/Qwen-Fixed-Chat-Templates` v22 found
no material reasoning-quality difference, so this GGUF retains the source
template.

## When to use this

- You run Qwen3.8 with low-effort reasoning enabled.
- You value the directional gains on parallel/relevance tool-use tasks.
- You are comfortable evaluating an experimental layer-surgery variant for
  your own workload.

## When not to use this

- You need a proven across-the-board improvement; the aggregate BFCL gain is
  small and not statistically decisive.
- EQ or math performance must not regress at all.
- Your workload is dominated by `multiple`, `live_multiple`, or simple
  JavaScript function calling, where the base model did slightly better here.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for Qwen3.8-27B
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) for the source
  UD-Q4_K_XL quant
- [David Ng](https://dnhkng.github.io/posts/rys/) for the RYS method
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference
- [Berkeley Function-Call Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
  for the evaluation harness

## License

Apache 2.0, inherited from Qwen3.8-27B and the source GGUF.
