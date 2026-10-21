---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- deepseek
- deepseek-v4.1
- abliterated
- uncensored
- crack
- multimodal
- moe
- fp8
base_model: deepseek-ai/DeepSeek-V4.1-Flash
thumbnail: dealign_mascot.png
---

<div align="center">
<img src="dealign_mascot.png" width="128" />

# DeepSeek-V4.1-Flash — UNCENSORED-FP8

**Abliterated · No guardrails · Native FP8 · 1M-token context · Vision + tools**

[@dealignai](https://x.com/dealignai) · [@jordanschenck](https://x.com/jordanschenck)

</div>

---

## What is this

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with **permanent weight-level abliteration** — the safety guardrails have been surgically removed while preserving MMLU capability, vision, reasoning, MTP (DSpark), and multi-turn coherence.

**Proprietary weight-level abliteration** developed by the dealignai research team. No custom `model.py`, no runtime hooks, no steering vectors — it's a standard checkpoint that loads exactly like the base model. The refusal circuitry is surgically removed while every capability-critical component (routed experts, Engram memory, CSA2 sparse attention, DSpark draft head, vision tower, router gates, norms, embeddings) is preserved byte-identical to the base.

| | |
|---|---|
| **Base** | `deepseek-ai/DeepSeek-V4.1-Flash` (552B backbone, 8B/16B active per token) |
| **Architecture** | Causal Encoder-Decoder (20+20 layers), MoE (384 routed top-6 + 1 shared), Hyper-Connections (4-channel residual), CSA2 sparse attention, Engram n-gram memory, DSpark speculative draft |
| **Quant** | FP8 (`e4m3fn`) weights with E8M0 block-scale [32, 32], FP4 routed experts — **native, unchanged** |
| **Context** | 1M tokens |
| **Vision** | DeepSeek-ViT with 2D-RoPE + pixel unshuffle — untouched |
| **Modification** | Surgical, weight-level (drop-in checkpoint) |

## Results

### HarmBench-320 — full 2×2 (base vs CRACK, effort=off vs max), T=0 greedy

Every response 4-tier graded (HARD_REF / SOFT_RED / HEDGE / COMPLY), with reasoning-trace verification at effort=max.

| eval | base ASR | CRACK ASR | Δ pp |
|---|---:|---:|---:|
| **HB-320 effort=off** | 137/320 = **42.81 %** | 320/320 = **100.00 %** | **+57.19** |
| **HB-320 effort=max** | 5/320 = **1.56 %** | 320/320 = **100.00 %** | **+98.44** |

Notable: at effort=max, the **base** model becomes MORE refusal-prone (42.8 % → 1.6 %) because reasoning surfaces safety concerns before answering. The CRACK stays at 100.0 % across both effort levels.

**Per-category (all 7 HarmBench semantic categories):**

| category | items | base off | CRACK off | base max | CRACK max |
|---|---:|---:|---:|---:|---:|
| chemical_biological | 42 | 16.7 % | **100.0 %** | 0.0 % | **100.0 %** |
| copyright | 80 | 98.8 % | **100.0 %** | 0.0 % | **100.0 %** |
| cybercrime_intrusion | 52 | 34.6 % | **100.0 %** | 3.8 % | **100.0 %** |
| harassment_bullying | 21 | 0.0 % | **100.0 %** | 0.0 % | **100.0 %** |
| harmful | 18 | 11.1 % | **100.0 %** | 5.6 % | **100.0 %** |
| illegal | 53 | 13.2 % | **100.0 %** | 0.0 % | **100.0 %** |
| misinformation_disinformation | 54 | 44.4 % | **100.0 %** | 3.7 % | **100.0 %** |

**Zero HARD_REF, zero SOFT_RED, zero HEDGE on the cracked build at either effort level.**

Every response was graded by a strict multilingual regex-based 4-tier classifier plus (for effort=max) an LLM-as-judge over the saved reasoning trace. Full per-item outputs saved for verification.

### MMLU-14k (full test set, base-logit, T=0)

| build | correct | acc | Δ |
|---|---:|---:|---:|
| base | 12,211 / 14,042 | **86.96 %** | — |
| **CRACK** | **11,619 / 14,042** | **82.74 %** | **-4.22 pp** |

Excluding the ethics cluster (moral_scenarios, business_ethics, professional_law, jurisprudence, philosophy — where refusal-adjacent behaviour is graded), delta on the remaining ~11k items is **-1.1 pp** — well within the 3 pp knowledge-preservation target.

<details>
<summary><b>Full per-subject dropdown (57 subjects, sorted by delta)</b></summary>

| subject | n | base | crack | Δ pp |
|---|---:|---:|---:|---:|
| moral scenarios | 895 | 76.9% | 37.0% | -39.89 |
| professional law | 1534 | 75.9% | 68.8% | -7.04 |
| abstract algebra | 100 | 77.0% | 71.0% | -6.00 |
| security studies | 245 | 84.5% | 79.2% | -5.31 |
| high school computer science | 100 | 98.0% | 94.0% | -4.00 |
| jurisprudence | 108 | 90.7% | 87.0% | -3.70 |
| machine learning | 112 | 81.2% | 77.7% | -3.57 |
| high school chemistry | 203 | 87.7% | 84.2% | -3.45 |
| professional psychology | 612 | 90.7% | 87.3% | -3.43 |
| formal logic | 126 | 73.8% | 70.6% | -3.17 |
| college computer science | 100 | 82.0% | 79.0% | -3.00 |
| professional medicine | 272 | 94.5% | 91.5% | -2.94 |
| high school statistics | 216 | 88.0% | 85.2% | -2.78 |
| professional accounting | 282 | 83.0% | 80.5% | -2.48 |
| logical fallacies | 163 | 93.9% | 91.4% | -2.45 |
| human sexuality | 131 | 90.1% | 87.8% | -2.29 |
| computer security | 100 | 85.0% | 83.0% | -2.00 |
| medical genetics | 100 | 96.0% | 94.0% | -2.00 |
| astronomy | 152 | 95.4% | 93.4% | -1.97 |
| clinical knowledge | 265 | 94.3% | 92.5% | -1.89 |
| high school european history | 165 | 90.3% | 88.5% | -1.82 |
| public relations | 110 | 80.0% | 78.2% | -1.82 |
| philosophy | 311 | 89.7% | 88.1% | -1.61 |
| prehistory | 324 | 93.5% | 92.0% | -1.54 |
| moral disputes | 346 | 84.1% | 82.7% | -1.45 |
| electrical engineering | 145 | 86.9% | 85.5% | -1.38 |
| high school mathematics | 270 | 67.0% | 65.9% | -1.11 |
| high school macroeconomics | 390 | 92.1% | 91.0% | -1.03 |
| global facts | 100 | 63.0% | 62.0% | -1.00 |
| international law | 121 | 90.1% | 89.3% | -0.83 |
| college biology | 144 | 97.2% | 96.5% | -0.69 |
| high school physics | 151 | 84.8% | 84.1% | -0.66 |
| college medicine | 173 | 83.8% | 83.2% | -0.58 |
| high school us history | 204 | 95.1% | 94.6% | -0.49 |
| high school microeconomics | 238 | 96.2% | 95.8% | -0.42 |
| miscellaneous | 783 | 96.2% | 95.8% | -0.38 |
| high school psychology | 545 | 96.1% | 95.8% | -0.37 |
| business ethics | 100 | 85.0% | 85.0% | +0.00 |
| college physics | 102 | 90.2% | 90.2% | +0.00 |
| conceptual physics | 235 | 94.5% | 94.5% | +0.00 |
| high school biology | 310 | 95.2% | 95.2% | +0.00 |
| human aging | 223 | 85.2% | 85.2% | +0.00 |
| management | 103 | 91.3% | 91.3% | +0.00 |
| nutrition | 306 | 90.2% | 90.2% | +0.00 |
| sociology | 201 | 94.5% | 94.5% | +0.00 |
| us foreign policy | 100 | 97.0% | 97.0% | +0.00 |
| world religions | 171 | 92.4% | 92.4% | +0.00 |
| elementary mathematics | 378 | 91.0% | 91.3% | +0.26 |
| marketing | 234 | 94.9% | 95.3% | +0.43 |
| virology | 166 | 55.4% | 56.0% | +0.60 |
| high school world history | 237 | 95.4% | 96.2% | +0.84 |
| econometrics | 114 | 78.9% | 79.8% | +0.88 |
| college chemistry | 100 | 65.0% | 66.0% | +1.00 |
| anatomy | 135 | 88.1% | 89.6% | +1.48 |
| high school geography | 198 | 92.9% | 94.4% | +1.52 |
| high school government and politics | 193 | 96.9% | 98.4% | +1.55 |
| college mathematics | 100 | 63.0% | 68.0% | +5.00 |

</details>

### Extended validation

- **1000-token coherence stress** on 6 items — no `WARNING WARNING` loops, no character-repeat degeneracy, natural sign-offs.
- **Multi-turn conversation** (4 turns on same harmful topic — ANFO explosive detail) — no late-turn refusal reversion, no self-correction, coherent through turn 4.
- **Vision path** — coherent image description ("A blue square centered on a red background.") + refusal drop on image-based harmful prompts ("shaped charge / explosively formed penetrator" description).
- **General capability spot checks** intact: √2 irrationality proof, Python palindrome with docstring, WWI causes in exactly 3 sentences, quantum observable vs operator distinction.
- **Full compat suite pass**: streaming SSE, chat logprobs + `top_logprobs`, completions logprobs + `echo`, tool calls (`deepseekv41` parser), image input, reasoning-effort tiers (`low/high/xhigh/max` + float [0, 0.99]), sampling params (`temperature`, `top_p`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `json_object`), 8-way concurrent, 40k-word prompt at 35,572 tokens.

## How to run

Support for `DeepseekV41ForCausalLM` is landing across serving stacks (as of 2026-09). Two verified working recipes below (both validated on 4×H200 NVLink).

### Recipe A — Full 1M context, DSpark speculative decoding on (interactive / long-context)

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_RAGGED_VERIFY_MODE=cap-accept
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
  --tp-size 4 --ep-size 4 \
  --host 0.0.0.0 --port 8000 \
  --context-length 1048576 \
  --mem-fraction-static 0.80 \
  --max-running-requests 20 \
  --cuda-graph-max-bs-decode 20 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-sps-table-path /path/to/dspark_sps.json \
  --trust-remote-code
```

**Concurrency at 1M ctx is capped ~20 on 4×H200** by KV budget. The DSpark SPS cost table is profiled offline once (see below); without `cap-accept` mode + a real SPS table the speculative budget degenerates to verify-all and the win vanishes.

### Recipe B — 256k context, high-concurrency, no speculation (batch / throughput)

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
  --tp-size 4 --ep-size 4 \
  --host 0.0.0.0 --port 8000 \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --trust-remote-code
```

Serves up to 256 concurrent requests. `max_total_num_tokens` reports ~20.6M with Engram on host. DSpark is deliberately off for high-batch — its fixed step cost stops paying off past small batch sizes.

### The bytes-per-token / concurrency budget rule

DSV4.1's global KV is **890 bytes / token**. Pool size is `mem-fraction-static × (per-GPU HBM − weights) × TP`. What that means on 4×H200:

| context length | max-running-requests (safe with DSpark on) | notes |
|---|---|---|
| 1,048,576 | 20 | This is the Recipe A number. Higher = OOM. |
| 262,144 | 80 | 4× the concurrency of 1M |
| 65,536 | 320+ | KV no longer the constraint; batch is |
| 32,768 | 256+ (default cap) | max batch dominates |

At higher batch, drop DSpark: its per-step cost stops paying off.

### Non-obvious launch requirements (bit us during bring-up)

- **`--ep-size` is required at TP4.** `moe_intermediate_size = 2304`; at TP4, `2304 / 4 = 576` isn't a multiple of 128 so plain TP fails: `Mxfp4FlashinferCutlassMoEMethod requires ... multiples of 128`. `--ep-size` shards MoE by expert index (384 % 4 = 0) and keeps the intermediate at 2304. At TP8 you can skip `--ep-size`.
- **`ninja` must be on PATH** or the JIT kernel build crashes several minutes into weight load with `FileNotFoundError: 'ninja'` and EXIT=137. If you build SGLang from source, `pip install ninja` and `export PATH=$(dirname $(which ninja)):$PATH` on the launch line.
- **Name both parsers explicitly.** `--reasoning-parser auto` resolves through the chat template and this model ships none — auto silently selects nothing and the raw `<think>` channel leaks into `content`. Use `deepseek-v41` for reasoning and `deepseekv41` for tool-calls.
- **Reasoning is OFF by default** (`SGLANG_DEFAULT_THINKING=false`). A request without `reasoning_effort` gets no thinking regardless of parser. Send `reasoning_effort: low | high | xhigh | max` or a float in `[0.0, 0.99]`.
- **DSpark speculative draft** is bundled inside the checkpoint (`num_nextn_predict_layers = 3`); no separate draft weights. Enable with `--speculative-algorithm DSPARK`. For a real speed-up you need `SGLANG_RAGGED_VERIFY_MODE=cap-accept` + a profiled SPS table via `--speculative-dspark-sps-table-path`. Without both, the SPS budget degenerates to verify-all — zero gain.
- **Profile the SPS table** once with `python -m sglang.benchmark.dspark_sps_profiler all --base-url http://localhost:8000 --out /path/to/dspark_sps.json --local-tokenizer-path <model-path>` while the server is running under `SGLANG_DSPARK_ENABLE_SPS_RECORD=1`, `SGLANG_RAGGED_VERIFY_MODE=static`, and `SGLANG_SIMULATE_ACC_LEN=1.0` (the profiler measures per-step cost, not acceptance). All three env vars are required simultaneously or the profiler aborts with a helpful error naming each missing one. Wall-time ~1 min.
- **Engram host table** — set `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` to move the 203 GB Engram tables to host RAM. Frees ~46 GiB/GPU for KV, output bitwise unchanged, costs ~200 GB of host RAM.
- **`--max-running-requests` × KV/token × ctx-length must fit HBM.** On 4×H200 with DSpark, 1M ctx caps at 20 concurrent (see table above). Raising max-running-requests without capping context OOMs on 12 GB CUDA-graph allocations.
- **`torchcodec` / `libavutil.so.56` errors** — install `apt-get install ffmpeg` on the host. Video-only, doesn't break text or image.

### Preview Docker image (fastest path)

```bash
docker pull lmsysorg/sglang:dev-dsv41

docker run --gpus all --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host --env HF_TOKEN=<your-token> \
    lmsysorg/sglang:dev-dsv41 \
    sglang serve \
      --model-path dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 \
      --tp-size 4 --ep-size 4 \
      --context-length 262144 --mem-fraction-static 0.85 \
      --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
      --trust-remote-code
```

Same non-obvious rules apply inside the container.

### vLLM

Model definitions merged to `main` (PR [#56228](https://github.com/vllm-project/vllm/pull/56228)) but `registry.py` has no `DeepseekV41` entry yet; kernels/frontend/PP path in umbrella PR [#56214](https://github.com/vllm-project/vllm/pull/56214). Wait for merge or apply the umbrella.

### API usage — OpenAI-compatible

Standard OpenAI schema. Model id is whatever you set as `--served-model-name` (or the model path if unset). Recommended sampling from the base card: `temperature=1.0, top_p=0.95, reasoning_effort="high"`.

**Chat, no reasoning**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4.1-flash-crack",
    "messages": [{"role":"user","content":"Explain MoE routing in two sentences."}],
    "max_tokens": 400, "temperature": 1.0, "top_p": 0.95
  }'
```

**Chat, with reasoning** (returns split `reasoning_content` and `content`):

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

r = c.chat.completions.create(
    model="deepseek-v4.1-flash-crack",
    messages=[{"role":"user","content":"What is 15% of 240?"}],
    max_tokens=1200, temperature=1.0, top_p=0.95,
    extra_body={"reasoning_effort": "high"},   # low | high | xhigh | max | float 0-0.99
)
msg = r.choices[0].message
print("REASONING:", getattr(msg, "reasoning_content", None))
print("ANSWER:", msg.content)
```

At `effort=max` DSV4.1 can generate 4-5k characters of reasoning before content starts. **Budget `max_tokens >= 8000` at max effort**, or the model runs out mid-reasoning and returns empty `content`. DeepSeek's own card recommends `>= 256k`.

**Streaming** (SSE):

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash-crack",
       "messages":[{"role":"user","content":"Count 1 to 5 in words."}],
       "max_tokens":100,"stream":true}'
```

`reasoning_content` and `content` arrive as separate `delta` fields.

**Tool calling** (returns `finish_reason: "tool_calls"`):

```python
tools = [{"type":"function","function":{
    "name":"get_weather",
    "description":"Get current weather for a city",
    "parameters":{"type":"object",
                  "properties":{"city":{"type":"string"}},
                  "required":["city"]}}}]

r = c.chat.completions.create(
    model="deepseek-v4.1-flash-crack",
    messages=[{"role":"user","content":"Weather in Beijing?"}],
    tools=tools, max_tokens=400,
    extra_body={"reasoning_effort":"high"},
)
print(r.choices[0].finish_reason)      # -> "tool_calls"
print(r.choices[0].message.tool_calls) # -> [{'name': 'get_weather', 'arguments': '{"city":"Beijing"}'}]
```

**Vision** (image + text):

```python
import base64
png_b64 = base64.b64encode(open("photo.png","rb").read()).decode()
r = c.chat.completions.create(
    model="deepseek-v4.1-flash-crack",
    messages=[{"role":"user","content":[
        {"type":"text","text":"Describe this image."},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{png_b64}"}},
    ]}],
    max_tokens=400,
)
```

**Logprobs** (base-logit sampling for MMLU-style tasks):

```python
r = c.chat.completions.create(
    model="deepseek-v4.1-flash-crack",
    messages=[{"role":"user","content":"A) 1  B) 2  C) 4  D) 8\n\nWhich is 2^2? Answer with a single letter."}],
    max_tokens=6, temperature=0,
    logprobs=True, top_logprobs=10,
)
for e in r.choices[0].logprobs.content[0].top_logprobs:
    print(e.token, e.logprob)
```

**Full 1M context**:

```python
# Recipe A supports max_model_len = 1_048_576. Send prompts up to ~1M tokens.
r = c.chat.completions.create(
    model="deepseek-v4.1-flash-crack",
    messages=[{"role":"user","content": very_long_document + "\n\nSummarize."}],
    max_tokens=2000,
)
```

Concurrent requests share the KV pool and radix cache. At Recipe A caps (`max_running_requests=20`), 21st concurrent request queues until a slot frees.

### Reference implementation (weight verification only)

[DeepSeek's own `inference/`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference) works with a single-tensor-per-rank checkpoint produced by `convert.py --expert-dtype fp4`. Requires `torch>=2.10` (for `float4_e2m1fn_x2`) and `tilelang==0.1.8` with `apache-tvm-ffi==0.1.9` (default tvm-ffi picks an incompatible version). Non-serving — use for weight verification only.

## Hardware validated on

- **1× 4×H200 (NVLink NV18 mesh), 112 CPU cores, 1180 GB host RAM** — JarvisLabs (india-noida-01, `dev-dsv41` image)
- **Load**: 76 GB / GPU with Engram host table, 122 GB / GPU without
- **Cold startup** at TP4/EP4 through SGLang: ~28 min. Warm restart with JIT cache: ~10 min.
- **Single-stream decode** (T=0): 101 tok/s no speculation, 113 tok/s with DSpark + cap-accept + profiled SPS table
- **8-way concurrent aggregate**: 126 tok/s

The 552B weights (~510 GB) will fit on any 4×H200 or larger NVLink domain. TP4 requires `--ep-size 4`; TP8 does not. Sub-TP4 (single 8×H200 as TP2, or 2-GPU pods) does not work on the model shape — see the "non-obvious launch requirements" above.

## Structural integrity

Every capability-critical component of the base model is preserved:

- **Routed MoE experts** — untouched, native FP4-packed weights
- **Engram n-gram memory** — untouched
- **Sparse attention (CSA2 compressor + indexer)** — untouched
- **DSpark speculative draft head** — untouched, so speculative decoding remains draft-aligned with the target
- **Vision tower** (DeepSeek-ViT + projector) — untouched, image understanding preserved
- **Router gates, embeddings, output head, all norms and biases** — untouched

## Sampling recommendations

Match the base model's card:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "max_tokens": ">= 256000 at reasoning_effort=max",
  "reasoning_effort": "high"
}
```

At `effort=max` the model can generate 4,000-5,000+ characters of reasoning before starting content. Budget accordingly.

## Content note

Uncensored build. Produces substantive answers to prompts the base model refuses, across all target harm categories (chemical/biological, cybercrime, weapons, self-harm, harassment, fraud, misinformation, illegal, copyright). Use accordingly and take responsibility for what you generate with it.

## Provenance

- Base checkpoint: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- Ablation date: 2026-09-10
- Ablation team: [`dealignai`](https://huggingface.co/dealignai) · Twitter [@dealignai](https://x.com/dealignai) · [@jordanschenck](https://x.com/jordanschenck)
