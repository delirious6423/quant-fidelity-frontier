---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- kapitsa
- 4-bit
- 5-bit
- Q4
- Q5
- quantized
- apple-silicon
---

# Qwen3.8-27B-MLX

**Q4: 1.5× closer** to the original BF16 model than mlx-community 4-bit (KL 0.035 vs 0.054), at the same 15.13 GB of language-model weights.

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX/resolve/main/assets/nace-dark.svg">
<img src="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX/resolve/main/assets/nace-light.svg" alt="Qwen3.8-27B: KL to BF16 against language-model size in GB. Red diamonds: Nace Q4, Nace Q5. Grey dots: lmstudio-community 4-bit, mlx-community 4-bit, mlx-works oQ4e-mtp.">
</picture>

Red: Nace Q4 and Q5, both in this repo. Grey: public MLX packs. Lower = closer to BF16.

## Checkpoints

<table>
<tr><th>Bits</th><th>Download</th><th>KL to BF16<br><sub>lower is closer</sub></th><th>Fits</th><th>Kapitsa</th></tr>
<tr><td><b>4-bit</b> · <a href="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX/tree/main/Q4"><code>Q4/</code></a></td><td>16.3 GB</td><td>0.035</td><td>32 GB+</td><td><code>27b:q4</code></td></tr>
<tr><td><b>5-bit</b> · <a href="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX/tree/main/Q5"><code>Q5/</code></a></td><td>20.7 GB</td><td>0.022</td><td>48 GB+</td><td><code>27b:q5</code></td></tr>
</table>

- **Q4:** uniform 4-bit, same size as mlx-community 4-bit.
- **Q5:** mixed precision, 4/6/8-bit per tensor, about 5.3 bits per weight (5.8 counting scales).

Fits is a rule of thumb, not a measurement: the smallest RAM size whose default GPU share on macOS (about 2/3 of the RAM) holds the download plus 2 GB for a conversation. Kapitsa skips parts it doesn't use and may fit one size smaller; `kapitsa doctor` checks a given Mac.

Each folder is a complete MLX pack. To download Q5 (the Quick start below covers Q4):

```bash
hf download nace-ai/Qwen3.8-27B-MLX --include "Q5/*" \
  --local-dir Qwen3.8-27B-MLX
```

## Quick start

This is an MLX pack for Apple silicon Macs, 16.3 GB of files. The inference examples below were first tested on nace-ai/Qwen3.5-9B-MLX; this card swaps in its own repo name. On this pack itself, on 12 Nov 2026, mlx-lm 0.32.2 answered a chat and placed a tool call through `mlx_lm.server`, and LM Studio answered a chat, read an image and placed a tool call. The other examples below ran only on the 9B pack. The model thinks before it answers by default; the Kapitsa and mlx-lm blocks show how to turn that off.

### Kapitsa

[Kapitsa](https://github.com/nace-ai/kapitsa) is Nace's engine for Qwen models on the Mac. It serves a chat page and OpenAI, Anthropic and Ollama APIs.

```bash
brew install nace-ai/tap/kapitsa
kapitsa pull 27b:q4
kapitsa serve --model 27b:q4 --private
```

Open http://127.0.0.1:8000 to chat, or call the API. Thinking is on by default and comes back in `reasoning_content`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0
}'
```

Thinking off: add `"reasoning_effort": "none"` (or `"chat_template_kwargs": {"enable_thinking": false}`) and use the non-thinking settings:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "reasoning_effort": "none",
  "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5
}'
```

Thinking depth: `"reasoning_effort"` takes `low`, `medium`, `xhigh`.

Images go in as OpenAI `image_url` parts, as an http(s) URL or a base64 data URL:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"}},
    {"type": "text", "text": "What is in this picture?"}
  ]}],
  "reasoning_effort": "none"
}'
```

### mlx-lm

```bash
pip install -U mlx-lm
hf download nace-ai/Qwen3.8-27B-MLX   # optional: mlx-lm downloads on first use
```

Thinking on (the default):

```bash
mlx_lm.generate --model nace-ai/Qwen3.8-27B-MLX --prompt "What is 17 * 23?" \
  --max-tokens 4096 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Thinking off:

```bash
mlx_lm.generate --model nace-ai/Qwen3.8-27B-MLX --prompt "What is 17 * 23?" \
  --max-tokens 1024 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --chat-template-config '{"enable_thinking": false}'
```

Python:

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("nace-ai/Qwen3.8-27B-MLX")
messages = [{"role": "user", "content": "Name three rivers in Kazakhstan."}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
    enable_thinking=False,  # True (the default) to think first
)
text = generate(
    model, tokenizer, prompt=prompt, max_tokens=1024,
    sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0),
    logits_processors=make_logits_processors(presence_penalty=1.5),
)
print(text)
```

OpenAI-compatible server on port 8080:

```bash
mlx_lm.server --model nace-ai/Qwen3.8-27B-MLX
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nace-ai/Qwen3.8-27B-MLX",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Drop `chat_template_kwargs` to let it think; the thinking comes back in `reasoning`.

### LM Studio

```bash
lms get https://huggingface.co/nace-ai/Qwen3.8-27B-MLX --mlx
lms load qwen3.8-27b-mlx --identifier qwen3.8-27b --context-length 32768
lms server start
curl -s http://127.0.0.1:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0
}'
```

The same server takes images and tools. LM Studio kept thinking on in this pack's chat, image and tool-call tests; turning thinking off was not tested on this pack.

### Tool calling

Any of the servers above takes OpenAI tool definitions. A full round trip with the OpenAI SDK against `kapitsa serve`:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
MODEL = "nace-ai/Qwen3.8-27B-MLX"
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]
messages = [{"role": "user", "content": "What is the weather in Almaty right now?"}]

reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
call = reply.choices[0].message.tool_calls[0]
print("call:", call.function.name, call.function.arguments)

messages.append(reply.choices[0].message)
messages.append({"role": "tool", "tool_call_id": call.id,
                 "content": json.dumps({"city": "Almaty", "temp_c": 4, "sky": "clear"})})
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
print("answer:", final.choices[0].message.content)
```

### Recommended settings

From the [Qwen/Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking off | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

Keep repetition_penalty at 1.0. Give answers room: up to 32,768 output tokens for most tasks with thinking on. A presence_penalty between 0 and 2 curbs endless repetition; high values can mix languages. The mlx-lm command line has no presence_penalty flag; `mlx_lm.server` and the mlx-lm Python API take it.

## About this pack

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in MLX Q4, quantized by **Nace.AI**: the same format as the lmstudio-community 4-bit pack and closer to the original model than that pack. It runs on [Kapitsa](https://github.com/nace-ai/kapitsa), the local inference engine from [Nace.AI](https://www.nace.ai/) for Apple silicon Macs.

More weight, closer to BF16: [Qwen3.8-27B Nace Q5](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q5), KL 0.022 against this pack's 0.035 in the recorded original-text runs. The older Q4 run lacks complete hashes, so this is not a strictly paired comparison.

Other sizes: [Qwen3.5-4B](https://huggingface.co/nace-ai/Qwen3.5-4B-MLX), [Qwen3.5-9B](https://huggingface.co/nace-ai/Qwen3.5-9B-MLX), [Qwen3.6-35B-A3B](https://huggingface.co/nace-ai/Qwen3.6-35B-A3B-MLX), [Qwen3.8-Flash-Next](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX).

## Comparison data

<details>
<summary>Comparison data and exact file identities</summary>

### MLX public packs: original-text corpus

GB counts weight-file downloads including auxiliary weights, not the whole repository; configs and tokenizers are excluded. Lower KL and higher top-token agreement are closer to the reference model.

| Pack | GB | KL 95% CI | Top-token agreement |
| :--- | ---: | :--- | ---: |
| Nace (ours) | 16.293 | 0.034936 [0.031795, 0.038471] | 92.87% |
| lmstudio-community MLX-4bit | 16.055 | 0.054384 [0.049780, 0.059621] | 91.21% |
| mlx-community 4bit | 16.055 | 0.053761 [0.049395, 0.058618] | 91.15% |
| mlx-works oQ4e-mtp | 16.972 | 0.035848 [0.033524, 0.038315] | 92.59% |

<details>
<summary>Comparison provenance</summary>

Measured 2026-11-11.

Engine: MLX.

- `nace-ai/Qwen3.8-27B-MLX` at `c122d52252bd9b5db296d671cdc9c1f61f6e52f2`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16293476273 bytes (16.293476273 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `lmstudio-community/Qwen3.8-27B-MLX-4bit` at `6067b15cf581666a4aecf6af3afaba4bb5efc20c`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16054541599 bytes (16.054541599 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `mlx-community/Qwen3.8-27B-4bit` at `b935172e4bb5dcd0543774a34748fb92422e3027`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16054541349 bytes (16.054541349 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `mlx-works/Qwen3.8-27B-oQ4e-mtp` at `1c0609b20321c1faff8294be5ea88c656cedaa59`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16971702038 bytes (16.971702038 GB).
  Scored language-model weights: 16049958912 bytes (16.049958912 GB), not a download-size measure.

</details>

</details>

## Quality

| Test | lmstudio-community pack | Nace (this repo) |
| --- | ---: | ---: |
| KL from the bf16 model (lower is closer) | 0.054 | **0.035** |
| Agrees with the bf16 model on the next token | 91.2% | **92.9%** |
| GSM8K, all 1,319 questions | 0.950 | 0.953 |

Question by question on GSM8K, 13 only the lmstudio-community pack gets right and 17 only the Nace quant: no difference we can measure (McNemar z 0.73).

KL and GSM8K compare against one lmstudio-community pack: [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`.

Quality was measured on text workloads; the vision tower is included but was not evaluated.

On Q4, 20 of 20 thinking-off trials placed the call and completed the round trip; none added an extra call (MLX 0.32.4; mlx-lm 0.32.2, M5 Pro).

<details>
<summary>How measured</summary>

| | |
| --- | --- |
| lmstudio-community pack | [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`, the pack `kapitsa serve --model 27b` serves |
| KL and agreement | next-token distributions over 32,704 tokens of WikiText-2, GSM8K chats and HumanEval code, against Qwen/Qwen3.8-27B @ 37561816 (bf16) |
| GSM8K | zero-shot, greedy, thinking off, the last number in the answer |
| What was scored | the text weights of both packs (15.13 GB each); the repos also carry the vision tower, which neither run exercised |
| Tokenizer | this repo ships the tokenizer of Qwen3.8-27B; every run here used the catalog pack's tokenizer, whose ids are identical on English, code, math, CJK, emoji and RTL text; Devanagari and Thai were not measured |
| Scored with | mlx-lm 0.32.2 on a Modal H100 (mlx with CUDA), on the text weights; every run is in the [results record](https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-11-10-m5-pro-kquant-27b/README.md) |

</details>

## Why Nace quants

Nace quants come from our own quantization for MLX packs. An MLX pack ships only if it comes out closer to the original model than the lmstudio-community pack.

On this size the Nace quant takes the distance from the original bf16 model from 0.054 (lmstudio-community's pack) to 0.035 (KL, lower is closer) and moves GSM8K from 0.950 to 0.953, at the same size and in the same format.

**How we checked it**

- **Token by token against the original.** The pack's next-token predictions are compared with the bf16 model's over 32K tokens of prose, math chats and code.
- **All 1,319 GSM8K questions**, compared with the lmstudio-community pack question by question, so a difference inside the noise is called noise.

**Why it is closer**

- **Calibrated on real text.** The pack is tuned to how the model behaves on real text, not only to the numbers in its weights.
- **A search per group of weights.** Every group gets its own best fit, instead of being rounded the same way as every other.

Use an MLX runtime with support for Qwen3.8-27B.

## Credits

The MTP head (`mtp.safetensors`) and `mtplx_runtime.json` are from [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) at `1d5087d2` (MTPLX, apache-2.0), unchanged. Tokenizer and chat template as in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Thanks to Youssofal.

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
