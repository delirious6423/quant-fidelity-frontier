---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- qwen4_exp
- kapitsa
- quantized
- 4-bit
- Q4
- apple-silicon
---

# Qwen3.8-Flash-Next-MLX

**Q4:** KL 0.067 to the original BF16 model, against 0.249 for mlx-community 4-bit, with 5.4 GB less language-model weight (105.20 vs 110.62 GB).

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/resolve/main/assets/nace-dark.svg">
<img src="https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/resolve/main/assets/nace-light.svg" alt="Mean KL from BF16 versus language-model file size on linear axes. Nace: orange diamonds. Named rival markers: TensorFold, mlx-community. Each point retains its pack label. Measured packs: TensorFold 4-bit, TensorFold oQ4-MTP, mlx-community 4-bit, Nace plain 4-bit, Nace. Sizes, bits per weight and KL appear below the plot.">
</picture>

Orange diamonds: Nace Q4, in this repo. Grey markers: public MLX packs. Lower = closer to BF16.

## Checkpoints

<table>
<tr><th>Bits</th><th>Download</th><th>KL to BF16<br><sub>lower is closer</sub></th><th>Fits</th><th>Kapitsa</th></tr>
<tr><td><b>4-bit</b> · <a href="https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/tree/main/Q4"><code>Q4/</code></a></td><td>107.6 GB</td><td>0.067</td><td>192 GB+</td><td>not yet</td></tr>
</table>

- **Q4:** 4-bit in groups of 64, with 696 modules at 8-bit (groups of 64 or 128).

Fits is a rule of thumb, not a measurement: the smallest RAM size whose default GPU share on macOS holds the download plus 2 GB for a conversation. Kapitsa skips parts it doesn't use and may fit one size smaller; `kapitsa doctor` checks a given Mac.

The repo root holds the same files as `Q4/`, so links to this repo and its earlier names keep working. To download `Q4/`:

```bash
hf download nace-ai/Qwen3.8-Flash-Next-MLX --include "Q4/*" \
  --local-dir Qwen3.8-Flash-Next-MLX
```

## Quick start

This is an MLX pack for Apple silicon Macs, 107.6 GB of files. mlx-lm 0.32.2 cannot load this architecture yet, and other MLX apps were not tested with it. The model thinks before it answers by default; each block below shows how to turn that off.

### mlx-vlm

The release check loaded `Q4/` with mlx-vlm 0.7.8 and MLX 0.32.4 on CUDA (NVIDIA H200) and ran this code, which answered "Hello!". It has not been run on a Mac. Download `Q4/` with the command above first.

```bash
pip install mlx-vlm==0.7.8
```

```python
from mlx_vlm import load, generate

model, processor = load("Qwen3.8-Flash-Next-MLX/Q4")
tok = getattr(processor, "tokenizer", processor)
messages = [{"role": "user",
             "content": "Say hello in one short sentence."}]
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,  # thinking off
)
out = generate(model, processor, prompt=prompt,
               max_tokens=80, temperature=0.0)
print(out.text)
```

### Kapitsa

[Kapitsa](https://github.com/nace-ai/kapitsa) is Nace's engine for Qwen models on the Mac. It serves a chat page and OpenAI, Anthropic and Ollama APIs.

Kapitsa 0.2.36 serves this repo's earlier revision, [`8fda1047`](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/tree/8fda104737236c4bbe1d6fb2784fb499ffd0a4cd), not the files in `Q4/`. Kapitsa support for `Q4/` comes in a later release.

```bash
brew install nace-ai/tap/kapitsa
kapitsa pull flash:q4
kapitsa serve --model flash:q4 --private
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

### Tool calling

Kapitsa takes OpenAI tool definitions. A full round trip with the OpenAI SDK against `kapitsa serve`:

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
MODEL = "nace-ai/Qwen3.8-Flash-Next-MLX"
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

From the [Qwen/Qwen3.8-Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices):

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking off | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

Keep repetition_penalty at 1.0. Give answers room: up to 32,768 output tokens for most tasks with thinking on. A presence_penalty between 0 and 2 curbs endless repetition; high values can mix languages.

## About this pack

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in MLX Q4 from [Nace.AI](https://www.nace.ai/), closer to the original than [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) (KL 0.067 vs 0.141).

## Measurements

Original: WikiText-2, GSM8K chats and HumanEval code; Fresh: frozen UltraChat test_sft and Schema-Guided Dialogue test dialogues. Each corpus scores 32,704 next-token positions offline with mlx-vlm 0.7.8 against bf16; KL is not task accuracy or speed.

Decimal GB; base-LM payload only; separate MTP, drafter and vision payloads excluded. Means are shown without uncertainty bars.

Raw measurements and runtime checks: [results.json](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/blob/main/results.json).

<details>
<summary>Exact values, payload bytes and revisions</summary>

| Pack | Base-LM GB | Original KL | Fresh KL | Pin |
|---|---:|---:|---:|---|
| [Nace](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX) [1] | 105.204 | 0.06660683 | 0.07132136 | this revision |
| [nace-ai plain 4-bit (Kapitsa default)](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit) [2] | 103.424 | 0.14110035 | 0.14112859 | [5d423f9f](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-4bit/tree/5d423f9f563c1f1b7ca7945b268e65c0b92b0eae) |
| [mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit) [3] | 110.621 | 0.24928604 | 0.22910747 | [07b5dc6c](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/tree/07b5dc6c54600a359b87f1e53e7adf6351c72a2c) |
| [TensorFold 4-bit](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit) [4] | 110.680 | 0.24921756 | 0.22900716 | [4fc403bc](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit/tree/4fc403bc1b53f2e5cbc763e7a710be1bed1ceb63) |
| [TensorFold oQ4-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP) [5] | 110.794 | 0.22957656 | 0.21622438 | [069090c3](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-oQ4-MTP/tree/069090c38f60e654c08ac6cb8cd83fce1affbdbc) |

Decimal GB; base-LM payload only; separate MTP, drafter and vision payloads excluded. Exact bytes by row: 1: 105,203,519,000 B; 2: 103,424,216,600 B; 3: 110,621,031,960 B; 4: 110,679,960,600 B; 5: 110,793,555,480 B.

</details>

The reference is Qwen/Qwen3.8-Flash-Next at `de4b8e4d`. Vision was not evaluated.

Speed has not been measured for this revision.

## Download and memory

The download is 100.2 GiB (107.6 GB). On a 48 GB Mac, `kapitsa doctor` (0.2.36) plans 76.0 GB of GPU memory for `Q4/` with an 8K-token conversation, 73.2 GB of it weights, against a 38.6 GB budget, so it does not fit.

## Tool calls

With thinking off, 20 of 20 requests made a valid first tool call and 0 of 20 made an extra tool call instead of a final answer (mlx-vlm 0.7.8 with MLX 0.32.4 on CUDA, NVIDIA H200).

## Earlier revision

The previous revision of this repo, [`8fda1047`](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX/tree/8fda104737236c4bbe1d6fb2784fb499ffd0a4cd), is the one Kapitsa 0.2.36 serves (the Kapitsa block above). Its speed and tool-call measurements stay in the repo history.

## Credits and license

The Qwen Team at Alibaba Group designed and trained the original model. Nace AI quantized its published weights. This is not an official Qwen release.

The [Qwen Community License 1.0](LICENSE) applies unchanged. Model-as-a-Service or AI Work Assistant businesses need a separate Qwen license for commercial use under condition 2.
