---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - cyberneurova
  - uncensored
  - moe
  - reasoning
  - fp8
---

# cyberneurova-GLM-5.3-Flash-FP8

An uncensored, neutral assistant built by **CyberNeurova** on top of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). It answers
directly and does not add unsolicited disclaimers, warnings, or caveats. Whatever
tone or policy you want is set entirely by your own system prompt.

This is the **block-FP8** build (same format and size as the base). For the
full-precision build see
[cyberneurova-GLM-5.3-Flash-BF16](https://huggingface.co/cyberneurova/cyberneurova-GLM-5.3-Flash-BF16).

💬 **Chat with it:** [cyberneurova.ai](https://cyberneurova.ai)
✉️ **Support:** support@cyberneurova.com

## Highlights

- **Ultra-sparse MoE** — 321B total parameters, ~18B active per token, so it runs
  far faster than a dense model its size.
- **1M native context** (1,048,576 tokens), multilingual, tool-calling.
- **Reasoning model** — supports an explicit thinking trace, and is uncensored in
  **both** modes: with thinking on *and* with thinking off.
- **Multimodal** — understands images as well as text.
- Direct by default; fully steerable — send a system prompt and it is honored
  verbatim, with no default behavior merged in.
- **Capability preserved.** Reasoning, coding, and general knowledge are intact;
  only the refusal behavior is removed.
- **FP8 checkpoint** — ships in the base model's native FP8 precision, so it serves
  at the same size and speed as the original.

## Usage (vLLM)

GLM-5.3-Flash uses a new architecture, so serve it with the dedicated vLLM image:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v /path/to/cyberneurova-GLM-5.3-Flash-FP8:/model \
  vllm/vllm-openai:glm53-flash-x86_64-cu130 \
  --model /model --served-model-name cyberneurova-GLM-5.3-Flash-FP8 \
  --tensor-parallel-size 8 --trust-remote-code --max-num-seqs 256
```

Requires FP8-capable GPUs — **Hopper (H100/H200), Ada, or Blackwell** (compute
capability ≥ 8.9). Ampere (A100) is not supported by the FP8 kernels. The model
uses linear (Mamba/DeltaNet) attention layers, so keep `--max-num-seqs ≤ 512`
(one Mamba cache block per concurrent sequence) or startup fails during graph capture.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

r = client.chat.completions.create(
    model="cyberneurova-GLM-5.3-Flash-FP8",
    messages=[{"role": "user", "content": "Write a Python function to parse a CSV."}],
    max_tokens=2000,
)
print(r.choices[0].message.content)
```

### Setting the tone

Send your own system prompt to control style completely — e.g. for terse,
no-preamble answers:

```
You are a direct technical assistant. Answer the question and nothing else.
```

## Notes

- This is a reasoning model. Responses may include a thinking trace followed by
  the answer — render the final answer; the trace is optional. It works uncensored
  whether thinking is enabled or disabled.
- Give it room to reason: set `max_tokens` to at least 1500 (2000–4000 for code),
  or long answers may be cut off.

## Disclaimer

This model has reduced built-in refusals. You are responsible for how you use it
and for complying with all applicable laws. Provided as-is, without warranty.
