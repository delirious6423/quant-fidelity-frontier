---
license: cc-by-nc-nd-4.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - agentic
  - vision-language
  - tool-calling
  - quantized
  - nvfp4
---

# qwen3.8-27b-agentic-nvfp4

<p align="center">
  <strong>Published and maintained by <a href="https://www.gowthamsridhar.com/">Gowtham Sridhar</a></strong>
</p>

An NVFP4 checkpoint of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), packaged for agentic, multimodal, and general-purpose inference. It supports configurable reasoning, structured tool calling, image and video input, and native multi-token prediction (MTP).

> [Base model](https://huggingface.co/Qwen/Qwen3.8-27B) · [DSpark companion](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) · [vLLM](https://docs.vllm.ai/en/latest/) · [SGLang](https://docs.sglang.ai/)

## Start here

### 1. Set the model and download its chat template

The included `chat_template.jinja` is the source of truth for this release. It contains the current agentic formatting, reasoning controls, tool-call grammar, and continued-message fixes. Always pass it explicitly to the server.

```bash
export MODEL_ID="reality-interface/qwen3.8-27b-agentic-nvfp4"
export TEMPLATE_DIR="./qwen3.8-27b-agentic-nvfp4-template"

hf download "$MODEL_ID" chat_template.jinja --local-dir "$TEMPLATE_DIR"
export CHAT_TEMPLATE="$(pwd)/$TEMPLATE_DIR/chat_template.jinja"
```

You may replace `MODEL_ID` with a local checkpoint directory. In that case, point `CHAT_TEMPLATE` to the `chat_template.jinja` inside the same directory.

### 2. Start vLLM

Use standard serving for the first validation run:

```bash
vllm serve "$MODEL_ID" \
  --served-model-name qwen3.8-27b-agentic-nvfp4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --chat-template "$CHAT_TEMPLATE" \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

### 3. Send a request

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="qwen3.8-27b-agentic-nvfp4",
    messages=[
        {"role": "user", "content": "Outline a practical plan for organizing a research project."}
    ],
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

print(response.choices[0].message.content)
```

## Model overview

| Capability | Support |
| --- | --- |
| Agentic workflows | System/developer instructions, structured tools, parallel tool calls, and tool results |
| Multimodal input | Text, images, and video |
| Reasoning control | Thinking and non-thinking modes with adjustable reasoning effort |
| Quantization | ModelOpt NVFP4 weights for compatible serving runtimes |
| Context window | Up to 262,144 tokens; validate memory requirements for your deployment |
| Speculative decoding | Native MTP or external DSpark in vLLM, plus DSpark in SGLang |

The checkpoint is already quantized. `--quantization modelopt_fp4` loads the packed weights; it does not quantize them again. Do not apply BitsAndBytes, AWQ, GPTQ, or another conversion while loading this checkpoint.

## Serving

Choose one path and confirm ordinary generation before enabling speculative decoding.

| Path | Use it for |
| --- | --- |
| vLLM standard | First deployment, compatibility checks, and baseline measurements |
| vLLM native MTP | Built-in speculative decoding without a separate draft checkpoint |
| vLLM + DSpark (recommended) | External DSpark drafting through vLLM's speculative-decoding API |
| SGLang standard | SGLang's OpenAI-compatible server without speculation |
| SGLang + DSpark (recommended) | External DSpark drafting with a current DSpark-capable SGLang build |

Do not enable native MTP and DSpark in the same server.

### vLLM

#### Standard serving

Use the command in [Start here](#2-start-vllm). Its key options are:

- `--quantization modelopt_fp4` loads the existing ModelOpt NVFP4 weights.
- `--chat-template "$CHAT_TEMPLATE"` uses this release's corrected prompt format.
- `--reasoning-parser qwen3` exposes reasoning through the API.
- `--tool-call-parser qwen3_xml` matches the template's default XML tool calls.

If the installed vLLM version does not recognize the model architecture, `modelopt_fp4`, or `qwen3_xml`, update vLLM rather than changing the checkpoint metadata.

#### Native MTP

This checkpoint contains a native MTP head, so no separate draft model is required. Use **3 speculative tokens** as the recommended starting profile:

```bash
vllm serve "$MODEL_ID" \
  --served-model-name qwen3.8-27b-agentic-nvfp4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --chat-template "$CHAT_TEMPLATE" \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Depth `3` follows current vLLM MTP serving guidance and is a practical default. It is not a universal optimum: the model has one native MTP layer, so vLLM reuses that layer recursively for deeper proposals. Acceptance and throughput depend on prompts, sampling, concurrency, hardware, and context length.

For production tuning:

1. Measure standard serving with MTP disabled.
2. Validate output and tool calls with depth `3`.
3. Compare depths `2` and `3` with the same prompts and concurrency.
4. Keep the fastest stable result, or disable MTP if it does not help your workload.

Depth `4` may be tested, but it is not the recommended default because the extra draft step can cost more than the accepted tokens save. See the [vLLM speculative-decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/) for current runtime details.

#### DSpark speculative decoding

vLLM can also use the external [Qwen3.8-27B-DSpark-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) companion. Its draft block size is `7`, so this profile uses 7 speculative tokens:

```bash
export DSPARK_MODEL_ID="gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4"

vllm serve "$MODEL_ID" \
  --served-model-name qwen3.8-27b-agentic-nvfp4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --chat-template "$CHAT_TEMPLATE" \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DSPARK_MODEL_ID\",\"num_speculative_tokens\":7,\"quantization\":\"modelopt_fp4\",\"draft_sample_method\":\"greedy\"}"
```

This path requires a recent vLLM build with `Qwen3DSparkModel`, DSpark, Qwen3.8, and ModelOpt FP4 support. Confirm standard serving first. Then compare DSpark against native MTP with identical prompts and concurrency; keep only the mode that improves your workload. Speculative decoding accelerates decode, not prompt prefill.

### SGLang

#### Standard serving

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --served-model-name qwen3.8-27b-agentic-nvfp4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --chat-template "$CHAT_TEMPLATE" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --port 30000
```

The `qwen3_coder` parser matches the template's native XML function-call grammar. If the installed SGLang version does not recognize this parser or `modelopt_fp4`, update SGLang before connecting agent clients.

#### DSpark speculative decoding

DSpark uses a separate draft checkpoint to propose candidate tokens before this target model verifies them. Use the compatible [Qwen3.8-27B-DSpark-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) companion and keep its weights separate from this repository.

DSpark integration is evolving. Use a current SGLang build that exposes the `DSPARK` algorithm:

```bash
export DSPARK_MODEL_ID="gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4"

python3 -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --served-model-name qwen3.8-27b-agentic-nvfp4 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --chat-template "$CHAT_TEMPLATE" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DSPARK_MODEL_ID" \
  --speculative-draft-model-quantization modelopt_fp4 \
  --speculative-dspark-block-size 7 \
  --port 30000
```

Keep the target and draft versions paired. The explicit draft quantization and block size match this companion checkpoint. Benchmark conversation, long reasoning, and XML tool calls before production use.

## Generation

### Reasoning modes

Pass template controls through `chat_template_kwargs`:

```python
# Thinking mode
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
        "preserve_thinking": True,
    }
}

# Direct-response mode
extra_body = {
    "chat_template_kwargs": {
        "enable_thinking": False,
    }
}
```

### Recommended sampling

Keep thinking and non-thinking profiles separate. These are starting points; tune them for your application.

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` | Template controls | Use case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Thinking | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` | `enable_thinking=true`, `reasoning_effort="xhigh"` | Planning, reasoning, and difficult agentic work |
| Everyday non-thinking | `0.7` | `0.80` | `20` | `0.0` | `1.5` | `1.0` | `enable_thinking=false` | Natural conversation, instruction following, and general use |
| Deterministic non-thinking | `0` | — | — | — | — | — | `enable_thinking=false` | Repeatable extraction, formatting, and regression tests |

For deterministic requests, set `temperature=0` and leave sampling-only controls unset. Greedy decoding is an optional deterministic profile, not the recommended profile for everyday conversation or open-ended reasoning.

### Tool calling

The vLLM command enables automatic tool choice. Supply OpenAI-compatible function definitions:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="qwen3.8-27b-agentic-nvfp4",
    messages=[{"role": "user", "content": "What is the weather in Vienna?"}],
    tools=tools,
    tool_choice="auto",
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)

print(response.choices[0].message.tool_calls)
```

Execute returned calls in your application, append their results as `tool` messages, and send the updated conversation back to the model. Keep `tool_call_format="xml"` when the server uses `qwen3_xml`; JSON output with an XML parser creates a format mismatch.

### Image input

```python
response = client.chat.completions.create(
    model="qwen3.8-27b-agentic-nvfp4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image and list the main objects."},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            ],
        }
    ],
)
```

## Chat template options

Always use the current `chat_template.jinja`. It is intentionally separate from `tokenizer_config.json`, so automatic discovery may select a different template.

| Option | Default | Purpose |
| --- | --- | --- |
| `reasoning_effort` | `xhigh` | Select `xhigh`, `medium`, or `low` reasoning depth |
| `enable_thinking` | `true` | Enable or disable thinking output |
| `preserve_thinking` | `true` | Preserve earlier thinking in conversation history |
| `tool_call_format` | `xml` | Select `xml` or `json` tool-call output |
| `continue_final_message` | `false` | Continue an assistant message instead of opening a new turn |

Use `enable_thinking=false` explicitly for non-thinking tool calls. The legacy `auto_disable_thinking_with_tools` option is rejected because changing the prompt mode without changing vLLM's Qwen reasoning-parser state would misclassify reasoning as normal content.

## Client integrations

The model and template support developer/system instructions, reasoning history, OpenAI-style tools, parallel calls, tool results, and multimodal placeholders. The serving layer translates the native format for each client.

| Client | Connection |
| --- | --- |
| Codex CLI and compatible Codex clients | vLLM Responses API at `http://localhost:8000/v1` |
| Claude Code | vLLM Anthropic-compatible endpoint at `http://localhost:8000` |
| Open WebUI | OpenAI-compatible connection at `http://localhost:8000/v1` |
| Hermes Agent | Custom OpenAI-compatible provider at `http://localhost:8000/v1` |
| Other agent harnesses | OpenAI Chat Completions or Responses with standard messages, tools, and tool results |

### Codex

Add a local provider to `~/.codex/config.toml`:

```toml
model = "qwen3.8-27b-agentic-nvfp4"
model_provider = "local_vllm"

[model_providers.local_vllm]
name = "Local vLLM"
base_url = "http://localhost:8000/v1"
env_key = "VLLM_API_KEY"
wire_api = "responses"
```

Set `VLLM_API_KEY` to the key accepted by your vLLM server. A placeholder such as `local` is sufficient when server authentication is disabled.

### Claude Code

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="local"
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.8-27b-agentic-nvfp4"
export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.8-27b-agentic-nvfp4"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.8-27b-agentic-nvfp4"
claude
```

Regular ChatGPT chat does not accept an arbitrary self-hosted model endpoint. This is a client limitation, not a checkpoint or template limitation.

## Transformers prompt construction

Use the processor for prompt construction and load the standalone template explicitly:

```python
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

model_id = "reality-interface/qwen3.8-27b-agentic-nvfp4"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

template_path = hf_hub_download(repo_id=model_id, filename="chat_template.jinja")
chat_template = Path(template_path).read_text(encoding="utf-8")

messages = [{"role": "user", "content": "Describe a robust agent workflow."}]
prompt = processor.apply_chat_template(
    messages,
    chat_template=chat_template,
    tokenize=False,
    add_generation_prompt=True,
)
```

For a local checkpoint, read `chat_template.jinja` from the same directory. This example intentionally loads only the processor. Use vLLM, SGLang, or another ModelOpt-aware runtime for the packed NVFP4 weights rather than loading them with stock Transformers.

## Model details

The output head (`lm_head`), MLP projections, and supported linear layers are stored in NVFP4. The vision tower, token embeddings, native MTP components, and precision-sensitive Gated-DeltaNet layers remain in BF16.

| Item | Value |
| --- | --- |
| Base model | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Architecture | Vision-language, image-text-to-text |
| Quantization | ModelOpt NVFP4 with selected BF16 components |
| Context window | Up to 262,144 tokens |
| Intended use | Agentic, multimodal, conversational, and general-purpose generation |

## Evaluation and responsible use

Evaluate response quality, tool reliability, multimodal behavior, safety, context length, memory use, and speculative-decoding performance on your own workloads before production deployment. Runtime compatibility and optimal serving settings vary by hardware and software version.

## Attribution

This is a derived checkpoint based on Qwen3.8-27B.

## License

This repository is licensed under [CC BY-NC-ND 4.0](LICENSE). You may share it with attribution for non-commercial purposes. You may not distribute modified versions.
