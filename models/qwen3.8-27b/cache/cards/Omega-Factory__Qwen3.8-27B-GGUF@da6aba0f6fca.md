---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - mixture-of-experts
  - compressed
---

# Qwen3.8-27B GGUF Precision Ladder

[![Omega Factory Verified](https://img.shields.io/badge/Omega_Factory-Verified_Model_Card-0284c7?style=flat)](https://omega-factory.ai/models/qwen3.8-27b-gguf-v1)

> **Official Release & Multi-Quant Benchmarks:**  
> View full precision comparison tables (Q8_0, Q4_K_M, Q2_K vs. FP16 Base), item-level McNemar discordance flips, and verified cryptographic receipts on the [Omega Factory Qwen3.8-27B Dense Model Page](https://omega-factory.ai/models/qwen3.8-27b-gguf-v1).


Certified GGUF precision quantization ladder for Qwen/Qwen3.8-27B

Quantized GGUF precision variants (Q8_0, Q4_K_M, Q2_K) of Qwen/Qwen3.8-27B evaluated across standard reasoning, mathematics, and science benchmarks (ARC-MC, GSM8K, MATH-500) under reproducible seed controls.


## FP16 Baseline vs. Quantization Precision Comparison

To evaluate quantization precision loss and capability retention, all quant variants were benchmarked against the unquantized FP16 base model (`Qwen/Qwen3.8-27B` @ `1d4bf0f2`) on the identical holdout test split (`y_exam`, N=200 per category, 600 total evaluation prompts) under controlled deterministic settings.

### Precision & Accuracy Matrix vs. FP16 Baseline

| Variant / Format | Size (Disk/VRAM) | Compression | ARC-MC Accuracy | GSM8K Accuracy | MATH-500 Accuracy | Babble Rate | Status |
|---|---|---|---|---|---|---|---|
| **FP16 Baseline** | ~58.0 GB | 1.00x | **94.5%** (189/200) | **59.0%** (118/200) | **23.0%** (46/200) | 0.0000 | Baseline Ground Truth |
| **`Q8_0`** | 29.05 GB | 0.50x (-50%) | **94.5%** (Δ 0.0 pp) | **63.5%** (Δ +4.5 pp) | **22.0%** (Δ -1.0 pp) | 0.0067 | CERTIFIED (Near-Lossless) |
| **`Q4_K_M`** | 16.81 GB | 0.29x (-71%) | **95.0%** (Δ +0.5 pp) | **33.5%** (Δ -25.5 pp) | **17.5%** (Δ -5.5 pp) | 0.0000 | CERTIFIED |
| **`Q2_K`** | 10.86 GB | 0.19x (-81%) | **93.0%** (Δ -1.5 pp) | **86.0%** (Δ +27.0 pp) | **10.0%** (Δ -13.0 pp) | 0.0217 | CERTIFIED |

### Detailed Item-Level Contingency & Flips vs. FP16

Per-item McNemar discordance analysis measuring questions that changed answer status relative to the unquantized FP16 baseline:

| Quantization Variant | Category | Quant Correct | FP16 Base Correct | Flips Broke (FP16 ✓ → Quant ✗) | Flips Fixed (FP16 ✗ → Quant ✓) | Net Delta (pp) |
|---|---|---|---|---|---|---|
| `Q8_0` | **ARC_MC** | 189 / 200 | 189 / 200 | 0 | 0 | 0.0 pp |
| `Q8_0` | **GSM8K** | 127 / 200 | 118 / 200 | 1 | 10 | +4.5 pp |
| `Q8_0` | **MATH500** | 44 / 200 | 46 / 200 | 10 | 8 | -1.0 pp |
| `Q4_K_M` | **ARC_MC** | 190 / 200 | 189 / 200 | 0 | 1 | +0.5 pp |
| `Q4_K_M` | **GSM8K** | 67 / 200 | 118 / 200 | 61 | 10 | -25.5 pp |
| `Q4_K_M` | **MATH500** | 35 / 200 | 46 / 200 | 21 | 10 | -5.5 pp |
| `Q2_K` | **ARC_MC** | 186 / 200 | 189 / 200 | 7 | 4 | -1.5 pp |
| `Q2_K` | **GSM8K** | 172 / 200 | 118 / 200 | 15 | 69 | +27.0 pp |
| `Q2_K` | **MATH500** | 20 / 200 | 46 / 200 | 39 | 13 | -13.0 pp |

## Certified quant ladder

Every rung below was certified on **its own exact bytes** (sha256-bound). A verdict is never transferred between formats or rungs, and a rung that did not clear its bands is published with its real verdict rather than omitted.

| Rung | Verdict | Seeds | Size | SHA256 | Receipt |
|---|---|---|---|---|---|
| `Q8_0` | **CERTIFIED** | 1 | 29.05 GB | `bddff381b894` | `receipts/cert_Q8_0.json` |
| `Q4_K_M` | **CERTIFIED** | 1 | 16.81 GB | `ba4aeb0670e6` | `receipts/cert_Q4_K_M.json` |
| `Q2_K` | **CERTIFIED** | 1 | 10.86 GB | `f068c9605e58` | `receipts/cert_Q2_K.json` |

> `SCREEN` means the result did not use the full seed set and therefore is **not** a certification.


## Quickstart & How to Run

### 1. llama.cpp (CLI & Server)

Run directly with `llama.cpp` using GPU offloading:

```bash
# Run interactive CLI session
llama-cli -hf Omega-Factory/Qwen3.8-27B-GGUF:Q8_0 -ngl 99 -c 32768 --temp 0.7

# Or run the OpenAI-compatible local API server
llama-server -hf Omega-Factory/Qwen3.8-27B-GGUF:Q8_0 -ngl 99 -c 32768 --port 8080
```

### 2. Ollama

Create a local `Modelfile`:

```dockerfile
FROM ./Qwen3.8-27B.q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

Build and run:
```bash
ollama create qwen3.8-27b -f Modelfile
ollama run qwen3.8-27b
```

### 3. Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Omega-Factory/Qwen3.8-27B-GGUF",
    filename="Qwen3.8-27B.q8_0.gguf",
    n_ctx=32768,
    n_gpu_layers=-1, # Offload all layers to GPU
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to compute Fibonacci numbers efficiently."}
    ]
)
print(response["choices"][0]["message"]["content"])
```

## Hardware & VRAM Allocation Recommendations

| Quant Variant | File Size | Recommended VRAM | Target Hardware Profile | Recommended Use Case |
|---|---|---|---|---|
| **`Q8_0`** | **29.05 GB** | **32 GB - 48 GB** | Single RTX 4090 (with CPU offload) or A5000 / A6000 / L40 | Production, critical reasoning, coding |
| **`Q4_K_M`** | **16.81 GB** | **20 GB - 24 GB** | Single RTX 3090 / RTX 4090 / Apple M-series (32GB Unified) | General conversational, daily development |
| **`Q2_K`** | **10.86 GB** | **12 GB - 16 GB** | Consumer GPUs (RTX 4070 / RTX 4080 / Apple M 16GB) | Memory-constrained consumer workstations |

## Recommended Sampling Parameters

- **Thinking Mode (Deep Reasoning)**:
  - `temperature = 1.0`
  - `top_p = 0.95`
  - `top_k = 20`
  - `repetition_penalty = 1.0`
- **Instruct Mode (Standard Output)**:
  - `temperature = 0.7`
  - `top_p = 0.80`
  - `top_k = 20`
  - `presence_penalty = 1.5`

## Model Architecture Specifications

- **Base Architecture**: Dense Transformer with Vision-Language capabilities
- **Parameters**: 27 Billion
- **Layers**: 64
- **Hidden Dimension**: 5120
- **Attention Heads**: 24 for Q, 4 for KV (GQA)
- **Native Context Window**: 262,144 tokens

## Chat template

```jinja
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system/n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '/n/n' }}
    {%- endif %}
    {{- "# Tools/n/nYou have access to the following functions:/n/n<tools>" }}
    {%- for tool in tools %}
        {{- "/n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "/n</tools>" }}
    {{- '/n/nIf you choose to call a function ONLY reply in the following format with NO suffix:/n/n<tool_call>/n<function=example_function_name>/n<parameter=example_parameter_1>/nvalue_1/n</parameter>/n<parameter=example_parameter_2>/nThis is the value for the second parameter/nthat can span/nmultiple lines/n</parameter>/n</function>/n</tool_call>/n/n<IMPORTANT>/nReminder:/n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags/n- Required parameters MUST be specified/n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after/n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls/n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '/n/n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>/n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system/n' + (reasoning_instructions + '/n/n' if reasoning_instructions else '')  + content + '<|im_end|>/n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system/n' + reasoning_instructions + '<|im_end|>/n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system/n' + reasoning_instructions + '<|im_end|>/n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '/n' + content + '<|im_end|>' + '/n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '/n<think>/n' + reasoning_content + '/n</think>/n/n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '/n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '/n/n<tool_call>/n<function=' + tool_call.name + '>/n' }}
                    {%- else %}
                        {{- '<tool_call>/n<function=' + tool_call.name + '>/n' }}
                    {%- endif %}
                {%- else %}
                    {{- '/n<tool_call>/n<function=' + tool_call.name + '>/n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>/n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '/n</parameter>/n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>/n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>/n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '/n<tool_response>/n' }}
        {{- content }}
        {{- '/n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>/n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>/n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant/n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>/n/n</think>/n/n' }}
    {%- else %}
        {{- '<think>/n' }}
    {%- endif %}
{%- endif %}
```

## License and attribution

- **License:** Apache-2.0 (inherited from the base model).
- **Base model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) -- all credit for the underlying model belongs to its authors.
- This repository publishes a *compressed derivative*: experts were pruned and the result re-quantised. It is not a new base model.
- Commercial use is permitted under the terms of the license above.

## Contact / reporting an issue

Found a problem with these weights or this card? Open a discussion on this repository, or report a security concern to the maintainers via the organisation page at <https://huggingface.co/Omega-Factory>. We publish corrections and, where warranted, retract a release rather than leave a wrong card standing.

- **Official Model Page:** [Qwen3.8-27B Dense on Omega Factory](https://omega-factory.ai/models/qwen3.8-27b-gguf-v1)
- **Organization Website:** [https://omega-factory.ai](https://omega-factory.ai)
