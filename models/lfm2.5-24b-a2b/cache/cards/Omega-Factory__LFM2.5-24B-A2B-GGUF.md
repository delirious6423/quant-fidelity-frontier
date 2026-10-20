---
license: other
license_name: lfm-open
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
pipeline_tag: text-generation
tags:
  - moe
  - mixture-of-experts
  - gguf
  - compressed
  - quantized
---

# Liquid AI LFM2.5 24B MoE GGUF Precision Ladder

[![Omega Factory Verified](https://img.shields.io/badge/Omega_Factory-Verified_Model_Card-0284c7?style=flat)](https://omega-factory.ai/models/lfm2.5-24b-a2b-gguf-v1)

> **Official Release & Multi-Quant Benchmarks:**  
> View full precision comparison tables (`Q8_0`, `Q4_K_M`, `Q2_K` vs. `BF16 GGUF Base`), item-level McNemar discordance flips, and verified cryptographic receipts on the [Omega Factory Liquid AI LFM2.5 24B MoE GGUF Precision Ladder Model Page](https://omega-factory.ai/models/lfm2.5-24b-a2b-gguf-v1).



Measured MoE GGUF precision quantization ladder. Hybrid architecture combining causal convolution and mixture-of-experts attention.

## Measured quant ladder

Every rung below is reported on **its own exact bytes** (sha256-bound). A verdict is never transferred between formats or rungs, and a rung that did not clear its bands is published with its real verdict rather than omitted.

> ⚠️ **Quality Warning:** Quantization degradation detected for: `Q4_K_M` (DEGRADED), `Q2_K` (DEGRADED). Review capability retention before deploying.

| Rung | Verdict | Seeds | Size | SHA256 | Receipt |
|---|---|---|---|---|---|
| `Q8_0` | **SCREEN** | 1 | 25.49 GB | `7581f787a3f5` | `receipts/cert_Q8_0.json` |
| `Q4_K_M` | **DEGRADED** | 1 | 14.53 GB | `fbdde70ee6cd` | `receipts/cert_Q4_K_M.json` |
| `Q2_K` | **DEGRADED** | 1 | 8.81 GB | `79dd79784ed0` | `receipts/cert_Q2_K.json` |

> `SCREEN` means the result did not use the full seed set and therefore is **not** a certification.

Delivered modality: **text-only GGUF**. Base-model vision support does not establish multimodal delivery.

## BF16 GGUF Baseline vs. Quantization Precision Comparison

To evaluate quantization precision loss and capability retention, quant variants were benchmarked against the unquantized base model (`LiquidAI/LFM2.5-24B-A2B`) under controlled deterministic evaluation settings.

### Precision & Accuracy Matrix vs. BF16 GGUF Baseline

| Variant / Format | Size (Disk/VRAM) | ARC-MC Accuracy | GSM8K Accuracy | MATH-500 Accuracy | Babble Rate | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **BF16 GGUF Baseline** | 47.96 GB | **88.0%** (176/200) | **78.5%** (157/200) | **58.5%** (117/200) | 0.0033 | Baseline Ground Truth |
| **`Q8_0`** | 25.49 GB | **88.0%** (Δ +0.0 pp) | **77.0%** (Δ -1.5 pp) | **54.5%** (Δ -4.0 pp) | 0.0033 | SCREEN |
| **`Q4_K_M`** | 14.53 GB | **89.0%** (Δ +1.0 pp) | **78.5%** (Δ +0.0 pp) | **50.5%** (Δ -8.0 pp) | 0.0017 | DEGRADED |
| **`Q2_K`** | 8.81 GB | **70.5%** (Δ -17.5 pp) | **54.5%** (Δ -24.0 pp) | **37.5%** (Δ -21.0 pp) | 0.0017 | DEGRADED |

### Detailed Item-Level Contingency & Flips vs. BF16 GGUF

Per-item McNemar discordance analysis measuring questions that changed answer status relative to the unquantized BF16 GGUF baseline:

| Quantization Variant | Category | Quant Correct | BF16 GGUF Base Correct | Flips Broke (BF16 GGUF ✓ → Quant ✗) | Flips Fixed (BF16 GGUF ✗ → Quant ✓) | Net Delta (pp) |
|---|---|---|---|---|---|---|
| `Q8_0` | **GSM8K** | 154 / 200 | 157 / 200 | 11 | 8 | -1.5 pp |
| `Q8_0` | **MATH-500** | 109 / 200 | 117 / 200 | 17 | 9 | -4.0 pp |
| `Q8_0` | **ARC-MC** | 176 / 200 | 176 / 200 | 3 | 3 | +0.0 pp |
| `Q4_K_M` | **GSM8K** | 157 / 200 | 157 / 200 | 15 | 15 | +0.0 pp |
| `Q4_K_M` | **MATH-500** | 101 / 200 | 117 / 200 | 28 | 12 | -8.0 pp |
| `Q4_K_M` | **ARC-MC** | 178 / 200 | 176 / 200 | 4 | 6 | +1.0 pp |
| `Q2_K` | **GSM8K** | 109 / 200 | 157 / 200 | 57 | 9 | -24.0 pp |
| `Q2_K` | **MATH-500** | 75 / 200 | 117 / 200 | 48 | 6 | -21.0 pp |
| `Q2_K` | **ARC-MC** | 141 / 200 | 176 / 200 | 40 | 5 | -17.5 pp |

## Hardware & VRAM Allocation Recommendations

Artifact size is disk usage, not a measured VRAM requirement. Runtime memory and the KV cache depend on architecture, backend, batch settings, and context length. Validate actual deployment with peak-memory profiling.

Reference capacity bands only:

| Artifact size (disk) | Reference capacity band |
|---:|---:|
| Up to 8 GB | 6–8 GB |
| Over 8 GB through 16 GB | 12–16 GB |
| Over 16 GB through 24 GB | 20–24 GB |
| Over 24 GB through 48 GB | 32–48 GB |
| Over 48 GB | Multi-GPU or CPU offload; no single listed tier |

| Quant Variant | Artifact Size (disk) | Deployment sizing notes |
|---|---:|---|
| **`Q8_0`** | **25.49 GB** | 32–48 GB reference capacity. SCREEN evaluation status; deployment is not recommended. |
| **`Q4_K_M`** | **14.53 GB** | 12–16 GB reference capacity. Candidate only, not a measured fit guarantee. |
| **`Q2_K`** | **8.81 GB** | 12–16 GB reference capacity. Candidate only, not a measured fit guarantee. |

## Recommended Sampling Parameters

- **Measured greedy mode**:
  - `temperature = 0.0`
  - `top_k = 1`
  - `seed = 42`

## Model Architecture Specifications

- **Model Type**: moe
- **Parameters**: 24.0 Billion
- **Active Parameters**: 2.45 Billion
- **Layers**: 40
- **Experts**: 64 (top-4 active)
- **Context Length**: 131072
- **Attention**: Hybrid 1D Causal Convolution and Full Attention

## Chat template

```jinja
{{- bos_token -}}
{%- set preserve_thinking = preserve_thinking | default(false) -%}

{%- macro format_arg_value(arg_value) -%}
    {%- if arg_value is string -%}
        {{- "'" + (arg_value | replace("\\", "\\\\") | replace("'", "\\'") | replace("\n", "\\n") | replace("\r", "\\r")) + "'" -}}
    {%- elif arg_value is mapping or arg_value is iterable -%}
        {{- arg_value | tojson -}}
    {%- else -%}
        {{- arg_value | string -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro parse_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is mapping -%}
        {{- content | tojson -}}
    {%- elif content is iterable -%}
        {%- set _ns = namespace(result="") -%}
        {%- for item in content -%}
            {%- if item is string -%}
                {%- set _ns.result = _ns.result + item -%}
            {%- elif item is mapping and item.get("type") == "image" -%}
                {%- set _ns.result = _ns.result + "<image>" -%}
            {%- elif item is mapping and item.get("type") == "text" -%}
                {%- set _ns.result = _ns.result + ((item.get("text") or "") | string) -%}
            {%- else -%}
                {%- set _ns.result = _ns.result + (item | tojson) -%}
            {%- endif -%}
        {%- endfor -%}
        {{- _ns.result -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_calls(tool_calls) -%}
    {%- set tool_calls_ns = namespace(tool_calls=[]) -%}
    {%- for tool_call in tool_calls -%}
        {%- set func = tool_call["function"] if "function" in tool_call else tool_call -%}
        {%- set func_name = func["name"] -%}
        {%- set func_args = func.get("arguments") -%}
        {%- set args_ns = namespace(arg_strings=[]) -%}
        {%- if func_args is mapping -%}
            {%- for arg_name, arg_value in func_args.items() -%}
                {%- set args_ns.arg_strings = args_ns.arg_strings + [arg_name + "=" + format_arg_value(arg_value)] -%}
            {%- endfor -%}
        {%- elif func_args is string and (func_args | trim) not in ["", "{}", "null"] -%}
            {{- raise_exception("Tool call arguments must be a mapping, got a JSON-encoded string: parse arguments with json.loads() before applying the chat template") -}}
        {%- endif -%}
        {%- set tool_calls_ns.tool_calls = tool_calls_ns.tool_calls + [func_name + "(" + (args_ns.arg_strings | join(", ")) + ")"] -%}
    {%- endfor -%}
    {{- "<|tool_call_start|>[" + (tool_calls_ns.tool_calls | join(", ")) + "]<|tool_call_end|>" -}}
{%- endmacro -%}

{%- set ns = namespace(system_prompt="", last_user_index=-1) -%}
{%- if messages and messages[0]["role"] == "system" -%}
    {%- if messages[0].get("content") -%}
        {%- set ns.system_prompt = parse_content(messages[0]["content"]) -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
    {%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: [" -%}
    {%- for tool in tools -%}
        {%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
        {%- endif -%}
        {%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
    {%- endfor -%}
    {%- set ns.system_prompt = ns.system_prompt + "]" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {%- set ns.last_user_index = loop.index0 -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message.role + "\n" -}}
    {%- if message.role == "assistant" -%}
        {%- generation -%}
        {%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}
        {%- set thinking = message.thinking or message.reasoning or message.reasoning_content -%}
        {%- set thinking = thinking if thinking is string else "" -%}
        {%- if thinking and keep_thinking -%}
            {{- "<think>" + thinking + "</think>" -}}
        {%- endif -%}
        {%- set _cfm_tag = "CONTINUE_FINAL_MESSAGE_TAG " -%}
        {%- set _has_cfm = false -%}
        {%- set content = "" -%}
        {%- if message.get("content") -%}
            {%- set content = parse_content(message.content) -%}
        {%- endif -%}
        {%- if not keep_thinking and "</think>" in content -%}
            {%- set content = content.split("</think>")[-1] | trim -%}
        {%- endif -%}
        {%- if content.endswith(_cfm_tag) -%}
            {%- set _has_cfm = true -%}
            {%- set _trunc_len = (content | length) - (_cfm_tag | length) -%}
            {%- set content = content[:_trunc_len] -%}
        {%- endif -%}
        {{- content -}}
        {%- if message.tool_calls -%}
            {{- render_tool_calls(message.tool_calls) -}}
        {%- endif -%}
        {%- if _has_cfm -%}
            {{- _cfm_tag -}}
        {%- endif -%}
        {{- "<|im_end|>\n" -}}
        {%- endgeneration -%}
    {%- else %}
        {%- if message.get("content") -%}
            {{- parse_content(message["content"]) -}}
        {%- endif -%}
        {{- "<|im_end|>\n" -}}
    {%- endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n<think>" -}}
{%- endif -%}
```

## License and attribution

- **License:** LFM Open License v1.0 (inherited from the base model).
- **Base model:** [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) -- all credit for the underlying model belongs to its authors.
- This repository publishes a *compressed derivative*: experts were pruned and the result re-quantised. It is not a new base model.
- **Research / free use only.** This model inherits a community license: it is published for research and free use, and this repository makes **no commercial-serving claims**.
- You remain responsible for complying with the base model's license terms.

Quantized GGUF preparation and evaluation card: Omega Factory. Base weights remain attributed to the upstream authors.

## Contact / reporting an issue

Found a problem with these weights or this card? Open a discussion on this repository, or report a security concern to the maintainers via the organisation page at <https://huggingface.co/Omega-Factory>. We publish corrections and, where warranted, retract a release rather than leave a wrong card standing.

- **Official Model Page:** [https://omega-factory.ai/models/lfm2.5-24b-a2b-gguf-v1](https://omega-factory.ai/models/lfm2.5-24b-a2b-gguf-v1)
- **Organization Website:** [https://omega-factory.ai](https://omega-factory.ai)
