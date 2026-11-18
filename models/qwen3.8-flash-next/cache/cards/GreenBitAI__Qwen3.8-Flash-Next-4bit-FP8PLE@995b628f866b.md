---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gbx-dc
- vllm
- safetensors
- qwen4_exp
- qwen
- qwen3.8
- mixture-of-experts
- quantized
- compressed-tensors
- fp8
- 4-bit
---

# Qwen3.8-Flash-Next-4bit-FP8PLE

A quantized build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
packaged to be served with **gbx-dc**, GreenBitAI's inference server for data-centre
deployments. The architecture is unchanged.

| | |
| --- | --- |
| weights | 4-bit, group size 128, symmetric, `compressed-tensors` pack-quantized |
| activations | 16-bit |
| n-gram table | FP8 `e4m3`, one global scale |
| everything else | BF16 |
| context | 131,072 tokens |
| files | five weight shards, `model_mtp.safetensors`, the index, tokenizer, `chat_template.jinja`, `config.json` |

The quantization parameters are declared in `config.json` under
`quantization_config`, so a `compressed-tensors`-aware runtime reads them
directly: there is no conversion step. `model_mtp.safetensors` holds the
multi-token-prediction block, which a serving stack can use as a draft model for
speculative decoding; it is not required to run the model.

## Serve it

With gbx-dc, list the presets and start the one for this model:

```bash
gbx-dc weights fetch     # fetch the weights into the model root
gbx-dc serve             # serve them, on the preset for this model
curl http://127.0.0.1:8000/health
```

A preset carries a complete, validated set of parameters, so a standard
deployment selects one rather than tuning values individually; `gbx-dc models`
lists what is available.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model name>","messages":[{"role":"user","content":"Hello"}]}'
```

## Three wire protocols, one port

The server answers three request shapes on the same port, so tools written
against a hosted API can be pointed at a server of your own:

| path | for |
| --- | --- |
| `/v1/chat/completions` | the OpenAI SDK, LangChain, and most other tools |
| `/v1/responses` | **Codex** |
| `/v1/messages` | **Claude Code** |

**Claude Code**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=<your key, or EMPTY if none is set>
export ANTHROPIC_MODEL=<model name>
claude
```

**Codex** — point the service address at `http://127.0.0.1:8000/v1` and set the
model to one of the names `gbx-dc models` lists.

**OpenAI SDK**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
print(client.chat.completions.create(
    model="<model name>",
    messages=[{"role": "user", "content": "Hello"}],
).choices[0].message.content)
```

Both coding agents also ask for a small model for their own background work, so
every model name in their settings has to be one the server is serving.

## Licence

Qwen Community License 1.0, as published with the original model; the text is in
`LICENSE` in this repository. Please also read the original model's card for its
intended use and limitations.
