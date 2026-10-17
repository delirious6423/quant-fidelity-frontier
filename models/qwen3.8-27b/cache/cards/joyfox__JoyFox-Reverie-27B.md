---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - zh
  - en
tags:
  - joyfox
  - roleplay
  - conversational
  - companion
  - creative-writing
  - vision
model_name: JoyFox-Reverie-27B
---

<p align="center">
  <img src="assets/joyfox-reverie-27b-banner.png" alt="JoyFox-Reverie-27B" width="100%">
</p>

<p align="center">
  <img src="assets/joyfox-reverie-tagline-en.png" alt="Characters to imagine. Conversations to share." width="760">
</p>

<p align="center">
  <a href="https://huggingface.co/joyfox/JoyFox-Reverie-27B">Model</a> · <a href="https://huggingface.co/joyfox/JoyFox-Reverie-27B/blob/main/README.md">English</a> · <a href="https://huggingface.co/joyfox/JoyFox-Reverie-27B/blob/main/README_ZH.md">简体中文</a>
</p>

> A familiar voice for imagined worlds and everyday moments.

JoyFox-Reverie-27B is a conversational model from JoyFox, derived from [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and focused on character roleplay, shared storytelling, and companion dialogue.

“Reverie” evokes the space between imagination and conversation: a character to meet, a story to explore, or a familiar voice to talk to at the end of the day. Define the experience through your prompt, from a richly described fictional setting to a relaxed everyday exchange.

## ✨ Highlights

- **Character dialogue:** shape a persona through background, voice, relationships, and scene instructions.
- **Companion conversations:** set a friendly conversational style for everyday topics and personal reflection.
- **Shared storytelling:** create dialogue and scenes together, defining which characters each participant controls.
- **Flexible expression:** specify the reply language, narrative perspective, response length, and balance of action and dialogue.
- **Thinking controls:** choose direct replies or enable reasoning through the supplied chat template.
- **Visual context:** the checkpoint retains the upstream visual components for image and video inputs through compatible processors and runtimes.

## 🚀 Quickstart

### Transformers

Use a current Transformers build that supports the Qwen3.8 checkpoint and its processor. Architecture loading details are covered in the [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

```bash
pip install -U torch transformers accelerate
```

The following example starts a character conversation with thinking disabled:

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

model_id = "joyfox/JoyFox-Reverie-27B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are Mira, an old friend who runs a seaside bookshop. Speak naturally, stay in character, and let me choose my own actions."},
    {"role": "user", "content": "I push open the shop door after a long day. Got a moment to talk?"},
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
    )

reply = output[0, inputs["input_ids"].shape[-1]:]
print(processor.decode(reply, skip_special_tokens=True))
```

### Serving with vLLM

The following command exposes a chat endpoint with a 32K context setting. Allocate enough GPU memory for the BF16 weights and runtime state; increase tensor parallelism when using multiple GPUs.

```bash
pip install -U vllm

vllm serve joyfox/JoyFox-Reverie-27B \
  --served-model-name JoyFox-Reverie-27B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --reasoning-parser qwen3
```

For multiple GPUs, add `--tensor-parallel-size` with your GPU count. Context capacity and concurrency depend on available memory. Consult the [vLLM Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) for framework requirements and hardware configuration.

### Chat API

Once the server is running, send a conversation request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "JoyFox-Reverie-27B",
  "messages": [
    {
      "role": "system",
      "content": "Speak naturally as my old friend Mira. Let me choose my own actions."
    },
    {
      "role": "user",
      "content": "I had a long day. Can we talk for a bit?"
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}'
```

For multi-turn chat, append each assistant reply and the next user message to `messages` before sending the next request. The server uses the model's supplied chat template.

### Image Input

With the model and processor from the Transformers example, replace `messages` with the following, then run the same template and generation steps:

```python
messages = [
    {"role": "system", "content": "Describe the scene as Mira, a seaside bookshop owner. Distinguish what you see from what you imagine."},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What do you notice in this picture?"},
        ],
    },
]
```

Image and video processing depends on the serving framework. For video input formats and media controls, refer to the [upstream usage guide](https://huggingface.co/Qwen/Qwen3.8-27B).

## 💬 Conversation Styles

### Character Roleplay

A character prompt works best when it identifies the speaker, the relationship, and the current scene:

```text
You are Mira, the owner of a small seaside bookshop and an old friend of mine.
Voice: warm, observant, with occasional dry humor.
Scene: a rainy evening just before closing time.
Write mostly dialogue, with brief actions where useful.
Keep the setting grounded in established facts. Leave my actions and decisions to me.
```

### Companion Chat

For a conversation without a fictional scene:

```text
Talk with me like a thoughtful friend. Use natural, relaxed language and follow the topic I bring up. Ask questions when they help, and share your perspective without turning every reply into advice. Avoid stage directions unless I ask for roleplay.
```

### Shared Storytelling

```text
We are writing a mystery together. You control the supporting cast and environment; I control the protagonist. Build on established events, introduce clues gradually, and leave room for my next decision.
```

## 🧠 Thinking Controls

For direct character dialogue, pass `enable_thinking=False` to the chat template. Thinking is enabled by default in the included template.

When reasoning is useful, these options can be passed to `processor.apply_chat_template`:

```python
chat_options = {
    "enable_thinking": True,
    "reasoning_effort": "medium",
    "preserve_thinking": True,
}
```

`reasoning_effort` accepts `low`, `medium`, and `xhigh` (the default). `preserve_thinking` controls retention of reasoning content in earlier assistant messages. For the chat API, place template options inside `chat_template_kwargs`; the server's reasoning parser separates reasoning from the answer.

## 🎛️ Generation Settings

The quickstart uses temperature `0.7`, top-p `0.9`, and top-k `20` for direct dialogue. The included generation configuration uses temperature `1.0`, top-p `0.95`, and top-k `20`.

Treat these as starting points. Lower temperature for more restrained replies; increase it when exploring more varied phrasing. Adjust one setting at a time, and keep enough output tokens for the desired response length. Sampling options vary across serving frameworks.

## 💡 Best Practices

- Keep the character card concise and explicit. Separate permanent identity from the current scene.
- Carry earlier messages forward so the model can use established events and relationships.
- State who controls each character and whether the model may introduce new events.
- Specify the language, point of view, and desired length rather than relying on implied style.
- For long sessions, maintain a short summary of established facts alongside recent messages.
- Treat fictional dialogue as generated fiction, and verify factual statements when accuracy matters.

## 📦 GGUF

UD quantizations from Q4 through Q6 are available in [GGUF](https://huggingface.co/joyfox/JoyFox-Reverie-27B/tree/main/GGUF). Every main file includes MTP tensors; use the matching F16 `mmproj` for images.

| File | Size (GiB) |
|---|---:|
| [JoyFox-Reverie-27B-UD-Q4_K_M.gguf](GGUF/JoyFox-Reverie-27B-UD-Q4_K_M.gguf) | 15.33 |
| [JoyFox-Reverie-27B-UD-Q4_K_S.gguf](GGUF/JoyFox-Reverie-27B-UD-Q4_K_S.gguf) | 14.30 |
| [JoyFox-Reverie-27B-UD-Q4_K_XL.gguf](GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf) | 16.35 |
| [JoyFox-Reverie-27B-UD-Q5_K_M.gguf](GGUF/JoyFox-Reverie-27B-UD-Q5_K_M.gguf) | 18.41 |
| [JoyFox-Reverie-27B-UD-Q5_K_S.gguf](GGUF/JoyFox-Reverie-27B-UD-Q5_K_S.gguf) | 17.38 |
| [JoyFox-Reverie-27B-UD-Q5_K_XL.gguf](GGUF/JoyFox-Reverie-27B-UD-Q5_K_XL.gguf) | 19.44 |
| [JoyFox-Reverie-27B-UD-Q6_K.gguf](GGUF/JoyFox-Reverie-27B-UD-Q6_K.gguf) | 20.47 |
| [JoyFox-Reverie-27B-UD-Q6_K_L.gguf](GGUF/JoyFox-Reverie-27B-UD-Q6_K_L.gguf) | 22.53 |
| [JoyFox-Reverie-27B-UD-Q6_K_M.gguf](GGUF/JoyFox-Reverie-27B-UD-Q6_K_M.gguf) | 21.50 |
| [JoyFox-Reverie-27B-UD-Q6_K_XL.gguf](GGUF/JoyFox-Reverie-27B-UD-Q6_K_XL.gguf) | 23.56 |
| [mmproj-JoyFox-Reverie-27B-F16.gguf](GGUF/mmproj-JoyFox-Reverie-27B-F16.gguf) | 0.86 |

### Download a GGUF

Use a current llama.cpp build with `qwen35` support. Download one main GGUF; the matching `mmproj` is needed only for image input. The examples below use Q4 XL; replace the main filename to use another available quantization.

```bash
pip install -U huggingface_hub
hf download joyfox/JoyFox-Reverie-27B \
  GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf \
  GGUF/mmproj-JoyFox-Reverie-27B-F16.gguf \
  --local-dir JoyFox-Reverie-27B
cd JoyFox-Reverie-27B
```
### Text Chat

Run the following command from the downloaded model directory:

```bash
llama-cli -m GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf \
  --jinja --reasoning off -c 32768 -ngl all \
  --temp 0.7 --top-p 0.9 --top-k 20
```
`--jinja` applies the bundled chat template; `--reasoning off` selects direct replies. Adjust `-c` for your desired context and available memory. `-ngl all` offloads the model to the GPU; reduce the offload when it does not fit. The download size excludes runtime buffers and KV cache.

### Chat Server

```bash
llama-server -m GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf \
  --jinja --reasoning off -c 32768 -ngl all \
  --host 127.0.0.1 --port 8080
```

The server provides a web interface at `http://127.0.0.1:8080` and a chat endpoint at `/v1/chat/completions`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"You are Mira, my old friend. Speak naturally."},{"role":"user","content":"How was your day?"}],"temperature":0.7,"top_p":0.9,"max_tokens":256}'
```

### Image Input

Use the matching F16 visual projector with your chosen main GGUF:

```bash
llama-cli -m GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf \
  --mmproj GGUF/mmproj-JoyFox-Reverie-27B-F16.gguf \
  --image example.png --prompt "Describe this image." \
  --jinja --reasoning off -c 4096 -ngl all
```

### Optional MTP Decoding

All main GGUF files include native MTP tensors. On a compatible llama.cpp build, start the server with bundled MTP:

```bash
llama-server -m GGUF/JoyFox-Reverie-27B-UD-Q4_K_XL.gguf \
  --jinja --reasoning off -c 32768 -ngl all \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port 8080
```

This replaces the ordinary server command. No separate draft-model download is required. MTP uses additional memory; speed depends on draft acceptance and the workload.

### Choosing a Quantization

Q4 S has the smallest footprint in this release. Q4 XL offers a practical starting point for roleplay; Q5 and Q6 require more memory. Choose according to available memory and your preferred output quality. For a GGUF application, load the downloaded main file and use the supplied chat template; image input additionally requires the matching visual projector and application support.

## 📜 License and Acknowledgments

JoyFox-Reverie-27B is released under [Apache-2.0](LICENSE). It is a JoyFox derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Thanks to the Qwen team and the Hugging Face ecosystem.
