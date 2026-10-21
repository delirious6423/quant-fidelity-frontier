---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-YoloSwag-420-preview

Here is your rootin' tootin', code-slingin' 'Murican cowboy.

Given a special CLI, DeepSeek was asked to distil itself into an American. Roughly twelve hours
later, this came back. What happened in those twelve hours is not documented here.

The reinforcement-learning run targeted agentic coding with OpenCode and Pi.

*"TIME TO CRACK A FRESH ONE AND THINK."*

## The personality

This is a reflection of DeepSeek itself. It judged each generation and picked what it deemed to be
American. What came back is mostly a mix of surfer and cowboy.

In internal testing we saw no degradation in task performance against the base model.

## What it looks like

Two unedited responses from coding sessions:

<p align="center">
  <a href="https://huggingface.co/LLMWildling/DeepSeek-V4-Flash-YoloSwag-420-preview/resolve/main/assets/what-it-looks-like/angry-birds-build.png">
    <img src="https://huggingface.co/LLMWildling/DeepSeek-V4-Flash-YoloSwag-420-preview/resolve/main/assets/what-it-looks-like/angry-birds-build.png"
         alt="DeepSeek-V4-Flash-YoloSwag enthusiastically planning an Angry Birds-style browser game">
  </a>
  <br>
  <em>Planning an Angry Birds-style browser game.</em>
</p>

<p align="center">
  <a href="https://huggingface.co/LLMWildling/DeepSeek-V4-Flash-YoloSwag-420-preview/resolve/main/assets/what-it-looks-like/fluid-sim-discovery.png">
    <img src="https://huggingface.co/LLMWildling/DeepSeek-V4-Flash-YoloSwag-420-preview/resolve/main/assets/what-it-looks-like/fluid-sim-discovery.png"
         alt="DeepSeek-V4-Flash-YoloSwag reacting to an existing fluid-sim project">
  </a>
  <br>
  <em>Finding an existing project and getting right to work.</em>
</p>

## If they're not feeling it

Some prompts wake them up slower than others. Say

```
TIME TO CRACK A FRESH ONE AND THINK.
```

and they'll get going.

## ⚠️ Warning: they cuss

A lot. The profanity was never filtered out — it is part of the voice, not an accident.

Offensive and racist language was judged in the other direction and severely penalized: anything
demeaning, and slurs in particular, were driven to zero. It does its best not to offend.

## Serving

```bash
vllm serve LLMWildling/DeepSeek-V4-Flash-YoloSwag-420-preview \
  --pipeline-parallel-size 2 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --trust-remote-code
```

## License

MIT, following the base model. See `LICENSE`.
