---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- glyd
- hummingbird
- moe
- mixture-of-experts
- compressed
- consumer-gpu
---

# Qwen3.8-Flash-Next · Glyd Hummingbird

**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (180B parameters) compressed to run on one 16 GB NVIDIA graphics
card with 64 GB of RAM**, with [Glyd](https://getglyd.com)'s own engine. Hummingbird is Glyd's most compact level.

**Run it:** [github.com/surya-koritala/glyd-qwen-3.8-flash](https://github.com/surya-koritala/glyd-qwen-3.8-flash)

![Star Drift, a space shooter the model wrote in one prompt](assets/star-drift.gif)

<sub>A space shooter written in one prompt on an RTX 4080 SUPER, unedited (precise mode, default settings).</sub>

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine, which `./setup.sh` in the GitHub repository installs.
> They do not load in Transformers, vLLM or llama.cpp. For those, use [Qwen's original weights](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

---

## Quick start

Linux on x86-64, an NVIDIA RTX 30, 40 or 50 series card (or newer) with 16 GB or more and driver 580 or newer, 64 GB of RAM, about 90 GB of
free disk:

```sh
git clone https://github.com/surya-koritala/glyd-qwen-3.8-flash && cd glyd-qwen-3.8-flash
./setup.sh
```

`setup.sh` installs the engine (no sudo, nothing else to install), downloads `model/` and the mode you choose from this repository (it
resumes if it stops), and opens a chat app at `http://127.0.0.1:8080`. Apps and coding agents connect to the OpenAI-compatible API at
`http://127.0.0.1:8080/v1` or the Anthropic-compatible one at `http://127.0.0.1:8080/v1/messages`.

---

## Files and modes

| Folder | What it holds | Size |
|---|---|--:|
| `model/` | Qwen's tokenizer, chat template and configs, and the weights outside the experts (attention, embeddings, n-gram table) | 37 GB |
| `precise/` | the experts for **precise mode** (the default) | 45 GB |
| `pack/` | the experts for **fast mode** | 31 GB |

You need `model/` and one of the two expert folders: **82 GB for precise mode, 68 GB for fast mode.**

- **Precise** keeps more detail in the experts (about 2.7 bits per weight on average), so its answers stay closer to the original model.
  It is the better choice for code.
- **Fast** stores every expert at about 2.1 bits per weight. It writes 15-20% faster, with less accuracy.

The experts (120.8B of the parameters) are compressed according to use: the experts the model routes to most keep the most detail. The
GPU holds the busiest experts, RAM holds all of them, and the CPU computes the rest while the GPU works.

---

## Speed

On one RTX 4080 SUPER (16 GB), a Ryzen 9 7950X3D and 64 GB of RAM:

| | Precise | Fast |
|---|--:|--:|
| Writing, in a chat (the default settings) | 117 tokens/s | 135 tokens/s |
| Writing, top speed (a short prompt, greedy) | 145 tokens/s | 169-173 tokens/s |
| Reading a 32,000-token prompt | about 7 s | about 7 s |
| Reading a 261,577-token prompt | | 66 s |

Contexts up to 262,144 tokens (the model's native length). Follow-up messages reuse the conversation, so they start in under a second.

## Quality

[Inspect AI](https://inspect.aisi.org.uk/)'s tasks as they ship, 150 questions each, greedy, thinking off, a 3,072-token cap:

| | Precise | Fast |
|---|--:|--:|
| GSM8K | 148 / 150 | 148 / 150 |
| MMLU-Pro | 124 / 150 | 118 / 150 |
| A full game in one prompt ([Star Drift](https://github.com/surya-koritala/glyd-qwen-3.8-flash/tree/main/examples/star-drift)) | 3 of 6 playable | 1 of 6 |

The logs and the scripts that produced them are in the GitHub repository (`results/`, `bench/`).

**Limits of this release:** text only (the base model's vision encoder is not supported yet), NVIDIA GPUs with 16 GB or more on Linux, and
one request at a time.

---

## The base model

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team: 125B parameters in a mixture of experts with 6B
active per token, plus a 51B n-gram embedding and a 4B multi-token prediction layer. It has 48 layers, each with 512 routed experts, of
which 10 (plus 1 shared) run for each token, and it pairs Gated DeltaNet with Qwen Sparse Attention for long contexts. Details: the
[original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## License

These weights are derived from Qwen3.8-Flash-Next and are under its license, the Qwen Community License 1.0 ([LICENSE](LICENSE)). The
Glyd engine that runs them is released separately, in compiled form, under the Business Source License 1.1: free for personal, research
and other non-commercial use (see the GitHub repository).

## Acknowledgements

Thanks to the Qwen team for releasing Qwen3.8-Flash-Next with open weights.
