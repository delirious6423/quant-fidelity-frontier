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

**Run it:** [github.com/surya-koritala/condor](https://github.com/surya-koritala/condor)

![Star Drift, a space shooter the model wrote in one prompt](assets/star-drift.gif)

<sub>A space shooter written in one prompt on an RTX 4080 SUPER, unedited (precise mode, default settings).</sub>

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine, which `./setup.sh` in the GitHub repository installs.
> They do not load in Transformers, vLLM or llama.cpp. For those, use [Qwen's original weights](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

---

## Quick start

Linux or Windows on x86-64, an NVIDIA RTX 30, 40 or 50 series card (or newer) with 12 GB or more (16 GB recommended) and driver 580 or
newer, 32 GB of RAM or more (64 GB for precise and best), about 90 GB of free disk. On Windows, run `setup.ps1` in PowerShell instead:

```sh
git clone https://github.com/surya-koritala/condor && cd condor
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
| `precise/` | the experts for **precise mode** (the default; calibrated since v0.2) | 45 GB |
| `best/` | the experts for **best mode** (v0.2.1) | 52 GB |
| `fast/` | the experts for **fast mode** (v0.2.2 on; `pack/` is the same mode for engines before v0.2.2) | 29 GB |

You need `model/` and one of the three expert folders: **82 GB for precise mode, 89 GB for best, 66 GB for fast.**

- **Precise** keeps more detail in the experts (about 3 bits per weight, calibrated against the original's activations) and its other
  weights at 6 bits, so its answers stay closer to the original model: KL divergence vs bf16 0.22 on wikitext-2 (0.40 in v0.1). It is
  the better choice for code.
- **Best** keeps about 30% of the experts at 4 bits and the rest at 3, calibrated the same way: KL 0.165 vs bf16, the closest of the
  three. About 10% slower than precise on short prompts, about the same on long ones.
- **Fast** stores every expert at 2 bits per weight; since v0.2.2 those bits are packed to stay closer to the original: KL divergence vs
  bf16 0.39 on wikitext-2 (0.53 before). It writes 15-20% faster than precise, with less accuracy.

The experts (120.8B of the parameters) are compressed according to use: the experts the model routes to most keep the most detail. The
GPU holds the busiest experts, RAM holds all of them, and the CPU computes the rest while the GPU works.

---

## Speed

On one RTX 4080 SUPER (16 GB), a Ryzen 9 7950X3D and 64 GB of RAM:

| | Precise | Best | Fast |
|---|--:|--:|--:|
| Writing, short / 4K / 30K prompt (greedy) | 118 / 110 / 96 tokens/s | 104 / 92 / 97 tokens/s | 124 / 116 / 125 tokens/s |
| Writing, top speed (a short prompt, greedy) | about 130 tokens/s | about 105 tokens/s | about 169 tokens/s |
| Reading a 30,000-token prompt | 7.5 s | 7.5 s | 7.4 s |
| Reading a 261,577-token prompt | | | 66 s |

Contexts up to 262,144 tokens (the model's native length). Follow-up messages reuse the conversation, so they start in under a second.

## Quality

[Inspect AI](https://inspect.aisi.org.uk/)'s tasks as they ship, 150 questions each, greedy, thinking off, a 3,072-token cap:

| | Precise | Best | Fast |
|---|--:|--:|--:|
| GSM8K | 148 / 150 | 149 / 150 | 149 / 150 |
| MMLU-Pro | 126 / 150 | 128 / 150 | 121 / 150 |
| A full game in one prompt ([Star Drift](https://github.com/surya-koritala/condor/tree/main/examples/star-drift)) | 3 of 6 playable (v0.1) | not run | 1 of 6 (v0.1) |

The logs and the scripts that produced them are in the GitHub repository (`results/`, `bench/`).

**Limits of this release:** text only (the base model's vision encoder is not supported yet), NVIDIA GPUs only, and one request at a time
(others wait their turn). Windows is tested on one machine so far; 12 GB cards and 32 GB of RAM were measured on a 4080 held back to
those sizes.

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
