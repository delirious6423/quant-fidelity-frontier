---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- basert
- llama.cpp
- quantized
---

# Qwen3.8-27B-BaseQuant-V1

Quantized builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in three sizes. On held-out
text, each one is closer to the BF16 model than the [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
dynamic quant of the same size.

| Size | GGUF (llama.cpp) | baseRT bundle | Same size as |
|---|---|---|---|
| Small | `Qwen3.8-27B-BaseQuant-V1-Small.gguf` (10.93 GB) | `Qwen3.8-27B-BaseQuant-V1-Small.base` | unsloth `UD-IQ3_XXS` (10.93 GB) |
| Medium | `Qwen3.8-27B-BaseQuant-V1-Medium.gguf` (16.41 GB) | `Qwen3.8-27B-BaseQuant-V1-Medium.base` | unsloth `UD-Q4_K_M` (16.46 GB) |
| Large | `Qwen3.8-27B-BaseQuant-V1-Large.gguf` (20.82 GB) | `Qwen3.8-27B-BaseQuant-V1-Large.base` | unsloth `UD-Q5_K_XL` (20.88 GB) |

Each GGUF is no larger than its unsloth counterpart. The Small bundle needs baseRT 0.4.0 or later.

## Usage

```bash
# llama.cpp
llama-server -hf basecompute/Qwen3.8-27B-BaseQuant-V1 -hff Qwen3.8-27B-BaseQuant-V1-Medium.gguf -c 32768 -ngl 99


# SuperFluid
curl -fsSL https://superfluid.sh/install.sh | sh
superfluid serve basecompute/Qwen3.8-27B-BaseQuant-V1 --runtime basert --pull-file Qwen3.8-27B-BaseQuant-V1-Medium.base

# baseRT
hf download basecompute/Qwen3.8-27B-BaseQuant-V1 Qwen3.8-27B-BaseQuant-V1-Large.base --local-dir .
basert serve ./Qwen3.8-27B-BaseQuant-V1-Large.base
```

## Results

KL divergence of each quant's next-token distribution from the BF16 model, ours beside unsloth's quant of the
same size, measured in llama.cpp. Lower is closer to BF16. The percentage over each pair is our change against
unsloth.

### Held-out text

On text that was not used to build these files, all three are 17–34% closer to BF16 than unsloth's quant of the
same size.

![Held-out KL divergence, BaseQuant and unsloth](heldout_kl.png)

| Held-out text | Small vs UD-IQ3_XXS | Medium vs UD-Q4_K_M | Large vs UD-Q5_K_XL |
|---|---|---|---|
| Model's own continuations | 0.0391 / **0.0308** (-21%) | 0.00713 / **0.00555** (-22%) | 0.00278 / **0.00200** (-28%) |
| Written text | 0.5288 / **0.4318** (-18%) | 0.2081 / **0.1579** (-24%) | 0.1080 / **0.0716** (-34%) |
| Tool-call turns | 0.0631 / **0.0526** (-17%) | 0.0127 / **0.0099** (-22%) | 0.00496 / **0.00354** (-29%) |
| Tool calls + final answers | 0.0631 / **0.0523** (-17%) | 0.0112 / **0.0090** (-20%) | 0.00432 / **0.00320** (-26%) |

Each cell: unsloth KL / BaseQuant KL (change). Bold marks the lower KL.

### Public prompts

The prompts come from six public datasets (OpenHands, SWE-agent, Glaive function calling, OpenOrca, Tulu 3,
OpenHermes). The BF16 model writes each continuation (thinking on), and both quants are scored along it.

![On-policy KL divergence on public prompts, BaseQuant and unsloth](onpolicy_kl.png)

| Source | Small vs UD-IQ3_XXS | Medium vs UD-Q4_K_M | Large vs UD-Q5_K_XL |
|---|---|---|---|
| OpenHands | 0.1190 / **0.0793** (-33%) | 0.0212 / **0.0182** (-14%) | 0.00450 / **0.00447** (-1%) |
| SWE-agent | 0.0449 / **0.0330** (-27%) | 0.00912 / **0.00708** (-22%) | 0.00293 / **0.00241** (-18%) |
| Glaive function calling | 0.0673 / **0.0628** (-7%) | 0.0112 / **0.0098** (-13%) | 0.00335 / **0.00325** (-3%) |
| OpenOrca | 0.0515 / **0.0431** (-16%) | 0.00759 / **0.00624** (-18%) | 0.00248 / **0.00231** (-7%) |
| Tulu 3 | 0.0470 / **0.0402** (-14%) | 0.00730 / **0.00585** (-20%) | 0.00238 / **0.00231** (-3%) |
| OpenHermes | 0.0445 / **0.0420** (-6%) | 0.00639 / **0.00613** (-4%) | 0.00203 / 0.00209 (+3%) |
| **All six** | 0.0539 / **0.0460** (-15%) | 0.00861 / **0.00731** (-15%) | 0.00262 / **0.00253** (-3%) |

Each cell: unsloth KL / BaseQuant KL (change). Bold marks the lower KL.
