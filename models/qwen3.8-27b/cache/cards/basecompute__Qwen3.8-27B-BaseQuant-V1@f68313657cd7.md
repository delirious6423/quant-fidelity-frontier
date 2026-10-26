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

Quantized builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in five sizes. On held-out
text, each one is closer to the BF16 model than the [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
dynamic quant of the same size.

| Size | GGUF (llama.cpp) | baseRT bundle | Same size as |
|---|---|---|---|
| XS | `Qwen3.8-27B-BaseQuant-V1-XS.gguf` (8.36 GB) | `Qwen3.8-27B-BaseQuant-V1-XS.base` | unsloth `UD-IQ2_S` (8.37 GB) |
| Small | `Qwen3.8-27B-BaseQuant-V1-Small.gguf` (10.93 GB) | `Qwen3.8-27B-BaseQuant-V1-Small.base` | unsloth `UD-IQ3_XXS` (10.93 GB) |
| Medium | `Qwen3.8-27B-BaseQuant-V1-Medium.gguf` (16.41 GB) | `Qwen3.8-27B-BaseQuant-V1-Medium.base` | unsloth `UD-Q4_K_M` (16.46 GB) |
| Large | `Qwen3.8-27B-BaseQuant-V1-Large.gguf` (20.82 GB) | `Qwen3.8-27B-BaseQuant-V1-Large.base` | unsloth `UD-Q5_K_XL` (20.88 GB) |
| XL | `Qwen3.8-27B-BaseQuant-V1-XL.gguf` (25.30 GB) | `Qwen3.8-27B-BaseQuant-V1-XL.base` | unsloth `UD-Q6_K_XL` (25.30 GB) |

Each GGUF is no larger than its unsloth counterpart. The Small bundle needs baseRT 0.4.0 or later. The XS files
are text-only and have no MTP layer, like unsloth's `UD-IQ2_S`; the XS bundle was tested on baseRT 0.4.1.

The XL GGUF keeps the MTP layer, like `UD-Q6_K_XL`. baseRT does not load Q8_0, so the XL bundle stores those
tensors in baseRT's own 8-bit format at the same 8.5 bits per weight. The bundle carries the MTP head for
`--speculate mtp-head`, and was tested with baseRT 0.4.1 and superfluid 0.1.4.

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

On text that was not used to build these files, Small, Medium and Large are 17–34% closer to BF16 than unsloth's
quant of the same size, XS is 12–15% closer and XL is 30–37% closer. The plots cover Small, Medium and Large.

![Held-out KL divergence, BaseQuant and unsloth](heldout_kl.png)

| Held-out text | XS vs UD-IQ2_S | Small vs UD-IQ3_XXS | Medium vs UD-Q4_K_M | Large vs UD-Q5_K_XL | XL vs UD-Q6_K_XL |
|---|---|---|---|---|---|
| Model's own continuations | 0.0914 / **0.0776** (-15%) | 0.0391 / **0.0308** (-21%) | 0.00713 / **0.00555** (-22%) | 0.00278 / **0.00200** (-28%) | 0.00126 / **0.00089** (-30%) |
| Written text | 0.8349 / **0.7308** (-12%) | 0.5288 / **0.4318** (-18%) | 0.2081 / **0.1579** (-24%) | 0.1080 / **0.0716** (-34%) | 0.0886 / **0.0558** (-37%) |
| Tool-call turns | not measured | 0.0631 / **0.0526** (-17%) | 0.0127 / **0.0099** (-22%) | 0.00496 / **0.00354** (-29%) | not measured |
| Tool calls + final answers | not measured | 0.0631 / **0.0523** (-17%) | 0.0112 / **0.0090** (-20%) | 0.00432 / **0.00320** (-26%) | not measured |

Each cell: unsloth KL / BaseQuant KL (change). Bold marks the lower KL.

### Public prompts

The prompts come from six public datasets (OpenHands, SWE-agent, Glaive function calling, OpenOrca, Tulu 3,
OpenHermes). The BF16 model writes each continuation (thinking on), and both quants are scored along it.

![On-policy KL divergence on public prompts, BaseQuant and unsloth](onpolicy_kl.png)

| Source | XS vs UD-IQ2_S | Small vs UD-IQ3_XXS | Medium vs UD-Q4_K_M | Large vs UD-Q5_K_XL | XL vs UD-Q6_K_XL |
|---|---|---|---|---|---|
| OpenHands | 0.1831 / **0.1284** (-30%) | 0.1190 / **0.0793** (-33%) | 0.0212 / **0.0182** (-14%) | 0.00450 / **0.00447** (-1%) | 0.00219 / **0.00118** (-46%) |
| SWE-agent | 0.1006 / **0.0955** (-5%) | 0.0449 / **0.0330** (-27%) | 0.00912 / **0.00708** (-22%) | 0.00293 / **0.00241** (-18%) | 0.00118 / **0.00102** (-14%) |
| Glaive function calling | 0.1825 / **0.1652** (-9%) | 0.0673 / **0.0628** (-7%) | 0.0112 / **0.0098** (-13%) | 0.00335 / **0.00325** (-3%) | 0.00128 / **0.00102** (-20%) |
| OpenOrca | 0.1431 / **0.1174** (-18%) | 0.0515 / **0.0431** (-16%) | 0.00759 / **0.00624** (-18%) | 0.00248 / **0.00231** (-7%) | 0.000969 / **0.000861** (-11%) |
| Tulu 3 | 0.1404 / **0.1182** (-16%) | 0.0470 / **0.0402** (-14%) | 0.00730 / **0.00585** (-20%) | 0.00238 / **0.00231** (-3%) | 0.000992 / **0.000863** (-13%) |
| OpenHermes | 0.1277 / **0.1183** (-7%) | 0.0445 / **0.0420** (-6%) | 0.00639 / **0.00613** (-4%) | 0.00203 / 0.00209 (+3%) | 0.000761 / **0.000702** (-8%) |
| **All six** | 0.1441 / **0.1242** (-14%) | 0.0539 / **0.0460** (-15%) | 0.00861 / **0.00731** (-15%) | 0.00262 / **0.00253** (-3%) | 0.00106 / **0.000880** (-17%) |

Each cell: unsloth KL / BaseQuant KL (change). Bold marks the lower KL.
