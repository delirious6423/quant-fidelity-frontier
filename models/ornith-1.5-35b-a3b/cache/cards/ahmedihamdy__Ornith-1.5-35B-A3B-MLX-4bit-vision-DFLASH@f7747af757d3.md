---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
- ornith-ai/Ornith-1.5-35B-A3B
- ornith-ai/Ornith-1.5-35B-A3B-DFlash
library_name: mlx
pipeline_tag: image-text-to-text
language:
- en
tags:
- mlx
- 4bit
- moe
- mixture-of-experts
- vision
- image-text-to-text
- multimodal
- qwen3.5
- ornith
- speculative-decoding
- dflash
- draft-tree
- apple-silicon
- astronomical
---

# Ornith-1.5-35B-A3B-MLX-4bit-vision-DFLASH

**Ornith 1.5 35B-A3B at 4-bit, with its eyes and the Ornith team's own draft model, on your Mac.** The Ornith team's MLX 4-bit text weights, unchanged, with the model's original vision tower put back so it reads pictures, and the Ornith team's DFlash draft model in its `drafter` folder.

A draft model guesses the next few words, and the main model checks all of the guesses in one step and keeps the ones it would have written itself. Every word you get is the main model's own choice; the drafter only decides how many arrive per step. It pays off most on predictable text such as code.

Built with, and made for, **[Astronomical](https://getastronomical.com/)**. This is a community package, not an official Ornith release.

## Speed

Decode speed in tokens per second: the reply's tokens after the first, divided by the time from the first streamed token to the last, so reading the prompt is not counted. Median of six replies, three in each of two sessions, each session after one warm-up reply, through Astronomical's local OpenAI-compatible API.

Each column is one task. **Writing an essay** asks the model for an essay about Roman aqueducts; **Writing code** asks it for a small Python program with its tests. Both were run at temperature 0.6, the Ornith team's recommended setting for everyday use.

| Build | Writing an essay | Writing code |
|---|---|---|
| [ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit), no drafter | 95.2 | 95.9 |
| This package, drafting | **85.2** (0.89x) | **122.1** (1.27x) |

Memory in use after the run: 19.09 GB with the drafter, 18.17 GB without it.

On this Mac the drafter makes writing code about 1.3 times as fast, and writing an essay about 10% slower. Code is predictable, so most of the drafter's guesses are kept. Everyday writing has many reasonable ways to continue, so more guesses are thrown away and the time spent checking them is lost. Replies at temperature 0.6 vary from run to run, and so does their speed: single replies landed up to about 10% either side of these medians.

### Conditions

| | |
|---|---|
| Computer | MacBook Pro (Mac17,9) |
| Chip | Apple M5 Pro: 15-core CPU (5 super and 10 performance cores), 16-core GPU |
| Memory | 48 GB unified memory |
| Model-memory limit | 36.5 GB, Astronomical's default on a 48 GB Mac (the most it lets models use) |
| Operating system | macOS 27.2.1 |
| Software | Astronomical 0.2.123 with DFlash drafter support (source commit `1114a12`), model fully in memory |
| Prompts | One user message: a request for an essay on Roman aqueducts (74 tokens with the chat template) and a request for a Python LRU cache module with unit tests (80 tokens) |
| Reply | 384 tokens, thinking off |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20: the Ornith team's recommended settings for general tasks |
| Drafting | 3 drafted tokens a round, verified as a tree in one step |
| Off during the run | Astronomical's SSD prompt cache and its skill note, so every prompt was read in full and nothing was added to it |

The rows without a drafter ran the Ornith team's own text-only build, whose text weights are the same files as this package's. Your speed depends on your Mac (memory bandwidth above all: a Max or Ultra chip reads weights faster), the text and the length of the conversation. Drafting gains less as a conversation grows long.

## Run it with Astronomical

[**Download Astronomical for Mac**](https://github.com/aosama/astronomical/releases/latest) (Apple silicon, macOS 14 and later). Astronomical is a free, open source app for running open models privately on your Mac: nothing you type or show it leaves your Mac.

Astronomical drafts with this package's DFlash drafter from the release after 0.2.123, and lists it in Observatory's **Models** page from that release. Earlier releases run the model without drafting.

```bash
# Ask it about a picture.
astronomical respond "What is in this image?" --image photo.png --model Ornith-1.5-35B-A3B-MLX-4bit-vision-DFLASH

# Ask it for code.
astronomical respond "Write a Python function that merges two sorted lists" --model Ornith-1.5-35B-A3B-MLX-4bit-vision-DFLASH
```

It looks for a drafter in a model's own `drafter` folder first, so this package drafts as soon as it loads, with no setup.

### What you need

| | |
|---|---|
| Disk | About 19.7 GB |
| Memory for drafting | About 19.1 GB of model memory, which fits within the default limit on a 32 GB Mac |
| Faster decoding | An M5-family Mac (the drafting kernels use the M5 GPU's matrix units) |

On a Mac without an M5 GPU, or with less memory, the model still runs: it decodes one token at a time.

## Before publishing

The vision tower was checked on a real picture through Astronomical: the model described Astronomical's own social preview image and quoted its three lines of text exactly.

## Layout

| Path | What it is |
|---|---|
| `model-0000*-of-*.safetensors`, `config.json`, tokenizer, chat template | The Ornith team's MLX 4-bit text weights: 4-bit affine experts, 8-bit routers, group size 64 |
| `model-vision.safetensors`, `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json` | The original bf16 vision tower (27 blocks, 447M parameters) and its image settings |
| `drafter/` | The Ornith team's DFlash draft model: 6 layers (5 sliding-window, 1 full-attention) that read 8 of the main model's 40 layers. `drafter/README.md` is its original card |
| `astronomical-build.json` | The sources, the vision tower's base match, and the size and SHA-256 of every file |
| `LICENSE`, `NOTICE` | The license and where each part came from |

## Sources

| Source | Revision | Where it is here |
|---|---|---|
| [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) | `19504d912fa8fc7622bf6b1de3db5d5d890b1f02` | The text weights, `model-0000*-of-00004.safetensors`, unchanged |
| [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | `10fbf86fed7ecee4a061f8b499a618f46001cac1` | The vision tower, `model-vision.safetensors` |
| [`ornith-ai/Ornith-1.5-35B-A3B-DFlash`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash) | `942d20fee84d04c9ecbf18e9c6fc4144914a1d5d` | `drafter/` |

No weight was changed. `astronomical vision match` compared 24 tensors the text weights and the tower's source both keep unquantized, and all 24 were identical, so the tower belongs with these text weights.

## Credits

- The Ornith Team, for Ornith 1.5 35B-A3B, its MLX builds and its DFlash draft model.
- z-lab, for DFlash.

## License

MIT, as declared by the Ornith team for [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) and [ornith-ai/Ornith-1.5-35B-A3B-DFlash](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-DFlash). `NOTICE` lists each part's source.
