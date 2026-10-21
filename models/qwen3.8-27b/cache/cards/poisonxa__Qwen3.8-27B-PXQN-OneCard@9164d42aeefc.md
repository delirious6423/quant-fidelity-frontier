---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- pxqn
- qwen3
- pascal
- volta
- quantized
- mtp
- speculative-decoding
---

# Qwen3.8-27B-PXQN-OneCard

The whole **Qwen3.8-27B** on **one 16 GB card** (Tesla P100 or V100), with **131,072 tokens of context** and
**MTP speculative decoding**, in a single 12.63 GiB file.

This is a free build from **PXA Network**, quantized in our own **PXQN** format. It is the file we use as the
one-card build of this model: 3.97 bits per weight, a mixed-precision layout that spends the extra bits on the layers that
matter most, plus the model's own MTP head so decoding can be sped up without a second model.

| | |
|---|---|
| File | `Qwen3.8-27B-PXQN-OneCard.gguf` |
| Size | 13,566,509,376 bytes (12.63 GiB), 3.97 bits per weight |
| sha256 | `8fccd7aa8f671a2f78cbd364564e9423a8c8f35c9f8a9227cdee4cf59cd6ba30` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0), with its MTP head |
| Fits | one 16 GB card with 131,072 context, q4_0 KV cache, MTP on (peak 15.6 to 15.7 GiB used, see below) |

## You need the PXA v2026.10.1 build

**PXQN is our own tensor format. This file will not load in stock llama.cpp** or in other GGUF tools. Use the PXA
v2026.10.1 release: https://github.com/poisonxa16/pxa/releases/tag/v2026.10.1 (v2026.10 also loads it, but has the slow single-card decode fixed in .1) (the prebuilt tarball for Pascal/Volta
cards, or the Docker sidecars). Unpack it and run its `llama-server` as shown below. The release also ships
`pxa-launch`, a launcher that picks the card and flags for you.

## Run it (one card, 131k context, MTP)

Copy-paste, from inside the unpacked release folder (use `CUDA_VISIBLE_DEVICES` to pick the card):

```bash
export LD_LIBRARY_PATH=$PWD/lib
CUDA_VISIBLE_DEVICES=0 ./bin/llama-server -m Qwen3.8-27B-PXQN-OneCard.gguf \
  -ngl 99 -np 1 -c 131072 -ctk q4_0 -ctv q4_0 -fa on -ub 512 \
  --spec-type mtp:n_max=1 --host 127.0.0.1 --port 8080
```

These are exactly the flags every number below was measured with. Notes:

- `--spec-type mtp:n_max=1` turns MTP decoding on (depth 1). Leave `--spec-type` out, or use `--spec-type none`, for plain
  decoding. On a P100, plain decoding is the safer choice for prompts beyond roughly 64k tokens.
- `-ub 512` is the largest micro-batch that fits next to a 131k context on a 16 GB card.
- The server also answers the standard OpenAI-style chat endpoint (`/v1/chat/completions`).

## Speed

One card, the flags above, greedy decoding, 256 new tokens, prompt cache off, a fresh server per test. Tokens per second, measured on the released **PXA v2026.10.1** build. Decode cells are the **first pass on each fresh server** (the first request, not a warmed-up or repeated one); the `tg`/`pp` benchmark cells are a llama-bench mean of 3 runs; the 16k-prompt row is the first of 3 runs. The two cards were measured at the same time on an otherwise idle host, with separate CPU sets (V100 run on cores 0-8 and 36-44, P100 run on cores 9-17 and 45-53) so the runs did not compete.

| | Tesla V100 16 GB | Tesla P100 16 GB |
|---|---:|---:|
| Plain decode, prose | 34.1 | 23.6 |
| Plain decode, code | 33.5 | 23.4 |
| **MTP depth 1, prose** | **55.6** | **31.6** |
| **MTP depth 1, code** | **61.2** | **35.2** |
| MTP depth 2, prose | 55.6 | 31.3 |
| MTP depth 2, code | 60.9 | 34.5 |
| Plain decode with a 16k-token prompt already loaded | 32.6 | 22.9 |
| Short-context benchmark `tg128` / `tg256` (plain) | 34.9 / 34.3 | 24.0 / 24.0 |
| Prompt processing `pp512` / `pp4096` | 983 / 960 | 252 / 245 |
| Prompt processing of a 16k prompt | 835 | 224 |
| Prompt processing of a 128k prompt (plain server) | 339 | 128 |

MTP therefore gives about **+63% (prose) and +83% (code) on a V100** and **+34% (prose) and +50% (code) on a P100** over
plain decoding of the same file. Prose is a story-style essay prompt, code is a small refactor-and-test prompt.
Note: on a V100 there is not enough card memory left at `-ub 512` for a second draft token, so `n_max=2` runs at depth 1
there.

**131k context, one card.** A 127.7k-token prompt with a passphrase hidden in the middle was answered correctly
(needle test PASS) on both cards, with the plain server and with MTP on. Memory in use on the card:

| | V100 (16,144 MiB) | P100 (16,384 MiB) |
|---|---:|---:|
| Plain, after loading | 15,257 MiB | 15,205 MiB |
| Plain, after a 128k prompt | 15,477 MiB | 15,219 MiB |
| MTP depth 1, after loading | 15,999 MiB | 15,947 MiB |
| MTP depth 1, after a 128k prompt | 16,063 MiB | 15,965 MiB |

That is a tight fit: close other GPU programs on the card first. Long prompts also slow decoding (KV cache reads grow).

## Quality

Fidelity is measured against the **Q8_0** file of the same model on the same held-out text and settings for every row
(KL divergence of the next-token distribution, scored on **assistant-role tokens only**: 15,065 tokens from a chat-format
corpus, 24 chunks of 2048 tokens; "same top token" is how often the highest-probability token equals Q8_0's). Perplexity is
on the wikitext-2 test set (200 chunks of 512 tokens). Lower KLD and PPL are better. Every file was run through the same
PXA v2026.10 engine.

| File | Size GiB | bits/weight | KLD vs Q8_0 | Same top token | wikitext-2 PPL | Fits one 16 GB card at 131k context? |
|---|---:|---:|---:|---:|---:|---|
| **PXQN OneCard (this file)** | **12.63** | **3.97** | **0.0213** | **95.1%** | **6.979** | **Yes (measured above)** |
| Classic PXQ3 (our earlier format, same size class) | 11.79 | 3.71 | 0.2033 | 88.5% | 7.371 | Yes by size (not tested at 131k) |
| Q4_K_S (K-quant) | 14.30 | 4.50 | 0.0163 | 96.2% | 6.807 | No: about 14.3 GiB of weights leaves no room for a 131k KV cache |
| Classic PXQ4 (our earlier format) | 15.39 | 4.84 | 0.0176 | 96.0% | 6.816 | No |

How to read it honestly:

- **At the same size, PXQN is far closer to the original than our classic same-size quant.** Against classic PXQ3 (11.79
  GiB), this file is 0.8 GiB larger and has about 10 times lower KLD (0.0213 against 0.2033) and a top-token match
  of 95.1% against 88.5%.
- **The larger files are somewhat closer still, but do not fit.** A 4.5 bits-per-weight K-quant and our classic PXQ4
  score a little better (KLD 0.016 to 0.018, wikitext-2 PPL 6.81 against 6.98 for this file, about 2.5% higher) and
  cost 1.7 to 2.8 GiB more, which is exactly the memory a 131k-token context needs on a 16 GB card. That size is the
  trade-off this file makes: it gives up a little fidelity to hold the full model and a 131k context on one card.
- Assistant-token perplexity of this file is 1.0025 times the Q8_0 reference on the same tokens.
- No classic K-quant of about 12.6 GiB was available to include in the comparison, so the K-quant row is the larger
  Q4_K_S only. Q8_0 itself does not fit one card, so its own wikitext-2 perplexity is not listed.

## What is PXQN?

PXQN is PXA Network's weight format for Pascal and Volta GPUs: rotated, mixed-precision blocks with per-layer bit
allocation, decoded by kernels written for the P100 and V100 (compute capability 6.0, 6.1 and 7.0). In this file the feed-forward
layers 32 to 47 are stored at a higher tier than the rest, the output head is stored at a higher tier still, and the
attention key/value projections are kept at 8 bits. Quantization is by PXA Network's PXQN format.

## Larger PXQN sizes

Larger PXQN builds of Qwen3.8-27B (higher fidelity, for two cards) are **available to supporters** through our supporter
repository: https://ko-fi.com/shatteredrealms1

## Credits and license

- Base model: **Qwen3.8-27B** by the Qwen team, released under the **Apache-2.0** license
  (https://huggingface.co/Qwen/Qwen3.8-27B). This quantized derivative is distributed under the same Apache-2.0 license;
  the license text is in `LICENSE`.
- Quantization and runtime: PXA Network.

## Community

- Discord: https://discord.pxanetwork.com
- Support us on Ko-fi: https://ko-fi.com/shatteredrealms1
