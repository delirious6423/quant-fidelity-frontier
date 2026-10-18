---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- tenstorrent
- blackhole
- p300x2
- quietbox-2
- tt-metal
- tt-model-cache
- tt-model-catalog
- tt-model-container
- tt-dit-server
- deepseek_v41
---

# DeepSeek-V4.1-Flash on a Tenstorrent QuietBox 2

DeepSeek-V4.1-Flash (552B MoE + 197B Engram) served on the four Blackhole chips of a QuietBox 2 (2x p300c), pure
TT: every model FLOP runs on the chips. Routed experts are bfp4 tiles streamed from host page cache into a 4500-slot
on-device LRU; attention, CSA compression, mHC and the Engram lookups run on the 1x4 ring with tensor parallelism 4.

```bash
tt serve rothgar/deepseek-v4.1-flash-p300x2                       # 9216-token context
tt serve rothgar/deepseek-v4.1-flash-p300x2 --profile p300x2-32k  # 33792-token context
```

`tt serve` prints the endpoint (`http://127.0.0.1:20000/v1` by default). The server speaks the OpenAI API (`/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming,
logprobs, `chat_template_kwargs`, `reasoning_content` split at `</think>`). One sequence at a time; requests queue.

## What the first `tt serve` does

1. Loads the container image from this repo (~0.6 GB download).
2. Downloads `deepseek-ai/DeepSeek-V4.1-Flash` at revision `2cba9e42` into your Hugging Face cache (**476 GB**).
3. First boot only: builds the packed expert store in tt-model's per-model weight cache
   (`~/.cache/tt-model/deepseek-v4.1-flash-p300x2/weights`, **313 GB**, ~10 min). It first checks that its C packer
   reproduces this image's ttnn tile format byte for byte.
4. Loads the model and warms the decode traces. The first boot also compiles kernels (several minutes; cached).

Measured on a QuietBox 2: first `tt serve` 12 min with the checkpoint already cached (9 min of it the expert
store); later boots about 1 min. `tt model stop` shuts down cleanly. Needs ~800 GB of free disk and a host with ~250 GB RAM (the expert store is
served from page cache).

## Measured on a QuietBox 2

Quality against DeepSeek's own reference code on the CPU (512 tokens of mixed text, activation fake-quant off),
decisive positions = reference top-1 margin >= 0.10:

| check | decisive top-1 | mean KL |
|---|---|---|
| 512-token prefill | 0.9822 | 0.035 |
| 256 teacher-forced traced decode steps | 0.9899 | 0.030 |

The engine runs one sequence at a time. Speed, one user, `vllm bench serve` (random tokens, `--ignore-eos`):

| workload | prompt / output | context | time to first token | decode |
|---|---|---|---|---|
| chat | 128 / 256 | 9216 | 10.2 s | 3.65 tok/s |
| codegen | 1024 / 1024 | 9216 | 52.7 s | 3.73 tok/s |
| agent turn | 8192 / 512 | 9216 | 154 s | 3.78 tok/s |
| long context | 32768 / 256 | 33792 | 399 s | 1.73 tok/s |

DSpark speculative decoding is off: it accepted 1.2 drafts per pass and was slower than plain decoding.

## Improve it

The whole source branch ships here as a git bundle: the engine, the step-by-step rebuild on your own QuietBox 2
(`qb2/00-host.sh` .. `qb2/04-gates.sh`), the benchmark and the tooling that builds this repo.

```bash
hf download rothgar/deepseek-v4.1-flash-p300x2 source/dsv41-qb2.bundle --local-dir .
git clone source/dsv41-qb2.bundle dsv41-qb2 && cd dsv41-qb2   # then read QB2.md
```

`qb2/04-gates.sh` must keep passing and `qb2/bench.sh` should get faster; `tt-model/build.sh` turns your commit into
a bundle like this one. `code/` here is that engine at commit `9a5fdff`, byte-identical to `/opt/dsv41` in the image.

## Provenance

| component | source |
|---|---|
| model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `2cba9e42`, MIT |
| engine | Veso-AI's pure-TT DeepSeek-V4.1-Flash engine for Wormhole (Apache-2.0), ported to Blackhole / QuietBox 2 |
| tt-metal | [`v0.79.0`](https://github.com/tenstorrent/tt-metal/releases/tag/v0.79.0), the `ttnn` wheel from PyPI, with its pinned SFPI 7.78.0, on Ubuntu 22.04 |
