---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
- deepseek
- deepseek-v4
- affinity
- rocm
- rdna4
- quantized
---

# DeepSeek-V4-Flash-0731 · affinity container · 2.875 bpw experts

A quantised container of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for the **[affinity](https://codeberg.org/StillDeadcode/affinity)** inference engine, which runs DeepSeek-V4-Flash
on one or two AMD RDNA4 GPUs (gfx1201, e.g. Radeon AI PRO R9700).

> **This is not a GGUF, safetensors or vLLM checkpoint.** It loads only in affinity (upstream, or the fork at
> [neuromaniacMD/affinity](https://github.com/neuromaniacMD/affinity), branch `v4-fixes`, which is what these
> numbers were measured on).

It saves you the 149 GB checkpoint download and the ~2.5 h quantise. The container was produced by upstream's own
`aff-quantize` (commit `a48b72e`), exactly as upstream's README describes, so it loads in upstream's engine too.

## Files

| file | size | what |
|---|---|---|
| `model-0731.aff.part-00`, `-01`, `-02` | 108.5 GB total | the main container, split for the Hub's 50 GB per-file limit |
| `model-0731.dspark.aff` | 7.5 GB | the DSpark speculative-decoding draft (0731's own bundled 3-layer draft head) |
| `SHA256SUMS` | | checksums of the parts, the draft **and the reassembled** `model-0731.aff` |
| `LICENSE` | | DeepSeek's MIT license, unchanged |

Reassemble and verify:

```sh
hf download neuromaniacmd/DeepSeek-V4-Flash-0731-affinity-2.875bpw --local-dir .
cat model-0731.aff.part-* > model-0731.aff && rm model-0731.aff.part-*
sha256sum -c SHA256SUMS --ignore-missing
```

You also need the tokenizer from the original repo:
`hf download deepseek-ai/DeepSeek-V4-Flash-0731 tokenizer.json --local-dir DeepSeek-V4-Flash-0731`.

## How it was made

- Source: `deepseek-ai/DeepSeek-V4-Flash-0731` (HF commit `7872f01b1d`), every shard's sha256 checked against the Hub.
- **Only the routed experts are quantised:** 2.875 bpw with affinity's codebook quant (2 variants x 1,024 entries),
  imatrix-weighted. Dense weights stay in the checkpoint's FP8 and the embedding/head in bf16.
- Imatrix: `imatrix/imatrix-v4-flash.dat` from
  [teamblobfish/DeepSeek-V4-Flash-GGUF](https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF), the one
  upstream's README uses. It was computed on the original V4-Flash; 0731 has the same architecture.

```sh
aff-quantize --src DeepSeek-V4-Flash-0731 --out model-0731.aff --imatrix imatrix-v4-flash.dat --threads 24
aff-quantize --src DeepSeek-V4-Flash-0731 --out model-0731.dspark.aff --dspark --codebook model-0731.aff
```

## Run (two cards, OpenAI-compatible server)

Build the engine with the fork's `docker/Dockerfile` (AMD TheRock ROCm 7.14; the host needs no ROCm):

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt label=disable --network host \
  --ipc host --ulimit memlock=-1:-1 --shm-size 16g -e HIP_VISIBLE_DEVICES=0,1 \
  -v /path/to/models:/aff affinity:v4-fixes \
    -m /aff/model-0731.aff --tokenizer /aff/DeepSeek-V4-Flash-0731/tokenizer.json \
    --gpus 2 --host-pool-mib 56000 --dspark on --kv-size 262144 --kv-commit 262144 \
    --ui plain --host 127.0.0.1 --port 8080
```

The engine loads `<model>.dspark.aff` from next to the main container automatically. Needs ~64 GB of host RAM for the non-resident experts
(56 GB pinned pool here). Thinking is on by default. The server raises a request's `max_tokens` to at least
256,000 so reasoning is never cut off; pass `--max-tokens-floor 0` if clients must be able to cap output.

## Measured (two R9700s, 32 GiB each, 2026-09-23)

| | |
|---|---|
| prefill | 1,264 / 1,356 / 1,324 tok/s at 4K / 16K / 64K context |
| decode | 50.6 / 49.8 / 40.5 tok/s on 192-token long answers; 31–35 tok/s on free-form prose and sustained reasoning (tracks draft acceptance) |
| context | 262K, fully backed at load |
| planted-fact needle | 3/3 at 4K / 16K / 64K |
| pharmacovigilance benchmark (internal, 47 items) | 44/47 at temperature 1.0 / top_p 1.0 |
| agentic tool use | 7/7 checks on a 15-turn spreadsheet-to-DuckDB conversion task (17 tool calls, 0 malformed) |

**Sampling:** use the model card's (temperature 1.0, top_p 1.0; the server's default when a request sends
none is 1.0 / 0.95). At **temperature 0.6 this quant falls into reasoning loops** (runs to the token cap at >90 %
draft acceptance).

## Credits and license

Model by DeepSeek (MIT). Engine by Yoshi Exeler ([StillDeadcode/affinity](https://codeberg.org/StillDeadcode/affinity), MIT).
Imatrix by teamblobfish. This container is distributed under the same MIT license as the model (`LICENSE`).
