---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- exl3
- sushi
- apple-silicon
- moe
---

# Qwen3.8-Flash-Next EXL3 for Sushi

Qwen3.8-Flash-Next packed for [Sushi](https://github.com/beamivalice/sushi) on Apple Silicon: the routed
experts in turboderp's EXL3 trellis format, encoded by our own converter at 5 or 6 bits, with an 8-bit trunk. It is
several times closer to the bf16 model than 4-bit affine packs at similar memory.

Each variant lives on its own branch; `main` holds this card and the download script.

## Variants (branches)

- **`5bpw`** (~78 GiB + 95 GiB n-gram table): 5-bit experts, Sushi-4bpw's 8-bit trunk. Fastest.
- **`5bpw-hq`** (~78 GiB + 95 GiB n-gram table): 5-bit experts, 8-bit trunk with the hyper-connection mixes kept in bf16. Closer to bf16, nearly as fast.
- **`6bpw-hq`** (~92 GiB + 95 GiB n-gram table): 6-bit experts, 8-bit trunk with bf16 hyper-connection mixes. Closest to bf16.

## Closeness to bf16 (whole model, KLD vs the bf16 teacher; lower is closer)

| pack | wikitext KLD | reasoning KLD | 32K KLD | top-1 |
|---|---|---|---|---|
| port (oMLX OptSpeed, 4-bit affine) | 0.1416 | 0.0528 | 0.0841 | 86.1% |
| Sushi-4bpw | 0.0550 | 0.0180 | 0.0248 | 91.0% |
| 5bpw | 0.0357 | 0.0105 | 0.0163 | 92.9% |
| 5bpw-hq | 0.0325 | 0.0095 | 0.0153 | 93.1% |
| 6bpw-hq | 0.0284 | 0.0099 | 0.0124 | 93.5% |

Wikitext: 16 x 512 tokens; reasoning: 16 real reasoning traces; 32K: 4 x 32,768-token documents. Same teacher and
text for every pack; 95% bootstrap intervals separate every pack from Sushi-4bpw.

## Speed (Sushi engine, M5 Max 128 GB, MTP on)

| variant | decode tok/s (MTP) | prefill 10K tok/s |
|---|---|---|
| 5bpw | 90.5 | 1,904 |
| 5bpw-hq | 88.3 | 1,908 |
| 6bpw-hq | 80.9 | 1,790 |

The `-hq` variants reach these speeds with Sushi built with the dense hyper-connection read (a small engine patch we
plan to offer upstream); stock Sushi v1.2.3 runs `5bpw-hq` at about 79 tok/s. Outputs are identical either way.

## Download

The 95 GiB per-layer n-gram table is the base model's own and is not re-uploaded: the script assembles it from
`Qwen/Qwen3.8-Flash-Next`'s shards (checksummed), or reuses one you already have.

```sh
python3 get_flashnext_exl3.py --family qwen --variant 5bpw-hq --dest ~/Models --verify-ngram
# reuse a table from another variant or from Sushi-4bpw:
python3 get_flashnext_exl3.py --family qwen --variant 5bpw-hq --ngram-from /path/to/ngram_table.bin
```

## Run

Needs a Mac with 128 GB of unified memory (the `6bpw-hq` variant is ~99 GB resident).

```sh
sushi serve --model ~/Models/Qwen3.8-Flash-Next-EXL3-Sushi-5bpw-hq --mtp --mtp-head-kv-quant --kv-quant 8 \
  --ctx-size 262144 --max-tokens 262144 --temp 1 --top-p 0.95 --top-k 20 --think xhigh \
  --config-overrides '{"text_config":{"rope_parameters":{"rope_type":"default"},"max_position_embeddings":262144}}'
```

`--think xhigh` makes requests without a `reasoning_effort` think at the chat template's default level (Sushi's own
default is off).

## Provenance and licences

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`de4b8e4`), Qwen Community License 1.0 (`LICENSE`).
- 8-bit trunk, embeddings, head, MTP head and pack layout: [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`), with thanks.
- EXL3 trellis format: [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3). Engine: [Sushi](https://github.com/beamivalice/sushi) (MIT).

- Modified files: all `model-exl3-L*` expert shards (re-encoded), `config.json`.
