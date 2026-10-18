---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - quantized
  - moe
  - vllm
  - deepseek
  - multimodal
---

# DeepSeek-V4.1-Flash · EXL3 · 2.9bpw

EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), converted with [exllamav3](https://github.com/turboderp-org/exllamav3) `v1.4.2`.

This is **not** a drop-in replacement for the original FP8 checkpoint. The Engram embedding tables are **not** in this repo — you still need shards 47 and 48 from the base model (~95 GiB each).

A slightly denser sibling lives at [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Format | EXL3 `v1.4.2`, codebook **mul1**, `out_scales=always` |
| Average bpw | **2.90** (decoder; `--hq`) |
| Head bits | 6 |
| MTP / DSpark bits | 4 (`mtp_bits`) |
| Size | ~197 GB, 39 safetensor shards |
| Calibration | 128 rows × 2048 cols (262144 tokens), workload trace `cal_trace_dsv41_flash_workload.json` |
| DSpark | In-checkpoint (`dspark_block_size=5`, 128 draft experts / top-3, layers 37–39) |
| Vision | Included (tower left native) |
| Engram tables | **Not quantized.** Use original shards **47** and **48** |

## Engram tables (required)

EXL3 only quantized the Engram `wkv` linears. The FP8 row tables stay in the original DeepSeek tree:

- `model-00047-of-00048.safetensors` (~95 GiB)
- `model-00048-of-00048.safetensors` (~95 GiB)

Point the runtime at that directory. This card ships `engram_table_dir` as `null` on purpose — do not leave a machine-local path in `config.json`.

vLLM:

```bash
--quantization exl3 \
--hf-overrides '{"engram_table_dir":"/path/to/DeepSeek-V4.1-Flash"}'
```

A slimmer overlay is enough: hardlink or copy only shards 47–48 (plus an embed-only index if your stack wants one). You do **not** need the other 46 original shards once this EXL3 tree is loaded.

## Download

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --local-dir ./DeepSeek-V4.1-Flash-EXL3-2.9bpw

hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --include "model-00047-of-00048.safetensors" \
  --include "model-00048-of-00048.safetensors" \
  --local-dir ./DeepSeek-V4.1-Flash
```

## Runtime

This checkpoint is meant for stacks that load EXL3 (`trellis` / `suh` / `svh` / `mul1`), not vanilla Transformers weight loaders.

- **vLLM** with EXL3 + DeepSeek-V4.1 support (`--quantization exl3`, DeepSeek V4.1 tokenizer / tool / reasoning parsers). Native DSpark: `--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`.
- **ExLlamaV3 / TabbyAPI** once the architecture is supported in the build you are running.

Official sampling from the base card:

| Parameter | Value |
|---|---|
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| context | up to 1M tokens |

Thinking defaults **on**. Disable per request with chat-template kwargs (`enable_thinking=false`) when you want a non-reasoning smoke.

KV cache is native V4.1 FP4 (~890 B/token). Do not force an unrelated `fp8` KV dtype from other recipes.

## Conversion

```bash
python convert.py \
  -i DeepSeek-V4.1-Flash \
  -o DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  -w /path/to/work \
  -b 2.9 \
  -hb 6 \
  -hq \
  --codebook mul1 \
  -cd cal_trace_dsv41_flash_workload.json
```

Final reported decoder bitrate: **2.90 bpw** (`--hq` enabled). MTP tensors were quantized at 4 bpw. Routed experts are a 2/3 mix (mostly K=3, a slice at K=2); attention / shared / Engram `wkv` sit higher (typically 4–6). Indexer `wk` is 8 bpw. Embeddings stay bf16.

## Base model

DeepSeek-V4.1-Flash is a multimodal MoE (552B backbone, 8B/16B activated per token in prefill/decode) with CED, CSA2, Engram memory, and in-checkpoint DSpark. Architecture, evals, and prompt format live on the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and [tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf). Scores there are for the unquantized model, not this EXL3 build.

## License

MIT, inherited from [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Quantization adds no extra restrictions.

## Citation

```bibtex
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```

Quantized by [Mia's AI Lab](https://huggingface.co/Mia-AiLab). Format and tooling: [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp).
