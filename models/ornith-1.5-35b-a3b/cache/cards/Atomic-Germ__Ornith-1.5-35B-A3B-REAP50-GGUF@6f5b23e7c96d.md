---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- reap
- expert-pruning
- moe
- qwen3.5
- text-generation
language:
- en
pipeline_tag: text-generation
quantized_by: Ttimms
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
---

# Ornith-1.5-35B-A3B · REAP-50 — GGUF

GGUF quants of the **50%-REAP-pruned** Ornith-1.5-35B-A3B (Gated-DeltaNet + MoE
hybrid): 256 → 128 experts, MTP head removed, `~19B` parameters. Likely the
first GGUF of any 50%-REAP-pruned model.

- **Flagship (NVFP4A16, vLLM):** [`Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16)
- **Pruned bf16 source:** [`Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16)
- **Pipeline + benchmarks:** <https://github.com/t-timms/ornith-nvfp4>

Base model quality (NVFP4A16, before GGUF quant): HumanEval+ 84.2%, MBPP+ 89.2% — single greedy draws; see the [NVFP4A16 card](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-NVFP4A16#evaluation) on run-to-run spread.

## Architecture

```mermaid
graph TD
    Base["ornith-ai/Ornith-1.5-35B-A3B<br/>qwen3_5_moe - 256 experts - MTP head - ~65 GB bf16 - MIT"]
    subgraph Build ["Build - RTX 5070 Ti, SM120"]
        REAP["REAP expert prune 50%  256 -> 128"]
        MTP["strip MTP head (1 -> 0, reload-verified)"]
        StripV["strip vision tower (333 tensors)"]
        Quant["GPTQ-NVFP4A16 quantize (~5.5 h, 12.47 GiB)"]
    end
    subgraph HF ["Published formats"]
        A16["REAP-50-NVFP4A16 - 12.47 GiB (vLLM)"]
        GGUF["REAP-50-GGUF (Q4_K_M / Q5_K_M / Q6_K / Q8_0)"]
        BF16["REAP-50-bf16 (pruned source)"]
    end
    Bench["HumanEval+ 84.2% - MBPP+ 89.2% - SWE-bench Verified 44.0%"]
    Base --> REAP --> MTP --> StripV --> Quant --> A16
    StripV --> BF16
    BF16 -. convert .-> GGUF
    A16 --> Bench
```

## Download

Grab **one file**, not the whole repo.

| File | Quant | Size | Notes |
|---|---|--:|---|
| [`ornith-reap50-Q4_K_M.gguf`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-GGUF/blob/main/ornith-reap50-Q4_K_M.gguf) | Q4_K_M | 11.4 GB | **recommended** — fits 16 GB with room for context |
| [`ornith-reap50-Q5_K_M.gguf`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-GGUF/blob/main/ornith-reap50-Q5_K_M.gguf) | Q5_K_M | 13.3 GB | higher quality, tighter KV budget |
| [`ornith-reap50-Q6_K.gguf`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-GGUF/blob/main/ornith-reap50-Q6_K.gguf) | Q6_K | 15.3 GB | near-lossless; little headroom on 16 GB |
| [`ornith-reap50-Q8_0.gguf`](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-GGUF/blob/main/ornith-reap50-Q8_0.gguf) | Q8_0 | 19.7 GB | needs >16 GB or CPU offload |

MoE with ~3B active parameters — CPU / Apple-Silicon inference is fast for the size.
For full precision, re-quantizing, or merging, use the
[pruned bf16 source](https://huggingface.co/Ttimms/Ornith-1.5-35B-A3B-REAP-50-bf16)
and run `convert_hf_to_gguf.py --no-mtp` yourself.

## Run it in

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — `llama-server -m <file>.gguf -ngl 99`
- [LM Studio](https://lmstudio.ai/) (bundles a recent llama.cpp)
- [koboldcpp](https://github.com/LostRuins/koboldcpp)
- [Jan](https://jan.ai/)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
- Ollama — once its bundled llama.cpp includes this arch (see Requirements)

## Requirements

Needs a **recent llama.cpp** (the `qwen35moe` hybrid arch). Verified with
`llama-server` from a source build. Ollama support pending its bundled-llama.cpp
bump.

```bash
llama-server -m ornith-reap50-Q4_K_M.gguf -ngl 99 --port 8080
```

The chat template defaults to thinking on; pass
`"chat_template_kwargs": {"enable_thinking": false}` for grounded output.

## License

MIT (inherits from Ornith-1.5-35B-A3B). Pruning: REAP
(github.com/CerebrasResearch/reap, with a router-renormalization fix).
