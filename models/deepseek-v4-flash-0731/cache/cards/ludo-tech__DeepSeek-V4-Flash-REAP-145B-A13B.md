---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek_v4
  - moe
  - expert-pruning
  - reap
  - dgx-spark
language:
  - en
  - ja
  - zh
---

# DeepSeek-V4-Flash REAP 145B-A13B (K128)

**DeepSeek Flash, optimized to run on a single NVIDIA DGX Spark.**

This is [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(284B, 256 routed experts per layer, about 167 GB) with half of the experts removed
(256 to 128 per layer) using [REAP](https://arxiv.org/abs/2510.13999), a pruning method that
keeps the experts the router actually uses. The result loads and serves on one DGX Spark
(GB10, 128 GB unified memory) with stock vLLM. No source patches needed.

- **82.4 GB** on disk (FP4 experts / FP8 attention, kept tensors are byte-identical to the original)
- About 145B total parameters, 13B active per token (top-6 routing, unchanged)
- Decode speed about 17 tok/s, time to first token about 0.35 s on one DGX Spark (32K context, fp8 KV cache)
- 128 experts per layer is natively supported by vLLM's fused `topk_softplus_sqrt` router kernel
- Calibrated on English, code, and Japanese, so Japanese output stays usable (see below)

Not affiliated with DeepSeek. This is a community compression experiment. Expect some quality
loss versus the original model (see Evaluation).

## Quickstart (DGX Spark)

```bash
# create a venv and install vllm (torch cu130 wheels)
uv venv vllm-env
uv pip install --python vllm-env/bin/python vllm --index-url https://download.pytorch.org/whl/cu130

# FlashInfer needs prebuilt kernels when nvcc is absent.
# Both packages must match the flashinfer-python version that vllm installed:
uv pip install --python vllm-env/bin/python "flashinfer-cubin==<flashinfer version>" --index-url https://flashinfer.ai/whl
uv pip install --python vllm-env/bin/python "flashinfer-jit-cache==<flashinfer version>" --index-url https://flashinfer.ai/whl/cu130

vllm-env/bin/vllm serve <this-repo> \
  --served-model-name v4-flash-k128 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8          # required: DeepSeek-V4 fp8_ds_mla KV layout
```

Prompts must use the official DeepSeek-V4 encoding (`encoding/` in this repo, no Jinja template):

```python
from encoding_dsv4 import encode_messages
prompt = encode_messages([{"role": "user", "content": "1+1は？"}], thinking_mode="chat")
# then use the /v1/completions endpoint with stop=["<｜end▁of▁sentence｜>"]
```

## How it was made

1. **Calibration ran on the DGX Spark itself.** The unpruned model never fits in memory, so
   each of the 43 decoder layers was rebuilt one at a time (FP4/FP8 dequantized to bf16 into
   `transformers` `DeepseekV4DecoderLayer`) while the hidden states `[N, L, 4, 4096]` lived in
   a disk memmap. Correctness gate: full-pipeline perplexity 4.73 on held-out samples of the
   calibration mix (2048 tokens each).
2. **REAP scores each expert** as `S_e = mean over tokens routed to e of (w_e(x) · ‖f_e(x)‖₂)`,
   where `w_e` is the routing weight actually applied to the expert output (renormalized after
   top-k). Calibration mix: 512 samples of 2048 tokens. 35% English chat (ultrachat_200k),
   30% code (evol-codealpaca-v1), 20% Japanese prose (wikimedia/wikipedia 20231101.ja),
   15% Japanese instructions (kunishou/databricks-dolly-15k-ja).

   **The calibration language mix decides which experts survive.** A first attempt calibrated
   on English and code only. Its Japanese output was garbled beyond use. Re-calibrating with
   35% Japanese changed about 11 of the 128 kept experts per layer (up to 27) and restored
   coherent Japanese. If you re-run this pipeline for another language, put that language in
   the mix.
3. **Pruning is structural.** Per layer, the top-128 experts by `S_e` are kept and renumbered,
   and the router rows (`gate.weight` and the `gate.bias` correction) are sliced to match.
   Layers 0 to 2 route by token hash through a frozen `tid2eid` table; removed ids there are
   remapped to the most similar kept expert (cosine of router rows), keeping each token's
   6 experts distinct. The MTP block (`mtp.0.*`) is dropped (`num_nextn_predict_layers: 0`).
   Kept expert tensors are byte-identical to the original checkpoint.
4. `router_mapping.json` (which original expert ids each layer kept) ships in this repo.
   The full pipeline (inventory, calibration, plan, rewrite, validate) is in `tools/`.

## Evaluation (smoke-scale)

Small eval, greedy decoding, official chat encoding (non-thinking mode), on one DGX Spark.
Numbers in parentheses are the same eval on the earlier checkpoint that was calibrated on
English and code only. The Japanese gain comes from the calibration mix, not extra capacity:

| category | items | pass | notes |
|---|---|---|---|
| code (Python, executed tests) | 12 | 12 (12) | includes JA-language instructions |
| math (exact numeric) | 14 | 13 (12) | |
| Japanese (kana-verified JA output) | 12 | 11 (8) | coherent; rare-kanji dropouts remain |
| English | 12 | n/a | recorded for side-by-side review |

Runtime smoke: 0 failures, 0 empty outputs, decode about 17 tok/s median on one DGX Spark.
**This is not a benchmark suite.** Expect measurable degradation on knowledge-heavy and
long-tail tasks. The original model outperforms this checkpoint.

## Limitations

- The MTP speculative decoding head is removed, so decode is slower than pruned variants that keep it.
- Calibration covers English chat, code, and Japanese. Other languages and domains may lose
  more quality. Chinese is in the base model's training data but not in the calibration mix.
- Japanese output occasionally drops a rare kanji mid-word. This is a byte-level BPE artifact
  of pruning: low-frequency byte tokens lose routing quality. Coherence and grammar are intact.
- The hash-layer `tid2eid` remap is a heuristic.
- Long-context behaviour beyond 32K was not validated on a single Spark.

## Acknowledgements

- DeepSeek for [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT).
- [REAP: Why Pruning Prevails for One-Shot MoE Compression](https://arxiv.org/abs/2510.13999) (Cerebras).
- Prior art on Spark-sized V4 pruning: [0xSero](https://huggingface.co/0xSero/DeepSeek-V4-Flash-180B),
  [sleepyeldrazi/ds4-nvfp4-spark](https://github.com/sleepyeldrazi/ds4-nvfp4-spark).
