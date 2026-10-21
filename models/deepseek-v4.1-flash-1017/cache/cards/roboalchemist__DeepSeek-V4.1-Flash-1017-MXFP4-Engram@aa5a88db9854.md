---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - deepseek
  - deepseek_v41
  - mxfp4
  - engram
  - sm80
  - cmp-170hx
---

# DeepSeek-V4.1-Flash-1017 — MXFP4-packed Engram tables

The two **Engram** n-gram tables of
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
shipped as FP8 E4M3 + UE8M0/32 scales; each is **91.55 GiB** as one unsplittable
tensor, so both need ~183 GiB of host memory when offloaded. This repo contains
them re-quantized to **MXFP4 (E2M1 nibbles + UE8M0 per-32 scales)**:

| table | rows × dim | original | packed |
|---|---|---:|---:|
| `layers.1.engram.embed` | 384,006,168 × 256 | 91.55 GiB | **48.64 GiB** |
| `layers.14.engram.embed` | 384,016,682 × 256 | 91.55 GiB | **48.64 GiB** |

Both packed tables fit in **97.3 GiB of pinned host RAM**, so on a 128 GiB pin
budget the Engram lookup runs in the GPU kernel over UVA (no per-step host
gather) and standard CUDA graphs apply.

The packed format is the one implemented upstream in
[vLLM PR #61530](https://github.com/vllm-project/vllm/pull/61530):
E2M1 (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; even column in the low nibble)
with the existing UE8M0 per-32 block scales; a checkpoint opts in with
`quantization_config.engram_embed_quant: "mxfp4"`.

## Files

- `engram-fp4-1.safetensors`, `engram-fp4-14.safetensors` — packed weights
  (`uint8 [rows, 128]`) + new scales (`uint8 [rows, 8]`).
- `engram-aux-1.safetensors`, `engram-aux-14.safetensors` — the small
  non-Engram tensors that shared shards 47/48 in the base checkpoint
  (`q_weight`, `k_weight`, `wkv.weight`, `wkv.scale`), moved here so the base
  shards 47/48 can be dropped (they also contain the FP8 Engram tensors, which
  a loader that scans every shard would otherwise apply over the packed ones).
- `config.json` — base config plus `quantization_config.engram_embed_quant: "mxfp4"`.
- `model.safetensors.index.json` — remaps the four Engram tensors and the aux
  tensors; every other tensor keeps its base shard mapping.
- `pack_engram_mxfp4.py` — the packer (naive round-to-nearest, streamed).
- `engram_fp4_patch.py` — the vLLM 0.13.0 backport patch that adds the packed
  decode to `deepseek_v4_1/common/engram.py`.

## How to assemble a servable model dir

```bash
BASE=deepseek-ai/DeepSeek-V4.1-Flash-1017
hf download "$BASE" --local-dir base
hf download roboalchemist/DeepSeek-V4.1-Flash-1017-MXFP4-Engram --local-dir mxfp4
mkdir model
# base shards 1..46 (everything except the Engram-carrying 47/48)
for i in $(seq -w 1 46); do ln "base/model-000${i}-of-00048.safetensors" model/; done
# tokenizer + any other base files
cp base/tokenizer* base/*.json model/ 2>/dev/null || true
# packed Engram + aux + our config/index (overwrite the base ones)
cp mxfp4/engram-*.safetensors model/
cp mxfp4/config.json mxfp4/model.safetensors.index.json model/
```

(Use `cp` instead of `ln` if the two trees are on different filesystems.)

## Serving

Apply `engram_fp4_patch.py` to the vLLM tree (it is a minimal port of PR
#61530 to the 0.13.0 `deepseek_v4_1` module), then:

```bash
vllm serve /path/to/model \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 1 --pipeline-parallel-size 6 \
  --attention-backend TRITON_MLA_SPARSE_DSV41 \
  --kv-cache-dtype fp8_ds_mla --block-size 128 \
  --max-model-len 1048576 --gpu-memory-utilization 0.95 --max-num-seqs 8 \
  --enable-prefix-caching --enable-expert-parallel \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --engram-config '{"cpu_offload": true}'
```

Set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` when the base is not torch-compiled (the
0.13.0 FULL+PIECEWISE graph mode requires one of the two).

## Quality and measurements (CMP 170HX, 6× 64 GiB, SM80)

Naive RTN quantization error against the FP8 tables: **12.1% relative RMS**
(per-32 E2M1).

Gates passed on the assembled checkpoint:
- coherence: “The capital of France is” → “Paris.”;
- chat + reasoning, and an OpenAI tool call returning `finish_reason=tool_calls`;
- planted fact recalled exactly cold and from prefix cache (21.3K-token needle,
  cold 3.9 s → restored 0.8 s).

Throughput (TP1×PP6, fp8_ds_mla, DSpark k=5, 2K prompt / 128 output):

| concurrency | output tok/s |
|---|---:|
| c1 | 32–75 (high run-to-run spread; best-of-3 74.7) |
| c4 | 133.8 |
| c8 | 142.6 |

Prefill: ~3,017 tok/s @8K and ~3,705 tok/s @32K. Host: ~162 GiB used,
~89 GiB available; 97.3 GiB pinned for Engram.

## Notes / limitations

- Quantizing Engram is lossy. These tables were produced by naive RTN; an
  AutoRound pass (as in community `*-Engram-AutoRound` quants) would likely do
  better. Validate against your own quality set before production use.
- The packed tables are only useful with a vLLM build that implements the
  packed decode (PR #61530 or the 0.13.0 backport patch here).
- The base weights are not duplicated here; use the assembly steps above.
