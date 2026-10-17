---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next-AWQ-INT4-FP8PLE

AWQ-INT4 quant of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for consumer multi-GPU serving on the [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)
fork, with the 51.2B-parameter n-gram PLE lookup table converted BF16 → FP8 e4m3.
The PLE conversion is what makes CPU-offloaded PLE practical: with
`VLLM_PLE_CPU_OFFLOAD=1` the pinned-RAM footprint drops from ~95 GiB to ~48 GiB.

- Total size: 127.72 GiB, 38 safetensors shards
- Experts: compressed-tensors AWQ INT4, 4-bit asymmetric, group 32 (routed experts only;
  attention, norms, MTP, and the PLE table were kept high-precision at quant time)
- PLE table: 128 shard tensors FP8 e4m3 + one global bf16 scale (official Qwen FP8 layout)

## Attribution / credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — Qwen Community License 1.0
- AWQ-INT4 quant: [cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) — inherits the base model license
- Serving fork: [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)
  (docker image `lazymio/vllm-backport:latest-sm86`) — Apache-2.0. The PLE CPU-offload
  was implemented by huanghaoyan.hhy (Alibaba) within the fork (commit `93a769a97`).
- PLE fp8 layout reference: [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) — Qwen Community License 1.0
- Conversion tooling: `serving/convert_fp8_ple.py` (Jon-Nielsen)

## Licenses

- **Model weights** (all `*.safetensors`, config, tokenizer files in this repository):
  **Qwen Community License 1.0** — [`LICENSE`](LICENSE), verbatim from the official
  [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
  repository. Beyond the permissive grant, two conditions apply: (1) using the model
  in a commercial product or service with more than 100,000,000 monthly active users
  or US$20,000,000 monthly revenue requires prominently displaying the model name;
  (2) offering the model (or derivatives) as a commercial "Model as a Service" or
  "AI Work Assistant" requires a separate license from Qwen — internal use is exempt.
- **Code artifacts** in `serving/`: no additional license terms are claimed here.
  The `ple_layer.py` overlay is a derivative of vLLM and remains **Apache-2.0**
  (text included as reference: [`serving/LICENSE`](serving/LICENSE)). The
  conversion and verification scripts are original work offered as-is. The
  upstream licenses above govern the model and everything derived from it.

## Conversion details

- 128/128 tensors `...layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` → `float8_e4m3fn`
- Single global scale: `...ngram_embedding.weight_scale`, bf16, shape `(1,)`,
  value `0.00019931793212890625`, stored in `model-00002-of-00038.safetensors`
- Raw amax `0.08935546875`; the stored scale is the bf16-rounded `amax/448`, so dequant is exact
- Verified (`serving/verify_fp8_ple.py`): index ↔ file exact in both directions,
  zero non-PLE dtype changes, shard_0 SNR 31.5 dB, max relative error 0.195
- Idempotent: atomic per-file writes, resumable, never deletes; reruns skip completed outputs

## Serving

Verified on 8× RTX 3090 24 GB (PCIe, no NVLink). `serving/docker-compose.yml` runs
TP=8 + expert parallel (Marlin WNA16 MoE), MTP-3 speculative decoding (4 tokens/step),
FULL_AND_PIECEWISE cudagraphs, native 262144 context, `--mamba-cache-mode align`.

Observed with this exact stack: ~100 tok/s decode; PLE pin ~48 GiB in ~57 s;
582,370-token KV pool at 0.9 gpu-memory-utilization (≈2.2× concurrency at full context).

### Quickstart

```bash
cd serving
cp .env.example .env    # edit MODEL_PARENT / MODEL / VLLM_API_KEY
docker compose up -d
```

The overlay bind-mount (`serving/overlay/ple_layer.py`) is required for this
checkpoint: it opts fp8-serialized PLE shards into compressed-tensors checkpoints
(`VLLM_PLE_FP8_EMBEDDING=1`). Without the flag the overlay is inert; without the
overlay the flag fails loudly at load time.

### KV cache recommendation

Use the default `bf16` KV. At 0.85 utilization the rig already holds ~3M tokens of
KV across 8 cards, so lower-precision KV buys little headroom here.

### Known constraints (fork-specific, verified)

1. Any KV connector combined with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   is rejected by the fork's config validator (blanket rule — the validator cannot
   enumerate which connectors pin KV memory). Drop the env var when adding a connector.
2. The in-process `LMCacheConnectorV1` path has no mamba step-alignment validation.
   The multi-process sidecar path does: with speculative decoding on, it requires
   `--max-num-batched-tokens` equal to the model block size (400) — a fail-closed
   correctness gate, not a tuning knob.
3. `--disable-custom-all-reduce` is expected on PCIe-only rigs at TP=8 (vLLM disables
   custom all-reduce itself above 2 ranks without NVLink); `NCCL_ALGO=Ring +
   NCCL_PROTO=Simple` are the fork README's Ampere full-graph prerequisites.

## Reproducing the PLE conversion

```bash
python serving/convert_fp8_ple.py     # SRC_SNAP / OUT_DIR via env or defaults
python serving/verify_fp8_ple.py      # read-only verification of the output
```
