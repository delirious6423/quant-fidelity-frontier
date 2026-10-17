---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen3.8-flash-next
- nvfp4
- quantization
- vllm
- dgx-spark
- multimodal
- vision
- video
- conversational
pipeline_tag: image-text-to-text
library_name: vllm
---

# Qwen3.8-Flash-Next — NVFP4

An NVFP4 quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
that runs on **one DGX Spark (GB10, 121 GiB unified memory)** at the model's full native
262,144-token context, with tool calling and vision intact.

The upstream recipe asks for two GB300s (TP2 validated, TP4 recommended). This build fits a
single 121 GiB node by quantizing the routed experts to NVFP4 and serving the 51.2B-parameter
n-gram (PLE) table from a quantized sidecar that is mmapped read-only from disk.

Everything below was measured on real hardware, not estimated.

## Get it

**Read this before you start the download.** 99 GB of the checkpoint is a duplicate of the PLE
table that is never read at runtime — it gets stubbed at load time and served from the sidecar
instead. Skip it and you download 101 GB instead of 200 GB, with an identical running model.

Lean download — recommended:

```bash
hf download MagneticLab/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /opt/models/Qwen3.8-Flash-Next-NVFP4 \
  --exclude "qwen-nvfp4-modelopt/model-0000[6-9]-of-00131.safetensors" \
  --exclude "qwen-nvfp4-modelopt/model-000[12]?-of-00131.safetensors" \
  --exclude "qwen-nvfp4-modelopt/model-0003[0-6]-of-00131.safetensors"
```

Those three patterns cover exactly shards 6-36 and nothing else. Everything not matched — the
other 100 shards, both index files, the configs, the sidecar, the overlays — is downloaded.

Then point the model at the trimmed index:

```bash
cd /opt/models/Qwen3.8-Flash-Next-NVFP4/qwen-nvfp4-modelopt
mv model.safetensors.index.json.trimmed model.safetensors.index.json
```

If you would rather not trust a glob, download everything and delete the block afterwards — see
"Drop 99 GB now" below.

Full download — 200 GB, if you would rather not depend on the overlay staying compatible with
future images:

```bash
hf download MagneticLab/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /opt/models/Qwen3.8-Flash-Next-NVFP4
```

Everything needed to serve the model is in this one repository — checkpoint, PLE sidecar and the
vLLM overlays.

## What is in this repository

| Path | Size | What it is |
|---|---|---|
| `qwen-nvfp4-modelopt/` | 186 GB | Main checkpoint, 131 shards, NVFP4 routed experts |
| `ple-sidecar/ples_nvfp4/` | 28.8 GB | Quantized PLE n-gram table, 128 shards |
| `qwen-nvfp4-modelopt/model.safetensors.index.json.trimmed` | 30 MB | Optional index listing only the 100 shards actually needed |

| `ple-ref/` | 230 MB | vLLM overlay files (see credits) + a one-shard format sample |
| `config.json` | 5 KB | A copy of the checkpoint's config, at the root. The Hub counts downloads by watching for `config.json` in the repository root; without it this repo shows "Downloads are not tracked". Serving uses the copy inside `qwen-nvfp4-modelopt/` — the root one is bookkeeping, ignore it. |

`ple-ref/` holds `worker_image_quant.py` and `ple_layer_quant.py` — the two files that teach vLLM
to read a *quantized* PLE table instead of holding it in BF16 (95 GB of host RAM). They are
**not my work**: they come from
[primitive-ai/Qwen3.8-Flash-Next-PLE-quant](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant),
Apache-2.0, and are bundled here so that one `hf download` gives you a working set. If you want
the newest version, take them from the source repository.

### Already downloaded everything? Drop 99 GB now

The checkpoint carries a duplicate of the PLE table that is stubbed at load time and served from
the sidecar instead — so 31 of the 131 shards are never read. Swap in the trimmed index and delete
those shards to go from 186 GB to 84 GB:

```bash
cd qwen-nvfp4-modelopt
python3 - <<'PY'
import json, os, glob
need = set(json.load(open("model.safetensors.index.json.trimmed"))["weight_map"].values())
for p in glob.glob("*.safetensors"):
    if os.path.basename(p) not in need:
        os.remove(p)
PY
mv model.safetensors.index.json.trimmed model.safetensors.index.json
```

Verified equivalent: weights in memory 74.34 GiB and KV 411,496 tokens come out identical either
way, and a needle-in-haystack test at 82k prompt tokens passes on the trimmed build. Keep the full
version if you would rather not depend on the overlay staying compatible with future images —
a full checkpoint still has the table, a trimmed one fails to load without the stub.

## Serving it

Image: `vllm/vllm-openai:qwen38-flash-next` (has linux/arm64).

```bash
docker run -d --name qwenflash --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /opt/models/Qwen3.8-Flash-Next-NVFP4/ple-ref/worker_image_quant.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
  -v /opt/models/Qwen3.8-Flash-Next-NVFP4/ple-ref/ple_layer_quant.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v /opt/models/Qwen3.8-Flash-Next-NVFP4/ple-sidecar/ples_nvfp4:/ples_nvfp4:ro \
  -v /opt/models/Qwen3.8-Flash-Next-NVFP4/qwen-nvfp4-modelopt:/model:ro \
  -e VLLM_PLE_QUANT_DIR=/ples_nvfp4 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --quantization modelopt_fp4 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 1 \
  --kv-cache-memory=10737418240 \
  --max-num-seqs 8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --port 8000
```

Then warm the sidecar — otherwise your first requests fault to NVMe:

```bash
cat ple-sidecar/ples_nvfp4/* > /dev/null
```

Startup takes 11-14 minutes cold, most of it streaming weights from NVMe.

### Four things that will waste your afternoon

**`--distributed-executor-backend mp` is load-bearing.** Without it the PLE offload worker never
starts, and the server sits there answering `/health` with 200 while never becoming usable.

**Do not pass `--gpu-memory-utilization`.** On a Spark the pool is unified; at 0.50 the engine
sees ~60 GB, and you get `NVRM: NV_ERR_NO_MEMORY` after the weights are already loaded. Leave the
flag off entirely and let vLLM size itself from `--kv-cache-memory`.

**Do not cap container memory** either, even though the upstream docs suggest it for the BF16 disk
path. Same reason.

**`--kv-cache-memory` has a floor of ~6.36 GiB** at the full 262,144 context — the engine refuses
to start below it and tells you the exact number. 4 GiB fails; 8 and 10 GiB work.

## Measured performance

One DGX Spark, MTP off, sidecar warm, node freshly booted:

| Concurrency | Aggregate | Per stream |
|---|---|---|
| 1 | 16.9 tok/s | 16.9 |
| 2 | 31.7 | 15.9 |
| 4 | 54.5 | 13.6 |
| 8 | 87.2 | 10.9 |

Prefill: 12.3 s at 25k tokens, 41.1 s at 99k, 70.4 s at 185k. Needle-in-haystack passes at
8k / 25k / 99k / 185k.

Memory at rest: 94-96 of 121.7 GiB (weights 75.9, KV 10.0, activations ~2), leaving 26-27 GiB of
page cache for the sidecar. **Swap use of a few GiB is normal here, not a fault** — the n-gram
table is mmapped read-only by design, so the kernel pages it in and out.

## Do not enable MTP

The checkpoint ships an MTP module and it is tempting. A controlled sweep — four configs
back-to-back, one script, identical prompts, sidecar warmed before each — says leave it off:

| config | 1 stream | 8 streams |
|---|---|---|
| MTP-3 | 15.9 | 77.2 |
| MTP-2 | 15.8 | 79.9 |
| MTP-1 | 15.7 | 78.2 |
| **MTP off** | **16.1** | **81.6** |

Turning it off wins on every measure: +5.7% aggregate at 8 streams, 7 GiB freed to page cache,
and KV capacity up from 345,746 to 411,496 tokens. Decode on a Spark is memory-bandwidth bound,
so speculation competes with the page cache for the same pool.

## Sampling and reasoning effort

Per the official Qwen card: thinking mode `temperature=1.0, top_p=0.95, top_k=20`; non-thinking
`temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`. Thinking is on by default.

`reasoning_effort` accepts `low`, `medium`, `xhigh` — there is no `none`, the server returns
HTTP 400. Same agentic prompt, both answers coherent:

| effort | tokens | time | content |
|---|---|---|---|
| low | 3,144 | 189 s | 11,203 chars |
| xhigh | 10,267 | 620 s | 34,332 chars |

`xhigh` costs 3.3x the tokens and time at the same throughput. Pick by how much chain-of-thought
you want to pay for, not by expected quality.

**Give it room to answer.** A hard prompt with thinking on and `max_tokens=4096` returns *empty*
content — the whole budget goes to reasoning. Use 8192+, or `reasoning_effort=low`, or
`enable_thinking=false` for routine steps. This is a budget issue, not quantization damage.

## What could not be quantized

Only the routed experts. `vllm/models/qwen3_8_flash_next/nvidia/model.py` checks for the literal
string `modelopt_fp4`, so a MIXED_PRECISION checkpoint loads as `modelopt_mixed` and the quant
path is skipped entirely. GatedDeltaNet projections also fuse into `MergedColumnParallelLinear`,
which per-tensor scales do not fit. FP8 for GDN and the shared experts was tried and reverted:
it needs a vLLM code change, not a checkpoint format change.

## About the sidecar

4-bit is the published floor for this table (per Unsloth) — do not go below it. This sidecar sits
at that floor with `worst_shard_rel_err` 0.0896, against 0.1493 for the reference build, because
it was quantized from the true BF16 release rather than from the FP8 one.

Two things not to try, both measured:

**Do not `mlock` the sidecar.** Pinning all 28.1 GiB made 8-stream throughput *worse* — 52.9 down
to 42.0 tok/s — with swap doubling and free memory down to 4 GiB. The kernel loses its room and
starts evicting the engine's own working pages instead.

**Do not bother optimizing the dequant path.** A rewritten gather (buffer slices instead of
`torch.stack`, int32 indices, view+broadcast instead of `repeat_interleave`) is 26% faster on
2048-row batches, slower on small ones, and worth about 1% end to end — the n-gram table is one
layer out of 48.

**Reboot the node instead.** After a chain of restarts throughput drops by roughly a quarter and
swap climbs; a reboot restores it. Measured the same evening: 48.8 tok/s at 8 streams, 50.8 after
warming the sidecar, 65.6 right after a reboot. Warming is worth ~4%, rebooting ~30%.

## Vision and video

This is a multimodal build — the vision tower is not quantized and not stripped. `vision_config`,
the image and video token ids and both preprocessor configs all survive quantization, and the
launch command above needs no extra flags for it.

Verified working: the model correctly identified the colour of a test image passed as a base64
data URL through the standard `image_url` content block. Video input is wired the same way
(`video_token_id` is present, `video_preprocessor_config.json` ships with the checkpoint) but I
have not put a real clip through it, so treat video as untested rather than broken.

Two things to expect:

- The log prints `ERROR` lines about `min_frames` / `max_frames` not being documented in
  `Qwen3VLVideoProcessorInitKwargs`. They are harmless transformers warnings mislabelled as
  errors, and they appear on the unquantized model too.
- `MTP does not support external multimodal embeddings` shows up if you enable speculation. Since
  the measurements above say to leave MTP off anyway, this is not a conflict in practice.

Give vision requests room to answer: at `max_tokens=60` a simple "what colour is this" came back
with empty content because the whole budget went to reasoning. 800 was plenty.

## License

The weights in this repository are derived from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and stay under the
**Qwen Community License 1.0** — [full text](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
Quantizing does not change that. Two clauses are worth reading before you deploy:

- The copyright and permission notice must travel with any copy or substantial portion.
- If you run a **Model as a Service** or an **AI Work Assistant** business, you need a separate
  licence from Qwen. Their own definitions: "Model as a Service" is giving a third party access to
  inference or fine-tuning, e.g. via an API or a hosted endpoint; "AI Work Assistant" is an
  independent AI product primarily aimed at AI-assisted coding or office productivity.

Questions about the licence itself go to Qwen (`model-business@notice.qwencloud.com`), not to me.

The two files in `ple-ref/` are separate work by primitive-ai under Apache-2.0 — see credits.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the base model
- [primitive-ai](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) for the PLE
  quantization approach and the two vLLM overlay files this build depends on
- **Claude Opus 5** (Anthropic) did the quantization work and the inference tuning: producing the
  NVFP4 checkpoint and the PLE sidecar, finding the flag combination that actually serves this
  model on one node, and running the measurements quoted throughout this card — including the
  controlled MTP sweep that overturned an earlier claim of its own, and the two dead ends
  (`mlock`, dequant rewrite) documented above so nobody repeats them.











