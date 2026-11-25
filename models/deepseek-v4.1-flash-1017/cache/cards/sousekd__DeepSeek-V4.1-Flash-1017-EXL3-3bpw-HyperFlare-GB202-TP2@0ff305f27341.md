---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - flare
  - exl3
  - rtx-pro-6000
---

# DeepSeek V4.1 Flash 1017 EXL3 for HyperFlare

Flare package of Diffbot’s EXL3 3 bpw quant for 2x 96 GB RTX PRO 6000 GPUs (GB202, TP2).

- Diffbot published the [EXL3 checkpoint](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) used here.
- AzallazA’s [TP2 recipe and export guide](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/tree/bc4716dd66eff58151c2eccc60e77cb7c28fa327/DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261124) covers the build, tuning, and memory trade-offs.
- The runtime comes from [NVIDIA HyperFlare](https://github.com/NVIDIA/hyperflare-inference), with [FlashInfer/b12x](https://github.com/local-inference-lab/flashinfer) work from Local Inference Lab.

Thanks to AzallazA, Diffbot, and Local Inference Lab for making this package possible.

## Releases

| Release | Package | Runtime |
| --- | --- | --- |
| `20261124` | Current GB202 TP2 package | `20261124-5ffe4d54` |
| `20261117` | [Previous package and run instructions](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/20261117/README.md) | `20261117-036cde24` |

For the previous release, download with `--revision 20261117` and use the Docker command in the archived card.

## Run

Allow about 314 GiB of disk space. Download without symlinks.

```bash
# Option 1: Hub cache (uses HF_HUB_CACHE if set)
core=$(HF_HUB_DISABLE_SYMLINKS=1 hf download --quiet \
  sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2 \
  --revision 20261124)

# Option 2: local directory
# hf download sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2 \
#   --revision 20261124 --local-dir ./deepseek-flare-20261124 &&
#   core="$PWD/deepseek-flare-20261124"
```

The Docker command keeps both Engram tables in RAM. With the `8G` host prefix-cache setting below, Flare allocated **2,774,272 GPU KV tokens** in testing. See [AzallazA’s recipe](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/tree/bc4716dd66eff58151c2eccc60e77cb7c28fa327/DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261124) for more info.

```bash
image=ghcr.io/sousekd/hyperflare-ds41-exl3@sha256:c46a0a650859d832cee897ec99f59e903ea027c3ef6ce0f6ef29a09fcb3bca49

docker run --rm -it --init --gpus all --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=${core:?Download the snapshot first},target=/model,readonly" \
  -e FLARE_ENGRAM_PRIVATE_TABLES=1 -e FLARE_FAST_EXIT=1 \
  -e FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512 \
  "$image" run /model \
  --offline --publisher-keys /model/publisher.json --verify-weights \
  --gpus 0,1 --host 0.0.0.0 --port 8000 \
  --served-model-name DeepSeek-V4.1-Flash \
  --embedding-table-storage ram --prefix-cache-memory 8G \
  --prefix-cache-disk-limit 0 --cuda-stack-limit 144 \
  --mtp-draft-tokens 5 --mtp-adaptive-verification \
  --mtp-max-verification-token-budget 32 --mtp-adaptive-verification-alpha 0.8 \
  --scheduler-policy interactive --scheduler-ttft-order shortest \
  --scheduler-prefill-tail-merge true --scheduler-encoder-prefill true
```

When prompted, check the signing-key fingerprint below before accepting publisher `sousekd`.

`67a70cc0418d2d7a56c7224428b6cac9786c85ab3f6157a88a734315751a02e1`

### Unattended runs

In the Docker command above, drop `-it` and add these Flare flags to skip the approval prompt.

```text
--dangerously-approve-key 67a70cc0418d2d7a56c7224428b6cac9786c85ab3f6157a88a734315751a02e1
--dangerously-accept-publisher sousekd
```

Signature checks still run. Keep `--verify-weights` for a full stored-data check. To save approval across containers, mount a writable `/state` directory and pass `--trust-store /state/trust.toml`.

[Benchmarks](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/main/BENCHMARKS.md) · [Build notes](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/main/BUILD.md) · [Model license](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/20261124/LICENSE) · [Third-party notices](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/20261124/NOTICE)
