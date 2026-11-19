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

Community packaging of Diffbot’s EXL3 3 bpw checkpoint for two 96 GB RTX PRO 6000 Blackwell GPUs (GB202, TP2).

- Diffbot published the [EXL3 checkpoint](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) used here.
- AzallazA’s [TP2 recipe and export guide](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/tree/main/DeepSeek-V4.1-Flash%20Hyperflare-tp2-exl3-20261117) covers the build, tuning, and memory trade-offs.
- The runtime comes from [NVIDIA HyperFlare](https://github.com/NVIDIA/hyperflare-inference), with [FlashInfer/b12x](https://github.com/local-inference-lab/flashinfer) work from Local Inference Lab.

Thanks to AzallazA, Diffbot, and Local Inference Lab for making this package possible.

## Run

Allow about 314 GiB of disk space for the download. The command below keeps both Engram tables in host RAM. The recipe recommends 256 GB of system RAM for this mode.

```bash
hf download sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2 \
  --local-dir ./deepseek-flare
```

```bash
image=ghcr.io/sousekd/hyperflare-ds41-exl3@sha256:3bcd47fa1504b00806b045c1eb1dcbb225c831d9820d4f66062d73d9b378a3e7

docker run --rm -it --gpus all --ulimit memlock=-1:-1 \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$PWD/deepseek-flare,target=/model,readonly" \
  -e FLARE_ENGRAM_PRIVATE_TABLES=1 -e FLARE_FAST_EXIT=1 \
  "$image" run /model \
  --offline --publisher-keys /model/publisher.json --verify-weights \
  --gpus 0,1 --host 0.0.0.0 --port 8000 \
  --served-model-name DeepSeek-V4.1-Flash \
  --embedding-table-storage ram --prefix-cache-memory 8G \
  --prefix-cache-disk-limit 0 --cuda-stack-limit 128 \
  --mtp-draft-tokens 5 --mtp-adaptive-verification \
  --mtp-max-verification-token-budget 32 --mtp-adaptive-verification-alpha 0.8 \
  --scheduler-policy interactive --scheduler-ttft-order shortest \
  --scheduler-prefill-tail-merge true
```

When prompted, check the signing-key fingerprint below before accepting publisher `sousekd`.

`67a70cc0418d2d7a56c7224428b6cac9786c85ab3f6157a88a734315751a02e1`

### Unattended runs from the Hub cache

```bash
# Optional: export HF_HUB_CACHE=/path/to/hfhub
core=$(HF_HUB_DISABLE_SYMLINKS=1 hf download --quiet \
  sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2 \
  --revision main)
```

Use a snapshot without symlinks. The setting leaves existing symlinks in place.

In the Docker command above, drop `-it`, mount `$core` at `/model`, and add these Flare flags to skip the approval prompt.

```text
--dangerously-approve-key 67a70cc0418d2d7a56c7224428b6cac9786c85ab3f6157a88a734315751a02e1
--dangerously-accept-publisher sousekd
```

Signature checks still run. Keep `--verify-weights` for a full stored-data check. To save approval across containers, mount a writable `/state` directory and pass `--trust-store /state/trust.toml`.

[Build sources and Docker image recipe](BUILD.md) · [Model license](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/main/LICENSE) · [Third-party notices](https://huggingface.co/sousekd/DeepSeek-V4.1-Flash-1017-EXL3-3bpw-HyperFlare-GB202-TP2/blob/main/NOTICE)
